# GPU Support Progress

Last updated: 2026-05-09 (multi-warp-per-block for SSV + Viterbi opt kernels)

## Current State

- `hmmsearch --gpu` is an opt-in, protein-only CUDA path. Target databases are built by `hmmseqdb` which produces both Easel `dsqdata` files and a `.gpudb` GPU-native format.
- GPU code lives under `src/cuda/` with stage-owned CUDA files (`p7_cuda_msv.cu`, `p7_cuda_bias.cu`, `p7_cuda_viterbi.cu`, `p7_cuda_forward.cu`, `p7_cuda_fb_parser.cu`, `p7_cuda_ssv.cu`) and shared runtime/profile ownership in `p7_cuda_runtime.cu` + `p7_cuda_internal.h`.
- `src/cuda_msv.h` is only a compatibility wrapper for existing callers; new CUDA-facing code should include `src/cuda/p7_cuda.h`.
- The default GPU MSV path uses a **fused SSV+null+bias+F1 gate kernel** (`cuda_ssv_null_bias_gate_kernel<STRIDE, WARPS>`) that computes SSV score, null score, bias filtersc, and F1 P-value in a single kernel launch per batch. The kernel is templated on STRIDE (compile-time stride per thread) and WARPS-per-block (one sequence per warp) with linear rbv layout (`d_rbv_lin[x * M + k]`) for coalesced access. Survivors are compacted via atomicAdd with in-kernel float score and overflow status written to per-survivor output arrays — D2H transfers only ~32KB per batch (nsurv entries) instead of 1.8MB (full nseq arrays). Supports both resident-database and chunk-based paths via `p7_cuda_SSVNullBiasGateResident()` and `p7_cuda_SSVNullBiasGateDsqdataChunk()`. The separate SSV-only path is retained for `--gpu-ssv-compare` debug mode.
- Exact hit parity is maintained by a two-stage GPU bias approach: (1) fast float32 kernel for all sequences (F1 gating pre-filter), (2) double-precision survivor kernel (`cuda_bias_filter_survivors_kernel`) recomputes filtersc with `log()` for ~4000 F1 survivors, achieving bit-identical parity with CPU `p7_bg_FilterScore`. No CPU `p7_bg_FilterScore` or `p7_MSVFilter` calls remain in the GPU path.
- Resident survivor-core work now keeps Viterbi/F3 pass decisions GPU-side in normal mode, copies only status/pass buffers when full scores are not needed, and materializes `ESL_SQ` metadata only for sequences entering CPU post-Fwd/domain work or compare diagnostics.
- GPU-side F1 gating kernel (`cuda_f1_gating_kernel` in `p7_cuda_bias.cu`) computes MSV P-values and bias-adjusted P-values on device, producing a compact survivor index list via atomicAdd. The batch loop in `hmmsearch_gpu.c` now iterates only over F1 survivors rather than all sequences in the batch. `p7_pipeline_Reuse` and `gpu_RestoreSeqStorage` are called only for survivors.
- Survivor loop is sorted by sequence length before processing, with ReconfigLength caching to skip redundant `p7_bg_SetLength`/`p7_oprofile_ReconfigLength` calls for same-length sequences.
- For post-Viterbi Forward prefilter in normal mode, F3 gating is now pure GPU decision; CPU Forward rerun at the F3 gray zone is retained only in compare/debug paths, not production gating.
- GPU Viterbi uses a **templated** register-based warp-shuffle kernel parameterized on `(STRIDE, WARPS)`. Each block packs WARPS warps × 32 threads, one sequence per warp; each warp owns a contiguous stripe of model nodes in registers, with cross-thread boundaries via `__shfl_up_sync`. STRIDE selects per-thread register-array size at compile time (true register residence for stride≤24, M≤768); a non-templated large-stride fallback handles stride>24 (M≤2048). Legacy shared-memory kernel retained as M>2048 fallback. WARPS per block is auto-tuned at runtime from `cudaGetDeviceProperties` (typically W=4 on sm_89), overridable via the hidden `--gpu-vit-warps` flag. Tiling strategy 4x larger than original (up to 4096 candidates/tile for small M) for better GPU occupancy.
- The stats report now includes exact exclusive timing buckets (`Exact io_read_unpack`, `Exact gpu_h2d`, `Exact gpu_kernel`, `Exact gpu_d2h`, `Exact host_survivor_orchestration`, `Exact cpu_postfwd_domain_null2_output`, `Exact other`) plus `Exact delta_vs_wall`. Legacy `Stage *` and `CUDA *` lines are retained for continuity but can overlap by construction.
- All GPU stages (SSV/MSV, Viterbi prefilter, Forward prefilter, FB parser) are enabled by default with `--gpu`. The Forward prefix kernel now supports models up to M≈2044 (T≤1024 threads) with `cudaFuncSetAttribute` for dynamic shared memory >48KB. Previously, models with M≥1024 fell back to a 4-lane kernel (only 4/32 threads active). The extension provides **12.5x Forward kernel speedup** for large models (M=1500: 525ms → 42ms). Debug compare flags and largem overrides remain available.
- The Easel `dsqdata` chunk-sizing change is applied at build time from `patches/easel-dsqdata-open-sized.patch`; do not edit the Easel submodule in place for this work.

### Engine Reuse (Phase 1)

The CUDA engine is now created once before the query loop in `hmmsearch.c` and reused across all queries. Only the per-query profile (`P7_CUDA_MSVPROFILE`) is rebuilt inside the loop. A `p7_cuda_engine_Reset()` clears per-query stats without destroying device allocations. This eliminates the ~260ms CUDA context init that was previously paid per query.

### GPU-Native Database Format (Phase 2, v2)

A `.gpudb` file format stores sequences pre-unpacked in GPU-ready layout:
- Memory-mappable, page-aligned data region with contiguous uint8_t residues
- Pre-computed int64 offsets and int32 lengths in an index region
- Written by `hmmseqdb` alongside dsqdata files
- Reader in `src/p7_gpudb.c` uses mmap for zero-copy host access
- `hmmsearch --gpu` auto-detects `.gpudb` sidecar; user can also pass `foo.gpudb` directly as the database argument (the `.gpudb` suffix is stripped for dsqdata open)
- **v2 (current)**: Embedded metadata section (name, accession, description, taxid per sequence) enables zero-I/O survivor materialization. When gpudb v2 metadata is present, `hmmsearch --gpu` skips `esl_dsqdata_OpenSized` entirely — no per-query fread/thread overhead.

Layout:
```
[P7_GPUDB_HEADER 80B]
[index: int64 offsets × nseq | int32 lengths × nseq]
[page-aligned padding]
[sequence data: sentinel + residues per seq, trailing sentinel]
[page-aligned padding]
[metadata index: int64 × nseq (byte offset into blob)]
[metadata blob: name\0 acc\0 desc\0 taxid(4B) per seq]
```

### Resident Database (Phase 4)

When GPU memory is sufficient, the entire database is uploaded once and reused across all queries:
- `p7_cuda_engine_UploadDatabase()` bulk-uploads sequence data, offsets, and lengths to device memory
- Per-query overhead reduced to only `tjb_by_seq` array (1 byte/seq) that changes between queries
- All kernel functions (MSV, Viterbi, Forward, FB parser) check `engine->resident_active` and use resident device pointers instead of per-batch uploads
- `p7_cuda_MSVFilterResident()` is the dedicated resident MSV entry point
- Automatic: if upload fails (insufficient GPU memory), falls back to per-batch streaming
- Resident pointer arithmetic uses `engine->d_resident_offsets + engine->resident_batch_seq0` for batch-relative addressing

## CPU-only Modules

These remain intentionally CPU-side in the current Resident Survivor Core scope:

- Post-Fwd domain definition and decoding: `p7_domaindef_ByPosteriorHeuristics()` and `p7_DomainDecoding()`.
- Null2 correction and reconstruction scoring.
- Hit object creation, thresholding, top-hits merge/sort/report, and CLI/reporting output.
- Sequence metadata object assembly/orchestration for sequences that reach CPU post-Fwd work.
- Database read/unpack and `hmmseqdb`/`dsqdata` host-side batch construction.

## Benchmark Snapshot

**Current best (2026-05-08):** Multi-query single-process benchmark (13 HMMs, 229K seqs, 97M residues)

| Config | Wall time | vs CPU-1 | vs CPU-4 |
|--------|-----------|----------|----------|
| CPU 1-thread | ~9.3s | 1.00x | — |
| CPU 4-thread | ~4.8s | — | 1.00x |
| GPU (fused, all stages) | ~1.27s | **~7.3x** | **~2.8x** |

- Flags: `--gpu` (fused SSV+null+bias+gate kernel, all stages default-on)
- Database: resident on GPU (229,290 seqs, 92.8 MB), gpudb v2 with embedded metadata
- Hit parity: **`cpu_only=0`, `gpu_only=0`** (exact GPU-vs-CPU parity, all 13 queries)
- Benchmark: `test-speed/x-hmmsearch-gpu-profmark` (multi-query single-process mode)
- Note: wall times vary ±0.2s between runs due to CUDA init jitter; speedup range 6.6x–7.3x vs CPU-1
- Key optimizations reaching current state:
  - **Fused SSV+null+bias+F1 kernel** — single launch replaces 5 separate kernels, eliminates inter-stage sync (246ms → 115ms `exact_other`)
  - **Survivor-indexed D2H** — transfers only nsurv×8 bytes (~32KB) instead of nseq×8 bytes (1.8MB); D2H: 28ms → 2.6ms
  - **Linear rbv layout** — `d_rbv_lin[x * M + k]` eliminates shared-memory lookup tables in fused kernel
  - **Templated Viterbi kernel** — stride as compile-time constant enables true register residence (2.9x Viterbi kernel speedup)
  - **Larger Viterbi tile sizes** — 4x increase (4096 candidates/tile for M≤700) with int64 overflow fix
  - Multi-query single-process benchmark (CUDA init paid once)
  - Pre-allocated CUDA events reused across all kernels
  - Bias model parameter caching (uploaded once per query, not per batch)
  - Viterbi/Forward ReconfigLength caching in post-processing loops
  - gpudb v2 embedded metadata eliminates dsqdata I/O entirely (~0.26s/query saved)
  - `madvise` hints (SEQUENTIAL for upload, RANDOM for metadata) on mmap'd gpudb
  - `cudaHostRegister` for DMA-accelerated initial GPU upload
  - **Forward prefix kernel extended** to T≤1024 (M≤2044) — large-model Forward 12.5x faster (525ms → 42ms for M=1500)
  - **Multi-warp-per-block for SSV + Viterbi opt kernels** — kernels now templated on `(STRIDE, WARPS)`, block packs W warps (one sequence per warp), W auto-tuned from `cudaGetDeviceProperties` with tie-break toward W=4 on sm_89. On the current 13-query profmark workload (M≈100–500, 229K seqs/batch), kernel-time delta vs W=1 is within run-to-run noise (≤1%) — the warp-shuffle DP already saturates SM compute at W=1. Infrastructure is in place for larger-M cases and to share the dispatch pattern with future kernels. Hidden flags `--gpu-ssv-warps`, `--gpu-vit-warps` for override.

**Benchmark run**: `benchmark-data/profmark-current/gpu-audit/forward-prefix-opt-run2/`

## GPU SSV/Fused Kernel (Default MSV Path, 2026-05-07)

The fused SSV+null+bias+gate kernel is the default GPU MSV path:
- **Architecture**: Templated kernel `cuda_ssv_null_bias_gate_kernel<STRIDE>` fuses 5 formerly separate stages (SSV, null score, bias filter, F1 Gumbel gate, survivor compaction) into a single kernel launch. Each warp (32 threads) processes one sequence end-to-end: SSV score → null score → bias filtersc → F1 P-value → atomic survivor write. In-kernel MSV fallback for ~0.3% of sequences needing J-state.
- **Profile layout**: Uses both `rbv` (SIMD-style unsigned byte, for shared-memory fallback path) and `rbv_lin` (linear `[x * M + k]` layout, for register-based fast path). The linear layout eliminates integer division / shared-memory q/z lookup tables from the SSV inner loop.
- **Survivor-indexed output**: Instead of writing raw integer scores for all 229K sequences and converting on host, the kernel writes float `usc` and overflow status only for sequences that pass both MSV F1 and bias F1 gates (~4K survivors). D2H transfers ~32KB instead of 1.8MB per batch.
- **Paths**: `p7_cuda_SSVNullBiasGateResident()` for resident-database mode, `p7_cuda_SSVNullBiasGateDsqdataChunk()` for chunk-based mode. Legacy SSV-only path (`p7_cuda_SSVFilterResident`/`p7_cuda_SSVFilterDsqdataChunk`) retained for `--gpu-ssv-compare` debug mode.
- **Parity**: fused path agrees with legacy GPU pipeline (zero hit diff). SSV scores bitwise-identical to monolithic MSV via `--gpu-ssv-compare`. Known ±3 survivor count difference vs legacy due to in-kernel bias using length-correct `tjb_b` per sequence (more accurate than legacy post-hoc approach).
- **Timing** (profmark, 229K sequences, all-13 aggregate):
  - Fused path total: 1.23s wall time
  - Legacy path total: 1.49s wall time
  - Improvement: **17.4%** end-to-end
  - D2H reduction: 28ms → 2.6ms (10.8x)
  - `exact_other` reduction: 246ms → 115ms (−53%)
- **Key optimizations**:
  1. Kernel fusion — eliminates 4 kernel launch + sync round-trips per batch
  2. Linear rbv layout — coalesced global memory access pattern for register-based SSV
  3. Survivor-indexed D2H — only transfer data for F1-passing sequences
  4. Templated STRIDE dispatch — compile-time constant enables register-based DP
- **Files**: `src/cuda/p7_cuda_ssv.cu`, `src/cuda/p7_cuda.h`, `src/hmmsearch_gpu.c`.

## Open Risks

- FB parser raw `p7X_SCALE` row diagnostics remain open (bounded posterior inputs are acceptable).

## Verification Guidance

- Use `profmark` for any serious GPU speed claim, not tutorial-sized inputs.
- Build target databases with `hmmseqdb` before running `hmmsearch --gpu`.
- Record CPU/GPU wall time, CUDA H2D/kernel/D2H time, batch sizes, CUDA batch counts, pass counts for MSV/bias/Viterbi/Forward, and final hit deltas.
- Compare each proposed GPU stage against the last accepted GPU baseline, not only against CPU.

## GPU nhmmer (Nucleotide Search) — 2026-05-09

A GPU-accelerated path for `nhmmer` is available via `--gpu`. Rebased on latest `h3-gpu` (fused SSV kernel, multi-warp-per-block, templated Viterbi).

- **Architecture**: GPU SSV longtarget kernel (warp-per-chunk, 64K chunks) + batch MSV/bias/F1 filter + GPU Viterbi pre-filter + GPU scanning Viterbi + GPU Forward pre-filter + GPU FB parser + GPU domain rescoring + threaded CPU hit reporting.
- **GPU domain rescoring** (`p7_cuda_DomainRescoreBatch`): Cross-window batching of all domains (6 kernels: Fwd/Bck/Decoding/OA/OATrace/Domcorr) + trim batching. Replaces `rescore_isolated_domain` (the 67-91% bottleneck). 4/6 kernels use T-thread-per-block with parallel prefix scan.
- **GPU FB parser** (`nhmmer_gpu_run_fb_parser_batch`): Batch Forward+Backward parser on GPU. Forward-Backward split: prefilter saves xf, Backward-only for F3 survivors.
- **Forward pre-filter** (`nhmmer_gpu_forward_prefilter`): GPU Forward score-only with F3*2.0 relaxed gate. Removes 45-60% of sub-windows before FB parser.
- **Skip-Forward optimization**: `p7_pli_postFwd_LongTarget()` injects GPU-precomputed xf/fwdsc, skipping redundant CPU Forward. Default-on with `--gpu`; hidden `--gpu-no-fwd-prefilter` restores the older CPU Fwd/Bwd continuation for diagnostics.
- **Batch filter** (`--gpu-batch`): Packs merged windows as synthetic ESL_DSQDATA_CHUNK (zero-copy), runs GPU MSV + null + bias batch scoring, applies F1 gating.
- **Viterbi pre-filter** (`--gpu-vit-prefilter`): GPU single-score Viterbi on batch survivors (warps_per_block=1 for short windows). Windows below F2 threshold skipped before scanning Viterbi.
- **Threading**: Post-vit workers distributed across N CPU threads with deep-copied per-thread state.
- **Engine reuse**: CUDA engine created once before query loop, saves ~250ms per additional query.
- **Hit parity**: MADE1 462/465 (3-hit diff, <1%), query_short 363/363 (exact match), query_medium 648/648 (exact match). Remaining differences from float32 vs double precision.
- **Performance**: GPU-4 FASTA: MADE1 1.01s, query_short 0.99s, query_medium 2.87s. GPU-4 overlap-nucdb: MADE1 0.64s, query_short 1.02s, query_medium 2.23s. CPU-4: 0.31s, 0.40s, 1.57s. GPU bottleneck: CPU workers (envelope-finding, 44-94% of wall time).
- **Files**: `src/nhmmer_gpu.c`, `src/nhmmer_internal.h`, `src/cuda/p7_cuda_ssv_longtarget.cu`, `src/cuda/p7_cuda_viterbi_longtarget.cu`, `src/cuda/p7_cuda_domain_rescore.cu`, `src/cuda/p7_cuda_fb_parser.cu`
- **Detailed docs**: See `nhmmer-gpu-progress.md` and `nhmmer-gpu-todo.md`.

## History

Detailed dated implementation notes were moved to `gpu-support-history.md`.
