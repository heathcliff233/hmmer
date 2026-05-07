# GPU Support Progress

Last updated: 2026-05-07 (Viterbi templated kernel + tile optimization)

## Current State

- `hmmsearch --gpu` is an opt-in, protein-only CUDA path. Target databases are built by `hmmseqdb` which produces both Easel `dsqdata` files and a `.gpudb` GPU-native format.
- GPU code lives under `src/cuda/` with stage-owned CUDA files (`p7_cuda_msv.cu`, `p7_cuda_bias.cu`, `p7_cuda_viterbi.cu`, `p7_cuda_forward.cu`, `p7_cuda_fb_parser.cu`, `p7_cuda_ssv.cu`) and shared runtime/profile ownership in `p7_cuda_runtime.cu` + `p7_cuda_internal.h`.
- `src/cuda_msv.h` is only a compatibility wrapper for existing callers; new CUDA-facing code should include `src/cuda/p7_cuda.h`.
- The default GPU MSV path uses the SSV kernel (register-optimized, 1.36x faster than monolithic MSV). The SSV kernel handles both resident-database and chunk-based paths via `p7_cuda_SSVFilterResident()` and `p7_cuda_SSVFilterDsqdataChunk()`.
- Exact hit parity is maintained by a two-stage GPU bias approach: (1) fast float32 kernel for all sequences (F1 gating pre-filter), (2) double-precision survivor kernel (`cuda_bias_filter_survivors_kernel`) recomputes filtersc with `log()` for ~4000 F1 survivors, achieving bit-identical parity with CPU `p7_bg_FilterScore`. No CPU `p7_bg_FilterScore` or `p7_MSVFilter` calls remain in the GPU path.
- Resident survivor-core work now keeps Viterbi/F3 pass decisions GPU-side in normal mode, copies only status/pass buffers when full scores are not needed, and materializes `ESL_SQ` metadata only for sequences entering CPU post-Fwd/domain work or compare diagnostics.
- GPU-side F1 gating kernel (`cuda_f1_gating_kernel` in `p7_cuda_bias.cu`) computes MSV P-values and bias-adjusted P-values on device, producing a compact survivor index list via atomicAdd. The batch loop in `hmmsearch_gpu.c` now iterates only over F1 survivors rather than all sequences in the batch. `p7_pipeline_Reuse` and `gpu_RestoreSeqStorage` are called only for survivors.
- Survivor loop is sorted by sequence length before processing, with ReconfigLength caching to skip redundant `p7_bg_SetLength`/`p7_oprofile_ReconfigLength` calls for same-length sequences.
- For post-Viterbi Forward prefilter in normal mode, F3 gating is now pure GPU decision; CPU Forward rerun at the F3 gray zone is retained only in compare/debug paths, not production gating.
- GPU Viterbi uses a **templated** register-based warp-shuffle kernel (32 threads/sequence, one warp per block). Each thread owns a contiguous stripe of model nodes; cross-thread boundaries use `__shfl_up_sync`. The kernel is a C++ template parameterized on stride (compile-time constant), enabling `nvcc` to place `reg_M/D/I` arrays in true registers for stride≤24 (M≤768). For stride>24 (M>768, ≤2048), a non-templated large-stride fallback is used. Legacy shared-memory kernel retained as M>2048 fallback. Tiling strategy 4x larger than original (up to 4096 candidates/tile for small M) for better GPU occupancy.
- The stats report now includes exact exclusive timing buckets (`Exact io_read_unpack`, `Exact gpu_h2d`, `Exact gpu_kernel`, `Exact gpu_d2h`, `Exact host_survivor_orchestration`, `Exact cpu_postfwd_domain_null2_output`, `Exact other`) plus `Exact delta_vs_wall`. Legacy `Stage *` and `CUDA *` lines are retained for continuity but can overlap by construction.
- All GPU stages (SSV/MSV, Viterbi prefilter, Forward prefilter, FB parser) are enabled by default with `--gpu`. Debug compare flags and largem overrides remain available.
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

**Current best (2026-05-07):** Multi-query single-process benchmark (13 HMMs, 229K seqs, 97M residues)

| Config | Wall time | vs CPU-1 | vs CPU-4 |
|--------|-----------|----------|----------|
| CPU 1-thread | 10.68s | 1.00x | — |
| CPU 4-thread | 4.79s | — | 1.00x |
| GPU (all stages) | 1.45s | **7.37x** | **3.30x** |

- Flags: `--gpu` (all stages now default-on)
- Database: resident on GPU (229,290 seqs, 92.8 MB), gpudb v2 with embedded metadata
- Hit parity: `cpu_only=0`, `gpu_only=0` across all 13 queries
- Benchmark: `test-speed/x-hmmsearch-gpu-profmark` (multi-query single-process mode)
- GPU utilization: **87%** SM utilization during kernel execution (`nvidia-smi` sampled)
- Key optimizations reaching current state:
  - All GPU stages default-on (no separate opt-in flags)
  - Multi-query single-process benchmark (CUDA init paid once)
  - Pre-allocated CUDA events reused across all kernels
  - Bias model parameter caching (uploaded once per query, not per batch)
  - Viterbi/Forward ReconfigLength caching in post-processing loops
  - gpudb v2 embedded metadata eliminates dsqdata I/O entirely (~0.26s/query saved)
  - `madvise` hints (SEQUENTIAL for upload, RANDOM for metadata) on mmap'd gpudb
  - `cudaHostRegister` for DMA-accelerated initial GPU upload
  - **Templated Viterbi kernel** — stride as compile-time constant enables true register residence (2.9x Viterbi kernel speedup)
  - **Larger Viterbi tile sizes** — 4x increase (4096 candidates/tile for M≤700) with int64 overflow fix

**Benchmark run**: `benchmark-data/profmark-current/gpu-audit/io-opt/`

## GPU SSV Kernel (Default MSV Path, 2026-05-07)

The SSV kernel is the default GPU MSV path:
- **Architecture**: Register-based SSV fast-path (no J-state, constant xB) with in-kernel MSV fallback for ~0.3% of sequences that need J-state. Each thread owns a contiguous stripe of model nodes in registers; cross-thread boundary values communicated via `__shfl_up_sync` (no `__syncthreads()` in the SSV inner loop). Precomputed q/z lookup tables in shared memory eliminate integer division from the inner loop.
- **Profile format**: uses `rbv` (unsigned byte) — same as monolithic MSV kernel. Does NOT use the CPU `sbv` (signed byte) profile that caused the earlier failed attempt.
- **Paths**: `p7_cuda_SSVFilterResident()` for resident-database mode, `p7_cuda_SSVFilterDsqdataChunk()` for chunk-based mode.
- **Parity**: bitwise-identical to monolithic MSV on all 13 profmark queries (229,290 sequences × 13 = ~3M comparisons, zero mismatches via `--gpu-ssv-compare`). Zero hit-level parity differences (cpu_only=0, gpu_only=0).
- **Kernel timing** (profmark, 229K sequences, all-13 aggregate):
  - Optimized SSV: 376.0ms total (28.9ms avg/query)
  - Monolithic MSV: 511.0ms total (39.3ms avg/query)
  - Speedup: **1.36x** kernel-level improvement
  - Best per-query gains: 1.57-1.79x for larger profiles (M > 200)
- **Key optimizations**:
  1. Precomputed q/z lookup tables — eliminates `% Q` and `/ Q` integer divisions (~50 cycles/iteration saved)
  2. Register-based DP — each thread keeps its stripe of prev values in registers instead of shared memory
  3. Warp shuffle for boundary — `__shfl_up_sync` replaces `__syncthreads()` + shared memory for the single cross-thread dependency
- **History**: Phase 1 was two-pass (15-20% slower). Phase 2 fused the fallback (matched MSV). Phase 3 exploits SSV's constant-xB property for register-based DP with 1.36x kernel speedup.
- **Files**: `src/cuda/p7_cuda_ssv.cu`, CLI flags in `hmmsearch.c`.

## Open Risks

- FB parser raw `p7X_SCALE` row diagnostics remain open (bounded posterior inputs are acceptable).

## Verification Guidance

- Use `profmark` for any serious GPU speed claim, not tutorial-sized inputs.
- Build target databases with `hmmseqdb` before running `hmmsearch --gpu`.
- Record CPU/GPU wall time, CUDA H2D/kernel/D2H time, batch sizes, CUDA batch counts, pass counts for MSV/bias/Viterbi/Forward, and final hit deltas.
- Compare each proposed GPU stage against the last accepted GPU baseline, not only against CPU.

## History

Detailed dated implementation notes were moved to `gpu-support-history.md`.
