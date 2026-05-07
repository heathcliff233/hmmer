# GPU Support Progress

Last updated: 2026-05-07

## Current State

- `hmmsearch --gpu` is an opt-in, protein-only CUDA path. Target databases are built by `hmmseqdb` which produces both Easel `dsqdata` files and a `.gpudb` GPU-native format.
- GPU code lives under `src/cuda/` with stage-owned CUDA files (`p7_cuda_msv.cu`, `p7_cuda_bias.cu`, `p7_cuda_viterbi.cu`, `p7_cuda_forward.cu`, `p7_cuda_fb_parser.cu`, `p7_cuda_ssv.cu`) and shared runtime/profile ownership in `p7_cuda_runtime.cu` + `p7_cuda_internal.h`.
- `src/cuda_msv.h` is only a compatibility wrapper for existing callers; new CUDA-facing code should include `src/cuda/p7_cuda.h`.
- The default GPU MSV path uses the SSV kernel (register-optimized, 1.36x faster than monolithic MSV). The SSV kernel handles both resident-database and chunk-based paths via `p7_cuda_SSVFilterResident()` and `p7_cuda_SSVFilterDsqdataChunk()`. The `--gpu-ssv` flag is now a no-op (SSV is always used).
- Exact hit parity is maintained by a two-stage GPU bias approach: (1) fast float32 kernel for all sequences (F1 gating pre-filter), (2) double-precision survivor kernel (`cuda_bias_filter_survivors_kernel`) recomputes filtersc with `log()` for ~4000 F1 survivors, achieving bit-identical parity with CPU `p7_bg_FilterScore`. No CPU `p7_bg_FilterScore` or `p7_MSVFilter` calls remain in the GPU path.
- Resident survivor-core work now keeps Viterbi/F3 pass decisions GPU-side in normal mode, copies only status/pass buffers when full scores are not needed, and materializes `ESL_SQ` metadata only for sequences entering CPU post-Fwd/domain work or compare diagnostics.
- GPU-side F1 gating kernel (`cuda_f1_gating_kernel` in `p7_cuda_bias.cu`) computes MSV P-values and bias-adjusted P-values on device, producing a compact survivor index list via atomicAdd. The batch loop in `hmmsearch_gpu.c` now iterates only over F1 survivors rather than all sequences in the batch. `p7_pipeline_Reuse` and `gpu_RestoreSeqStorage` are called only for survivors.
- Survivor loop is sorted by sequence length before processing, with ReconfigLength caching to skip redundant `p7_bg_SetLength`/`p7_oprofile_ReconfigLength` calls for same-length sequences.
- For post-Viterbi Forward prefilter in normal mode, F3 gating is now pure GPU decision; CPU Forward rerun at the F3 gray zone is retained only in compare/debug paths, not production gating.
- GPU Viterbi is active by default for M≤2048 (was M≤512); GPU Forward for M≤1024 (was M≤256). Tiling strategy includes an M>2048 tier for very large models.
- The stats report now includes exact exclusive timing buckets (`Exact io_read_unpack`, `Exact gpu_h2d`, `Exact gpu_kernel`, `Exact gpu_d2h`, `Exact host_survivor_orchestration`, `Exact cpu_postfwd_domain_null2_output`, `Exact other`) plus `Exact delta_vs_wall`. Legacy `Stage *` and `CUDA *` lines are retained for continuity but can overlap by construction.
- Experimental/default-off flags remain available for later-stage work: `--gpu-vit-prefilter`, `--gpu-fwd-prefilter`, `--gpu-fb-parser`, and their compare/min-batch controls.
- The Easel `dsqdata` chunk-sizing change is applied at build time from `patches/easel-dsqdata-open-sized.patch`; do not edit the Easel submodule in place for this work.

### Engine Reuse (Phase 1)

The CUDA engine is now created once before the query loop in `hmmsearch.c` and reused across all queries. Only the per-query profile (`P7_CUDA_MSVPROFILE`) is rebuilt inside the loop. A `p7_cuda_engine_Reset()` clears per-query stats without destroying device allocations. This eliminates the ~260ms CUDA context init that was previously paid per query.

### GPU-Native Database Format (Phase 2)

A `.gpudb` file format stores sequences pre-unpacked in GPU-ready layout:
- Memory-mappable, page-aligned data region with contiguous uint8_t residues
- Pre-computed int64 offsets and int32 lengths in an index region
- Written by `hmmseqdb` alongside dsqdata files
- Reader in `src/p7_gpudb.c` uses mmap for zero-copy host access
- `hmmsearch --gpu` auto-detects `.gpudb` sidecar; user can also pass `foo.gpudb` directly as the database argument (the `.gpudb` suffix is stripped for dsqdata open)

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

| Config | Wall time | vs CPU-1 |
|--------|-----------|----------|
| CPU 1-thread | 9.58s | 1.00x |
| GPU (resident DB, all stages, SSV default) | 6.49s | **1.48x** (aggregate), **1.51x** (median) |

- Flags: `--gpu --gpu-vit-prefilter --gpu-fwd-prefilter --gpu-fb-parser`
- Database: resident on GPU (229,290 seqs, 92.8 MB)
- Hit parity: `cpu_only=0`, `gpu_only=0` across all 13 queries
- SSV compare: zero mismatches (bitwise-identical to monolithic MSV)
- Key changes from prior baseline (1.32x → 1.48x):
  - SSV kernel as default MSV path (1.36x faster kernel)
  - Viterbi M-limit raised from 512 to 2048 (more queries use GPU Viterbi)
  - Forward M-limit raised from 256 to 1024
  - Survivor loop: sorted by length + ReconfigLength caching + CPU MSV fallback eliminated

**GPU time breakdown (~6.5s search wall):**

| Category | Time | Notes |
|----------|------|-------|
| GPU kernels | ~0.49s | SSV + Vit 0.48 + Fwd 0.25 + Bias 0.06 + Bck 0.05 |
| I/O (dsqdata read) | 0.19s | Reading chunk metadata |
| Domain def + null2 | 0.15s | CPU-only, after Forward |
| Survivor loop | 0.003s | Sorted by length, ReconfigLength cached |
| Other (score convert, F1 gate, dispatch) | ~0.56s | Between kernel launches |

**Per-query profmark results** (in `benchmark-data/profmark-current/gpu-audit/ssv-default-full/`):
- GPU vs 1-thread: every query faster, 1.31x–2.74x range
- Per-query speedup variation driven by model size and survivor count

**Key historical milestones (superseded runs in `benchmark-data/profmark-current/gpu-audit/`):**
- bias-elim-default (2026-05-07, pre-SSV-default): CPU 9.57s, GPU 7.25s, 1.32x aggregate
- smem-opt-run (2026-05-06, pre-engine-reuse): CPU 10.44s, GPU 6.83s, 1.53x (per-query separate processes, CUDA init paid 13x)
- Pre-smem baseline (all-13, later-stage flags): CPU 29.69s, GPU 16.57s, 1.792x speedup.
- MSV/bias-only baseline (all-13): CPU 28.42s, GPU 19.49s, 1.458x speedup.

## GPU SSV Kernel (Default MSV Path, 2026-05-07)

The SSV kernel is now the default GPU MSV path (no `--gpu-ssv` flag needed):
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

- Later-stage prefilters (Viterbi, Forward, FB parser) are parity-clean on checked samples but need broader validation before becoming default.
- FB parser raw `p7X_SCALE` row diagnostics remain open (bounded posterior inputs are acceptable).

## Verification Guidance

- Use `profmark` for any serious GPU speed claim, not tutorial-sized inputs.
- Build target databases with `hmmseqdb` before running `hmmsearch --gpu`.
- Record CPU/GPU wall time, CUDA H2D/kernel/D2H time, batch sizes, CUDA batch counts, pass counts for MSV/bias/Viterbi/Forward, and final hit deltas.
- Compare each proposed GPU stage against the last accepted GPU baseline, not only against CPU.

## History

Detailed dated implementation notes were moved to `gpu-support-history.md`.
