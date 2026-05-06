# GPU Support Progress

Last updated: 2026-05-06

## Current State

- `hmmsearch --gpu` is an opt-in, protein-only CUDA path. It requires target input built by `hmmseqdb` as Easel protein `dsqdata`; ordinary FASTA stays on the CPU path.
- GPU code lives under `src/cuda/` with stage-owned CUDA files (`p7_cuda_msv.cu`, `p7_cuda_bias.cu`, `p7_cuda_viterbi.cu`, `p7_cuda_forward.cu`, `p7_cuda_fb_parser.cu`) and shared runtime/profile ownership in `p7_cuda_runtime.cu` + `p7_cuda_internal.h`.
- `src/cuda_msv.h` is only a compatibility wrapper for existing callers; new CUDA-facing code should include `src/cuda/p7_cuda.h`.
- The default accepted GPU path accelerates MSV and computes the biased-composition filter score in CUDA batches while reusing the MSV-uploaded sequence batch.
- Exact hit parity currently depends on CPU-compatible checks at the bias-corrected F1 boundary: CPU `p7_bg_FilterScore()` supplies the final bias score for GPU MSV survivors, and CPU `p7_MSVFilter()` can rescue bias-boundary rejects through the optimized CPU SSV shortcut.
- Resident survivor-core work now keeps Viterbi/F3 pass decisions GPU-side in normal mode, copies only status/pass buffers when full scores are not needed, and materializes `ESL_SQ` metadata only for sequences entering CPU post-Fwd/domain work or compare diagnostics.
- GPU-side F1 gating kernel (`cuda_f1_gating_kernel` in `p7_cuda_bias.cu`) computes MSV P-values and bias-adjusted P-values on device, producing a compact survivor index list via atomicAdd. The batch loop in `hmmsearch_gpu.c` now iterates only over F1 survivors rather than all sequences in the batch. `p7_pipeline_Reuse` and `gpu_RestoreSeqStorage` are called only for survivors.
- For post-Viterbi Forward prefilter in normal mode, F3 gating is now pure GPU decision; CPU Forward rerun at the F3 gray zone is retained only in compare/debug paths, not production gating.
- The stats report now includes exact exclusive timing buckets (`Exact io_read_unpack`, `Exact gpu_h2d`, `Exact gpu_kernel`, `Exact gpu_d2h`, `Exact host_survivor_orchestration`, `Exact cpu_postfwd_domain_null2_output`, `Exact other`) plus `Exact delta_vs_wall`. Legacy `Stage *` and `CUDA *` lines are retained for continuity but can overlap by construction.
- Experimental/default-off flags remain available for later-stage work: `--gpu-vit-prefilter`, `--gpu-fwd-prefilter`, `--gpu-fb-parser`, and their compare/min-batch controls.
- The Easel `dsqdata` chunk-sizing change is applied at build time from `patches/easel-dsqdata-open-sized.patch`; do not edit the Easel submodule in place for this work.

## CPU-only Modules

These remain intentionally CPU-side in the current Resident Survivor Core scope:

- Post-Fwd domain definition and decoding: `p7_domaindef_ByPosteriorHeuristics()` and `p7_DomainDecoding()`.
- Null2 correction and reconstruction scoring.
- Hit object creation, thresholding, top-hits merge/sort/report, and CLI/reporting output.
- Sequence metadata object assembly/orchestration for sequences that reach CPU post-Fwd work.
- Database read/unpack and `hmmseqdb`/`dsqdata` host-side batch construction.

## Benchmark Snapshot

**Current best (2026-05-06):** smem-optimized run in `benchmark-data/profmark-current/gpu-audit/smem-opt-run/`
- Flags: `--gpu-vit-prefilter --gpu-fwd-prefilter --gpu-fb-parser`
- Profmark aggregate: CPU wall 10.44 sec, GPU wall 6.83 sec, speedup 1.53x, `cpu_only=0`, `gpu_only=0`.
- Manual `time` measurements (same 13 queries):
  - CPU single-thread: 13.58s | CPU 4-thread: 5.55s | GPU cold: 7.68s | GPU warm (amortized init): ~4.30s
  - GPU vs CPU-1: 1.77x | GPU vs CPU-4 cold: 0.72x | GPU vs CPU-4 warm: 1.29x
- GPU wall breakdown (6.83s):
  - CUDA context init: 3.38s (49.5%) — 260ms × 13 queries, amortizable with engine reuse
  - Host sync/blocking: 2.24s (32.8%) — host waiting for synchronous GPU ops
  - GPU kernel execution: 0.49s (7.2%)
  - CPU survivor work: 0.43s (6.3%)
  - CPU postfwd/domain/null2: 0.12s (1.7%)
  - Host in-loop overhead: 0.11s (1.6%)
  - GPU H2D + D2H transfers: 0.07s (0.9%)

**Key historical milestones (superseded runs in `benchmark-data/profmark-current/gpu-audit/`):**
- Pre-smem baseline (all-13, later-stage flags): CPU 29.69s, GPU 16.57s, 1.792x speedup.
- MSV/bias-only baseline (all-13): CPU 28.42s, GPU 19.49s, 1.458x speedup.
- Compare-mode validation: zero `CUDAVIT`/`CUDAFWD` mismatches; `CUDAFB` bounded (max_mocc ≤ 0.000007, max_btot ≤ 0.000012, max_etot ≤ 0.000019).
- Large-profile Viterbi/Forward activation: tested and reverted — current CUDA kernels are slower than CPU SSE for large profiles.
- GPU-side F1 gating: adopted for architectural cleanliness, but performance gain was only ~5-10ms/query (not the ~200ms hypothesized).

## Standalone GPU SSV Kernel (2026-05-07)

A standalone SSV kernel is available via `--gpu-ssv`:
- **Architecture**: Register-based SSV fast-path (no J-state, constant xB) with in-kernel MSV fallback for ~0.3% of sequences that need J-state. Each thread owns a contiguous stripe of model nodes in registers; cross-thread boundary values communicated via `__shfl_up_sync` (no `__syncthreads()` in the SSV inner loop). Precomputed q/z lookup tables in shared memory eliminate integer division from the inner loop.
- **Profile format**: uses `rbv` (unsigned byte) — same as monolithic MSV kernel. Does NOT use the CPU `sbv` (signed byte) profile that caused the earlier failed attempt.
- **Parity**: bitwise-identical to monolithic MSV on all 13 profmark queries (229,290 sequences × 13 = ~3M comparisons, zero mismatches via `--gpu-ssv-compare`). Zero hit-level parity differences (msv_only=0, ssv_only=0).
- **Kernel timing** (profmark, 229K sequences, all-13 aggregate):
  - Optimized SSV: 376.0ms total (28.9ms avg/query)
  - Monolithic MSV: 511.0ms total (39.3ms avg/query)
  - Speedup: **1.36x** kernel-level improvement
  - Best per-query gains: 1.57-1.79x for larger profiles (M > 200)
- **Key optimizations** (2026-05-07):
  1. Precomputed q/z lookup tables — eliminates `% Q` and `/ Q` integer divisions (~50 cycles/iteration saved)
  2. Register-based DP — each thread keeps its stripe of prev values in registers instead of shared memory
  3. Warp shuffle for boundary — `__shfl_up_sync` replaces `__syncthreads()` + shared memory for the single cross-thread dependency
- **History**: Phase 1 was two-pass (15-20% slower). Phase 2 fused the fallback (matched MSV). Phase 3 (current) exploits SSV's constant-xB property for register-based DP with 1.36x kernel speedup.
- **Status**: opt-in, parity-verified, 1.36x faster than monolithic MSV. Strong candidate for default GPU MSV path.
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
