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

## Open Risks

- CUDA-native SSV-equivalent boundary deferred (direct diagonal port did not reproduce CPU `p7_SSVFilter()` behavior).
- Later-stage prefilters (Viterbi, Forward, FB parser) are parity-clean on checked samples but need broader validation before becoming default.
- FB parser raw `p7X_SCALE` row diagnostics remain open (bounded posterior inputs are acceptable).

## Verification Guidance

- Use `profmark` for any serious GPU speed claim, not tutorial-sized inputs.
- Build target databases with `hmmseqdb` before running `hmmsearch --gpu`.
- Record CPU/GPU wall time, CUDA H2D/kernel/D2H time, batch sizes, CUDA batch counts, pass counts for MSV/bias/Viterbi/Forward, and final hit deltas.
- Compare each proposed GPU stage against the last accepted GPU baseline, not only against CPU.

## History

Detailed dated implementation notes were moved to `gpu-support-history.md`.
