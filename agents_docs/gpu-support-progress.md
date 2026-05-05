# GPU Support Progress

Last updated: 2026-05-05

## Current State

- `hmmsearch --gpu` is an opt-in, protein-only CUDA path. It requires target input built by `hmmseqdb` as Easel protein `dsqdata`; ordinary FASTA stays on the CPU path.
- GPU code lives under `src/cuda/` with stage-owned CUDA files (`p7_cuda_msv.cu`, `p7_cuda_bias.cu`, `p7_cuda_viterbi.cu`, `p7_cuda_forward.cu`, `p7_cuda_fb_parser.cu`) and shared runtime/profile ownership in `p7_cuda_runtime.cu` + `p7_cuda_internal.h`.
- `src/cuda_msv.h` is only a compatibility wrapper for existing callers; new CUDA-facing code should include `src/cuda/p7_cuda.h`.
- The default accepted GPU path accelerates MSV and computes the biased-composition filter score in CUDA batches while reusing the MSV-uploaded sequence batch.
- Exact hit parity currently depends on CPU-compatible checks at the bias-corrected F1 boundary: CPU `p7_bg_FilterScore()` supplies the final bias score for GPU MSV survivors, and CPU `p7_MSVFilter()` can rescue bias-boundary rejects through the optimized CPU SSV shortcut.
- Experimental/default-off flags remain available for later-stage work: `--gpu-vit-prefilter`, `--gpu-fwd-prefilter`, `--gpu-fb-parser`, and their compare/min-batch controls.
- The Easel `dsqdata` chunk-sizing change is applied at build time from `patches/easel-dsqdata-open-sized.patch`; do not edit the Easel submodule in place for this work.

## Benchmark Snapshot

- Correctness-first all-13 profmark run in `benchmark-data/profmark-current/gpu-audit/cpu-bias-msv-rescue-all13/` with `--gpu-vit-prefilter --gpu-fwd-prefilter --gpu-fb-parser`: CPU wall 26.82 sec, GPU wall 16.32 sec, aggregate speedup 1.643x, median speedup 1.508x, `cpu_only=0`, `gpu_only=0`.
- Broader 12-query profmark run in `benchmark-data/profmark-current/gpu-audit/broader12-nocompare/` with the same later-stage flags: CPU wall 7.78 sec, GPU wall 7.05 sec, aggregate speedup 1.104x, `cpu_only=0`, `gpu_only=0`.
- The broader 12-query MSV/bias baseline in `benchmark-data/profmark-current/gpu-audit/broader12-msvbias-baseline/` recorded GPU wall 7.88 sec; the later-stage path saved 0.83 sec but regressed a few short profiles slightly, so later stages remain default-off pending auto-gating policy.
- Compare-mode runs are validation evidence, not speed claims. The current compare runs produced zero `CUDAVIT` and zero `CUDAFWD` diagnostics; `CUDAFB` raw scale-row diagnostics remain, but domain-decoding input maxima stayed below 0.00002 in the all-13 compare run.
- Viterbi same-algorithm optimization (2026-05-05) in `benchmark-data/profmark-current/gpu-audit/vit-samealgo-all13-nocompare/`: CPU wall 28.66 sec, GPU wall 26.61 sec, aggregate speedup 1.077x, `cpu_only=0`, `gpu_only=0`; CUDA Viterbi total (H2D+kernel+D2H) 0.945 sec vs CPU stage Viterbi 3.861 sec from CPU logs (about 4.09x stage speedup; kernel 0.941 sec).
- Same iteration, broader 12 non-compare in `benchmark-data/profmark-current/gpu-audit/vit-samealgo-broader12-nocompare/`: CPU wall 9.82 sec, GPU wall 7.85 sec, aggregate speedup 1.251x, `cpu_only=0`, `gpu_only=0`; CUDA Viterbi total 0.792 sec, while many profiles had no surviving CPU Viterbi work in the GPU run (`stage_vit` near 0 in GPU logs), so CPU-stage comparison should be taken from CPU outputs when evaluating stage-only speedup.
- Compare validation for the same iteration: `benchmark-data/profmark-current/gpu-audit/vit-samealgo-all13-compare/` and `.../vit-samealgo-broader12-compare/` both had `cpu_only=0`, `gpu_only=0`, and zero `CUDAPREVIT`, `CUDAVIT`, `CUDAFWD`, `CUDAFB` diagnostics.

## Open Risks

- A fully CUDA-native SSV-equivalent boundary remains deferred. A direct diagonal CUDA port over CPU `sbv` was tested and reverted because it did not reproduce CPU `p7_SSVFilter()` behavior.
- Viterbi and Forward score prefilters are parity-clean on the checked profmark samples but still need broader validation and a default/auto-gating policy.
- Forward/Backward parser handoff has useful timing evidence and bounded posterior/domain input differences, but raw `p7X_SCALE` row diagnostics are still open.
- Backward, domain definition, null2, hit storage, thresholding, and output should remain CPU-side for accepted/default paths.

## Verification Guidance

- Use `profmark` for any serious GPU speed claim, not tutorial-sized inputs.
- Build target databases with `hmmseqdb` before running `hmmsearch --gpu`.
- Record CPU/GPU wall time, CUDA H2D/kernel/D2H time, batch sizes, CUDA batch counts, pass counts for MSV/bias/Viterbi/Forward, and final hit deltas.
- Compare each proposed GPU stage against the last accepted GPU baseline, not only against CPU.

## History

Detailed dated implementation notes were moved to `agents_docs/gpu-support-history.md`.
