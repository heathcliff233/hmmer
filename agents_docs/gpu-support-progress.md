# GPU Support Progress

Last updated: 2026-05-05

## Current State

- `hmmsearch --gpu` is an opt-in, protein-only CUDA path. It requires target input built by `hmmseqdb` as Easel protein `dsqdata`; ordinary FASTA stays on the CPU path.
- GPU code lives under `src/cuda/` with stage-owned CUDA files (`p7_cuda_msv.cu`, `p7_cuda_bias.cu`, `p7_cuda_viterbi.cu`, `p7_cuda_forward.cu`, `p7_cuda_fb_parser.cu`) and shared runtime/profile ownership in `p7_cuda_runtime.cu` + `p7_cuda_internal.h`.
- `src/cuda_msv.h` is only a compatibility wrapper for existing callers; new CUDA-facing code should include `src/cuda/p7_cuda.h`.
- The default accepted GPU path accelerates MSV and computes the biased-composition filter score in CUDA batches while reusing the MSV-uploaded sequence batch.
- Exact hit parity currently depends on CPU-compatible checks at the bias-corrected F1 boundary: CPU `p7_bg_FilterScore()` supplies the final bias score for GPU MSV survivors, and CPU `p7_MSVFilter()` can rescue bias-boundary rejects through the optimized CPU SSV shortcut.
- Resident survivor-core work now keeps Viterbi/F3 pass decisions GPU-side in normal mode, copies only status/pass buffers when full scores are not needed, and materializes `ESL_SQ` metadata only for sequences entering CPU post-Fwd/domain work or compare diagnostics.
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

- Correctness-first all-13 profmark run in `benchmark-data/profmark-current/gpu-audit/cpu-bias-msv-rescue-all13/` with `--gpu-vit-prefilter --gpu-fwd-prefilter --gpu-fb-parser`: CPU wall 26.82 sec, GPU wall 16.32 sec, aggregate speedup 1.643x, median speedup 1.508x, `cpu_only=0`, `gpu_only=0`.
- Broader 12-query profmark run in `benchmark-data/profmark-current/gpu-audit/broader12-nocompare/` with the same later-stage flags: CPU wall 7.78 sec, GPU wall 7.05 sec, aggregate speedup 1.104x, `cpu_only=0`, `gpu_only=0`.
- The broader 12-query MSV/bias baseline in `benchmark-data/profmark-current/gpu-audit/broader12-msvbias-baseline/` recorded GPU wall 7.88 sec; the later-stage path saved 0.83 sec but regressed a few short profiles slightly, so later stages remain default-off pending auto-gating policy.
- Compare-mode runs are validation evidence, not speed claims. The current compare runs produced zero `CUDAVIT` and zero `CUDAFWD` diagnostics; `CUDAFB` raw scale-row diagnostics remain, but domain-decoding input maxima stayed below 0.00002 in the all-13 compare run.
- Viterbi same-algorithm optimization (2026-05-05) in `benchmark-data/profmark-current/gpu-audit/vit-samealgo-all13-nocompare/`: CPU wall 28.66 sec, GPU wall 26.61 sec, aggregate speedup 1.077x, `cpu_only=0`, `gpu_only=0`; CUDA Viterbi total (H2D+kernel+D2H) 0.945 sec vs CPU stage Viterbi 3.861 sec from CPU logs (about 4.09x stage speedup; kernel 0.941 sec).
- Same iteration, broader 12 non-compare in `benchmark-data/profmark-current/gpu-audit/vit-samealgo-broader12-nocompare/`: CPU wall 9.82 sec, GPU wall 7.85 sec, aggregate speedup 1.251x, `cpu_only=0`, `gpu_only=0`; CUDA Viterbi total 0.792 sec, while many profiles had no surviving CPU Viterbi work in the GPU run (`stage_vit` near 0 in GPU logs), so CPU-stage comparison should be taken from CPU outputs when evaluating stage-only speedup.
- Compare validation for the same iteration: `benchmark-data/profmark-current/gpu-audit/vit-samealgo-all13-compare/` and `.../vit-samealgo-broader12-compare/` both had `cpu_only=0`, `gpu_only=0`, and zero `CUDAPREVIT`, `CUDAVIT`, `CUDAFWD`, `CUDAFB` diagnostics.
- Post-Viterbi Forward/Backward refinement run (2026-05-05) added a pure-GPU F3 decision policy in normal mode and enabled the intended block-parallel Forward parser batch launch.
  - Compare mode all-13 run: `benchmark-data/profmark-current/gpu-audit/postvit-fwdbck-refine-all13-compare-20260505/`
    - Command:
      - `python3 test-speed/x-hmmsearch-gpu-profmark . benchmark-data/profmark-current/work benchmark-data/profmark-current/gpu-audit/postvit-fwdbck-refine-all13-compare-20260505 14-3-3 2-Hacid_dh 23S_rRNA_IVP 2OG-FeII_Oxy_2 3keto-disac_hyd 4TM_phosphoesterase 7tm_1 7tm_3 A2M_BRD AAA_15 ATG2_CAD Nup192 Tra1_ring --gpu-vit-prefilter --gpu-fwd-prefilter --gpu-fb-parser --gpu-previt-compare --gpu-vit-compare --gpu-fwd-compare --gpu-fb-compare --gpu-vit-min-seqs 1 --gpu-fwd-min-seqs 1`
    - Aggregate wall summary: CPU wall 30.86 sec, GPU wall 81.64 sec (compare mode intentionally heavy), `cpu_only=0`, `gpu_only=0`.
    - Compare diagnostics:
      - `CUDAVIT_SUMMARY` present for all 13 queries with `status_mismatch=0`, `pass_mismatch=0`, `score_drift=0`.
      - `CUDAFWD`: zero lines (no Forward pass/fail or score mismatches reported).
      - `CUDAFB`: present (139 lines) with bounded DomainDecoding inputs; maxima from diagnostics sweep:
        - `max_mocc <= 0.000007`
        - `max_btot <= 0.000012`
        - `max_etot <= 0.000019`
      - `CUDAFB` raw `xmx` row deltas remain (up to `max_fwd=max_bck=0.9375`) but did not produce hit-set parity failures.
  - Non-compare all-13 run: `benchmark-data/profmark-current/gpu-audit/postvit-fwdbck-refine-all13-nocompare-20260505/`
    - Command:
      - `python3 test-speed/x-hmmsearch-gpu-profmark . benchmark-data/profmark-current/work benchmark-data/profmark-current/gpu-audit/postvit-fwdbck-refine-all13-nocompare-20260505 14-3-3 2-Hacid_dh 23S_rRNA_IVP 2OG-FeII_Oxy_2 3keto-disac_hyd 4TM_phosphoesterase 7tm_1 7tm_3 A2M_BRD AAA_15 ATG2_CAD Nup192 Tra1_ring --gpu-vit-prefilter --gpu-fwd-prefilter --gpu-fb-parser --gpu-vit-min-seqs 1 --gpu-fwd-min-seqs 1`
    - Aggregate wall summary: CPU wall 29.69 sec, GPU wall 16.57 sec, aggregate speedup 1.792x, `cpu_only=0`, `gpu_only=0`.
    - Stage-speed gate:
      - `CPU_FWD_BCK = sum(stage_fwd + stage_bck) = 2.622675 sec`
      - `GPU_KERNEL_FWD_BCK = sum(gpu_fwd_kernel + gpu_bck_kernel) = 0.643210 sec`
      - `CPU_FWD_BCK / GPU_KERNEL_FWD_BCK = 4.077x` (passes 3.0x gate).
  - Regression sanity baseline (MSV/bias-only path): `benchmark-data/profmark-current/gpu-audit/postvit-fwdbck-refine-baseline-all13-20260505/`
    - Command:
      - `python3 test-speed/x-hmmsearch-gpu-profmark . benchmark-data/profmark-current/work benchmark-data/profmark-current/gpu-audit/postvit-fwdbck-refine-baseline-all13-20260505 14-3-3 2-Hacid_dh 23S_rRNA_IVP 2OG-FeII_Oxy_2 3keto-disac_hyd 4TM_phosphoesterase 7tm_1 7tm_3 A2M_BRD AAA_15 ATG2_CAD Nup192 Tra1_ring`
    - Baseline aggregate: CPU wall 28.42 sec, GPU wall 19.49 sec, aggregate speedup 1.458x, `cpu_only=0`, `gpu_only=0`.
    - Stage-counter sanity versus refined non-compare run:
      - Total `past_msv`, `past_bias`, `past_vit`, `past_fwd` matched exactly (`117182`, `54504`, `4152`, `210`).
  - Current bottleneck signal from the same non-compare all-13 run:
    - Largest aggregate bucket is host survivor continuation (`gpu_survivor = 4.481900 sec`).
    - Next largest CPU-side stage totals are `stage_msv_host = 3.123544 sec`, `stage_vit = 2.578773 sec`, and `stage_fwd = 2.191749 sec`.
    - This confirms end-to-end speed is still mostly constrained by CPU continuation after GPU filtering, even though FWD/BCK kernel compute passes the 3x stage-speed gate.
- Resident survivor-core timing/decision migration run (2026-05-05) added exact-sum timing, GPU-side Viterbi/F3 decision buffers, delayed sequence metadata materialization, and zero-copy `P7_OMX` special-state views over batched parser output.
  - Compare mode all-13 run: `benchmark-data/profmark-current/gpu-audit/survivor-core-all13-compare-20260505/`
    - Command:
      - `python3 test-speed/x-hmmsearch-gpu-profmark . benchmark-data/profmark-current/work benchmark-data/profmark-current/gpu-audit/survivor-core-all13-compare-20260505 14-3-3 2-Hacid_dh 23S_rRNA_IVP 2OG-FeII_Oxy_2 3keto-disac_hyd 4TM_phosphoesterase 7tm_1 7tm_3 A2M_BRD AAA_15 ATG2_CAD Nup192 Tra1_ring --gpu-vit-prefilter --gpu-fwd-prefilter --gpu-fb-parser --gpu-previt-compare --gpu-vit-compare --gpu-fwd-compare --gpu-fb-compare --gpu-vit-min-seqs 1 --gpu-fwd-min-seqs 1`
    - Aggregate wall summary: CPU wall 28.73 sec, GPU wall 80.56 sec (compare mode intentionally heavy), `cpu_only=0`, `gpu_only=0`.
    - Boundary diagnostics: `CUDAPREVIT` status/pass mismatches were zero; `CUDAVIT_SUMMARY` totals had `status_mismatch=0`, `pass_mismatch=0`, `score_drift=0`; `CUDAFWD` emitted zero mismatch lines.
    - `CUDAFB` emitted 139 bounded parser diagnostics; all GPU outputs printed `Exact delta_vs_wall: 0.000000000 sec [OK]`.
  - Non-compare all-13 run: `benchmark-data/profmark-current/gpu-audit/survivor-core-all13-nocompare-20260505/`
    - Command:
      - `python3 test-speed/x-hmmsearch-gpu-profmark . benchmark-data/profmark-current/work benchmark-data/profmark-current/gpu-audit/survivor-core-all13-nocompare-20260505 14-3-3 2-Hacid_dh 23S_rRNA_IVP 2OG-FeII_Oxy_2 3keto-disac_hyd 4TM_phosphoesterase 7tm_1 7tm_3 A2M_BRD AAA_15 ATG2_CAD Nup192 Tra1_ring --gpu-vit-prefilter --gpu-fwd-prefilter --gpu-fb-parser --gpu-vit-min-seqs 1 --gpu-fwd-min-seqs 1`
    - Aggregate wall summary: CPU wall 29.64 sec, GPU wall 18.14 sec, aggregate speedup 1.634x, `cpu_only=0`, `gpu_only=0`.
    - Exact timing reconciled on all 13 GPU outputs (`Exact delta_vs_wall` within `1e-6` and printed `[OK]`).
    - Stage-speed gate still passed: `CPU_FWD_BCK / GPU_KERNEL_FWD_BCK = 4.678x`.
    - Performance gates did not pass on this run: `gpu_survivor = 5.330384 sec` versus target `<= 3.14 sec`, and GPU wall 18.14 sec versus target `<= 16.57 sec`.
  - Large-profile activation experiments were measured and reverted:
    - Widening both Viterbi and Forward/parser activation was parity-clean on `ATG2_CAD`, `Nup192`, and `Tra1_ring`, but GPU wall rose to 135.69 sec in compare mode because large-profile Forward/parser kernels dominated.
    - Widening only Viterbi reduced hot3 `gpu_survivor` from the previous 4.330335 sec to 1.771558 sec, but worsened hot3 non-compare GPU wall from 9.34 sec to 11.88 sec because the current large-profile CUDA Viterbi kernel was slower than the CPU SSE continuation.
    - The accepted code therefore retains the existing conservative Viterbi/Forward activation caps while preserving the new GPU-side decision APIs for the profile sizes where they are beneficial.

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
