# GPU Support History (Condensed)

Last updated: 2026-05-05

Purpose: keep a compact record of major GPU attempts, outcomes, and rejected directions. Detailed “daybook” logs are intentionally removed.

## Baseline Direction

- Goal: opt-in protein `hmmsearch --gpu` path with correctness-first parity, then staged acceleration of later filters.
- Accepted core path: CUDA MSV + CUDA bias batching, with CPU-compatible bias/F1 boundary checks to preserve hit parity.

## What Was Tried And Failed

### 1) Tutorial-only timing as a speed claim
- Attempt: used small tutorial datasets for end-to-end speed claims.
- Outcome: rejected as insufficient evidence.
- Failure reason: tutorial-sized inputs hide real bottlenecks and are not representative.
- Decision: require profmark-scale validation for GPU speed claims.

### 2) Aggressive larger MSV batches as default
- Attempt: increased search batch residues (for example 12M/16M) expecting wall-time gains.
- Outcome: not adopted as default.
- Failure reason: fewer launches and faster kernels did not consistently improve end-to-end wall time.
- Decision: keep measured default around 32,768 seqs / 8M residues until broader evidence supports change.

### 3) Naive CUDA SSV-style shortcut from CPU `sbv` profile
- Attempt: direct diagonal signed-byte CUDA shortcut to mimic CPU SSV behavior.
- Outcome: reverted.
- Failure reason: did not match CPU `p7_SSVFilter()` behavior; caused sensitivity regressions.
- Decision: do not pursue this shortcut; any future SSV work must match CPU banded behavior or keep CPU-compatible rescue.

### 4) Early CUDA Viterbi prototype without strong parity/perf envelope
- Attempt: post-bias CUDA Viterbi candidate scoring prototype.
- Outcome: removed.
- Failure reason: runtime regressions and parity issues in tested form.
- Decision: reintroduce only with strict score/status parity checks and profmark speed evidence.

### 5) Initial GPU bias path with duplicate transfer cost
- Attempt: CUDA bias scoring that re-uploaded sequence batch separately.
- Outcome: replaced.
- Failure reason: extra transfer overhead erased expected benefit.
- Decision: reuse MSV-uploaded sequence buffers for bias stage.

## What Was Tried And Worked (But Not Yet Default)

### 1) CUDA Viterbi prefilter (opt-in)
- Outcome: parity-clean on checked all-13 and broader-12 compare runs.
- Limitation: still experimental/default-off pending broader validation and policy.

### 2) CUDA Forward score prefilter (opt-in)
- Outcome: parity-clean on checked compare runs; useful in combination with later stages.
- Limitation: remains opt-in; short-profile and launch-overhead tradeoffs require gating policy.

### 3) CUDA Forward/Backward parser handoff (opt-in)
- Outcome: final hit parity on checked prepared runs; meaningful speedups on selected non-compare runs.
- Limitation: raw scale-row diagnostics remain; requires broader acceptance criteria before default use.

## Current Correctness Boundary

- Active sensitivity risk boundary is bias-corrected F1 near CPU SSV/MSV shortcut behavior.
- Current accepted mitigation:
  - CPU `p7_bg_FilterScore()` supplies final bias score for GPU MSV survivors.
  - CPU `p7_MSVFilter()` rescue remains available at the bias/F1 boundary.
- Rationale: this boundary currently preserves exact parity on prepared all-13 evidence while keeping GPU acceleration useful.

## Current Status Summary

- Exact final hit parity is demonstrated on prepared all-13 and broader-12 checked samples when using current accepted boundary handling.
- Later stages (`--gpu-vit-prefilter`, `--gpu-fwd-prefilter`, `--gpu-fb-parser`) are promising but remain opt-in/default-off by policy.
- Open work is maintained in `gpu-support-todo.md`; current accepted state is maintained in `gpu-support-progress.md`.
