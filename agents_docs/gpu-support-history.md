# GPU Support History (Condensed)

Last updated: 2026-05-06

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
- Failure reason: did not match CPU `p7_SSVFilter()` behavior; caused sensitivity regressions. The `sbv` (signed byte) profile uses a different score representation than the `rbv` (unsigned byte) profile used by the GPU MSV kernel.
- Decision: do not pursue this shortcut; any future SSV work must use the `rbv` profile and match the existing GPU MSV scoring logic, not the CPU SSV diagonal traversal.

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

## What Was Tried And Worked

### 1) Later-stage CUDA prefilters (opt-in, default-off)
- CUDA Viterbi, Forward, and Forward/Backward parser: all parity-clean on checked compare runs.
- Remain opt-in (`--gpu-vit-prefilter`, `--gpu-fwd-prefilter`, `--gpu-fb-parser`) pending broader validation and auto-gating policy for short profiles.

### 2) GPU-side F1 gating
- Moved MSV/bias P-value gating to a CUDA kernel; batch loop now iterates only over compact survivor indices.
- Adopted for architectural cleanliness. Performance gain was minimal (~5-10ms/query) because `exact_other` is dominated by CUDA host-side API overhead, not the per-sequence CPU loop.

### 3) Bulk smem copy
- Replaced 32K individual `memcpy` calls per batch with a single bulk copy of the dsqdata chunk's contiguous `smem` buffer (L+1 offset spacing).
- Pack loop halved (24ms → 12ms), warm-query wall reduced 19%.
- Key insight: sequences in `chu->smem` are already contiguous with overlapping sentinels; CUDA kernels access only positions 1..L.
- Limitation: multi-chunk batch views (`smem=NULL`) fall back to per-sequence copy.

## Current Correctness Boundary

- Active sensitivity risk boundary is bias-corrected F1 near CPU SSV/MSV shortcut behavior.
- Current accepted mitigation:
  - CPU `p7_bg_FilterScore()` supplies final bias score for GPU MSV survivors.
  - CPU `p7_MSVFilter()` rescue remains available at the bias/F1 boundary.
- Rationale: this boundary currently preserves exact parity on prepared all-13 evidence while keeping GPU acceleration useful.

### 4) Standalone GPU SSV kernel (2026-05-06 → 2026-05-07)
- Attempt: extract the inline SSV fast-path from the monolithic MSV kernel into a standalone architecture.
- **Phase 1** (two-pass): SSV kernel first (no J-state), then MSV fallback kernel only for uncertain sequences (~0.3%). Used `rbv` (unsigned byte) profile — same as the monolithic MSV kernel — avoiding the `sbv` pitfall of attempt #3. Parity-verified but ~15–20% slower due to host round-trip between passes.
- **Phase 2** (fused): eliminated the two-pass overhead by fusing the MSV fallback directly into the SSV kernel. The fused kernel runs SSV as the primary fast-path with in-kernel MSV fallback for sequences that need J-state. No intermediate copies, no second kernel launch. Performance-equivalent to monolithic MSV (<0.2% delta).
- **Phase 3** (register-optimized): exploited SSV's constant-xB property for register-based DP. Key changes: (1) precomputed q/z lookup tables in shared memory eliminate integer division from inner loop, (2) each thread owns a contiguous stripe of model nodes in registers instead of shared memory, (3) `__shfl_up_sync` replaces `__syncthreads()` for the single cross-thread boundary value. Result: 1.36x kernel speedup over monolithic MSV (376ms vs 511ms on all-13 profmark), with 1.57-1.79x for larger profiles.
- Outcome: adopted as opt-in (`--gpu-ssv`), 1.36x faster than monolithic MSV on all-13 profmark.
- Parity: bitwise-identical scores to monolithic MSV on all-13 profmark (229,290 sequences × 13 queries, zero mismatches via `--gpu-ssv-compare`). Zero hit-level parity differences.
- Files: `src/cuda/p7_cuda_ssv.cu`, CLI flags in `hmmsearch.c`.

## Current Status Summary

- Exact final hit parity is demonstrated on prepared all-13 and broader-12 checked samples when using current accepted boundary handling.
- Later stages (`--gpu-vit-prefilter`, `--gpu-fwd-prefilter`, `--gpu-fb-parser`) are promising but remain opt-in/default-off by policy.
- Open work is maintained in `gpu-support-todo.md`; current accepted state is maintained in `gpu-support-progress.md`.
