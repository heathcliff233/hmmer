# GPU Support History (Condensed)

Last updated: 2026-05-07

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

### 1) Later-stage CUDA prefilters (now default-on)
- CUDA Viterbi, Forward, and Forward/Backward parser: all parity-clean on checked compare runs.
- Now enabled by default with `--gpu` (previously required separate opt-in flags).

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
  - Double-precision GPU survivor kernel (`cuda_bias_filter_survivors_kernel`) recomputes filtersc with `log()` for F1 survivors, achieving bit-identical parity with CPU `p7_bg_FilterScore`. No CPU bias or MSV calls remain in the GPU path.
- Rationale: two-stage approach (fast float32 pre-filter + exact double-precision recompute for survivors) gives both speed and exact parity. CPU MSV fallback was removed as dead code since the double-precision GPU bias is authoritative.

### 5) Double-precision GPU survivor bias kernel (2026-05-07)
- Problem: GPU float32 bias kernel (`logf()`) diverges from CPU (`log()`) by up to 4.75 nats, breaking downstream F2/F3 threshold decisions. CPU `p7_bg_FilterScore` was required for exact parity.
- Failed attempt: full double-precision bias kernel for all 229K sequences — 14x slower (0.06s → 0.85s), not viable.
- Solution: keep fast float32 kernel for F1 gating (conservative pre-filter), add a second kernel that recomputes filtersc with double-precision `log()` only for ~4000 F1 survivors.
- Precision recipe (matching CPU exactly): float DP values, `(float)log((double)max)` for scale factors, float accumulation, `logf` for length correction. Per-sequence `t[0][0] = L/(L+1)` computed in-kernel.
- Outcome: zero CUDAPREVIT mismatches (bit-identical to CPU). CPU `p7_bg_FilterScore` completely eliminated from GPU path.
- Files: `cuda_bias_filter_survivors_kernel` in `src/cuda/p7_cuda_bias.cu`, wrapper `p7_cuda_BiasFilterSurvivors`.

### 4) Standalone GPU SSV kernel (2026-05-06 → 2026-05-07)
- Attempt: extract the inline SSV fast-path from the monolithic MSV kernel into a standalone architecture.
- **Phase 1** (two-pass): SSV kernel first (no J-state), then MSV fallback kernel only for uncertain sequences (~0.3%). Used `rbv` (unsigned byte) profile — same as the monolithic MSV kernel — avoiding the `sbv` pitfall of attempt #3. Parity-verified but ~15–20% slower due to host round-trip between passes.
- **Phase 2** (fused): eliminated the two-pass overhead by fusing the MSV fallback directly into the SSV kernel. The fused kernel runs SSV as the primary fast-path with in-kernel MSV fallback for sequences that need J-state. No intermediate copies, no second kernel launch. Performance-equivalent to monolithic MSV (<0.2% delta).
- **Phase 3** (register-optimized): exploited SSV's constant-xB property for register-based DP. Key changes: (1) precomputed q/z lookup tables in shared memory eliminate integer division from inner loop, (2) each thread owns a contiguous stripe of model nodes in registers instead of shared memory, (3) `__shfl_up_sync` replaces `__syncthreads()` for the single cross-thread boundary value. Result: 1.36x kernel speedup over monolithic MSV (376ms vs 511ms on all-13 profmark), with 1.57-1.79x for larger profiles.
- **Phase 4** (default, 2026-05-07): made SSV the default GPU MSV path. Added `p7_cuda_SSVFilterResident()` for resident-database mode. Monolithic MSV retained only for `--gpu-ssv-compare` debug mode.
- Outcome: adopted as default GPU MSV kernel. 1.36x faster than monolithic MSV, contributing to 1.32x → 1.48x aggregate speedup.
- Parity: bitwise-identical scores to monolithic MSV on all-13 profmark (229,290 sequences × 13 queries, zero mismatches via `--gpu-ssv-compare`). Zero hit-level parity differences.
- Files: `src/cuda/p7_cuda_ssv.cu`, CLI flags in `hmmsearch.c`.

### 6) Survivor loop optimization (2026-05-07)
- Problem: per-survivor overhead from (a) CPU MSV fallback for bias-boundary sequences, (b) redundant ReconfigLength calls for same-length sequences.
- Solution: (a) removed CPU MSV fallback entirely — double-precision GPU bias kernel is authoritative and bit-identical to CPU. (b) sort survivors by sequence length before processing, cache last ReconfigLength call to skip redundant `p7_bg_SetLength`/`p7_oprofile_ReconfigLength`.
- Outcome: survivor loop time reduced from ~0.01s to ~0.003s. Combined with SSV default and M-limit changes, aggregate speedup improved from 1.32x to 1.48x.
- Files: `src/hmmsearch_gpu.c` (PreViterbiBoundary, survivor loop).

### 7) Viterbi/Forward M-limit expansion (2026-05-07)
- Problem: GPU Viterbi was gated at M≤512 and Forward at M≤256, despite shared memory easily supporting larger models.
- Solution: raised Viterbi default to M≤2048 (at M=2048, Qw=256, shmem=24576 bytes per group, fits in 48KB). Raised Forward default to M≤1024. Added M>2048 tiling tier with smaller batch sizes.
- Outcome: more queries use GPU Viterbi/Forward instead of falling back to CPU. No parity issues.
- Files: `src/hmmsearch_gpu.c` (M gates, tiling function).

## Current Status Summary

- Exact final hit parity is demonstrated on prepared all-13 and broader-12 checked samples when using current accepted boundary handling.
- Later stages (Viterbi prefilter, Forward prefilter, FB parser) are now default-on with `--gpu`.
- Open work is maintained in `gpu-support-todo.md`; current accepted state is maintained in `gpu-support-progress.md`.

### 8) GPU pipeline overlap and efficiency refinements (2026-05-07)
- Problem: GPU pipeline had redundant synchronization, per-call event creation, and repeated bias parameter uploads. Combined: ~0.05s wasted per query.
- Solution: (a) use pre-allocated engine events for all SSV/bias functions (eliminated ~126 event create/destroy per query), (b) remove redundant `cudaEventSynchronize` after synchronous `cudaMemcpy`, (c) cache bias model parameters with `engine->bias_params_uploaded` flag, (d) add ReconfigLength caching to Viterbi/Forward post-processing loops.
- Rejected sub-optimization: CPU null score computation (`logf` on host) broke parity — x86 and CUDA `logf` implementations differ enough to flip borderline threshold decisions. GPU null kernel retained.
- Outcome: event overhead eliminated, ReconfigLength calls reduced. Combined with multi-query benchmark fix, GPU achieves 1.92x vs CPU-4 and 4.19x vs CPU-1 on all-13 profmark.
- Files: `src/cuda/p7_cuda_ssv.cu`, `src/cuda/p7_cuda_bias.cu`, `src/cuda/p7_cuda_runtime.cu`, `src/cuda/p7_cuda_internal.h`, `src/hmmsearch_gpu.c`.

### 9) Multi-query profmark benchmark fix (2026-05-07)
- Problem: `test-speed/x-hmmsearch-gpu-profmark` invoked hmmsearch once per query (13 separate processes). Each GPU invocation paid ~0.35s CUDA initialization overhead, inflating GPU total by ~4.55s. This made GPU appear 0.6x vs CPU-4, when in reality a single-process multi-query search shows 2.20x vs CPU-4.
- Solution: rewrote the profmark runner to concatenate all HMMs into one file, `hmmpress` it, and run hmmsearch once for CPU and once for GPU. Per-query hit parity parsed from tblout (column 3 = query name). Per-query GPU timing parsed from output blocks between `Query:` and `//` delimiters.
- Outcome: benchmark now correctly reports GPU 1.92x vs CPU-4 and 4.19x vs CPU-1.
- Files: `test-speed/x-hmmsearch-gpu-profmark`.
