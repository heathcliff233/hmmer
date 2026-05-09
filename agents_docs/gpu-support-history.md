# GPU Support History (Condensed)

Last updated: 2026-05-09

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

### 10) gpudb v2 embedded metadata — zero-I/O resident path (2026-05-07)
- Problem: 10.5% of GPU wall time (0.258s/2.45s) spent reading dsqdata metadata (names, acc, desc) for all 229K sequences every query, even though only ~15 reach the hit stage.
- Solution: extended `.gpudb` format to v2 with embedded metadata section (name\0 acc\0 desc\0 taxid per seq). When gpudb v2 has metadata, `hmmsearch --gpu` skips `esl_dsqdata_OpenSized` entirely. Added `madvise` hints (SEQUENTIAL for upload, RANDOM for metadata) and `cudaHostRegister` for DMA-accelerated GPU upload.
- Outcome: `io_read_unpack` dropped from 0.258s to 0.000s. GPU wall 2.45s → 1.54s (5.86x vs CPU-1).
- Files: `src/p7_gpudb.h`, `src/p7_gpudb.c`, `src/hmmsearch.c`, `src/hmmsearch_gpu.c`, `src/cuda/p7_cuda_runtime.cu`.

### 11) Register-based warp-shuffle Viterbi kernel (2026-05-07)
- Problem: Viterbi kernel was dominant GPU cost (0.918s, 59.6% of wall). Old kernel used 8 threads per sequence (75% warp waste) with shared-memory DP rows.
- Solution: rewrote as 32-thread/warp register-based kernel matching SSV design. Each thread owns contiguous model-node stripe; cross-thread boundaries via `__shfl_up_sync`. Two variants: small (M≤768, stride≤24) and large (M≤2048, stride≤64).
- Outcome: Zero parity errors. Architecture cleaner (no wasted threads, less shmem), but Viterbi kernel time flat — compiler places register arrays in local memory (L1 cache) rather than true registers, making access patterns similar to shared memory. Bottleneck identified as global memory bandwidth (scattered profile accesses), not thread utilization.
- Lesson: for Viterbi's 3-state DP (3 arrays × stride × int16), the register array approach that worked for SSV (1 array × stride × uint8) doesn't translate to true register residence. Future improvement requires profile layout changes or shared-memory preloading.
- Files: `src/cuda/p7_cuda_viterbi.cu`, `src/cuda/p7_cuda_internal.h`.

### 12) Templated Viterbi kernel + tile optimization (2026-05-07)
- Problem: ptxas confirmed Viterbi "register" arrays were placed on stack (384 bytes stack frame, 48 regs) because `stride` was a runtime variable preventing compile-time array sizing. Additionally, tile sizes (1024 max for M≤700) underutilized RTX 4090's 128 SMs, and the tile heuristic had an int32 overflow bug.
- Solution: (a) converted kernel to C++ template `cuda_viterbi_opt_kernel<STRIDE>` with `#pragma unroll` on inner loops. Dispatch via switch(stride) for stride 1–24 (VIT_OPT_MAX_STRIDE_SMALL); larger strides fall back to non-templated variant. (b) Increased Viterbi tile sizes 4x (1024→4096 for M≤700). (c) Fixed integer overflow in `gpu_ChooseTileCandidates`: `tile_n * tile_res` can exceed int32; changed to `int64_t` arithmetic.
- Compiler results (stride=8, M=246): 79 registers, 48-byte stack frame, **zero spills** (was 48 regs + 384-byte stack). Stack residual is `__syncthreads` ABI frame, not DP arrays.
- Outcome: Viterbi kernel **2.9x faster** (0.475s → 0.162s). Launches reduced 43% (91 → 52). GPU SM utilization 78% → 87%. Overall wall 1.63s → 1.45s (10.9% faster). Zero parity errors.
- Lesson: `#pragma unroll` with compile-time template parameter is necessary and sufficient for nvcc to place arrays in true registers. The `break` inside the unrolled loop (for `j >= my_count`) does not prevent register allocation — nvcc still unrolls and uses predicated execution.
- Files: `src/cuda/p7_cuda_viterbi.cu`, `src/hmmsearch_gpu.c`.

### 13) Fused SSV+null+bias+F1 gate kernel with survivor-indexed D2H (2026-05-07)
- Problem: After all stage-level kernel optimizations, `exact_other` (inter-stage overhead) was 246ms — 17% of wall time. This overhead came from 5 separate kernel launches per batch (SSV, null, bias, F1 gating, survivor bias) plus host-side score conversion, intermediate D2H transfers, and multiple synchronization points. Additionally, full-array D2H transferred 1.8MB/batch (raw scores + overflow for 229K sequences) but only ~4K survivors (1.7%) were actually used.
- Solution: Three-part optimization:
  1. **Kernel fusion**: merged SSV + null score + bias filter + F1 Gumbel gate into a single templated kernel (`cuda_ssv_null_bias_gate_kernel<STRIDE>`). Each warp computes SSV score, null score, bias filtersc, and F1 P-value for its sequence in one pass. Survivors written to compact list via `atomicAdd`.
  2. **Linear rbv layout**: added `d_rbv_lin[x * M + k]` profile layout for coalesced inner-loop access (eliminates shared-memory q/z lookup tables from SSV fast-path within fused kernel).
  3. **Survivor-indexed D2H**: kernel writes float `usc` and overflow status directly into per-survivor arrays at the `atomicAdd` point. Host D2H transfers only `nsurv` entries (~32KB) instead of full `nseq` arrays (1.8MB). Eliminates host-side score conversion loop entirely.
- Intermediate attempts that failed:
  - Register-cached rbv offsets (`int rbv_off[STRIDE]`): 80 bytes extra registers caused spilling, regression from 34ms to 55ms. Reverted.
  - `__ldg()` explicit cache hints: no improvement — `const uint8_t*` already uses read-only cache on sm_89. Reverted.
- Results:
  - `exact_other` reduced from 246ms to 115ms (−131ms, −53%)
  - D2H reduced from 28ms to 2.6ms across 13 queries (10.8x reduction)
  - Host `score_convert` eliminated (−1ms)
  - Overall fused path: **1.23s** vs legacy 1.49s (17.4% faster)
  - Hit parity: fused path agrees with legacy GPU path (zero diff). Known precision difference in in-kernel bias gate vs host-side bias causes ±3 survivor count difference per query (accepted: fused bias is length-correct, more accurate).
- API change: `p7_cuda_SSVNullBiasGateResident()` and `p7_cuda_SSVNullBiasGateDsqdataChunk()` now output survivor-indexed `float *survivor_scores` and `int *survivor_statuses` instead of full-array `float *scores` and `int *statuses`.
- Files: `src/cuda/p7_cuda_ssv.cu`, `src/cuda/p7_cuda_internal.h`, `src/cuda/p7_cuda_runtime.cu`, `src/cuda/p7_cuda.h`, `src/hmmsearch_gpu.c`.

### 14) Survivor sort parity fix (2026-05-08)
- Problem: `cpu_only=44` on profmark — 44 hits found by CPU but missed by GPU, including slam-dunk true positives (E=1.5e-39). The bug was introduced by commit ce86e5ac (fused SSV+null+bias+gate kernel with survivor-indexed D2H).
- Root cause: the survivor length-sort in `hmmsearch_gpu.c` (insertion sort for ReconfigLength cache optimization) rearranged `gpu_f1_survivor_idx[]` but did NOT co-sort the parallel `gpu_surv_scores[]` and `gpu_surv_statuses[]` arrays. After sorting, survivor `si`'s index pointed to the correct sequence, but its score/status came from a different survivor. This caused correct-scoring hits to receive wrong (lower) scores and get filtered out at the pre-Viterbi boundary.
- Why not caught earlier: the previous commit's parity comparison was "fused vs legacy GPU" (which uses per-sequence indexed scores, unaffected by the sort). The GPU-vs-CPU comparison was not checked at the same time.
- Solution: 7-line fix — the insertion sort now co-moves all three parallel arrays (`idx`, `scores`, `statuses`) in lockstep.
- Outcome: `cpu_only=0`, `gpu_only=0` on all 13 profmark queries. No speed regression (kernel time unchanged; slight increase in domain-def time from correctly processing 44 more hits).
- Lesson: when parallel arrays are indexed by a compacted survivor position (not by sequence index), any reordering must co-sort all parallel arrays together.
- Files: `src/hmmsearch_gpu.c` (survivor sort block).

### 15) Multi-warp-per-block for SSV + Viterbi opt kernels (2026-05-09)
- Problem: every DP kernel launched as `<<<nseq, 32>>>` — one warp per block, one sequence per block. On sm_89 (RTX 4090, 48 warps/SM max) this caps at theoretical occupancy of 16 blocks/SM × 1 warp/block = 16 warps/SM = 33% of the 48-warp budget.
- Solution: templatize `cuda_ssv_null_bias_gate_kernel` and `cuda_viterbi_opt_kernel` on `(STRIDE, WARPS)`. Each block packs `WARPS` warps of 32 threads, one sequence per warp. Block-shared `s_q`/`s_z` lookup table (read-only, depends only on the profile); per-warp `__shared__` arrays for `use_full_msv[WARPS]`; `__syncthreads()` retained only for the lookup-table prefill, all other barriers are `__syncwarp()` (warp-scoped). `__shfl_*_sync(0xffffffff, ...)` already warp-scoped — unchanged. `__launch_bounds__` clamped to ≤24 blocks/SM for sm_89.
- Auto-tuner: `p7_cuda_DefaultWarpsPerBlock()` queries `cudaGetDeviceProperties` once, picks the W ∈ {1,2,3,4,6,8} maximizing resident warps per SM (with tie-break toward W=4, the empirical sweet spot on sm_89). Hidden flags `--gpu-ssv-warps` / `--gpu-vit-warps` for manual override.
- Verification: `cpu_only=0 gpu_only=0` on the 13-query profmark; no `SSV_COMPARE_MISMATCH` lines at any W; `Exact delta_vs_wall [OK]`.
- Outcome: aggregate speedup on the 13-query profmark held at ~7.3× vs CPU-1 — within run-to-run noise of the W=1 (old shape) baseline. Per-W wall-time medians (one query, 3 trials each): W=1 1.29s, W=2 1.29s, W=3 1.26s, W=4 1.24s, W=6 1.25s, W=8 1.39s. Per-kernel time totals across 13 queries are essentially flat between W=1 and W=4 (SSV ~0.355s, Viterbi ~0.20s, Forward ~0.18s).
- Honest read: the warp-shuffle DP already saturates SM compute pipes at W=1 despite low theoretical occupancy. Multi-warp packing did not produce a measurable kernel-time win on this workload. Infrastructure retained because (a) it costs nothing on this workload (parity preserved, no slowdown), (b) it should help when M is large or batches are small enough that the SM is genuinely empty, and (c) the dispatch pattern is shared with future kernels.
- Files: `src/cuda/p7_cuda_ssv.cu`, `src/cuda/p7_cuda_viterbi.cu`, `src/cuda/p7_cuda_runtime.cu`, `src/cuda/p7_cuda.h`, `src/cuda/p7_cuda_stub.c`, `src/hmmsearch.c`, `src/hmmsearch_internal.h`, `src/hmmsearch_gpu.c`.

### 16) GPU profile reuse (grow-only reallocation) — REVERTED (2026-05-09)
- Problem: 0.39s of the 1.34s multi-query GPU wall time was process-level overhead, hypothesized to be primarily per-query `cudaMalloc`/`cudaFree` (6 pairs × 13 queries = 78 malloc+free calls).
- Attempt: implemented `p7_cuda_msvprofile_Reconfigure()` — reuses device allocations across queries with grow-only reallocation. Only re-uploads data via `cudaMemcpy` on subsequent queries; avoids malloc/free except when the new model is larger.
- Verification: parity preserved (`cpu_only=0 gpu_only=0`), zero compare mismatches.
- Outcome: **no speedup**; process wall time unchanged or slightly worse (1.43–1.51s vs 1.34–1.40s baseline, within thermal noise). The inter-query overhead did decrease by ~0.06s, but per-query work increased by a similar amount.
- Root cause of failure: the `cudaMalloc`/`cudaFree` overhead is negligible compared to the `cudaMemcpy` transfers (which remain the same). The 0.39s gap is dominated by: (1) one-time CUDA context init (~100–200ms, unavoidable), and (2) time between queries outside the per-query stopwatch (HMM file reads, `esl_dsqdata_Close` thread joins). Neither is addressable by profile buffer reuse.
- Lesson: `cudaMalloc` on sm_89 is fast (sub-millisecond bookkeeping); the expensive part is the data transfer, not the allocation. Process-level startup overhead is dominated by CUDA driver initialization, not per-query GPU memory management.
- Decision: reverted. Do not pursue this direction further.

### 17) Forward kernel multi-group packing — REVERTED (2026-05-09)
- Problem: for short-M queries (M≈100–300), `cuda_forward_score_prefix_kernel` uses 128–512 threads/block. Multiple sequences packed per block via named barriers (`bar.sync b, n`) could improve SM utilization.
- Attempt: templatized kernel on `GROUPS ∈ {1, 2, 4, 8}`, named barriers for per-group sync, per-group shared memory slabs.
- Outcome: **18% slower** at G=2/4/8 vs G=1 baseline.
- Root cause: named-barrier synchronization overhead exceeded any occupancy gain. The kernel is instruction-throughput-bound, not occupancy-bound.
- Decision: reverted per plan's explicit ≥5% improvement gate. Do not pursue multi-group packing for the Forward prefix kernel.

### 18) nhmmer scanning Viterbi threshold fix (2026-05-09)
- Problem: GPU nhmmer scanning Viterbi produced ~7x more sub-windows than CPU (e.g., 3475 vs 498 for MADE1), causing `p7_domaindef` to run ~7x longer and making GPU 3-4x slower than CPU-4. Hit parity was badly broken: MADE1 151 vs 465, query_medium 218 vs 648.
- Root cause: two bugs in `cuda_compute_viterbi_thresholds_kernel` in `src/cuda/p7_cuda_viterbi_longtarget.cu`:
  1. **`esl_gumbel_invsurv_device` used `logf(-logf(p))` instead of `logf(-logf(1.0f - p))`** — computing the Gumbel inverse CDF instead of inverse survival function. With F2=0.003, this produced invP≈-11.8 instead of ≈-1.3, lowering thresholds by ~5265 int16 units (0.693 × 10.53 × 721.3).
  2. **Bias nullsc subtraction used `null(loc_window_len)` instead of `null(window_len)`** — the GPU bias filter computes scores using `window_len`, so the null subtraction to isolate composition bias must also use `window_len`, matching `p7_pli_postSSV_LongTarget`.
- Fix: one-line change to Gumbel function (`logf(p)` → `logf(1.0f - p)`), plus nullsc split into `nullsc_loc` and `nullsc_win`.
- Outcome: MADE1 1.35s/151hits → 0.91s/462hits, query_short 1.92s/119hits → 1.40s/363hits, query_medium 5.34s/218hits → 2.85s/648hits. Hit parity now matches CPU within float32 precision (0-3 hit difference).
- Lesson: always compare GPU device-computed values back to CPU reference at each intermediate step; a wrong sign or missing `1-p` is invisible until you download and compare the threshold vector itself.
- Files: `src/cuda/p7_cuda_viterbi_longtarget.cu` (threshold kernel fix).
