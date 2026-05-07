# GPU Timing Analysis

Last updated: 2026-05-07 (benchmark refreshed after default-on flag cleanup)

## Test Configuration

- **Queries**: 13 profmark HMMs (from `benchmark-data/profmark-current/gpu-audit/dbl-bias-run/queries/`)
- **Database**: `pmark.test.gpudb` (229,290 sequences, 92.8 MB, resident on GPU)
- **Flags**: `--gpu --cpu 0` (single-threaded, GPU-only MSV path)
- **Engine reuse**: yes (CUDA init paid once, amortized across 13 queries)

## Timing Instrumentation Reference

The GPU path reports three layers of timing. Understanding their semantics avoids confusion:

### Layer 1: "CUDA *" lines (per-stage device timings)

Measured via `cudaEvent` pairs around individual H2D/kernel/D2H operations inside each CUDA stage function. These are **device-side elapsed times** and can overlap with host work.

### Layer 2: "Stage *" lines (CPU pipeline stage timings)

Wall-clock time spent in each CPU pipeline stage function (`p7_MSVFilter`, `p7_ViterbiFilter`, `p7_ForwardParser`, etc.). In GPU mode, "Stage MSV host" includes the host-side GPU sync wait (not just CPU MSV compute). These **overlap** with CUDA timings by construction.

### Layer 3: "Exact *" lines (exclusive wall-clock buckets)

Designed to partition wall time into non-overlapping buckets that sum to `exact_wall`. Key semantics:

| Bucket | What it measures |
|--------|-----------------|
| `io_read_unpack` | `esl_dsqdata_Read` + chunk unpacking, before GPU work starts |
| `gpu_h2d` | Sum of all CUDA H2D event-measured transfers (MSV + null + bias + vit + fwd + bck) |
| `gpu_kernel` | Sum of all CUDA kernel event-measured times |
| `gpu_d2h` | Sum of all CUDA D2H event-measured transfers |
| `gpu_submit_overhead` | Host time in batch submission logic (currently 0 with resident DB) |
| `gpu_wait_barrier` | Host time blocked on `cudaEventSynchronize` in MSV path |
| `gpu_event_overhead` | Host time in event create/destroy (currently 0 with resident DB) |
| `gpu_dispatch_wait` | Host time blocked waiting for dispatch slot |
| `host_survivor_orchestration` | Per-survivor `gpu_BindSeqView` + `p7_bg_SetLength` + `p7_oprofile_ReconfigLength` |
| `cpu_survivor_total` | Explicit CPU Forward calls inside the survivor loop (only when GPU Fwd is off) |
| `cpu_postfwd_domain_null2_output` | CPU domain definition + null2 + output (post-Forward work) |
| `survivor_loop_other` | **Wall time of the entire survivor processing loop** (lines 935–1069 in `hmmsearch_gpu.c`). Includes bias filter, PreViterbi boundary, CPU pipeline stages, and seq materialization. NOT "unaccounted residual" — it is the loop's total wall clock. |
| `vit_fwd_dispatch` | Wall time of the Viterbi+Forward+FB dispatch block (lines 1071–1387). Includes GPU Vit/Fwd kernel launches, CPU fallback Vit/Fwd, and FB parser batch. **Overlaps** with `gpu_kernel` and `cpu_survivor_total` when later-stage GPU flags are active. |
| `other` | `wall - known_sum`. Residual unaccounted time. |

**Important**: With all GPU stages active (default), `vit_fwd_dispatch` **overlaps** with `gpu_kernel` — the "Exact" buckets are not fully exclusive.

### Verification

`Exact delta_vs_wall` must print `[OK]` (within 1e-6 sec). This confirms `exact_total == exact_wall`.

## Mode 1: MSV+bias only (historical, no longer the default)

This mode is retained for reference only. With all stages now default-on, Mode 1 is only reachable if the code is modified to disable later stages.

Wall time: ~2.94s total (historical). CPU Viterbi (33.6%), GPU kernel (19.7%), CPU Fwd+Bck (26.5%) were the main costs.

## Current Default: Full GPU (`--gpu`, all stages on)

Wall time: **2.45s** total, **0.188s** avg/query (13 queries, 229K seqs, resident DB)

| Component | Time (s) | % of wall | Notes |
|-----------|----------|-----------|-------|
| GPU Viterbi kernel | 0.475 | 19.4% | Dominant single cost |
| Stage MSV host (read+kernel) | 0.398 | 16.2% | Includes I/O overlap |
| Unaccounted residual (exact_other) | 0.334 | 13.6% | Inter-stage sync, host overhead |
| I/O read + unpack | 0.258 | 10.5% | dsqdata metadata loading |
| GPU Forward kernel | 0.243 | 9.9% | |
| CPU post-Fwd/domain/null2 | 0.162 | 6.6% | Domain def + null2 + output |
| GPU Bias kernel | 0.059 | 2.4% | |
| GPU Backward kernel | 0.049 | 2.0% | |
| GPU H2D + D2H | 0.003 | 0.1% | Negligible with resident DB |

**GPU utilization**: ~34% of wall time is GPU kernel execution (0.827s kernel / 2.45s wall). The remaining time is CPU-bound post-processing, I/O, and inter-stage overhead.

**Filter funnel** (avg per query): 229K → 4213 past MSV → 4052 past bias → 288 past Vit → 15 past Fwd

## Optimization Opportunities (ranked by impact, current benchmark)

### 1. GPU Viterbi kernel — 19.4% of wall

The single largest cost (0.475s across 13 queries). Average 544 candidates/launch, ~217K residues/launch.
- **Opportunity**: Occupancy tuning (shared memory vs register pressure), warp-level DP with shuffle-based state propagation (as done for SSV), profile tiling for large M.

### 2. Unaccounted residual (exact_other) — 13.6% of wall

0.334s not attributed to any measured bucket. Likely inter-stage synchronization, host-side score conversion, F1 gating logic, and survivor list construction between kernel launches.
- **Opportunity**: Instrument further to identify. Fuse MSV + null + bias into a single kernel to eliminate 2 sync points per batch. Move score conversion to GPU.

### 3. I/O read + unpack — 10.5% of wall

0.258s reading dsqdata metadata. With resident DB, sequence data is already on GPU; this cost is chunk metadata (names, accessions, descriptions) loaded for potential hit reporting.
- **Opportunity**: Lazy-load metadata only for sequences that reach the hit stage (~15 per query). Or overlap I/O with GPU kernel via double-buffering.

### 4. GPU Forward kernel — 9.9% of wall

0.243s total. Same optimization approaches as Viterbi apply.
- **Opportunity**: Occupancy tuning, kernel fusion with Backward (both traverse the same sequence), async overlap with CPU post-Fwd work.

### 5. CPU domain definition — 6.6% of wall

0.162s for `p7_domaindef_ByPosteriorHeuristics()` on ~15 sequences/query that pass Forward.
- **Opportunity**: Parallelize across survivors (thread pool for domain work). Eventually move posterior decoding to GPU (high complexity, deferred).

### 6. Kernel fusion (MSV → bias) — reduces sync overhead

Currently MSV and bias are separate kernel launches with host-side sync between them. Fusing would eliminate one round-trip.
- **Opportunity**: Single kernel that runs SSV → null → bias → F1 gating, producing compact survivor list directly.

## Comparison: GPU vs CPU

Current benchmark (13 queries, 229K seqs, single-process, all GPU stages default-on):

| Config | Wall time | Speedup |
|--------|-----------|---------|
| CPU 1-thread | 10.68s | 1.00x |
| CPU 4-thread | 4.79s | 2.23x |
| GPU (all stages) | 2.45s | **4.36x** vs CPU-1, **1.96x** vs CPU-4 |

**Critical benchmark note**: The profmark runner was previously invoking hmmsearch once per query (13 separate processes), paying ~0.35s CUDA init each time = 4.55s pure overhead. This made GPU appear 0.6x vs CPU-4. The fix: `test-speed/x-hmmsearch-gpu-profmark` now concatenates all HMMs and runs a single hmmsearch process for CPU and GPU respectively, matching real multi-query usage.

