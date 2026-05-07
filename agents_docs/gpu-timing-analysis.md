# GPU Timing Analysis

Last updated: 2026-05-07 (fused SSV+null+bias+gate kernel with survivor-indexed D2H)

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

## Current Default: Full GPU (`--gpu`, fused kernel, all stages on)

Wall time: **1.23s** total, **0.095s** avg/query (13 queries, 229K seqs, resident DB, fused SSV+null+bias+gate)

| Component | Time (s) | % of wall | Notes |
|-----------|----------|-----------|-------|
| GPU kernel (all stages) | 0.795 | 64.6% | SSV+null+bias+gate fused + Viterbi + Forward + Backward |
| CPU post-Fwd/domain/null2 | 0.165 | 13.4% | Domain def + null2 + output |
| Inter-stage overhead (exact_other) | 0.115 | 9.3% | Reduced from 0.246s by kernel fusion |
| Vit/Fwd dispatch overlap | 0.100 | 8.1% | Overlaps with gpu_kernel |
| GPU D2H | 0.003 | 0.2% | Survivor-indexed: ~32KB/batch vs prior 1.8MB |
| GPU H2D | 0.002 | 0.2% | Negligible with resident DB |
| Host survivor orchestration | 0.010 | 0.8% | BindSeqView + ReconfigLength |
| I/O read + unpack | 0.000 | 0.0% | Eliminated by gpudb v2 |

**GPU utilization**: ~65% of wall time is GPU kernel execution (0.795s kernel / 1.23s wall). The remaining time is CPU-bound post-processing and inter-stage overhead.

**Filter funnel** (avg per query): 229K → 4213 past MSV+bias (fused F1 gate) → 288 past Vit → 15 past Fwd

## Optimization Opportunities (ranked by impact, current benchmark)

### 1. Inter-stage overhead (exact_other) — 9.3% of wall

0.115s not attributed to measured kernel/D2H buckets. Likely remaining sources: Viterbi/Forward batch construction, CUDA event record overhead, host-side survivor list sorting, and F1 survivor bias recomputation.
- **Opportunity**: Profile with nsys to identify exact hotspots. Consider fusing F1 survivor bias into the main fused kernel (currently still a separate `cuda_bias_filter_survivors_kernel` call for double-precision recompute).

### 2. CPU domain definition — 13.4% of wall

0.165s for `p7_domaindef_ByPosteriorHeuristics()` on ~15 sequences/query that pass Forward.
- **Opportunity**: Parallelize across survivors (thread pool for domain work). Eventually move posterior decoding to GPU (high complexity, deferred).

### 3. GPU Forward kernel — part of gpu_kernel

Same optimization approaches as Viterbi. The Forward kernel could be templated similarly to Viterbi.
- **Opportunity**: Template on stride for register residence, kernel fusion with Backward.

### 4. Viterbi/Forward dispatch overlap — 8.1% of wall

The dispatch block overlaps with GPU kernel time. Reducing dispatch overhead or hiding more work behind kernel execution could shrink this.
- **Opportunity**: Overlap CPU survivor work with next query's GPU kernel via streams or pipelining.

## Comparison: GPU vs CPU

Current benchmark (13 queries, 229K seqs, single-process, fused kernel, all GPU stages default-on):

| Config | Wall time | Speedup |
|--------|-----------|---------|
| CPU 1-thread | 10.68s | 1.00x |
| CPU 4-thread | 4.79s | 2.23x |
| GPU (fused, all stages) | 1.23s | **8.68x** vs CPU-1, **3.89x** vs CPU-4 |
| GPU (legacy pipeline) | 1.49s | 7.17x vs CPU-1, 3.21x vs CPU-4 |

**Critical benchmark note**: The profmark runner was previously invoking hmmsearch once per query (13 separate processes), paying ~0.35s CUDA init each time = 4.55s pure overhead. This made GPU appear 0.6x vs CPU-4. The fix: `test-speed/x-hmmsearch-gpu-profmark` now concatenates all HMMs and runs a single hmmsearch process for CPU and GPU respectively, matching real multi-query usage.

