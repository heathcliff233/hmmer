# GPU Timing Analysis

Last updated: 2026-05-07

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

**Important**: In Mode 1 (default `--gpu`), `vit_fwd_dispatch` is ~0 because Vit/Fwd run inside `survivor_loop_other`. In Mode 2 (with `--gpu-vit-prefilter` etc.), `vit_fwd_dispatch` becomes large and **overlaps** with `gpu_kernel` — the "Exact" buckets are no longer fully exclusive.

### Verification

`Exact delta_vs_wall` must print `[OK]` (within 1e-6 sec). This confirms `exact_total == exact_wall`.

## Mode 1: Default `--gpu` (MSV+bias on GPU only)

Wall time: **2.94s** total, **0.226s** avg/query

| Component | Time (s) | % of wall | Notes |
|-----------|----------|-----------|-------|
| CPU Viterbi (survivors) | 0.99 | 33.6% | Largest single cost |
| GPU MSV+bias kernel | 0.58 | 19.7% | Device compute |
| CPU Backward (survivors) | 0.43 | 14.5% | |
| CPU Forward (survivors) | 0.35 | 12.0% | |
| Survivor loop overhead | 0.21 | 7.3% | Seq materialization, bg/oprofile reconfig, alloc |
| I/O read + unpack | 0.19 | 6.5% | dsqdata chunk read |
| CPU domain definition | 0.10 | 3.4% | |
| CPU bias filter | 0.08 | 2.6% | GPU double-precision survivor bias |
| GPU H2D + D2H transfers | 0.01 | 0.4% | Negligible with resident DB |
| Host survivor orchestration | 0.02 | 0.7% | BindSeqView + SetLength + ReconfigLength |

**GPU utilization**: 20% of wall time is GPU kernel execution. The remaining 80% is CPU-bound survivor processing.

## Mode 2: Full GPU (`--gpu-vit-prefilter --gpu-fwd-prefilter --gpu-fb-parser`)

Wall time: **2.07s** total, **0.159s** avg/query (1.42x faster than Mode 1)

| Component | Time (s) | % of wall | Notes |
|-----------|----------|-----------|-------|
| GPU kernels (all stages) | 1.31 | 63.3% | MSV+Vit+Fwd+Bck+bias |
| Vit/Fwd dispatch + sync | 0.96 | 46.4% | Includes GPU launch + CPU fallback |
| CPU Viterbi (residual) | 0.48 | 23.4% | Large-M models fall back to CPU |
| I/O read + unpack | 0.23 | 11.2% | |
| CPU Forward (residual) | 0.21 | 10.0% | |
| Survivor loop overhead | 0.20 | 9.6% | |
| CPU domain + null2 | 0.16 | 7.9% | |
| CPU Backward (residual) | 0.10 | 4.6% | |

**GPU utilization**: 63% of wall time is GPU kernel execution.

## Optimization Opportunities (ranked by impact)

### 1. CPU Viterbi — 33.6% (Mode 1), 23.4% residual (Mode 2)

The single largest bottleneck. The GPU Viterbi prefilter helps but:
- M≤512 limit excludes large models (use `--gpu-vit-largem` to override, but current kernel is slower than CPU SSE for large M)
- Per-survivor dispatch overhead is high
- **Opportunity**: Optimize GPU Viterbi kernel for large M (tiling, shared memory), or batch more aggressively to amortize launch overhead

### 2. CPU Forward + Backward — 26.5% combined (Mode 1)

Second largest cost. GPU prefilters reduce but don't eliminate.
- **Opportunity**: Full GPU domain definition would eliminate most of this. The FB parser already exists but domain definition (`p7_domaindef_ByPosteriorHeuristics`) remains CPU-only.

### 3. Survivor loop overhead — 7.3% (Mode 1), 9.6% (Mode 2)

Per-survivor costs: `gpu_MaterializeSeq` (copies sequence from gpudb mmap to ESL_SQ), `p7_bg_SetLength`, `p7_oprofile_ReconfigLength`, `p7_omx_GrowTo`.
- **Opportunity**: Pre-allocate survivor buffers to max expected length, batch ReconfigLength calls, avoid per-sequence malloc/free in the hot loop.

### 4. I/O read + unpack — 6.5% (Mode 1), 11.2% (Mode 2)

Fixed per-query cost (~15-21ms/query). With resident DB, this is reading chunk metadata from dsqdata, not sequence data.
- **Opportunity**: Overlap I/O with GPU kernel execution via double-buffering. Or eliminate dsqdata read entirely when using gpudb resident path (metadata is already in the gpudb index).

### 5. Vit/Fwd dispatch sync — 46.4% (Mode 2 only)

In full-GPU mode, the dispatch block (GPU Viterbi launch → GPU Forward launch → FB parser batch) is blocking.
- **Opportunity**: Use async CUDA streams to overlap Viterbi D2H with Forward H2D. Overlap dispatch with CPU work on other survivors. Increase batch sizes to reduce number of launches.

### 6. GPU kernel optimization — 19.7% (Mode 1), 63.3% (Mode 2)

In Mode 2, GPU kernel becomes the dominant cost (good sign — CPU bottlenecks are addressed). The SSV kernel (`--gpu-ssv`) already provides 1.36x over monolithic MSV.
- **Opportunity**: Make SSV the default MSV path. Occupancy tuning for Viterbi/Forward kernels. Kernel fusion (MSV → bias in single launch).

## Comparison: GPU vs CPU-4

With engine reuse, resident DB, and multi-query single-process benchmarking (amortizing CUDA init once):

| Config | Wall time | Speedup |
|--------|-----------|---------|
| CPU 1-thread | 8.93s | 1.00x |
| CPU 4-thread | 4.79s | 1.86x |
| GPU (all stages) | 2.50s | **4.19x** vs CPU-1, **1.92x** vs CPU-4 |

**Critical benchmark note**: The profmark runner was previously invoking hmmsearch once per query (13 separate processes), paying ~0.35s CUDA init each time = 4.55s pure overhead. This made GPU appear 0.6x vs CPU-4. The fix: `test-speed/x-hmmsearch-gpu-profmark` now concatenates all HMMs and runs a single hmmsearch process for CPU and GPU respectively, matching real multi-query usage.

