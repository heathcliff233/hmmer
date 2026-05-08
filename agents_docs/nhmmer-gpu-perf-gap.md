# nhmmer GPU vs CPU Performance Gap

Last updated: 2026-05-09

## Current Benchmark (chr22, single query)

| Path | MADE1 (M=80) | query_short (M=151) | query_medium (M=501) |
|------|:---:|:---:|:---:|
| CPU-4 | 0.33s / 154 | 0.45s / 120 | 1.64s / 215 |
| GPU-4 FASTA | 1.35s / 154 | 1.92s / 122 | 5.34s / 261 |

**GPU is ~3-4x slower than CPU-4** on single-query benchmarks (improved from 5-6x).

## Root Causes

### 1. Serial GPU pipeline with CPU worker bottleneck

The GPU pipeline stages run serially, and CPU workers (domain definition + hit reporting) still consume 75–94% of total wall time. The GPU saves ~50% of Forward computation via the prefilter→Backward-only split, but CPU `p7_domaindef_ByPosteriorHeuristics` dominates.

### 2. CUDA fixed overhead (~0.5s)

- CUDA context initialization: ~0.5s first query, ~0s subsequent (engine reuse)
- Amortized over multi-query workloads

### 3. GPU generates more survivors than CPU (parity issue)

GPU scanning Viterbi uses fixed `xw_*` profile parameters (not reconfigured per window length), making it more permissive than CPU's per-window `p7_oprofile_ReconfigLength`. For query_medium: 261 hits vs 215.

## Historical Improvement

| Version | MADE1 time | Key change |
|---------|:---:|:---:|
| Pre-batching (per-domain GPU calls) | 34s | 5000+ individual GPU calls × 5ms overhead each |
| Cross-window batching + trim batching | 1.74s | Single GPU call for all domains |
| Parallel-thread domain kernels | 1.37s | T threads/block with prefix scan (4 of 6 kernels) |
| Forward-Backward split (prefilter saves xf) | 1.35s | Eliminates redundant Forward computation |

**25x improvement** from serial per-domain calls to current design.

## What Would Further Improve It

| Fix | Estimated savings | Effort | Notes |
|-----|---------|--------|-------|
| Move envelope-finding to GPU | Eliminate 75-94% bottleneck | Very high | Would need GPU posterior decoding heuristics |
| Pinned memory + async D2H | ~10-20ms per batch | Low | Overlap D2H with CPU work |
| Fix per-window xw_* reconfigure | ~46 fewer false-positive domains | Medium | Reduces wasted work |
| OptAcc kernel parallelization | Small | Medium | Needs max-prefix scan |
| Multi-query amortization | ~0.5s | Done | Engine reuse implemented |
| Nucdb format | ~0.4s | Done | Eliminates FASTA parsing |
| Redundant Forward elimination | ~0.5-5s | Done | Prefilter saves xf for Backward-only |
| Parallel-thread domain kernels | ~0.04-0.4s | Done | 4/6 kernels use T threads with prefix scan |

## When GPU Nhmmer Makes Sense

Currently the GPU path is NOT faster than CPU-4 for any single-query workload. The dominant bottleneck is now CPU workers (envelope-finding + hit reporting), not GPU kernels. However:

- All GPU kernel stages combined are only 0.2–0.6s
- Multi-query + nucdb can amortize fixed costs
- Moving envelope-finding to GPU would be the path to beating CPU-4

## Benchmark Script

Run `test-speed/x-nhmmer-gpu-bench` for quick comparison.
