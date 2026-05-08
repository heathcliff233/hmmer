# nhmmer GPU vs CPU Performance Gap

Last updated: 2026-05-08

## Current Benchmark (chr22, single query)

| Path | MADE1 (M=80) | query_short (M=151) | query_medium (M=501) |
|------|:---:|:---:|:---:|
| CPU-4 | 0.33s / 154 | 0.45s / 120 | 1.64s / 215 |
| GPU-4 FASTA | 1.74s / 156 | 2.07s / 124 | 10.3s / 264 |

**GPU is ~5-6x slower than CPU-4** on single-query benchmarks.

## Root Causes

### 1. Single-thread-per-block domain kernels (primary bottleneck)

The GPU domain rescoring kernels use 1 CUDA thread per domain (one block per domain). For MADE1 with M=80, Q=5, each domain iterates N=20 cells × L positions serially. With ~3000 domains and 6 sequential kernel launches, the RTX 4090's 128 SMs run ~23 waves of blocks per kernel. The total kernel time is ~27ms per batch — fast in absolute terms, but the **overhead dominates**: H2D transfers, kernel launches, D2H trace transfers.

Domain kernel time breakdown for MADE1 (per strand):
- Primary batch (3000 domains): ~27ms kernels
- Trim batch (1100 domains): ~12ms kernels
- H2D/D2H + overhead: remaining time

### 2. CUDA fixed overhead (~0.5s)

- CUDA context initialization: ~0.5s first query, ~0s subsequent (engine reuse)
- Amortized over multi-query workloads

### 3. Envelope-finding still on CPU

`p7_domaindef_ByPosteriorHeuristics(envelopes_only=TRUE)` runs on CPU to identify domain boundaries. This involves the posterior decoding heuristics (DomainDecoding + region/envelope identification). Not a major bottleneck (~0.1-0.3s) but serializes GPU work.

### 4. GPU generates more survivors than CPU (parity issue)

GPU scanning Viterbi uses fixed `xw_*` profile parameters (not reconfigured per window length), making it more permissive than CPU's per-window `p7_oprofile_ReconfigLength`. For query_medium: 264 hits vs 215.

## Historical Improvement

| Version | MADE1 time | Bottleneck |
|---------|:---:|:---:|
| Pre-batching (per-domain GPU calls) | 34s | 5000+ individual GPU calls × 5ms overhead each |
| Cross-window batching + trim batching | 1.74s | Single-thread kernels + fixed overhead |

**18x improvement** from batching alone, without changing kernel design.

## What Would Further Improve It

| Fix | Estimated savings | Effort | Notes |
|-----|---------|--------|-------|
| Warp-cooperative domain kernels | ~2-4x faster kernels | High | Use Q threads per block with warp shuffle; inner loop parallelism |
| Pinned memory + async D2H | ~10-20ms per batch | Low | Overlap D2H with CPU work |
| Eliminate trace D2H (generate alidisplay on GPU) | ~15ms per batch (large trace arrays) | Very high | Complex; alidisplay needs host strings |
| Fix per-window xw_* reconfigure | ~49 fewer false-positive domains | Medium | Reduces wasted work |
| Multi-query amortization | ~0.5s | Done | Engine reuse implemented |
| Nucdb format | ~0.4s | Done | Eliminates FASTA parsing |

## When GPU Nhmmer Makes Sense

Currently the GPU path is NOT faster than CPU-4 for any single-query workload. The single-thread kernel design means short domains can't exploit the GPU's massive parallelism. However:

- The domain rescoring bottleneck is **eliminated** (was 34s, now 0.08s kernel time)
- Remaining overhead is structural (CUDA init, H2D/D2H, envelope-finding)
- Multi-query + nucdb can amortize fixed costs
- Warp-cooperative kernels would be the path to beating CPU-4

## Benchmark Script

Run `test-speed/x-nhmmer-gpu-bench` for quick comparison.
