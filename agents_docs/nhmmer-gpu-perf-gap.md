# nhmmer GPU vs CPU Performance Gap

Last updated: 2026-05-07

## Current Benchmark (chr22, single query)

| Path | MADE1 (M=80) | query_short (M=151) | query_medium (M=501) |
|------|:---:|:---:|:---:|
| CPU-1 | 0.97s | 1.29s | 6.32s |
| CPU-4 | 0.36s | 0.46s | 1.87s |
| GPU-4 FASTA | 2.87s | 6.47s | 46.5s |
| GPU-4 nucdb | 2.31s | — | — |

**GPU is 6-25x slower than CPU-4** on single-query benchmarks.

## Root Causes

### 1. CPU downstream dominates (90%+ of GPU wall time for M≥150)

After GPU SSV + scanning Viterbi identify sub-windows, CPU threads run ForwardParser + Backward + domain definition on each survivor. Cost per window: O(window_length × M).

- query_short (M=151): 372 surviving windows → ~5s CPU downstream
- query_medium (M=501): 693 surviving windows → ~45s CPU downstream

The extra GPU hits (372 vs 363, 693 vs 648) amplify this — GPU is doing MORE CPU work than the CPU-only path.

### 2. Fixed overhead (~0.5s)

- CUDA context initialization: ~0.5s first query, ~0s subsequent (engine reuse)
- Amortized over multi-query workloads

### 3. GPU generates more survivors than CPU (parity issue)

GPU scanning Viterbi uses fixed `xw_*` profile parameters (not reconfigured per window length), making it more permissive than CPU's per-window `p7_oprofile_ReconfigLength`. This causes:
- More false-positive sub-windows survive to CPU downstream
- Each false positive costs O(window_length × M) CPU time
- For M=501: 45 extra hits × ~65ms each ≈ 3s wasted

### 4. Kernel cost scales with M

SSV kernel: O(L × M/8) per strand (inner Q-loop).
Scanning Viterbi: O(L × M/8) per window, shared memory = O(M) per warp → reduced occupancy.

## What Would Fix It

| Fix | Savings | Effort | Notes |
|-----|---------|--------|-------|
| GPU ForwardParser | ~40s for M=501 | Very high | Only fix for large models |
| Fix per-window xw_* reconfigure | ~3s for M=501 | Medium | Eliminate false positives |
| Avoid CPU downstream for false positives | ~3s for M=501 | Low | Tighter GPU thresholds |
| Multi-query amortization | ~0.5s | Done | Engine reuse implemented |
| Nucdb format | ~0.4s | Done | Eliminates FASTA parsing |

## When GPU Nhmmer Makes Sense

Currently the GPU path is NOT faster than CPU-4 for any single-query workload. It's a development scaffold toward GPU ForwardParser.

Potential future wins:
- **Multi-query + nucdb**: Engine reuse + no FASTA parsing per query
- **GPU ForwardParser**: Would eliminate the 90%+ bottleneck
- **Very large databases**: PCIe bandwidth amortized over more residues (but CPU scales linearly too)

## Benchmark Script

Run `test-speed/x-nhmmer-gpu-bench` for quick comparison.
