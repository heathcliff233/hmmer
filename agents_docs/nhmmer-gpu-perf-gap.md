# nhmmer GPU vs CPU Performance Gap

Last updated: 2026-05-08

## Current Benchmark (chr22, single query)

| Path | MADE1 (M=80) | query_short (M=151) | query_medium (M=501) |
|------|:---:|:---:|:---:|
| CPU-4 | 0.30s / 154 | 0.45s / 120 | 1.80s / 215 |
| GPU-4 FASTA | 1.49s / 153 | 2.22s / 120 | 7.96s / 226 |

**GPU is 4-7x slower than CPU-4** on single-query benchmarks.

## Root Causes

### 1. `rescore_isolated_domain` dominates (67-91% of GPU pipeline time)

After GPU SSV + scanning Viterbi + Forward prefilter + GPU FB parser identify windows and compute parser-level Forward/Backward matrices, `p7_domaindef_ByPosteriorHeuristics` calls `rescore_isolated_domain` for each domain. This runs full `p7_Forward` + `p7_Backward` + `p7_OptimalAccuracy` + `p7_OATrace` on each domain subsequence — O(domain_length × M) per domain.

- MADE1 (M=80): CPU workers 0.67s / 0.99s pipeline (68%)
- query_short (M=151): CPU workers 1.95s / 2.14s pipeline (91%)
- query_medium (M=501): CPU workers 5.57s / 6.25s pipeline (89%)

### 2. GPU FB parser helps but doesn't eliminate the bottleneck

The GPU FB parser replaces the initial `p7_ForwardParser` + `p7_BackwardParser` (parser-level DP). Improvement:

| Query | CPU workers (no GPU FB) | CPU workers (with GPU FB) | Savings |
|-------|:---:|:---:|:---:|
| MADE1 | 0.76s | 0.67s | 12% |
| query_short | 1.69s | 1.56s | 8% |
| query_medium | 7.91s | 5.57s | 30% |

The remaining time is `rescore_isolated_domain` (full DP, not parser), which the GPU FB parser cannot replace.

### 3. Fixed overhead (~0.5s)

- CUDA context initialization: ~0.5s first query, ~0s subsequent (engine reuse)
- Amortized over multi-query workloads

### 4. GPU generates more survivors than CPU (parity issue)

GPU scanning Viterbi uses fixed `xw_*` profile parameters (not reconfigured per window length), making it more permissive than CPU's per-window `p7_oprofile_ReconfigLength`. For query_medium: 226 hits vs 215.

## What Would Fix It

| Fix | Savings | Effort | Notes |
|-----|---------|--------|-------|
| GPU full Forward/Backward for rescore_isolated_domain | ~5s for M=501 | Very high | Only fix for large models; needs per-domain DP |
| GPU OptimalAccuracy + OATrace | ~1-2s for M=501 | Very high | Completes full domaindef on GPU |
| Fix per-window xw_* reconfigure | ~11 fewer false-positive windows | Medium | Reduces wasted domaindef calls |
| Multi-query amortization | ~0.5s | Done | Engine reuse implemented |
| Nucdb format | ~0.4s | Done | Eliminates FASTA parsing |
| GPU Forward prefilter | Done | Done | Removes ~50-60% of windows before FB |
| GPU FB parser | Done | Done | Replaces parser-level Fwd+Bck (8-30% CPU worker savings) |

## When GPU Nhmmer Makes Sense

Currently the GPU path is NOT faster than CPU-4 for any single-query workload. The bottleneck is `rescore_isolated_domain` inside domaindef, which runs full Forward+Backward+OptimalAccuracy per domain on CPU.

Potential future wins:
- **Multi-query + nucdb**: Engine reuse + no FASTA parsing per query
- **GPU full Forward/Backward**: Would replace `rescore_isolated_domain` (the 67-91% bottleneck)
- **Very large databases**: PCIe bandwidth amortized over more residues

## Benchmark Script

Run `test-speed/x-nhmmer-gpu-bench` for quick comparison.
