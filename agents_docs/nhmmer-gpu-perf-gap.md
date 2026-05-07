# nhmmer GPU vs CPU-4 Performance Gap

Last updated: 2026-05-07

## Current State

GPU+batch+vit+vlt (all GPU stages) vs CPU-4 on single-query chr22 benchmarks:

| Query | CPU-4 | GPU+batch+vit | GPU+batch+vit+vlt | Ratio (vlt) |
|-------|-------|---------------|-------------------|-------------|
| MADE1 (M=80) | 0.35s | 0.93s | 0.95s | 2.7x slower |
| query_short (M=151) | 0.41s | 0.97s | 1.09s | 2.7x slower |
| query_medium (M=501) | 1.80s | 2.77s | 3.71s | 2.1x slower |

## Fixed Overhead Breakdown

These costs are independent of model size and dominate for short/medium models:

| Source | Cost | Amortizable? |
|--------|------|-------------|
| CUDA context init | ~0.5s | Yes (engine reuse across queries) |
| FASTA sequential I/O | ~0.4s sys | No (inherent to esl_sqio_ReadWindow) |
| Reverse complement | ~0.05s | No (50MB alloc + memcpy + complement) |

Total fixed overhead: ~0.9s per query (cold) or ~0.4s (warm, engine reused).

## GPU Scanning Viterbi Bottleneck

With `--gpu-vit-longtarget`, scanning Viterbi runs on GPU but the remaining CPU work (ForwardParser + domain definition on surviving windows) is now the dominant cost. For query_medium with 648 hits, CPU post-Viterbi processing takes ~1-2s.

The GPU scanning Viterbi eliminated the CPU threshold computation bottleneck (previously serial bias filter per window) by computing thresholds on GPU: bias filter via existing batch kernel, null scores computed analytically in a threshold kernel. Kernel launch batches 8 warps/block for better occupancy.

## Paths to Parity with CPU-4

### 1. GPU ForwardParser / domain definition (saves ~1-2s for large models)

The dominant remaining CPU cost. Would require GPU implementations of:
- `p7_ForwardParser` (Forward algorithm with sparse DP)
- `p7_domaindef` (posterior decoding, domain envelope identification)

**Effort**: Very high. Complex algorithms with many data dependencies.
**Impact**: Would make GPU competitive for large models.

### 2. Eliminate FASTA I/O overhead (saves ~0.4s)

Replace `esl_sqio_ReadWindow` with memory-mapped I/O or a pre-built binary format.

**Effort**: Medium. Requires new sequence format or mmap support in Easel.
**Impact**: 0.4s savings per query.

### 3. Multi-query amortization (already implemented)

Engine reuse saves ~0.25s per additional query. For workloads with 10+ queries, CUDA init becomes negligible.

**Effort**: Done.
**Impact**: Already competitive for multi-query workloads.

### 4. Parallel strand processing

Overlapping GPU SSV scan of one strand with CPU downstream of the other.

**Effort**: Low-medium. Requires async CUDA stream management.
**Impact**: Modest (0.1-0.3s).

## Priority Recommendation

1. **GPU Forward/domain** (very high effort, but the only path to beating CPU-4 for large models)
2. **FASTA I/O** (medium effort, saves fixed ~0.4s per query)
3. **Parallel strands** (low effort, modest gain)
