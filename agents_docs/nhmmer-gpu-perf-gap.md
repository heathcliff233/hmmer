# nhmmer GPU vs CPU-4 Performance Gap

Last updated: 2026-05-07

## Current State

GPU-4+batch+vit is 1.2x–2.4x slower than CPU-4 on single-query chr22 benchmarks:

| Query | CPU-4 | GPU-4+batch+vit | Ratio |
|-------|-------|-----------------|-------|
| MADE1 (M=80) | 0.32s | 0.76s | 2.4x slower |
| query_short (M=151) | 0.44s | 0.82s | 1.9x slower |
| query_medium (M=501) | 1.79s | 2.15s | 1.2x slower |

Multi-query (3 combined): CPU-4 2.97s vs GPU-4+batch+vit 3.47s (1.17x slower).

## Fixed Overhead Breakdown

These costs are independent of model size and dominate for short/medium models:

| Source | Cost | Amortizable? |
|--------|------|-------------|
| CUDA context init | ~0.5s | Yes (engine reuse across queries) |
| FASTA sequential I/O | ~0.4s sys | No (inherent to esl_sqio_ReadWindow) |
| Reverse complement | ~0.05s | No (50MB alloc + memcpy + complement) |

Total fixed overhead: ~0.9s per query (cold) or ~0.4s (warm, engine reused).

## Variable Cost: CPU Scanning Viterbi

After GPU SSV + batch filter + Viterbi pre-filter, the remaining work is CPU scanning Viterbi + Forward + domain on surviving windows. For query_medium:

- ~8000 merged windows per strand
- Viterbi pre-filter removes some, but most pass (model is large enough that windows are genuine)
- CPU scanning Viterbi is the dominant remaining cost (~1-2s for query_medium)

## Paths to Parity with CPU-4

### 1. Eliminate FASTA I/O overhead (saves ~0.4s)

Replace `esl_sqio_ReadWindow` with memory-mapped I/O or a pre-built binary format (like dsqdata for proteins). The CPU path avoids this cost by distributing sequence blocks across threads via a work queue; the GPU path reads the full sequence sequentially.

**Effort**: Medium. Requires new sequence format or mmap support in Easel.
**Impact**: 0.4s savings → query_medium 1.75s (matches CPU-4).

### 2. GPU scanning Viterbi kernel (saves ~1-2s for large models)

The existing GPU Viterbi computes a single max score per sequence. nhmmer needs `p7_ViterbiFilter_longtarget` which scans within a window and emits multiple sub-windows where the score exceeds a threshold. A GPU implementation would need to:

- Process each window as a warp (similar to SSV longtarget)
- Track running Viterbi score and emit sub-window coordinates when score > threshold
- Handle the J-state and multi-hit model

**Effort**: High. Fundamentally different kernel architecture from single-score Viterbi.
**Impact**: Would eliminate the dominant CPU cost for large models. Could make GPU 2-3x faster than CPU-4 for M>500.

### 3. Multi-query amortization (already implemented)

Engine reuse saves ~0.25s per additional query. For workloads with 10+ queries, CUDA init becomes negligible. Current multi-query gap is only 1.17x.

**Effort**: Done.
**Impact**: Already competitive for multi-query workloads.

### 4. Parallel strand processing

Currently forward and reverse strands are processed sequentially. Overlapping GPU SSV scan of one strand with CPU downstream of the other could save ~0.1-0.3s.

**Effort**: Low-medium. Requires async CUDA stream management.
**Impact**: Modest (0.1-0.3s).

## Priority Recommendation

1. **FASTA I/O** (medium effort, gets to parity for large models)
2. **GPU scanning Viterbi** (high effort, but would make GPU definitively faster)
3. **Parallel strands** (low effort, modest gain)

For the current architecture (GPU SSV + CPU downstream), the ceiling is approximately CPU-4 speed minus CUDA init. Beating CPU-4 requires either eliminating I/O overhead or moving scanning Viterbi to GPU.
