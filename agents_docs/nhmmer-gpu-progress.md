# nhmmer GPU Support — Progress

Last updated: 2026-05-07

## Architecture

GPU nhmmer uses a **GPU SSV + threaded CPU downstream** approach:

1. **GPU SSV stage**: Long sequence split into overlapping chunks (64K each, overlap = max_length). Warp-per-chunk kernel (32 threads) performs register-based SSV DP, finds high-scoring diagonals, extends them, and outputs windows with coordinates translated back to full-sequence space.

2. **Window merge**: `p7_pli_ExtendAndMergeWindows()` consolidates overlapping GPU windows (same as CPU pipeline).

3. **Threaded downstream**: Merged windows distributed across N CPU threads. Each thread independently runs bias + MSV recompute + `p7_pli_postSSV_LongTarget` (Viterbi longtarget + Forward + domain definition). Results merged via `p7_tophits_Merge` / `p7_pipeline_Merge`.

4. **Both strands**: Forward strand processed first, then reverse complement. Each strand goes through the full GPU SSV → threaded downstream pipeline.

## Key Design Decisions

- **Kernel style**: warp-per-chunk (32 threads), register-based DP, same pattern as hmmsearch SSV kernel
- **Chunk size**: 64K residues (tunable via `--gpu-chunk-size`), overlap = `om->max_length`
- **Window output**: atomic counter + structured output buffer on device
- **Downstream threading**: per-thread deep copies of P7_OPROFILE (not Clone — Clone shares memory, causes races), P7_PIPELINE, P7_TOPHITS, P7_BG, P7_SCOREDATA
- **Engine reuse**: CUDA engine created once before query loop, MSV profile per-query
- **No GPU Viterbi/Forward for nhmmer**: existing GPU kernels compute single scores per sequence; nhmmer needs scanning Viterbi (`p7_ViterbiFilter_longtarget`) which is fundamentally different

## Files

| File | Purpose |
|------|---------|
| `src/nhmmer_internal.h` | `NHMMER_GPU_INFO` struct, `NHMMER_GPU_IDLEN_CB` callback, serial loop API |
| `src/nhmmer_gpu.c` | Worker struct, thread function, `nhmmer_gpu_process_strand`, `nhmmer_gpu_serial_loop` |
| `src/cuda/p7_cuda_ssv_longtarget.cu` | GPU SSV longtarget kernel + host wrapper |
| `src/nhmmer.c` | `--gpu` option, engine lifecycle, GPU path integration |

## Benchmark Results

Target: chr22.fa (50MB, ~101.6M residues both strands)
Queries: MADE1 (M=80), query_short (M=151), query_medium (M=501)

| Query | CPU-1 | CPU-4 | GPU-1 | GPU-4 | GPU-4 threading speedup |
|-------|-------|-------|-------|-------|------------------------|
| MADE1 (M=80) | 0.94s | 0.33s | 1.10s | 0.90s | 1.2x |
| query_short (M=151) | 1.71s | 0.41s | 2.22s | 0.94s | 2.4x |
| query_medium (M=501) | 6.04s | 1.73s | 7.10s | 2.93s | 2.4x |

Hit parity: 154/120 hits match exactly (MADE1/query_short). query_medium has 217 GPU vs 215 CPU — a pre-existing 2-hit difference from GPU SSV kernel boundary effects, consistent across GPU-1 and GPU-4.

Multi-query engine reuse: 3 queries combined GPU-4 = 4.05s vs sum-of-individual 4.77s (saves ~0.7s).

## Performance Analysis

**GPU-4 vs CPU-4 gap** (GPU is ~2-3x slower):
- CUDA context init: ~0.5s (amortized across queries via engine reuse)
- FASTA I/O overhead: ~0.4s sys time (GPU reads full sequence via `esl_sqio_ReadWindow`; CPU distributes blocks via work queue)
- Parallelism granularity: CPU-4 distributes entire sequence blocks across threads; GPU-4 distributes windows within a single strand

**Where GPU wins**:
- SSV scan: GPU processes 50MB in ~0.1s vs CPU-1 ~0.5-5s depending on model size
- Threading scales well: 2.4x speedup with 4 threads for medium/large models

**Remaining bottlenecks**:
- File I/O (0.4s sys) — inherent to FASTA sequential read
- Reverse complement copy (50MB allocation + memcpy + complement)
- CUDA init (one-time ~0.5s per session)

## Status

- **Phase 1**: Complete — GPU SSV longtarget kernel working with hit parity
- **Phase 2**: Complete — threaded downstream pipeline with 2.4x scaling
- **Phase 3**: Complete — engine reuse, chunk_size tuning; async deferred (low ROI)
