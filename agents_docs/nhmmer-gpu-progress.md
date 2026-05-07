# nhmmer GPU Support — Progress

Last updated: 2026-05-08

## Architecture

GPU nhmmer uses a **GPU SSV + GPU filters + GPU scanning Viterbi + GPU Forward prefilter + GPU FB parser + threaded CPU downstream** pipeline. All GPU stages are default-on with `--gpu`.

```
Input FASTA/nucdb
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ GPU SSV Longtarget (warp-per-chunk, 64K chunks)     │
│   → overlapping chunks, register-based SSV DP       │
│   → outputs windows with global coordinates         │
└──────────────────────┬──────────────────────────────┘
                       ▼
        p7_pli_ExtendAndMergeWindows (CPU)
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ GPU Batch Filter (MSV + null + bias + F1 gating)    │
│   → synthetic ESL_DSQDATA_CHUNK (zero-copy)         │
│   → removes non-survivors                           │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ GPU Viterbi Pre-filter (single-score, F2 gating)    │
│   → removes windows that can't produce sub-windows  │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ GPU Scanning Viterbi (warp-per-window, 8 warps/blk) │
│   → full DP per window, emits sub-windows           │
│   → GPU threshold: bias kernel + analytic null      │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ GPU Forward Pre-filter (F3*2.0 relaxed gate)        │
│   → GPU Forward score-only on sub-windows           │
│   → removes ~50-60% of windows before FB + domain   │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ GPU FB Parser (batch ForwardBackward xmx)           │
│   → batch Forward+Backward parser on all survivors  │
│   → returns xmx arrays for domaindef, fwd/bck scores│
│   → auto fallback to CPU if GPU fails               │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ Threaded CPU Downstream (N threads)                 │
│   → F3 gate with GPU fwdsc, bind GPU xmx, domaindef│
│   → rescore_isolated_domain (full Fwd+Bck per domain)│
│   → merge: p7_tophits_Merge + p7_pipeline_Merge    │
└─────────────────────────────────────────────────────┘
```

Both strands processed sequentially (forward, then reverse complement).

## Files

| File | Purpose |
|------|---------|
| `src/nhmmer.c` | CLI (`--gpu` option), engine lifecycle, nucdb detection |
| `src/nhmmer_internal.h` | `NHMMER_GPU_INFO`, `NHMMER_GPU_WINDOW_BATCH` structs |
| `src/nhmmer_gpu.c` | GPU orchestration: batch filter, Viterbi prefilter, scanning Vit, threads, nucdb loop |
| `src/cuda/p7_cuda_ssv_longtarget.cu` | SSV longtarget kernel + `SSVLongtargetResident` |
| `src/cuda/p7_cuda_viterbi_longtarget.cu` | Scanning Viterbi kernel + threshold kernel |
| `src/cuda/p7_cuda_runtime.cu` | Engine create/destroy/reset, nucdb upload/release |
| `src/cuda/p7_cuda_internal.h` | Engine struct (all device buffers) |
| `src/cuda/p7_cuda.h` | Public API declarations |
| `src/p7_nucdb.c` | Nucdb format: Write/Open/Close (mmap-based) |
| `src/p7_nucdb.h` | Nucdb structs: `P7_NUCDB_HEADER`, `P7_NUCDB_CHUNK_IDX`, `P7_NUCDB_SEQ_IDX` |
| `src/hmmnucdb.c` | CLI tool: `hmmnucdb [opts] <seqfile> <nucdb>` |

## CLI

```sh
# Basic GPU nhmmer (all GPU stages default-on)
src/nhmmer --gpu --cpu 4 --noali query.hmm target.fa

# With nucdb (pre-built binary format, eliminates FASTA parsing)
src/hmmnucdb target.fa target.nucdb        # build once
src/nhmmer --gpu --cpu 4 --noali query.hmm target.nucdb.nucdb

# Nucdb with overlap (enables GPU-resident path when overlap >= model max_length)
src/hmmnucdb --overlap 2001 target.fa target-overlap.nucdb
src/nhmmer --gpu --cpu 4 --noali query.hmm target-overlap.nucdb.nucdb

# Hidden flags (group 99, not shown in help)
--gpu-batch             # SSV/bias batch on GPU (default-on with --gpu)
--gpu-vit-prefilter     # Viterbi pre-filter (default-on with --gpu)
--gpu-vit-longtarget    # Scanning Viterbi (default-on with --gpu)
--gpu-fwd-prefilter     # Forward pre-filter (wired, not implemented)
--gpu-chunk-size N      # Chunk size (default 65536)
--gpu-device N          # CUDA device selection
```

## Benchmark Results (2026-05-08)

**Target**: chr22.fa (50MB, ~101.6M residues both strands)
**System**: RTX 4090, 4 CPU threads

| Path | MADE1 (M=80) | query_short (M=151) | query_medium (M=501) |
|------|:---:|:---:|:---:|
| CPU-4 | 0.30s / 154 | 0.45s / 120 | 1.80s / 215 |
| GPU-4 FASTA | 1.49s / 153 | 2.22s / 120 | 7.96s / 226 |

### GPU Timing Breakdown (pipeline time, both strands)

| Stage | MADE1 | query_short | query_medium |
|-------|:---:|:---:|:---:|
| SSV longtarget | 0.09s (9%) | 0.09s (4%) | 0.25s (4%) |
| batch filter | 0.02s (2%) | 0.02s (1%) | 0.10s (2%) |
| scanning Viterbi | 0.004s (<1%) | 0.004s (<1%) | 0.01s (<1%) |
| Forward prefilter | 0.04s (5%) | 0.04s (2%) | 0.17s (3%) |
| GPU FB parser | 0.17s (17%) | 0.03s (1%) | 0.15s (2%) |
| CPU workers | 0.67s (68%) | 1.95s (91%) | 5.57s (89%) |
| **Pipeline total** | **0.99s** | **2.14s** | **6.25s** |

### Parity Notes

- **MADE1 (M=80)**: 153 vs 154 (1-hit FP difference from GPU Forward score)
- **query_short (M=151)**: Perfect parity (120=120)
- **query_medium (M=501)**: GPU reports 226 vs CPU 215 (11 extra hits from fixed `xw_*` profile parameters in scanning Viterbi)

## Performance Analysis

### Why GPU is slower than CPU-4

1. **Fixed overhead (~0.5s)**: CUDA context initialization (amortized for multi-query)
2. **CPU domaindef dominates**: `rescore_isolated_domain` runs full `p7_Forward` + `p7_Backward` + `p7_OptimalAccuracy` + `p7_OATrace` per domain. This is O(domain_length × M) per domain and accounts for 67-91% of pipeline time.
3. **GPU FB parser helps but doesn't eliminate the bottleneck**: Replaces parser-level Forward+Backward (~19% improvement for M=501), but `rescore_isolated_domain` still runs on CPU.
4. **Extra GPU hits amplify CPU work**: GPU passes more windows → more domaindef calls

### GPU FB Parser Impact

| Query | Without GPU FB | With GPU FB | Improvement |
|-------|:---:|:---:|:---:|
| MADE1 | CPU workers 0.76s | CPU workers 0.67s | 12% |
| query_short | CPU workers 1.69s | CPU workers 1.56s | 8% |
| query_medium | CPU workers 7.91s | CPU workers 5.57s | 30% |

The GPU FB parser replaces the initial `p7_ForwardParser` + `p7_BackwardParser` calls (parser-level DP, which computes only xmx special states). The remaining CPU time is dominated by `rescore_isolated_domain` (full DP per domain).

### Where GPU wins

- Multi-query workloads (amortize engine init)
- Nucdb format (saves ~0.4s FASTA parsing overhead)

### Key bottleneck: `rescore_isolated_domain`

| Model size | Dominant cost | Potential fix |
|-----------|--------------|---------------|
| M≤100 | Fixed overhead (CUDA init, I/O) | Multi-query, nucdb |
| M=100-500 | rescore_isolated_domain (full Fwd+Bck per domain) | GPU full Forward/Backward (not parser) |
| M>500 | rescore_isolated_domain + kernel arithmetic | GPU full Forward/Backward + kernel tuning |

## Nucdb Format

Pre-chunked, mmap'd nucleotide database. Eliminates FASTA parsing and enables GPU-resident data path.

```
File layout (page-aligned):
  Metadata:  sequence names (null-terminated strings)
  Seq index: P7_NUCDB_SEQ_IDX[nseq] (name_offset, length, chunk ranges)
  Chunk idx: P7_NUCDB_CHUNK_IDX[nchunks] (data_offset, length, seq_id, seq_offset, strand)
  Data:      [sentinel][residues] per chunk, contiguous, both strands
```

Build-time options: `--chunk-size` (default 65536), `--overlap` (default 0), `--fwd-only`.

**Resident path**: When `--overlap >= model_max_length` AND `chunk_size` matches runtime, the kernel reads directly from GPU-uploaded nucdb data (no H2D per sequence). Pre-stored RC chunks eliminate `esl_sq_ReverseComplement`.

## Status Summary

All phases complete including GPU FB parser. Branch `worktree-h3-gpu-nhmmer` contains commits on top of `h3-gpu`:

```
(pending) gpu: GPU ForwardBackward parser for nhmmer domaindef
2432fbc5 gpu: nhmmer performance optimizations (per-window thresholds, Forward pre-filter, kernel tuning)
f0de5475 docs: comprehensive nhmmer GPU documentation and benchmark scripts
98e997c1 gpu: pre-stored RC from nucdb + GPU-resident SSV longtarget path
2e51ca54 cleanup: remove --gpu-compare debug flag and update nhmmer GPU docs
221a1fa3 gpu: nucleotide GPU database format (nucdb)
9f2eee46 gpu: port h3-gpu optimizations to nhmmer
6fffc72d gpu: GPU scanning Viterbi for nhmmer with GPU threshold computation
23a203ef gpu: batch MSV/bias filter and Viterbi pre-filter for nhmmer
4c38dd80 gpu: threaded downstream pipeline for nhmmer with CUDA engine reuse
(+ base commits from h3-gpu: protein GPU acceleration)
```

## Future Work

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| High | GPU full Forward/Backward (not parser) for rescore_isolated_domain | Very high | Only path to beating CPU-4 for M>100 |
| Medium | Fix parity for query_medium (per-window xw_* reconfigure) | Medium | Eliminates false positives (226→215) |
| Low | GPU OptimalAccuracy + OATrace | Very high | Would eliminate remaining domaindef CPU work |
| Low | Async strand overlap | Low | ~0.1-0.3s savings |
