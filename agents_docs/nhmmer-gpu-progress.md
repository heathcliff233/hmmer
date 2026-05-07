# nhmmer GPU Support — Progress

Last updated: 2026-05-07

## Architecture

GPU nhmmer uses a **GPU SSV + GPU filters + GPU scanning Viterbi + threaded CPU downstream** pipeline. All GPU stages are default-on with `--gpu`.

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
│ Threaded CPU Downstream (N threads)                 │
│   → post-Viterbi: Forward, Backward, domain def    │
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

## Benchmark Results (2026-05-07)

**Target**: chr22.fa (50MB, ~101.6M residues both strands)
**System**: CUDA-enabled GPU, 4 CPU threads

| Path | MADE1 (M=80) | query_short (M=151) | query_medium (M=501) |
|------|:---:|:---:|:---:|
| CPU-1 | 0.97s / 465 | 1.29s / 363 | 6.32s / 648 |
| CPU-4 | 0.36s / 465 | 0.46s / 363 | 1.87s / 648 |
| GPU-4 FASTA | 2.87s / 465 | 6.47s / 372 | 46.5s / 693 |
| GPU-4 nucdb | 2.31s / 465 | — | — |
| GPU-4 nucdb-resident | 2.44s / 465 | — | — |

### Parity Notes

- **MADE1 (M=80)**: Perfect parity (465=465) across all paths
- **query_short (M=151)**: GPU reports 372 vs CPU 363 (9 extra hits)
- **query_medium (M=501)**: GPU reports 693 vs CPU 648 (45 extra hits)

Extra GPU hits come from the scanning Viterbi generating more sub-windows for larger models. The GPU uses fixed `xw_*` profile parameters (not reconfigured per window length), making it slightly more permissive. This is a known pre-existing discrepancy.

## Performance Analysis

### Why GPU is slower than CPU-4

1. **Fixed overhead (~0.5s)**: CUDA context initialization (amortized for multi-query)
2. **CPU downstream dominates**: ForwardParser + domain definition = O(window_length × M) per surviving window. For query_medium with 648+ hits, this is ~40s of CPU work
3. **GPU scanning Viterbi shared memory**: O(M) bytes per warp → reduced SM occupancy for large M
4. **Extra hits amplify CPU work**: GPU passes more windows → more CPU Forward/Backward calls

### Where GPU wins

- Multi-query workloads (amortize engine init)
- Nucdb format (saves ~0.4s FASTA parsing overhead)
- When GPU Forward/domain is eventually implemented

### Key bottlenecks by model size

| Model size | Dominant cost | Potential fix |
|-----------|--------------|---------------|
| M≤100 | Fixed overhead (CUDA init, I/O) | Multi-query, nucdb |
| M=100-500 | CPU downstream (Forward/Backward) | GPU ForwardParser |
| M>500 | CPU downstream + kernel arithmetic | GPU ForwardParser + kernel tuning |

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

All phases complete. Branch `worktree-h3-gpu-nhmmer` contains 8 commits on top of `h3-gpu`:

```
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
| High | GPU ForwardParser/domain definition | Very high | Only path to beating CPU-4 for M>100 |
| Medium | Fix parity for query_short/medium (per-window xw_* reconfigure) | Medium | Eliminates false positives |
| Low | Async strand overlap | Low | ~0.1-0.3s savings |
| Low | GPU Forward pre-filter | Medium | Reduces CPU downstream load |
