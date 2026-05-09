# nhmmer GPU Support — Progress

Last updated: 2026-05-09

## Architecture

GPU nhmmer uses a **GPU SSV + GPU filters + GPU scanning Viterbi + GPU Forward prefilter + GPU FB parser + GPU domain rescoring + threaded CPU hit reporting** pipeline. All GPU stages are default-on with `--gpu`.

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
│   → GPU Forward parser on sub-windows (saves xmx)   │
│   → removes ~50-60% of windows before Backward      │
│   → survivor xf compacted for Backward-only stage   │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ GPU Backward Parser (batch Backward-only xmx)       │
│   → uses pre-computed Forward xf from prefilter     │
│   → runs Backward parser only on F3 survivors       │
│   → returns xf+xb arrays for domaindef envelope-find│
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ CPU: Envelope finding (domaindef heuristics)        │
│   → p7_domaindef_ByPosteriorHeuristics(envelopes)   │
│   → identifies domain envelopes (ienv, jenv)        │
│   → reparameterizes model per domain                │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ GPU Domain Rescoring (batched CUDA kernels)         │
│   → cross-window batching: all domains at once      │
│   → 6 kernels: Fwd + Bck + Decoding + OA + Trace   │
│     + Domcorrection                                 │
│   → trim batching: trimmed envelopes in 2nd batch   │
│   → returns envsc, domcorr, traces per domain       │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ CPU: Hit reporting                                  │
│   → build P7_ALIDISPLAY from GPU traces             │
│   → score computation, p7_tophits_CreateNextHit     │
│   → coordinate transforms (strand-aware)            │
└─────────────────────────────────────────────────────┘
```

Both strands processed sequentially (forward, then reverse complement).

## Files

| File | Purpose |
|------|---------|
| `src/nhmmer.c` | CLI (`--gpu` option), engine lifecycle, nucdb detection |
| `src/nhmmer_internal.h` | `NHMMER_GPU_INFO`, `NHMMER_GPU_WINDOW_BATCH` structs |
| `src/nhmmer_gpu.c` | GPU orchestration: batch filter, Viterbi, scanning Vit, domain rescore, hit reporting |
| `src/cuda/p7_cuda_ssv_longtarget.cu` | SSV longtarget kernel + `SSVLongtargetResident` |
| `src/cuda/p7_cuda_viterbi_longtarget.cu` | Scanning Viterbi kernel + threshold kernel |
| `src/cuda/p7_cuda_fb_parser.cu` | Forward/Backward parser batch: Forward-only, Backward-only, and combined F+B modes |
| `src/cuda/p7_cuda_domain_rescore.cu` | Domain rescoring: 6 GPU kernels (Fwd/Bck/Decoding/OA/OATrace/Domcorr) |
| `src/cuda/p7_cuda_runtime.cu` | Engine create/destroy/reset, nucdb upload/release |
| `src/cuda/p7_cuda_internal.h` | Engine struct (all device buffers including domain rescore grow-only buffers) |
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

## Benchmark Results (2026-05-09)

**Target**: chr22.fa (50MB, ~101.6M residues both strands)
**System**: RTX 4090, 4 CPU threads

| Path | MADE1 (M=80) | query_short (M=151) | query_medium (M=501) |
|------|:---:|:---:|:---:|
| CPU-4 | 0.32s / 465 | 0.48s / 363 | 1.66s / 648 |
| GPU-4 FASTA | 0.91s / 462 | 1.40s / 363 | 2.85s / 648 |

### GPU Domain Rescoring Performance

Cross-window batching + trim batching replaces per-domain GPU calls with two batched calls per strand:

| Query | Domains/strand | GPU kernel time | Trim domains | Trim kernel time |
|-------|:---:|:---:|:---:|:---:|
| MADE1 | ~3000 | 27ms | ~1100 | 12ms |
| query_short | ~600 | varies | ~300 | varies |
| query_medium | ~300 | varies | ~100 | varies |

Previous per-window approach: MADE1 took 34s (5000+ individual GPU calls at ~5ms each). Cross-window batching: **1.74s** (18x improvement). Forward-Backward split (prefilter saves xf, Backward-only): **1.35s** (further 1.3x). Scanning Viterbi threshold fix: **0.91s** (further 1.5x).

### Parity Notes

- **MADE1 (M=80)**: 462 vs 465 (3-hit difference, <1%)
- **query_short (M=151)**: 363 vs 363 (exact match)
- **query_medium (M=501)**: 648 vs 648 (exact match)

Remaining differences are from float32 vs double precision in Forward/Backward accumulation. Domain rescoring uses nj=0 (unihit mode) matching CPU behavior.

## GPU Domain Rescoring Architecture

### Three-Phase Design (in `nhmmer_gpu_worker_process_post_vit_gpu`)

**Phase 1: Envelope collection (CPU)**
- Iterates over all Viterbi survivor windows
- Runs F3 gate, `p7_domaindef_ByPosteriorHeuristics(envelopes_only=TRUE)`
- Collects domain envelopes: `NHMMER_GPU_PDOM` struct (ienv, jenv, Ld, rfv_copy)
- Reparameterizes model per domain (composition-adjusted emissions)
- Stores per-window context in `NHMMER_GPU_WCTX`

**Phase 2: GPU batch call**
- Single `p7_cuda_DomainRescoreBatch` call for all ~3000-6000 domains
- Bulk H2D: staging buffers → single memcpy for DSQ and RFV
- 6 sequential kernels (Fwd, Bck, Decoding, OA, OATrace, Domcorr)
- Single-thread-per-block design (one CUDA block per domain)
- Grow-only engine buffers (never shrink between calls)

**Phase 3: Unpack + trim batch + hit reporting (3 sub-phases)**
- **3a**: Build traces from GPU output, create P7_ALIDISPLAY, identify trim candidates
- **3b**: Single GPU batch call for all ~1000-2000 trim domains
- **3c**: Apply trim results, report hits (coordinate transform, scoring, p7_tophits)

### CUDA Kernels (`p7_cuda_domain_rescore.cu`)

| Kernel | Purpose | Design |
|--------|---------|--------|
| `cuda_domain_fwd_full_kernel` | Full Forward with matrix storage | T threads/block, parallel prefix scan for D-state, tree reduction for xE |
| `cuda_domain_bck_full_kernel` | Full Backward | T threads/block, reverse prefix scan, reads forward xmx for scaling |
| `cuda_domain_decoding_kernel` | Posterior Decoding (fwd*bck/total) | T threads/block, strided element-wise, no prefix scan needed |
| `cuda_domain_optacc_kernel` | Optimal Accuracy DP | 1 thread/block (deferred: needs max-prefix scan) |
| `cuda_domain_oatrace_kernel` | OA Traceback (reverse) | 1 thread/block (inherently sequential traceback) |
| `cuda_domain_fwd_scoreonly_kernel` | Domcorrection Forward (orig rfv) | T threads/block, parallel prefix scan, score-only (no matrix storage) |

### Grow-Only Engine Buffers

Device allocations in `P7_CUDA_ENGINE` that only grow (never freed between calls):
- `d_dom_dpf[4]`: DP matrices (fwd, bck, pp, oa) — keyed on total_dp_cells
- `d_dom_xmx[4]`: Special state matrices — keyed on total_xmx
- `d_dom_rfv_all`, `d_dom_dsq_all`: Packed per-domain data
- `d_dom_*_ptrs`, `d_dom_lengths`, etc.: Per-domain metadata — keyed on ndomains
- `d_dom_trace_*`: Trace output arrays — keyed on ndomains × max_trace_len

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

All phases complete including GPU domain rescoring. Branch `worktree-h3-gpu-nhmmer` is rebased on top of latest `h3-gpu` (fused SSV+null+bias+gate kernel, multi-warp-per-block SSV/Viterbi, templated kernels, ~7.3x protein profmark). Commits:

```
d1a7a4a6 gpu: fix nhmmer domain rescoring parity by using unihit nj=0
93c51c5b gpu: eliminate redundant Forward in nhmmer prefilter→FB pipeline
1e0116e0 gpu: GPU domain rescoring with cross-window batching for nhmmer
775fa2a5 gpu: GPU ForwardBackward parser for nhmmer domaindef
8fe7522d gpu: nhmmer performance optimizations (per-window thresholds, Forward pre-filter, kernel tuning)
8e6eae3d docs: comprehensive nhmmer GPU documentation and benchmark scripts
97cb6e06 gpu: pre-stored RC from nucdb + GPU-resident SSV longtarget path
3cd1287a cleanup: remove --gpu-compare debug flag and update nhmmer GPU docs
c274036f gpu: nucleotide GPU database format (nucdb)
5d8c80d7 gpu: port h3-gpu optimizations to nhmmer
fd4c34e3 gpu: GPU scanning Viterbi for nhmmer with GPU threshold computation
4787499c gpu: batch MSV/bias filter and Viterbi pre-filter for nhmmer
09e6c38f gpu: threaded downstream pipeline for nhmmer with CUDA engine reuse
(+ h3-gpu base: fused SSV kernel, multi-warp-per-block, templated Viterbi, Forward prefix M≤2044)
```

## Future Work

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| High | Multi-threaded warp-cooperative domain kernels | High | Would parallelize within each domain, reducing kernel time 4-8x |
| Low | Async strand overlap | Low | ~0.1-0.3s savings |
