# nhmmer GPU Support — Progress

Last updated: 2026-05-10

## Architecture

GPU nhmmer uses a **GPU SSV + GPU filters + GPU scanning Viterbi + GPU Forward prefilter + GPU Forward/Backward parser handoff + threaded CPU domain hit reporting** pipeline by default. The GPU Forward prefilter is default-on with `--gpu` and reduces post-Viterbi survivors before CPU workers. GPU Forward/Backward parser matrix reuse is also default-on with `--gpu`; it hands GPU xmx matrices to CPU domain processing after enforcing the exact F3 gate. The previous extra single-score GPU Viterbi prefilter has been removed; GPU scanning Viterbi is the CPU-equivalent Viterbi boundary.

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
│ GPU Scanning Viterbi (warp-per-window, 8 warps/blk) │
│   → full DP per window, emits sub-windows           │
│   → GPU threshold: bias kernel + analytic null      │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ GPU Forward Pre-filter (exact F3 gate)              │
│   → GPU Forward parser on sub-windows               │
│   → removes windows before parser/domain handoff    │
│   → survivor list compacted for GPU FB parser       │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ GPU Forward/Backward Parser Handoff                 │
│   → reuses GPU Forward xmx/fwdsc from prefilter     │
│   → runs GPU Backward xmx and passes matrices to CPU│
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
│ CPU domain definition / hit reporting               │
│   → p7_domaindef_ByPosteriorHeuristics              │
│   → null2/domain scoring and output remain CPU-side │
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
| `src/nhmmer.c` | CLI (`--gpu` option), engine lifecycle, shared nucdb open/upload, timing accounting |
| `src/nhmmer_internal.h` | `NHMMER_GPU_INFO`, `NHMMER_GPU_WINDOW_BATCH` structs |
| `src/nhmmer_gpu.c` | GPU orchestration: batch filter, Viterbi, scanning Vit, shared nucdb reuse, domain rescore, hit reporting |
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
# Basic GPU nhmmer (default path: GPU filters/scanning Viterbi/Forward prefilter, CPU domain continuation)
src/nhmmer --gpu --cpu 16 --noali query.hmm target.fa

# With fast nucdb (default overlap enables GPU-resident SSV when overlap >= model max_length)
src/hmmnucdb target.fa target.nucdb        # build once; default --overlap 2001
src/nhmmer --gpu --cpu 16 --noali query.hmm target.nucdb.nucdb

# Ordinary no-overlap nucdb for diagnostics
src/hmmnucdb --overlap 0 target.fa target-o0.nucdb
src/nhmmer --gpu --cpu 16 --noali query.hmm target-o0.nucdb.nucdb

# Hidden flags (group 99, not shown in help)
--gpu-batch             # SSV/bias batch on GPU (default-on with --gpu)
--gpu-vit-longtarget    # Scanning Viterbi (default-on with --gpu)
--gpu-fwd-prefilter     # Deprecated no-op: GPU Fwd/Bwd parser reuse is default-on
--gpu-no-fwd-prefilter  # Diagnostic: use older CPU Fwd/Bwd continuation after GPU Forward prefilter
--gpu-cpu-postmsv       # Bypass GPU scanning Viterbi/Fwd; CPU postSSV path
--gpu-compare           # Debug: compare GPU vs CPU scores per stage
--gpu-chunk-size N      # Chunk size (default 65536)
--gpu-device N          # CUDA device selection
```

## Benchmark Results (2026-05-10)

**Target**: chr22.fa (50MB, ~101.6M residues both strands)
**System**: RTX 4090, 16 CPU threads by default

`test-speed/x-nhmmer-gpu-bench` now defaults to an all-sample multi-query run with `--cpu 16` for both CPU and GPU paths. It concatenates `MADE1`, `query_short`, and `query_medium` into `benchmark-data/nucleotide-bench/work/nhmmer-gpu-all-samples.hmm`, then runs one `nhmmer` process per path. This is the preferred benchmark for current GPU nhmmer because CUDA engine creation and resident `.nucdb` upload are shared across searches. The old one-query-at-a-time behavior remains available with `NHMMER_GPU_BENCH_MODE=per-query`; set `NHMMER_GPU_BENCH_CPU=N` to override the default thread count.

Current combined all-sample result:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.535s | 1476 |
| CPU-16 | 1.008s | 1476 |
| GPU-16 FASTA | 2.035s | 1476 |
| GPU-16 nucdb, no overlap | 1.710s | 1476 |
| GPU-16 overlap-nucdb | 1.536s | 1476 |

Current parity script result:

| Query/path | Result |
|------------|:---:|
| MADE1 FASTA | 465 = 465 |
| MADE1 nucdb | 465 = 465 |
| query_short FASTA | 363 = 363 |
| query_medium FASTA | 648 = 648 |

Recent per-query baseline:

| Path | MADE1 (M=80) | query_short (M=151) | query_medium (M=501) |
|------|:---:|:---:|:---:|
| CPU-1 | 0.835s / 465 | 1.324s / 363 | 5.809s / 648 |
| CPU-4 | 0.278s / 465 | 0.388s / 363 | 1.713s / 648 |
| GPU-4 FASTA | 0.706s / 465 | 0.703s / 363 | 1.668s / 648 |
| GPU-4 nucdb | 0.724s / 465 | 0.817s / 363 | 1.493s / 648 |
| GPU-4 overlap-nucdb | 0.613s / 465 | 0.644s / 363 | 1.669s / 648 |

The older 2.109s overlap `.nucdb` result was stale: it predated the GPU window-ordering fixes and mixed old default runs with targeted Fwd/Bwd handoff runs.

### Combined-Run Timing Accounting

`nhmmer --gpu` now reports both search-stage timing and outside-search timing. The combined overlap `.nucdb` run showed one shared CUDA engine creation (`0.446s`) and one shared `.nucdb` upload (`0.029s`) for all three queries, followed by per-query reconstruction/search/report buckets. In that run, process elapsed time was `2.205s`, summed GPU loop wall time was `1.710s`, and process time outside the GPU search loops was `0.495s`.

| Query | Search stages | GPU loop wall | Query elapsed | CPU workers | Domain workflow | nucdb reconstruct | Query outside search |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MADE1 | 0.150s | 0.236s | 0.237s | 0.062s | 0.059s | 0.080s | 0.001s |
| query_short | 0.219s | 0.302s | 0.303s | 0.123s | 0.119s | 0.077s | 0.001s |
| query_medium | 1.102s | 1.172s | 1.174s | 0.756s | 0.738s | 0.065s | 0.002s |

Process-level line from that run:

| Bucket | Time |
|--------|:---:|
| CUDA engine create | 0.446s |
| shared nucdb open/mmap | 0.000s |
| shared nucdb upload | 0.029s |
| query pre-search setup | 0.002s |
| GPU loop wall total | 1.710s |
| query post-loop cleanup | 0.000s |
| post-search/report | 0.003s |
| CUDA reset | 0.000s |
| CUDA destroy | 0.003s |
| process outside search | 0.495s |
| process elapsed | 2.205s |

Current strict query_medium parity audit on 2026-05-10:

| Path | Target | Main-output hit lines | `--tblout` rows |
|------|--------|:---:|:---:|
| CPU-4 | `chr22.fa` | 648 | 215 |
| GPU default, fast overlap nucdb | `chr22-overlap.nucdb.nucdb` | 648 | 215 |

The CPU and GPU `--tblout` rows had no diff. `hmmnucdb` defaults to the fast overlap construction (`--overlap 2001`) and `nhmmer --gpu` defaults to GPU Fwd/Bwd parser handoff. The code-path verification is nonzero `GPU FB parser` time with zero CPU Forward/Backward stage time.

Current fast `.nucdb` GPU breakdown for query_medium:

| Bucket | Time |
|--------|:---:|
| SSV longtarget | 0.109s |
| extend+merge | 0.002s |
| batch filter | 0.055s |
| scanning Viterbi | 0.122-0.242s |
| Forward prefilter | 0.050s |
| GPU FB parser | 0.011s |
| CPU workers | 0.288s |
| worker domain workflow | 0.256s |
| worker CPU Backward | 0.000s |

Current query_medium launch/occupancy instrumentation on fast overlap `.nucdb`:

| Stage | Launch shape | Occupancy | Grid coverage |
|-------|--------------|:---:|:---:|
| SSV longtarget | 2 launches, last grid=800, block=32, smem=1002B | 50.0% theoretical, 24 active warps/SM of 48; 97.3% device-active in SSV wall | 6.25x on 128 SMs |
| Scanning Viterbi | 2 launches, last grid=1097, block=32, smem=24192B | 8.3% theoretical, 4 active physical warps/SM of 48; 46.5-94.0% device-active in Viterbi CUDA call depending on first-call allocation | 8.51x on 128 SMs |

The 16-thread baseline changes the conclusion: GPU-16 remains hit-clean but is slower than CPU-16 on the combined all-sample benchmark. Focused query_medium shows why the occupancy counter alone was misleading. SSV is a one-warp-per-block kernel with only 50% theoretical occupancy, but it is already about 97% device-active; grouping multiple chunks per block raised theoretical occupancy and did not improve wall time. Scanning Viterbi was wasting lanes 8-31 of each physical warp for nucleotide DP; it now packs four 8-lane DP groups per physical warp, reducing the repeated scan kernel from about `0.125-0.128s` to about `0.112-0.116s` in focused runs. A score-only Forward prefilter experiment failed parity badly because `p7_cuda_ForwardScoreDsqdataSubset()` is not parser-equivalent for this nucleotide F3 gate, so the accepted path still uses the GPU Forward parser xmx handoff. The final combined benchmark remains run-to-run volatile (`1.374-1.536s` observed for GPU-16 overlap `.nucdb` after the subwarp change; `1.418s` after parser event reuse), so the measured kernel improvement does not yet translate to a stable end-to-end win over CPU-16. The remaining combined gap is CUDA setup, `.nucdb` reconstruction, SSV, Forward/parser work, Viterbi allocation variance, and residual CPU domain workflow together exceeding the strong CPU-16 baseline.

Focused repeats show run-to-run variance in CUDA engine setup and scanning Viterbi. After dynamic survivor-window scheduling, GPU CPU-domain workflow is no longer inflated relative to CPU-16 for query_medium: CPU-16 wall-stage trace measured domain at `0.250s`; GPU-16 focused repeats measured worker domain at `0.239-0.258s` and GPU loop wall at `0.617-0.814s` depending mostly on CUDA Viterbi allocation variance.

### CPU Wall-Stage Trace

`HMMER_NHMMER_CPU_WALL_TRACE=1` prints max-across-worker stage times before CPU worker pipelines are merged. This is intentionally separate from the normal `# Stage ... time` lines, which remain summed across workers. Current CPU-16 query_medium wall-stage trace:

| Stage | Wall estimate |
|-------|:---:|
| SSV | 0.239s |
| MSV host | 0.015s |
| bias | 0.044s |
| Viterbi | 0.141s |
| Forward | 0.044s |
| Backward | 0.032s |
| domain | 0.250s |

### Dynamic CPU-Continuation Scheduling

The GPU continuation workers now pull post-Forward survivor windows from a shared work index instead of receiving contiguous equal-window slices. This better matches the CPU-only work-queue behavior because domain-definition cost is driven by region/envelope complexity, not just survivor-window count. The change also computes overlap accounting from global neighboring survivor windows, preserving parity while avoiding worker-boundary artifacts.

### Historical GPU Domain Rescoring Performance

Cross-window batching + trim batching replaces per-domain GPU calls with two batched calls per strand:

| Query | Domains/strand | GPU kernel time | Trim domains | Trim kernel time |
|-------|:---:|:---:|:---:|:---:|
| MADE1 | ~3000 | 27ms | ~1100 | 12ms |
| query_short | ~600 | varies | ~300 | varies |
| query_medium | ~300 | varies | ~100 | varies |

Previous per-window approach: MADE1 took 34s (5000+ individual GPU calls at ~5ms each). Cross-window batching: **1.74s** (18x improvement). Forward-Backward split (prefilter saves xf, Backward-only): **1.35s** (further 1.3x). Scanning Viterbi threshold fix: **0.91s** (further 1.5x). Nucdb + overlap-nucdb paths: **0.64s** (GPU-resident SSV, zero per-chunk H2D).

### Default Matrix Reuse

The default path always runs the GPU Forward prefilter with the exact F3 gate, then reuses GPU Forward/Backward parser xmx for surviving windows and uses CPU domain processing. This path matched default GPU hit counts on the current chr22 smoke set for MADE1, query_short, and query_medium on both FASTA and nucdb.

GPU Backward parser matrix handoff was tested after fixing the CPU diagnostic length configuration to use `window_len` rather than `min(window_len, om->max_length)`. `--gpu-compare` shows Forward/Backward parser scores match CPU to float noise. query_short showed a 1 bp envelope-start shift in one coordinate comparison, and a similar 1 bp envelope-start jitter can occur between repeated default `--gpu` runs, so this is a boundary-sensitivity issue rather than a hit-count regression.

Current Forward-prefilter evidence: on query_medium fast overlap `.nucdb`, the default Fwd/Bwd handoff kept 341/707 post-Viterbi windows (48.2%) in the focused timing run and the speed script measured 1.232s versus 1.669s for CPU-4. Worker CPU Backward was 0.000s, confirming the parser handoff rather than the older CPU Forward/Backward continuation.

Implementation:
- `NHMMER_GPU_WORKER` carries `prefilter_xf`, `prefilter_fwdsc`, `prefilter_xf_offset` (non-owning pointers into shared buffers)
- `nhmmer_gpu_worker_process_post_fwd()` injects xf into `pli->oxf->xmx`, reconstructs `totscale`, calls `p7_pli_postFwd_LongTarget()`
- `p7_pli_postFwd_LongTarget()` in `p7_pipeline.c` performs F3 check using provided fwdsc, then Backward + domaindef + hit reporting
- Gated on `use_skip_fwd = info->do_gpu_fwd && prefilter_xf != NULL`; `info->do_gpu_fwd` is default-on with `--gpu`

### Parity Notes

- **MADE1 (M=80)**: FASTA 462 vs 465 (within 1% tolerance); nucdb 462 vs 462 exact in the parity script.
- **query_short (M=151)**: 363 vs 363 exact.
- **query_medium (M=501)**: 648 vs 648 exact.

Remaining MADE1 FASTA delta is at the GPU long-target window/filter boundary; default `--gpu` now reuses GPU Forward/Backward parser matrices, then keeps domain processing and hit reporting on CPU.

### Window Ordering Fixes (2026-05-10)

The root cause of inflated GPU worker time was not CPU Forward/Backward. GPU kernels emit windows via `atomicAdd`, so output order is not guaranteed. `p7_pli_ExtendAndMergeWindows()` only merges adjacent windows, so unsorted GPU windows defeated merge and inflated downstream domain-definition work. The first affected boundary is the SSV long-target output before the MSV-window extend/merge. The scanning-Viterbi path has the same ordering requirement, and it must merge seeds in parent MSV-window-local coordinates before converting to target coordinates.

Fixes:
- Sort GPU SSV windows by strand/sequence/coordinate before the first `p7_pli_ExtendAndMergeWindows()` call.
- Sort GPU Viterbi seeds by `(window_id, position, model_k)` before extension/merge.
- Extend and merge GPU scanning-Viterbi seeds in parent-window-local coordinates, then convert merged windows back to target coordinates.
- Clamp overlap accounting for `pos_past_vit` and `pos_past_fwd` so overlapping windows cannot underflow residue counters.
- Use exact F3 gating in the GPU Forward prefilter.

### Removed Single-Score Viterbi Prefilter (2026-05-10)

The previous single-score GPU Viterbi prefilter (`--gpu-vit-prefilter`) was removed from nucleotide `nhmmer`. It was an extra GPU-only gate before the real long-target scanning Viterbi stage and had no direct CPU-reference equivalent. The accepted path is `.nucdb` overlap input with GPU SSV, GPU batch MSV/null/bias filtering, GPU scanning Viterbi, GPU Forward prefilter, and GPU Fwd/Bwd parser handoff.

### Nucdb Reverse-Strand Coordinate Fix (2026-05-10)

For `.nucdb` reverse-complement chunks, reconstructed `ESL_SQ` metadata now mirrors `esl_sq_ReverseComplement()` by setting `start = n` and `end = 1`. This prevents negative reverse-strand coordinate transforms in GPU `.nucdb` output and allowed strict CPU FASTA versus GPU overlap `.nucdb` tabular rows to match for query_medium in the smoke audit.

### Bias Filter Precision Fix (2026-05-10)

Fixed two precision bugs in the GPU scanning Viterbi threshold path:

1. **`cuda_bias_filter_kernel`**: Was using a fixed `t[0][0]` transition for all windows (uploaded once at init). Now computes per-window `t00 = L/(L+1)` matching CPU's `p7_bg_SetLength` per-window behavior. Also switched from `logf()` to `(float)log()` matching CPU's double-precision log in `esl_hmm_Forward`.

2. **`cuda_compute_viterbi_thresholds_kernel`**: The bias score is computed against the full parent `window_len`, so the kernel must subtract `nullsc_win` to isolate composition bias before scaling it to the local window length. The compare-side diagnostic uses the same calculation.

3. **Compare-side Viterbi diagnostic**: The CPU reference path in `--gpu-compare` now recomputes `nullsc_win` for each shortened candidate window before subtracting it from the bias filter score. This matches `p7_pli_postSSV_LongTarget()` and removes a diagnostic-only threshold mismatch.

## Historical GPU Domain Rescoring Architecture

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

Build-time options: `--chunk-size` (default 65536), `--overlap` (default 2001), `--fwd-only`.

**Resident path**: When `--overlap >= model_max_length` AND `chunk_size` matches runtime, the kernel reads directly from GPU-uploaded nucdb data (no H2D per sequence). Pre-stored RC chunks eliminate `esl_sq_ReverseComplement`.

## Status Summary

Default `nhmmer --gpu` now uses GPU SSV/batch filtering/Viterbi/Forward prefilter plus GPU Forward/Backward parser matrix handoff before CPU domain/hit reporting. `--gpu-fwd-prefilter` is deprecated compatibility spelling; hidden `--gpu-no-fwd-prefilter` restores the older CPU Fwd/Bwd continuation for diagnostics. Historical implementation notes below include GPU domain-rescoring work that is not part of the accepted default path.

```
a7936ac8 gpu: fix scanning Viterbi threshold bug + add --gpu-compare and --gpu-cpu-postmsv
063207a2 build: ignore src/hmmnucdb and suppress easel submodule dirty state
a5a49fc0 gpu: multi-thread domain rescore kernels + split Forward/Backward parser API
26a6a5e7 gpu: rebase compatibility fix and doc refresh for h3-gpu latest
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
