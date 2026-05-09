# nhmmer GPU vs CPU Performance Gap

Last updated: 2026-05-09

## TL;DR

GPU-4 nhmmer is **~1.8-3.3x slower than CPU-4** on single-query FASTA workloads
(~1.4-2.6x slower with overlap-nucdb). After fixing the scanning Viterbi
threshold bug (wrong Gumbel invsurv function), the GPU now has near-exact hit
parity with CPU (0-3 hit difference). The remaining gap is CUDA init overhead
plus envelope-finding runtime. The skip-Forward optimization
(`p7_pli_postFwd_LongTarget`) eliminates redundant CPU Forward when
`--gpu-fwd-prefilter` is set.

## Current Benchmark (RTX 4090, chr22)

| Path | MADE1 (M=80) | query_short (M=151) | query_medium (M=501) |
|------|:---:|:---:|:---:|
| CPU-1 | 1.08s / 465 | 1.33s / 363 | 5.67s / 648 |
| CPU-4 | 0.31s / 465 | 0.40s / 363 | 1.57s / 648 |
| GPU-4 FASTA | 1.01s / 462 | 0.99s / 363 | 2.87s / 648 |
| GPU-4 nucdb | 0.78s / 462 | 0.97s / 363 | 2.50s / 648 |
| GPU-4 overlap-nucdb | 0.64s / 462 | 1.02s / 363 | 2.23s / 648 |
| Ratio (GPU-4 FASTA vs CPU-4) | 3.3x | 2.5x | 1.8x |

## Aligned per-stage Gantt — query_medium (M=501) on chr22

Same stage names on both rows. Each bar is positioned at where the stage
actually runs in wall-time. Resolution: 1 char ≈ 0.1s. Both axes start at
t = 0; CPU-4 finishes at 1.82s, GPU-4 at 5.32s.

```
                            0     0.5    1.0    1.5    2.0    2.5    3.0    3.5    4.0    4.5    5.0
                            |─────|──────|──────|──────|──────|──────|──────|──────|──────|──────|

load + setup        CPU-4   ▏                                                                          0.05s
                    GPU-4   █████████                                                                  0.88s

SSV                 CPU-4   ·██████                                                                    0.64s
                    GPU-4            ██                                                                0.16s

null + bias + MSV   CPU-4         █                                                                    0.13s
                    GPU-4              █                                                               0.09s

Viterbi             CPU-4          ███                                                                 0.33s
                    GPU-4               ·                                                              0.01s

Forward             CPU-4             █                                                                0.12s
                    GPU-4               ████                                                           0.39s

Backward            CPU-4              ▊                                                               0.07s
                    GPU-4                   █                                                          0.12s

envelope-find       CPU-4               █████                                                          0.50s
                    GPU-4                    ███████████████████████                                   2.32s
                                              (parallel across 4 worker threads)

GPU dom rescore     CPU-4   —— (CPU rescores inline inside envelope-find) ——
                    GPU-4                    ███████████████                                           1.49s
                                              (mutex-serialized, overlaps envelope-find)

GPU trim            CPU-4   ——
                    GPU-4                                   █████████                                  0.91s
                                                            (mutex-serialized, overlaps envelope-find)

hit output          CPU-4                   ·                                                          ~0
                    GPU-4                                                                       ██     0.18s

────────────────────────────────────────────────────────────────────────────────────────────────
WALL                CPU-4   ██████████████████                                                         1.82s
                    GPU-4   █████████████████████████████████████████████████████                      5.32s
```

### How to read this

- **CPU-4 stage values** come from `p7_pli_Statistics` (sums across 4 worker
  threads, divided by 4 to give per-thread wall ≈ total wall when balanced).
  CPU stages are interleaved per-window in reality, but cumulatively each
  stage's total wall is what's shown — the bars are laid end-to-end in the
  order each stage executes within a window.
- **GPU-4 stage values** come from `NHMMER_GPU_INFO::t_*` (per-strand
  stages, genuinely sequential) and worker sub-buckets in
  `nhmmer_gpu_worker_process_post_vit_gpu`. The first six GPU rows are
  truly back-to-back; the last four (envelope-find / dom rescore / trim /
  output) all run **concurrently** inside the `t_cpu_workers = 3.66s`
  block (1.67s → 5.32s) — envelope-find runs in parallel across threads,
  dom rescore and trim hold a single CUDA mutex, output runs at the tail.

### Stage-by-stage

| Stage | CPU-4 wall | GPU-4 wall | Δ |
|-------|:---:|:---:|:---:|
| load + setup | 0.05s | 0.88s | **−0.83s** ← CUDA init + FASTA |
| SSV | 0.64s | 0.16s | **+0.48s** ← GPU 4× faster |
| null + bias + MSV | 0.13s | 0.09s | +0.04s |
| Viterbi (prefilter + scanning) | 0.33s | 0.01s | **+0.32s** ← GPU 33× faster |
| Forward | 0.12s | 0.39s | −0.27s ← GPU does 2 passes (prefilter + Backward-only) |
| Backward | 0.07s | 0.12s | −0.05s |
| envelope-find (`p7_domaindef`) | 0.50s | 2.32s | **−1.82s** ← biggest single regression |
| GPU dom rescore + trim | (inline) | 2.40s ¹ | −2.40s ← new GPU-only stage, mutex-bound |
| hit output | ~0 | 0.18s | −0.18s |
| **WALL TOTAL** | **1.82s** | **5.32s** | **−3.50s** |

¹ summed wall of mutex-held calls; concurrent in real time with envelope-find.

### Why each loss happens

- **envelope-find (−1.82s)**: GPU pipeline pushes **7,364 windows** into
  `p7_domaindef_ByPosteriorHeuristics` vs ~3,800 the CPU pipeline lets
  through its F3 P-value gate. GPU writes `xf`/`xb` into pinned-host
  buffers, so the domaindef call hits cold L1/L2.

- **Forward (−0.27s)**: GPU runs Forward in two passes — a prefilter pass
  that produces `xf` for the F3 gate, then a Backward-only batch that
  re-uses the saved `xf`. CPU runs Forward once. PCIe D2H of `xf` costs
  more than the kernel saves on small-M models.

- **GPU dom rescore + trim (−2.40s)**: New stage with no CPU equivalent.
  CPU does this inline inside envelope-find. Made worse by the
  `pthread_mutex_lock(gpu_domain_mutex)` that serializes all four worker
  threads through one CUDA context.

- **CUDA init (−0.83s)**: First-query overhead. Multi-query workloads
  amortize this away.

## Direct comparison — query_medium (M=501)

| Path | Wall | Hits | vs CPU-4 |
|------|:---:|:---:|:---:|
| CPU-1 | 6.40s | 215 | 3.5x slower |
| **CPU-4** | **1.82s** | **215** | **1.0x** |
| GPU-1 | 7.63s | 218 | 4.2x slower |
| GPU-4 (FASTA) | 5.32s | 218 | 2.9x slower |

GPU-4 is even slower than CPU-1, despite committing the GPU + 4 CPU threads.

## What Would Further Improve It

| Fix | Estimated savings (query_medium) | Effort | Status |
|-----|---------|--------|--------|
| Fix GPU scanning Viterbi threshold bug | **~2.5s** (halved wall time) | Low | **Done** |
| Move envelope-finding to GPU | Eliminate ~1-2s bottleneck | Very high | Open |
| Drop GPU domain-rescore mutex (per-thread streams) | ~0.5-1.0s @ cpu 4 | Medium | Open |
| Pinned memory + async D2H | ~10-20ms per batch | Low | Open |
| OptAcc kernel parallelization | Small | Medium | Open (needs max-prefix scan) |
| Multi-query amortization | ~0.5s | Done | Engine reuse implemented |
| Nucdb format | ~0.4s | Done | Eliminates FASTA parsing |
| Overlap-nucdb (GPU-resident SSV) | ~0.3-0.6s | Done | Zero per-chunk H2D |
| Skip-Forward optimization | Varies | Done | `p7_pli_postFwd_LongTarget()` uses GPU xf |
| Redundant Forward elimination | ~0.5–5s | Done | Prefilter saves xf for Backward-only |
| Parallel-thread domain kernels | ~0.04–0.4s | Done | 4/6 kernels use T threads |

## Hit parity

GPU domain rescoring uses nj=0 (unihit mode) matching CPU behavior.

After the scanning Viterbi threshold fix:
- **MADE1 (M=80)**: 462 vs 465 (3-hit difference, <1%)
- **query_short (M=151)**: 363 vs 363 (exact match)
- **query_medium (M=501)**: 648 vs 648 (exact match)

Remaining differences are from float32 vs double precision in Forward/Backward
accumulation.

## Historical Improvement (MADE1)

| Version | MADE1 time | Key change |
|---------|:---:|:---:|
| Pre-batching (per-domain GPU calls) | 34s | 5000+ individual GPU calls × 5ms overhead each |
| Cross-window batching + trim batching | 1.74s | Single GPU call for all domains |
| Parallel-thread domain kernels | 1.37s | T threads/block with prefix scan (4 of 6 kernels) |
| Forward-Backward split (prefilter saves xf) | 1.35s | Eliminates redundant Forward computation |
| **Scanning Viterbi threshold fix** | **0.91s** | Fix Gumbel invsurv + nullsc mismatch |

**37x improvement** from serial per-domain calls to current design.

## Root Cause of Prior 3-4x Slowdown (FIXED)

Two bugs in `cuda_compute_viterbi_thresholds_kernel` caused GPU scanning
Viterbi to produce ~7x more sub-windows than CPU, inflating `p7_domaindef`
runtime (75-94% of wall) by ~7x:

1. **Primary bug: wrong Gumbel inverse survival function** (~5000 int16 units).
   `esl_gumbel_invsurv_device` computed `mu - log(-log(p))/lambda` (inverse
   CDF) instead of `mu - log(-log(1-p))/lambda` (inverse survival). With
   F2=0.003, this produced `invP ≈ -11.8` instead of the correct `≈ -1.3`,
   lowering thresholds by `0.693 × 10.53 × 721.3 ≈ 5265` int16 units.

2. **Secondary bug: nullsc mismatch for bias subtraction** (variable, up to
   ~2500 units for long windows). The kernel subtracted `null(loc_window_len)`
   from `bias_scores[i]` to isolate composition bias, but `bias_scores[i]` was
   computed using `window_len`. When `window_len > max_length`, the kernel used
   the wrong null model base, making `filtersc` more negative and further
   lowering thresholds.

## Reproducing

```sh
# CPU per-stage (sums across worker threads; divide by --cpu N for per-thread wall)
src/nhmmer --cpu 4 --noali \
    benchmark-data/nucleotide-bench/work/query_medium.hmm \
    benchmark-data/nucleotide-bench/work/chr22.fa \
    2>&1 | grep '^# Stage'

# GPU per-stage (parallel CPU stages reported as max-across-workers,
# mutex-serialized GPU stages summed)
src/nhmmer --gpu --cpu 4 --noali \
    benchmark-data/nucleotide-bench/work/query_medium.hmm \
    benchmark-data/nucleotide-bench/work/chr22.fa \
    2>&1 | grep -E 'GPU timing|^  '
```

## Benchmark Script

Run `test-speed/x-nhmmer-gpu-bench` for quick CPU-1 / CPU-4 / GPU-4 wall
comparison.
