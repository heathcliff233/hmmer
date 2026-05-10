# nhmmer GPU vs CPU Performance Breakdown

Last updated: 2026-05-10

## TL;DR

The previous “GPU is slower because CPU workers redo Forward/Backward and GPU domain rescoring is mutex-bound” analysis is stale. Current default `nhmmer --gpu` uses the GPU Forward prefilter and GPU Forward/Backward parser handoff by default, so CPU Forward/Backward continuation is gone in the accepted path.

The real issue behind the stale 1.91s versus 2.109s comparison was apples-to-oranges benchmarking plus GPU window merge bugs. GPU SSV windows and GPU Viterbi seeds are emitted with `atomicAdd`, so they are unordered; `p7_pli_ExtendAndMergeWindows()` only merges adjacent windows. The GPU path also needed to extend/merge scanning-Viterbi seeds in parent MSV-window-local coordinates, matching CPU, before converting back to target coordinates. These issues inflated downstream CPU domain workflow. The current fix sorts SSV windows before the first long-target merge, sorts Viterbi seeds by `(window_id, position, model_k)`, extends/merges Viterbi seeds in local coordinates, then converts back to target coordinates.

## Current query_medium Result

Target: chr22/hg38 benchmark. Query: `query_medium.hmm` (M=501). Threads: `--cpu 4`. GPU: RTX 4090. Date: 2026-05-10.

| Path | Target | Output rows | HMMER elapsed | `/usr/bin/time` wall |
|------|--------|:---:|:---:|:---:|
| CPU-4 | `chr22.fa` | 648 | 1.669s speed script | 1.77s focused |
| GPU default, no-overlap nucdb | `chr22.nucdb.nucdb` | 648 | 1.340s | - |
| GPU default, fast overlap nucdb | `chr22-overlap.nucdb.nucdb` | 648 | 1.232s speed script | 1.53-1.86s focused repeats |

The speed-script result now shows the expected fast-overlap `.nucdb` win after removing the extra single-score GPU Viterbi prefilter. The focused `--tblout` audit produced 215 CPU rows and 215 GPU rows with no diff.

## Current Combined Benchmark

`test-speed/x-nhmmer-gpu-bench` now defaults to a combined all-sample benchmark. It runs `MADE1`, `query_short`, and `query_medium` in one `nhmmer` process per path, so CUDA engine creation and resident `.nucdb` upload are measured once instead of repeated for each query.

Current all-sample result on 2026-05-10:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.209s | 1476 |
| CPU-4 | 2.373s | 1476 |
| GPU-4 FASTA | 2.159s | 1476 |
| GPU-4 no-overlap `.nucdb` | 1.864s | 1476 |
| GPU-4 overlap `.nucdb` | 1.684s | 1476 |

Parity after the update was clean: MADE1 FASTA 465=465, MADE1 `.nucdb` 465=465, query_short FASTA 363=363, query_medium FASTA 648=648.

The combined overlap `.nucdb` timing run accounted for the previously confusing wall-time gap:

| Bucket | MADE1 | query_short | query_medium |
|--------|:---:|:---:|:---:|
| Search stages | 0.150s | 0.219s | 1.102s |
| GPU loop wall | 0.236s | 0.302s | 1.172s |
| Query elapsed | 0.237s | 0.303s | 1.174s |
| CPU workers | 0.062s | 0.123s | 0.756s |
| Domain workflow | 0.059s | 0.119s | 0.738s |
| nucdb reconstruct | 0.080s | 0.077s | 0.065s |
| Query outside search | 0.001s | 0.001s | 0.002s |

Shared setup/teardown for that process: CUDA engine create 0.446s, `.nucdb` open/mmap 0.000s, `.nucdb` upload 0.029s, CUDA destroy 0.003s, summed GPU loop wall 1.710s, process outside search 0.495s, process elapsed 2.205s.

The previous in-search mismatch was static GPU CPU-continuation scheduling. GPU workers were assigned contiguous equal-window slices, but domain-definition cost depends on region/envelope complexity. Dynamic survivor-window scheduling now lets workers pull work from a shared index and brings GPU-16 query_medium domain wall time (`0.239-0.258s`) in line with the CPU-16 wall-stage trace (`0.250s`). The remaining gap is mostly CUDA setup, `.nucdb` reconstruction, and GPU-side kernel/allocation variance rather than inflated CPU domain workflow.

## Stage Breakdown

CPU-4 HMMER stage totals are summed across worker threads, so divide by four for a rough per-thread wall comparison. GPU `NHMMER_GPU_INFO` buckets are wall buckets around sequential GPU stages plus max-across-worker CPU continuation buckets. The GPU column shows the focused repeat range after removing the extra single-score Viterbi prefilter.

| Stage | CPU-4 summed | CPU-4 approx wall | GPU fast `.nucdb` wall |
|-------|:---:|:---:|:---:|
| SSV | 2.343s | 0.586s | 0.103-0.109s |
| MSV/null/bias | 0.536s | 0.134s | 0.054-0.064s batch filter + 0.002s worker bias/null |
| Viterbi | 1.307s | 0.327s | 0.138-0.289s scanning Viterbi |
| Forward | 0.469s | 0.117s | 0.028-0.039s Forward prefilter |
| Backward | 0.278s | 0.070s | 0.011-0.014s GPU FB parser; 0.000s CPU Backward |
| Domain workflow | 1.992s | 0.498s | 0.782-0.799s |
| Output/null2 | 0.000s | ~0.000s | 0.002-0.004s hit reporting |
| Total measured wall | - | 1.77s focused | 1.53-1.86s focused repeats |

GPU device-side filtering is faster than CPU for SSV, Viterbi, and Forward/Backward. After dynamic scheduling, CPU domain workflow is no longer the unexplained mismatch against CPU-16; the remaining GPU-side cost is setup/reconstruction plus CUDA Viterbi/parser overhead.

Current launch occupancy from the instrumented query_medium fast overlap `.nucdb` run:

| Stage | Last launch | Theoretical occupancy | Grid/SM coverage |
|-------|-------------|:---:|:---:|
| SSV longtarget | grid=800, block=32, smem=1002B | 50.0% (24 active warps/SM of 48), 97.5% device-active in SSV wall | 6.25x |
| Scanning Viterbi | grid=1097, block=128, smem=24192B | 33.3% (16 active warps/SM of 48), 94.6% device-active in Viterbi CUDA call | 8.51x |

The GPU kernels are not starved by a single CUDA engine setup or too few blocks on chr22/query_medium; both launch enough blocks to cover all 128 SMs multiple times and their measured GPU-processing buckets are mostly device-active. SSV occupancy is capped by one warp per block, while scanning Viterbi occupancy is capped by dynamic shared memory. The current small GPU-side optimization keeps the Viterbi bias-score device buffer persistent on the CUDA engine, removing per-call `cudaMalloc`/`cudaFree`. Further optimization should focus on improving per-window DP efficiency/allocation reuse or reducing CPU domain workflow and `.nucdb` reconstruction cost, not on creating more CUDA engines.

## Timing Breakdown From Current GPU Run

```
GPU pipeline: vit_lt_in=8710 vit_lt_out=707 post_vit=707 post_fwd=341 hits=478
GPU timing breakdown (0.975s search stages; 1.024s GPU loop wall):
  SSV longtarget:      0.109s
    utilization:       97.5% device-active in SSV wall
  extend+merge:        0.002s
  batch filter:        0.054s
  scanning Viterbi:    0.132s
    utilization:       94.6% device-active in Viterbi CUDA call
  Forward prefilter:   0.025s
  GPU FB parser:       0.010s
  CPU workers:         0.643s
    null scoring:      0.000s
    bias scoring:      0.002s
    CPU Backward:      0.000s
    domain workflow:   0.633s
    hit reporting:     0.000s
```

`hits=478` in the diagnostic line is the internal `P7_TOPHITS` count before final output formatting. The comparable main-output hit-line count was 648 for both CPU-4 and GPU; strict `--tblout` rows were 215 for both with no diff.

## Reproducing

```sh
src/nhmmer --cpu 4 --noali \
    benchmark-data/nucleotide-bench/work/query_medium.hmm \
    benchmark-data/nucleotide-bench/work/chr22.fa

src/nhmmer --gpu --cpu 4 --noali \
    benchmark-data/nucleotide-bench/work/query_medium.hmm \
    benchmark-data/nucleotide-bench/work/chr22-overlap.nucdb.nucdb
```

Run commands sequentially to avoid GPU contention. For benchmark claims, report the query, target database, flags, HMMER elapsed, external wall time, output-row count, and GPU timing breakdown.
