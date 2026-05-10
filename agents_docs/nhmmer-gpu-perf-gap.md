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

GPU device-side filtering is faster than CPU for SSV, Viterbi, and Forward/Backward. The dominant remaining GPU-side cost is not a CUDA kernel; it is CPU domain workflow after parser handoff.

## Timing Breakdown From Current GPU Run

```
GPU pipeline: vit_lt_in=8710 vit_lt_out=707 post_vit=707 post_fwd=341 hits=478
GPU timing breakdown repeat range (1.141-1.319s total):
  SSV longtarget:      0.103-0.109s
  extend+merge:        0.003s
  batch filter:        0.054-0.064s
  scanning Viterbi:    0.138-0.289s
  Forward prefilter:   0.028-0.039s
  GPU FB parser:       0.011-0.014s
  CPU workers:         0.801-0.812s
    null scoring:      0.000s
    bias scoring:      0.002s
    CPU Backward:      0.000s
    domain workflow:   0.782-0.799s
    hit reporting:     0.002-0.004s
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
