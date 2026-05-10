# nhmmer GPU vs CPU Performance Breakdown

Last updated: 2026-05-10

## TL;DR

The previous “GPU is slower because CPU workers redo Forward/Backward and GPU domain rescoring is mutex-bound” analysis is stale. Current default `nhmmer --gpu` uses the GPU Forward prefilter and GPU Forward/Backward parser handoff by default, so CPU Forward/Backward continuation is gone in the accepted path.

The real issue behind the stale 1.91s versus 2.109s comparison was apples-to-oranges benchmarking plus a GPU scanning-Viterbi window merge bug. GPU Viterbi seeds are emitted with `atomicAdd`, so they were unsorted; `p7_pli_ExtendAndMergeWindows()` only merges adjacent windows. The GPU path also extended/merged seeds after converting them to absolute target coordinates, unlike CPU, which extends/merges in parent MSV-window-local coordinates. That inflated downstream CPU domain workflow. The current fix sorts seeds by `(window_id, position, model_k)`, extends/merges in local coordinates, then converts back to target coordinates.

## Current query_medium Result

Target: chr22/hg38 benchmark. Query: `query_medium.hmm` (M=501). Threads: `--cpu 4`. GPU: RTX 4090. Date: 2026-05-10.

| Path | Target | Output rows | HMMER elapsed | `/usr/bin/time` wall |
|------|--------|:---:|:---:|:---:|
| CPU-4 | `chr22.fa` | 648 | 1.87s | 1.89s |
| GPU default, no-overlap nucdb | `chr22.nucdb.nucdb` | 648 | 1.36s | 1.71s |
| GPU default, fast overlap nucdb | `chr22-overlap.nucdb.nucdb` | 648 | 1.40s | 1.75s |

The small ordinary-vs-overlap reversal in this single run is within run-to-run noise and workload shape; overlap `.nucdb` remains the default construction because it enables GPU-resident SSV and avoids per-sequence SSV H2D transfers when the query fits the overlap.

## Stage Breakdown

CPU-4 HMMER stage totals are summed across worker threads, so divide by four for a rough per-thread wall comparison. GPU `NHMMER_GPU_INFO` buckets are wall buckets around sequential GPU stages plus max-across-worker CPU continuation buckets.

| Stage | CPU-4 summed | CPU-4 approx wall | GPU fast `.nucdb` wall |
|-------|:---:|:---:|:---:|
| SSV | 2.597s | 0.649s | 0.107s |
| MSV/null/bias | 0.562s | 0.141s | 0.099s batch filter + 0.002s worker bias/null |
| Viterbi | 1.352s | 0.338s | 0.015s scanning Viterbi |
| Forward | 0.484s | 0.121s | 0.038s Forward prefilter |
| Backward | 0.290s | 0.072s | 0.013s GPU FB parser; 0.000s CPU Backward |
| Domain workflow | 2.101s | 0.525s | 1.013s |
| Output/null2 | 0.001s | ~0.000s | 0.005s |
| Total elapsed | - | 1.87s | 1.40s HMMER elapsed |

GPU device-side filtering is faster than CPU for SSV, Viterbi, and Forward/Backward. The dominant remaining GPU-side cost is not a CUDA kernel; it is CPU domain workflow after parser handoff.

## Timing Breakdown From Current GPU Run

```
GPU pipeline: vit_lt_in=874 vit_lt_out=874 post_vit=874 post_fwd=511 hits=718
GPU timing breakdown (1.297s total):
  SSV longtarget:      0.107s
  extend+merge:        0.001s
  batch filter:        0.099s
  scanning Viterbi:    0.015s
  Forward prefilter:   0.038s
  GPU FB parser:       0.013s
  CPU workers:         1.024s
    null scoring:      0.000s
    bias scoring:      0.002s
    CPU Backward:      0.000s
    domain workflow:   1.013s
    hit reporting:     0.005s
```

`hits=718` in the diagnostic line is the internal `P7_TOPHITS` count before final output formatting. The comparable output-row count was 648 for both CPU-4 and GPU.

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
