# nhmmer GPU vs CPU Performance Breakdown

Last updated: 2026-05-12

This file explains the current GPU/CPU performance gap. Historical run logs and
obsolete hypotheses were removed; use git history for old intermediate numbers.

## 2026-05-12 P1 attempt: GPU domcorrection Forward (no wall-time win)

The 2nd-pass `p7_Forward` inside `rescore_isolated_domain` was moved to a
batched GPU call after `pthread_join`. Parity exact (1476/6294 hits). Wall
time unchanged within noise:

| Path (chr22x5 query_medium, 16 threads, 5-run median) | gpu_loop_wall | domain workflow | GPU domcorr |
|---|:---:|:---:|:---:|
| GPU domcorr on (P1 default) | 2.443s | 1.070s | ~0.007s (2405 envs / 10 launches) |
| GPU domcorr off (`--gpu-cpu-domcorr`) | 2.431s | 1.066s | 0s |

Why P1 alone doesn't help: the 2nd Forward runs on 50–200 residue envelope
windows across 16 threads in parallel, is already SSE-vectorized, and each
call is ~50–100 µs. Deferring it to GPU saves only ~15 ms per worker, of
which the per-launch overhead + D2H patch consumes most. The remaining CPU
`rescore_isolated_domain` work (1st Forward, Backward, Decoding,
OptimalAccuracy, OATrace, alidisplay) is the dominant cost and still runs
on CPU.

## Summary

The accepted GPU path no longer redoes CPU Forward/Backward after GPU scanning
Viterbi. Default `nhmmer --gpu` uses GPU Forward parser F3 gating plus GPU
Backward parser handoff, then CPU domain definition/output.

All serial `<<<1,1>>>` kernels (prefix sums, extend/merge, F1 ordered
compaction) have been replaced with host-side code, eliminating kernel launch
overhead and synchronization points between the main compute kernels.

Current gap drivers are:

- GPU SSV, F1, scanning Viterbi, and parser kernel time.
- Parser matrix D2H needed by the CPU domain workflow.
- CPU domain definition/output after the GPU parser handoff.
- Shared CUDA setup and `.nucdb` upload, especially on small workloads.

Recent non-causes:

- Serial `<<<1,1>>>` kernels are gone — prefix sums, extend/merge, and F1
  ordered compaction all run on host now.
- Full `.nucdb` host strand reconstruction on the default resident overlap path
  has been removed.
- CPU Forward/Backward continuation is not in the accepted default path.
- SSV/Viterbi seed sorting and extend/merge CPU islands are no longer in the
  default path.
- Batch/parser sequence H2D is `0 bytes` on the fast overlap `.nucdb` path.

## Current Benchmarks

System: RTX 4090, `--cpu 16` unless stated otherwise.

Standard chr22 combined benchmark:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 9.962s | 1476 |
| CPU-16 | 1.111s | 1476 |
| GPU-16 FASTA | 1.955s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.371s | 1476 |
| GPU-16 overlap `.nucdb` | 1.196s | 1476 |

Larger chr22x5 benchmark (median of 3 runs):

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-16 FASTA | 4.893s | 6294 |
| GPU-16 overlap `.nucdb` | ~3.75s process / ~3.37s GPU loop | 6294 |

## Current 5x GPU Breakdown

Standalone GPU-16 overlap `.nucdb`, all three sample queries, chr22x5:

| Bucket | Time |
|--------|:---:|
| Internal process elapsed | ~3.75s |
| GPU loop wall | ~3.37s |
| Process outside search | ~0.38s |
| CUDA engine create | ~0.21s |
| Shared `.nucdb` upload | ~0.10s |
| `.nucdb` reconstruct | 0.000s |

GPU loop stage totals:

| Stage | Time |
|-------|:---:|
| SSV longtarget | ~0.90s |
| extend+merge | ~0.003s |
| batch filter | ~0.17s |
| scanning Viterbi | ~0.70s |
| GPU Forward/Backward parser | ~0.15s |
| CPU workers | ~1.65s |
| CPU domain workflow inside workers | ~1.53s |

## What Was Fixed

### Serial <<<1,1>>> Kernel Elimination

Five serial single-thread CUDA kernels have been replaced with host-side code:

| Original kernel | Path | Replacement |
|---|---|---|
| `cuda_ssv_longtarget_prefix_kernel<<<1,1>>>` | SSV | Host prefix sum |
| `cuda_ssv_windows_extend_merge_kernel<<<1,1>>>` | SSV | Host extend/merge |
| `cuda_viterbi_longtarget_prefix_kernel<<<1,1>>>` | Viterbi | Host prefix sum |
| `cuda_viterbi_windows_extend_merge_kernel<<<1,1>>>` | Viterbi | Host extend/merge/split |
| `cuda_f1_compact_ordered_kernel<<<1,1>>>` | F1 batch | Host compaction with D2H mask/filtersc |

The pattern for each: D2H per-chunk/window counts → host prefix sum →
H2D offsets → device compact kernel → D2H compact results → host extend/merge.
For F1, the compaction is entirely host-side: D2H pass mask + filter scores →
host compact loop → H2D compacted survivors back to device (needed by downstream
Viterbi threshold kernel).

Measured effect on chr22x5 (SSV stage sum):

| Metric | Before | After |
|--------|:---:|:---:|
| SSV longtarget stage | 1.182s | ~0.90s |
| GPU loop wall | 3.801s | ~3.37s |

### Full `.nucdb` Host Reconstruction

Before the latest fix, the default fast `.nucdb` GPU path still reconstructed
full forward and reverse-complement host `ESL_SQ`/`dsq` objects from mapped
`.nucdb` chunks. That was outside the GPU search loop and redundant because the
accepted path already reads SSV, F1, Viterbi, and parser sequence bytes from the
resident device `.nucdb` buffer.

Current behavior:

- Top-level `.nucdb` loop creates metadata-only `ESL_SQ` shells.
- CPU domain workers materialize only survivor-window slices.
- Diagnostic fallback paths that still dereference `sq->dsq` keep the old full
  reconstruction.

### CPU Domain Scheduling

GPU CPU-continuation workers use dynamic survivor-window scheduling. This fixed
the previous imbalance where static equal-window slices assigned expensive domain
regions unevenly. Current remaining CPU-worker cost is real domain/output work,
not mostly thread imbalance.

### Sequence Transfers

The fast overlap `.nucdb` path no longer packs/uploads sequence bytes for F1 or
parser windows:

- F1 windows map back to resident chunks or use device gather for boundaries.
- Scanning Viterbi consumes F1 resident metadata directly.
- Parser windows use resident device sequence bytes.
- Forward xmx stays on device through F3 survivor selection and Backward launch.

## Kernel Utilization Notes

SSV:

- One warp per chunk/block.
- Theoretical occupancy is capped around 50% on the RTX 4090 setup.
- Grid coverage is enough to occupy all SMs on chr22/query_medium and chr22x5.
- Raising theoretical occupancy with more warps per block did not improve wall
  time — register spill from multi-warp `prev_reg[64]` sharing outweighs the
  occupancy gain. The compiler achieves 20 regs / 0 spill in the single-warp
  layout vs 36 regs / 64B spill with multi-warp.

Scanning Viterbi:

- Nucleotide DP uses four 8-lane groups per physical warp.
- This lowers physical-warp occupancy counters but improves useful lane usage.
- The focused scan kernel improved versus the older one-8-lane-group mapping,
  but Viterbi remains a visible bucket on larger targets.

Conclusion: low theoretical occupancy is not sufficient evidence of starvation.
The useful metrics are device-active time, kernel wall time, and whether the
stage remains a large end-to-end bucket.

## Current Root Cause

For small chr22, GPU-16 only narrowly trails or matches CPU-16 because fixed
CUDA setup/upload and CPU domain workflow are large relative to total work.

For chr22x5, GPU-16 is faster (~3.75s vs ~4.89s, ~23% speedup), but the
speedup is limited because nearly half of the GPU loop is still CPU
worker/domain workflow:

- CPU workers: ~1.65s of ~3.37s GPU loop wall.
- CPU domain workflow: ~1.53s.
- SSV + scanning Viterbi together: ~1.60s.

The next meaningful improvements are therefore:

1. Reduce SSV and scanning Viterbi kernel wall time.
2. Reduce parser matrix D2H or consume more parser/domain data on GPU.
3. Move more domain definition/null2/output work to GPU, or overlap CPU domain
   work with later GPU work.
4. Keep engine/setup amortized with multi-query or larger-target runs.

## Reproducing

```sh
# Standard combined benchmark
NHMMER_GPU_BENCH_CPU=16 test-speed/x-nhmmer-gpu-bench .

# CPU-16 5x baseline
HMMER_NHMMER_CPU_WALL_TRACE=1 \
src/nhmmer --cpu 16 --noali \
  benchmark-data/nucleotide-bench/work/nhmmer-gpu-all-samples.hmm \
  benchmark-data/nucleotide-bench/work/chr22x5.fa

# GPU-16 5x fast .nucdb
src/nhmmer --gpu --cpu 16 --noali \
  benchmark-data/nucleotide-bench/work/nhmmer-gpu-all-samples.hmm \
  benchmark-data/nucleotide-bench/work/chr22x5-overlap.nucdb.nucdb
```

Run benchmark commands sequentially to avoid GPU contention. For performance
claims, report target, query set, flags, hit count, external wall time, internal
GPU process timing, and the GPU timing breakdown.
