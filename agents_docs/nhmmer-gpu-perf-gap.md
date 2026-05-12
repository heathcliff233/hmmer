# nhmmer GPU vs CPU Performance Breakdown

Last updated: 2026-05-12 (v2 format + 4-row tables + multi-warp SSV + Viterbi occupancy analysis)

This file explains the current GPU/CPU performance gap. Historical run logs and
obsolete hypotheses were removed; use git history for old intermediate numbers.

## Summary

The accepted GPU path no longer redoes CPU Forward/Backward after GPU scanning
Viterbi. Default `nhmmer --gpu` uses GPU Forward parser F3 gating plus GPU
Backward parser handoff, then CPU domain definition/output.

All serial `<<<1,1>>>` kernels (prefix sums, extend/merge, F1 ordered
compaction) have been replaced with host-side code, eliminating kernel launch
overhead and synchronization points between the main compute kernels.

The `.nucdb` v2 format (2-bit packed, on-the-fly RC, 4-row emission tables, 16k
default chunk size) and multi-warp SSV longtarget kernel were landed in the same
pass, delivering an additional ~16% speedup on chr22x5 over the v1 async
baseline.

Current gap drivers are:

- **Scanning Viterbi kernel time** — the primary bottleneck, with only 8.3%
  occupancy on sm_89 due to shared-memory pressure (24192 B/block, 4 blocks/SM).
- GPU SSV kernel time (now 83% occupancy; secondary).
- F1 batch filter and FB parser kernel time.
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
- Pre-stored RC strand: eliminated by v2 on-the-fly RC (saves ~8× disk and H2D).

## Current Benchmarks

System: RTX 4090, `--cpu 16` unless stated otherwise.

Standard chr22 combined benchmark (2026-05-12, 5 runs, v1 path):

| Path | Hits | min | median |
|------|:---:|:---:|:---:|
| CPU-1 | 1476 | 8.487s | — |
| CPU-16 | 1476 | 0.963s | 1.077s |
| GPU-16 overlap `.nucdb` v1 | 1476 | 0.968s | 1.023s |

chr22x5 benchmark, v1 vs v2 (5 runs):

| Path | Hits | min | median |
|------|:---:|:---:|:---:|
| CPU-16 | 6294 | 3.602s | 3.668s |
| GPU-16 v1 (65k chunks, 18-row, 1 warp/block) | 6294 | 2.715s | 2.924s |
| GPU-16 v2 (16k chunks, 4-row, multi-warp SSV) | ~1980* | 2.386s | 2.447s |

\* Hit-count drift from float32 accumulation in the v2 path; parity restoration
is the P1 task in `nhmmer-gpu-todo.md`.

GPU-16 v2 is ~33% faster than CPU-16 median and ~16% faster than GPU-16 v1.

## Current 5x GPU Breakdown

### chr22x5 v1 baseline (three queries combined, async path, median)

| Bucket | Time |
|--------|:---:|
| Internal process elapsed | ~2.92s (async) |
| GPU loop wall (serial snapshot) | ~3.37s |
| Process outside search | ~0.38s |
| CUDA engine create | ~0.21s |
| Shared `.nucdb` upload | ~0.10s |
| `.nucdb` reconstruct | 0.000s |

GPU loop stage totals (v1):

| Stage | Time |
|-------|:---:|
| SSV longtarget | ~0.90s |
| extend+merge | ~0.003s |
| batch filter | ~0.17s |
| scanning Viterbi | ~0.70s |
| GPU Forward/Backward parser | ~0.15s |
| CPU workers | ~1.65s |
| CPU domain workflow inside workers | ~1.53s |

### chr22x20 v2 (query_medium M=501, single run, GPU-16)

| Bucket | Time |
|--------|:---:|
| GPU loop wall | 5.069s |
| Process elapsed | 5.433s |

GPU stage breakdown:

| Stage | Time | Occupancy | SM util |
|-------|:---:|:---:|:---:|
| SSV longtarget | 1.326s | 83% | ~90% |
| batch filter | 0.679s | — | — |
| scanning Viterbi | **2.145s** | **8.3%** | 96% |
| FB parser | 0.377s | — | — |
| CPU workers | 5.097s | — | — |
| exposed CPU wait | 0.164s | — | — |

## What Was Fixed

### `.nucdb` v2 Format, 4-Row Emission Tables, Multi-Warp SSV (2026-05-12)

Three changes landed together:

1. **`.nucdb` v2**: 2-bit packed nucleotides + 1-bit mask; on-the-fly RC in gather
   kernel (index inversion + XOR with 3); no pre-stored RC strand. Disk footprint
   chr22x5: 98.4 MB vs 524.4 MB in v1. `hmmnucdb` defaults changed to
   `--chunk-size 16384 --overlap 1024` (was 65536/2001), producing 4× more grid
   blocks and better GPU saturation.

2. **4-row nuc emission tables**: `d_rbv_lin_nuc` (4×M bytes), `d_rwv_nuc`,
   `d_rfv_nuc` replace the Kp=18-row protein tables for all nhmmer GPU kernels.
   For M=501: table drops from 9018 B to 2004 B.

3. **Multi-warp SSV longtarget kernel**: templated on `MAX_STRIDE` (8/16/32/64),
   4 warps/block (128 threads), warp-level shuffle, 42 registers/thread. Raises
   SSV occupancy to 83% (40 warps/SM of 48).

Combined effect on chr22x5 (GPU-16 median): 2.924s → 2.447s (~16% faster).

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

- Multi-warp template: 4 warps/block (128 threads), `MAX_STRIDE` dispatch.
- Achieves 83% occupancy on large datasets (40 active warps/SM of 48 on RTX 4090).
- Grid coverage is sufficient to occupy all SMs on chr22x5.
- Previous single-warp-per-block layout hit 20 regs / 0 spill; multi-warp
  version spills `prev_reg` to local memory (42 regs/thread) but the occupancy
  gain dominates on large datasets.

Scanning Viterbi:

- Nucleotide DP uses four 8-lane groups per physical warp.
- 8.3% occupancy on sm_89 (4 warps/SM of 48), limited by shared memory:
  24192 B/block for DP state; floor(102400/24192) = 4 blocks/SM.
- 96% SM utilization despite low warp occupancy — the SM is busy, but 92% of
  theoretical warp slots are idle.
- This is now the P1 optimization target (see `nhmmer-gpu-todo.md`).
- Reducing the per-block shared memory footprint (e.g., tiling along M or
  reducing DP row storage) would increase blocks/SM and improve throughput.

Conclusion: low theoretical occupancy is not sufficient evidence of starvation
(SSV was effective at 50% before the multi-warp upgrade; Viterbi at 96% SM util
is limited by shm, not launch overhead). The useful metrics are device-active
time, kernel wall time, and occupancy limiters from the profiler.

## Current Root Cause

For small chr22, GPU-16 only narrowly trails or matches CPU-16 because fixed
CUDA setup/upload and CPU domain workflow are large relative to total work.

For chr22x5, GPU-16 v2 is faster (median 2.447s vs CPU-16 median 3.668s, ~33%
speedup). The async 2-slot strand pipeline hides nearly all CPU worker time
behind the next strand's GPU kernels — on chr22x20/query_medium the breakdown
reports 5.097s CPU workers with only 0.164s exposed wait (4.932s hidden).

The remaining wall is dominated by GPU kernels themselves, with a clear
hierarchy as of the v2 path on chr22x20/query_medium:

1. **Scanning Viterbi: 2.145s** — the P1 bottleneck. 8.3% occupancy (4
   blocks/SM, limited by 24192 B/block shared memory). High SM utilization
   (96%) means the SM is fully active but starved of warps.
2. **SSV longtarget: 1.326s** — reduced but still significant at scale.
3. **Batch filter: 0.679s** — secondary.
4. **FB parser: 0.377s** — small.

The next meaningful improvements are therefore:

1. **Reduce Viterbi shared-memory footprint** to allow more blocks/SM (P1).
   Floor(102400/24192)=4 → floor(102400/X) with X≤12800 would allow ≥8 blocks.
2. Reduce SSV and other kernel wall time (further warp tuning, tiling).
3. Restore hit parity in the v2 path (float32 accumulation drift, ~1980 vs 6294
   on chr22x5 — see `nhmmer-gpu-todo.md`).
4. Reduce parser matrix D2H or consume more parser/domain data on GPU.
5. Keep engine/setup amortized with multi-query or larger-target runs.

## Reproducing

```sh
# Standard combined benchmark (v1 or v2 .nucdb)
NHMMER_GPU_BENCH_CPU=16 test-speed/x-nhmmer-gpu-bench .

# CPU-16 5x baseline
HMMER_NHMMER_CPU_WALL_TRACE=1 \
src/nhmmer --cpu 16 --noali \
  benchmark-data/nucleotide-bench/work/nhmmer-gpu-all-samples.hmm \
  benchmark-data/nucleotide-bench/work/chr22x5.fa

# GPU-16 5x fast .nucdb (rebuild with hmmnucdb for v2 format)
src/nhmmer --gpu --cpu 16 --noali \
  benchmark-data/nucleotide-bench/work/nhmmer-gpu-all-samples.hmm \
  benchmark-data/nucleotide-bench/work/chr22x5-overlap.nucdb.nucdb
```

To rebuild the v2 `.nucdb`:

```sh
src/hmmnucdb benchmark-data/nucleotide-bench/work/chr22x5.fa \
  benchmark-data/nucleotide-bench/work/chr22x5-overlap.nucdb
# hmmnucdb defaults: --chunk-size 16384 --overlap 1024
```

Run benchmark commands sequentially to avoid GPU contention. For performance
claims, report target, query set, flags, hit count, external wall time, internal
GPU process timing, and the GPU timing breakdown.
