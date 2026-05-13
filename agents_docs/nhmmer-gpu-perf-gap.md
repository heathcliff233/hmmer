# nhmmer GPU vs CPU Performance Breakdown

Last updated: 2026-05-13 (register-based scanning Viterbi; fill_slice still present)

This file explains the current GPU/CPU performance gap. Historical run logs and
obsolete hypotheses were removed; use git history for old intermediate numbers.

## Summary

The accepted GPU path uses nuc-specific Forward/Backward engines (`fwdback_nuc.c`)
that read directly from mmap'd 2-bit packed data via `P7_NUCSEQVIEW`, plus a
nuc-optimized emission score rebuild that skips degenerate code computation.

Current gap drivers are:

- **CPU Forward/Backward Q-loop** — the single largest time consumer. Q=126
  SSE iterations per residue (M=501), called 4–5× per domain envelope. AVX2
  (Q=63) would halve this.
- **`fill_slice` decode** — still called per window for `p7_alidisplay_Create`.
  Costs ~1s/query on chr22x20. Planned for elimination.
- **Scanning Viterbi kernel time** — 1.21s on chr22x20/query_medium (reduced
  from 2.02s by register-based kernel, still the dominant GPU stage).
- GPU SSV kernel time (~1.3s, 83% occupancy; secondary).
- F1 batch filter and FB parser kernel time.

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

System: RTX 4090, `--cpu 16` unless stated otherwise. 2026-05-13.

### chr22x20 (~1GB, 20 chromosomes)

Per-query:

| Query | M | GPU-16 | CPU-16 | Speedup |
|-------|---|--------|--------|---------|
| MADE1 | 80 | 1.44s | 2.91s | 2.0× |
| query_short | 151 | 1.67s | 2.81s | 1.7× |
| query_medium | 501 | 5.14s | 8.95s | 1.7× |

Multi-query (12 queries = 3 models × 4 copies):

| Path | Wall time (median of 3) | Speedup |
|------|:-----------------------:|---------|
| GPU-16 | **25.8s** | **2.3×** |
| CPU-16 | 59.8s | baseline |

### chr22x5 (~250MB, 5 chromosomes)

| Query | M | GPU-16 | CPU-16 | Speedup |
|-------|---|--------|--------|---------|
| MADE1 | 80 | 0.79s | 1.04s | 1.3× |
| query_short | 151 | 0.68s | 0.81s | 1.2× |
| query_medium | 501 | 1.90s | 2.74s | 1.4× |

## Current GPU Breakdown (chr22x20, query_medium M=501, GPU-16)

| Bucket | Time |
|--------|:---:|
| GPU loop wall | ~4.3s |
| Process elapsed | ~5.1s |

GPU stage breakdown:

| Stage | Time | Notes |
|-------|:---:|-------|
| SSV longtarget | ~1.3s | 83% occupancy, multi-warp |
| batch filter | ~0.7s | |
| scanning Viterbi | **1.21s** | register kernel (was 2.02s with shmem) |
| FB parser | ~0.4s | |
| CPU workers | ~4.5s | |
| exposed CPU wait | ~0.2s | 4.3s hidden behind GPU |

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

- **Register kernel** (M≥256): STRIDE=24 template, 4 warps/block, 184 regs/thread,
  8 warps/SM (16.7% occupancy). DP state held in register arrays; shared memory
  reduced to 2*M bytes (lookup tables only). Achieves 1.67× speedup over shmem
  kernel for M=501.
- **Shared-memory kernel** (M<256 or M>768): nucleotide DP with four 8-lane groups
  per physical warp. 8.3% occupancy (4 blocks/SM, limited by 24192 B/block shmem).
- STRIDE=16 template (33% occupancy) blocked by nvcc speculative-read bug.

Conclusion: low theoretical occupancy is not sufficient evidence of starvation
(SSV was effective at 50% before the multi-warp upgrade; Viterbi at 96% SM util
is limited by shm, not launch overhead). The useful metrics are device-active
time, kernel wall time, and occupancy limiters from the profiler.

## Current Root Cause

For small chr22, GPU-16 only narrowly trails or matches CPU-16 because fixed
CUDA setup/upload and CPU domain workflow are large relative to total work.

For larger databases, the GPU advantage grows with dataset size because the async
2-slot pipeline keeps both slots busy through the run:
- chr22x5: 1.2–1.4× per query
- chr22x20: 1.7–2.0× per query, **2.3× for 12-query batch**

The remaining wall is dominated by GPU kernels themselves, with current hierarchy
on chr22x20/query_medium (register kernel active):

1. **SSV longtarget: ~1.3s** — the largest GPU stage at this scale.
2. **Scanning Viterbi: 1.21s** — reduced from 2.02s by register kernel (1.67×).
3. **Batch filter: ~0.7s** — secondary.
4. **FB parser: ~0.4s** — small.

CPU worker time (~4.5s) is almost fully hidden behind the next strand's GPU work
(only ~0.2s exposed wait).

The next meaningful improvements are:

1. **AVX2 Forward/Backward** (P0): Q=126→63 would halve CPU domain F/B time,
   further reducing the remaining exposed CPU wait.
2. **Eliminate `fill_slice`** (P1): saves ~1s/query in CPU worker decode overhead.
3. **STRIDE=16 register kernel** (P2): would raise Viterbi occupancy from 16.7%
   to 33% if the nvcc speculative-read bug can be worked around.
4. Keep engine/setup amortized with multi-query or larger-target runs.

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
