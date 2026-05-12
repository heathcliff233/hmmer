# nhmmer GPU Support - Progress

Last updated: 2026-05-11

This file records the current accepted `nhmmer --gpu` state. Older per-change
benchmark logs were intentionally removed; use git history for that level of
detail.

## Current Architecture

Default `nhmmer --gpu` is a nucleotide long-target pipeline:

1. GPU SSV long-target scan.
2. Host-side SSV-window extend/merge (was device `<<<1,1>>>` kernel).
3. GPU MSV/null/bias/F1 batch gate + host-side ordered compaction.
4. GPU scanning Viterbi.
5. Host-side Viterbi seed extend/merge/split (was device `<<<1,1>>>` kernel).
6. GPU Forward parser F3 gate.
7. GPU Backward parser handoff.
8. Threaded CPU domain definition, null2, hit storage, thresholding, and output.

The accepted path no longer has a separate single-score GPU Viterbi prefilter.
Scanning Viterbi is the CPU-equivalent Viterbi boundary. `--gpu-fwd-prefilter`
is retained as a deprecated no-op compatibility spelling; the GPU
Forward/Backward parser handoff is default-on. Use hidden
`--gpu-no-fwd-prefilter` only for diagnostics.

All serial `<<<1,1>>>` kernels (prefix sums, extend/merge, F1 compaction) have
been replaced with host-side code. The pattern is: D2H per-chunk/window counts
→ host prefix sum → H2D offsets → device compact kernel → D2H results → host
extend/merge. This eliminates kernel launch overhead and device synchronization
points between main compute kernels.

## Fast `.nucdb` Path

`hmmnucdb` builds a mmap-backed nucleotide database with pre-chunked forward and
reverse-complement strands. Its default overlap is `2001`, which is enough for
the current chr22 benchmark models.

The default resident overlap `.nucdb` path:

- Uploads `.nucdb` chunk data once per process.
- Runs SSV directly from resident device chunks.
- Maps F1 and parser windows back to resident chunks.
- Uses device gather for boundary-spanning windows.
- Keeps F1-to-Viterbi sequence bytes device-resident.
- Keeps Forward xmx device-resident through F3 gate, survivor offset creation,
  survivor compaction, and Backward.
- Avoids full host forward/reverse `ESL_SQ` reconstruction. The top-level
  pipeline gets metadata-only sequence shells, and CPU domain workers materialize
  only the survivor-window slice they need.

Diagnostic fallback paths that still read `sq->dsq` keep full host
reconstruction.

## Important Files

| File | Role |
|------|------|
| `src/nhmmer.c` | CLI, engine lifecycle, shared `.nucdb` open/upload, timing output |
| `src/nhmmer_internal.h` | `NHMMER_GPU_INFO` and GPU window/batch structs (public API) |
| `src/cuda/nhmmer_cuda_internal.h` | Cross-TU contract: worker/OMX structs, shared macros, internal decls |
| `src/nhmmer_gpu.c` | CPU-only stubs (~50 lines) for non-CUDA builds |
| `src/nhmmer_gpu_seqhelpers.c` | `.nucdb` ESL_SQ reconstruction (full / shell / slice) |
| `src/nhmmer_gpu_windows.c` | Windowlist helpers + synthetic-chunk lifecycle |
| `src/nhmmer_gpu_workers.c` | CPU worker pool, OMX special-state binding, scalar Viterbi debug |
| `src/cuda/p7_cuda_nhmmer_filters.c` | Scratch arenas, batch SSV/bias/F1 filter, nucdb resident mapping |
| `src/cuda/p7_cuda_nhmmer_viterbi.c` | Scanning Viterbi longtarget orchestration |
| `src/cuda/p7_cuda_nhmmer_fwd.c` | GPU Forward prefilter + Forward/Backward parser dispatch |
| `src/cuda/p7_cuda_nhmmer_strand.c` | Per-strand orchestration (FASTA and `.nucdb` paths) |
| `src/cuda/p7_cuda_nhmmer_search.c` | Public outer loops (`serial_loop`, `nucdb_loop`, `nucdb_upload`) |
| `src/cuda/p7_cuda_ssv_longtarget.cu` | SSV long-target kernels, host prefix sum + extend/merge |
| `src/cuda/p7_cuda_viterbi_longtarget.cu` | Scanning Viterbi kernels, host prefix sum + extend/merge/split |
| `src/cuda/p7_cuda_ssv.cu` | Fused SSV/null/bias/F1 gate kernel, host F1 compaction |
| `src/cuda/p7_cuda_fb_parser.cu` | GPU Forward/Backward parser handoff |
| `src/cuda/p7_cuda_runtime.cu` | CUDA engine, reset/destroy, `.nucdb` upload/release |
| `src/p7_nucdb.c`, `src/p7_nucdb.h` | `.nucdb` mmap format and helpers |
| `src/hmmnucdb.c` | `.nucdb` builder |

## CLI Reference

```sh
# Default GPU nhmmer
src/nhmmer --gpu --cpu 16 --noali query.hmm target.fa

# Fast resident .nucdb path
src/hmmnucdb target.fa target.nucdb
src/nhmmer --gpu --cpu 16 --noali query.hmm target.nucdb.nucdb

# No-overlap .nucdb diagnostic path
src/hmmnucdb --overlap 0 target.fa target-o0.nucdb
src/nhmmer --gpu --cpu 16 --noali query.hmm target-o0.nucdb.nucdb
```

Hidden diagnostics:

| Option | Meaning |
|--------|---------|
| `--gpu-batch` | GPU SSV/bias batch gate; default-on with `--gpu` |
| `--gpu-vit-longtarget` | GPU scanning Viterbi; default-on with `--gpu` |
| `--gpu-no-fwd-prefilter` | Disable default GPU Forward/Backward parser reuse |
| `--gpu-cpu-postmsv` | Bypass GPU scanning Viterbi/Fwd; CPU postSSV diagnostic |
| `--gpu-compare` | Compare GPU and CPU scores/stages |
| `--gpu-chunk-size N` | Runtime chunk size, default `65536` |
| `--gpu-device N` | CUDA device selection |

## Current Benchmarks

System: RTX 4090, 16 CPU threads unless stated otherwise.

Standard chr22 combined benchmark:

```sh
NHMMER_GPU_BENCH_CPU=16 test-speed/x-nhmmer-gpu-bench .
```

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 9.962s | 1476 |
| CPU-16 | 1.111s | 1476 |
| GPU-16 FASTA | 1.955s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.371s | 1476 |
| GPU-16 overlap `.nucdb` | 1.196s | 1476 |

Larger chr22x5 benchmark:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-16 FASTA | 4.893s | 6294 |
| GPU-16 overlap `.nucdb` | ~3.75s process / ~3.37s GPU loop (median of 3 runs) | 6294 |

The chr22x5 target is local benchmark data:

- `benchmark-data/nucleotide-bench/work/chr22x5.fa`
- `benchmark-data/nucleotide-bench/work/chr22x5-overlap.nucdb.nucdb`

## Current 5x GPU Timing Breakdown

Standalone GPU-16 overlap `.nucdb`, all three sample queries, chr22x5 (median
of 3 runs):

| Bucket | Time |
|--------|:---:|
| Internal process elapsed | ~3.75s |
| GPU loop wall | ~3.37s |
| Process outside search | ~0.38s |
| CUDA engine create | ~0.21s |
| Shared `.nucdb` upload | ~0.10s |

GPU loop stage totals across the three queries:

| Stage | Time |
|-------|:---:|
| SSV longtarget | ~0.90s |
| extend+merge | ~0.003s |
| batch filter | ~0.17s |
| scanning Viterbi | ~0.70s |
| GPU Forward/Backward parser | ~0.15s |
| CPU workers | ~1.65s |
| CPU domain workflow inside workers | ~1.53s |

Interpretation: full `.nucdb` host reconstruction is no longer a hidden
outside-search cost. Remaining wall time is visible search work plus shared CUDA
setup/upload and CPU domain/output workflow.

## Timing Semantics

GPU timing output has three layers:

- `CUDA *`: CUDA event timings for device-side sub-buckets.
- `GPU timing breakdown`: wall buckets around sequential GPU pipeline stages and
  max-across-worker CPU continuation buckets.
- `GPU outside-search timing` / `GPU process timing`: engine setup, shared
  `.nucdb` open/upload, per-query profile setup/cleanup, CUDA reset/destroy, and
  total process elapsed.

Some sub-buckets overlap by design, especially dispatch/wall buckets with CUDA
event buckets. Trust `process_elapsed`, `gpu_loop_wall`, and
`process_outside_search` for top-level accounting.

CPU threaded `Stage *` totals are summed across workers. Set
`HMMER_NHMMER_CPU_WALL_TRACE=1` to print max-across-worker wall-stage estimates
for CPU-only `nhmmer`.

## Verification

Current accepted verification:

```sh
git diff --check
make -C src -j2 nhmmer
NHMMER_GPU_BENCH_CPU=16 test-speed/x-nhmmer-gpu-parity .
```

Parity result: `4 passed, 0 failed`.

## Current Performance Conclusions

- GPU-16 is hit-count clean against CPU-16 on the smoke benchmark.
- On standard chr22, GPU-16 fast `.nucdb` is close to CPU-16 but not a stable
  large win.
- On chr22x5, GPU-16 fast `.nucdb` (~3.75s) is faster than CPU-16 (4.89s),
  roughly 23% speedup.
- CPU domain workflow remains a large bucket because domain definition, null2,
  hit storage, thresholding, and output remain CPU-side.
- SSV and scanning Viterbi have enough grid coverage; low theoretical occupancy
  alone is not the root cause. SSV is one warp per block, and scanning Viterbi
  uses four 8-lane nucleotide DP groups per physical warp.
- All serial `<<<1,1>>>` kernels have been eliminated; host-side prefix
  sum/extend/merge reduced SSV stage time by ~24%.
- Further work should focus on GPU kernel throughput, reducing parser matrix D2H
  or moving more domain workflow to GPU, and overlapping CPU domain work with
  later GPU strand/query work where correctness allows.
