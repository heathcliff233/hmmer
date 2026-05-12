# nhmmer GPU Support - Progress

Last updated: 2026-05-12 (v2 format + 4-row tables + multi-warp SSV)

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
Scanning Viterbi is the CPU-equivalent Viterbi boundary. GPU Forward/Backward
parser handoff is default-on and unconditional; the previous `--gpu-fwd-prefilter`
/ `--gpu-no-fwd-prefilter` diagnostic flags were removed along with all
non-default branches in the 2026-05-12 async-pass cleanup.

All serial `<<<1,1>>>` kernels (prefix sums, extend/merge, F1 compaction) have
been replaced with host-side code. The pattern is: D2H per-chunk/window counts
→ host prefix sum → H2D offsets → device compact kernel → D2H results → host
extend/merge. This eliminates kernel launch overhead and device synchronization
points between main compute kernels.

## Async 2-slot Strand Pipeline (landed 2026-05-12)

The outer `.nucdb` loop is a 2-slot double buffer: the main thread drives
strand N+1's GPU pipeline while strand N's CPU domain workers run in background
pthreads. A single `P7_CUDA_ENGINE` still serializes all GPU stages (only one
slot on the device at a time), but CPU worker time — previously dead time on
the GPU — is now hidden behind the next strand's GPU kernels. On
chr22x5/query_medium the breakdown reports `CPU workers 1.762s`,
`main-thread wait 0.144s`, `overlap saved 1.618s`. New timing fields:
`info->t_worker_wait` (wall spent in `pthread_join`) and
`info->t_overlap_saved`.

See `src/cuda/nhmmer_gpu_slot.{c,h}` for slot lifecycle, and
`src/cuda/p7_cuda_nhmmer_search.c` for the 2-slot ring driver.

## `.nucdb` v2 Format (landed)

`hmmnucdb` v2 packs nucleotides at 2 bits/residue (A=0, C=1, G=2, T=3) plus a
1-bit/residue mask for non-ACGT ambiguity characters.  Reverse complement is
computed on-the-fly by the GPU gather kernel (index inversion + XOR with 3) —
no pre-stored RC strand is written to disk or uploaded to the device.

Disk layout:

- 3 bits per input residue (2-bit packed + 1-bit mask).
- chr22x5: 98.4 MB on disk, down from 524.4 MB in v1 (forward + RC, 8 bits each).

Device payload:

- 2 bits/residue (packed forward strand only; RC computed at kernel launch).
- Reduces H2D upload by ~8× vs v1.

`hmmnucdb` new defaults: `--chunk-size 16384 --overlap 1024` (was 65536/2001).
The 4× smaller chunks produce 4× more blocks in the grid, improving GPU
saturation for all kernel stages.  The runtime now reads chunk size and overlap
directly from the `.nucdb` header, with no hardcoded 65536 check.

The overlap requirement is `overlap >= om->max_length`; the `hmmnucdb` default
of 1024 satisfies all current benchmark models (max_length ≤ 1024 for M ≤ 501).
Larger models may need `--overlap` set explicitly.

## 4-Row Nucleotide Emission Tables (landed)

GPU kernels for SSV longtarget, batch filter, scanning Viterbi, and the FB
parser now use nuc-specific emission tables with 4 rows (one per canonical
residue A/C/G/T) instead of the full Kp=18-row protein table:

| Table | Layout | Size (M=501) |
|-------|--------|:---:|
| `d_rbv_lin_nuc` | 4×M bytes | 2004 B (was 9018 B) |
| `d_rwv_nuc` | 4×Qw×8×int16 | proportional |
| `d_rfv_nuc` | 4×Qf×4×float | proportional |

The gather kernel unpacks 2-bit residue codes (0–3) directly into the 4-row
table index.  All kernel launch sites pass the nuc tables when available (i.e.,
when the profile alphabet is DNA/RNA and a v2 `.nucdb` is in use).

Relevant files: `src/cuda/p7_cuda_internal.h` (struct fields), 
`src/cuda/p7_cuda_runtime.cu` (`CreateNucTables`), `src/cuda/p7_cuda.h`
(declaration).

## Multi-Warp SSV Longtarget Kernel (landed)

The SSV longtarget kernel is now templated on `MAX_STRIDE` (8/16/32/64) and
uses warp-level shuffle intrinsics for inter-lane communication, replacing the
previous single-warp-per-block layout.  Key parameters:

- `SSV_LT_WARPS_PER_BLOCK = 4` (128 threads/block).
- Each warp processes one chunk independently.
- Dispatch selects `MAX_STRIDE` at runtime based on `om->M`.
- Compiler reports 42 registers/thread; minor spills to local memory stack for
  `prev_reg` (which is now shared across 4 warps).
- Measured occupancy: 83% on large datasets (40 active warps/SM of 48 on RTX
  4090 sm_89).

## Fast `.nucdb` Path

The default resident overlap `.nucdb` path:

- Uploads `.nucdb` v2 chunk data once per process (forward strand only, 2 bits/residue).
- Computes RC on-the-fly in the gather kernel; no RC storage on disk or device.
- Runs SSV directly from resident device chunks using `d_rbv_lin_nuc` (4-row).
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
| `src/cuda/p7_cuda_nhmmer_strand.c` | Per-strand GPU phase (`nhmmer_gpu_run_strand_gpu_phase`), slot-owned outputs |
| `src/cuda/p7_cuda_nhmmer_search.c` | Outer `.nucdb` loop, 2-slot ring driver (`submit_strand`, `nhmmer_gpu_nucdb_loop`) |
| `src/cuda/nhmmer_gpu_slot.c`, `nhmmer_cuda_internal.h` | Per-slot state, init/retire/launch-workers, pthread pool |
| `src/cuda/p7_cuda_ssv_longtarget.cu` | SSV long-target kernels, host prefix sum + extend/merge |
| `src/cuda/p7_cuda_viterbi_longtarget.cu` | Scanning Viterbi kernels, host prefix sum + extend/merge/split |
| `src/cuda/p7_cuda_ssv.cu` | Fused SSV/null/bias/F1 gate kernel, host F1 compaction |
| `src/cuda/p7_cuda_fb_parser.cu` | GPU Forward/Backward parser handoff |
| `src/cuda/p7_cuda_runtime.cu` | CUDA engine, reset/destroy, `.nucdb` upload/release |
| `src/p7_nucdb.c`, `src/p7_nucdb.h` | `.nucdb` mmap format and helpers |
| `src/hmmnucdb.c` | `.nucdb` builder |

## CLI Reference

`nhmmer --gpu` requires an overlap `.nucdb` target (v2 format recommended;
v1 is still accepted but lacks the 4-row nuc tables and multi-warp SSV). FASTA
and no-overlap `.nucdb` are rejected with a diagnostic (no CPU fallback inside
the GPU path). `hmmnucdb` defaults to `--chunk-size 16384 --overlap 1024`.

```sh
src/hmmnucdb target.fa target.nucdb
src/nhmmer --gpu --cpu 16 --noali query.hmm target.nucdb.nucdb
```

Hidden diagnostics (only one remains after the 2026-05-12 cleanup):

| Option | Meaning |
|--------|---------|
| `--gpu-device N` | CUDA device selection |

## Current Benchmarks

System: RTX 4090, 16 CPU threads. Numbers from `test-speed/x-nhmmer-gpu-bench`
(process-lifetime wall via `date +%s%N`), 5 runs each, 2026-05-12.

Standard chr22 combined benchmark (MADE1 + query_short + query_medium), v1
baseline:

| Path | Hits | min | median |
|------|:---:|:---:|:---:|
| CPU-1 | 1476 | 8.487s | — |
| CPU-16 | 1476 | 0.963s | 1.077s |
| GPU-16 overlap `.nucdb` v1 | 1476 | 0.968s | 1.023s |

chr22x5 benchmark, v1 vs v2 comparison:

| Path | Hits | min | median |
|------|:---:|:---:|:---:|
| CPU-16 | 6294 | 3.602s | 3.668s |
| GPU-16 v1 (65k chunks, 18-row) | 6294 | 2.715s | 2.924s |
| GPU-16 v2 (16k chunks, 4-row, multi-warp) | ~1980* | 2.386s | 2.447s |

\* ~1980 hits reflects float32 accumulation drift introduced in the v2 path;
the Viterbi occupancy fix and hit-parity restoration are in progress (see
`nhmmer-gpu-todo.md`).

GPU-16 v2 is ~33% faster than CPU-16 and ~16% faster than GPU-16 v1.

The chr22x5 target is local benchmark data:

- `benchmark-data/nucleotide-bench/work/chr22x5.fa`
- `benchmark-data/nucleotide-bench/work/chr22x5-overlap.nucdb.nucdb`

## Current 5x GPU Timing Breakdown (v2 path, chr22x20, query_medium M=501)

Single-run, GPU-16, `nhmmer --gpu --cpu 16`, chr22x20 (20× chr22), query_medium:

| Bucket | Time |
|--------|:---:|
| GPU loop wall | 5.069s |
| Process elapsed | 5.433s |

GPU stage breakdown:

| Stage | Time | Notes |
|-------|:---:|-------|
| SSV longtarget | 1.326s | 83% occupancy, ~90% SM utilization |
| batch filter | 0.679s | |
| scanning Viterbi | 2.145s | **bottleneck**: 8.3% occupancy, 96% SM utilization |
| FB parser | 0.377s | |
| CPU workers | 5.097s | only 0.164s exposed wait; 4.932s hidden behind GPU |

Scanning Viterbi occupancy is limited to 8.3% (4 warps/SM of 48) by shared
memory: 24192 B/block for DP state, floor(102400/24192) = 4 blocks/SM. See
`nhmmer-gpu-todo.md` for the planned occupancy optimization.

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

- GPU-16 v2 is ~33% faster than CPU-16 on chr22x5 (median 2.447s vs 3.668s).
  GPU-16 v1 was ~27% faster; v2 gains ~16% additional speedup from 4× smaller
  chunks, 4-row emission tables, and multi-warp SSV.
- On standard chr22, GPU-16 and CPU-16 remain a tie (~1.0s median both); too
  few strands to fill both async slots regardless of v2 improvements.
- The async 2-slot strand pipeline (2026-05-12) hides nearly all CPU worker
  time behind the next strand's GPU kernels — on chr22x20/query_medium only
  0.164s of 5.097s worker time is exposed wait.
- **New bottleneck**: scanning Viterbi at 2.145s on chr22x20/query_medium.
  Occupancy is 8.3% (4 warps/SM), limited by shared-memory pressure (24192 B/block
  → floor(102400/24192) = 4 blocks/SM). This is now the P1 optimization target.
- SSV longtarget runs at 83% occupancy (40 warps/SM) with the new multi-warp
  template; no longer a primary bottleneck at this scale.
- All serial `<<<1,1>>>` kernels have been eliminated; host-side prefix
  sum/extend/merge reduced SSV stage time by ~24% vs the pre-async baseline.
- Further work: Viterbi shared-memory reduction to raise occupancy, intra-stage
  async (`cudaMemcpyAsync`, per-slot streams), on-GPU prefix sums — see
  `nhmmer-gpu-todo.md`.
