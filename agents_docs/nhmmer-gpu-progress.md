# nhmmer GPU Support - Progress

Last updated: 2026-05-13 (register-based scanning Viterbi kernel)

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
8. Threaded CPU domain definition with nuc-optimized Forward/Backward.

The CPU domain workers use:
- `P7_NUCSEQVIEW` zero-copy view into mmap'd nucdb packed+mask data
- `p7_Forward_nuc` / `p7_Backward_nuc` reading 2-bit residues on the fly
- `UpdateFwdEmissionScores_nuc` rebuilding only 5 emission rows (ACGT + N)
- `fill_slice` decode still called for `p7_alidisplay_Create` (dsq needed
  for alignment display only; planned for elimination)

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

## Register-Based Scanning Viterbi Kernel (landed 2026-05-13)

The scanning Viterbi kernel now uses a register-based DP path for models with
M≥256 (template `STRIDE=24`, max M=768). This eliminates shared-memory bottleneck
for the DP state arrays:

**Key design**:
- M/D/I state arrays kept in `int16_t reg_M[STRIDE]` register arrays
- `#pragma unroll` is essential — without it nvcc spills to local memory (DRAM)
- Inter-lane communication via `__shfl_up_sync` for carry propagation
- Shared memory reduced to `2*M` bytes (only for `s_q[]` / `s_z[]` lookup tables)
- 4 warps/block, 184 regs/thread, 8 warps/SM (16.7% occupancy)

**Performance** (chr22x20, query_medium M=501, GPU-16):
- Scan kernel: **1.212s** (was 2.023s with shared-memory kernel) = **1.67× speedup**
- GPU loop wall: 4.3s (was 5.1s) = 14.5% overall improvement

**Limitations**:
- M<256: falls back to shared-memory kernel (register kernel has edge-case bug
  producing excess seed windows for very short models like M=80)
- STRIDE=16 template (which would allow M≤512 at 33% occupancy) blocked by nvcc
  speculative-read issue in `#pragma unroll`
- Controlled via `HMMER_VLT_REG` env var (0=force shmem, 1=force reg)

Files: `src/cuda/p7_cuda_viterbi_longtarget.cu` (kernel + dispatch logic).

## Nuc-Specific CPU Forward/Backward (landed 2026-05-13)

CPU domain workers now use a specialized Forward/Backward path optimized for the
nucleotide 2-bit+mask format:

**`P7_NUCSEQVIEW`** (`src/p7_nucdb.h`): a zero-copy view into mmap'd nucdb packed
data. Inline accessor `p7_nucseqview_residue()` returns residue code 0–3 or -1
(masked/N) via 2-bit unpack + mask check. `p7_nucseqview_subview()` creates
sub-envelope views without copying.

**`fwdback_nuc.c`** (`src/impl_sse/fwdback_nuc.c`): Forward/Backward engines that
replace `rp = om->rfv[dsq[i]]` with:
```c
int code = p7_nucseqview_residue(nsv, i);
rp = (code >= 0) ? om->rfv[code] : rfv_N;
```
Same Q-loop structure as standard engines; the per-residue overhead is 3 ALU ops
+ 1 highly-predictable branch (mask is 0 for >99% of genomic positions).

**`UpdateFwdEmissionScores_nuc`** (`src/impl_sse/p7_oprofile.c`): only rebuilds
emission scores for 5 residue codes (ACGT + N). Skips all 13 IUPAC degenerate
codes and their `esl_abc_FExpectScVec` computation. Eliminates 72% of
`esl_sse_expf` calls per `reparameterize_model` invocation.

**`reparameterize_model_nuc`** (`src/p7_domaindef.c`): counts residues directly
from `P7_NUCSEQVIEW` packed data when available; falls back to
`esl_sq_CountResidues` on the dsq when nsv is NULL.

Gating: `ddef->nuc_mode = TRUE` (set on GPU worker pipeline creation) +
`ddef->nsv != NULL` (set per window from `nhmmer_gpu_nucdb_build_nsv()`).

Impact: ~9% reduction in domain workflow wall time on chr22/query_medium;
measurable improvement in `reparameterize_model` cost. The Q-loop itself is
unchanged — it remains the dominant cost and the target for AVX2 widening
(see `nhmmer-gpu-todo.md` P0).

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
| `src/nhmmer_gpu_seqhelpers.c` | `.nucdb` ESL_SQ reconstruction (full / shell / slice) + `build_nsv` |
| `src/nhmmer_gpu_windows.c` | Windowlist helpers + synthetic-chunk lifecycle |
| `src/nhmmer_gpu_workers.c` | CPU worker pool, OMX special-state binding, nsv wiring |
| `src/impl_sse/fwdback_nuc.c` | Nuc-specific SSE Forward/Backward (reads P7_NUCSEQVIEW) |
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

System: RTX 4090, 16 CPU threads. Wall time via `date +%s%N`. 2026-05-13.

### chr22x20 (~1GB, 20 chromosomes)

Per-query (single query, wall-clock):

| Query | M | GPU-16 | CPU-16 | Speedup |
|-------|---|--------|--------|---------|
| MADE1 | 80 | 1.44s | 2.91s | 2.0× |
| query_short | 151 | 1.67s | 2.81s | 1.7× |
| query_medium | 501 | 5.14s | 8.95s | 1.7× |

Multi-query (12 queries = 3 models × 4 copies, 3 runs):

| Path | Wall time (median) | Speedup |
|------|:------------------:|---------|
| GPU-16 | **25.8s** | **2.3×** |
| CPU-16 | 59.8s | baseline |

### chr22x5 (~250MB, 5 chromosomes)

| Query | M | GPU-16 | CPU-16 | Speedup |
|-------|---|--------|--------|---------|
| MADE1 | 80 | 0.79s | 1.04s | 1.3× |
| query_short | 151 | 0.68s | 0.81s | 1.2× |
| query_medium | 501 | 1.90s | 2.74s | 1.4× |

### Standard chr22 combined (MADE1 + query_short + query_medium)

| Path | Hits | Wall |
|------|:---:|:---:|
| CPU-16 | 1476 | ~1.1s |
| GPU-16 | ~1458 | ~1.2s |

On chr22 the GPU is essentially a tie with CPU-16 — too few strands to fill
both async slots.

The chr22x5 target is local benchmark data:

- `benchmark-data/nucleotide-bench/work/chr22x5.fa`
- `benchmark-data/nucleotide-bench/work/chr22x5-overlap.nucdb.nucdb`

## Current GPU Timing Breakdown (chr22x20, query_medium M=501)

Single-run, GPU-16, `nhmmer --gpu --cpu 16`, chr22x20 (20× chr22), query_medium:

| Bucket | Time |
|--------|:---:|
| GPU loop wall | ~4.3s |
| Process elapsed | ~5.1s |

GPU stage breakdown:

| Stage | Time | Notes |
|-------|:---:|-------|
| SSV longtarget | ~1.3s | 83% occupancy, multi-warp template |
| batch filter | ~0.7s | |
| scanning Viterbi | **1.21s** | register kernel (was 2.02s with shmem kernel) |
| FB parser | ~0.4s | |
| CPU workers | ~4.5s | only 0.2s exposed wait; 4.3s hidden behind GPU |

The register-based Viterbi kernel (STRIDE=24, 4 warps/block, 184 regs/thread)
eliminates the shared-memory bottleneck. The kernel uses `#pragma unroll` with
register arrays for M/D/I state, achieving 1.67× speedup over the previous
shared-memory kernel for M=501.

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

- **GPU-16 is 2.3× faster than CPU-16** on chr22x20 with 12 queries (25.8s vs
  59.8s). The speedup grows with dataset size due to async strand overlap.
- Per-query on chr22x20: 1.7–2.0× depending on model size.
- Per-query on chr22x5: 1.2–1.4× (still amortizing setup cost at this scale).
- On standard chr22, GPU-16 and CPU-16 remain a tie (~1.1s); too few strands to
  fill both async slots.
- The register-based scanning Viterbi kernel (landed 2026-05-13) provides 1.67×
  scan kernel speedup for M≥256, reducing the former P1 bottleneck from 2.02s to
  1.21s on chr22x20/query_medium.
- The async 2-slot strand pipeline hides nearly all CPU worker time behind the
  next strand's GPU kernels — on chr22x20/query_medium: 4.5s workers with only
  0.2s exposed wait.
- SSV longtarget runs at 83% occupancy (40 warps/SM) with the multi-warp template.
- All serial `<<<1,1>>>` kernels have been eliminated.
- Further work: AVX2 F/B (P0), `fill_slice` elimination (P1), STRIDE=16 register
  kernel for higher occupancy (P2) — see `nhmmer-gpu-todo.md`.
