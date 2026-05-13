# nhmmer GPU Support — TODO

## Current Status (2026-05-13)

Default path: GPU SSV longtarget → GPU batch F1 gate → GPU scanning Viterbi →
GPU Forward/Backward parser → CPU domain workers (nuc-optimized F/B).

Latest optimizations:
- `.nucdb` v2: 2-bit packed + 1-bit mask; on-the-fly RC; 16k chunks / 1k overlap
- 4-row device emission tables; multi-warp SSV (83% occupancy)
- Single-buffer Viterbi DP (halved shared memory)
- **Register-based scanning Viterbi kernel** (M≥256): eliminates shared-memory DP,
  keeps M/D/I state in register arrays with `#pragma unroll`. 1.67× scan kernel
  speedup on query_medium (2.02s → 1.21s on chr22x20).
- **Nuc-specific CPU F/B**: `fwdback_nuc.c` reads directly from mmap'd 2-bit
  packed data via `P7_NUCSEQVIEW`; `UpdateFwdEmissionScores_nuc` skips degenerate
  codes (72% fewer exp() calls per reparameterize)

Parity: GPU 2600/2820 MADE1, 1860/2020 query_short, 2420/2460 query_medium
(float32 drift in Forward/Backward accumulation; all within documented tolerance).

chr22x20, 12 queries (3×4), GPU-16: ~25.8s; CPU-16: ~59.8s (**2.3× speedup**).

## P0 — AVX2 Forward/Backward for nuc path (highest impact)

**Problem**: The CPU domain definition F/B inner loop runs Q = ceil(M/4) = 126
iterations per residue (M=501) using 4-wide SSE `__m128`. This is the dominant
CPU cost at ~4s per strand on chr22x20/query_medium.

**Optimization**: With AVX2 `__m256` (8-wide floats), Q drops from 126 → 63.
With AVX-512 `__m512` (16-wide), Q drops to 32. The 2-bit+mask format means
residues are always codes 0–3, so the emission lookup is a trivial
`rfv[code][q]` with no branch for 99%+ of positions.

**Approach**:
1. Create `src/impl_avx2/fwdback_nuc.c` with AVX2 Forward/Backward engines
2. Stripe layout: 8 floats per vector → `Q = ceil(M/8)`, rightshift by 8
3. Transition table `tfv` and emission table `rfv` need 8-wide repack
4. The existing `P7_NUCSEQVIEW` accessor works unchanged
5. Runtime dispatch: check AVX2 support, fall back to SSE `fwdback_nuc.c`

**Expected impact**: ~2× speedup on F/B (Q halved, each iteration same cost).
Domain workflow 4.0s → ~2.0s on chr22x20/query_medium. End-to-end: GPU loop
wall (~5s) would clearly dominate, shifting bottleneck back to GPU.

**Considerations**:
- AVX2 rightshift (`_mm256_permutevar8x32_ps` + zero-mask) is more complex than
  SSE `_mm_shuffle_ps` but well-understood
- DD serialization still needs 4 passes (ceil(8/8) = 1 per shift? no — DD
  wrap-around needs ceil(M/8) passes regardless of vector width)
- Could template on vector width for maintainability

## P1 — Eliminate `fill_slice` decode (medium impact)

**Problem**: `nhmmer_gpu_nucdb_fill_slice` decodes the full 2-bit→byte window
per survivor (~2000 residues × ~3000 windows on chr22x20). Costs ~1s/query
(~20% of CPU worker time). The decoded dsq is used only by:
1. `esl_sq_CountResidues` → already handled by nuc-mode packed counting
2. `p7_alidisplay_Create` → reads trace positions only (50–200 residues)

**Approach**:
1. `p7_alidisplay_Create_nuc`: reads `nsv[tr->i[z]]` on the fly for each
   aligned position instead of `sq->dsq[tr->i[z]]`
2. Remove `fill_slice` call entirely from the worker loop
3. `slice_sq` becomes metadata-only (name/length, no dsq allocation)

**Expected impact**: Eliminates ~1s per query on chr22x20 (the 0.97s gap between
total CPU worker time and domain workflow time in single-thread measurement).

## P2 — Scanning Viterbi further optimization (medium impact)

Register-based kernel (landed 2026-05-13) reduced scanning Viterbi from 2.02s to
1.21s (1.67× speedup) for M≥256. The shared-memory kernel remains active for
M<256 (where the serial chain is already short enough).

Current state for query_medium (M=501, chr22x20):
- Register kernel: STRIDE=24, 4 warps/block, 184 regs/thread, 16.7% occupancy
- Still the dominant GPU stage at 1.21s (down from 2.15s)

Remaining approaches:
1. **STRIDE=16 template** (M≤512): would give 33% occupancy (16 warps/SM) but
   currently blocked by nvcc speculative-read bug in `#pragma unroll` that causes
   OOB twv accesses for iterations beyond `my_count`
2. **Register kernel for small M**: M=80 produces spurious extra seeds (>64 per
   window); root cause is striping layout difference vs shmem kernel. Needs
   investigation if small-M Viterbi becomes a bottleneck.
3. **Intra-kernel chunking**: position-level parallelism (splitting windows into
   sub-chunks); implemented but did not help due to overlap overhead dominating

## P3 — `reparameterize_model` elimination for uniform composition

Each `rescore_isolated_domain` call runs `reparameterize_model_nuc` 2–3 times
(rebuild emission scores) + one Forward just to estimate bias correction. For
windows where composition is nearly uniform (most genomic DNA), the
reparameterization changes scores by < 0.01 nats.

Skip reparameterize when: `max(|obs_freq[x] - 0.25|) < threshold` for all x.
Saves 1–2 full emission score rebuilds + 1 Forward per domain.

## P4 — Intra-stage async (low-medium impact)

Convert synchronous `cudaMemcpy` / `cudaEventSynchronize` at stage boundaries to
`cudaMemcpyAsync` on compute streams. Estimated: tens of ms saved per strand
from host launch overhead overlap.

## Completed (reference only)

- `.nucdb` v2 format (2-bit packed, on-the-fly RC)
- 4-row nuc emission tables (`d_rbv_lin_nuc`, `d_rwv_nuc`, `d_rfv_nuc`)
- Multi-warp SSV longtarget (4 warps/block, 83% occupancy)
- Single-buffer Viterbi DP (halved shared memory)
- **Register-based scanning Viterbi** (M≥256, STRIDE=24, 1.67× kernel speedup)
- Smaller default chunks (16384 vs 65536)
- Nuc-specific `UpdateFwdEmissionScores_nuc` + `reparameterize_model_nuc`
- `fwdback_nuc.c` Forward/Backward with `P7_NUCSEQVIEW` packed-data read
- 2-slot async strand pipeline (CPU workers overlap with next strand's GPU)
- Serial <<<1,1>>> kernel elimination (host prefix sum + extend/merge)
- GPU Forward/Backward parser handoff (CPU skips Fwd/Bwd recomputation)
- Forward-Backward split (Backward-only after prefilter saves xf)
- Dynamic CPU worker scheduling (work-stealing queue)
- **Gather elimination**: All GPU stages (SSV, F1 gate, scanning Viterbi, FB parser)
  now read directly from 2-bit packed nucdb data via `p7_nucdb_fetch1()`. The
  legacy `cuda_nhmmer_gather_windows_kernel` (byte-unpack) is dead code. Eliminates
  per-strand gather kernel launches and ~350MB intermediate device memory traffic.
