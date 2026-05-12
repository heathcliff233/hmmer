# nhmmer GPU Support — TODO

## Current Status (2026-05-13)

Default path: GPU SSV longtarget → GPU batch F1 gate → GPU scanning Viterbi →
GPU Forward/Backward parser → CPU domain workers (nuc-optimized F/B).

Latest optimizations:
- `.nucdb` v2: 2-bit packed + 1-bit mask; on-the-fly RC; 16k chunks / 1k overlap
- 4-row device emission tables; multi-warp SSV (83% occupancy)
- Single-buffer Viterbi DP (halved shared memory)
- **Nuc-specific CPU F/B**: `fwdback_nuc.c` reads directly from mmap'd 2-bit
  packed data via `P7_NUCSEQVIEW`; `UpdateFwdEmissionScores_nuc` skips degenerate
  codes (72% fewer exp() calls per reparameterize)

Parity: MADE1 462/465, query_short 360/363, query_medium 636/648 (float32 drift).

chr22x20, 12 queries (3×4), GPU-16: ~28.3s median; CPU-16: ~60s median (~2.1× speedup).

## P0 — AVX2 Forward/Backward for nuc path (highest impact)

**Problem**: The CPU domain definition F/B inner loop runs Q = ceil(M/4) = 126
iterations per residue (M=501) using 4-wide SSE `__m128`. This is the dominant
CPU cost at ~4s per strand on chr22x20/query_medium.

**Optimization**: With AVX2 `__m256` (8-wide floats), Q drops from 126 → 63.
With AVX-512 `__m512` (16-wide), Q drops to 32. The 2-bit+mask format means
residues are always codes 0–3, so the emission lookup is a trivial
`rfv[code][q]` with no branch for 99%+ of positions.

**Approach**:
1. Create `src/impl_sse/fwdback_nuc_avx2.c` with AVX2 Forward/Backward engines
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

## P2 — Scanning Viterbi occupancy (high impact, high effort)

Current: 8.3% occupancy (4 blocks/SM), limited by shared memory (24192 B/block
for the DP state with single-buffer optimization). At 96% SM utilization the SM
is fully busy but starved of warps.

Target: 8 blocks/SM requires ≤12800 B/block (~47% further reduction).

Approaches:
1. **Tile along M**: process DP in strips; reduce per-block storage from O(M) to
   O(K) at cost of multiple kernel passes
2. **int8 DP rows**: halves row storage where score range allows
3. **Register-file DP for small M**: no shared memory needed for M≤256

This is the single highest-impact GPU kernel optimization: halving Viterbi time
(2.1s → ~1.0s on chr22x20) reduces GPU loop wall by ~25%.

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
- Smaller default chunks (16384 vs 65536)
- Nuc-specific `UpdateFwdEmissionScores_nuc` + `reparameterize_model_nuc`
- `fwdback_nuc.c` Forward/Backward with `P7_NUCSEQVIEW` packed-data read
- 2-slot async strand pipeline (CPU workers overlap with next strand's GPU)
- Serial <<<1,1>>> kernel elimination (host prefix sum + extend/merge)
- GPU Forward/Backward parser handoff (CPU skips Fwd/Bwd recomputation)
- Forward-Backward split (Backward-only after prefilter saves xf)
- Dynamic CPU worker scheduling (work-stealing queue)
