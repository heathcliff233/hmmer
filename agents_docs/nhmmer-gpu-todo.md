# nhmmer GPU Support — TODO

## All Phases Complete

Phases 1–6 are done. See `nhmmer-gpu-progress.md` for details.

## Phase 7 (Current): Pre-stored RC + GPU-Resident — COMPLETE

- [x] Reconstruct RC strand from nucdb RC chunks (skip `esl_sq_ReverseComplement`)
- [x] `p7_cuda_engine_UploadNucdb` / `ReleaseNucdb` / `NucdbDevPtr`
- [x] `p7_cuda_SSVLongtargetResident` (zero H2D, direct device pointer)
- [x] Resident path activates when nucdb overlap >= model max_length
- [x] Hit parity verified: 465=465 on MADE1 for all paths

## Known Issues

- **query_short/medium parity**: GPU reports more hits (372 vs 363, 693 vs 648) due to fixed `xw_*` profile parameters in scanning Viterbi kernel. CPU reconfigures per window length; GPU does not.
- **GPU slower than CPU-4**: Dominated by CPU ForwardParser/domain on surviving windows. See `nhmmer-gpu-perf-gap.md`.

## Future Work (Not Currently Planned)

| Item | Effort | Impact | Blocker |
|------|--------|--------|---------|
| GPU ForwardParser + domain def | Very high | Would make GPU faster than CPU-4 | Complex algorithms |
| Per-window xw_* reconfigure on GPU | Medium | Fixes parity (372→363, 693→648) | Need to pass per-window L to kernel |
| GPU Forward pre-filter | Medium | Reduces CPU downstream survivors | Needs `p7_pli_postSSV_LongTarget` split |
| Async strand overlap | Low | ~0.1-0.3s | Diminishing returns |
| FM-index GPU path | High | Alternative to FASTA scanning | FM-index is CPU-optimized |
