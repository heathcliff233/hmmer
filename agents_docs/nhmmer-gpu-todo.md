# nhmmer GPU Support — TODO

## All Phases Complete (including GPU FB Parser)

Phases 1–7 are done. Phase 8 (GPU FB Parser) is complete. See `nhmmer-gpu-progress.md` for details.

## Phase 8: GPU FB Parser — COMPLETE

- [x] `nhmmer_gpu_run_fb_parser_batch`: batch Forward+Backward parser on GPU for all post-Forward-prefilter survivors
- [x] `nhmmer_gpu_worker_process_post_vit_gpu`: GPU FB worker with F3 gate, xmx binding, domaindef, hit reporting
- [x] `NHMMER_OMX_BINDING` / `nhmmer_gpu_BindOmxXmx` / `nhmmer_gpu_RestoreOmxXmx`: swap P7_OMX fields to point at GPU buffers
- [x] Sub-batching at 128M floats per sub-batch (avoids GPU OOM on large window sets)
- [x] Per-window CPU fallback when GPU status != eslOK
- [x] Timing instrumentation: `t_gpu_fb_parser` in breakdown
- [x] Transparent fallback: if GPU FB fails, CPU Forward+Backward path takes over
- [x] Hit parity verified: MADE1 153/154 (1-hit FP diff), query_short 120=120, query_medium 226/215

### Impact

GPU FB parser replaces parser-level `p7_ForwardParser` + `p7_BackwardParser`. Savings: 8-30% reduction in CPU worker time. Remaining CPU time dominated by `rescore_isolated_domain` (full Forward+Backward+OptimalAccuracy per domain).

## Known Issues

- **MADE1 parity**: GPU reports 153 vs CPU 154 (1-hit difference from GPU Forward score FP precision at F3 boundary)
- **query_medium parity**: GPU reports 226 vs CPU 215 (extra hits from fixed `xw_*` profile parameters in scanning Viterbi kernel)
- **GPU slower than CPU-4**: Dominated by `rescore_isolated_domain` in CPU domaindef. See `nhmmer-gpu-perf-gap.md`.

## Future Work (Not Currently Planned)

| Item | Effort | Impact | Blocker |
|------|--------|--------|---------|
| GPU full Forward/Backward for `rescore_isolated_domain` | Very high | Would eliminate 67-91% bottleneck | Per-domain subsequence DP, varying lengths |
| GPU OptimalAccuracy + OATrace | Very high | Completes full domaindef on GPU | Complex traceback algorithm |
| Per-window xw_* reconfigure on GPU | Medium | Fixes parity (226→215 for query_medium) | Need to pass per-window L to kernel |
| Async strand overlap | Low | ~0.1-0.3s | Diminishing returns |
| FM-index GPU path | High | Alternative to FASTA scanning | FM-index is CPU-optimized |
