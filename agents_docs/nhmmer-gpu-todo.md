# nhmmer GPU Support — TODO

## All Phases Complete (including GPU Domain Rescoring)

Phases 1–8 are done. Phase 9 (GPU Domain Rescoring) is complete. See `nhmmer-gpu-progress.md` for details.

## Phase 9: GPU Domain Rescoring — COMPLETE

- [x] `p7_cuda_DomainRescoreBatch`: batched Forward+Backward+Decoding+OptimalAccuracy+OATrace+Domcorrection
- [x] 6 CUDA kernels: single-thread-per-block, one block per domain
- [x] Grow-only engine buffers (no per-call cudaMalloc/cudaFree)
- [x] Bulk H2D transfers (staging buffers → single memcpy)
- [x] Cross-window domain batching (all ~3000 domains in one GPU call)
- [x] Trim batching (all ~1100 trim domains in second GPU call)
- [x] Three-phase design in `nhmmer_gpu_worker_process_post_vit_gpu`
- [x] `p7_domaindef_ByPosteriorHeuristics` with `envelopes_only=TRUE`
- [x] Per-domain model reparameterization (composition-adjusted emissions)
- [x] `nhmmer_gpu_trace_from_gpu`: GPU trace → P7_TRACE conversion
- [x] P7_ALIDISPLAY creation from GPU traces
- [x] Envelope trimming with batched re-rescore
- [x] Hit reporting with correct coordinate transforms

### Impact

GPU domain rescoring replaces `rescore_isolated_domain` (the 67-91% bottleneck). Performance improvement: MADE1 34s → 1.74s (18x), query_short 120s → 2.07s (53x). Still ~5x slower than CPU-4 due to single-thread-per-block kernel design (short domains don't exploit GPU SIMT parallelism).

## Known Issues

- **MADE1 parity**: GPU reports 156 vs CPU 154 (2-hit difference, within 1% tolerance)
- **query_short parity**: GPU reports 124 vs CPU 120 (4-hit difference)
- **query_medium parity**: GPU reports 264 vs CPU 215 (extra hits from fixed `xw_*` profile parameters in scanning Viterbi kernel)
- **GPU slower than CPU-4**: Domain kernels are single-threaded per block; short domains (L~50-170) can't exploit warp parallelism

## Future Work (Not Currently Planned)

| Item | Effort | Impact | Notes |
|------|--------|--------|-------|
| Multi-threaded warp-cooperative domain kernels | High | 4-8x kernel speedup | Use warp shuffle for inner loop; would need Q threads per block |
| Per-window xw_* reconfigure on GPU | Medium | Fixes parity (264→215 for query_medium) | Need to pass per-window L to scanning Viterbi kernel |
| Async strand overlap | Low | ~0.1-0.3s | Diminishing returns |
| FM-index GPU path | High | Alternative to FASTA scanning | FM-index is CPU-optimized |
