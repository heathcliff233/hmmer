# nhmmer GPU Support — TODO

## All Phases Complete (including GPU Domain Rescoring + Kernel Parallelization + FB Split)

Phases 1–8 are done. Phase 9 (GPU Domain Rescoring) is complete. Phase 10 (kernel parallelization + FB split) is complete. See `nhmmer-gpu-progress.md` for details.

## Phase 10: Kernel Parallelization + Forward-Backward Split — COMPLETE

- [x] Parallel-thread domain kernels: 4 of 6 kernels ported from 1-thread-per-block to T-thread-per-block
  - [x] `cuda_domain_fwd_full_kernel`: strided M/I + (a,b) prefix scan for D-state + xE tree reduction
  - [x] `cuda_domain_bck_full_kernel`: reverse prefix scan + xB tree reduction
  - [x] `cuda_domain_decoding_kernel`: strided element-wise (no prefix scan needed)
  - [x] `cuda_domain_fwd_scoreonly_kernel`: same as fwd_full without matrix write
  - [ ] `cuda_domain_optacc_kernel`: deferred (needs max-prefix scan)
  - [ ] `cuda_domain_oatrace_kernel`: deferred (inherently sequential traceback)
- [x] Forward-Backward parser split: `fb_parser_subset_ex` with `run_modes` parameter
  - [x] `p7_cuda_ForwardParserDsqdataSubset`: Forward-only, saves xmx
  - [x] `p7_cuda_BackwardParserDsqdataSubset`: Backward-only, takes xf as input
- [x] Prefilter → Backward-only pipeline: eliminates redundant Forward computation
  - [x] Prefilter calls ForwardParser (saves xmx + scores)
  - [x] F3 gating compacts survivor xf
  - [x] FB stage calls BackwardParser only with pre-computed xf

### Impact

Parallel-thread kernels: ~1.3-1.5x kernel speedup on domain rescoring. Forward-Backward split: eliminates ~50% of FB computation (no redundant Forward). Combined: MADE1 1.74s → 1.35s (1.3x), query_medium 10.3s → 5.34s (1.9x).

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

- **MADE1 parity**: GPU reports 151 vs CPU 154 (3-hit difference, <2%)
- **query_short parity**: GPU reports 119 vs CPU 120 (1-hit difference, <1%)
- **query_medium parity**: GPU reports 218 vs CPU 215 (3-hit difference, <2%)
- Remaining differences are from float32 vs double precision in Forward/Backward accumulation
- **GPU slower than CPU-4**: CPU workers (envelope-finding) dominate at 75-94% of GPU wall time

## Latest Benchmark (2026-05-09, RTX 4090, chr22 50MB)

| Config | MADE1 (M=80) | query_short (M=151) | query_medium (M=501) |
|--------|:---:|:---:|:---:|
| CPU-1 | 1.12s / 154 hits | 1.39s / 120 hits | 6.35s / 215 hits |
| CPU-4 | 0.33s / 154 hits | 0.45s / 120 hits | 1.64s / 215 hits |
| GPU-4 FASTA | 1.35s / 151 hits | 1.92s / 119 hits | 5.34s / 218 hits |

### GPU Timing Breakdown (FASTA path)

| Stage | MADE1 | query_short | query_medium |
|-------|:---:|:---:|:---:|
| SSV longtarget | 0.072s (8.3%) | 0.089s (4.8%) | 0.146s (1.5%) |
| extend+merge | 0.001s (0.1%) | 0.001s (0.0%) | 0.001s (0.0%) |
| batch filter | 0.013s (1.5%) | 0.018s (1.0%) | 0.094s (1.0%) |
| scanning Viterbi | 0.054s (6.3%) | 0.003s (0.2%) | 0.014s (0.1%) |
| Forward prefilter | 0.045s (5.2%) | 0.029s (1.6%) | 0.179s (1.8%) |
| GPU FB parser | 0.029s (3.4%) | 0.022s (1.2%) | 0.181s (1.8%) |
| **CPU workers** | **0.645s (75.1%)** | **1.685s (91.2%)** | **9.261s (93.8%)** |

**Key finding**: CPU workers (domain definition + hit reporting via `p7_pli_postViterbi_LongTarget`) consume 75–94% of GPU wall time. All GPU kernel stages combined are only 0.2–0.6s.

### GPU Timing Breakdown (nucdb path)

| Stage | MADE1 | query_short | query_medium |
|-------|:---:|:---:|:---:|
| SSV longtarget | 0.228s (20.7%) | 0.102s (4.6%) | 0.154s (1.8%) |
| extend+merge | 0.001s (0.1%) | 0.001s (0.1%) | 0.001s (0.0%) |
| batch filter | 0.016s (1.5%) | 0.027s (1.2%) | 0.099s (1.2%) |
| scanning Viterbi | 0.004s (0.4%) | 0.006s (0.3%) | 0.014s (0.2%) |
| Forward prefilter | 0.000s (0.0%) | 0.000s (0.0%) | 0.000s (0.0%) |
| GPU FB parser | 0.000s (0.0%) | 0.000s (0.0%) | 0.000s (0.0%) |
| **CPU workers** | **0.852s (77.3%)** | **2.063s (93.8%)** | **8.205s (96.8%)** |

Note: Forward prefilter and GPU FB parser are not used in the nucdb path (those stages only exist in the FASTA path).

## Open Performance Work

### P1 — Move envelope-finding to GPU (high impact, very high effort)

CPU workers are 75–94% of total time. The bottleneck is `p7_domaindef_ByPosteriorHeuristics()` running per-envelope Forward/Backward on CPU. Moving envelope-finding to GPU would eliminate the single largest bottleneck.

### P2 — Parallelize OptAcc kernel (low impact, medium effort)

`cuda_domain_optacc_kernel` still uses 1 thread/block. Needs a max-prefix scan (instead of sum-prefix scan used in Forward/Backward). Lower priority since OptAcc is not the dominant kernel.

## Future Work (Not Currently Planned)

| Item | Effort | Impact | Notes |
|------|--------|--------|-------|
| GPU envelope-finding | Very high | Eliminate 75-94% bottleneck | Needs GPU posterior decoding heuristics |
| OptAcc kernel parallelization | Medium | Small | Needs max-prefix scan |
| Async strand overlap | Low | ~0.1-0.3s | Diminishing returns |
| FM-index GPU path | High | Alternative to FASTA scanning | FM-index is CPU-optimized |
| CUDA init amortization | N/A | Already done | Engine created once before query loop |
| Redundant Forward elimination | N/A | Already done | Prefilter saves xf for Backward-only |
| Parallel-thread domain kernels | N/A | Already done | 4/6 kernels use T threads with prefix scan |
| Domain rescore nj fix | N/A | Already done | GPU now uses nj=0 (unihit) matching CPU |
