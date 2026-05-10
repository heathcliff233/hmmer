# nhmmer GPU Support — TODO

## All Phases Complete (including GPU Domain Rescoring + Kernel Parallelization + FB Split + Skip-Forward)

Phases 1–8 are done. Phase 9 (GPU Domain Rescoring) is complete. Phase 10 (kernel parallelization + FB split) is complete. Phase 11 (skip-Forward optimization) is complete. See `nhmmer-gpu-progress.md` for details.

## Phase 11: Skip-Forward Optimization — COMPLETE

- [x] `p7_pli_postFwd_LongTarget()` in `p7_pipeline.c`: skips Forward recomputation, uses GPU-precomputed xf/fwdsc
- [x] `nhmmer_gpu_worker_process_post_fwd()`: injects GPU xf into pli->oxf->xmx, reconstructs totscale
- [x] Per-window xf offset computation for multi-worker partitioning (`surv_xf_offsets`)
- [x] Gated on `use_skip_fwd = info->do_gpu_fwd && prefilter_xf != NULL`; `info->do_gpu_fwd` is default-on with `--gpu`
- [x] Thread dispatch: `nhmmer_gpu_thread_func_post_fwd` for skip-Forward, `nhmmer_gpu_thread_func_post_vit` for standard path
- [x] `--gpu-compare` and `--gpu-cpu-postmsv` flags for debug/comparison

### Impact

Eliminates redundant CPU Forward computation in post-filter workers when GPU Forward prefilter has already computed xf. Savings most visible on workloads with many Forward survivors (query_medium). This is now the default `--gpu` path; hidden `--gpu-no-fwd-prefilter` is the diagnostic fallback to the older CPU Fwd/Bwd continuation.

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

Parallel-thread kernels: ~1.3-1.5x kernel speedup on domain rescoring. Forward-Backward split: eliminates ~50% of FB computation (no redundant Forward). Combined: MADE1 1.74s → 1.35s (1.3x), query_medium 10.3s → 5.34s (1.9x). Scanning Viterbi threshold fix: MADE1 1.35s → 0.91s (1.5x), query_medium 5.34s → 2.85s (1.9x).

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

- **MADE1 parity**: GPU reports 462 vs CPU 465 (3-hit difference, <1%)
- **query_short parity**: GPU reports 363 vs CPU 363 (exact match)
- **query_medium parity**: GPU reports 648 vs CPU 648 (exact match)
- Remaining differences are from float32 vs double precision in Forward/Backward accumulation
- **GPU slower than CPU-4**: CUDA init overhead + envelope-finding runtime (GPU FASTA is 1.8-3.3x slower; overlap-nucdb is 1.4-2.6x slower)

## Latest Benchmark (2026-05-09, RTX 4090, chr22 50MB)

| Config | MADE1 (M=80) | query_short (M=151) | query_medium (M=501) |
|--------|:---:|:---:|:---:|
| CPU-1 | 1.08s / 465 hits | 1.33s / 363 hits | 5.67s / 648 hits |
| CPU-4 | 0.31s / 465 hits | 0.40s / 363 hits | 1.57s / 648 hits |
| GPU-4 FASTA | 1.01s / 462 hits | 0.99s / 363 hits | 2.87s / 648 hits |
| GPU-4 nucdb | 0.78s / 462 hits | 0.97s / 363 hits | 2.50s / 648 hits |
| GPU-4 overlap-nucdb | 0.64s / 462 hits | 1.02s / 363 hits | 2.23s / 648 hits |

### GPU Timing Breakdown (FASTA path, MADE1)

| Stage | Time | % |
|-------|:---:|:---:|
| SSV longtarget | 0.211s | 45.9% |
| extend+merge | 0.001s | 0.1% |
| batch filter | 0.023s | 5.0% |
| scanning Viterbi | 0.004s | 0.8% |
| Forward prefilter | 0.018s | 4.0% |
| GPU FB parser | 0.000s | 0.0% |
| **CPU workers** | **0.203s** | **44.2%** |

**Key finding**: CPU workers (domain definition + hit reporting) consume 44–94% of GPU wall time depending on model size. All GPU kernel stages combined are only 0.2–0.3s.

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
| Skip-Forward optimization | N/A | Already done | `p7_pli_postFwd_LongTarget()` uses GPU xf |
| Parallel-thread domain kernels | N/A | Already done | 4/6 kernels use T threads with prefix scan |
| Domain rescore nj fix | N/A | Already done | GPU now uses nj=0 (unihit) matching CPU |
| Overlap-nucdb GPU-resident path | N/A | Already done | Zero per-chunk H2D for SSV |
