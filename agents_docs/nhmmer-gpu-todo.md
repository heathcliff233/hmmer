# nhmmer GPU Support — TODO

## Current Status

The accepted default path is GPU filters + GPU scanning Viterbi + exact-F3 GPU Forward prefilter + GPU Forward/Backward parser handoff + CPU domain/hit processing. `hmmnucdb` defaults to overlap chunking (`--overlap 2001`) for the fast GPU-resident SSV path. Historical phase notes below include GPU domain-rescoring experiments that are not the accepted default continuation path.

Current query_medium smoke result on chr22, RTX 4090, `--cpu 4`: speed script measured CPU-4 FASTA 1.669s / 648 main-output hit lines and GPU default fast overlap `.nucdb` 1.232s / 648 hit lines; focused repeats showed CPU-4 wall 1.77s and GPU fast `.nucdb` wall 1.53-1.86s. Strict `--tblout` CPU FASTA versus GPU fast `.nucdb` rows both produced 215 rows with no diff. GPU worker time is mostly CPU domain workflow; CPU Backward is 0.000s in the default handoff path.

## Completed: GPU Forward/Backward Parser Handoff

- [x] `p7_pli_postFwd_LongTarget()` in `p7_pipeline.c`: skips Forward recomputation, uses GPU-precomputed xf/fwdsc
- [x] `nhmmer_gpu_worker_process_post_fwd()`: injects GPU xf into pli->oxf->xmx, reconstructs totscale
- [x] Per-window xf offset computation for multi-worker partitioning (`surv_xf_offsets`)
- [x] Gated on `use_skip_fwd = info->do_gpu_fwd && prefilter_xf != NULL`; `info->do_gpu_fwd` is default-on with `--gpu`
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

## Historical: GPU Domain Rescoring Experiment

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

This work proved cross-window batching and kernel parallelization ideas, but the accepted default currently keeps domain definition, null2, hit storage, thresholding, and output CPU-side after GPU parser handoff. See `nhmmer-gpu-progress.md` for the active architecture.

## Known Issues

- **MADE1 parity**: GPU reports 462 vs CPU 465 (3-hit difference, <1%)
- **query_short parity**: GPU reports 363 vs CPU 363 (exact match)
- **query_medium parity**: GPU reports 648 vs CPU 648 (exact match)
- Remaining differences are from float32 vs double precision in Forward/Backward accumulation
- **Main remaining cost**: CPU domain workflow after GPU parser handoff

## Historical Benchmark (2026-05-09, RTX 4090, chr22 50MB)

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

**Historical finding**: CPU workers (domain definition + hit reporting) consumed 44–94% of GPU wall time depending on model size. Current query_medium fast `.nucdb` worker time is still dominated by CPU domain workflow, but total wall is now faster than CPU-4 in the smoke run after the GPU window-ordering fixes.

## Open Performance Work

### P1 — Move envelope-finding to GPU (high impact, very high effort)

CPU workers are still the largest GPU timing bucket. The bottleneck is `p7_domaindef_ByPosteriorHeuristics()` and related CPU domain workflow after parser handoff. Moving this workflow to GPU would eliminate the largest remaining bottleneck, but it is high-risk because domain definition is tightly coupled to posterior decoding and hit reporting semantics.

### P2 — Parallelize OptAcc kernel (low impact, medium effort)

`cuda_domain_optacc_kernel` still uses 1 thread/block. Needs a max-prefix scan (instead of sum-prefix scan used in Forward/Backward). Lower priority since OptAcc is not the dominant kernel.

## Future Work (Not Currently Planned)

| Item | Effort | Impact | Notes |
|------|--------|--------|-------|
| GPU domain workflow | Very high | Reduce largest remaining bucket | Needs GPU posterior decoding/domain semantics |
| OptAcc kernel parallelization | Medium | Small | Needs max-prefix scan |
| Async strand overlap | N/A | Already done (2026-05-12) | 2-slot double-buffered strand pipeline; CPU workers of strand N overlap with GPU of strand N+1. chr22x5 median CPU-16 4.019s → GPU-16 2.924s (~27% faster, 5-run). chr22 is a tie (~1.0s both). |
| FM-index GPU path | High | Alternative to FASTA scanning | FM-index is CPU-optimized |
| CUDA init amortization | N/A | Already done | Engine created once before query loop |
| Redundant Forward elimination | N/A | Already done | Prefilter saves xf for Backward-only |
| GPU Fwd/Bwd parser handoff | N/A | Already done | `p7_pli_postFwd_LongTarget()` uses GPU xf |
| Parallel-thread domain kernels | N/A | Already done | 4/6 kernels use T threads with prefix scan |
| Domain rescore nj fix | N/A | Already done | GPU now uses nj=0 (unihit) matching CPU |
| Overlap-nucdb GPU-resident path | N/A | Already done | Zero per-chunk H2D for SSV |

## Remaining Async Opportunities (post strand-level pipelining)

With strand-level overlap (2-slot double buffer) in place, the remaining
per-strand GPU pipeline is still fully serial on the default stream with
synchronous `cudaMemcpy` / `cudaEventSynchronize` at every stage boundary.
Future async passes, in rough order of expected wall-time return vs
implementation cost:

1. **Intra-stage async H2D/D2H.** Convert all `cudaMemcpy` in SSV,
   F1-resident, Viterbi FromF1, and FB-parser stages to
   `cudaMemcpyAsync` on a per-engine compute stream. Replace host
   `cudaEventSynchronize` barriers with `cudaStreamWaitEvent` between
   stages. Estimated saving: tens of ms of host launch overhead per
   strand on overlap `.nucdb`; larger on FASTA targets (not supported
   here). Unlocks overlapping host prefix-sum work with prior stage D2H.

2. **Per-slot compute stream.** Today one engine-global default stream
   serializes everything. A per-slot stream lets slot N+1's GPU stages
   begin while slot N's D2H is still draining; combined with (1),
   enables `t_gpu_fb_d2h` ↔ `t_ssv_launch` overlap at slot boundaries.
   Requires either duplicating subsets of device scratch per slot or a
   disciplined single-stream ordering with cross-stream events.
   Currently the single `P7_CUDA_ENGINE` serializes both slots — this
   is the cleanest place to add true GPU-side pipelining.

3. **On-GPU prefix sum for SSV / Viterbi counts.** The 2026-05 commits
   moved `<<<1,1>>>` prefix sums to the host to kill kernel launch
   overhead, but now the SSV / VLT stages round-trip
   `d_*_win_count → host → h_*_win_offsets → d_*_win_offsets` via
   synchronous `cudaMemcpy`. A cub block scan on a single warp would
   eliminate the round-trip; would stack with (1)+(2) to keep the
   stage fully on device.

4. **Keep `xf`/`xb` on device during CPU domain definition.** The
   parser currently D2Hs compact xf/xb matrices for all FB survivors
   (~32KB to MB per strand) because the CPU worker's posterior-decoding
   path needs them. Eliminating this transfer requires moving null2 +
   domain envelope refinement onto GPU, which overlaps with the P1
   item above.

5. **Multi-GPU strand dispatch.** With 2-slot double buffering in place,
   a per-GPU engine + round-robin strand dispatch is a natural next
   step for HPC hosts.

6. **Dead-code cleanup.** `nhmmer_gpu_forward_prefilter`,
   `nhmmer_gpu_run_fb_parser_batch`, `nhmmer_gpu_batch_filter`
   (host-packed), and the `nhmmer_gpu_window_batch_*` helpers are
   no longer called after the Phase 1 path pruning. Deleting them
   simplifies downstream async work and trims build time.

7. **Thread-budget tuning.** Each slot currently gets `info->ncpus`
   workers, so in async steady-state we peak at `2 * ncpus` worker
   threads for brief windows at slot transitions. Evaluate
   `ncpus_per_slot = (ncpus + 1) / 2` and expose a
   `--gpu-async-workers-per-slot` flag if profiling shows contention.
