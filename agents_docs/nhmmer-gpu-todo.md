# nhmmer GPU Support — TODO

## Current Status

The accepted default path is GPU filters + GPU scanning Viterbi + exact-F3 GPU Forward prefilter + GPU Forward/Backward parser handoff + CPU domain/hit processing. `hmmnucdb` defaults to overlap chunking (`--chunk-size 16384 --overlap 1024`) for the fast GPU-resident SSV path.

Latest changes (2026-05-12):

- `.nucdb` v2 format: 2-bit packed nucleotides + 1-bit mask; on-the-fly RC; disk chr22x5 98.4 MB vs 524.4 MB v1.
- 4-row nuc emission tables (`d_rbv_lin_nuc`, `d_rwv_nuc`, `d_rfv_nuc`): 4 rows vs Kp=18, M=501 table 2004 B vs 9018 B.
- Multi-warp SSV longtarget kernel: 4 warps/block, `MAX_STRIDE` template, 83% occupancy (40/48 warps/SM).
- Smaller default chunk size (16384 vs 65536): 4× more grid blocks, better GPU saturation.
- chr22x5 GPU-16 median: 2.447s (v2) vs 2.924s (v1 async) vs 3.668s (CPU-16); ~33% faster than CPU-16.

Outstanding: hit-count drift in v2 path (~1980 vs 6294 on chr22x5); scanning Viterbi occupancy is 8.3% (the P1 optimization target).

Historical phase notes below include GPU domain-rescoring experiments that are not the accepted default continuation path.

## Completed: `.nucdb` v2 Format + 4-Row Tables + Multi-Warp SSV (2026-05-12)

- [x] `.nucdb` v2: 2-bit packed A/C/G/T (0/1/2/3) + 1-bit non-ACGT mask per residue
- [x] On-the-fly reverse complement in gather kernel (index inversion + XOR with 3)
- [x] No pre-stored RC strand on disk or device
- [x] `hmmnucdb` defaults: `--chunk-size 16384 --overlap 1024`
- [x] Runtime reads chunk/overlap from `.nucdb` header (removed hardcoded 65536 check)
- [x] `d_rbv_lin_nuc` (4×M bytes), `d_rwv_nuc`, `d_rfv_nuc` — 4-row nuc emission tables
- [x] `CreateNucTables` in `p7_cuda_runtime.cu`; declaration in `p7_cuda.h`; fields in `p7_cuda_internal.h`
- [x] All kernel launch sites pass nuc tables when available
- [x] Multi-warp SSV longtarget: template `MAX_STRIDE` (8/16/32/64), 4 warps/block (`SSV_LT_WARPS_PER_BLOCK=4`), warp-level shuffle, 42 regs/thread, 83% occupancy

### Impact

chr22x5 GPU-16 median: 2.924s (v1) → 2.447s (v2), ~16% further speedup on top of async baseline.
SSV occupancy: single-warp → 83% (40/48 warps/SM). Disk: 98.4 MB vs 524.4 MB.

## Completed: GPU Forward/Backward Parser Handoff

- [x] `p7_pli_postFwd_LongTarget()` in `p7_pipeline.c`: skips Forward recomputation, uses GPU-precomputed xf/fwdsc
- [x] `nhmmer_gpu_worker_process_post_fwd()`: injects GPU xf into pli->oxf->xmx, reconstructs totscale
- [x] Per-window xf offset computation for multi-worker partitioning (`surv_xf_offsets`)
- [x] Gated on `use_skip_fwd = info->do_gpu_fwd && prefilter_xf != NULL`; `info->do_gpu_fwd` is default-on with `--gpu`
- [x] `--gpu-compare` and `--gpu-cpu-postmsv` flags for debug/comparison (removed in 2026-05-12 cleanup)

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

- **v2 hit-count drift**: GPU v2 reports ~1980 vs CPU/GPU-v1 6294 on chr22x5 (float32 accumulation change in the v2 path introduces score boundary drift). This is the P0 correctness issue before v2 can replace v1 as the accepted default.
- **MADE1 parity (v1)**: GPU reports 462 vs CPU 465 (3-hit difference, <1%)
- **query_short parity (v1)**: GPU reports 363 vs CPU 363 (exact match)
- **query_medium parity (v1)**: GPU reports 648 vs CPU 648 (exact match)
- Remaining v1 differences are from float32 vs double precision in Forward/Backward accumulation

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

### P0 — Restore hit parity in v2 path (correctness, must fix before v2 is default)

The v2 path reports ~1980 hits on chr22x5 vs 6294 for CPU/GPU-v1. Root cause is
float32 score accumulation drift introduced with the 4-row table or 2-bit unpack
path. Must be diagnosed and fixed before v2 replaces v1 as the accepted smoke
benchmark.

Approach: binary-search the stage where scores diverge (SSV → F1 gate → Viterbi
→ F3 gate → FB parser). Add a `--gpu-v2-compare` flag that runs both v1 and v2
emission table paths on the same chunks and reports per-sequence score diffs.

### P1 — Reduce scanning Viterbi shared-memory footprint (high impact, high effort)

Scanning Viterbi occupancy is 8.3% (4 blocks/SM) on sm_89, limited by 24192 B
per block for DP state. At 96% SM utilization, the SM is fully busy but starved
of warps — reducing shared memory is the only way to raise occupancy.

Current: floor(102400 / 24192) = 4 blocks/SM.
Target: floor(102400 / X) ≥ 8 blocks/SM requires X ≤ 12800 B/block (~47% reduction).

Approaches:

1. **Tile along M**: process the DP in strips of K residues; reduce per-block
   storage from O(M) to O(K) at the cost of multiple kernel passes.
2. **Reduce row precision**: use int8 for DP rows where range allows; halves
   row storage.
3. **Register-file DP**: for small M, keep DP rows entirely in registers (no
   shared memory needed); fall back to shared memory for large M.

This is the single highest-impact open item: halving Viterbi time on chr22x20
(from 2.145s) would reduce GPU loop wall by ~25-30%.

### P2 — Move envelope-finding to GPU (high impact, very high effort)

CPU workers are still a large GPU timing bucket. The bottleneck is
`p7_domaindef_ByPosteriorHeuristics()` and related CPU domain workflow after
parser handoff. Moving this workflow to GPU would eliminate the largest remaining
CPU-side cost, but it is high-risk because domain definition is tightly coupled
to posterior decoding and hit reporting semantics.

### P3 — Parallelize OptAcc kernel (low impact, medium effort)

`cuda_domain_optacc_kernel` still uses 1 thread/block. Needs a max-prefix scan
(instead of sum-prefix scan used in Forward/Backward). Lower priority since
OptAcc is not the dominant kernel.

## Future Work (Not Currently Planned)

| Item | Effort | Impact | Notes |
|------|--------|--------|-------|
| GPU domain workflow | Very high | Reduce largest remaining bucket | Needs GPU posterior decoding/domain semantics |
| OptAcc kernel parallelization | Medium | Small | Needs max-prefix scan |
| Async strand overlap | N/A | Already done (2026-05-12) | 2-slot double-buffered strand pipeline; CPU workers of strand N overlap with GPU of strand N+1. chr22x5 median CPU-16 3.668s → GPU-16 2.447s (v2, ~33% faster). |
| FM-index GPU path | High | Alternative to FASTA scanning | FM-index is CPU-optimized |
| CUDA init amortization | N/A | Already done | Engine created once before query loop |
| Redundant Forward elimination | N/A | Already done | Prefilter saves xf for Backward-only |
| GPU Fwd/Bwd parser handoff | N/A | Already done | `p7_pli_postFwd_LongTarget()` uses GPU xf |
| Parallel-thread domain kernels | N/A | Already done | 4/6 kernels use T threads with prefix scan |
| Domain rescore nj fix | N/A | Already done | GPU now uses nj=0 (unihit) matching CPU |
| Overlap-nucdb GPU-resident path | N/A | Already done | Zero per-chunk H2D for SSV |
| `.nucdb` v2 format | N/A | Already done (2026-05-12) | 2-bit packed, on-the-fly RC, 98.4 MB vs 524.4 MB chr22x5 |
| 4-row nuc emission tables | N/A | Already done (2026-05-12) | 4 rows vs Kp=18, M=501: 2004 B vs 9018 B |
| Multi-warp SSV longtarget | N/A | Already done (2026-05-12) | 4 warps/block, 83% occupancy |
| 4× smaller default chunk size | N/A | Already done (2026-05-12) | 16384 vs 65536; 4× more grid blocks |

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
