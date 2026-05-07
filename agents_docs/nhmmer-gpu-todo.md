# nhmmer GPU Support — TODO

## Phase 1: GPU SSV LongTarget — COMPLETE

- [x] Add `--gpu`, `--gpu-device`, `--gpu-chunk-size` options to `src/nhmmer.c`
- [x] Create `src/nhmmer_gpu.c` with GPU serial loop
- [x] Create `src/nhmmer_internal.h` with NHMMER_GPU_INFO struct
- [x] Update `src/Makefile.in` to compile `nhmmer_gpu.o` and CUDA file
- [x] Update `configure.ac` CUDA_OBJ to include `cuda/p7_cuda_ssv_longtarget.o`
- [x] Create `src/cuda/p7_cuda_ssv_longtarget.cu` (warp-per-chunk kernel)
- [x] Wire GPU SSV into serial loop with both-strand handling
- [x] Call `p7_pli_ExtendAndMergeWindows()` on GPU window results
- [x] CPU downstream: bias filter, MSV recompute, postSSV pipeline
- [x] Fix hang: skip `esl_threads_AddThread` when `--gpu` active
- [x] Fix id_length_list: callback mechanism for sequence ID/length registration
- [x] Hit parity verified on chr22 benchmark (MADE1, query_short)

## Phase 2: Threaded Downstream Pipeline — COMPLETE

- [x] Add `ncpus` and `go` fields to `NHMMER_GPU_INFO`
- [x] Implement `NHMMER_GPU_WORKER` struct with per-thread state
- [x] Deep-copy oprofile (`p7_oprofile_Copy`, not Clone) to avoid race conditions
- [x] Thread function distributes merged windows across N CPU threads
- [x] Merge results via `p7_tophits_Merge` / `p7_pipeline_Merge`
- [x] Single-threaded fallback when ncpus <= 1
- [x] Hit parity verified: GPU-1 and GPU-4 produce identical hit counts

## Phase 3: Optimization — COMPLETE

- [x] Chunk_size tuning: 64K optimal (32K slightly better for short models, 128K+ worse)
- [x] CUDA engine reuse across queries: hoisted outside query loop, saves ~250ms/query
- [ ] Async strand overlap (deferred — GPU SSV is ~0.1s, not worth complexity)

## Phase 4: Batch GPU Filters — COMPLETE

- [x] `NHMMER_GPU_WINDOW_BATCH` struct with synthetic ESL_DSQDATA_CHUNK (zero-copy)
- [x] `nhmmer_gpu_window_batch_init/pack/free` helpers
- [x] GPU batch MSV + null + bias + F1 gating (`--gpu-batch`)
- [x] Handle eslERANGE (overflow) as pass in batch filter
- [x] GPU Viterbi single-score pre-filter (`--gpu-vit-prefilter`)
- [x] CLI flags: `--gpu-batch`, `--gpu-vit-prefilter`, `--gpu-fwd-prefilter`
- [x] Hit parity verified: MADE1 154=154, query_short 120=120, query_medium 217=217 (batch only)
- [x] Viterbi pre-filter: query_medium 215 (removes 2 boundary hits, matches CPU count)
- [x] Speed: query_medium GPU-4+batch+vit 2.15s vs GPU-4 baseline 3.23s (1.5x improvement)

## Phase 5: GPU Scanning Viterbi — COMPLETE

- [x] Create `src/cuda/p7_cuda_viterbi_longtarget.cu` with warp-per-window scanning Viterbi kernel
- [x] Warp-based DP: 8 active lanes for int16 SIMD, `__shfl_up_sync` for cross-lane communication
- [x] In-kernel window emission via atomicAdd when xE >= threshold, with DP state reset
- [x] Lazy-F D-state propagation for DD transitions
- [x] Host wrapper: pack windows into pinned buffer, grow-on-demand engine-persistent device buffers
- [x] `--gpu-vit-longtarget` CLI flag in `src/nhmmer.c`
- [x] Post-Viterbi worker thread that skips MSV/bias stages, grows oxf matrix per window
- [x] Made `p7_pli_postViterbi_LongTarget` public for direct post-Viterbi entry
- [x] GPU threshold computation: null scores computed analytically, bias via existing GPU kernel
- [x] Batched kernel launch: 8 warps/block for better GPU occupancy
- [x] Hit parity: MADE1 462 (vs 465 CPU, 3-hit discrepancy from fixed xw_*), query_short 363=363, query_medium 648=648
- [x] Update `configure.ac` CUDA_OBJ to include `cuda/p7_cuda_viterbi_longtarget.o`

## Phase 6: Rebase + Optimizations + Nucdb — COMPLETE

- [x] Rebased onto h3-gpu (engine reuse, bias kernels, pre-allocated events, SSV default-on, resident database)
- [x] Default-on GPU stages: `--gpu-batch`, `--gpu-vit-prefilter`, `--gpu-vit-longtarget` auto-enabled with `--gpu`
- [x] GPU bias filter replaces CPU `p7_bg_FilterScore` in Viterbi pre-filter
- [x] Persistent scratch arrays in NHMMER_GPU_INFO (grow-only, no per-batch malloc/free)
- [x] Engine reset between queries (`p7_cuda_engine_Reset`)
- [x] Nucleotide GPU database format (`p7_nucdb`): pre-chunked, mmap'd, both strands
- [x] `hmmnucdb` CLI tool to build .nucdb from FASTA
- [x] `nhmmer_gpu_nucdb_loop()` processes nucdb directly (eliminates FASTA parsing)
- [x] Removed `--gpu-compare` debug flag
- [x] Hit parity verified: 465=465 (GPU FASTA), 465=465 (GPU nucdb) vs CPU on MADE1

## Future Work (Not Planned)

- GPU Forward pre-filter for sub-windows (requires splitting p7_pli_postSSV_LongTarget)
- GPU-side ForwardParser/domain definition to eliminate CPU post-Viterbi bottleneck
- FM-index GPU path (FM-index remains CPU-only)
- Per-window profile reconfiguration on GPU (would fix MADE1 3-hit discrepancy)
- GPU-resident nucdb: `p7_cuda_SSVLongtargetResident()` to skip H2D transfer
- Use pre-stored RC strand from nucdb directly (skip runtime `esl_sq_ReverseComplement`)
