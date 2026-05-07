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
- [x] CLI flags: `--gpu-batch`, `--gpu-vit-prefilter`, `--gpu-fwd-prefilter`, `--gpu-compare`
- [x] Hit parity verified: MADE1 154=154, query_short 120=120, query_medium 217=217 (batch only)
- [x] Viterbi pre-filter: query_medium 215 (removes 2 boundary hits, matches CPU count)
- [x] Speed: query_medium GPU-4+batch+vit 2.15s vs GPU-4 baseline 3.23s (1.5x improvement)

## Future Work (Not Planned)

- GPU Forward pre-filter for sub-windows (requires splitting p7_pli_postSSV_LongTarget)
- GPU scanning Viterbi kernel (`p7_ViterbiFilter_longtarget` equivalent)
- Memory-mapped sequence I/O to reduce sys time (~0.4s overhead)
- FM-index GPU path (FM-index remains CPU-only)
- GPU-side bias filter for nhmmer windows
- `--gpu-compare` mode implementation (score-level comparison)
