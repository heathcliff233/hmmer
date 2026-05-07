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
- [ ] GPU-side FB parser for nhmmer (deferred — requires scanning Viterbi kernel)

## Future Work (Not Planned)

- GPU scanning Viterbi kernel (`p7_ViterbiFilter_longtarget` equivalent)
- Memory-mapped sequence I/O to reduce sys time (~0.4s overhead)
- FM-index GPU path (FM-index remains CPU-only)
- GPU-side bias filter for nhmmer windows
