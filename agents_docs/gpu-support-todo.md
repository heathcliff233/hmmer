# GPU Support TODO

This is the live TODO for future GPU work. For detailed dated implementation history, see `gpu-support-history.md`; for external CUDA-HMM reference notes, see `cuda-hmm-reference.md`.

## Locked Constraints

- Keep GPU support native to the existing C/autotools build. Do not add CMake, vendor external CUDA-HMM code, link external libraries, or treat external projects as dependencies.
- Keep GPU code in the additive `src/cuda/` subsystem. Do not turn it into an `impl_*` backend or replace the selected SSE/NEON/VMX CPU implementation.
- Keep `hmmsearch --gpu` protein-only for now. Leave `phmmer`, `jackhmmer`, `hmmscan`, daemon/cache paths, and nucleotide programs for later milestones.
- Do not change pressed HMM database files (`.h3m/.h3i/.h3f/.h3p`) as part of current GPU work.
- Do not silently fall back to CPU when the user explicitly requests `--gpu`; fail with a direct diagnostic.

## Current Design Boundaries

- HMMER C side owns CLI options, validation, pipeline accounting, hit reporting, and CPU continuation.
- `src/cuda/` owns CUDA device memory, profile upload, sequence batch upload/reuse, kernels, runtime stats, and CUDA error translation; keep stage ownership aligned with `p7_cuda_msv.cu`, `p7_cuda_bias.cu`, `p7_cuda_viterbi.cu`, `p7_cuda_forward.cu`, and `p7_cuda_fb_parser.cu`, with shared lifecycle/runtime code in `p7_cuda_runtime.cu`.
- `src/hmmsearch_gpu.c` owns the `hmmsearch --gpu` serial dsqdata loop, GPU batch packing, survivor staging, and debug compare diagnostics.
- `src/cuda_msv.h` exists only for source compatibility. Prefer `src/cuda/p7_cuda.h` for new CUDA-facing code.

## Open Work

### High-impact optimizations (ranked by current benchmark impact)

- **GPU Viterbi kernel optimization** (0.475s, 19.4% of wall): Dominant single GPU cost. Average 544 candidates/launch, ~217K residues/launch. Approaches: occupancy tuning (shmem vs register pressure), warp-level shuffle-based DP (as done for SSV), profile tiling for large M to reduce shmem footprint.

- **Reduce unaccounted residual** (0.334s, 13.6% of wall): The `exact_other` bucket captures inter-stage sync, host-side score conversion, F1 gating logic, and survivor list construction between kernel launches. Approaches:
  - Fuse MSV + null + bias into a single kernel (eliminates 2 sync points per batch)
  - Move score conversion and F1 gating to GPU
  - Overlap survivor loop CPU work with next batch's kernel via CUDA streams

- **Reduce I/O overhead** (0.258s, 10.5% of wall): dsqdata metadata loading (names, accessions) for potential hit reporting, even though only ~15 sequences/query reach the hit stage. Approaches:
  - Lazy-load metadata only for sequences that reach the hit stage
  - Overlap I/O with GPU kernel via double-buffering
  - Eliminate dsqdata read entirely when using gpudb resident path (store metadata in gpudb index)

- **GPU Forward kernel optimization** (0.243s, 9.9% of wall): Same strategies as Viterbi. Additional opportunity: kernel fusion with Backward (both traverse the same sequence), async overlap with CPU post-Fwd work.

- **CPU domain definition** (0.162s, 6.6% of wall): `p7_domaindef_ByPosteriorHeuristics()` on ~15 sequences/query that pass Forward. Approaches: parallelize across survivors with thread pool, or eventually move posterior decoding to GPU (high complexity, deferred).

- **Kernel fusion (MSV → bias)**: MSV and bias are separate launches with host-side sync between them. A single kernel that runs SSV → null → bias → F1 gating would eliminate one round-trip and produce the compact survivor list directly.

### SSV kernel — future optimizations
The optimized SSV kernel (`src/cuda/p7_cuda_ssv.cu`) is now the default GPU MSV path. Remaining kernel-level optimization opportunities:
- **Adaptive thread count**: current kernel uses 32 threads/block regardless of M. For small M (< 64), a single warp is sufficient; for large M (> 512), multiple warps with shared-memory partitioning could improve occupancy.
- **Early termination**: since ~99.7% of sequences complete in the SSV fast-path, explore whether the SSV section can be further optimized (e.g., skip the warp reduction when the local max is clearly below threshold).
- **rbv access pattern optimization**: with contiguous node ownership, explore whether L1 cache prefetching or texture memory for the rbv profile improves memory-bound queries.

### Medium-priority work
- Consider larger batch sizes (>32K seqs) to reduce per-batch CUDA API call count.
- Profile/candidate-shape auto-gating for short queries where CUDA launches regress wall time.

### Completed
- CUDA engine reuse across queries (~3.1s saved)
- GPU-native database format (.gpudb) with mmap-based reader
- Resident database (whole-DB GPU upload, H2D reduced to ~0.02s)
- Batched CPU bias pre-computation → superseded by double-precision GPU survivor kernel
- SSV as default GPU MSV kernel (1.36x faster than monolithic MSV)
- Viterbi M≤2048, Forward M≤1024, M>2048 tiling tier
- Survivor loop optimization (sort-by-length, ReconfigLength caching, CPU MSV fallback removed)
- Pre-allocated CUDA events, bias parameter caching, redundant sync removal
- Multi-query single-process profmark benchmark
- All GPU stages (Viterbi prefilter, Forward prefilter, FB parser) default-on with `--gpu`

### Lower-priority / deferred
- `dsqdata` v2 length-index extension for GPU batch planning without chunk unpacking.
- Broaden parser-state validation beyond raw `p7X_SCALE` row differences.
- CUDA context init reduction (~0.51s): explore `cudaSetDevice` lazy init, persistent context across process invocations, or CUDA MPS.

## Validation Checklist

- Non-CUDA build: `./configure --disable-cuda`, `make -C src hmmsearch generic_msv_utest`, and `hmmsearch --gpu` against dsqdata must fail with “HMMER was built without CUDA support”.
- CUDA build: `./configure --enable-cuda --with-cuda-arch=sm_89`, `make -C src hmmsearch hmmseqdb generic_msv_utest`, and `make -C src/impl_sse msvfilter_utest msvfilter_benchmark`.
- Functional smoke: CPU and GPU `hmmsearch --cpu 0` hit names should match on `tutorial/globins4.hmm` versus `hmmseqdb`-built `tutorial/globins45.fa` dsqdata.
- Profmark validation: record wall time, CUDA timing, batch sizes/counts, pass counts, and `cpu_only`/`gpu_only` deltas with `test-speed/x-hmmsearch-gpu-profmark` and summarize with `test-speed/x-hmmsearch-gpu-profmark-summary`.
- Exact timing validation: in GPU stats output, `Exact delta_vs_wall` must print `[OK]` and stay within `1e-6` seconds.

## Deferred Scope

- GPU `hmmscan`, pressed HMM database GPU indexing, GPU `phmmer`/`jackhmmer`, daemon/cache GPU integration, and nucleotide `nhmmer`/`nhmmscan`.
- GPU null2, domain definition, hit storage, thresholding, or output.
- Rewriting CPU optimized implementations around CUDA abstractions.
