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

- **GPU Viterbi kernel optimization** ~~(0.475s, 19.4% of wall)~~ → **0.162s after template optimization (2.9x kernel speedup)**. The kernel is now a C++ template parameterized on `STRIDE` with `#pragma unroll` on inner loops. For stride≤24 (M≤768, covering typical profmark queries), `nvcc` places `reg_M/D/I` in true registers (79 regs, 48-byte stack frame, zero spills at stride=8). For stride>24, a non-templated fallback retains the old local-memory behavior. Tile sizes increased 4x (4096 candidates/tile for M≤700) with int64 overflow fix in the residue-density heuristic. Remaining approaches:
  - Node-contiguous profile layout (`rwv_node[k * Kp + x]`) for coalesced cross-warp access
  - Preload transition/emission scores into shared memory per residue
  - Kernel fusion (MSV → bias → Vit in single kernel to avoid re-reading profile)

- **Reduce unaccounted residual** ~~(0.334s, 13.6% of wall)~~ → **0.115s (9.3% of wall) after kernel fusion and survivor-indexed D2H**. Remaining sources: Viterbi/Forward batch construction, CUDA event record overhead, host-side survivor list sorting, F1 survivor double-precision bias recompute. Approaches:
  - Fuse F1 survivor double-precision bias into the main fused kernel (avoid separate `cuda_bias_filter_survivors_kernel` launch)
  - Overlap survivor loop CPU work with next batch's kernel via CUDA streams
  - Profile with nsys to identify exact remaining hotspots

- **Reduce I/O overhead** ~~(0.258s, 10.5% of wall)~~: **DONE** — gpudb v2 embeds metadata; dsqdata is skipped entirely on the resident path. `io_read_unpack` dropped from 0.258s to 0.000s.

- **GPU Forward kernel optimization** ~~(0.243s, 9.9% of wall)~~: The prefix kernel (parallel prefix scan for DD recurrence) was extended from T≤512 to T≤1024, covering models up to M≈2044. Models with M≥1024 previously fell back to a 4-lane serial kernel (4/32 threads active); now they use the efficient prefix kernel with full warp utilization. For M=1500: 525ms → 42ms (**12.5x speedup**). For profmark models (M≤313), already used prefix kernel — no change. Remaining: kernel fusion with Backward, async overlap with CPU post-Fwd work.

- **CPU domain definition** (0.162s, 6.6% of wall): `p7_domaindef_ByPosteriorHeuristics()` on ~15 sequences/query that pass Forward. Approaches: parallelize across survivors with thread pool, or eventually move posterior decoding to GPU (high complexity, deferred).

- **Kernel fusion (MSV → bias)**: ~~MSV and bias are separate launches with host-side sync between them. A single kernel that runs SSV → null → bias → F1 gating would eliminate one round-trip and produce the compact survivor list directly.~~ **DONE** — fused into `cuda_ssv_null_bias_gate_kernel<STRIDE>`. Inter-stage overhead reduced 53% (246ms → 115ms). D2H reduced 10.8x via survivor-indexed output.

### SSV/Fused kernel — future optimizations
The fused SSV+null+bias+gate kernel (`src/cuda/p7_cuda_ssv.cu`) is now the default GPU MSV path. Remaining kernel-level optimization opportunities:
- **Early termination**: since ~99.7% of sequences complete in the SSV fast-path, explore whether the SSV section can be further optimized (e.g., skip the warp reduction when the local max is clearly below threshold).
- **Double-precision bias fusion**: integrate the survivor double-precision bias recompute directly into the fused kernel, eliminating the separate `cuda_bias_filter_survivors_kernel` launch.
- **In-kernel null/bias precision**: current in-kernel bias uses float32 `expf`/`logf` which diverges slightly from the double-precision survivor recompute. Explore whether the survivor bias kernel can be eliminated entirely by computing exact-enough bias in the fused kernel.

### Medium-priority work
- Consider larger batch sizes (>32K seqs) to reduce per-batch CUDA API call count.
- Profile/candidate-shape auto-gating for short queries where CUDA launches regress wall time.

### Completed
- CUDA engine reuse across queries (~3.1s saved)
- GPU-native database format (.gpudb) with mmap-based reader
- Resident database (whole-DB GPU upload, H2D reduced to ~0.02s)
- Batched CPU bias pre-computation → superseded by double-precision GPU survivor kernel
- SSV as default GPU MSV kernel (1.36x faster than monolithic MSV)
- Viterbi M≤2048, Forward M≤2044, M>2048 tiling tier
- Survivor loop optimization (sort-by-length, ReconfigLength caching, CPU MSV fallback removed)
- Pre-allocated CUDA events, bias parameter caching, redundant sync removal
- Multi-query single-process profmark benchmark
- All GPU stages (Viterbi prefilter, Forward prefilter, FB parser) default-on with `--gpu`
- gpudb v2 embedded metadata — eliminates dsqdata I/O entirely on resident path (0.258s → 0.000s)
- madvise hints + cudaHostRegister for GPU upload DMA acceleration
- Viterbi register-based warp-shuffle kernel (32 threads/seq, cleaner architecture, parity-verified)
- Viterbi templated kernel — stride as compile-time constant for true register residence (2.9x kernel speedup, 0.475s → 0.162s)
- Viterbi tile sizes 4x increase (4096 candidates/tile for M≤700) + int64 overflow fix in tile heuristic
- Fused SSV+null+bias+F1 gate kernel (`cuda_ssv_null_bias_gate_kernel<STRIDE>`) — single launch replaces 5 kernels, inter-stage overhead −53%
- Linear rbv layout (`d_rbv_lin[x * M + k]`) for coalesced fused-kernel inner-loop access
- Survivor-indexed D2H — transfers only nsurv entries (~32KB) instead of full nseq arrays (1.8MB); D2H 28ms → 2.6ms
- Forward prefix kernel extended to T≤1024 (M≤2044) with dynamic shared memory — large-model Forward 12.5x faster (525ms → 42ms for M=1500)
- Multi-warp-per-block for fused SSV and Viterbi opt kernels (templated on `(STRIDE, WARPS)`, one sequence per warp; W auto-tuned from `cudaGetDeviceProperties` with tie-break toward W=4 on sm_89; hidden `--gpu-ssv-warps` / `--gpu-vit-warps` for override). Infrastructure change — kernel time within noise of W=1 baseline on the current 13-query profmark because the warp-shuffle DP is already compute-saturated at W=1.

### Lower-priority / deferred
- `dsqdata` v2 length-index extension for GPU batch planning without chunk unpacking.
- Broaden parser-state validation beyond raw `p7X_SCALE` row differences.
- CUDA context init reduction (~100–200ms one-time): explore `cudaSetDevice` lazy init, persistent context across process invocations, or CUDA MPS.
- GPU profile buffer reuse across queries was tried and reverted — `cudaMalloc` is sub-ms, the real per-query cost is `cudaMemcpy` which is unavoidable when the model changes. See history entry #16.

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
