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

### High-impact optimizations
- **CUDA engine reuse across queries** (highest priority): move `p7_cuda_engine_Create` outside the per-query loop so the ~260ms CUDA context init is paid once per process. Saves ~3.1s across 13 queries. The engine struct already supports reuse; barrier is the per-query create/destroy lifecycle in `hmmsearch.c`.
- **CUDA stream-based overlap**: pipeline H2D of next batch with kernel of current batch, reducing the 2.24s host sync/blocking time.

### SSV kernel optimizations (new, `--gpu-ssv` path)
The standalone SSV kernel (`src/cuda/p7_cuda_ssv.cu`) is parity-verified but ~15–20% slower than monolithic MSV due to two-pass overhead. Optimizations to recover and exceed monolithic performance:
- **Eliminate intermediate host round-trip**: instead of copying `d_ssv_status` back to host and building fallback index on CPU, use a device-side compaction kernel (thrust `copy_if` or a custom atomic-based compaction) to build `d_fallback_idx` entirely on GPU. Saves one `cudaMemcpy` + `cudaEventSynchronize` between passes.
- **Fuse SSV classification into the kernel**: the SSV kernel currently writes `raw_sc` and `ssv_status` separately; fuse the score conversion for OK sequences directly into the SSV kernel to avoid a second pass over OK results.
- **Warp-level SSV specialization**: the SSV inner loop has no cross-iteration J-state dependency (constant `xB`), making it amenable to warp-shuffle reductions without shared-memory barriers. Explore replacing `__syncthreads()` with warp-cooperative patterns for M ≤ 512.
- **Adaptive thread count**: current kernel uses 32 threads/block regardless of M. For small M (< 64), a single warp is sufficient; for large M (> 512), multiple warps with shared-memory partitioning could improve occupancy.
- **Skip fallback kernel launch when nfallback=0**: currently the fallback path is gated by `if (nfallback > 0)` but the event create/destroy overhead is always paid. Move event lifecycle inside the conditional.
- **Profile the fallback kernel**: the fallback kernel processes only 0.2–0.3% of sequences but takes 5–11ms. This is disproportionate — investigate whether kernel launch overhead dominates or whether fallback sequences are genuinely harder (longer, more complex).

### Medium-priority work
- Decide default policy for later-stage flags (`--gpu-vit-prefilter`, `--gpu-fwd-prefilter`, `--gpu-fb-parser`). Needs broader validation and auto-gating for short profiles.
- Eliminate multi-chunk view fallback path (~50% of batches still use per-sequence copy because `gpu_pending_max_chunks=2` creates multi-chunk views).
- Consider larger batch sizes (>32K seqs) to reduce per-batch CUDA API call count.

### Lower-priority / deferred
- `dsqdata` v2 length-index extension for GPU batch planning without chunk unpacking.
- Profile/candidate-shape auto-gating for short queries where CUDA launches regress wall time.
- Broaden parser-state validation beyond raw `p7X_SCALE` row differences.
- Once SSV kernel is optimized to match or beat monolithic MSV, make `--gpu-ssv` the default GPU MSV path and remove the monolithic kernel.

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
