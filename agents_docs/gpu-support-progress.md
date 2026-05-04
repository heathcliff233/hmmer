# GPU Support Progress

Last updated: 2026-05-05

## Objective

Implement an opt-in pure GPU MSV path for protein `hmmsearch --gpu`, without CPU fallback for the requested GPU mode, and validate it with correctness and timing benchmarks.

## Current State

- TODO exists in `agents_docs/gpu-support-todo.md`.
- An opt-in CUDA MSV implementation is present for protein `hmmsearch --gpu` on protein `dsqdata` target input built with `hmmseqdb`.
- `origin/cuda` remains historical reference material only.
- Build detection is wired through `configure --enable-cuda` and `--with-cuda-arch=sm_XX`.
- `nvcc` and CUDA runtime libraries are present in the environment.
- `hmmseqdb` builds existing Easel protein `dsqdata`; `hmmsearch --gpu` rejects ordinary FASTA target input.
- The Easel `dsqdata` chunk-sizing change is applied at build time from `patches/easel-dsqdata-open-sized.patch`, not edited in place in the submodule.

## Required Deliverables

1. CUDA runtime and MSV module under `src/`.
2. `hmmsearch --gpu` opt-in CLI path.
3. No fallback to CPU when GPU mode is explicitly requested.
4. Benchmark and timing check proving GPU speedup versus CPU MSV on the same fixture set.
5. Progress updates recorded here as work lands.

## Verification Targets

- CUDA MSV parity against CPU `p7_MSVFilter()`.
- Overflow/high-score parity.
- End-to-end `hmmsearch --gpu` versus CPU `hmmsearch`.
- Timing comparison that records wall-clock and GPU kernel time.
- Error handling for unsupported builds/devices/databases/options.

## Timing Requirement

GPU work must not be treated as complete until timing evidence shows a real speedup on at least one representative benchmark fixture.

Minimum timing evidence to record:

- CPU MSV wall time.
- GPU MSV wall time.
- GPU kernel elapsed time.
- Host/device transfer time.
- Batch size used.

## Work Log

- 2026-05-04: Reviewed TODO, build hooks, `hmmsearch` pipeline, and historical `origin/cuda` branch.
- 2026-05-04: Added this progress tracker.
- 2026-05-04: Verified CUDA toolchain availability and identified `test-speed/` as the local timing harness family to extend.
- 2026-05-04: Implemented CUDA MSV module, `hmmsearch --gpu`, post-MSV pipeline split, CUDA timing output, and focused benchmark hooks.
- 2026-05-04: `src/impl_sse/msvfilter_benchmark --gpu -N100000 -L400 tutorial/globins4.hmm` showed CPU MSV 54,181.8 Mc/s and CUDA kernel 220,927.2 Mc/s. End-to-end `hmmsearch --gpu` remains slower on small/no-hit FASTA because sequence parsing, CUDA startup, and downstream CPU work dominate.
- 2026-05-04: Fixed archive rebuild rules so switching between CUDA and non-CUDA configure modes rebuilds `libhmmer.a` with the configured CUDA object or stub.
- 2026-05-04: Non-CUDA build check passed: `./configure --disable-cuda && make -C src hmmsearch`; explicit `src/hmmsearch --gpu --cpu 0 tutorial/globins4.hmm tutorial/globins45.fa` exits with `HMMER was built without CUDA support`.
- 2026-05-04: CUDA build check passed: `./configure --enable-cuda --with-cuda-arch=sm_89`, `make -C src hmmsearch generic_msv_utest`, and `make -C src/impl_sse msvfilter_utest msvfilter_benchmark`.
- 2026-05-04: Correctness checks passed: CPU and GPU `hmmsearch --cpu 0` outputs matched on `tutorial/globins4.hmm` versus `tutorial/globins45.fa` after stripping comments/status; `src/generic_msv_utest` and `src/impl_sse/msvfilter_utest` passed.
- 2026-05-04: Clean MSV timing check: `src/impl_sse/msvfilter_benchmark --gpu -N100000 -L400 tutorial/globins4.hmm` reported CPU MSV wall time 0.165818 sec, CPU MSV 35,943.0 Mc/s, CUDA MSV wall time 0.044423 sec, CUDA H2D 0.009416 sec, CUDA kernel 0.026952 sec, CUDA D2H 0.002627 sec, CUDA kernel 221,131.0 Mc/s. This is a real device-kernel speedup on the same generated sequence block.
- 2026-05-04: End-to-end timing harness `test-speed/x-hmmsearch-gpu-msv . tutorial/globins4.hmm tutorial/globins45.fa benchmark-data/gpu-msv/logs/hmmsearch-gpu-msv-timing-final.txt` recorded CPU wall 0.020620 sec, GPU wall 0.341298 sec, GPU H2D 0.000022 sec, GPU kernel 0.000214 sec, GPU D2H 0.000028 sec, 45 sequences, 6,519 residues, and end-to-end speedup 0.060417 on the tiny tutorial input.
- 2026-05-04: User requested real `benchmark-data/profmark-current` benchmarking because tutorial timing is too small. Verified local profmark data: `pmark.test.fa` has 229,290 sequences and 97,077,299 residues; `pmark.tbl` has 4,714 `ok` query families.
- 2026-05-04: Initial 10-family profmark subset on full `pmark.test.fa` showed GPU was not complete: GPU was slower end-to-end on shorter profiles and lost positives for `4TM_phosphoesterase`, `7tm_1`, and `AAA_15`.
- 2026-05-04: Added per-sequence CUDA MSV length parameters, larger configurable GPU batches (`--gpu-batch-seqs`), and finite GPU MSV survivor slack (`--gpu-msv-slack`) so the GPU path can preserve sensitivity without forcing every MSV survivor through all later CPU filters.
- 2026-05-04: Profmark sensitivity check: with `--gpu-msv-slack 2`, long-profile families `ATG2_CAD`, `Tra1_ring`, and `Nup192` had zero CPU-only reported hits versus CPU `hmmsearch`; `ATG2_CAD` and `Tra1_ring` had zero GPU-only hits, and `Nup192` had zero GPU-only hits.
- 2026-05-04: Profmark timing check on full `pmark.test.fa` with `--gpu-batch-seqs 8192 --gpu-msv-slack 2`: `ATG2_CAD` CPU 6.21 sec, GPU 6.88 sec, CUDA kernel 0.955711 sec; `Tra1_ring` CPU 4.53 sec, GPU 4.97 sec, CUDA kernel 0.779093 sec; `Nup192` CPU 5.02 sec, GPU 4.75 sec, CUDA kernel 0.784063 sec. `Nup192` is a real end-to-end speedup on the profmark dataset while preserving CPU sensitivity.
- 2026-05-04: Added reusable profmark timing harness `test-speed/x-hmmsearch-gpu-profmark`. Harness smoke run: `test-speed/x-hmmsearch-gpu-profmark . benchmark-data/profmark-current/work benchmark-data/profmark-current/gpu-msv/harness Nup192 --gpu-batch-seqs 8192 --gpu-msv-slack 2` recorded CPU 4.590 sec, GPU 4.870 sec, speedup 0.943, zero CPU-only hits, zero GPU-only hits, H2D 0.015330 sec, kernel 0.783702 sec, D2H 0.001272 sec.
- 2026-05-04: Added `hmmseqdb <seqfile> <seqdb>` and switched `hmmsearch --gpu` to require protein `dsqdata`; ordinary FASTA now fails explicitly with a missing dsqdata index diagnostic. The CUDA path reads `ESL_DSQDATA_CHUNK` directly and preserves target metadata for CPU post-MSV stages.
- 2026-05-04: Added `esl_dsqdata_OpenSized()` so GPU search can request chunk sizing before loader threads start. For safety, CLI `--gpu-batch-res` is currently capped at 1,572,864 residues, matching the existing dsqdata guaranteed large-sequence/chunk bound; an attempted 4,000,000-residue chunk exposed a `free(): invalid pointer` path and is intentionally rejected until the reader path is hardened.
- 2026-05-04: Dsqdata smoke checks passed: `src/hmmseqdb tutorial/globins45.fa benchmark-data/gpu-msv/tutorial-dsq`; `src/hmmsearch --gpu --cpu 0 ... benchmark-data/gpu-msv/tutorial-dsq`; CPU/GPU reported hit names matched. Raw FASTA under `--gpu` rejected as expected.
- 2026-05-04: Stable dsqdata profmark long-profile run on full `pmark.test.gpudb` with `--gpu-batch-seqs 8192 --gpu-batch-res 1572864 --gpu-msv-slack 2`: `ATG2_CAD` CPU 6.190 sec, GPU 8.240 sec, CPU-only 0, GPU-only 0, H2D 0.025552 sec, kernel 1.600307 sec, D2H 0.001135 sec; `Tra1_ring` CPU 4.510 sec, GPU 6.140 sec, CPU-only 0, GPU-only 2, H2D 0.021998 sec, kernel 1.293523 sec, D2H 0.001092 sec; `Nup192` CPU 4.590 sec, GPU 6.050 sec, CPU-only 0, GPU-only 0, H2D 0.021564 sec, kernel 1.272845 sec, D2H 0.001040 sec. This preserves CPU sensitivity but does not show end-to-end speedup on the dsqdata path yet.
- 2026-05-04: Current direct MSV timing after dsqdata changes: `src/impl_sse/msvfilter_benchmark --gpu -N100000 -L400 tutorial/globins4.hmm` reported CPU MSV wall time 0.105566 sec, CUDA MSV wall time 0.049826 sec, H2D 0.005057 sec, kernel 0.039756 sec, D2H 0.001127 sec, CUDA kernel 149,913.1 Mc/s. Device MSV still speeds up; end-to-end dsqdata search is dominated by reader/unpack and downstream CPU survivor costs.
- 2026-05-04: Verification after dsqdata integration passed: `make -C easel libeasel.a`, `make -C src hmmsearch hmmseqdb`, `make -C easel esl_dsqdata_utest`, `make -C src generic_msv_utest`, `make -C src/impl_sse msvfilter_utest msvfilter_benchmark`, `easel/esl_dsqdata_utest`, `src/generic_msv_utest`, and `src/impl_sse/msvfilter_utest`.
- 2026-05-05: Updated repository guidance to reflect the current GPU state: `hmmsearch --gpu` requires `hmmseqdb`/protein `dsqdata`, Easel is patched at build time, and profmark remains the benchmark for GPU claims.

## Remaining Scope

- TODO validation categories for a true `dsqdata` v2 length-index extension remain open; current implementation uses existing Easel dsqdata and per-sequence lengths after chunk unpacking.
- Current `hmmsearch --gpu` uses dsqdata input with GPU replacing MSV only; bias, Viterbi, Forward/Backward, domain definition, null2, hit storage, thresholding, and output remain on CPU as intended for the first MSV milestone.
- GPU is not universally faster. Direct MSV benchmarks show real device speedup, but stable end-to-end dsqdata profmark runs are slower so far because dsqdata read/unpack and downstream CPU work dominate. More work is needed before claiming the dsqdata `hmmsearch --gpu` path is complete by the end-to-end timing criterion.
