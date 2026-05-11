# Agent Guide

HMMER is biological sequence analysis software. It searches sequence databases for homologs: protein or DNA/RNA sequences related by common ancestry, often with shared structure or function even when raw sequence identity is low.

HMMER's central method is the profile hidden Markov model, or profile HMM: a probabilistic model of a sequence family. A profile HMM is usually built from a multiple sequence alignment (MSA) that captures conserved positions, variable positions, and insertion/deletion patterns. HMMER can also build a profile from one sequence for pairwise-style searches.

This is a coupled C/autotools HMMER3 worktree. Treat it as one project with three build-time components:

- `src/`: HMMER library, command-line programs, optimized DP implementations, daemon/cache code, FM-index support, tests, examples, and benchmarks.
- `easel/`: required sibling library for sequence/alignment I/O, alphabets, random utilities, work queues, threads, tests, and autoconf macros.
- `libdivsufsort/`: suffix-array support used by `makehmmerdb` and FM-index target databases.

Easel is not a leaf dependency. HMMER uses it for core runtime objects (`ESL_MSA`, `ESL_SQ`, `ESL_ALPHABET`), command-line parsing, error/memory conventions, statistics/randomization, vector helpers, threading/work queues, MPI helpers, miniapps, tests, documentation tooling, and autoconf macros.

## Common Workflows

- Build a profile HMM from an alignment with `hmmbuild`.
- Search a sequence database for homologs with `hmmsearch`, `phmmer`, or `jackhmmer`.
- Scan sequences against a pressed profile database with `hmmscan`.
- Search long DNA/RNA targets with `nhmmer`, or scan nucleotide profile databases with `nhmmscan`.
- Prepare profile databases with `hmmpress`, and FM-index nucleotide target databases with `makehmmerdb`.

Search results are reported as hits and domains with bit scores, E-values, coordinates, alignments, and per-stage pipeline statistics. Much of the code exists to make profile HMM searches fast while preserving their statistical behavior.

## Build Quick Reference

CPU-only development build:

```sh
./configure --enable-threads
make -j$(nproc)
make check
```

CUDA development build:

```sh
./configure --enable-cuda --with-cuda-arch=sm_89 --enable-threads
make -j$(nproc)
```

Important targets:

- `make`: builds Easel, `libdivsufsort`, HMMER programs, and profmark.
- `make dev`: also builds test drivers, benchmarks, examples, and extra development tools.
- `make tests`: compiles test drivers.
- `make check`: runs Easel and HMMER test suites.
- `make -C src hmmsearch nhmmer hmmseqdb hmmnucdb`: rebuilds the main GPU-facing programs.
- `make -C src/impl_sse msvfilter_utest`: rebuilds a selected optimized implementation unit test.

## GPU Status Snapshot

- The current `hmmsearch --gpu` path is protein-only, opt-in, and requires an `hmmseqdb`-built protein target database.
- `hmmseqdb` also writes a `.gpudb` GPU-native mmap database. Current `.gpudb` v2 embeds sequence metadata (name, accession, description, taxid), so `hmmsearch --gpu` can skip `dsqdata` per-query I/O entirely when metadata is present.
- The current `nhmmer --gpu` path is nucleotide long-target search, opt-in, and works on ordinary FASTA or a prebuilt `.nucdb` from `hmmnucdb`. `.nucdb` is mmap-based, pre-chunked, stores both strands by default, and `hmmnucdb` now defaults to overlap chunking (`--overlap 2001`) so typical GPU runs use the GPU-resident SSV path. Use `hmmnucdb --overlap 0` only for an ordinary no-overlap diagnostic database.
- The repository applies an Easel `dsqdata` chunk-sizing patch at build time from `patches/easel-dsqdata-open-sized.patch`; do not edit the submodule in place for this work.
- Keep GPU work in the existing autotools build. Do not add CMake to this repository for external CUDA-HMM references; those projects are design references only. Borrow high-level implementation ideas where useful, but do not copy code, vendor sources, link external libraries, or add external build support.
- In this configured workspace, `nvcc` is `/usr/local/cuda/bin/nvcc`; exporting `CUDA_HOME=/usr/local/cuda` is useful for reproducibility but the generated Makefile already records the absolute compiler path.
- GPU code lives in `src/cuda/`, with public declarations in `src/cuda/p7_cuda.h`. `src/cuda_msv.h` is a compatibility wrapper only; include `src/cuda/p7_cuda.h` for new CUDA-facing work. Non-CUDA builds use `src/cuda/p7_cuda_stub.c`; explicit `--gpu` must fail with a direct "HMMER was built without CUDA support" diagnostic rather than silently falling back.
- `hmmsearch --gpu` is serial on the CPU side (`--cpu 0` required) and protein-only. It accelerates fused SSV/MSV + null + bias + F1 gate, Viterbi prefilter, Forward prefilter, and Forward/Backward parser by default. Domain definition, null2, hit storage, thresholding, and output remain CPU-side.
- The default protein prefilter is `cuda_ssv_null_bias_gate_kernel<STRIDE, WARPS>` in `src/cuda/p7_cuda_ssv.cu`: one sequence per warp, fused SSV/null/bias/F1 gating, in-kernel MSV fallback for rare SSV misses, survivor compaction via `atomicAdd`, and survivor-indexed D2H transfers. The legacy separate-kernel path is only for diagnostics via `--gpu-legacy-pipeline`/compare flags.
- Exact protein hit parity currently depends on GPU-side double-precision survivor bias recomputation (`p7_cuda_BiasFilterSurvivors`), not CPU `p7_bg_FilterScore()`/`p7_MSVFilter()` rescue. Do not reintroduce CPU fallback at the bias/F1 boundary without evidence that it is needed.
- `hmmsearch --gpu` separates dsqdata load sizing (`--gpu-load-seqs`, `--gpu-load-res`) from CUDA search-batch sizing (`--gpu-batch-seqs`, `--gpu-batch-res`) and can pack multiple loaded chunks into one CUDA batch. Defaults remain 32,768 sequences / 8M residues for both load and search.
- Protein GPU resident mode uploads the database once with `p7_cuda_engine_UploadDatabase()` when memory allows; otherwise it streams batches. Engine reuse creates the CUDA engine once before the query loop and resets stats per query, so multi-query profmark runs amortize CUDA initialization.
- Current protein profmark snapshot: all-13 multi-query single-process run, resident `.gpudb` v2, all GPU stages on, exact hit parity (`cpu_only=0`, `gpu_only=0`), roughly 1.27s GPU wall versus ~9.3s CPU-1 and ~4.8s CPU-4 on the documented RTX 4090 setup. Treat `benchmark-data/profmark-current` as the real protein benchmark, not tutorial-sized inputs.
- `nhmmer --gpu` currently enables GPU SSV long-target scanning, GPU SSV-window extend/merge, nucleotide-specific GPU batch MSV/null/bias/F1 gating, GPU scanning Viterbi, GPU Viterbi-seed extend/merge/split, a GPU Forward prefilter for post-Viterbi survivor reduction, GPU Forward/Backward parser matrix reuse, and threaded CPU domain/hit processing by default. In the default `.nucdb` overlap path, post-SSV merged F1 windows are mapped back to the resident `.nucdb` device chunks; chunk-contained windows use resident offsets directly, and boundary-spanning windows use a small device gather into the active F1 batch, so batch F1 no longer uploads packed host sequence bytes. In the default biased path, scanning Viterbi consumes the device-resident F1 packed batch and survivor metadata directly, so the F1-to-Viterbi handoff no longer repacks or reuploads survivor sequence bytes and no longer materializes a separate Viterbi offsets/lengths array before scanning. The normal path no longer calls CPU `p7_pli_ExtendAndMergeWindows()` between SSV/F1 or Viterbi/parser; `--gpu-compare` still keeps the old CPU merge path for diagnostics. On FASTA and `.nucdb` inputs the Forward-to-Backward parser handoff keeps the full Forward xmx on device, computes the exact F3 survivor gate on GPU, skips the all-window Forward xmx D2H transfer, computes survivor xmx offsets on GPU, compacts survivor xmx on GPU, and runs Backward from device-resident survivor indices/offsets without re-uploading Forward xmx or survivor metadata. Because GPU F3 already enforced the gate, the post-parser CPU worker uses the no-F3 post-Fwd/Bwd path and no longer recomputes CPU null/bias F3 scores. On resident overlap `.nucdb`, the parser batch now describes windows with metadata only, gathers parser sequence bytes on GPU, and delays F3 survivor-index and compact Forward-xmx downloads until after Backward has consumed the device buffers. The default resident overlap `.nucdb` path no longer reconstructs full forward/reverse host `ESL_SQ` strands; it creates metadata-only sequence shells and materializes only survivor-window slices from mapped `.nucdb` chunks for the CPU domain workflow. Diagnostic fallback paths that still dereference `sq->dsq` keep the old full reconstruction. Domain definition/null2/hit reporting remain CPU-side. There is no separate single-score GPU Viterbi prefilter in the nucleotide path; scanning Viterbi is the CPU-equivalent Viterbi boundary. `--gpu-fwd-prefilter` is retained as a deprecated no-op compatibility flag; use hidden `--gpu-no-fwd-prefilter` only for diagnostics against the older conservative CPU Forward/Backward continuation.
- A nucleotide `p7_cuda_DomainRescoreBatch()` API exists for cross-window domain/trim batches, but the accepted default `nhmmer --gpu` path still keeps domain definition, null2, hit storage, thresholding, and output CPU-side after the GPU parser handoff. Current chr22 smoke evidence is hit-count clean, while exact envelope coordinates can show the same 1 bp jitter already seen between repeated default GPU runs.
- Current nucleotide benchmark snapshot: chr22/hg38 target, RTX 4090, 16 CPU threads. After making GPU SSV longtarget emit deterministic chunk-ordered windows, compacting SSV windows on GPU for one ordered D2H transfer, moving SSV-window extend/merge to GPU using the CPU MAXL prefix/suffix tables, removing the CPU SSV-window sort before the first long-target extend/merge, moving nucleotide batch F1 gating to a B1-scaled CUDA gate that skips unused survivor-status D2H in production and emits ordered survivors through a GPU pass-mask compaction step, making GPU scanning-Viterbi emit per-window ordered seed slots and compact them on GPU so the normal path no longer CPU-sorts raw Viterbi seeds, moving Viterbi seed extend/merge/max-window split to GPU, removing the extra single-score GPU Viterbi prefilter, changing GPU CPU-continuation workers from static equal-window slices to dynamic survivor-window scheduling, remapping scanning Viterbi to four 8-lane nucleotide DP groups per physical warp, reusing engine-owned Viterbi long-target CUDA streams, replacing the Viterbi device-wide sync with an event sync, making default scanning Viterbi read F1-packed device windows directly, making overlap `.nucdb` F1 use resident device chunks plus device gather for chunk-spanning windows, making overlap `.nucdb` parser batches gather resident sequence bytes on GPU instead of packing host subsequences, keeping FASTA and `.nucdb` Forward parser xmx resident through GPU F3 gating/survivor offset construction/survivor compaction/Backward, skipping the redundant post-parser CPU F3 null/bias recomputation after the GPU F3 gate, delaying F3 survivor-index and compact Forward-xmx D2H until after GPU Backward consumes the device buffers, removing the host synchronization between GPU Forward and Backward parser kernels, removing avoidable parser/SSV/F1 handoff `cudaDeviceSynchronize()` barriers and the redundant compact-xmx D2D copy before Backward, using persistent host scratch for F1 survivor indices/bias/metadata, parser indices/x-offsets/F3 survivor source indices, and resident `.nucdb` chunk/window metadata, reusing metadata scratch in Viterbi, reusing engine-owned host result scratch for SSV/Viterbi long-target window downloads, filling GPU-emitted window lists directly, and replacing full `.nucdb` host-strand reconstruction with metadata shells plus per-survivor CPU-domain slices, default overlap `.nucdb` is hit-count clean versus CPU in the smoke script (`MADE1`, `query_short`, `query_medium`: 4/4 parity checks passed with 16 threads). `test-speed/x-nhmmer-gpu-bench` now defaults to CPU/GPU `--cpu 16` and a combined multi-query run over all three samples so CUDA engine creation and resident `.nucdb` upload are amortized. Latest audit rerun on 2026-05-11: CPU-1 9.962s / 1476 hits, CPU-16 1.111s / 1476 hits, GPU-16 FASTA 1.955s / 1476 hits, GPU-16 no-overlap `.nucdb` 1.371s / 1476 hits, GPU-16 overlap `.nucdb` 1.196s / 1476 hits. On the 5x chr22 target, the full-reconstruction fix changed the GPU overlap `.nucdb` reconstruction bucket from about `0.295s` to `0.000s`; a clean standalone rerun measured GPU-16 `4.348s` / 6294 hits versus CPU-16 `4.893s` / 6294 hits, with GPU internal process elapsed `4.260s`, GPU loop wall `3.801s`, process outside search `0.459s`, CUDA engine create `0.274s`, and shared `.nucdb` upload `0.119s`. The benchmark remains volatile; do not read one run as a stable wall-time win by itself. The remaining default GPU path gap is GPU kernel time, parser matrix D2H, and the allowed CPU domain workflow. Per-query mode remains available with `NHMMER_GPU_BENCH_MODE=per-query`; `NHMMER_GPU_BENCH_CPU=N` overrides the default thread count.
- GPU timing output now includes search-stage timing plus outside-search accounting. For `nhmmer --gpu`, stderr reports GPU search stages, GPU loop wall, `.nucdb` reconstruction, pre-search setup, post-loop cleanup, post-search/report time, query outside-search time, query elapsed, CUDA engine creation/reset/destroy, shared `.nucdb` open/upload, process outside-search time, and total process elapsed. SSV longtarget and scanning Viterbi also report launch shape, dynamic shared memory, theoretical occupancy, grid/SM coverage, and device-active utilization rate for their GPU processing wall buckets. CUDA event timings remain device-side sub-buckets. CPU pipeline `Stage *` totals are still summed across worker threads, but `HMMER_NHMMER_CPU_WALL_TRACE=1` prints a max-across-worker CPU wall-stage estimate for CPU-only threaded `nhmmer`.
- GPU batch/F1 and FB parser timing now report host prep where applicable, H2D bytes/event time, kernel event time, and D2H event time. Current focused query_medium evidence showed overlap `.nucdb` batch F1 sequence upload at `0 bytes` after resident chunk/gather mapping, parser sequence upload at `0 bytes`, parser kernels around `0.015s`, and parser matrix D2H around `0.003s`. The remaining parser optimization target is reducing parser kernel/matrix-D2H cost or moving more domain workflow off CPU, not uploading reconstructed full strands.

## Core Terms

- `homolog`: a sequence related to another by common ancestry.
- `domain`: a conserved sequence region, often one part of a larger protein or DNA sequence.
- `MSA`: multiple sequence alignment; the usual training input for a profile HMM.
- `profile HMM`: probabilistic sequence-family model with match, insert, and delete behavior.
- `Plan7`: HMMER's profile HMM architecture; many core types use the `P7_` prefix.
- `profile`: a scoring form of an HMM configured for a search context.
- `optimized profile`: vectorized scoring form used by accelerated filters and pressed databases.
- `hit`: a reported match between a query/model and a target sequence/model.
- `E-value`: expected number of false positives at a score threshold; lower is more significant.
- `bit score`: log-odds score in bits; higher is better.
- `MSV/SSV`, `Viterbi`, `Forward/Backward`, `null2`: major filter/scoring/correction stages in the search pipeline.
- `pressed HMM database`: binary `.h3m/.h3i/.h3f/.h3p` files produced by `hmmpress`.
- `FM-index`: nucleotide target database format produced by `makehmmerdb` for accelerated `nhmmer` searches.

## Internal Architecture

Most CLI programs are thin orchestration layers around shared model construction, HMM/profile I/O, accelerated filtering/DP, hit reporting, and output code.

`src/hmmer.h` is the central API map. It is intentionally broad because model construction, optimized profiles, filters, DP, hit reporting, daemon code, and FM-index support share core structures. Do not introduce private abstractions that pretend these subsystems are independent.

The main object flow is:

1. Easel reads inputs as `ESL_MSA`, `ESL_SQ`, `ESL_SQFILE`, `ESL_MSAFILE`, and `ESL_ALPHABET`.
2. `P7_BUILDER` constructs models from MSAs or single sequences.
3. `P7_HMM` stores the core Plan7 model.
4. `P7_PROFILE` and `P7_OPROFILE` configure generic and optimized scoring profiles.
5. `P7_PIPELINE` coordinates filters, DP stages, thresholds, counters, and reporting mode.
6. `P7_DOMAINDEF` defines domains/envelopes and alignment displays.
7. `P7_TOPHITS` stores, thresholds, sorts, merges, de-duplicates, serializes, and outputs hits.

Key structures to recognize early:

- `P7_HMM`, `P7_PROFILE`, `P7_OPROFILE`: model and scoring-profile layers.
- `P7_BG`, `P7_TRACE`, `P7_GMX`, `P7_OMX`: background model, tracebacks, and generic/optimized DP matrices.
- `P7_HMMFILE`: HMM text/binary/pressed database reader.
- `P7_PIPELINE`, `P7_DOMAINDEF`, `P7_TOPHITS`: search workflow, domain definition, and hit reporting.
- `P7_SCOREDATA`, `P7_HMM_WINDOWLIST`: nucleotide long-target and FM-index windowing support.

Major execution families:

- Model build and I/O: `hmmbuild`, HMM readers/writers, `hmmpress`, `hmmfetch`, `hmmconvert`.
- Protein search: `hmmsearch`, `phmmer`, `jackhmmer`, and `hmmscan`. GPU acceleration is currently in `hmmsearch --gpu` only.
- Nucleotide search: `nhmmer` and `nhmmscan`, using long-target window logic and nucleotide-specific E-values. GPU acceleration is currently in `nhmmer --gpu` only.
- FM-index databases: `makehmmerdb` plus `libdivsufsort` and FM-index search helpers.
- Daemon/cache paths: `hmmpgmd`, `hmmpgmd_shard`, cached sequence/HMM data, and command/status serialization.
- Tests/docs/benchmarks: `testsuite/`, `documentation/`, `tutorial/`, `profmark/`, `test-speed/`, and `autobuild/`.

## Code Conventions

- Follow Easel/HMMER error and allocation style: `eslOK` returns, `ESL_ALLOC`, `ESL_XFAIL`, and `ERROR:` cleanup labels.
- Use `ESL_OPTIONS` tables for CLI parsing; do not hand-roll option parsing.
- Use `P7_`/`p7_` prefixes for HMMER types and functions, and `ESL_`/`esl_` prefixes for Easel.
- Unit tests are commonly compiled from source files via `p7*_TESTDRIVE` defines in `src/Makefile.in`.
- CUDA support is additive. Do not turn it into an `impl_cuda` replacement for `src/impl_sse`, `src/impl_neon`, or `src/impl_vmx`.
- Do not change pressed HMM database files (`.h3m/.h3i/.h3f/.h3p`) as part of GPU work unless that is explicitly the task.

## Task Index

Use `agents_docs/` only for files relevant to the task:

- `agents_docs/architecture.md`: repository layout, central API shape, dependency boundaries.
- `agents_docs/config-and-optimized-impl.md`: autotools feature gates, generated files, active SIMD implementation, optimized DP layout.
- `agents_docs/data-flow.md`: shared model/search/scan data flow.
- `agents_docs/easel-integration.md`: Easel library layout and the concrete ways HMMER depends on Easel.
- `agents_docs/program-families.md`: CLI ownership and execution families.
- `agents_docs/search-pipelines.md`: search pipeline, domain definition, E-values, scan mode.
- `agents_docs/model-build-and-io.md`: `hmmbuild`, single-sequence model construction, HMM files, pressed databases.
- `agents_docs/nucleotide-and-fm-index.md`: `nhmmer`, `nhmmscan`, FM-index databases, nucleotide coordinate risks.
- `agents_docs/parallel-daemon-cache.md`: threads, MPI, daemon programs, sequence/HMM caches.
- `agents_docs/tests-docs-benchmarks.md`: build targets, tests, docs, benchmarks, local benchmark data.
- `agents_docs/gpu-support-progress.md`: current protein `hmmsearch --gpu` implementation status, benchmark results, and remaining gaps. Prefer this over older stale GPU notes when docs disagree.
- `agents_docs/gpu-support-todo.md`: protein GPU open work, validation checklist, and deferred scope. Some historical/deferred wording may lag current `nhmmer --gpu` support; verify against code.
- `agents_docs/gpu-support-history.md`: dated protein GPU attempts, reversions, and lessons learned.
- `agents_docs/gpu-timing-analysis.md`: GPU timing instrumentation semantics, overlap caveats, profiling results, and optimization opportunities.
- `agents_docs/nhmmer-gpu-progress.md`: current nucleotide `nhmmer --gpu` architecture, file map, benchmark results, parity notes, and performance analysis.
- `agents_docs/nhmmer-gpu-todo.md`: nucleotide GPU phase checklist, known issues, and remaining performance work.
- `agents_docs/nhmmer-gpu-perf-gap.md`: why current GPU `nhmmer` remains slower than CPU-16, with root-cause timing analysis.
- `agents_docs/cuda-hmm-reference.md`: reference notes for the external `divinrkz/cuda-hmm` Viterbi/Forward/Backward implementation, including which high-level CUDA DP ideas are useful and why its code/build system should not be copied into HMMER.
- `agents_docs/nucleotide-benchmark.md`: nucleotide benchmark setup (chr22 target, query HMMs, FM-index, baseline timings, rebuild instructions).

## Code Navigation

For code changes, read in this order before editing:

1. `src/hmmer.h` for shared structs, exposed API, compile-time feature gates, and the broad include graph.
2. The relevant CLI entrypoint in `src/*.c`.
3. The owning implementation module named by the entrypoint or function prefix.
4. `src/Makefile.in` to confirm whether changed code is library code, a program object, a unit-test driver, an example, a benchmark, or architecture-specific code.

For protein GPU work, read:

1. `src/hmmsearch.c` for CLI validation, GPU option defaults, engine lifecycle, `.gpudb`/`dsqdata` opening, and query loop integration.
2. `src/hmmsearch_gpu.c` for batch assembly, fused SSV/null/bias/F1 orchestration, resident versus streaming paths, survivor processing, compare diagnostics, and CPU post-Fwd continuation.
3. `src/cuda/p7_cuda.h` and `src/cuda/p7_cuda_internal.h` for CUDA API boundaries and engine-owned buffers.
4. Stage-owned CUDA files: `p7_cuda_ssv.cu`, `p7_cuda_bias.cu`, `p7_cuda_viterbi.cu`, `p7_cuda_forward.cu`, `p7_cuda_fb_parser.cu`, and `p7_cuda_runtime.cu`.
5. Database helpers: `src/hmmseqdb.c`, `src/p7_gpudb.c`, and `src/p7_gpudb.h`.

For nucleotide GPU work, read:

1. `src/nhmmer.c` for CLI options, engine lifecycle, `.nucdb` detection, and per-query GPU handoff.
2. `src/nhmmer_internal.h` and `src/nhmmer_gpu.c` for `NHMMER_GPU_INFO`, window batches, GPU long-target orchestration, GPU Forward survivor reduction, default parser-matrix reuse, CPU domain handoff, and hit reporting.
3. `src/cuda/p7_cuda_ssv_longtarget.cu`, `src/cuda/p7_cuda_viterbi_longtarget.cu`, `src/cuda/p7_cuda_fb_parser.cu`, and `src/cuda/p7_cuda_domain_rescore.cu`.
4. Database helpers: `src/hmmnucdb.c`, `src/p7_nucdb.c`, and `src/p7_nucdb.h`.

Prefer existing project patterns and helper APIs over new layering.

## Verification And Data

Do not report full configure/build/check verification unless `easel/` is present and the configure/build actually ran. Fresh checkouts need Easel before `autoconf` can regenerate `configure`, because `configure.ac` includes macros from `easel/m4`.

Benchmark datasets are local working data, not repository content. Put downloaded/generated benchmark data under ignored `benchmark-data/`; see `agents_docs/tests-docs-benchmarks.md` for the protein `profmark` dataset and nucleotide benchmark smoke data.

For protein GPU work, prefer the protein `profmark` dataset plus an `hmmseqdb`-built target database (`dsqdata` and `.gpudb`). Record CPU/GPU wall time, CUDA H2D/kernel/D2H time, batch sizes/counts, pass counts for MSV/bias/Viterbi/Forward, and final hit deltas in `agents_docs/gpu-support-progress.md` as the work lands.

For nucleotide GPU work, use the chr22 nucleotide benchmark under `benchmark-data/nucleotide-bench/` and the scripts in `test-speed/` (`x-nhmmer-gpu-bench`, `x-nhmmer-gpu-parity`) when available. Record CPU-1, CPU-16, GPU FASTA, GPU `.nucdb`, overlap `.nucdb`, hit counts, per-stage timing, and parity notes in `agents_docs/nhmmer-gpu-progress.md`.
