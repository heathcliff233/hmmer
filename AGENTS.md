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

GPU status snapshot:

- The current `hmmsearch --gpu` path is protein-only, opt-in, and requires `hmmseqdb`/Easel protein `dsqdata` input.
- The repository applies an Easel `dsqdata` chunk-sizing patch at build time from `patches/easel-dsqdata-open-sized.patch`; do not edit the submodule in place for this work.
- Keep GPU work in the existing autotools build. Do not add CMake to this repository for external CUDA-HMM references; those projects are design references only. Borrow high-level implementation ideas where useful, but do not copy code, vendor sources, link external libraries, or add external build support.
- In this configured workspace, `nvcc` is `/usr/local/cuda/bin/nvcc`; exporting `CUDA_HOME=/usr/local/cuda` is useful for reproducibility but the generated Makefile already records the absolute compiler path.
- CUDA currently accelerates MSV plus the biased-composition filter in the GPU path. Bias reuses the uploaded MSV sequence batch; avoid a second dsq transfer when adding adjacent GPU stages.
- `hmmsearch --gpu` separates dsqdata load sizing (`--gpu-load-seqs`, `--gpu-load-res`) from CUDA search-batch sizing (`--gpu-batch-seqs`, `--gpu-batch-res`) and can pack multiple loaded chunks into one CUDA filter batch. The measured default remains 32,768 sequences / 8M residues for both load and search; larger 12M search batches reduced launches but did not improve all-13 profmark wall time.
- Current profmark runs show real end-to-end speedups on representative protein queries. The current all-13 stage-count run in `benchmark-data/profmark-current/gpu-stage-suitability/counts-13/` recorded CPU wall 25.41 sec, GPU wall 14.66 sec, aggregate speedup 1.733x, four CPU-only hits from known SSV/MSV sensitivity gaps, and no GPU-only hits. Treat `benchmark-data/profmark-current` as the real benchmark, not tutorial-sized inputs.
- The active sensitivity issue is SSV, not Viterbi/Forward. CPU SSE `p7_MSVFilter()` may accept the `p7_SSVFilter()` shortcut, while GPU v1 computes full MSV only. Some borderline CPU-only hits can therefore still appear even though later GPU Viterbi/Forward score comparisons are clean; keep this as a deferred SSV/MSV parity task.
- Remaining GPU costs are mainly CPU survivor continuation after bias. Viterbi is still a logical bottleneck, but the first broad candidate-only CUDA prototype was rejected as slower and not parity-safe; the later opt-in score-prefilter path remains experimental.
- Forward/Backward is only partially suitable as a simple GPU target: `p7_ForwardParser()` supplies both the F3 score and `P7_OMX` state needed by Backward/domain. A newer `--gpu-fb-parser` handoff exists as scaffolding for parser `xmx` state and batches F3 survivors per search chunk. It now has block-parallel short-profile parser Forward and Backward kernels with shared-memory opt-in for larger blocks. On `benchmark-data/profmark-current` two-query runs over `14-3-3` and `ATG2_CAD`, parser-only non-compare recorded CPU wall 11.29 sec, GPU wall 5.63 sec, aggregate speedup 2.005x, and zero hit deltas; with `--gpu-fwd-prefilter` it recorded CPU wall 11.41 sec, GPU wall 5.51 sec, aggregate speedup 2.071x, and zero hit deltas. The path remains experimental because raw `p7X_SCALE` row diagnostics persist, even though the checked `p7_DomainDecoding()` posterior inputs differ only at about 1e-5. Backward, domain definition, null2, hit storage, thresholding, and output should stay CPU-side for accepted/default paths.

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
- Protein search: `hmmsearch`, `phmmer`, `jackhmmer`, and `hmmscan`.
- Nucleotide search: `nhmmer` and `nhmmscan`, using long-target window logic and nucleotide-specific E-values.
- FM-index databases: `makehmmerdb` plus `libdivsufsort` and FM-index search helpers.
- Daemon/cache paths: `hmmpgmd`, `hmmpgmd_shard`, cached sequence/HMM data, and command/status serialization.
- Tests/docs/benchmarks: `testsuite/`, `documentation/`, `tutorial/`, `profmark/`, `test-speed/`, and `autobuild/`.

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
- `agents_docs/gpu-support-todo.md`: planned CUDA GPU support, `hmmsearch --gpu`, GPU-capable sequence database construction, and open validation/tuning work.
- `agents_docs/gpu-support-progress.md`: current GPU implementation status, benchmark results, and remaining gaps.
- `agents_docs/cuda-hmm-reference.md`: reference notes for the external `divinrkz/cuda-hmm` Viterbi/Forward/Backward implementation, including which high-level CUDA DP ideas are useful and why its code/build system should not be copied into HMMER.

## Code Navigation

For code changes, read in this order before editing:

1. `src/hmmer.h` for shared structs, exposed API, compile-time feature gates, and the broad include graph.
2. The relevant CLI entrypoint in `src/*.c`.
3. The owning implementation module named by the entrypoint or function prefix.
4. `src/Makefile.in` to confirm whether changed code is library code, a program object, a unit-test driver, an example, a benchmark, or architecture-specific code.

Prefer existing project patterns and helper APIs over new layering.

## Verification And Data

Do not report full configure/build/check verification unless `easel/` is present and the configure/build actually ran. Fresh checkouts need Easel before `autoconf` can regenerate `configure`, because `configure.ac` includes macros from `easel/m4`.

Benchmark datasets are local working data, not repository content. Put downloaded/generated benchmark data under ignored `benchmark-data/`; see `agents_docs/tests-docs-benchmarks.md` for the protein `profmark` dataset and nucleotide benchmark smoke data.

For GPU work, prefer the protein `profmark` dataset plus an `hmmseqdb`-built `dsqdata` target database. Record both CPU and GPU wall time, kernel time, transfer time, batch size, and sensitivity deltas in `agents_docs/gpu-support-progress.md` as the work lands.
