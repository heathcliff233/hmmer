# Architecture

HMMER is a C/autotools project centered on profile hidden Markov models. The top-level build expects `easel/`, `libdivsufsort/`, `src/`, `testsuite/`, `documentation/`, and benchmark/support directories to move together.

## Repository Layout

- `src/`: core HMMER library code, installed CLI programs, auxiliary programs, optimized implementations, daemon/cache code, FM-index code, test drivers, examples, benchmarks, and stats programs.
- `src/impl_sse`, `src/impl_vmx`, `src/impl_neon`: vectorized acceleration layers selected by `configure`. The active implementation is built as `src/impl`.
- `libdivsufsort/`: suffix array construction used by `makehmmerdb` for FM-index databases.
- `easel/`: required external-but-sibling library. It supplies sequence/MSA I/O, alphabets, random utilities, work queues, threads, sqc tests, and autoconf macros.
- `testsuite/`: integration data and `testsuite.sqc`.
- `documentation/`, `tutorial/`: user guide, man pages, and runnable tutorial material.
- `profmark/`, `test-speed/`, `autobuild/`: benchmarks, performance harnesses, and automation scripts.

## Central API

`src/hmmer.h` is the all-encompassing HMMER header. Its opening comment is literal: the project has cross-dependencies and only limited modularity. Future edits should assume shared structs are intentionally visible across subsystems.

Use `src/hmmer.h` as the first source map for:

- model and profile objects,
- DP matrices and optimized profile APIs,
- pipeline and hit-reporting contracts,
- FM-index structs and functions,
- MPI/thread feature gates,
- builder configuration,
- daemon-facing data structures that rely on core HMMER types.

Architecture-specific optimized APIs enter through `impl_{sse,vmx,neon}/impl_*.h`, especially `P7_OPROFILE` and `P7_OMX`.

## Core Objects

- `P7_HMM`: the core Plan7 model, including transition/emission probabilities, annotations, cutoffs, E-value parameters, composition, offsets, alphabet, and max-length metadata.
- `P7_PROFILE`: generic scoring profile derived from `P7_HMM`; holds transition/emission scores and length-model state.
- `P7_OPROFILE`: optimized/vector profile in the selected implementation; required by accelerated MSV/SSV/Viterbi/Forward stages and pressed database files.
- `P7_BG`: null/background model and bias filter state.
- `P7_TRACE`: traceback path used for alignments and model construction.
- `P7_HMMFILE`: text/binary/pressed HMM file reader with staged `.h3m`, `.h3f`, `.h3p`, `.h3i` support.
- `P7_GMX`: generic DP matrix for reference/generic algorithms.
- `P7_OMX`: optimized DP matrix for vector filters and DP.
- `P7_PIPELINE`: accelerated comparison pipeline state, thresholds, counters, domain definition object, long-target settings, and reporting mode.
- `P7_DOMAINDEF`: reusable workflow state for domain definition, posterior decoding, envelopes, null2, and alignments.
- `P7_TOPHITS`: ranked hit list with target/domain thresholding, output, merge, duplicate removal, and MPI serialization.
- `P7_BUILDER`: model-construction configuration for MSA and single-sequence builders.
- `P7_SCOREDATA`: compact score data used by long-target diagonal recovery and extension.
- `P7_HMM_WINDOWLIST`: dynamic list of target windows passed between long-target stages.

## Dependency Shape

Most CLI programs are thin orchestration layers around shared builder, HMM file, profile, pipeline, top-hit, and output functions. Do not assume a change is isolated because it starts in one command. For example, `p7_pipeline.c` affects protein searches, nucleotide long-target searches, scan mode, daemon behavior, counters, and output thresholds.

`src/Makefile.in` is the ownership checkpoint. It shows which files are linked into `libhmmer.a`, which are CLI entrypoints, which unit tests are compiled by defining `p7*_TESTDRIVE`, and which examples/benchmarks exercise the same implementation files.
