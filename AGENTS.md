# Agent Guide

This is a coupled C/autotools HMMER3 worktree. Read it as one project with three build-time components:

- `src/`: HMMER library objects, command-line programs, optimized DP implementations, daemon support, FM-index support, unit tests, examples, benchmarks.
- `easel/`: required sibling library used throughout HMMER. This checkout currently does not contain `easel/`, but `configure.ac`, `Makefile.in`, `src/Makefile.in`, documentation tooling, and tests all expect it.
- `libdivsufsort/`: suffix-array support used by `makehmmerdb` and FM-index target databases.

## Build And Test

For a fresh source checkout, restore or clone `easel/` first. `configure.ac` includes macros from `easel/m4`, so `autoconf` cannot regenerate `configure` without Easel.

Typical workflow:

1. `autoconf` if `configure.ac` or included m4 inputs changed.
2. `./configure`
3. `make`
4. `make tests` to compile test drivers.
5. `make check` to run Easel checks plus `testsuite/testsuite.sqc`.

`make check` requires `python3`; both the top-level and `testsuite/Makefile.in` check for it. Do not report full build/test verification unless `easel/` is present and the configure/build actually ran.

## Navigation Rule

Start every code change by reading these in order:

1. `src/hmmer.h` for the shared structs, exposed API, compile-time feature gates, and the intentionally broad include graph.
2. The relevant CLI entrypoint in `src/*.c`.
3. The owning implementation module named by the entrypoint or function prefix.
4. `src/Makefile.in` to confirm whether a changed source is library code, a program object, a unit-test driver, an example, a benchmark, or architecture-specific code.

HMMER is not cleanly layered. `src/hmmer.h` is deliberately all-encompassing because profile construction, optimized profiles, filters, DP, hit reporting, daemon code, and FM-index support share core structures. Avoid introducing private abstractions that pretend these dependencies do not exist.

## Program Map

Installed programs from `src/Makefile.in`:

- Build/search: `hmmbuild`, `hmmsearch`, `hmmscan`, `phmmer`, `jackhmmer`, `nhmmer`, `nhmmscan`.
- Database preparation: `hmmpress`, `makehmmerdb`.
- Model and alignment utilities: `hmmalign`, `hmmemit`, `hmmfetch`, `hmmconvert`, `hmmstat`, `hmmlogo`, `alimask`.
- Simulation/daemon: `hmmsim`, `hmmpgmd`, `hmmpgmd_shard`.

Auxiliary programs are built but not installed: `hmmc2`, `hmmerfm-exactmatch`.

## Major Execution Families

- Protein profile search: `hmmsearch` reads HMM queries and searches protein sequence databases through `p7_Pipeline()`.
- Protein model scan: `hmmscan` reads sequences and scans pressed HMM databases; pressed profiles are staged from `.h3f` and `.h3p`.
- Protein sequence search: `phmmer` builds a single-sequence profile with `p7_SingleBuilder()` before sequence database search.
- Iterative protein search: `jackhmmer` repeats search, inclusion, MSA construction, and model rebuilding.
- Nucleotide long-target search: `nhmmer` accepts HMM/MSA/sequence queries, builds nucleotide HMMs when needed, searches long DNA/RNA targets, handles strand and window logic, can use FM-index targets, and recomputes nhmmer-specific E-values.
- Nucleotide scan: `nhmmscan` scans DNA/RNA sequences against pressed nucleotide profile databases through the long-target pipeline.
- HMM database preparation: `hmmpress` writes `.h3m`, `.h3f`, `.h3p`, `.h3i`.
- FM-index database preparation: `makehmmerdb` writes FM metadata and data using `libdivsufsort`.
- Daemon search: `hmmpgmd` and `hmmpgmd_shard` use cached sequence/HMM data plus command/status serialization.
- Tests/docs/benchmarks: `src/Makefile.in`, `testsuite/testsuite.sqc`, `tutorial/`, `documentation/`, `profmark/`, `test-speed/`, and `autobuild/`.

## Agent Docs

Additional agent-facing notes live in `agents_docs/` and are tracked directly in this repository. Use those files for architecture, data-flow, pipeline, nucleotide/FM-index, daemon/cache, and test navigation before making broad changes.
