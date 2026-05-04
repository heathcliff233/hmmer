# Easel Integration

Easel is HMMER's sibling C support library for biological sequence analysis. It is built and linked as part of this worktree, and HMMER code uses Easel types and conventions throughout normal runtime paths.

## Easel Layout

- `easel/easel.h` and `easel.c`: foundation layer for status codes, exceptions, allocation macros, constants, and portability.
- `easel/esl_*.{c,h}`: reusable modules for alphabets, sequence/MSA I/O, statistics, randomization, vector operations, clustering, threading, MPI helpers, and data structures.
- `easel/miniapps/`: command-line tools such as `esl-sfetch`, `esl-afetch`, `esl-shuffle`, `esl-seqstat`, `esl-reformat`, and `esl-alistat`.
- `easel/devkit/`: developer tooling, including `sqc` test execution and documentation helpers.
- `easel/testsuite/`: Easel's own SQC tests.
- `easel/demotic/`: Perl parsers/adapters for benchmark and regression workflows involving BLAST/FASTA/HMMER outputs.
- `easel/m4/`: autoconf macros used by HMMER for SIMD, pthread, MPI, PIC, and compiler feature checks.
- `easel/esl_msa_testfiles/` and `easel/formats/`: format fixtures and examples used by Easel tests and format handling.

## Build Integration

HMMER's `configure.ac` includes Easel macros from `easel/m4`, configures `easel/esl_config.h`, and sets `HMMER_ESLDIR=easel`. Top-level `make`, `make dev`, `make tests`, and `make check` descend into Easel before building HMMER components.

HMMER program and benchmark link lines include `-leasel`, and HMMER sources include Easel headers through `src/hmmer.h` and CLI-specific includes. Documentation builds also consume Easel miniapp manpages through `documentation/userguide/Makefile.in`.

If `configure.ac` or included m4 inputs change, Easel must be present before running `autoconf`.

## Runtime APIs Used By HMMER

The most common HMMER-facing Easel types are:

- `ESL_ALPHABET` and `ESL_DSQ`: biological alphabet and digitized residue representation.
- `ESL_MSA` and `ESL_MSAFILE`: multiple sequence alignments and alignment file readers.
- `ESL_SQ`, `ESL_SQFILE`, and `ESL_SQ_BLOCK`: sequences, sequence database readers, and sequence blocks.
- `ESL_GETOPTS` and `ESL_OPTIONS`: CLI option tables, parsing, validation, environment processing, and help output.
- `ESL_RANDOMNESS` and `ESL_RAND64`: reproducible random number generators.
- `ESL_STOPWATCH`: timing and CPU/elapsed time reporting.
- `ESL_WORK_QUEUE` and `ESL_THREADS`: threaded reader/worker coordination.
- `ESL_KEYHASH`, `ESL_FILEPARSER`, `ESL_DMATRIX`, and vector/statistical helpers used in parsing, tables, calibration, scoring, and benchmarks.

`src/hmmer.h` directly includes foundational Easel headers such as `easel.h`, `esl_alphabet.h`, `esl_getopts.h`, `esl_msa.h`, `esl_random.h`, `esl_rand64.h`, `esl_sq.h`, `esl_scorematrix.h`, and `esl_stopwatch.h`.

## Coding Conventions

HMMER follows Easel's C error and allocation style in many modules:

- return `eslOK` on success and Easel status codes such as `eslFAIL`, `eslEFORMAT`, `eslEINVAL`, or `eslEMEM` on failure;
- use `ESL_ALLOC`, `ESL_REALLOC`, `ESL_XFAIL`, and `ESL_XEXCEPTION` in functions with `int status` and an `ERROR:` cleanup path;
- use `esl_fatal()` for unrecoverable CLI failures;
- use Easel option tables (`ESL_OPTIONS`) for command-line programs instead of hand-rolled option parsing.

When editing code that already uses Easel conventions, keep the local style. Do not mix unrelated allocation/error styles in the same function unless the surrounding module already does so.

## HMMER Data Flow Boundary

Easel owns biological input parsing and generic sequence/alignment representations. HMMER converts those into `P7_*` objects:

1. Easel opens and reads sequence/alignment files.
2. `ESL_ALPHABET` determines residue alphabet and digitization.
3. `ESL_MSA` or `ESL_SQ` feeds `P7_BUILDER` or `p7_SingleBuilder()`.
4. HMMER constructs `P7_HMM`, then `P7_PROFILE` and `P7_OPROFILE`.
5. Easel sequence objects remain target/query carriers as HMMER pipelines score and report hits.

This boundary is especially visible in `hmmbuild`, `hmmsearch`, `hmmscan`, `phmmer`, `jackhmmer`, `nhmmer`, and `nhmmscan`.

For planned protein GPU search, Easel `dsqdata` is the intended base for GPU-capable sequence databases. Per-sequence length metadata should be written during sequence database construction, then consumed by future `hmmsearch --gpu` batch planning. See `agents_docs/gpu-support-todo.md`.

## Miniapps And Tests

Easel miniapps are part of normal HMMER workflows:

- `esl-sfetch --index` creates SSI indexes required by `profmark/create-profmark` sequence databases.
- `esl-afetch` fetches named alignments for benchmark driver scripts.
- `esl-shuffle` synthesizes random DNA/protein sequence in tests.
- `esl-seqstat` summarizes benchmark and tutorial sequence sets.
- `esl-reformat` converts alignment/sequence formats used in docs, tests, and examples.

`make check` runs Easel's own tests before HMMER's `testsuite/testsuite.sqc`. HMMER's tests use Easel's `devkit/sqc` runner, so `python3` and Easel's devkit are part of full verification.

## Benchmark Integration

`profmark/create-profmark` is heavily Easel-based: it reads Stockholm MSAs and FASTA sequence databases, uses SSI indexes, performs pairwise identity and clustering/independent-set splitting, generates random/lognormal lengths, shuffles/composes synthetic segments, and writes benchmark MSAs/FASTA/tables.

`profmark/rocplot` uses Easel option parsing, file parsing, key hashes, random Dirichlet sampling, vector operations, and statistics.

`easel/demotic/` contains parser/adaptor scripts used by historical benchmark comparisons against BLAST, FASTA, HMMER2, and HMMER3 outputs.
