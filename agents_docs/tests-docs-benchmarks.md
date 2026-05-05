# Tests, Docs, And Benchmarks

Use `src/Makefile.in` and `testsuite/testsuite.sqc` to choose verification. Full configure/build/check requires the sibling `easel/` tree because HMMER builds and tests Easel as part of the top-level workflow.

## Build Targets

Top-level `Makefile.in`:

- `make`: builds Easel, `libdivsufsort`, HMMER programs, and `profmark`.
- `make dev`: builds programs plus test drivers, stats, benchmarks, and examples.
- `make tests`: compiles test drivers for checks.
- `make check`: requires `python3`, compiles needed pieces, runs Easel checks, then runs `testsuite/testsuite.sqc`.
- `make pdf`: builds documentation PDFs.

`src/Makefile.in`:

- `PROGS`: installed CLI programs.
- `AUXPROGS`: built but not installed.
- `OBJS`: `libhmmer.a` library objects.
- `UTESTS`: unit-test drivers compiled from implementation files with `p7*_TESTDRIVE` defines.
- `ITESTS`: integration test executables.
- `EXAMPLES`, `BENCHMARKS`, `STATS`: source-local examples, performance drivers, and statistical harnesses.

## Unit Tests And Examples

Unit tests are generated from source files by Makefile rules. For example, `p7_hmm_utest` compiles `p7_hmm.c` with a test driver define. If you add a new source file or change a module with existing embedded tests, check `UTESTS`, `EXAMPLES`, and `BENCHMARKS` in `src/Makefile.in`.

Useful source-local tests include:

- filter/DP: `generic_msv_utest`, `generic_viterbi_utest`, `generic_fwdback_utest`, `generic_decoding_example`;
- model/profile: `p7_hmm_utest`, `p7_profile_utest`, `modelconfig_utest`, `seqmodel_utest`;
- I/O/reporting: `p7_hmmfile_utest`, `p7_tophits_utest`, `p7_alidisplay_utest`;
- daemon stats/status: `p7_hmmd_search_stats_utest`, `hmmd_search_status_utest`, `hmmpgmd2msa_utest`;
- FM/long-target support: `p7_scoredata_utest`.

## Integration Tests

`testsuite/testsuite.sqc` is the main integration suite. It calls scripts in `testsuite/`, including:

- build/search variation tests,
- `hmmalign`, `hmmconvert`, `hmmemit`, `hmmbuild` behavior tests,
- stdin/rewind/error-format cases,
- `i18-nhmmer-generic.pl` and `iss159-nhmmer-overlap.py` for nucleotide search behavior,
- `i20-fmindex-core.pl` for FM-index behavior; this script exists, but `testsuite/testsuite.sqc` currently comments it out with a note that the fmindex test is not active,
- `i19-hmmpgmd-ga.pl` and `i22-hmmpgmd-shard-ga.pl` for daemon paths.

Run `make check` for broad behavior changes. For a focused change, run the relevant unit test and one or more integration scripts through the configured test harness, then escalate to `make check` when risk touches shared pipeline, output, file formats, or parallel behavior.

GPU changes need dedicated checks before performance claims: CUDA MSV/bias parity against the CPU filter boundary, GPU-capable `dsqdata` length metadata round trips, `hmmseqdb` construction smoke tests, and end-to-end `hmmsearch --gpu` versus CPU output comparisons. See `agents_docs/gpu-support-todo.md`.

## Documentation

- `documentation/man/`: installed man pages. Edit `*.man.in` sources; generated `*.man` files are configured outputs.
- `documentation/userguide/`: User Guide and daemon guide sources; it also consumes Easel miniapp manpages and Easel manpage processing tools. `titlepage*.tex` and `copyright.tex` are configured from `.in` files.
- `tutorial/`: sample data used for user-facing examples.
- `release-notes/`: release history and notes.

If CLI options or output change, update the man page and user guide in the same patch unless the user explicitly asked for code-only work.

## Benchmarks And Automation

- `profmark/`: profile search benchmarking and comparison harnesses.
- `test-speed/`: speed/performance scripts.
- `autobuild/`: platform build automation.
- `contrib/`: extra experimental/support code that is not part of the installed program set.

Use these for performance-sensitive changes in DP filters, optimized implementations, FM-index code, or daemon/cache behavior. They are not substitutes for correctness tests.

Benchmark datasets should stay local and untracked. Use ignored `benchmark-data/` for downloaded inputs, generated datasets, and run logs; prefer `.git/info/exclude` for local-only ignores when the data is only for one checkout.

GPU note: the current `hmmsearch --gpu` path expects protein `dsqdata` from `hmmseqdb`, not raw FASTA. It accelerates MSV plus the biased-composition filter and reports separate CUDA MSV and CUDA bias H2D/kernel/D2H timing. The build applies `patches/easel-dsqdata-open-sized.patch` before building Easel, so treat GPU validation as a top-level build-and-run flow, not a direct submodule edit.

### Protein `profmark` Dataset

`profmark/create-profmark` creates the standard protein sensitivity benchmark dataset. The dataset is not checked in. Build it from external inputs:

- Pfam SEED Stockholm alignments, for example `Pfam-A.seed.gz` from the Pfam current release.
- UniProt/Swiss-Prot FASTA, for example `uniprot_sprot.fasta.gz` from the UniProt current release.

Typical local build:

```sh
mkdir -p benchmark-data/profmark-current/{downloads,work,logs}
curl -L -C - --fail -o benchmark-data/profmark-current/downloads/Pfam-A.seed.gz https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.seed.gz
curl -L -C - --fail -o benchmark-data/profmark-current/downloads/uniprot_sprot.fasta.gz https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
sha256sum benchmark-data/profmark-current/downloads/Pfam-A.seed.gz benchmark-data/profmark-current/downloads/uniprot_sprot.fasta.gz > benchmark-data/profmark-current/logs/downloads.sha256
gzip -dc benchmark-data/profmark-current/downloads/Pfam-A.seed.gz > benchmark-data/profmark-current/work/Pfam-A.seed
gzip -dc benchmark-data/profmark-current/downloads/uniprot_sprot.fasta.gz > benchmark-data/profmark-current/work/uniprot_sprot.fasta
easel/miniapps/esl-sfetch --index benchmark-data/profmark-current/work/uniprot_sprot.fasta
profmark/create-profmark -S 42 benchmark-data/profmark-current/work/pmark benchmark-data/profmark-current/work/Pfam-A.seed benchmark-data/profmark-current/work/uniprot_sprot.fasta
```

Expected outputs are `pmark.train.msa`, `pmark.test.fa`, `pmark.tbl`, `pmark.pos`, and `pmark.neg`. `pmark.tbl` includes both successful and failed family splits; filter field 8 for `ok` when selecting benchmarkable families. Validate with `wc -l pmark.tbl pmark.pos pmark.neg` and `easel/miniapps/esl-seqstat pmark.test.fa`.

For GPU work, prefer `benchmark-data/profmark-current/work/pmark.test.fa` plus an `hmmseqdb`-built `pmark.test.gpudb` target database. Keep `--gpu-load-seqs`, `--gpu-load-res`, `--gpu-batch-seqs`, `--gpu-batch-res`, `--gpu-msv-slack`, and CUDA batch count in the run logs, and record CPU vs GPU wall time, GPU kernel time, transfer time, CUDA bias timing, and any sensitivity differences.

Use `test-speed/x-hmmsearch-gpu-profmark` to run CPU/GPU query subsets and write `logs/profmark-gpu-summary.tsv`. Use `test-speed/x-hmmsearch-gpu-profmark-summary <summary.tsv>` to aggregate wall time, stage totals, pass counts, CUDA batch counts, sensitivity deltas, and the theoretical upper bound for a free Forward score prefilter that still reruns CPU Forward for F3 survivors.

### Nucleotide Benchmark Data

There is no separate `profmark`-style nucleotide benchmark dataset checked into this tree. Nucleotide benchmark and regression coverage comes from tutorial/test assets:

- `tutorial/MADE1.sto` and `tutorial/dna_target.fa` exercise `hmmbuild`, `nhmmer`, `nhmmscan`, and `makehmmerdb`.
- `testsuite/i18-nhmmer-generic.pl` synthesizes a larger pseudochromosome search case.
- `testsuite/i20-fmindex-core.pl` builds a small FM-index database and checks exact-match behavior; it may be commented out in `testsuite/testsuite.sqc`.

Typical local nucleotide data build:

```sh
mkdir -p benchmark-data/nucleotide-current/{work,logs}
cp tutorial/MADE1.sto benchmark-data/nucleotide-current/work/MADE1.sto
cp tutorial/dna_target.fa benchmark-data/nucleotide-current/work/dna_target.fa
src/hmmbuild benchmark-data/nucleotide-current/work/MADE1.hmm benchmark-data/nucleotide-current/work/MADE1.sto > benchmark-data/nucleotide-current/logs/hmmbuild-made1.log 2>&1
src/makehmmerdb benchmark-data/nucleotide-current/work/dna_target.fa benchmark-data/nucleotide-current/work/dna_target.fm > benchmark-data/nucleotide-current/logs/makehmmerdb-dna-target.log 2>&1
src/nhmmer benchmark-data/nucleotide-current/work/MADE1.hmm benchmark-data/nucleotide-current/work/dna_target.fa > benchmark-data/nucleotide-current/logs/nhmmer-made1-fasta.out 2> benchmark-data/nucleotide-current/logs/nhmmer-made1-fasta.err
src/nhmmer benchmark-data/nucleotide-current/work/MADE1.hmm benchmark-data/nucleotide-current/work/dna_target.fm > benchmark-data/nucleotide-current/logs/nhmmer-made1-fm.out 2> benchmark-data/nucleotide-current/logs/nhmmer-made1-fm.err
src/hmmpress -f benchmark-data/nucleotide-current/work/MADE1.hmm > benchmark-data/nucleotide-current/logs/hmmpress-made1.log 2>&1
src/nhmmscan benchmark-data/nucleotide-current/work/MADE1.hmm benchmark-data/nucleotide-current/work/dna_target.fa > benchmark-data/nucleotide-current/logs/nhmmscan-made1.out 2> benchmark-data/nucleotide-current/logs/nhmmscan-made1.err
```

This is a functional local nucleotide benchmark/smoke dataset, not a full sensitivity benchmark.

## Easel Verification Notes

Easel is part of full verification, not an optional external package. `make check` runs Easel's SQC suite before HMMER's suite, and HMMER tests use `easel/devkit/sqc`. Benchmark data workflows also use Easel miniapps such as `esl-sfetch`, `esl-afetch`, `esl-shuffle`, `esl-seqstat`, and `esl-reformat`.
