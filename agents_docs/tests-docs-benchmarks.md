# Tests, Docs, And Benchmarks

Use `src/Makefile.in` and `testsuite/testsuite.sqc` to choose verification. This checkout currently lacks `easel/`, so full configure/build/check requires restoring Easel first.

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

## Documentation

- `documentation/man/`: installed man pages.
- `documentation/userguide/`: User Guide and daemon guide sources; it also consumes Easel manpage processing tools.
- `tutorial/`: sample data used for user-facing examples.
- `release-notes/`: release history and notes.

If CLI options or output change, update the man page and user guide in the same patch unless the user explicitly asked for code-only work.

## Benchmarks And Automation

- `profmark/`: profile search benchmarking and comparison harnesses.
- `test-speed/`: speed/performance scripts.
- `autobuild/`: platform build automation.

Use these for performance-sensitive changes in DP filters, optimized implementations, FM-index code, or daemon/cache behavior. They are not substitutes for correctness tests.

## Current Worktree Verification Notes

Because `easel/` is missing here, docs-only changes can be verified with file inventory and Git status. Build/test verification should be reported as blocked until Easel is restored.
