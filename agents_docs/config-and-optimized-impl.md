# Build Config And Optimized Implementations

HMMER's build is controlled by `configure.ac`, generated Makefiles, generated config headers, and one selected vector DP implementation. Build-system changes can affect C feature gates, Easel, documentation outputs, tests, and the active `src/impl` implementation.

## Autotools Inputs

`configure.ac` configures HMMER, Easel, and `libdivsufsort` together. It includes Easel macros from `easel/m4`, so Easel must be present before regenerating `configure`.

Primary generated Makefiles:

- top level: `Makefile`;
- HMMER source and tests: `src/Makefile`, `testsuite/Makefile`, `profmark/Makefile`;
- active optimized implementation: `src/impl_${impl_choice}/Makefile`;
- documentation: `documentation/Makefile`, `documentation/man/Makefile`, `documentation/userguide/Makefile`, `documentation/userguide/inclusions/Makefile`;
- bundled suffix-array library: `libdivsufsort/Makefile`;
- Easel: `easel/Makefile`, `easel/miniapps/Makefile`, `easel/testsuite/Makefile`, `easel/documentation/Makefile`.

If `configure.ac`, `*.in`, or `easel/m4` macros change, check whether `autoconf` and `./configure` need to be rerun before claiming verification.

## Configure Feature Gates

Important options in `configure.ac`:

- `--enable-sse`, `--enable-neon`, `--enable-vmx`: select the vector DP implementation. Autodetection normally chooses one; explicitly enabling more than one is invalid.
- `--enable-threads`: enables POSIX-threaded search/build paths and defines `HMMER_THREADS`/`HAVE_PTHREAD`.
- `--enable-mpi`: uses MPI compiler detection, defines `HMMER_MPI`/`HAVE_MPI`, and adds MPI unit tests and benchmarks.
- `--enable-asan`, `--enable-tsan`: add sanitizer flags; ASan and TSan are mutually exclusive.
- `--enable-gcov`, `--enable-gprof`, `--enable-debugging`, `--enable-fortify=x`: alter compiler flags or debug defines.
- `--enable-pic`: enables position-independent code flags through Easel macros.
- `--with-gsl`: links GNU Scientific Library support and defines `HAVE_LIBGSL`.

Feature gates are visible to HMMER through `src/p7_config.h` and to Easel through `easel/esl_config.h`. Do not add compile-time conditionals without checking both headers and the existing `HMMER_*`, `HAVE_*`, and `eslENABLE_*` conventions.

## Generated Files

Configured/generated files are build artifacts, not hand-maintained source:

- `src/p7_config.h` from `src/p7_config.h.in`;
- `easel/esl_config.h` and `easel/decoy_config.h`;
- `libdivsufsort/divsufsort.h` from `libdivsufsort/divsufsort.h.in`;
- generated `Makefile`s from corresponding `Makefile.in` files;
- configured man pages in `documentation/man/*.man` from `*.man.in`;
- configured user guide title/copyright TeX files.

`configure` also creates a build-tree `src/impl` symlink to the selected `src/impl_${impl_choice}` directory. Testsuite SQC tests rely on this link. In source navigation, inspect the concrete implementation directory as well as `src/impl` if the tree is already configured.

## Optimized Implementation Shape

The vectorized DP code lives in:

- `src/impl_sse/`: Intel/AMD SSE implementation;
- `src/impl_neon/`: ARM NEON implementation;
- `src/impl_vmx/`: Power/Altivec VMX implementation.

These directories provide the active implementations of optimized objects and algorithms declared through `src/hmmer.h` and `impl_*.h`:

- `P7_OPROFILE`: striped/interleaved optimized profile layout for byte, word, and float scoring stages;
- `P7_OMX`: optimized DP matrix layout;
- MSV/SSV, Viterbi, Forward/Backward, null2, decoding, optimal-accuracy, stochastic traceback, MPI serialization, profile I/O, and related test/benchmark drivers.

The generic algorithms in `src/generic_*.c` are not drop-in replacements for optimized filters. Changes to score scaling, profile layout, matrix allocation, or disk serialization must be checked against the selected implementation and pressed database files.

CUDA support remains separate from this SIMD implementation selection. The opt-in protein `hmmsearch --gpu` path lives in `src/cuda/`; it is not an `impl_cuda` replacement, and the selected SSE/NEON/VMX implementation remains available for CPU stages. See `gpu-support-progress.md` for the current state and `gpu-support-todo.md` for open work.

## CUDA Build Configuration

CUDA is controlled by `--enable-cuda` and `--with-cuda-arch=sm_XX` (default `sm_89`). The `configure.ac` CUDA block (lines ~453-479):

- Searches `/usr/local/cuda/bin` and `/usr/local/cuda-12.6/bin` for `nvcc`
- Links `-lcudart`
- Compiles CUDA objects: `p7_cuda_runtime.o`, `p7_cuda_msv.o`, `p7_cuda_bias.o`, `p7_cuda_viterbi.o`, `p7_cuda_forward.o`, `p7_cuda_fb_parser.o`
- Uses `easel/m4/esl_cuda.m4` for CUDA macro support

Non-CUDA builds use `src/cuda/p7_cuda_stub.c` to provide stub symbols. When CUDA is disabled, `hmmsearch --gpu` must fail with "HMMER was built without CUDA support".

## Implementation Makefiles

Each `src/impl_*/Makefile.in` has its own:

- `OBJS` for `libhmmerimpl.a`;
- `UTESTS` such as optimized filter, Forward/Backward, null2, decoding, stochastic traceback, optimal-accuracy, and I/O tests;
- `BENCHMARKS` and `EXAMPLES`;
- optional MPI test/benchmark substitutions when `--enable-mpi` is active.

Top-level `make`, `make dev`, `make tests`, and `make check` descend into the selected implementation through `src/impl`. If a change touches optimized code, run the implementation-specific unit test when possible, then broader HMMER tests for pipeline-visible behavior.

## `libdivsufsort`

`libdivsufsort/` is a bundled lightweight suffix-array/BWT library. HMMER builds `libdivsufsort.a` and links it into programs through `src/Makefile.in`, but its HMMER-facing purpose is narrow: `makehmmerdb` uses it while constructing FM-index nucleotide target databases.

Do not treat `libdivsufsort` as a general sequence/search dependency. For FM-index behavior, start in `src/makehmmerdb.c`, `src/fm_general.c`, `src/fm_alphabet.c`, and `src/fm_ssv.c`; use `libdivsufsort/` when suffix-array or BWT construction itself is relevant.
