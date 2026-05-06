# CLAUDE.md

HMMER is biological sequence analysis software that uses profile hidden Markov models (profile HMMs) to search for sequence homologs. This is the `h3-gpu` development branch adding opt-in CUDA GPU acceleration to protein `hmmsearch`.

## Quick Reference

### Build (CPU-only)

```sh
./configure --enable-threads
make -j$(nproc)
make check          # requires python3; runs Easel then HMMER test suites
```

### Build (CUDA GPU)

```sh
./configure --enable-cuda --with-cuda-arch=sm_89 --enable-threads
make -j$(nproc)
```

`nvcc` is at `/usr/local/cuda/bin/nvcc`. The generated Makefile records the absolute compiler path.

### Key Build Targets

| Target | What it does |
|--------|-------------|
| `make` | Easel + libdivsufsort + HMMER programs + profmark |
| `make dev` | Above + test drivers, stats, benchmarks, examples |
| `make tests` | Compile test drivers |
| `make check` | Full test suite (Easel SQC + HMMER testsuite.sqc) |
| `make -C src hmmsearch` | Rebuild just hmmsearch |
| `make -C src/impl_sse msvfilter_utest` | Rebuild a specific unit test |

### Running GPU hmmsearch

```sh
# Build GPU-capable target database
src/hmmseqdb target.fa target.gpudb

# Run GPU search (protein-only, opt-in)
src/hmmsearch --gpu query.hmm target.gpudb

# With later-stage GPU acceleration (experimental, default-off)
src/hmmsearch --gpu --gpu-vit-prefilter --gpu-fwd-prefilter --gpu-fb-parser query.hmm target.gpudb

# With standalone SSV kernel (experimental, parity-verified)
src/hmmsearch --gpu --gpu-ssv query.hmm target.gpudb

# Debug: compare SSV+fallback scores to monolithic MSV (prints mismatches to stderr)
src/hmmsearch --gpu --gpu-ssv-compare query.hmm target.gpudb
```

### Running profmark Benchmarks

```sh
# Run CPU/GPU profmark comparison
python3 test-speed/x-hmmsearch-gpu-profmark . benchmark-data/profmark-current/work \
  benchmark-data/profmark-current/gpu-audit/<run-name> <query-names...> [--gpu-flags...]

# Summarize results
python3 test-speed/x-hmmsearch-gpu-profmark-summary benchmark-data/profmark-current/gpu-audit/<run-name>/logs/profmark-gpu-summary.tsv
```

## Repository Structure

```
src/               HMMER library, CLI programs, optimized DP, tests
src/cuda/          CUDA GPU kernels and runtime (p7_cuda_*.cu)
src/impl_sse/      Active SSE/AVX vectorized implementation (symlinked as src/impl)
src/impl_neon/     ARM NEON implementation
src/impl_vmx/     PowerPC VMX implementation
easel/             Sibling library: sequence/alignment I/O, alphabets, utilities, macros
libdivsufsort/     Suffix-array library for FM-index databases
patches/           Build-time patches (applied to Easel before build)
testsuite/         Integration tests and test data
profmark/          Protein sensitivity benchmark harness
test-speed/        Performance scripts and GPU profmark runners
benchmark-data/    Local untracked benchmark datasets and run logs
agents_docs/       Detailed architecture documentation (see index below)
```

## Code Conventions

- `src/hmmer.h` is the central API map. Read it first for structs, exposed API, and feature gates.
- Follow Easel error/allocation style: `eslOK` returns, `ESL_ALLOC`, `ESL_XFAIL`, `ERROR:` cleanup labels.
- Use `ESL_OPTIONS` tables for CLI parsing; never hand-roll option parsing.
- Prefix types/functions with `P7_` or `p7_` for HMMER, `ESL_` or `esl_` for Easel.
- Unit tests are compiled from source files via `p7*_TESTDRIVE` defines in `src/Makefile.in`.
- CUDA code lives in `src/cuda/`; new CUDA-facing code should include `src/cuda/p7_cuda.h` (not `src/cuda_msv.h`).

## Important Constraints

- **Easel submodule**: Do not edit `easel/` directly for GPU work. The `dsqdata` chunk-sizing change comes from `patches/easel-dsqdata-open-sized.patch` applied at build time.
- **No CMake**: Keep CUDA in the existing autotools build. Do not add CMake for any reason.
- **GPU scope**: `hmmsearch --gpu` is protein-only. `hmmscan`, `phmmer`, `jackhmmer`, daemon, and nucleotide programs remain CPU-only.
- **Hit parity**: GPU path must preserve exact hit parity with CPU on profmark datasets (`cpu_only=0`, `gpu_only=0`).
- **Pressed HMM files**: Do not change `.h3m/.h3i/.h3f/.h3p` format as part of GPU work.
- **Configure requires Easel**: `configure.ac` includes macros from `easel/m4`; Easel must be present before `autoconf`.
- **Benchmark data**: Use `benchmark-data/` (gitignored) for datasets and run logs, not `tutorial/` inputs for speed claims.

## GPU Architecture Summary

The GPU path accelerates MSV + biased-composition filter in CUDA batches. Current state:

- Default path: CUDA MSV + bias with CPU-compatible F1 boundary checks
- Opt-in standalone SSV: `--gpu-ssv` (register-optimized kernel: SSV fast-path with in-kernel MSV fallback, 1.36x faster than monolithic MSV)
- Opt-in later stages: `--gpu-vit-prefilter`, `--gpu-fwd-prefilter`, `--gpu-fb-parser`
- Latest all-13 profmark (with later stages): 1.53x speedup (CPU 10.44s → GPU 6.83s), zero parity errors
- GPU vs CPU-4 (4-thread): currently 0.72x cold (slower due to 260ms/query CUDA init), 1.29x warm
- Dominant overhead: CUDA context initialization (260ms per query, 49.5% of GPU wall); amortizable by reusing engine across queries
- Remaining bottleneck: host sync/blocking (32.8%), CPU survivor continuation (8.0%)
- CPU-side modules: domain definition, null2, hit reporting, sequence metadata assembly
- Sequence packing uses bulk `smem` copy (single memcpy of dsqdata's contiguous buffer) with L+1 offset spacing
- SSV kernel status: parity-verified, 1.36x faster than monolithic MSV (register-based DP, precomputed q/z lookups, warp shuffle); strong candidate for default GPU MSV path

## Verification Checklist

- Non-CUDA build: `--disable-cuda`, confirm `--gpu` fails with diagnostic
- CUDA build: `--enable-cuda --with-cuda-arch=sm_89`, build succeeds
- Smoke test: `hmmsearch --gpu --cpu 0` hit names match CPU on tutorial inputs
- Profmark: record wall time, CUDA timing, pass counts, `cpu_only`/`gpu_only` deltas
- Timing: `Exact delta_vs_wall` in GPU output must print `[OK]` (within 1e-6 sec)

## agents_docs/ Index

Detailed architecture documentation lives in `agents_docs/`. Read in this order for context:

| File | Purpose |
|------|---------|
| [architecture.md](agents_docs/architecture.md) | Repository layout, central API (`hmmer.h`), core objects, dependency shape |
| [config-and-optimized-impl.md](agents_docs/config-and-optimized-impl.md) | Autotools feature gates, generated files, SIMD impl selection, CUDA build |
| [data-flow.md](agents_docs/data-flow.md) | Input → model → profile → pipeline → hits data flow |
| [search-pipelines.md](agents_docs/search-pipelines.md) | Pipeline stages, domain definition, long-target, scan mode |
| [program-families.md](agents_docs/program-families.md) | CLI programs: search, scan, build, utilities, daemon |
| [model-build-and-io.md](agents_docs/model-build-and-io.md) | `hmmbuild`, single-sequence models, HMM files, pressed databases |
| [easel-integration.md](agents_docs/easel-integration.md) | Easel library layout, build integration, runtime APIs, conventions |
| [nucleotide-and-fm-index.md](agents_docs/nucleotide-and-fm-index.md) | `nhmmer`, `nhmmscan`, FM-index databases, coordinate risks |
| [parallel-daemon-cache.md](agents_docs/parallel-daemon-cache.md) | Threads, MPI, daemon programs, sequence/HMM caches |
| [tests-docs-benchmarks.md](agents_docs/tests-docs-benchmarks.md) | Build targets, tests, docs, profmark dataset, nucleotide data |
| [gpu-support-progress.md](agents_docs/gpu-support-progress.md) | Current GPU state, benchmark results, open risks |
| [gpu-support-todo.md](agents_docs/gpu-support-todo.md) | GPU open work, validation checklist, deferred scope |
| [gpu-support-history.md](agents_docs/gpu-support-history.md) | Failed/reverted GPU attempts and lessons learned |
| [cuda-hmm-reference.md](agents_docs/cuda-hmm-reference.md) | External cuda-hmm reference (design cues only, no code/build import) |

## Code Navigation

For code changes, read in this order:

1. `src/hmmer.h` — shared structs, API, feature gates
2. The relevant CLI entrypoint (`src/hmmsearch.c`, `src/nhmmer.c`, etc.)
3. The owning implementation module (`p7_pipeline.c`, `p7_domaindef.c`, etc.)
4. `src/Makefile.in` — confirms whether code is library, program, test, or benchmark
5. For GPU: `src/hmmsearch_gpu.c` (orchestration) → `src/cuda/p7_cuda.h` → individual `.cu` files
