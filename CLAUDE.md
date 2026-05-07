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

# Run GPU search (protein-only, opt-in; all GPU stages on by default)
src/hmmsearch --gpu query.hmm target.gpudb

# Debug: compare SSV scores to monolithic MSV (prints mismatches to stderr)
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

The GPU path accelerates SSV/MSV + biased-composition filter + Viterbi + Forward in CUDA batches. Current state:

- Default MSV path: **fused SSV+null+bias+F1 gate kernel** (`cuda_ssv_null_bias_gate_kernel<STRIDE>`) computes all pre-filter stages in a single kernel launch. Templated on STRIDE with linear rbv layout for coalesced access. Survivors compacted via atomicAdd with in-kernel float score output (D2H only ~32KB/batch vs 1.8MB). Supports both resident-database and chunk-based paths.
- All GPU stages enabled by default with `--gpu`: fused SSV+null+bias+gate → Viterbi prefilter (M≤2048) → Forward prefilter (M≤1024) → FB parser
- Latest all-13 profmark (multi-query single-process): **8.68x vs CPU-1** (10.68s → 1.23s), **3.89x vs CPU-4** (4.79s → 1.23s), zero fused-vs-legacy parity errors
- Survivor loop: sorted by sequence length with ReconfigLength caching; CPU MSV fallback eliminated (double-precision GPU bias is authoritative)
- CPU-side modules: domain definition, null2, hit reporting, sequence metadata assembly
- Sequence packing uses bulk `smem` copy (single memcpy of dsqdata's contiguous buffer) with L+1 offset spacing
- `.gpudb` v2 format embeds sequence metadata (name, acc, desc, taxid); when present, `hmmsearch --gpu` skips dsqdata entirely (zero per-query I/O)

## GPU Timing Instrumentation

The GPU output reports three timing layers (see `agents_docs/gpu-timing-analysis.md` for full semantics):

- **"CUDA *" lines**: device-side event timings per stage (can overlap with host work)
- **"Stage *" lines**: CPU pipeline stage wall-clock (overlap with CUDA timings by construction)
- **"Exact *" lines**: exclusive wall-clock buckets that sum to `exact_wall`

Key gotcha: `survivor_loop_other` is the **total wall time** of the survivor loop, not "unaccounted residual". With all GPU stages active, `vit_fwd_dispatch` **overlaps** with `gpu_kernel` — the Exact buckets are not fully exclusive.

## GPU Expert Flags (hidden, docgroup 99)

These flags are not shown in `hmmsearch -h` output but are defined in `src/hmmsearch.c`:

```
--gpu-vit-largem       Allow GPU Viterbi on large models M>2048 (requires --gpu)
--gpu-fwd-largem       Allow GPU Forward on large models M>1024 (requires --gpu)
--gpu-ssv-compare      Debug: compare SSV scores to monolithic MSV
```

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
| [gpu-timing-analysis.md](agents_docs/gpu-timing-analysis.md) | GPU timing instrumentation semantics, profiling results, optimization opportunities |
| [cuda-hmm-reference.md](agents_docs/cuda-hmm-reference.md) | External cuda-hmm reference (design cues only, no code/build import) |

## Code Navigation

For code changes, read in this order:

1. `src/hmmer.h` — shared structs, API, feature gates
2. The relevant CLI entrypoint (`src/hmmsearch.c`, `src/nhmmer.c`, etc.)
3. The owning implementation module (`p7_pipeline.c`, `p7_domaindef.c`, etc.)
4. `src/Makefile.in` — confirms whether code is library, program, test, or benchmark
5. For GPU: `src/hmmsearch_gpu.c` (orchestration) → `src/cuda/p7_cuda.h` → individual `.cu` files
