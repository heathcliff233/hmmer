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

### Running GPU nhmmer

```sh
# All GPU stages default-on with --gpu (SSV + batch filter + Viterbi)
src/nhmmer --gpu --cpu 4 --noali query.hmm target.fa

# With pre-built nucdb (eliminates FASTA parsing, ~20% faster)
src/hmmnucdb target.fa target.nucdb
src/nhmmer --gpu --cpu 4 --noali query.hmm target.nucdb.nucdb

# Nucdb with overlap (enables GPU-resident path, zero H2D for SSV)
src/hmmnucdb --overlap 2001 target.fa target-overlap.nucdb
src/nhmmer --gpu --cpu 4 --noali query.hmm target-overlap.nucdb.nucdb

# Tune chunk size (default 64K)
src/nhmmer --gpu --cpu 4 --gpu-chunk-size 32768 query.hmm target.fa
```

### Running nhmmer GPU Benchmarks

```sh
# Full timing comparison (CPU-1, CPU-4, GPU-4 FASTA, GPU-4 nucdb)
test-speed/x-nhmmer-gpu-bench .

# Quick parity check (MADE1 must match exactly)
test-speed/x-nhmmer-gpu-parity .
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
- **GPU scope**: `hmmsearch --gpu` is protein-only (requires `.gpudb`). `nhmmer --gpu` runs full GPU pipeline (SSV + batch filter + Viterbi + scanning Viterbi + Forward prefilter + FB parser) with threaded CPU downstream (works on plain FASTA or `.nucdb`). `hmmscan`, `phmmer`, `jackhmmer`, and daemon remain CPU-only.
- **Hit parity**: GPU nhmmer path preserves near-exact hit parity with CPU. MADE1: 462 vs 465 (3-hit difference). query_short: 363 vs 363 (exact match). query_medium: 648 vs 648 (exact match). Remaining differences are from float32 vs double precision in Forward/Backward accumulation.
- **Pressed HMM files**: Do not change `.h3m/.h3i/.h3f/.h3p` format as part of GPU work.
- **Configure requires Easel**: `configure.ac` includes macros from `easel/m4`; Easel must be present before `autoconf`.
- **Benchmark data**: Use `benchmark-data/` (gitignored) for datasets and run logs, not `tutorial/` inputs for speed claims.

## Nucleotide Benchmark

Target: human chr22 (hg38, 50MB FASTA, ~101.6M residues both strands). FM-index also available.

```sh
# Quick baseline
time src/nhmmer --cpu 1 --noali benchmark-data/nucleotide-bench/work/MADE1.hmm benchmark-data/nucleotide-bench/work/chr22.fa

# FM-index accelerated
time src/nhmmer --cpu 1 --noali benchmark-data/nucleotide-bench/work/MADE1.hmm benchmark-data/nucleotide-bench/work/chr22.fmdb
```

Queries: MADE1 (M=80, ~1s), query_short (M=151, ~1.5s), query_medium (M=501, ~6.4s), query_long (M=2001, stress only). See `agents_docs/nucleotide-benchmark.md` for full setup and rebuild instructions.

### Current nhmmer GPU Performance (RTX 4090, chr22)

| Path | MADE1 (M=80) | query_short (M=151) | query_medium (M=501) |
|------|:---:|:---:|:---:|
| CPU-1 | 1.08s / 465 | 1.33s / 363 | 5.67s / 648 |
| CPU-4 | 0.31s / 465 | 0.40s / 363 | 1.57s / 648 |
| GPU-4 FASTA | 1.01s / 462 | 0.99s / 363 | 2.87s / 648 |
| GPU-4 nucdb | 0.78s / 462 | 0.97s / 363 | 2.50s / 648 |
| GPU-4 overlap-nucdb | 0.64s / 462 | 1.02s / 363 | 2.23s / 648 |

GPU nhmmer now supports a **skip-Forward optimization**: when `--gpu-fwd-prefilter` gates windows through the GPU Forward pre-filter, workers use `p7_pli_postFwd_LongTarget()` which injects GPU-precomputed xf matrices and skips redundant CPU Forward recomputation. Only Backward + domain definition + hit reporting run on CPU. Without `--gpu-fwd-prefilter`, the standard `p7_pli_postViterbi_LongTarget()` path runs full CPU Forward+Backward.

GPU domain rescoring uses batched CUDA kernels (Forward+Backward+Decoding+OptimalAccuracy+OATrace+Domcorrection) with cross-window batching and trim batching. Domain rescoring uses nj=0 (unihit mode) matching CPU behavior. Remaining gap vs CPU-4 is from CUDA init overhead (~0.5s) plus CPU workers (envelope-finding, 44-93% of pipeline). The overlap-nucdb format eliminates FASTA parsing and enables GPU-resident SSV (zero per-chunk H2D). GPU-4 overlap-nucdb is 1.7-3.6x faster than CPU-1 for short models. Remaining hit count differences (0-3 hits, <1%) are from float32 vs double precision in Forward/Backward accumulation.

## GPU Architecture Summary

The GPU path accelerates SSV/MSV + biased-composition filter + Viterbi + Forward in CUDA batches. Current state:

- Default MSV path: **fused SSV+null+bias+F1 gate kernel** (`cuda_ssv_null_bias_gate_kernel<STRIDE, WARPS>`) computes all pre-filter stages in a single kernel launch. Templated on STRIDE and WARPS-per-block (one sequence per warp) with linear rbv layout for coalesced access. Survivors compacted via atomicAdd with in-kernel float score output (D2H only ~32KB/batch vs 1.8MB). Supports both resident-database and chunk-based paths.
- GPU Viterbi opt path: `cuda_viterbi_opt_kernel<STRIDE, WARPS>`, same multi-warp-per-block layout (one sequence per warp). W is auto-tuned at runtime from `cudaGetDeviceProperties` (typically W=4 on sm_89), overridable via `--gpu-ssv-warps` / `--gpu-vit-warps`.
- All GPU stages enabled by default with `--gpu`: fused SSV+null+bias+gate → Viterbi prefilter (M≤2048) → Forward prefilter (M≤2044) → FB parser (hmmsearch), Forward prefilter → FB parser (nhmmer)
- Latest all-13 profmark (multi-query single-process): **~7.3x vs CPU-1**, exact hit parity (`cpu_only=0`, `gpu_only=0`)
- Survivor loop: sorted by sequence length (co-sorting scores/statuses) with ReconfigLength caching; CPU MSV fallback eliminated (double-precision GPU bias is authoritative)
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
--gpu-fwd-largem       Allow GPU Forward on large models M>2044 (requires --gpu)
--gpu-ssv-compare      Debug: compare SSV scores to monolithic MSV
--gpu-ssv-warps        Warps per block for fused SSV kernel (0=auto)
--gpu-vit-warps        Warps per block for Viterbi opt kernel (0=auto)
```

Nhmmer-specific expert flags (defined in `src/nhmmer.c`):

```
--gpu-fwd-prefilter    GPU Forward pre-filter: skip-Forward optimization (docgroup 12)
--gpu-batch            GPU batch SSV/bias filtering on merged windows
--gpu-vit-prefilter    GPU Viterbi as pre-filter before scanning Viterbi
--gpu-vit-longtarget   GPU scanning Viterbi for sub-window detection
--gpu-cpu-postmsv      Bypass GPU scanning Viterbi/Fwd; use CPU postSSV
--gpu-compare          Debug: compare GPU filter scores to CPU per stage
--gpu-chunk-size N     Chunk size in residues (default 65536)
--gpu-device N         CUDA device id
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
| [nucleotide-benchmark.md](agents_docs/nucleotide-benchmark.md) | Nucleotide benchmark setup: chr22 target, query HMMs, baseline timings |
| [nhmmer-gpu-progress.md](agents_docs/nhmmer-gpu-progress.md) | GPU nhmmer architecture, file map, benchmark results, performance analysis |
| [nhmmer-gpu-todo.md](agents_docs/nhmmer-gpu-todo.md) | GPU nhmmer phase checklist (all phases complete), known issues |
| [nhmmer-gpu-perf-gap.md](agents_docs/nhmmer-gpu-perf-gap.md) | Why GPU is slower than CPU-4, root causes, what would fix it |

## Code Navigation

For code changes, read in this order:

1. `src/hmmer.h` — shared structs, API, feature gates
2. The relevant CLI entrypoint (`src/hmmsearch.c`, `src/nhmmer.c`, etc.)
3. The owning implementation module (`p7_pipeline.c`, `p7_domaindef.c`, etc.)
4. `src/Makefile.in` — confirms whether code is library, program, test, or benchmark
5. For GPU: `src/hmmsearch_gpu.c` (orchestration) → `src/cuda/p7_cuda.h` → individual `.cu` files
