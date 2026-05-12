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

`nhmmer --gpu` now requires an overlap `.nucdb` target (default overlap from
`hmmnucdb`). FASTA targets and no-overlap `.nucdb` targets are rejected; there
is no longer a CPU fallback inside the GPU path.

```sh
# Build an overlap .nucdb (hmmnucdb defaults to --overlap 2001)
src/hmmnucdb target.fa target.nucdb
src/nhmmer --gpu --cpu 16 --noali query.hmm target.nucdb.nucdb
```

### Running nhmmer GPU Benchmarks

```sh
# Full timing comparison (CPU-1, CPU-N, GPU-N overlap-nucdb)
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
- **Hit parity**: GPU nhmmer path preserves hit parity with CPU. Combined chr22 benchmark with 16 threads: all modes report 1476 hits across MADE1 + query_short + query_medium (4/4 parity checks passed). On 5x chr22: 6294 hits GPU = CPU. Remaining 1-hit differences on individual queries come from float32 vs double precision in Forward/Backward accumulation.
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

Latest combined benchmark (all three queries, 16 CPU threads, 5 runs, audit 2026-05-12). Script `test-speed/x-nhmmer-gpu-bench` uses process-lifetime wall bracketed by `date +%s%N` around each `nhmmer` invocation; cross-checked against `/usr/bin/time -f "%e"`.

chr22 (50MB):

| Path | Hits | min | median |
|------|:---:|:---:|:---:|
| CPU-1 | 1476 | 8.487s | — |
| CPU-16 | 1476 | 0.963s | 1.077s |
| GPU-16 overlap `.nucdb` | 1476 | 0.968s | 1.023s |

On chr22 the GPU result is essentially a tie with CPU-16 (median ~5% faster, within run-to-run jitter).

chr22x5 (~250MB, 5x chr22):

| Path | Hits | min | median |
|------|:---:|:---:|:---:|
| CPU-16 | 6294 | 3.882s | 4.019s |
| GPU-16 overlap `.nucdb` | 6294 | 2.715s | 2.924s |

GPU-16 is ~27% faster than CPU-16 on chr22x5 median. The async overlap win grows with dataset size because there are enough strands to keep both slots busy through the run.

`hmmnucdb` defaults to overlap chunking (`--overlap 2001`), required by `nhmmer --gpu`; use `--overlap 0` only for diagnostic databases that the GPU path won't consume. `nhmmer --gpu` unconditionally runs GPU batch F1 filtering, scanning Viterbi (FromF1), exact-F3 Forward prefilter, and GPU Forward/Backward parser handoff, with all filter/parser sequence data resident on device. Pre-2026-05-12 the path was gated by `--gpu-batch` / `--gpu-vit-longtarget` / `--gpu-fwd-prefilter` / `--gpu-no-fwd-prefilter` / `--gpu-cpu-postmsv` / `--gpu-compare` / `--gpu-chunk-size`; those flags were removed along with their branches.

The outer loop is now a 2-slot double-buffered strand pipeline: the main thread drives strand N+1's GPU pipeline while strand N's CPU domain workers finish in background threads. On chr22x5/query_medium, the breakdown shows `CPU workers 1.762s` with `main-thread wait 0.144s` and `overlap saved 1.618s` — roughly all of the worker time is hidden behind the next strand's GPU kernels. The remaining bottleneck is the unavoidable GPU kernel wall (SSV + scanning Viterbi dominate); see `agents_docs/nhmmer-gpu-todo.md` for the remaining intra-stage async opportunities.

## GPU Architecture Summary

The GPU path accelerates SSV/MSV + biased-composition filter + Viterbi + Forward in CUDA batches. Current state:

- Default MSV path: **fused SSV+null+bias+F1 gate kernel** (`cuda_ssv_null_bias_gate_kernel<STRIDE, WARPS>`) computes all pre-filter stages in a single kernel launch. Templated on STRIDE and WARPS-per-block (one sequence per warp) with linear rbv layout for coalesced access. Survivors compacted via atomicAdd with in-kernel float score output (D2H only ~32KB/batch vs 1.8MB). Supports both resident-database and chunk-based paths.
- GPU Viterbi opt path: `cuda_viterbi_opt_kernel<STRIDE, WARPS>`, same multi-warp-per-block layout (one sequence per warp). W is auto-tuned at runtime from `cudaGetDeviceProperties` (typically W=4 on sm_89), overridable via `--gpu-ssv-warps` / `--gpu-vit-warps`.
- All GPU stages enabled by default with `--gpu`: fused SSV+null+bias+gate → Viterbi prefilter (M≤2048) → Forward prefilter (M≤2044) → FB parser (hmmsearch); SSV longtarget → batch F1 gate → scanning Viterbi → Forward prefilter → FB parser (nhmmer). SSV longtarget kernel uses the linear `d_rbv_lin[x * M + k]` layout (same as protein fused SSV), eliminating shared memory lookup tables and simplifying inner-loop address math. Per-query `ssv_scores` upload is cached across strands/sequences.
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
--gpu-device N           CUDA device id
```

The prior `--gpu-chunk-size`, `--gpu-batch`, `--gpu-vit-longtarget`,
`--gpu-fwd-prefilter`, `--gpu-no-fwd-prefilter`, `--gpu-cpu-postmsv`, and
`--gpu-compare` flags were removed together with the non-default GPU nhmmer
paths (FASTA, no-overlap `.nucdb`, CPU-continuation post-SSV). `nhmmer --gpu`
unconditionally runs the resident-overlap-`.nucdb` pipeline: SSV longtarget →
batch F1 gate → scanning Viterbi (FromF1) → GPU Forward/Backward parser →
CPU domain workers.

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
| [nhmmer-gpu-perf-gap.md](agents_docs/nhmmer-gpu-perf-gap.md) | GPU vs CPU-16 performance gap analysis, root causes, remaining bottlenecks |

## Code Navigation

For code changes, read in this order:

1. `src/hmmer.h` — shared structs, API, feature gates
2. The relevant CLI entrypoint (`src/hmmsearch.c`, `src/nhmmer.c`, etc.)
3. The owning implementation module (`p7_pipeline.c`, `p7_domaindef.c`, etc.)
4. `src/Makefile.in` — confirms whether code is library, program, test, or benchmark
5. For GPU: `src/hmmsearch_gpu.c` (orchestration) → `src/cuda/p7_cuda.h` → individual `.cu` files
