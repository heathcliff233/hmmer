# Parallel, Daemon, And Cache Paths

HMMER supports serial, POSIX-threaded, MPI, and daemon execution. Parallel code usually keeps worker-local pipeline and hit-list objects, then merges results.

## Thread Patterns

The major search programs use Easel threads/work queues when built with `HMMER_THREADS`:

- `hmmsearch`,
- `hmmscan`,
- `phmmer`,
- `jackhmmer`,
- `nhmmer`,
- `nhmmscan`,
- `hmmbuild` for parallel model construction.

Common pattern:

- parse options and create shared immutable inputs,
- create per-worker `P7_PIPELINE`, `P7_TOPHITS`, background/profile copies as needed,
- dispatch sequence or model blocks through `ESL_WORK_QUEUE`,
- call pipeline/build functions in each worker,
- merge `P7_TOPHITS` and `P7_PIPELINE` counters at the end.

Do not share mutable pipeline, matrix, top-hit, or background objects across workers unless the existing code already does so.

## MPI Patterns

MPI is gated by `HMMER_MPI`. Search programs use master/worker loops, packed messages, and helper serialization:

- `p7_pipeline_MPISend()` / `p7_pipeline_MPIRecv()`,
- `p7_tophits_MPISend()` / `p7_tophits_MPIRecv()`,
- HMM/MSA/sequence serialization through Easel/HMMER helpers.

MPI code often duplicates serial/thread logic. When changing option parsing, thresholds, output counters, or result merging, inspect MPI paths before assuming behavior is consistent.

## Daemon Programs

Daemon entrypoints:

- `hmmpgmd`,
- `hmmpgmd_shard`.

Important files:

- `hmmpgmd.c`, `hmmpgmd.h`: daemon process and shared definitions.
- `hmmdmstr.c`, `hmmdmstr_shard.c`: master-side coordination.
- `hmmdwrkr.c`, `hmmdwrkr_shard.c`: worker-side search.
- `hmmdutils.c`: options, utilities, and shared setup.
- `hmmd_search_status.c`, `p7_hmmd_search_stats.c`: command/status/stat tracking.
- `hmmpgmd2msa.c`: daemon hit/alignment conversion support.

Daemon code depends on pthreads; some files explicitly error when pthread support is absent.

## Sequence And HMM Caches

Cache files:

- `cachedb.c`, `cachedb.h`,
- `cachedb_shard.c`, `cachedb_shard.h`,
- `p7_hmmcache.c`, `p7_hmmcache.h`.

Core objects:

- `P7_SEQCACHE`: sequence database cache used by daemon search paths.
- HMM cache objects in `p7_hmmcache.*`: model/profile cache integration.

Cache changes can affect memory lifetime, daemon startup, shard routing, and search result identity. Check both regular and shard daemon tests when changing these paths.

## Verification Targets

Relevant integration tests include `testsuite/i19-hmmpgmd-ga.pl` and `testsuite/i22-hmmpgmd-shard-ga.pl`. For non-daemon parallel behavior, search-specific integration scripts plus `make check` are the normal verification path.

## GPU And Threading

The current `hmmsearch --gpu` path uses `--cpu 0` semantics (single-threaded host with GPU offload). It does not use Easel work queues for GPU batch dispatch. The serial dsqdata loop in `src/hmmsearch_gpu.c` reads chunks, packs GPU batches, launches CUDA kernels, and processes survivors sequentially. Multi-GPU or host-thread-parallel GPU dispatch is deferred.
