# Program Families

The installed program list is defined in `src/Makefile.in`. Treat each CLI file as an orchestration layer around shared HMMER library code.

## Protein Profile Search: `hmmsearch`

`hmmsearch` searches HMM queries against a protein sequence database. It opens HMM input through `P7_HMMFILE`, creates profile/background/pipeline objects, and runs target sequences through `p7_Pipeline()` in `p7_SEARCH_SEQS` mode.

Start with `src/hmmsearch.c`, then inspect `p7_pipeline.c`, `p7_tophits.c`, `p7_domaindef.c`, HMM file/profile code, and the relevant DP/filter implementation.

The opt-in `hmmsearch --gpu` path is protein-only and requires a target database built by `hmmseqdb` as Easel protein `dsqdata`. It batches `dsqdata` targets through CUDA MSV plus biased-composition filtering, then continues accepted survivors through the existing CPU pipeline. See `gpu-support-progress.md`.

## Protein Model Scan: `hmmscan`

`hmmscan` searches protein query sequences against a pressed HMM database. It relies on `hmmpress` output files and staged reads: the MSV filter profile first, then the rest of the optimized profile only for models that need later stages.

Start with `src/hmmscan.c`, `src/p7_hmmfile.c`, `src/hmmpress.c`, and `p7_pipeline.c`.

## Protein Sequence Search: `phmmer`

`phmmer` searches a protein sequence query against a protein database by building a single-sequence model with `p7_SingleBuilder()`. After construction, it follows the ordinary sequence search pipeline.

Start with `src/phmmer.c`, then `seqmodel.c`, `p7_builder.c`, `p7_pipeline.c`, and `p7_tophits.c`.

## Iterative Protein Search: `jackhmmer`

`jackhmmer` repeats search, hit inclusion, alignment construction, and model rebuilding. Changes can cross search thresholds, inclusion logic, alignment output, and builder behavior.

Start with `src/jackhmmer.c`, then `p7_tophits.c`, `tracealign.c`, `p7_builder.c`, and `p7_pipeline.c`.

## Nucleotide Long-Target Search: `nhmmer`

`nhmmer` is a first-class subsystem, not a protein search wrapper. It accepts HMM, MSA, or sequence queries; builds nucleotide HMMs when needed; searches DNA/RNA targets; handles top/bottom/both-strand logic; windows long targets; supports FM-index target databases; and recomputes nhmmer-style E-values from window sizes.

Start with `src/nhmmer.c`, then `p7_pipeline.c`, `generic_msv.c`, `generic_viterbi.c`, `fm_ssv.c`, `p7_scoredata.c`, `p7_hmmwindow.c`, and `p7_tophits.c`.

## Nucleotide Scan: `nhmmscan`

`nhmmscan` scans DNA/RNA query sequences against pressed DNA profile databases using the long-target pipeline in scan mode. It uses staged pressed HMM reads like `hmmscan`, but passes query sequence windows through `p7_Pipeline_LongTarget()`.

Start with `src/nhmmscan.c`, then `p7_hmmfile.c`, `p7_pipeline.c`, `p7_scoredata.c`, `p7_tophits.c`, and `hmmpress.c`.

## HMM Database Preparation: `hmmpress`

`hmmpress` turns an HMM file into pressed database files:

- `.h3m`: binary core HMMs,
- `.h3f`: MSV filter part of optimized profiles,
- `.h3p`: remaining optimized profile data,
- `.h3i`: SSI index for retrieval.

Start with `src/hmmpress.c` and `src/p7_hmmfile.c`.

## FM-Index Database Preparation: `makehmmerdb`

`makehmmerdb` builds target databases for accelerated nucleotide long-target search. It uses `libdivsufsort` to construct suffix arrays/BWT/FM-index data and writes metadata plus FM blocks.

Start with `src/makehmmerdb.c`, `src/fm_general.c`, `src/fm_alphabet.c`, and `libdivsufsort/`.

Do not reuse `makehmmerdb` for the protein GPU sequence database builder. The HMMER-facing command is `hmmseqdb`, which writes GPU-capable Easel protein `dsqdata`; nucleotide/FM-index GPU metadata belongs to later `nhmmer` work.

## GPU Sequence Database Preparation: `hmmseqdb`

`hmmseqdb` builds GPU-capable Easel protein `dsqdata` target databases from FASTA input. This is the required input format for `hmmsearch --gpu`.

Start with `src/hmmseqdb.c`. The output format uses Easel's `dsqdata` chunked binary layout with the chunk-sizing patch from `patches/easel-dsqdata-open-sized.patch`.

Usage: `src/hmmseqdb target.fa target.gpudb`

## Daemon Search

`hmmpgmd` and `hmmpgmd_shard` serve searches from cached sequence/model databases. This path has its own command/status serialization, worker code, and cache integration while still using common profile/pipeline/tophits objects.

Start with `src/hmmpgmd.c`, `src/hmmpgmd.h`, `src/hmmdmstr.c`, `src/hmmdwrkr.c`, `src/hmmdutils.c`, `src/cachedb.c`, `src/p7_hmmcache.c`, and the shard variants when relevant.

## Model Utilities

- `hmmalign`: align sequences to HMMs.
- `hmmemit`: emit sequences from HMMs.
- `hmmfetch`: retrieve models from HMM files/databases.
- `hmmconvert`: convert HMM formats.
- `hmmstat`: summarize HMM files.
- `hmmlogo`: produce HMM logo data.
- `alimask`: mask alignment columns.
- `hmmsim`: simulation/statistics support.

Each utility has a matching `src/<program>.c`; follow from there to shared `p7_hmmfile.c`, `p7_hmm.c`, builder, trace, or output modules as needed.
