# Nucleotide And FM-Index

Nucleotide search is a first-class HMMER subsystem. `nhmmer` and `nhmmscan` use the long-target pipeline and have distinct coordinate, strand, window, and E-value behavior.

## `nhmmer` Target Detection

`src/nhmmer.c` distinguishes ordinary sequence targets from FM-index targets:

- ordinary sequence formats use Easel `ESL_SQFILE` reading,
- FM-index databases use `eslSQFILE_FMINDEX`,
- `-` stdin is only valid for sequence input, not FM-index input,
- FM-index mode reads metadata/data into `FM_CFG`/`FM_DATA` and sends windows into `p7_Pipeline_LongTarget()`.

When query input is an MSA or sequence, `nhmmer` builds a nucleotide HMM first. It also applies `--watson`, `--crick`, or both-strand defaults through `P7_PIPELINE.strands`.

## `nhmmscan`

`nhmmscan` searches DNA/RNA query sequences against pressed nucleotide profile databases. It uses `p7_SCAN_MODELS` plus `long_targets=TRUE`. It scans both strands unless options restrict the search and the alphabet supports complements.

Unlike `nhmmer`, `nhmmscan` is not an FM-index target search path; it reads pressed HMM profiles and applies the long-target pipeline to query sequences.

## FM-Index Database Build

`makehmmerdb` builds FM-index target databases:

- reads sequence blocks with Easel,
- records sequence and ambiguity metadata,
- uses `libdivsufsort` for suffix arrays,
- writes BWT/FM-index data and suffix-array samples,
- writes metadata followed by FM data blocks,
- can build forward-only indexes, though HMMER search normally needs both directions.

Start with `src/makehmmerdb.c`, then `src/fm_general.c`, `src/fm_alphabet.c`, and `libdivsufsort/`.

## FM Structs

Defined in `src/hmmer.h`:

- `FM_METADATA`: alphabet, block, sequence, ambiguity, and sampling metadata for FM-index files.
- `FM_DATA`: loaded FM-index data, BWT, sampled suffix array, occurrence counts, and text data.
- `FM_CFG`: runtime FM-index thresholds/configuration and `FM_METADATA`.
- `FM_DIAGLIST`: dynamic list of diagonal seeds from FM traversal.
- `P7_SCOREDATA`: optimized profile-derived data used to score and extend SSV/MSV diagonals.
- `P7_HMM_WINDOWLIST`: sequence windows promoted from FM/SSV/MSV stages into later pipeline stages.

## FM Search Flow

1. `nhmmer` opens the FM-index target and initializes `FM_CFG`.
2. Forward/backward `FM_DATA` blocks are read.
3. `p7_SSVFM_longlarget()` finds high-scoring seeds with FM traversal.
4. Seeds are extended to diagonals, converted to sequence windows, and stored in `P7_HMM_WINDOWLIST`.
5. `p7_pli_ExtendAndMergeWindows()` expands/merges windows using `P7_SCOREDATA`.
6. `p7_Pipeline_LongTarget()` extracts sequence windows, rescoring them through later filters/DP and reporting hits.

## SSE Constraint

FM-index acceleration depends on SSE-oriented SSV/MSV score data and vector profile behavior. Treat FM-index performance code as tied to the optimized implementation, not as a portable generic replacement. If changing configure feature gates or non-SSE implementations, check FM-index availability and build assumptions explicitly.

## Coordinate Risks

FM-index code converts between FM coordinates, segment coordinates, original sequence coordinates, complement coordinates, and output coordinates. High-risk areas:

- windows crossing FM block or original sequence boundaries,
- complement strand conversion,
- overlap removal between adjacent blocks,
- duplicate removal after both-strand search,
- tabular output fields for `hmmfrom`, `hmm to`, `alifrom`, `ali to`, envelope start/end, strand, and sequence/model length.
