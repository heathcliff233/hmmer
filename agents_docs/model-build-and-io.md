# Model Build And I/O

Model construction and HMM file handling are shared by almost every program. Small changes can affect searches, scans, utilities, daemon caches, and compatibility with old HMM files.

## `hmmbuild`

`src/hmmbuild.c` is the CLI entrypoint for profile HMM construction from MSAs. It configures `P7_BUILDER`, opens MSA input through Easel, and writes HMMs through `p7_hmmfile_Write*()` paths.

Key builder concerns:

- alphabet selection: autodetect, `--amino`, `--dna`, `--rna`;
- relative sequence weighting: PB, GSC, BLOSUM, none, given;
- effective sequence weighting;
- priors: default mixture Dirichlet, none, Laplace;
- calibration and E-value parameters;
- model mask and consensus handling;
- DNA/RNA max-length and window options: `--w_beta`, `--w_length`;
- optional post-processed MSA output.

Implementation files:

- `p7_builder.c`: `P7_BUILDER`, `p7_Builder()`, `p7_SingleBuilder()`, configuration.
- `build.c`: MSA-to-HMM construction mechanics.
- `seqmodel.c`: single-sequence model construction for `phmmer` and related paths.
- `p7_prior.c`, `eweight.c`, `modelconfig.c`, `modelstats.c`: priors, weighting, model configuration, calibration/statistics.

## Single-Sequence Build

`phmmer` and some `nhmmer` query paths build a model from an `ESL_SQ` with `p7_SingleBuilder()`. This is not equivalent to reading an MSA with one sequence in every detail because score-system setup, background composition, and naming are handled by the single-sequence path.

## HMM Files

`src/p7_hmmfile.c` handles HMM file opening, format detection, text/binary parsing, legacy compatibility, pressed database side files, offsets, and MPI serialization support.

Be careful with:

- alphabet compatibility across multi-HMM files,
- old HMMER2/HMMER3 format conversion,
- `hmm->offset` for indexing/fetching,
- `hmm->max_length` used by nucleotide long-target search,
- statistical parameters and cutoff flags.

## Pressed Databases

`hmmpress` writes four files:

- `.h3m`: binary core HMMs,
- `.h3f`: binary optimized profiles, MSV filter part only,
- `.h3p`: binary optimized profiles, remainder excluding the MSV filter part,
- `.h3i`: SSI index for retrieval from `.h3m`.

`hmmscan` and `nhmmscan` depend on this split for staged reads. A format or offset change must be checked against both scan programs and `hmmfetch`.

## Utilities

- `hmmfetch`: indexed or sequential HMM retrieval.
- `hmmconvert`: format conversion, including legacy HMM handling.
- `hmmemit`: emits sequences from HMMs.
- `hmmalign`: aligns sequences to HMMs using traces and profile configuration.
- `hmmstat`: reports model summaries and statistics.
- `hmmlogo`: computes data for profile logos.
- `alimask`: masks alignment columns before or around model-building workflows.

Use utility-specific tests in `testsuite/` when modifying output format, file parsing, or compatibility behavior.
