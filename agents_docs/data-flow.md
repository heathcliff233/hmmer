# Data Flow

HMMER's common path moves biological sequence/alignment data into HMMER model objects, then into profile/scoring objects, then through filters/DP, then into hit reporting.

## Shared Model/Search Flow

1. Input data comes from Easel:
   - `ESL_MSA` for multiple sequence alignments.
   - `ESL_SQ` for single query or target sequences.
   - `ESL_SQFILE`, `ESL_MSAFILE`, and `ESL_ALPHABET` own I/O and alphabet detection.
2. Model construction uses `P7_BUILDER`:
   - `p7_Builder()` builds from `ESL_MSA`.
   - `p7_SingleBuilder()` builds a profile from one `ESL_SQ`.
   - weighting, priors, calibration, effective sequence count, model mask, consensus, and max-length/window parameters are resolved here.
3. The core model is `P7_HMM`.
4. `P7_PROFILE` is configured from the HMM and target length/model length assumptions for generic scoring.
5. `P7_OPROFILE` is created for the selected vector implementation and configured for MSV/SSV, Viterbi, and Forward stages.
6. `P7_PIPELINE` coordinates filters and DP:
   - MSV/SSV,
   - bias filter,
   - Viterbi,
   - Forward/Backward,
   - posterior decoding/domain definition,
   - null2 correction,
   - thresholds and counters.
7. `P7_DOMAINDEF` stores reusable domain-definition state and produces domain coordinates, envelopes, posterior traces, and alignment displays.
8. `P7_TOPHITS` stores ranked target/domain hits, applies reporting/inclusion thresholds, removes nucleotide duplicates when needed, merges worker results, and writes output.
9. Output functions emit text, tabular target/domain tables, Xfam-style tables, alignment/MSA output, and tail metadata.

## Search Sequence

Protein sequence/profile searches typically call:

- CLI entrypoint parses options and opens query/target files.
- HMM or single sequence query becomes `P7_HMM`, `P7_PROFILE`, `P7_OPROFILE`, and `P7_BG`.
- Worker-local `P7_PIPELINE` and `P7_TOPHITS` are created.
- Each target `ESL_SQ` runs through `p7_Pipeline()`.
- Worker results merge with `p7_tophits_Merge()` and `p7_pipeline_Merge()`.
- `p7_tophits_Threshold()`, sorting, and output formatting produce final reports.

## Scan Sequence

Scan mode inverts the outer loops:

- Query sequences are read from Easel.
- Pressed HMM databases are opened through `P7_HMMFILE`.
- The MSV-only optimized profile is read first from `.h3f`.
- If a target/model passes early filters, the rest of the optimized profile is read later from `.h3p`.
- Hits are reported with `P7_PIPELINE` mode `p7_SCAN_MODELS`, which changes labels, lengths, and table semantics.

## Long-Target Sequence

Nucleotide long-target search uses windows instead of treating a whole chromosome-like target as one ordinary sequence:

- Query HMMs get max-length/window metadata from `P7_BUILDER` or command-line overrides.
- `P7_SCOREDATA` gives prefix/suffix and diagonal data used to extend seeds to plausible windows.
- `P7_HMM_WINDOWLIST` carries SSV/MSV and Viterbi windows.
- Each final window is rescored with the normal later stages and added to `P7_TOPHITS`.
- `p7_tophits_ComputeNhmmerEvalues()` and duplicate removal are part of final nucleotide reporting.

## Ownership Hints

- Construction bugs usually start in `hmmbuild.c`, `p7_builder.c`, `build.c`, `seqmodel.c`, `p7_prior.c`, `modelconfig.c`, or `modelstats.c`.
- HMM read/write bugs usually start in `p7_hmmfile.c`, then `hmmpress.c`, `hmmfetch.c`, or `hmmconvert.c`.
- Search behavior usually starts in the CLI file plus `p7_pipeline.c`, `p7_domaindef.c`, `p7_tophits.c`, and the relevant DP/filter implementation.
- Long-target behavior usually starts in `nhmmer.c`, `nhmmscan.c`, `p7_pipeline.c`, `generic_msv.c`, `generic_viterbi.c`, `fm_ssv.c`, and `p7_scoredata.c`.
