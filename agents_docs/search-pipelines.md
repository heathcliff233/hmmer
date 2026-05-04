# Search Pipelines

`src/p7_pipeline.c` owns the central accelerated comparison workflow. It is shared by ordinary protein searches, pressed database scans, nucleotide long-target searches, and daemon paths.

## Normal Pipeline

The ordinary pipeline is used by protein profile search, protein sequence search after model construction, and much of scan mode. The high-level stages are:

1. Configure `P7_PIPELINE`, `P7_OPROFILE`, and `P7_BG` for the current model and sequence length.
2. Run MSV filter.
3. Apply biased-composition filter unless disabled.
4. Run Viterbi filter.
5. Run Forward, and when needed Backward.
6. Decode posteriors and define domains through `P7_DOMAINDEF`.
7. Apply null2 correction.
8. Add target/domain records to `P7_TOPHITS`.
9. Apply reporting and inclusion thresholds.

The exact thresholds are controlled by `--F1`, `--F2`, `--F3`, `-E`, `-T`, domain thresholds, inclusion thresholds, bit cutoffs, and `--max`/bias options. A threshold change often affects printed counters, daemon stats, and tests that inspect outputs.

## Domain Definition

`P7_DOMAINDEF` is reusable state, not a final result container. It manages posterior decoding, envelopes, domain clustering, null2 correction, and alignment display construction. Start in `p7_domaindef.c` for boundary or posterior changes; check `p7_alidisplay.c`, `p7_tophits.c`, and tabular/text outputs after changing it.

## Long-Target Pipeline

Long-target search uses `p7_Pipeline_LongTarget()` and window lists:

1. Create an initial SSV/MSV window list.
2. If an FM-index target is present, call `p7_SSVFM_longlarget()` in `fm_ssv.c`.
3. Otherwise, call long-target SSV/MSV over sequence data.
4. Extend and merge windows with `p7_pli_ExtendAndMergeWindows()`.
5. Split or adjust windows that cross sequence boundaries or exceed maximum window length.
6. Rescore each surviving window with bias, MSV, Viterbi, Forward/Backward, and domain/hit reporting.
7. Track long-target counters such as positions past MSV/Viterbi/Forward.

Important files:

- `p7_pipeline.c`: orchestration, thresholding, long-target windows, final scoring.
- `generic_msv.c`: generic long-target MSV window discovery.
- `generic_viterbi.c`: long-target Viterbi window discovery.
- `fm_ssv.c`: FM-index SSV seeding, diagonal extension, and FM-to-sequence window conversion.
- `p7_scoredata.c`: compact score data for diagonal/window decisions.
- `p7_hmmwindow.c`: window-list allocation and reuse.

## Nucleotide E-Values And Duplicates

Long-target nucleotide reporting differs from ordinary protein reporting. `p7_tophits_ComputeNhmmerEvalues()` computes E-values from p-values and standardized window lengths. `p7_tophits_RemoveDuplicates()` removes duplicate windows/hits, and its behavior depends on sort keys and bit-cutoff mode.

Changing long-target coordinates, strands, or window merging requires checking text output, tabular output, duplicate removal, and E-value computation together.

## Scan Mode Staged Reads

`hmmscan` and `nhmmscan` use pressed HMM databases. The file set is:

- `.h3m`: core model,
- `.h3f`: optimized profile data needed for MSV filtering,
- `.h3p`: rest of optimized profile,
- `.h3i`: SSI index.

The pipeline mode is `p7_SCAN_MODELS`, so output labels and lengths use model-scan semantics. Do not reuse `p7_SEARCH_SEQS` assumptions in scan-mode table changes.
