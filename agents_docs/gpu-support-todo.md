# GPU Support TODO

This is a planning document for future GPU work. Cross-check implemented behavior against `agents_docs/gpu-support-progress.md` and the code before relying on any older milestone wording here.

## Current Direction

The first GPU milestone was an opt-in CUDA acceleration path for protein `hmmsearch`, initially focused on MSV. The current efficiency pass is evaluating later pipeline stages one by one and only keeping stages that are suitable for GPU and fast enough on `profmark`.

Locked decisions:

- Use native CUDA, not Triton. HMMER is a C/autotools CLI codebase, and CUDA can be called from C through a small C ABI around `.cu` implementation files.
- Do not introduce CMake for GPU work in this repository. External CUDA HMM projects may be inspected for high-level algorithm and implementation ideas, but HMMER GPU code must integrate through the existing autotools build. Borrow ideas only; do not copy code, vendor external sources, link external libraries, build the external project here, or add external build support.
- Add GPU code under a new additive `src/cuda/` area. Do not make it the active `impl_*` backend and do not replace the selected SSE/NEON/VMX implementation.
- Start with `hmmsearch --gpu`; leave `phmmer`, `jackhmmer`, `hmmscan`, daemon paths, and nucleotide programs for later milestones.
- Accelerate full MSV-compatible filtering first, not SSV-only.
- Keep Viterbi, Forward/Backward, domain definition, null2, hit storage, thresholding, and output on CPU until each is evaluated. Bias filtering has a CUDA batch prototype in the current GPU path because its score boundary is narrow and it can reuse the MSV-uploaded sequence batch.
- Do not change the pressed HMM database files (`.h3m/.h3i/.h3f/.h3p`) in v1.
- Treat the old `origin/cuda` branch as reference material only. It targets a substantially different codebase/version and should not be ported mechanically.

## Current Codebase Facts

The normal protein sequence-search path is centered on `p7_Pipeline()` in `src/p7_pipeline.c`.

For ordinary `hmmsearch`, the relevant early sequence of work is:

1. `hmmsearch.c` reads target sequences as `ESL_SQ`/`ESL_SQ_BLOCK`.
2. The caller configures `P7_BG` and `P7_OPROFILE` for each target length.
3. `p7_Pipeline()` computes the base null score with `p7_bg_NullOne()`.
4. `p7_Pipeline()` calls `p7_MSVFilter()`.
5. The host computes the MSV bit score and Gumbel P-value and applies `pli->F1`.
6. GPU mode computes the biased-composition filter score in batch after MSV, then survivors continue through later CPU stages.

The MSV implementation is part of the selected optimized implementation. In the SSE path, `src/impl_sse/msvfilter.c:p7_MSVFilter()` first tries `p7_SSVFilter()` and falls back to full byte MSV DP when SSV cannot produce a result. The byte profile conversion rules live in the optimized profile implementation, for example `src/impl_sse/p7_oprofile.c`.

`P7_OPROFILE` is intentionally implementation-specific. A CUDA MSV profile should therefore be a separate GPU-specific object built from generic/profile data, not a flattened or copied SSE/NEON/VMX layout.

## Intended GPU Design

Add a small CUDA runtime layer with C-callable entry points. The HMMER C side should own command-line options, input validation, pipeline accounting, hit reporting, and CPU continuation. The CUDA side should own device memory, CUDA streams/events, profile upload, sequence batch upload, and MSV kernels.

Planned objects and interfaces:

- `P7_CUDA_MSVPROFILE`: GPU-specific MSV profile data derived from generic/profile scores and MSV byteification rules.
- `P7_CUDA_ENGINE` or equivalent runtime object: device selection, stream ownership, buffers, and batch limits.
- `P7_CUDA_MSV_RESULT`: per-sequence status and score data sufficient for the host to reconstruct `usc`, detect overflow/high-score pass cases, and decide whether the sequence passed `F1`.

Initial kernel strategy:

- Map one warp to one sequence/profile comparison.
- Implement correctness-first full byte MSV DP using shared memory for the byte shift.
- Return enough integer state to reconstruct the CPU-style MSV score on host.
- Treat MSV overflow/high-score behavior as a pass into downstream CPU stages.
- Tune later with warp shuffle instructions and alternative block-level mapping only after CPU/GPU parity is proven.

The host should compute or reconstruct:

- sequence null score;
- MSV `usc`;
- bias filter pass/fail from the GPU-provided filter score;
- MSV bit score;
- MSV Gumbel P-value;
- `pli->F1` pass/fail;
- pipeline counters.

Avoid GPU floating-point decisions in v1 where host reconstruction can preserve CPU behavior more closely.

## Pipeline Integration TODO

Split `p7_Pipeline()` without changing its public CPU behavior:

- keep the existing `p7_Pipeline()` as the CPU reference path;
- factor the downstream work after a successful MSV filter into a helper that accepts known `nullsc` and `usc`;
- call that helper from both the existing CPU path and the future `hmmsearch --gpu` survivor path.

The post-MSV helper must preserve:

- zero-length and over-limit sequence behavior;
- bias filter behavior and `--nobias`/`--max` semantics, including the split helper that accepts a precomputed filter score;
- scan/search mode distinctions, even though v1 GPU should use search mode only;
- thresholding, counters, domain definition, and output ordering;
- existing error conventions and `pli->errbuf` usage.

For explicit `--gpu`, failures should be clear and non-silent:

- HMMER built without CUDA support;
- no usable CUDA device;
- invalid or non-GPU-capable sequence database;
- CUDA runtime failure;
- internal CPU/GPU validation mismatch in debug/test modes.

Do not silently fall back to CPU when the user explicitly requested GPU.

## Sequence Database TODO

Sequence length should be handled during sequence database construction, not discovered ad hoc during GPU search.

Use Easel `dsqdata` as the base for protein GPU sequence databases:

- add a GPU-capable v2 format or compatible extension;
- store per-sequence length in index metadata;
- preserve existing `max_seqlen`, `nseq`, `nres`, packed digitized sequence data, and sequence metadata;
- keep existing v1 readers compatible for current users;
- reject unsupported huge protein sequences at construction time with a clear diagnostic;
- expose a HMMER-facing builder command, tentatively `hmmseqdb <seqfile> <seqdb>`.

Why construction-time lengths matter:

- MSV length-dependent byte parameters need sequence length.
- Null score depends on target length.
- GPU batch planning should be possible before unpacking sequence payloads.
- Length-aware batching can reduce wasted work and improve memory transfer planning.

For v1, `hmmsearch --gpu` should require a GPU-capable protein `dsqdata` database. Ordinary FASTA should remain on the CPU path.

Current implementation status:

- `hmmseqdb` exists and wraps protein `dsqdata` construction for GPU search.
- `hmmsearch --gpu` now rejects ordinary FASTA input and expects an `hmmseqdb`-built target database.
- A build-time patch under `patches/easel-dsqdata-open-sized.patch` adjusts Easel chunk sizing for GPU runs; treat that as a bridge, not the final v2 format.
- The open design gap is a true `dsqdata` v2 length-index extension that stores per-sequence lengths in the database itself rather than relying on chunk unpacking.

Do not conflate this with `makehmmerdb`: `makehmmerdb` is currently the nucleotide/FM-index database builder for `nhmmer`. Nucleotide GPU support should later revisit `makehmmerdb` for long-target length/window metadata, but that is not part of the first protein `hmmsearch` GPU milestone.

## Build And CLI TODO

Build-system work:

- add optional CUDA detection to `configure.ac`;
- generate an appropriate `HAVE_CUDA`/`HMMER_CUDA` style feature define following local conventions;
- add `src/cuda/` build rules without disturbing selected `src/impl_*`;
- make CUDA architecture configurable instead of hardcoding a single `sm_*`;
- keep non-CUDA builds clean and warning-free.

CLI work:

- add `hmmsearch --gpu` as an explicit opt-in;
- add hidden or expert knobs for batch sizing, such as `--gpu-batch-seqs` and `--gpu-batch-res`;
- report GPU use in the `hmmsearch` header;
- make invalid combinations fail early with direct messages.

Future CLI:

- add `hmmseqdb` to build GPU-capable protein sequence databases;
- document that v1 GPU search expects `hmmseqdb`/GPU-capable `dsqdata` input.

## Validation TODO

Correctness tests should come before performance tuning.

Required test categories:

- `dsqdata` v2 write/read round trip, including per-sequence lengths.
- Compatibility tests showing existing v1 `dsqdata` readers still work.
- `hmmseqdb` smoke tests for protein FASTA input and rejection of unsupported inputs.
- CUDA MSV parity against CPU `p7_MSVFilter()` across varied model sizes, residue distributions, and sequence lengths.
- CUDA SSV shortcut parity with the optimized CPU `p7_MSVFilter()` path. The CPU SSE path first tries `p7_SSVFilter()` and accepts that score when it is safe enough, only falling back to full MSV on `eslENORESULT`; GPU v1 currently runs full MSV only, so some borderline CPU-reported hits can differ. Track this as a deferred sensitivity/parity task, not an efficiency blocker for the current MSV-only tuning pass.
- Overflow/high-score behavior parity.
- End-to-end `hmmsearch --gpu` versus CPU `hmmsearch` on small deterministic fixtures.
- Error tests for non-CUDA builds, missing devices, non-GPU-capable sequence databases, and invalid options.

Performance tests should separately measure:

- sequence DB read/unpack cost;
- host-to-device transfer cost;
- kernel throughput;
- CUDA bias H2D/kernel/D2H cost when bias is enabled;
- survivor handoff cost into CPU post-MSV stages;
- sensitivity of throughput to batch sequence count and residue count.

Current bottleneck interpretation:

- CPU survivor continuation after GPU MSV/bias is larger than CUDA kernel, transfer, and dsqdata read costs in the current profmark runs.
- Load/search batch packing is implemented. `--gpu-load-seqs`/`--gpu-load-res` control dsqdata reader chunking, while `--gpu-batch-seqs`/`--gpu-batch-res` control the CUDA search batch. The GPU path can pack multiple loaded chunks into one CUDA MSV/bias batch. Current all-13 profmark evidence says larger 12M/16M search batches reduce launch count and kernel time but do not improve wall time because CPU survivor continuation dominates; keep the 32,768 sequence / 8M residue defaults unless new profiling says otherwise.
- Null scoring is currently too small to justify moving to GPU as an isolated optimization.
- Bias filtering is suitable for GPU only when it reuses the MSV sequence batch upload; a separate bias dsq upload erased most of the benefit.
- Viterbi remains a bounded candidate because it is a scalar filter. The current CUDA prefilter has a conservative `M <= 512` gate and is useful on the prepared all-13 profmark set when combined with the Forward score and parser-state handoff. It still needs a default-policy decision because long profiles currently fall back to CPU Viterbi and the remaining sensitivity gap is upstream SSV/MSV, not Viterbi. Any future long-profile integration needs a faster long-profile design or a validated auto-gating rule, plus explicit score/overflow parity against the optimized 16-bit CPU filter.
- External dense-HMM CUDA examples are design references only, not portable kernels. The inspected `divinrkz/cuda-hmm` repository implements dense textbook HMM Viterbi/Forward/Backward, not HMMER Plan7; see `agents_docs/cuda-hmm-reference.md`.
- Do not build, vendor, copy, or add build-system support for `divinrkz/cuda-hmm` inside this worktree. CMake is therefore unnecessary for this task. The useful output from that inspection is the reference assessment and any HMMER-specific design ideas, not a dependency or source import.
- Forward/Backward is partially suitable, but the boundary is not just a score. `p7_ForwardParser()` produces the Forward score used for F3 filtering and also writes `P7_OMX` special-state rows/scale factors consumed by `p7_BackwardParser()` and `p7_domaindef_ByPosteriorHeuristics()`. The current Forward prefilter is therefore limited to very short profiles (`M <= 256`) and reruns CPU Forward near the F3 cutoff unless `--gpu-fb-parser` will compute parser state for an accepted GPU Forward score. The `--gpu-fb-parser` state-handoff prototype batches F3 survivors per search chunk and has block-parallel short-profile parser Forward/Backward kernels with dynamic shared-memory opt-in. It has prepared all-13 profmark speed evidence, but it is not an accepted default path: raw `p7X_SCALE` rows can still differ and validation beyond the prepared 13-query set is still pending.
- Parser-state validation should treat posterior/domain inputs as the meaningful sensitivity check, not raw scale rows alone. Current compare output reports `max_mocc`, `max_btot`, and `max_etot`; the all-13 compare run kept these below 0.00002 even when raw `p7X_SCALE` row differences were larger. Broader validation should require final hit parity plus bounded posterior/domain input differences on more profmark queries.
- Stage-count evidence from `benchmark-data/profmark-current/gpu-stage-suitability/counts-2/logs/profmark-gpu-summary.tsv` shows why Forward is still worth studying: on full `pmark.test.gpudb`, `14-3-3` reduced 287 Viterbi survivors to 4 Forward survivors, and `ATG2_CAD` reduced 542 to 57, with zero hit deltas in that run.
- Broader five-query stage counts (`benchmark-data/profmark-current/gpu-stage-suitability/counts-5/logs/profmark-gpu-summary.tsv`) put an upper bound on narrow Forward score offload. Across `14-3-3`, `AAA_15`, `ATG2_CAD`, `Tra1_ring`, and `Nup192`, 1,902 sequences passed Viterbi and 113 passed Forward. A theoretical free GPU Forward prefilter that still reruns CPU Forward for F3 survivors could save only about 1.75 sec from a 10.19 sec GPU run, so a real kernel must be well batched, parity-safe, and measured against the current MSV+bias baseline before acceptance.
- Current all-13 stage counts (`benchmark-data/profmark-current/gpu-stage-suitability/counts-13/logs/profmark-gpu-summary.tsv`) confirm the same interpretation on the prepared amino query set: 4,123 sequences passed Viterbi and 206 passed Forward. A theoretical free GPU Forward prefilter that still reruns CPU Forward for F3 survivors could save at most about 1.97 sec from a 14.66 sec GPU run, so the incremental upper bound is about 1.16x.
- Current combined all-13 timing (`benchmark-data/profmark-current/gpu-audit/vit-fwd-fb-parser-all13/logs/profmark-gpu-summary.tsv`) shows the later-stage path can beat the same-build MSV/bias baseline: 17.84 sec GPU wall versus 20.96 sec for `baseline-current-all13-rerun`, with the same final hit deltas. The remaining CPU-only hits match the baseline and are still SSV/MSV, not Viterbi/Forward/Backward.
- Do not use a naive direct diagonal CUDA SSV port over `sbv` as the parity fix. A reverted local experiment showed that this does not reproduce CPU `p7_SSVFilter()`'s banded traversal and worsened affected-family sensitivity. Future work should either map the CPU banded SSV algorithm exactly enough to match acceptance, or add a deliberate CPU-compatible SSV boundary check for the small set of borderline sequences.
- Backward is not a useful standalone GPU stage in the current design. It totaled only 0.250 sec in the all-13 baseline run and consumes Forward scaling/state before immediately feeding domain definition. The current parser experiment shows short-profile CUDA Backward itself can be made very fast; the remaining issue is the coupled Forward/Backward/domain-state handoff and CPU continuation, not an isolated Backward kernel.
- Domain definition is a poor near-term standalone GPU target. It runs after the Forward filter has already reduced candidates sharply and it is workflow-heavy: posterior decoding, region/envelope definition, stochastic clustering, null2/domain corrections, and alignment display ownership all interact with `P7_DOMAINDEF` and `P7_TOPHITS`.

Use ignored local `benchmark-data/` for larger datasets and run logs.

Current benchmark guidance:

- Use `profmark` for any serious GPU speed claim.
- Use `hmmseqdb` to build the target database before running `hmmsearch --gpu`.
- Keep `--gpu-load-seqs`, `--gpu-load-res`, `--gpu-batch-seqs`, `--gpu-batch-res`, `--gpu-msv-slack`, and CUDA batch count in the logs.
- Record pass counts for MSV, bias, Viterbi, and Forward, plus sensitivity deltas and wall-clock timing. Kernel speedup alone is not enough. Compare each proposed GPU stage against the last accepted GPU baseline, not just against CPU.
- Summarize `test-speed/x-hmmsearch-gpu-profmark` output with `test-speed/x-hmmsearch-gpu-profmark-summary <summary.tsv>` before making stage-suitability claims.

## Deferred Work

Do not include these in the first implementation milestone unless explicitly requested:

- GPU Viterbi, Forward/Backward, null2, or domain definition until each stage has a concrete suitability check and benchmark.
- GPU `hmmscan`, pressed HMM DB format changes, or profile database GPU indexing.
- GPU `phmmer` or `jackhmmer`.
- GPU daemon/cache integration.
- Nucleotide `nhmmer`/`nhmmscan` GPU acceleration.
- Rewriting CPU optimized implementations around CUDA abstractions.
