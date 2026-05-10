# nhmmer GPU vs CPU Performance Breakdown

Last updated: 2026-05-10

## TL;DR

The previous “GPU is slower because CPU workers redo Forward/Backward and GPU domain rescoring is mutex-bound” analysis is stale. Current default `nhmmer --gpu` uses the GPU Forward prefilter and GPU Forward/Backward parser handoff by default, so CPU Forward/Backward continuation is gone in the accepted path. On FASTA and `.nucdb`, the Forward-to-Backward handoff now keeps Forward xmx device-resident, computes the F3 survivor gate and compact xmx offsets on GPU, compacts survivor xmx on GPU, and runs Backward from device-resident survivor metadata without re-uploading Forward xmx or survivor indices/offsets.

The real issue behind the stale 1.91s versus 2.109s comparison was apples-to-oranges benchmarking plus GPU window merge bugs. GPU window output order matters because `p7_pli_ExtendAndMergeWindows()` only merges adjacent windows. The current accepted path emits SSV windows in deterministic chunk order and emits scanning-Viterbi seeds into per-input-window slots that are compacted on GPU in parent-window order. It still extends/merges scanning-Viterbi seeds in parent MSV-window-local coordinates, matching CPU, before converting back to target coordinates.

## Historical query_medium Result

Target: chr22/hg38 benchmark. Query: `query_medium.hmm` (M=501). Threads: `--cpu 4`. GPU: RTX 4090. Date: 2026-05-10.

| Path | Target | Output rows | HMMER elapsed | `/usr/bin/time` wall |
|------|--------|:---:|:---:|:---:|
| CPU-4 | `chr22.fa` | 648 | 1.669s speed script | 1.77s focused |
| GPU default, no-overlap nucdb | `chr22.nucdb.nucdb` | 648 | 1.340s | - |
| GPU default, fast overlap nucdb | `chr22-overlap.nucdb.nucdb` | 648 | 1.232s speed script | 1.53-1.86s focused repeats |

The speed-script result now shows the expected fast-overlap `.nucdb` win after removing the extra single-score GPU Viterbi prefilter. The focused `--tblout` audit produced 215 CPU rows and 215 GPU rows with no diff.

## Current Combined Benchmark

`test-speed/x-nhmmer-gpu-bench` now defaults to a combined all-sample benchmark with `--cpu 16` for CPU and GPU paths. It runs `MADE1`, `query_short`, and `query_medium` in one `nhmmer` process per path, so CUDA engine creation and resident `.nucdb` upload are measured once instead of repeated for each query.

Current all-sample result on 2026-05-10:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 7.843s | 1476 |
| CPU-16 | 1.059s | 1476 |
| GPU-16 FASTA | 1.933s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.742s | 1476 |
| GPU-16 overlap `.nucdb` | 1.677s | 1476 |

Latest all-sample result after `.nucdb` GPU F3 gating and Forward-xmx compaction:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 7.940s | 1476 |
| CPU-16 | 0.941s | 1476 |
| GPU-16 FASTA | 1.975s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.472s | 1476 |
| GPU-16 overlap `.nucdb` | 1.270s | 1476 |

Latest all-sample result after caching reconstructed `.nucdb` host strands across queries:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.147s | 1476 |
| CPU-16 | 0.991s | 1476 |
| GPU-16 FASTA | 2.314s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.374s | 1476 |
| GPU-16 overlap `.nucdb` | 1.213s | 1476 |

Latest all-sample result after also making compact GPU F3/Backward handoff the FASTA default:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.808s | 1476 |
| CPU-16 | 1.079s | 1476 |
| GPU-16 FASTA | 2.214s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.666s | 1476 |
| GPU-16 overlap `.nucdb` | 1.450s | 1476 |

Latest all-sample result after adding a nucleotide-specific GPU F1 gate:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.823s | 1476 |
| CPU-16 | 1.090s | 1476 |
| GPU-16 FASTA | 2.380s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.568s | 1476 |
| GPU-16 overlap `.nucdb` | 1.202s | 1476 |

Latest all-sample result after ordered SSV output removed the first CPU qsort:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 7.971s | 1476 |
| CPU-16 | 0.999s | 1476 |
| GPU-16 FASTA | 1.902s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.445s | 1476 |
| GPU-16 overlap `.nucdb` | 1.165s | 1476 |

Latest all-sample result after also skipping the unused nucleotide F1 survivor-status D2H/store in production and compacting ordered SSV windows on GPU for one D2H transfer:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 9.362s | 1476 |
| CPU-16 | 1.038s | 1476 |
| GPU-16 FASTA | 1.756s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.207s | 1476 |
| GPU-16 overlap `.nucdb` | 1.236s | 1476 |

Parity after the update was clean: MADE1 FASTA 465=465, MADE1 `.nucdb` 465=465, query_short FASTA 363=363, query_medium FASTA 648=648.

Latest all-sample result after ordered GPU F1 survivor compaction removed the host survivor `qsort()`:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.189s | 1476 |
| CPU-16 | 1.035s | 1476 |
| GPU-16 FASTA | 1.948s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.207s | 1476 |
| GPU-16 overlap `.nucdb` | 1.240s | 1476 |

Parity after the ordered F1 update remained clean: MADE1 FASTA 465=465, MADE1 `.nucdb` 465=465, query_short FASTA 363=363, query_medium FASTA 648=648.

Latest all-sample result after device-resident F3 survivor metadata handoff:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 9.058s | 1476 |
| CPU-16 | 1.050s | 1476 |
| GPU-16 FASTA | 1.894s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.405s | 1476 |
| GPU-16 overlap `.nucdb` | 1.184s | 1476 |

Parity after the device-survivor update remained clean: MADE1 FASTA 465=465, MADE1 `.nucdb` 465=465, query_short FASTA 363=363, query_medium FASTA 648=648.

Latest all-sample result after removing avoidable parser-handoff `cudaDeviceSynchronize()` barriers:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 7.966s | 1476 |
| CPU-16 | 1.134s | 1476 |
| GPU-16 FASTA | 2.133s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.205s | 1476 |
| GPU-16 overlap `.nucdb` | 1.275s | 1476 |

Parity after the no-sync update remained clean: MADE1 FASTA 465=465, MADE1 `.nucdb` 465=465, query_short FASTA 363=363, query_medium FASTA 648=648.

Latest all-sample result after removing the redundant compact-xmx D2D copy before Backward:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.387s | 1476 |
| CPU-16 | 1.001s | 1476 |
| GPU-16 FASTA | 1.740s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.382s | 1476 |
| GPU-16 overlap `.nucdb` | 1.516s | 1476 |

Parity after the compact-xmx D2D removal remained clean: MADE1 FASTA 465=465, MADE1 `.nucdb` 465=465, query_short FASTA 363=363, query_medium FASTA 648=648.

Latest all-sample result after removing avoidable SSV/F1 handoff sync barriers:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.015s | 1476 |
| CPU-16 | 1.021s | 1476 |
| GPU-16 FASTA | 1.786s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.288s | 1476 |
| GPU-16 overlap `.nucdb` | 1.321s | 1476 |

Parity after the SSV/F1 no-sync update remained clean: MADE1 FASTA 465=465, MADE1 `.nucdb` 465=465, query_short FASTA 363=363, query_medium FASTA 648=648.

Latest all-sample result after Viterbi metadata scratch reuse:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.320s | 1476 |
| CPU-16 | 1.018s | 1476 |
| GPU-16 FASTA | 1.892s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.206s | 1476 |
| GPU-16 overlap `.nucdb` | 1.266s | 1476 |

Parity after the persistent F1 scratch update remained clean: MADE1 FASTA 465=465, MADE1 `.nucdb` 465=465, query_short FASTA 363=363, query_medium FASTA 648=648.

Latest all-sample result after SSV/Viterbi host-output scratch reuse:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.881s | 1476 |
| CPU-16 | 1.020s | 1476 |
| GPU-16 FASTA | 1.702s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.332s | 1476 |
| GPU-16 overlap `.nucdb` | 1.547s | 1476 |

Parity after SSV/Viterbi host-output scratch reuse remained clean: MADE1 FASTA 465=465, MADE1 `.nucdb` 465=465, query_short FASTA 363=363, query_medium FASTA 648=648.

Latest all-sample result after resident `.nucdb` chunk-metadata scratch reuse:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 9.435s | 1476 |
| CPU-16 | 0.896s | 1476 |
| GPU-16 FASTA | 1.763s | 1476 |
| GPU-16 no-overlap `.nucdb` | 1.352s | 1476 |
| GPU-16 overlap `.nucdb` | 1.234s | 1476 |

Parity after resident `.nucdb` chunk-metadata scratch reuse remained clean: MADE1 FASTA 465=465, MADE1 `.nucdb` 465=465, query_short FASTA 363=363, query_medium FASTA 648=648.

The cache reduced one concrete CPU island in multi-query `.nucdb` runs: a direct combined overlap `.nucdb` timing run reported `nucdb reconstruct` as `0.075s`, then `0.000s`, then `0.000s` for MADE1/query_short/query_medium. Process elapsed was `1.439s` with `1.079s` summed GPU loop wall. The end-to-end speed script remains noisy, so this should be read as a reduction in the reconstruction bucket, not a stable wall-time win over CPU-16.

The accepted nucleotide GPU F1 gate is separate from the older failed protein-gate experiment. The failed path did not preserve `B1`-scaled bias semantics and reordered survivors. The accepted path uses the nucleotide gate and now preserves original window order on GPU before CPU-side window-list compaction.

The combined overlap `.nucdb` timing run accounted for the previously confusing wall-time gap:

| Bucket | MADE1 | query_short | query_medium |
|--------|:---:|:---:|:---:|
| Search stages | 0.150s | 0.219s | 1.102s |
| GPU loop wall | 0.236s | 0.302s | 1.172s |
| Query elapsed | 0.237s | 0.303s | 1.174s |
| CPU workers | 0.062s | 0.123s | 0.756s |
| Domain workflow | 0.059s | 0.119s | 0.738s |
| nucdb reconstruct | 0.080s | 0.077s | 0.065s |
| Query outside search | 0.001s | 0.001s | 0.002s |

Shared setup/teardown for that process: CUDA engine create 0.446s, `.nucdb` open/mmap 0.000s, `.nucdb` upload 0.029s, CUDA destroy 0.003s, summed GPU loop wall 1.710s, process outside search 0.495s, process elapsed 2.205s.

The previous in-search mismatch was static GPU CPU-continuation scheduling. GPU workers were assigned contiguous equal-window slices, but domain-definition cost depends on region/envelope complexity. Dynamic survivor-window scheduling now lets workers pull work from a shared index and brings GPU-16 query_medium domain wall time (`0.248s` in the focused run) in line with the CPU-16 wall-stage trace (`0.250s`). The remaining gap against CPU-16 is not a worker imbalance; it is CUDA setup, first-use `.nucdb` reconstruction/cache fill, GPU SSV/Viterbi/Forward parser work, scanning-Viterbi sort/merge islands, and residual CPU domain workflow added together.

## Historical CPU-4 Stage Breakdown

CPU-4 HMMER stage totals are summed across worker threads, so divide by four for a rough per-thread wall comparison. GPU `NHMMER_GPU_INFO` buckets are wall buckets around sequential GPU stages plus max-across-worker CPU continuation buckets. The GPU column shows the focused repeat range after removing the extra single-score Viterbi prefilter.

| Stage | CPU-4 summed | CPU-4 approx wall | GPU fast `.nucdb` wall |
|-------|:---:|:---:|:---:|
| SSV | 2.343s | 0.586s | 0.103-0.109s |
| MSV/null/bias | 0.536s | 0.134s | 0.009-0.036s batch filter in the latest combined `.nucdb` timing + 0.002s worker bias/null |
| Viterbi | 1.307s | 0.327s | 0.138-0.289s scanning Viterbi |
| Forward | 0.469s | 0.117s | 0.028-0.039s Forward prefilter |
| Backward | 0.278s | 0.070s | 0.011-0.014s GPU FB parser; 0.000s CPU Backward |
| Domain workflow | 1.992s | 0.498s | 0.782-0.799s |
| Output/null2 | 0.000s | ~0.000s | 0.002-0.004s hit reporting |
| Total measured wall | - | 1.77s focused | 1.53-1.86s focused repeats |

GPU device-side filtering is faster than CPU-4 stage approximations, but CPU-16 is the relevant baseline now and it is much stronger. After dynamic scheduling, CPU domain workflow is no longer the unexplained mismatch against CPU-16; the combined GPU path is slower because it pays CUDA setup/reconstruction plus GPU SSV/Viterbi/Forward parser overhead before the remaining CPU domain workflow.

Current launch occupancy from the instrumented query_medium fast overlap `.nucdb` run:

| Stage | Last launch | Theoretical occupancy | Grid/SM coverage |
|-------|-------------|:---:|:---:|
| SSV longtarget | grid=800, block=32, smem=1002B | 50.0% (24 active warps/SM of 48), device-active in SSV wall | 6.25x |
| Scanning Viterbi | grid=1097, block=32, smem=24192B | 8.3% physical-warp occupancy (4 active warps/SM of 48), about 94% device-active in the current focused Viterbi CUDA call after first-call buffer growth | 8.51x |

The GPU kernels are not starved by a single CUDA engine setup or too few blocks on chr22/query_medium; both launch enough blocks to cover all 128 SMs multiple times. SSV occupancy is capped by one warp per block, but the kernel is already device-active and two-/four-warp-per-block experiments raised theoretical occupancy without improving wall time. Scanning Viterbi was the better target: the old kernel used only lanes 0-7 of each physical warp. The current kernel maps four independent 8-lane nucleotide DP groups into each physical warp. That lowers the physical-warp occupancy percentage but improves useful lane occupancy and reduced the repeated scan kernel from about `0.125-0.128s` to about `0.112-0.116s` in focused runs. Viterbi long-target also reuses engine-owned CUDA streams now, removing the per-call stream create/destroy path and making the current focused stream/sync timing round to `0.000s`; the default biased path now reads the F1-packed device batch directly, so detailed timing shows Viterbi host pack at `0 bytes` and Viterbi sequence H2D at `0.000s`. A score-only Forward prefilter experiment failed parity because it is not parser-equivalent for the F3 gate, so the production path still uses GPU Forward parser xmx before GPU Backward/parser handoff. The FASTA and `.nucdb` paths now remove the largest Forward-xmx host round trip by keeping all-window Forward xmx on device, computing F3 survivor selection and compact offsets on GPU from GPU Forward scores, compacting survivor xmx on GPU, and launching Backward from device-resident survivor metadata. The overlap `.nucdb` path now avoids host sequence packing for F1 and parser after GPU extend/merge by remapping windows to resident chunks and gathering boundary-spanning windows on GPU. Parser batches describe windows with metadata only, and F3 survivor-index plus compact Forward-xmx downloads are delayed until after Backward has consumed the device buffers. The `.nucdb` path also caches reconstructed host strands after first use. Ordered SSV output removes one CPU sort, ordered F1 survivor compaction plus bias-scratch reuse removes the host survivor `qsort()` and extra CPU bias-score handoff copy, persistent F1 survivor/metadata scratch removes per-batch survivor and metadata heap allocation, resident `.nucdb` SSV reuses chunk metadata scratch, direct window-list fill removes per-output `p7_hmmwindow_new()` calls, Viterbi reuses host metadata and output scratch, Viterbi seed slots remove the normal CPU raw-seed sort, SSV/Viterbi extend-merge helpers remove the default CPU merge islands, direct OMX binding removes parser-matrix host-to-host copies, and avoidable parser/SSV/F1 handoff synchronization barriers are gone. End-to-end GPU-16 overlap `.nucdb` remains volatile, so further optimization should focus on GPU kernel throughput, parser matrix D2H, and residual CPU domain workflow, not on creating more CUDA engines.

## Timing Breakdown From Current GPU Run

```
GPU pipeline: vit_lt_in=8753 vit_lt_out=711 post_vit=711 post_fwd=344 hits=481
GPU timing breakdown (0.584s search stages; 0.584s GPU loop wall):
  SSV longtarget:      0.125s
    utilization:       device-active in SSV wall
  extend+merge:        0.000s
  batch filter:        0.035s
  scanning Viterbi:    0.126s
    scan kernel:       0.116s
    alloc/grow:        0.005s
    stream/sync:       0.001s
    utilization:       92.2% device-active in Viterbi CUDA call
  Forward prefilter:   0.000s
  GPU FB parser:       0.024s
  CPU workers:         0.274s
    null scoring:      0.000s
    bias scoring:      0.003s
    CPU Backward:      0.000s
    domain workflow:   0.251s
    hit reporting:     0.000s
```

After the resident parser handoff, synchronization cleanup, ordered Viterbi seed compaction, and buffer ownership cleanup, query_medium in the overlap `.nucdb` focused run reported scanning Viterbi `0.120s`, host pack `0.000s (0 bytes)`, H2D `0.000s`, scan kernel `0.117s`, GPU FB parser `0.023s`, parser H2D `0.000s (0 bytes)`, parser kernels `0.016s`, parser D2H `0.002s`, CPU workers `0.266s`, and domain workflow `0.240s`. Combined benchmark numbers from the same update were CPU-1 `8.177s`, CPU-16 `1.052s`, GPU-16 FASTA `2.317s`, GPU-16 no-overlap `.nucdb` `1.423s`, and GPU-16 overlap `.nucdb` `1.257s`, all with `1476` hits. At that point the normal path no longer CPU-sorted raw Viterbi seeds, but CPU-16 remained faster because post-Viterbi extend/merge, parser matrix D2H, and CPU domain workflow remained. The later GPU extend/merge update removed that post-Viterbi CPU merge island.

After reusing parser-side host scratch for sequence indices, xmx offsets, and
F3 survivor source indices, the 2026-05-11 combined 16-thread benchmark remained
hit-clean: CPU-1 `8.210s`, CPU-16 `1.081s`, GPU-16 FASTA `1.933s`, GPU-16
no-overlap `.nucdb` `1.287s`, and GPU-16 overlap `.nucdb` `1.300s`, all with
`1476` hits. The focused query_medium fast `.nucdb` run showed parser host prep
rounding to `0.000s`; this confirms the scratch reuse removes allocation/control
churn but is not the main performance lever. At that point the remaining parser gap was still
sequence-byte H2D after CPU Viterbi seed extend/merge. Removing it required a
resident window gather path with explicit parent-window/local-coordinate metadata
preserved through `p7_pli_ExtendAndMergeWindows()`.

Parser transfer instrumentation on the next run quantified that gap: query_medium
fast `.nucdb` uploaded `585307` parser sequence bytes, with parser H2D event time
rounding to `0.000s`, parser kernels `0.017s`, parser D2H `0.005s`, and total GPU
FB parser `0.029s`. The same instrumentation update stayed hit-clean in the
combined benchmark: CPU-1 `8.457s`, CPU-16 `0.995s`, GPU-16 FASTA `2.268s`,
GPU-16 no-overlap `.nucdb` `1.305s`, and GPU-16 overlap `.nucdb` `1.296s`, all
with `1476` hits. A full reconstructed-strand resident upload would move far more
bytes than the current parser window pack for this workload unless it is reused
across queries, so it should not be treated as an obvious fix by itself.

Adding the same breakdown to the batch F1 stage showed the larger post-SSV
sequence movement: query_medium fast `.nucdb` batch filter `0.037s`, batch H2D
`0.001s` for `11049026` packed bytes, batch kernels `0.032s`, and batch D2H
`0.000s`; parser H2D in the same run remained `585307` bytes with event time
rounded to `0.000s`. Combined benchmark after this instrumentation was CPU-1
`9.304s`, CPU-16 `1.018s`, GPU-16 FASTA `2.276s`, GPU-16 no-overlap `.nucdb`
`1.724s`, and GPU-16 overlap `.nucdb` `1.163s`, all with `1476` hits. This
rules out host/device sequence copy time as the main reason GPU-16 trails
CPU-16 on this workload; the expensive pieces are GPU kernel work, required
parser matrix D2H for CPU domain workflow, and the CPU domain workflow itself.

The new F1 timing split tightened that conclusion further. In focused query_medium
overlap `.nucdb` runs, batch filter wall time is about `0.037-0.040s`, with
`f1 gate` about `0.033-0.034s` and ordered `f1 compact` about `0.001s`. The
batch-filter overhead is therefore dominated by the fused gate kernel itself,
not by survivor compaction or host/device copy time.

The next F1-to-Viterbi cleanup removed the F1-resident path's separate Viterbi
metadata materialization: scanning Viterbi now indexes the already resident F1
offsets/lengths through the survivor-index array. Focused query_medium overlap
`.nucdb` timing after that change reported scanning Viterbi `0.114s`, host pack
`0.000s (0 bytes)`, H2D `0.000s`, scan kernel `0.112s`, and stream/sync
rounded to `0.000s`. Repeated combined benchmarks remained noisy and did not
show a stable end-to-end win: GPU-16 overlap `.nucdb` was `1.529-1.575s` versus
CPU-16 `0.962-1.005s`, all with `1476` hits.

The next `.nucdb` residency updates removed both batch F1 and parser host
sequence packing for overlap `.nucdb`. Post-merge windows now map back to
resident chunks when possible; windows that cross a chunk boundary use a small
device gather into the active batch. Focused query_medium overlap `.nucdb`
timing after F1 gather reported batch F1 H2D `0.000s (0 bytes)` while parser
still uploaded `585307` bytes. The follow-up parser gather reduced parser H2D
to `0.000s (0 bytes)` as well: batch filter `0.038s`, scanning Viterbi `0.113s`
with host pack/H2D still `0 bytes / 0.000s`, GPU FB parser `0.026s`, parser
kernels `0.015s`, parser D2H `0.004s`, CPU workers `0.295s`, and domain
workflow `0.260s`. Parity remained clean (`4 passed, 0 failed`). The combined
16-thread benchmark from the parser-gather run was CPU-1 `8.642s`, CPU-16
`0.968s`, GPU-16 FASTA `1.820s`, GPU-16 no-overlap `.nucdb` `1.317s`, and
GPU-16 overlap `.nucdb` `1.233s`, all with `1476` hits. This removes the
remaining pre-parser sequence H2D volume on the fast `.nucdb` path, but the
end-to-end gap versus CPU-16 remains because the dominant focused buckets are
still SSV/Viterbi kernels, parser matrix D2H, and CPU domain workflow.

`hits=478` in the diagnostic line is the internal `P7_TOPHITS` count before final output formatting. The comparable main-output hit-line count was 648 for both CPU-16 and GPU-16; strict `--tblout` rows were 215 for both with no diff.

The next handoff cleanup moved the remaining default pre-parser
`p7_pli_ExtendAndMergeWindows()` work to CUDA-managed helpers. SSV longtarget
now compacts ordered raw hits and extends/merges them on GPU before the host
copies a ready `P7_HMM_WINDOW` list; scanning Viterbi now extends/merges/splits
ordered seed slots on GPU before the parser handoff. The normal path no longer
calls CPU SSV-window extend/merge or CPU Viterbi-seed extend/merge; `--gpu-compare`
still keeps the old CPU path for diagnostics. The important correctness detail
is that SSV extension must use `om->max_length`, not model length `M`; using `M`
made windows too short, changed downstream hit counts (`666` instead of `648`
for query_medium), and was fixed before accepting the update. After the fix,
`test-speed/x-nhmmer-gpu-parity .` with 16 threads was strict-clean (`4 passed,
0 failed`), and the focused query_medium overlap `.nucdb` run returned `648`
hit lines with `extend+merge` and `vit extend` both rounding to `0.000s`.
That run showed GPU loop wall `0.799s`, SSV `0.149s`, batch filter `0.174s`,
scanning Viterbi `0.131s`, GPU FB parser `0.025s`, CPU workers `0.274s`, and
domain workflow `0.245s`. The combined 16-thread benchmark was CPU-1 `8.335s`,
CPU-16 `1.061s`, GPU-16 FASTA `1.866s`, GPU-16 no-overlap `.nucdb` `1.345s`,
and GPU-16 overlap `.nucdb` `1.298s`, all with `1476` hits. This closes the
remaining CPU merge island before the parser; the remaining gap is GPU kernel
time, parser matrix D2H, and the allowed CPU domain workflow.

The follow-up post-parser cleanup removed the redundant CPU F3 null/bias
recomputation after GPU Forward/Backward parser handoff. Since the GPU Forward
parser already enforced the exact F3 gate, the default worker now calls the
no-F3 post-Fwd/Bwd path and leaves only CPU domain definition/output after the
parser handoff. Focused query_medium overlap `.nucdb` stayed hit-clean (`648`
hit lines) with CPU worker null, bias, and Backward buckets all rounding to
`0.000s`; CPU domain workflow was `0.243s`. `test-speed/x-nhmmer-gpu-parity .`
with 16 threads remained strict-clean (`4 passed, 0 failed`). The first combined
16-thread benchmark was CPU-1 `7.917s`, CPU-16 `0.936s`, GPU-16 FASTA `1.780s`,
GPU-16 no-overlap `.nucdb` `1.228s`, and GPU-16 overlap `.nucdb` `1.138s`, all
with `1476` hits. A fresh audit rerun stayed hit-clean but showed the run-to-run
spread: CPU-1 `8.115s`, CPU-16 `1.056s`, GPU-16 FASTA `2.413s`, GPU-16
no-overlap `.nucdb` `1.472s`, and GPU-16 overlap `.nucdb` `1.531s`, all with
`1476` hits. The remaining end-to-end gap is therefore no longer CPU filter work
in the GPU processing pipeline; it is GPU kernel time, parser matrix D2H,
CUDA/setup noise, and the allowed CPU domain/output workflow.

## Reproducing

```sh
src/nhmmer --cpu 16 --noali \
    benchmark-data/nucleotide-bench/work/query_medium.hmm \
    benchmark-data/nucleotide-bench/work/chr22.fa

src/nhmmer --gpu --cpu 16 --noali \
    benchmark-data/nucleotide-bench/work/query_medium.hmm \
    benchmark-data/nucleotide-bench/work/chr22-overlap.nucdb.nucdb
```

Run commands sequentially to avoid GPU contention. For benchmark claims, report the query, target database, flags, HMMER elapsed, external wall time, output-row count, and GPU timing breakdown.
