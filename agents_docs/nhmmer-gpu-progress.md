# nhmmer GPU Support — Progress

Last updated: 2026-05-11

## Architecture

GPU nhmmer uses a **GPU SSV + GPU window extend/merge + GPU filters + GPU scanning Viterbi + GPU Viterbi seed extend/merge + GPU Forward prefilter + GPU Forward/Backward parser handoff + threaded CPU domain hit reporting** pipeline by default. The GPU Forward prefilter is default-on with `--gpu` and reduces post-Viterbi survivors before CPU workers. GPU Forward/Backward parser matrix reuse is also default-on with `--gpu`; it hands GPU xmx matrices to CPU domain processing after enforcing the exact F3 gate. In the default biased path, scanning Viterbi consumes the F1 packed device batch through GPU survivor metadata, so the F1-to-Viterbi handoff no longer repacks or reuploads survivor sequence bytes. On FASTA and `.nucdb` inputs, the Forward-to-Backward handoff now keeps the all-window Forward xmx on device, computes the F3 survivor gate and survivor xmx offsets on GPU, compacts survivor xmx on GPU, and runs Backward from device-resident survivor indices/offsets without re-uploading Forward xmx or survivor metadata. The previous extra single-score GPU Viterbi prefilter has been removed; GPU scanning Viterbi is the CPU-equivalent Viterbi boundary.

```
Input FASTA/nucdb
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ GPU SSV Longtarget (warp-per-chunk, 64K chunks)     │
│   → overlapping chunks, register-based SSV DP       │
│   → emits deterministic chunk-ordered windows       │
│   → extends/merges windows on GPU using MAXL tables │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ GPU Batch Filter (MSV + null + bias + F1 gating)    │
│   → nucleotide B1-scaled F1 gate on GPU             │
│   → survivor-indexed bias-score D2H                 │
│   → CPU restores original window order              │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ GPU Scanning Viterbi (warp-per-window, 8 warps/blk) │
│   → reads F1-packed device windows in default path   │
│   → full DP per window, emits sub-windows           │
│   → GPU threshold: bias kernel + analytic null      │
│   → extends/merges/splits Viterbi seeds on GPU      │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ GPU Forward Pre-filter (exact F3 gate)              │
│   → GPU Forward parser on sub-windows               │
│   → GPU F3 gate removes windows                     │
│   → survivor indices/offsets remain device-resident │
│   → survivor xmx compacted on GPU                   │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ GPU Forward/Backward Parser Handoff                 │
│   → reuses GPU Forward xmx/fwdsc from prefilter     │
│   → runs GPU Backward from device survivor metadata │
│   → passes survivor matrices to CPU domain workflow │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ CPU: Envelope finding (domaindef heuristics)        │
│   → p7_domaindef_ByPosteriorHeuristics(envelopes)   │
│   → identifies domain envelopes (ienv, jenv)        │
│   → reparameterizes model per domain                │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ CPU domain definition / hit reporting               │
│   → p7_domaindef_ByPosteriorHeuristics              │
│   → null2/domain scoring and output remain CPU-side │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ CPU: Hit reporting                                  │
│   → build P7_ALIDISPLAY from GPU traces             │
│   → score computation, p7_tophits_CreateNextHit     │
│   → coordinate transforms (strand-aware)            │
└─────────────────────────────────────────────────────┘
```

Both strands processed sequentially (forward, then reverse complement).

## Files

| File | Purpose |
|------|---------|
| `src/nhmmer.c` | CLI (`--gpu` option), engine lifecycle, shared nucdb open/upload, timing accounting |
| `src/nhmmer_internal.h` | `NHMMER_GPU_INFO`, `NHMMER_GPU_WINDOW_BATCH` structs |
| `src/nhmmer_gpu.c` | GPU orchestration: batch filter, Viterbi, scanning Vit, shared nucdb reuse, domain rescore, hit reporting |
| `src/cuda/p7_cuda_ssv_longtarget.cu` | SSV longtarget kernel + `SSVLongtargetResident` |
| `src/cuda/p7_cuda_viterbi_longtarget.cu` | Scanning Viterbi kernel + threshold kernel |
| `src/cuda/p7_cuda_fb_parser.cu` | Forward/Backward parser batch: Forward-only, Backward-only, and combined F+B modes |
| `src/cuda/p7_cuda_domain_rescore.cu` | Domain rescoring: 6 GPU kernels (Fwd/Bck/Decoding/OA/OATrace/Domcorr) |
| `src/cuda/p7_cuda_runtime.cu` | Engine create/destroy/reset, nucdb upload/release |
| `src/cuda/p7_cuda_internal.h` | Engine struct (all device buffers including domain rescore grow-only buffers) |
| `src/cuda/p7_cuda.h` | Public API declarations |
| `src/p7_nucdb.c` | Nucdb format: Write/Open/Close (mmap-based) |
| `src/p7_nucdb.h` | Nucdb structs: `P7_NUCDB_HEADER`, `P7_NUCDB_CHUNK_IDX`, `P7_NUCDB_SEQ_IDX` |
| `src/hmmnucdb.c` | CLI tool: `hmmnucdb [opts] <seqfile> <nucdb>` |

## CLI

```sh
# Basic GPU nhmmer (default path: GPU filters/scanning Viterbi/Forward prefilter, CPU domain continuation)
src/nhmmer --gpu --cpu 16 --noali query.hmm target.fa

# With fast nucdb (default overlap enables GPU-resident SSV when overlap >= model max_length)
src/hmmnucdb target.fa target.nucdb        # build once; default --overlap 2001
src/nhmmer --gpu --cpu 16 --noali query.hmm target.nucdb.nucdb

# Ordinary no-overlap nucdb for diagnostics
src/hmmnucdb --overlap 0 target.fa target-o0.nucdb
src/nhmmer --gpu --cpu 16 --noali query.hmm target-o0.nucdb.nucdb

# Hidden flags (group 99, not shown in help)
--gpu-batch             # SSV/bias batch on GPU (default-on with --gpu)
--gpu-vit-longtarget    # Scanning Viterbi (default-on with --gpu)
--gpu-fwd-prefilter     # Deprecated no-op: GPU Fwd/Bwd parser reuse is default-on
--gpu-no-fwd-prefilter  # Diagnostic: use older CPU Fwd/Bwd continuation after GPU Forward prefilter
--gpu-cpu-postmsv       # Bypass GPU scanning Viterbi/Fwd; CPU postSSV path
--gpu-compare           # Debug: compare GPU vs CPU scores per stage
--gpu-chunk-size N      # Chunk size (default 65536)
--gpu-device N          # CUDA device selection
```

## Benchmark Results (2026-05-10)

**Target**: chr22.fa (50MB, ~101.6M residues both strands)
**System**: RTX 4090, 16 CPU threads by default

`test-speed/x-nhmmer-gpu-bench` now defaults to an all-sample multi-query run with `--cpu 16` for both CPU and GPU paths. It concatenates `MADE1`, `query_short`, and `query_medium` into `benchmark-data/nucleotide-bench/work/nhmmer-gpu-all-samples.hmm`, then runs one `nhmmer` process per path. This is the preferred benchmark for current GPU nhmmer because CUDA engine creation and resident `.nucdb` upload are shared across searches. The old one-query-at-a-time behavior remains available with `NHMMER_GPU_BENCH_MODE=per-query`; set `NHMMER_GPU_BENCH_CPU=N` to override the default thread count.

Current combined all-sample result:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 7.843s | 1476 |
| CPU-16 | 1.059s | 1476 |
| GPU-16 FASTA | 1.933s | 1476 |
| GPU-16 nucdb, no overlap | 1.742s | 1476 |
| GPU-16 overlap-nucdb | 1.677s | 1476 |

Latest combined all-sample result after `.nucdb` GPU F3 gating and Forward-xmx compaction:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 7.940s | 1476 |
| CPU-16 | 0.941s | 1476 |
| GPU-16 FASTA | 1.975s | 1476 |
| GPU-16 nucdb, no overlap | 1.472s | 1476 |
| GPU-16 overlap-nucdb | 1.270s | 1476 |

Latest combined all-sample result after caching reconstructed `.nucdb` host strands across queries:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.147s | 1476 |
| CPU-16 | 0.991s | 1476 |
| GPU-16 FASTA | 2.314s | 1476 |
| GPU-16 nucdb, no overlap | 1.374s | 1476 |
| GPU-16 overlap-nucdb | 1.213s | 1476 |

Latest combined all-sample result after also making compact GPU F3/Backward handoff the FASTA default:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.808s | 1476 |
| CPU-16 | 1.079s | 1476 |
| GPU-16 FASTA | 2.214s | 1476 |
| GPU-16 nucdb, no overlap | 1.666s | 1476 |
| GPU-16 overlap-nucdb | 1.450s | 1476 |

Latest combined all-sample result after adding a nucleotide-specific GPU F1 gate:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.823s | 1476 |
| CPU-16 | 1.090s | 1476 |
| GPU-16 FASTA | 2.380s | 1476 |
| GPU-16 nucdb, no overlap | 1.568s | 1476 |
| GPU-16 overlap-nucdb | 1.202s | 1476 |

Latest combined all-sample result after ordered GPU SSV output removed the first CPU qsort:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 7.971s | 1476 |
| CPU-16 | 0.999s | 1476 |
| GPU-16 FASTA | 1.902s | 1476 |
| GPU-16 nucdb, no overlap | 1.445s | 1476 |
| GPU-16 overlap-nucdb | 1.165s | 1476 |

Latest combined all-sample result after also skipping the unused nucleotide F1 survivor-status D2H/store in production and compacting ordered SSV windows on GPU for one D2H transfer:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 9.362s | 1476 |
| CPU-16 | 1.038s | 1476 |
| GPU-16 FASTA | 1.756s | 1476 |
| GPU-16 nucdb, no overlap | 1.207s | 1476 |
| GPU-16 overlap-nucdb | 1.236s | 1476 |

Latest combined all-sample result after ordered GPU F1 cleanup and direct OMX binding:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.189s | 1476 |
| CPU-16 | 1.035s | 1476 |
| GPU-16 FASTA | 1.948s | 1476 |
| GPU-16 nucdb, no overlap | 1.207s | 1476 |
| GPU-16 overlap-nucdb | 1.240s | 1476 |

Latest combined all-sample result after device-resident F3 survivor metadata handoff:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 9.058s | 1476 |
| CPU-16 | 1.050s | 1476 |
| GPU-16 FASTA | 1.894s | 1476 |
| GPU-16 nucdb, no overlap | 1.405s | 1476 |
| GPU-16 overlap-nucdb | 1.184s | 1476 |

Latest combined all-sample result after removing avoidable parser-handoff sync barriers:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 7.966s | 1476 |
| CPU-16 | 1.134s | 1476 |
| GPU-16 FASTA | 2.133s | 1476 |
| GPU-16 nucdb, no overlap | 1.205s | 1476 |
| GPU-16 overlap-nucdb | 1.275s | 1476 |

Latest combined all-sample result after removing the redundant compact-xmx D2D copy before Backward:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.387s | 1476 |
| CPU-16 | 1.001s | 1476 |
| GPU-16 FASTA | 1.740s | 1476 |
| GPU-16 nucdb, no overlap | 1.382s | 1476 |
| GPU-16 overlap-nucdb | 1.516s | 1476 |

Latest combined all-sample result after removing avoidable SSV/F1 handoff sync barriers:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.015s | 1476 |
| CPU-16 | 1.021s | 1476 |
| GPU-16 FASTA | 1.786s | 1476 |
| GPU-16 nucdb, no overlap | 1.288s | 1476 |
| GPU-16 overlap-nucdb | 1.321s | 1476 |

Latest combined all-sample result after Viterbi metadata scratch reuse:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.320s | 1476 |
| CPU-16 | 1.018s | 1476 |
| GPU-16 FASTA | 1.892s | 1476 |
| GPU-16 nucdb, no overlap | 1.206s | 1476 |
| GPU-16 overlap-nucdb | 1.266s | 1476 |

Latest combined all-sample result after SSV/Viterbi host-output scratch reuse:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 8.881s | 1476 |
| CPU-16 | 1.020s | 1476 |
| GPU-16 FASTA | 1.702s | 1476 |
| GPU-16 nucdb, no overlap | 1.332s | 1476 |
| GPU-16 overlap-nucdb | 1.547s | 1476 |

Parity after SSV/Viterbi host-output scratch reuse remained clean: MADE1 FASTA 465=465, MADE1 `.nucdb` 465=465, query_short FASTA 363=363, query_medium FASTA 648=648.

Latest combined all-sample result after resident `.nucdb` chunk-metadata scratch reuse:

| Path | Time | Hits |
|------|:---:|:---:|
| CPU-1 | 9.435s | 1476 |
| CPU-16 | 0.896s | 1476 |
| GPU-16 FASTA | 1.763s | 1476 |
| GPU-16 nucdb, no overlap | 1.352s | 1476 |
| GPU-16 overlap-nucdb | 1.234s | 1476 |

Parity after resident `.nucdb` chunk-metadata scratch reuse remained clean: MADE1 FASTA 465=465, MADE1 `.nucdb` 465=465, query_short FASTA 363=363, query_medium FASTA 648=648.

SSV longtarget now writes windows into fixed per-chunk slots, computes per-chunk output offsets on the GPU, compacts populated slots on GPU with global coordinates, and can return an already extended/merged `P7_HMM_WINDOW` list with one D2H transfer into engine-owned host scratch. This removes the CPU `qsort()` and the normal CPU `p7_pli_ExtendAndMergeWindows()` call before F1 while preserving hit parity. If a chunk exceeds the ordered-output capacity, the wrapper now fails explicitly instead of silently truncating windows. The F1 cleanup removes an unused survivor-status device store/D2H copy in production, builds an ordered pass mask on GPU before downloading compact survivor indices and bias scores, passes the ordered bias-score scratch directly to scanning Viterbi without an extra CPU copy, reuses persistent host survivor and metadata scratch instead of allocating per batch, reuses resident `.nucdb` chunk offset/length scratch instead of allocating it per strand, and now lets scanning Viterbi read the F1-packed device batch directly in the default biased path. That removes the old Viterbi survivor host-pack loop and sequence H2D copy for the normal path; the later ordered Viterbi seed-slot update also removes the normal CPU raw-seed sort, and the current default Viterbi helper returns already extended/merged/split parser windows. The wrapper also reuses host metadata/output scratch, fills downloaded GPU window lists directly when host lists are still needed, and binds downloaded GPU parser matrices directly into OMX objects for CPU domain workflow. The compact F3/Backward handoff now keeps F3 survivor indices and compact xmx offsets on the GPU between Forward gating and Backward, removing the previous survivor metadata D2H/H2D round trip. Backward now reads compact Forward xmx directly from the compact device buffer, so the previous compact-xmx D2D copy back into the main parser buffer is gone. Avoidable parser/SSV/F1 handoff synchronization barriers are removed; following CUDA events and D2H copies enforce the required ordering. The CPU still receives survivor windows and parser matrices because the allowed post-parser domain workflow needs host objects. The combined wall-time benchmark remains noisy; these changes should be read as communication/control reductions, not stable end-to-end wins by themselves.

The accepted GPU F1 gate is not the older protein helper. It preserves nucleotide `B1`-scaled bias semantics, returns survivor-indexed bias scores for scanning Viterbi, and now preserves original window order before D2H. CPU-side window-list compaction/merge is kept only in `--gpu-compare` diagnostics; full MSV/null/bias score arrays, unordered survivor sorting, unused status arrays, an extra CPU bias-score handoff copy, and parser-matrix host-to-host copies are no longer part of the accepted path.

Current parity script result:

| Query/path | Result |
|------------|:---:|
| MADE1 FASTA | 465 = 465 |
| MADE1 nucdb | 465 = 465 |
| query_short FASTA | 363 = 363 |
| query_medium FASTA | 648 = 648 |

Recent per-query baseline:

| Path | MADE1 (M=80) | query_short (M=151) | query_medium (M=501) |
|------|:---:|:---:|:---:|
| CPU-1 | 0.835s / 465 | 1.324s / 363 | 5.809s / 648 |
| CPU-4 | 0.278s / 465 | 0.388s / 363 | 1.713s / 648 |
| GPU-4 FASTA | 0.706s / 465 | 0.703s / 363 | 1.668s / 648 |
| GPU-4 nucdb | 0.724s / 465 | 0.817s / 363 | 1.493s / 648 |
| GPU-4 overlap-nucdb | 0.613s / 465 | 0.644s / 363 | 1.669s / 648 |

The older 2.109s overlap `.nucdb` result was stale: it predated the GPU window-ordering fixes and mixed old default runs with targeted Fwd/Bwd handoff runs.

### Combined-Run Timing Accounting

`nhmmer --gpu` now reports both search-stage timing and outside-search timing. The combined overlap `.nucdb` run showed one shared CUDA engine creation (`0.446s`) and one shared `.nucdb` upload (`0.029s`) for all three queries, followed by per-query reconstruction/search/report buckets. In that run, process elapsed time was `2.205s`, summed GPU loop wall time was `1.710s`, and process time outside the GPU search loops was `0.495s`.

After adding a shared `.nucdb` host-strand cache, reconstructed `ESL_SQ` objects are owned by `P7_NUCDB` and reused across queries in the same process. A direct combined overlap `.nucdb` timing run showed reconstruction only on the first query:

| Query | Search stages | GPU loop wall | Query elapsed | CPU workers | Domain workflow | nucdb reconstruct |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| MADE1 | 0.102s | 0.177s | 0.179s | 0.024s | 0.018s | 0.075s |
| query_short | 0.154s | 0.154s | 0.156s | 0.065s | 0.052s | 0.000s |
| query_medium | 0.748s | 0.748s | 0.752s | 0.302s | 0.263s | 0.000s |

Process-level line from that run:

| Bucket | Time |
|--------|:---:|
| CUDA engine create | 0.268s |
| shared nucdb open/mmap | 0.000s |
| shared nucdb upload | 0.034s |
| query pre-search setup | 0.003s |
| GPU loop wall total | 1.079s |
| query post-loop cleanup | 0.001s |
| post-search/report | 0.004s |
| CUDA reset | 0.000s |
| CUDA destroy | 0.003s |
| process outside search | 0.359s |
| process elapsed | 1.439s |

After SSV/Viterbi host-output scratch reuse, a direct combined overlap `.nucdb` timing repeat showed:

| Query | Search stages | GPU loop wall | Query elapsed | CPU workers | Domain workflow | nucdb reconstruct |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| MADE1 | 0.097s | 0.154s | 0.156s | 0.022s | 0.019s | 0.057s |
| query_short | 0.153s | 0.153s | 0.154s | 0.048s | 0.043s | 0.000s |
| query_medium | 0.708s | 0.708s | 0.720s | 0.277s | 0.252s | 0.000s |

Process-level line from that run:

| Bucket | Time |
|--------|:---:|
| CUDA engine create | 0.307s |
| shared nucdb open/mmap | 0.000s |
| shared nucdb upload | 0.020s |
| query pre-search setup | 0.011s |
| GPU loop wall total | 1.014s |
| query post-loop cleanup | 0.000s |
| post-search/report | 0.003s |
| CUDA reset | 0.000s |
| CUDA destroy | 0.004s |
| process outside search | 0.379s |
| process elapsed | 1.394s |

The overlap `.nucdb` path is now resident through the F1-to-scanning-Viterbi boundary. SSV reads resident chunks on GPU; after CPU `p7_pli_ExtendAndMergeWindows()`, merged F1 windows are mapped back to resident chunk offsets when they fit within one chunk, and chunk-spanning windows use a small CUDA gather into the active F1 batch. Scanning Viterbi then reads that same active F1 batch through survivor metadata, so the overlap `.nucdb` path no longer packs or uploads F1 sequence bytes from the host. The later parser gather update applies the same resident-window mapping to Forward/Backward parser input, so parser sequence bytes are also gathered on GPU; parser matrices are still downloaded for the CPU domain workflow.

After resident `.nucdb` chunk-metadata scratch reuse, a direct combined overlap `.nucdb` timing repeat showed:

| Query | Search stages | GPU loop wall | Query elapsed | CPU workers | Domain workflow | nucdb reconstruct |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| MADE1 | 0.207s | 0.257s | 0.258s | 0.024s | 0.019s | 0.050s |
| query_short | 0.127s | 0.127s | 0.128s | 0.047s | 0.042s | 0.000s |
| query_medium | 0.584s | 0.584s | 0.587s | 0.274s | 0.251s | 0.000s |

Process-level line from that run:

| Bucket | Time |
|--------|:---:|
| CUDA engine create | 0.311s |
| shared nucdb open/mmap | 0.000s |
| shared nucdb upload | 0.016s |
| query pre-search setup | 0.002s |
| GPU loop wall total | 0.968s |
| query post-loop cleanup | 0.001s |
| post-search/report | 0.003s |
| CUDA reset | 0.000s |
| CUDA destroy | 0.003s |
| process outside search | 0.374s |
| process elapsed | 1.342s |

After the F1-to-Viterbi resident handoff, the combined benchmark stayed hit-clean and the detailed overlap `.nucdb` run showed the targeted communication bucket removed from scanning Viterbi:

| Benchmark path | Wall | Hits |
|----------------|:---:|:---:|
| CPU-1 | 8.745s | 1476 |
| CPU-16 | 1.042s | 1476 |
| GPU-16 FASTA | 1.819s | 1476 |
| GPU-16 nucdb, no overlap | 1.374s | 1476 |
| GPU-16 overlap-nucdb | 1.409s | 1476 |

| Query | Search stages | GPU loop wall | Query elapsed | Viterbi host pack | Viterbi H2D | CPU workers | Domain workflow |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MADE1 | 0.218s | 0.273s | 0.275s | 0 bytes | 0.000s | 0.023s | 0.019s |
| query_short | 0.129s | 0.129s | 0.130s | 0 bytes | 0.000s | 0.048s | 0.042s |
| query_medium | 0.569s | 0.569s | 0.572s | 0 bytes | 0.000s | 0.289s | 0.266s |

The next `.nucdb` residency update added a post-merge F1 window mapper and
device gather path. Chunk-contained windows pass resident `.nucdb` offsets and
lengths directly to the F1 gate. Windows spanning the boundary between two
overlapped chunks upload only small span metadata and gather sequence bytes on
GPU into the active F1 batch before running the same nucleotide F1 gate; this
keeps Viterbi-from-F1 unchanged and avoids host sequence packing for overlap
`.nucdb` F1. Verification on 2026-05-11:

| Check | Result |
|-------|:---:|
| `make -C src -j2 nhmmer hmmnucdb` | pass |
| `test-speed/x-nhmmer-gpu-parity .` | 4 passed, 0 failed |

Focused query_medium fast overlap `.nucdb` timing after resident F1 gather:

| Bucket | Time / volume |
|--------|:---:|
| Process elapsed | 0.954s |
| GPU loop wall | 0.622s |
| Search stages | 0.572s |
| SSV longtarget | 0.111s |
| Batch filter | 0.035s |
| Batch F1 H2D | 0.000s, 0 bytes |
| Batch kernels | 0.033s |
| Scanning Viterbi | 0.118s |
| Viterbi host pack / H2D | 0 bytes / 0.000s |
| GPU FB parser | 0.047s |
| Parser H2D | 0.000s, 585307 bytes |
| CPU workers | 0.260s |
| Domain workflow | 0.234s |

Combined 16-thread benchmark from the same update:

| Benchmark path | Wall | Hits |
|----------------|:---:|:---:|
| CPU-1 | 8.128s | 1476 |
| CPU-16 | 1.064s | 1476 |
| GPU-16 FASTA | 2.416s | 1476 |
| GPU-16 nucdb, no overlap | 1.312s | 1476 |
| GPU-16 overlap-nucdb | 1.395s | 1476 |

The follow-up resident parser-batch gather reuses the same window-to-`.nucdb`
mapping for the GPU Forward/F3/Backward parser batch. The parser still downloads
Forward/Backward xmx matrices for the CPU domain workflow, but it no longer
packs or uploads parser sequence bytes on the overlap `.nucdb` path.
Verification after this parser gather stayed clean on 2026-05-11:

| Check | Result |
|-------|:---:|
| `make -C src -j2 nhmmer hmmnucdb` | pass |
| `test-speed/x-nhmmer-gpu-parity .` | 4 passed, 0 failed |

Focused query_medium fast overlap `.nucdb` timing after resident parser gather:

| Bucket | Time / volume |
|--------|:---:|
| Process elapsed | 1.130s |
| GPU loop wall | 0.645s |
| Search stages | 0.584s |
| SSV longtarget | 0.111s |
| Batch filter | 0.038s |
| Batch F1 H2D | 0.000s, 0 bytes |
| Scanning Viterbi | 0.113s |
| Viterbi host pack / H2D | 0 bytes / 0.000s |
| GPU FB parser | 0.026s |
| Parser host prep | 0.001s |
| Parser H2D | 0.000s, 0 bytes |
| Parser kernels | 0.015s |
| Parser D2H | 0.004s |
| CPU workers | 0.295s |
| Domain workflow | 0.260s |

Combined 16-thread benchmark from the same update:

| Benchmark path | Wall | Hits |
|----------------|:---:|:---:|
| CPU-1 | 8.642s | 1476 |
| CPU-16 | 0.968s | 1476 |
| GPU-16 FASTA | 1.820s | 1476 |
| GPU-16 nucdb, no overlap | 1.317s | 1476 |
| GPU-16 overlap-nucdb | 1.233s | 1476 |

The next verified parser-side cleanup reuses host scratch for parser sequence
indices, parser xmx offsets, and parser F3 survivor source indices. It preserves
the same host-visible post-Viterbi windows and the same GPU Forward/Backward
parser kernels, so it is a control/allocation reduction rather than a new
resident parser coordinate path. Verification on 2026-05-11:

| Check | Result |
|-------|:---:|
| `git diff --check` | pass |
| `make -C src -j2 nhmmer hmmnucdb` | pass |
| `test-speed/x-nhmmer-gpu-parity .` | 4 passed, 0 failed |

Combined 16-thread benchmark from the same run:

| Benchmark path | Wall | Hits |
|----------------|:---:|:---:|
| CPU-1 | 8.210s | 1476 |
| CPU-16 | 1.081s | 1476 |
| GPU-16 FASTA | 1.933s | 1476 |
| GPU-16 nucdb, no overlap | 1.287s | 1476 |
| GPU-16 overlap-nucdb | 1.300s | 1476 |

Focused query_medium fast overlap `.nucdb` timing after the parser scratch reuse
reported search stages `0.679s`, GPU loop wall `0.726s`, SSV `0.213s`, batch
filter `0.037s`, scanning Viterbi `0.121s` with `0` Viterbi host-pack bytes/H2D,
GPU FB parser `0.046s`, parser host prep rounded to `0.000s`, CPU workers
`0.261s`, and domain workflow `0.234s`. At that point, before the later resident
parser gather, the parser scratch change was parity-safe but did not remove the
remaining parser sequence-byte H2D.

Parser transfer instrumentation added before the resident parser gather showed
why a full-strand resident parser upload was not automatically the next win. A
focused query_medium fast `.nucdb` run reported GPU FB parser `0.029s`, parser
host prep `0.000s`, parser sequence H2D event time rounded to `0.000s` for
`585307` packed bytes, parser kernels `0.017s`, and parser D2H `0.005s`. The
combined benchmark from the same instrumentation update remained hit-clean:

| Benchmark path | Wall | Hits |
|----------------|:---:|:---:|
| CPU-1 | 8.457s | 1476 |
| CPU-16 | 0.995s | 1476 |
| GPU-16 FASTA | 2.268s | 1476 |
| GPU-16 nucdb, no overlap | 1.305s | 1476 |
| GPU-16 overlap-nucdb | 1.296s | 1476 |

That historical parser sequence upload was small for query_medium relative to a
full chromosome-strand upload, which is why the accepted follow-up was a true
window gather/reuse design rather than a blind upload of the whole reconstructed
strand per query.

That resident parser gather has now landed, so the preceding paragraph is
historical rather than current guidance. The current overlap `.nucdb` parser path
does not pack host subsequences; it describes the post-Viterbi windows with
metadata only, gathers parser sequence bytes on GPU, and keeps F3 survivor
indices plus compact Forward xmx on device until Backward has consumed them. The
CPU still receives survivor windows and parser matrices afterward for the
allowed domain workflow. Verification after delaying the F3 survivor-index D2H
and compact Forward-xmx D2H until after Backward:

| Check | Result |
|-------|:---:|
| `make -C src -j2 nhmmer hmmnucdb` | pass |
| `test-speed/x-nhmmer-gpu-parity .` | 4 passed, 0 failed |

Focused query_medium fast overlap `.nucdb` timing:

| Bucket | Time / volume |
|--------|:---:|
| Process elapsed | 1.052s |
| GPU loop wall | 0.599s |
| Search stages | 0.540s |
| SSV longtarget | 0.106s |
| Batch filter | 0.034s |
| Batch F1 H2D | 0.000s, 0 bytes |
| Scanning Viterbi | 0.111s |
| Viterbi host pack / H2D | 0 bytes / 0.000s |
| GPU FB parser | 0.024s |
| Parser host prep | 0.001s |
| Parser H2D | 0.000s, 0 bytes |
| Parser kernels | 0.015s |
| Parser D2H | 0.003s |
| CPU workers | 0.264s |
| Domain workflow | 0.242s |

Combined 16-thread benchmark from the same update stayed hit-clean but did not
show a stable wall-time win:

| Benchmark path | Wall | Hits |
|----------------|:---:|:---:|
| CPU-1 | 9.136s | 1476 |
| CPU-16 | 1.102s | 1476 |
| GPU-16 FASTA | 2.470s | 1476 |
| GPU-16 nucdb, no overlap | 1.369s | 1476 |
| GPU-16 overlap-nucdb | 1.556s | 1476 |

Read this as a communication-order cleanup: Forward-to-Backward now stays
device-side until Backward completes, but total wall time remains dominated by
SSV/Viterbi kernel time, parser matrix D2H for CPU domain workflow, and the CPU
domain workflow itself.

A follow-up synchronization cleanup removed two remaining host barriers in the
GPU section: Viterbi long-target now waits on the kernel event rather than a
device-wide sync before the D2H count copy, and Forward/Backward parser uses
separate timing events for Forward and Backward so the Backward launch follows
Forward in stream order without a host event synchronization between the kernels.
After a follow-up buffer ownership cleanup for the Viterbi per-window seed-count
storage, verification stayed clean:

| Check | Result |
|-------|:---:|
| `make -C src -j2 nhmmer hmmnucdb` | pass |
| `test-speed/x-nhmmer-gpu-parity .` | 4 passed, 0 failed |

Focused query_medium fast overlap `.nucdb` timing after the ownership cleanup:

| Bucket | Time / volume |
|--------|:---:|
| Process elapsed | 0.903s |
| GPU loop wall | 0.602s |
| Search stages | 0.556s |
| SSV longtarget | 0.111s |
| Batch filter | 0.035s |
| Batch F1 H2D | 0.000s, 0 bytes |
| Scanning Viterbi | 0.116s |
| Viterbi host pack / H2D | 0 bytes / 0.000s |
| GPU FB parser | 0.024s |
| Parser host prep | 0.001s |
| Parser H2D | 0.000s, 0 bytes |
| Parser kernels | 0.016s |
| Parser D2H | 0.003s |
| CPU workers | 0.270s |
| Domain workflow | 0.250s |

Combined 16-thread benchmark from the same update:

| Benchmark path | Wall | Hits |
|----------------|:---:|:---:|
| CPU-1 | 8.576s | 1476 |
| CPU-16 | 1.079s | 1476 |
| GPU-16 FASTA | 2.055s | 1476 |
| GPU-16 nucdb, no overlap | 1.368s | 1476 |
| GPU-16 overlap-nucdb | 1.207s | 1476 |

This improved the GPU handoff shape rather than changing the algorithmic split:
at that point the CPU still sorted/extended/merged GPU Viterbi seeds before
parser, and CPU domain workflow still consumed the downloaded parser matrices.

The next Viterbi boundary update mirrors the ordered SSV output design for
scanning-Viterbi seeds. The kernel now writes seeds into fixed per-input-window
slots, a GPU prefix pass computes per-window output offsets, and a GPU compact
pass downloads seeds in parent-window order. The normal path therefore no longer
CPU-sorts raw Viterbi seeds before filling the local-coordinate window list;
`--gpu-compare` keeps a diagnostic sort for comparison snapshots. If any input
window emits more than `VIT_LT_MAX_WINDOWS_PER_INPUT` seeds, the CUDA helper
fails explicitly instead of silently truncating.

Verification stayed clean:

| Check | Result |
|-------|:---:|
| `make -C src -j2 nhmmer hmmnucdb` | pass |
| `test-speed/x-nhmmer-gpu-parity .` | 4 passed, 0 failed |

Focused query_medium fast overlap `.nucdb` timing:

| Bucket | Time / volume |
|--------|:---:|
| Process elapsed | 1.011s |
| GPU loop wall | 0.599s |
| Search stages | 0.555s |
| SSV longtarget | 0.111s |
| Batch filter | 0.035s |
| Batch F1 H2D | 0.000s, 0 bytes |
| Scanning Viterbi | 0.120s |
| Viterbi host pack / H2D | 0 bytes / 0.000s |
| Viterbi seed sort | 0.000s |
| Viterbi extend | 0.000s |
| GPU FB parser | 0.023s |
| Parser host prep | 0.001s |
| Parser H2D | 0.000s, 0 bytes |
| Parser kernels | 0.016s |
| Parser D2H | 0.002s |
| CPU workers | 0.266s |
| Domain workflow | 0.240s |

Combined 16-thread benchmark from the same update:

| Benchmark path | Wall | Hits |
|----------------|:---:|:---:|
| CPU-1 | 8.177s | 1476 |
| CPU-16 | 1.052s | 1476 |
| GPU-16 FASTA | 2.317s | 1476 |
| GPU-16 nucdb, no overlap | 1.423s | 1476 |
| GPU-16 overlap-nucdb | 1.257s | 1476 |

This removes the normal CPU Viterbi seed sort, but not the CPU
`p7_pli_ExtendAndMergeWindows()` step that turns local seeds into merged
post-Viterbi windows. Moving that step is the remaining pre-parser CPU island and
requires preserving HMMER's exact local-coordinate merge/split semantics.

Batch F1 transfer instrumentation from the next focused query_medium fast
`.nucdb` trace showed the larger post-SSV sequence movement: batch filter
`0.037s`, batch H2D `0.001s` for `11049026` packed bytes, batch kernels `0.032s`,
and batch D2H `0.000s`. In the same run, parser H2D remained small at `585307`
bytes with event time rounded to `0.000s`, parser kernels were `0.015s`, and
parser D2H `0.002s`. This confirms that the current gap is not primarily PCIe
copy time; the useful next targets are reducing F1/Viterbi/parser kernel work,
matrix D2H volume needed by CPU domain workflow, and the CPU domain workflow
itself.

Combined 16-thread benchmark after this instrumentation stayed hit-clean:

| Benchmark path | Wall | Hits |
|----------------|:---:|:---:|
| CPU-1 | 9.304s | 1476 |
| CPU-16 | 1.018s | 1476 |
| GPU-16 FASTA | 2.276s | 1476 |
| GPU-16 nucdb, no overlap | 1.724s | 1476 |
| GPU-16 overlap-nucdb | 1.163s | 1476 |

Process-level line from that run:

| Bucket | Time |
|--------|:---:|
| CUDA engine create | 0.273s |
| shared nucdb open/mmap | 0.000s |
| shared nucdb upload | 0.032s |
| query pre-search setup | 0.002s |
| GPU loop wall total | 0.971s |
| query post-loop cleanup | 0.000s |
| post-search/report | 0.003s |
| CUDA reset | 0.000s |
| CUDA destroy | 0.003s |
| process outside search | 0.351s |
| process elapsed | 1.322s |

Earlier timing before the `.nucdb` strand cache:

| Query | Search stages | GPU loop wall | Query elapsed | CPU workers | Domain workflow | nucdb reconstruct | Query outside search |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MADE1 | 0.150s | 0.236s | 0.237s | 0.062s | 0.059s | 0.080s | 0.001s |
| query_short | 0.219s | 0.302s | 0.303s | 0.123s | 0.119s | 0.077s | 0.001s |
| query_medium | 1.102s | 1.172s | 1.174s | 0.756s | 0.738s | 0.065s | 0.002s |

Process-level line from that earlier run:

| Bucket | Time |
|--------|:---:|
| CUDA engine create | 0.446s |
| shared nucdb open/mmap | 0.000s |
| shared nucdb upload | 0.029s |
| query pre-search setup | 0.002s |
| GPU loop wall total | 1.710s |
| query post-loop cleanup | 0.000s |
| post-search/report | 0.003s |
| CUDA reset | 0.000s |
| CUDA destroy | 0.003s |
| process outside search | 0.495s |
| process elapsed | 2.205s |

Current strict query_medium parity audit on 2026-05-10:

| Path | Target | Main-output hit lines | `--tblout` rows |
|------|--------|:---:|:---:|
| CPU-4 | `chr22.fa` | 648 | 215 |
| GPU default, fast overlap nucdb | `chr22-overlap.nucdb.nucdb` | 648 | 215 |

The CPU and GPU `--tblout` rows had no diff. `hmmnucdb` defaults to the fast overlap construction (`--overlap 2001`) and `nhmmer --gpu` defaults to GPU Fwd/Bwd parser handoff. The code-path verification is nonzero `GPU FB parser` time with zero CPU Forward/Backward stage time.

Current focused fast `.nucdb` GPU breakdown for query_medium after compact-xmx D2D removal remains dominated by allowed post-parser CPU domain workflow plus GPU SSV/Viterbi work:

| Bucket | Time |
|--------|:---:|
| SSV longtarget | 0.111s |
| extend+merge | 0.001s |
| batch filter | 0.037s |
| scanning Viterbi | 0.125s |
| Forward prefilter | 0.000s |
| GPU FB parser | 0.031s |
| CPU workers | 0.331s |
| worker domain workflow | 0.297s |
| worker CPU Backward | 0.000s |

Current query_medium launch/occupancy instrumentation on fast overlap `.nucdb`:

| Stage | Launch shape | Occupancy | Grid coverage |
|-------|--------------|:---:|:---:|
| SSV longtarget | 2 launches, last grid=800, block=32, smem=1002B | 50.0% theoretical, 24 active warps/SM of 48; 97.9% device-active in SSV wall | 6.25x on 128 SMs |
| Scanning Viterbi | 2 launches, last grid=1103, block=32, smem=24192B | 8.3% theoretical, 4 active physical warps/SM of 48; about 94.4% device-active in the current focused Viterbi CUDA call after first-call buffer growth | 8.55x on 128 SMs |

The 16-thread baseline changes the conclusion: GPU-16 remains hit-clean but is slower than CPU-16 on the combined all-sample benchmark. Focused query_medium shows why the occupancy counter alone was misleading. SSV is a one-warp-per-block kernel with only 50% theoretical occupancy, but it is already device-active; grouping multiple chunks per block raised theoretical occupancy and did not improve wall time. Scanning Viterbi was wasting lanes 8-31 of each physical warp for nucleotide DP; it now packs four 8-lane DP groups per physical warp, reducing the repeated scan kernel from about `0.125-0.128s` to about `0.112-0.116s` in focused runs, though current runs still show variance. Viterbi long-target now reuses two engine-owned nonblocking CUDA streams instead of creating and destroying streams on every call, and the default biased path reads F1-packed device windows directly instead of repacking/reuploading survivor sequence bytes. A score-only Forward prefilter experiment failed parity badly because `p7_cuda_ForwardScoreDsqdataSubset()` is not parser-equivalent for this nucleotide F3 gate, so the accepted path still uses the GPU Forward parser xmx handoff. The FASTA and `.nucdb` paths now avoid the all-window Forward xmx D2H transfer, CPU F3 survivor loop, survivor Forward xmx H2D transfer, and survivor metadata H2D before Backward by gating, computing compact offsets, and compacting Forward xmx on GPU before Backward. The batch F1 stage now uses a nucleotide-specific GPU gate, only copies ordered survivor indices/bias scores back, reuses persistent host survivor/metadata scratch, fills downloaded GPU window lists directly, and exposes the packed device batch to Viterbi; the Viterbi wrapper also reuses host metadata/output scratch. The GPU parser handoff binds downloaded xmx buffers directly into OMX objects for CPU domain workflow. Ordered SSV output removes the first CPU SSV-window sort, ordered Viterbi seed slots remove the normal CPU raw-seed sort, and avoidable parser/SSV/F1 handoff synchronization barriers are gone. SSV-window and scanning-Viterbi seed extend/merge now run in CUDA-managed helpers in the default path; residual CPU domain workflow remains. The final combined benchmark remains run-to-run volatile (`1.163-1.556s` observed for GPU-16 overlap `.nucdb` across recent runs), so measured micro-improvements do not yet translate to a stable end-to-end win over CPU-16.

Focused repeats show run-to-run variance in CUDA engine setup and scanning Viterbi. After dynamic survivor-window scheduling, GPU CPU-domain workflow is no longer inflated relative to CPU-16 for query_medium: CPU-16 wall-stage trace measured domain at `0.250s`; GPU-16 focused repeats measured worker domain at `0.239-0.258s` and GPU loop wall at `0.617-0.814s` depending mostly on CUDA Viterbi allocation variance.

### CPU Wall-Stage Trace

`HMMER_NHMMER_CPU_WALL_TRACE=1` prints max-across-worker stage times before CPU worker pipelines are merged. This is intentionally separate from the normal `# Stage ... time` lines, which remain summed across workers. Current CPU-16 query_medium wall-stage trace:

| Stage | Wall estimate |
|-------|:---:|
| SSV | 0.239s |
| MSV host | 0.015s |
| bias | 0.044s |
| Viterbi | 0.141s |
| Forward | 0.044s |
| Backward | 0.032s |
| domain | 0.250s |

### Dynamic CPU-Continuation Scheduling

The GPU continuation workers now pull post-Forward survivor windows from a shared work index instead of receiving contiguous equal-window slices. This better matches the CPU-only work-queue behavior because domain-definition cost is driven by region/envelope complexity, not just survivor-window count. The change also computes overlap accounting from global neighboring survivor windows, preserving parity while avoiding worker-boundary artifacts.

### Historical GPU Domain Rescoring Performance

Cross-window batching + trim batching replaces per-domain GPU calls with two batched calls per strand:

| Query | Domains/strand | GPU kernel time | Trim domains | Trim kernel time |
|-------|:---:|:---:|:---:|:---:|
| MADE1 | ~3000 | 27ms | ~1100 | 12ms |
| query_short | ~600 | varies | ~300 | varies |
| query_medium | ~300 | varies | ~100 | varies |

Previous per-window approach: MADE1 took 34s (5000+ individual GPU calls at ~5ms each). Cross-window batching: **1.74s** (18x improvement). Forward-Backward split (prefilter saves xf, Backward-only): **1.35s** (further 1.3x). Scanning Viterbi threshold fix: **0.91s** (further 1.5x). Nucdb + overlap-nucdb paths: **0.64s** (GPU-resident SSV, zero per-chunk H2D).

### Default Matrix Reuse

The default path always runs the GPU Forward prefilter with the exact F3 gate, then reuses GPU Forward/Backward parser xmx for surviving windows and uses CPU domain processing. This path matched default GPU hit counts on the current chr22 smoke set for MADE1, query_short, and query_medium on both FASTA and nucdb.

GPU Backward parser matrix handoff was tested after fixing the CPU diagnostic length configuration to use `window_len` rather than `min(window_len, om->max_length)`. `--gpu-compare` shows Forward/Backward parser scores match CPU to float noise. query_short showed a 1 bp envelope-start shift in one coordinate comparison, and a similar 1 bp envelope-start jitter can occur between repeated default `--gpu` runs, so this is a boundary-sensitivity issue rather than a hit-count regression.

Current Forward-prefilter evidence: on query_medium fast overlap `.nucdb`, the default Fwd/Bwd handoff kept 341/707 post-Viterbi windows (48.2%) in the focused timing run and the speed script measured 1.232s versus 1.669s for CPU-4. Worker CPU Backward was 0.000s, confirming the parser handoff rather than the older CPU Forward/Backward continuation.

Implementation:
- `NHMMER_GPU_WORKER` carries `prefilter_xf`, `prefilter_fwdsc`, `prefilter_xf_offset` (non-owning pointers into shared buffers)
- `nhmmer_gpu_worker_process_post_fwd()` injects xf into `pli->oxf->xmx`, reconstructs `totscale`, calls `p7_pli_postFwd_LongTarget()`
- `p7_pli_postFwd_LongTarget()` in `p7_pipeline.c` performs F3 check using provided fwdsc, then Backward + domaindef + hit reporting
- Gated on `use_skip_fwd = info->do_gpu_fwd && prefilter_xf != NULL`; `info->do_gpu_fwd` is default-on with `--gpu`

### Parity Notes

- **MADE1 (M=80)**: FASTA 462 vs 465 (within 1% tolerance); nucdb 462 vs 462 exact in the parity script.
- **query_short (M=151)**: 363 vs 363 exact.
- **query_medium (M=501)**: 648 vs 648 exact.

Remaining MADE1 FASTA delta is at the GPU long-target window/filter boundary; default `--gpu` now reuses GPU Forward/Backward parser matrices, then keeps domain processing and hit reporting on CPU.

### Window Ordering Fixes (2026-05-10)

The root cause of inflated GPU worker time was not CPU Forward/Backward. GPU kernels emitted windows via `atomicAdd`, so output order was not guaranteed. `p7_pli_ExtendAndMergeWindows()` only merges adjacent windows, so unsorted GPU windows defeated merge and inflated downstream domain-definition work. The SSV long-target output now uses per-chunk slots and ordered host download, and scanning-Viterbi now uses per-input-window seed slots plus GPU compaction, so the normal path no longer needs CPU qsorts before either merge. The current default path also performs SSV-window and scanning-Viterbi seed extend/merge on GPU; `--gpu-compare` keeps the CPU merge path only for diagnostics.

Fixes:
- Emit GPU SSV windows in deterministic chunk order before the first `p7_pli_ExtendAndMergeWindows()` call.
- Emit GPU Viterbi seeds in deterministic parent-window order before extension/merge.
- Extend and merge GPU scanning-Viterbi seeds in parent-window-local coordinates, then convert merged windows back to target coordinates.
- Clamp overlap accounting for `pos_past_vit` and `pos_past_fwd` so overlapping windows cannot underflow residue counters.
- Use exact F3 gating in the GPU Forward prefilter.

### Removed Single-Score Viterbi Prefilter (2026-05-10)

The previous single-score GPU Viterbi prefilter (`--gpu-vit-prefilter`) was removed from nucleotide `nhmmer`. It was an extra GPU-only gate before the real long-target scanning Viterbi stage and had no direct CPU-reference equivalent. The accepted path is `.nucdb` overlap input with GPU SSV, GPU batch MSV/null/bias filtering, GPU scanning Viterbi, GPU Forward prefilter, and GPU Fwd/Bwd parser handoff.

### Nucdb Reverse-Strand Coordinate Fix (2026-05-10)

For `.nucdb` reverse-complement chunks, reconstructed `ESL_SQ` metadata now mirrors `esl_sq_ReverseComplement()` by setting `start = n` and `end = 1`. This prevents negative reverse-strand coordinate transforms in GPU `.nucdb` output and allowed strict CPU FASTA versus GPU overlap `.nucdb` tabular rows to match for query_medium in the smoke audit.

### Bias Filter Precision Fix (2026-05-10)

Fixed two precision bugs in the GPU scanning Viterbi threshold path:

1. **`cuda_bias_filter_kernel`**: Was using a fixed `t[0][0]` transition for all windows (uploaded once at init). Now computes per-window `t00 = L/(L+1)` matching CPU's `p7_bg_SetLength` per-window behavior. Also switched from `logf()` to `(float)log()` matching CPU's double-precision log in `esl_hmm_Forward`.

2. **`cuda_compute_viterbi_thresholds_kernel`**: The bias score is computed against the full parent `window_len`, so the kernel must subtract `nullsc_win` to isolate composition bias before scaling it to the local window length. The compare-side diagnostic uses the same calculation.

3. **Compare-side Viterbi diagnostic**: The CPU reference path in `--gpu-compare` now recomputes `nullsc_win` for each shortened candidate window before subtracting it from the bias filter score. This matches `p7_pli_postSSV_LongTarget()` and removes a diagnostic-only threshold mismatch.

## Historical GPU Domain Rescoring Architecture

### Three-Phase Design (in `nhmmer_gpu_worker_process_post_vit_gpu`)

**Phase 1: Envelope collection (CPU)**
- Iterates over all Viterbi survivor windows
- Runs F3 gate, `p7_domaindef_ByPosteriorHeuristics(envelopes_only=TRUE)`
- Collects domain envelopes: `NHMMER_GPU_PDOM` struct (ienv, jenv, Ld, rfv_copy)
- Reparameterizes model per domain (composition-adjusted emissions)
- Stores per-window context in `NHMMER_GPU_WCTX`

**Phase 2: GPU batch call**
- Single `p7_cuda_DomainRescoreBatch` call for all ~3000-6000 domains
- Bulk H2D: staging buffers → single memcpy for DSQ and RFV
- 6 sequential kernels (Fwd, Bck, Decoding, OA, OATrace, Domcorr)
- Single-thread-per-block design (one CUDA block per domain)
- Grow-only engine buffers (never shrink between calls)

**Phase 3: Unpack + trim batch + hit reporting (3 sub-phases)**
- **3a**: Build traces from GPU output, create P7_ALIDISPLAY, identify trim candidates
- **3b**: Single GPU batch call for all ~1000-2000 trim domains
- **3c**: Apply trim results, report hits (coordinate transform, scoring, p7_tophits)

### CUDA Kernels (`p7_cuda_domain_rescore.cu`)

| Kernel | Purpose | Design |
|--------|---------|--------|
| `cuda_domain_fwd_full_kernel` | Full Forward with matrix storage | T threads/block, parallel prefix scan for D-state, tree reduction for xE |
| `cuda_domain_bck_full_kernel` | Full Backward | T threads/block, reverse prefix scan, reads forward xmx for scaling |
| `cuda_domain_decoding_kernel` | Posterior Decoding (fwd*bck/total) | T threads/block, strided element-wise, no prefix scan needed |
| `cuda_domain_optacc_kernel` | Optimal Accuracy DP | 1 thread/block (deferred: needs max-prefix scan) |
| `cuda_domain_oatrace_kernel` | OA Traceback (reverse) | 1 thread/block (inherently sequential traceback) |
| `cuda_domain_fwd_scoreonly_kernel` | Domcorrection Forward (orig rfv) | T threads/block, parallel prefix scan, score-only (no matrix storage) |

### Grow-Only Engine Buffers

Device allocations in `P7_CUDA_ENGINE` that only grow (never freed between calls):
- `d_dom_dpf[4]`: DP matrices (fwd, bck, pp, oa) — keyed on total_dp_cells
- `d_dom_xmx[4]`: Special state matrices — keyed on total_xmx
- `d_dom_rfv_all`, `d_dom_dsq_all`: Packed per-domain data
- `d_dom_*_ptrs`, `d_dom_lengths`, etc.: Per-domain metadata — keyed on ndomains
- `d_dom_trace_*`: Trace output arrays — keyed on ndomains × max_trace_len

## Nucdb Format

Pre-chunked, mmap'd nucleotide database. Eliminates FASTA parsing and enables GPU-resident data path.

```
File layout (page-aligned):
  Metadata:  sequence names (null-terminated strings)
  Seq index: P7_NUCDB_SEQ_IDX[nseq] (name_offset, length, chunk ranges)
  Chunk idx: P7_NUCDB_CHUNK_IDX[nchunks] (data_offset, length, seq_id, seq_offset, strand)
  Data:      [sentinel][residues] per chunk, contiguous, both strands
```

Build-time options: `--chunk-size` (default 65536), `--overlap` (default 2001), `--fwd-only`.

**Resident path**: When `--overlap >= model_max_length` AND `chunk_size` matches runtime, the kernel reads directly from GPU-uploaded nucdb data (no H2D per sequence). Pre-stored RC chunks eliminate `esl_sq_ReverseComplement`.

## Status Summary

Default `nhmmer --gpu` now uses GPU SSV/batch filtering/Viterbi/Forward prefilter plus GPU Forward/Backward parser matrix handoff before CPU domain/hit reporting. `--gpu-fwd-prefilter` is deprecated compatibility spelling; hidden `--gpu-no-fwd-prefilter` restores the older CPU Fwd/Bwd continuation for diagnostics. Historical implementation notes below include GPU domain-rescoring work that is not part of the accepted default path.

Recent timing instrumentation split the nucleotide F1 stage into two GPU pieces: the fused gate kernel is the real cost, while ordered survivor compaction is tiny. In focused query_medium overlap `.nucdb` runs, batch filter wall time is about `0.037-0.040s`, with `f1 gate` about `0.033-0.034s` and `f1 compact` about `0.001s`. That means the remaining F1 optimization target is kernel throughput and memory efficiency, not the ordered compaction step.

The F1-resident scanning Viterbi path now indexes the existing F1 device offsets/lengths through the survivor-index array instead of launching a separate metadata materialization kernel and allocating/filling `d_vlt_offsets`/`d_vlt_lengths` for that path. Focused query_medium overlap `.nucdb` timing after this cleanup showed scanning Viterbi `0.114s`, host pack `0.000s (0 bytes)`, H2D `0.000s`, scan kernel `0.112s`, and stream/sync rounded to `0.000s`. The same update stayed parity-clean (`4/4`) and repeated combined benchmarks were still volatile: CPU-16 `0.962-1.005s`, GPU-16 FASTA `1.750-1.816s`, GPU-16 no-overlap `.nucdb` `1.523-1.535s`, and GPU-16 overlap `.nucdb` `1.529-1.575s`, all with `1476` hits.

The overlap `.nucdb` batch F1 path now avoids host sequence packing too. After
SSV-window extend/merge, F1 windows are mapped back to resident `.nucdb`
chunks; windows contained in one chunk read resident offsets directly, and
boundary-spanning windows use a device gather into the active F1 batch. Focused
query_medium fast `.nucdb` timing after this change showed batch F1 H2D
`0.000s (0 bytes)`, batch filter `0.035s`, scanning Viterbi host pack/H2D
still `0 bytes / 0.000s`, and parser H2D still `585307` bytes. Verification
remained clean (`test-speed/x-nhmmer-gpu-parity .`: `4 passed, 0 failed`).
The combined 16-thread benchmark from that run was CPU-1 `8.128s`, CPU-16
`1.064s`, GPU-16 FASTA `2.416s`, GPU-16 no-overlap `.nucdb` `1.312s`, and
GPU-16 overlap `.nucdb` `1.395s`, all with `1476` hits.

The default path now also moves the remaining pre-parser window extend/merge
bookkeeping onto the GPU. SSV longtarget can return already extended and
adjacent-merged `P7_HMM_WINDOW` records, using the same prefix/suffix MAXL tables
as `p7_pli_ExtendAndMergeWindows()`. Scanning Viterbi can return already
extended, merged, max-window-split, target-coordinate parser windows. A parity
bug in the first SSV port came from using model length instead of
`om->max_length`; after passing the correct MAXL value, the focused
query_medium overlap `.nucdb` run returned `648` main-output hit lines, matching
CPU-16, with `extend+merge` and `vit extend` both rounding to `0.000s`. The
same focused run reported process elapsed `1.082s`, GPU loop wall `0.799s`,
SSV `0.149s`, batch filter `0.174s` (no sequence H2D), scanning Viterbi
`0.131s` with host pack/H2D `0 bytes / 0.000s`, GPU FB parser `0.025s`,
parser H2D `0.000s (0 bytes)`, parser kernels `0.016s`, parser D2H `0.003s`,
CPU workers `0.274s`, and domain workflow `0.245s`. `test-speed/x-nhmmer-gpu-parity`
with 16 threads was strict-clean for MADE1, query_short, and query_medium
(`4 passed, 0 failed`). The combined 16-thread benchmark from the same update
was CPU-1 `8.335s`, CPU-16 `1.061s`, GPU-16 FASTA `1.866s`, GPU-16 no-overlap
`.nucdb` `1.345s`, and GPU-16 overlap `.nucdb` `1.298s`, all with `1476` hits.

The next post-parser cleanup removed the redundant CPU F3 null/bias recomputation
after the GPU Forward/Backward parser handoff. GPU F3 already selected survivors,
so the worker now calls `p7_pli_postFwdBwdNoF3_LongTarget()` and goes straight
from GPU matrices into CPU domain definition/output. Focused query_medium overlap
`.nucdb` after this change stayed at `648` hit lines and reported CPU worker
null, bias, and Backward buckets all rounding to `0.000s`; CPU domain workflow
was `0.243s`. `test-speed/x-nhmmer-gpu-parity .` with 16 threads remained
strict-clean (`4 passed, 0 failed`). The first combined 16-thread benchmark was
CPU-1 `7.917s`, CPU-16 `0.936s`, GPU-16 FASTA `1.780s`, GPU-16 no-overlap
`.nucdb` `1.228s`, and GPU-16 overlap `.nucdb` `1.138s`, all with `1476` hits.
A fresh audit rerun on the same implementation stayed hit-clean but slower:
CPU-1 `8.115s`, CPU-16 `1.056s`, GPU-16 FASTA `2.413s`, GPU-16 no-overlap
`.nucdb` `1.472s`, and GPU-16 overlap `.nucdb` `1.531s`, all with `1476` hits.

```
a7936ac8 gpu: fix scanning Viterbi threshold bug + add --gpu-compare and --gpu-cpu-postmsv
063207a2 build: ignore src/hmmnucdb and suppress easel submodule dirty state
a5a49fc0 gpu: multi-thread domain rescore kernels + split Forward/Backward parser API
26a6a5e7 gpu: rebase compatibility fix and doc refresh for h3-gpu latest
d1a7a4a6 gpu: fix nhmmer domain rescoring parity by using unihit nj=0
93c51c5b gpu: eliminate redundant Forward in nhmmer prefilter→FB pipeline
1e0116e0 gpu: GPU domain rescoring with cross-window batching for nhmmer
775fa2a5 gpu: GPU ForwardBackward parser for nhmmer domaindef
8fe7522d gpu: nhmmer performance optimizations (per-window thresholds, Forward pre-filter, kernel tuning)
8e6eae3d docs: comprehensive nhmmer GPU documentation and benchmark scripts
97cb6e06 gpu: pre-stored RC from nucdb + GPU-resident SSV longtarget path
3cd1287a cleanup: remove --gpu-compare debug flag and update nhmmer GPU docs
c274036f gpu: nucleotide GPU database format (nucdb)
5d8c80d7 gpu: port h3-gpu optimizations to nhmmer
fd4c34e3 gpu: GPU scanning Viterbi for nhmmer with GPU threshold computation
4787499c gpu: batch MSV/bias filter and Viterbi pre-filter for nhmmer
09e6c38f gpu: threaded downstream pipeline for nhmmer with CUDA engine reuse
(+ h3-gpu base: fused SSV kernel, multi-warp-per-block, templated Viterbi, Forward prefix M≤2044)
```

## Future Work

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| High | Multi-threaded warp-cooperative domain kernels | High | Would parallelize within each domain, reducing kernel time 4-8x |
| Low | Async strand overlap | Low | ~0.1-0.3s savings |
