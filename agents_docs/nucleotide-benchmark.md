# Nucleotide Benchmark Setup

Last updated: 2026-05-07

## Purpose

Provide a reproducible nucleotide search benchmark for `nhmmer` performance work, including future GPU acceleration. The benchmark uses real genomic data (human chromosome 22, hg38) and DNA profile HMMs of varying model lengths.

## Target Database

- **File**: `benchmark-data/nucleotide-bench/work/chr22.fa`
- **Source**: UCSC hg38 `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz`
- **Size**: 50 MB FASTA, 50,818,468 residues (both strands searched = ~101.6M residues)
- **FM-index**: `benchmark-data/nucleotide-bench/work/chr22.fmdb` (64 MB, built with `makehmmerdb`)

chr22 is the smallest human autosome — large enough for meaningful timing but small enough that single-query runs complete in seconds.

## Query HMMs

| Query | M (nodes) | Source | chr22 time (1-thread) | Hits |
|-------|-----------|--------|----------------------|------|
| MADE1.hmm | 80 | Dfam alignment (tutorial/MADE1.sto) | ~1.0s | 154 |
| query_short.hmm | 151 | Single-seq from chr22:25532700-25532850 | ~1.5s | 120 |
| query_medium.hmm | 501 | Single-seq from chr22:34459500-34460000 | ~6.4s | 215 |
| query_long.hmm | 2001 | Single-seq from chr22:36340000-36342000 | ~25s+ | stress only |

MADE1 is the primary benchmark query — it's a real transposable element model with known biology and abundant chr22 hits.

## Running the Benchmark

### Basic CPU baseline (single-thread)

```sh
time src/nhmmer --cpu 1 --noali \
  benchmark-data/nucleotide-bench/work/MADE1.hmm \
  benchmark-data/nucleotide-bench/work/chr22.fa
```

### FM-index accelerated

```sh
time src/nhmmer --cpu 1 --noali \
  benchmark-data/nucleotide-bench/work/MADE1.hmm \
  benchmark-data/nucleotide-bench/work/chr22.fmdb
```

### Multi-query sweep

```sh
for hmm in MADE1 query_short query_medium; do
  echo "--- $hmm ---"
  time src/nhmmer --cpu 1 --noali \
    benchmark-data/nucleotide-bench/work/${hmm}.hmm \
    benchmark-data/nucleotide-bench/work/chr22.fa 2>&1 | grep -E "(Mc/sec|Total number|Elapsed)"
done
```

### Multi-thread comparison

```sh
for t in 1 2 4; do
  echo "--- threads=$t ---"
  time src/nhmmer --cpu $t --noali \
    benchmark-data/nucleotide-bench/work/MADE1.hmm \
    benchmark-data/nucleotide-bench/work/chr22.fa 2>&1 | grep "Elapsed"
done
```

## Baseline Results (2026-05-07)

Single-thread, chr22 FASTA target:

| Query | M | Wall time | Mc/sec | Hits |
|-------|---|-----------|--------|------|
| MADE1 | 80 | 1.05s | 7,761 | 154 |
| query_short | 151 | 1.56s | 9,862 | 120 |
| query_medium | 501 | 6.40s | 7,971 | 215 |

FM-index target (MADE1 only): 0.72s (1.46x faster than FASTA).

## Rebuilding the Benchmark Data

```sh
# Download chr22
wget -O benchmark-data/nucleotide-bench/work/chr22.fa.gz \
  "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
gunzip benchmark-data/nucleotide-bench/work/chr22.fa.gz

# Build FM-index
src/makehmmerdb benchmark-data/nucleotide-bench/work/chr22.fa \
  benchmark-data/nucleotide-bench/work/chr22.fmdb

# Build MADE1 HMM
src/hmmbuild --dna benchmark-data/nucleotide-bench/work/MADE1.hmm tutorial/MADE1.sto

# Extract query sequences and build HMMs
easel/miniapps/esl-sfetch --index benchmark-data/nucleotide-bench/work/chr22.fa
easel/miniapps/esl-sfetch -c 25532700..25532850 benchmark-data/nucleotide-bench/work/chr22.fa chr22 > benchmark-data/nucleotide-bench/work/query_short.fa
easel/miniapps/esl-sfetch -c 34459500..34460000 benchmark-data/nucleotide-bench/work/chr22.fa chr22 > benchmark-data/nucleotide-bench/work/query_medium.fa
easel/miniapps/esl-sfetch -c 36340000..36342000 benchmark-data/nucleotide-bench/work/chr22.fa chr22 > benchmark-data/nucleotide-bench/work/query_long.fa
for name in short medium long; do
  src/hmmbuild --dna -n "query_${name}" \
    benchmark-data/nucleotide-bench/work/query_${name}.hmm \
    benchmark-data/nucleotide-bench/work/query_${name}.fa
done
```

## nhmmer Pipeline Overview

`nhmmer` uses the same filter cascade as protein `hmmsearch` but adapted for long DNA targets:

1. **SSV/MSV filter**: scans windows of the target for high-scoring diagonal segments.
2. **Bias filter**: composition-bias correction.
3. **Viterbi filter**: full Viterbi on candidate windows.
4. **Forward filter**: Forward algorithm on Viterbi survivors.
5. **Domain definition**: envelope extraction and alignment on Forward survivors.

Key differences from protein search:
- Target is a single long sequence (chromosome), not many short sequences.
- Both strands are searched (forward + reverse complement).
- Window-based scanning with overlap, not per-sequence batching.
- FM-index can accelerate the initial SSV/MSV stage by skipping non-seed regions.
- E-values use a different statistical framework (window-based, not per-sequence).

## Relevance to GPU Work

The protein GPU path (`hmmsearch --gpu`) batches many short sequences per CUDA kernel launch. Nucleotide search is fundamentally different:
- One very long target sequence instead of many short ones.
- Window-based scanning requires different parallelization strategy.
- FM-index acceleration is orthogonal to GPU — it reduces the search space before filters run.

GPU acceleration for `nhmmer` would likely target:
- Parallel window scoring (many windows per kernel launch).
- Long-target Viterbi/Forward on candidate windows.
- Potentially FM-index seed extension on GPU.

This benchmark provides the baseline for measuring any future GPU nucleotide acceleration.
