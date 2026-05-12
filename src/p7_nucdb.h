/* p7_nucdb.h
 * GPU-native nucleotide database format (v2): 2-bit packed residues + mask.
 *
 * v2 layout:
 *   - Forward-strand residues only; the reverse complement is generated on
 *     the fly by the GPU kernels and by the CPU slice-fill helper (index
 *     inversion + 2-bit XOR with 3).
 *   - Each canonical base is packed as a 2-bit code (A=0, C=1, G=2, T=3).
 *   - Each non-ACGT residue (IUPAC degenerates R/Y/M/K/S/W/H/B/V/D, plus N,
 *     gap, *, ~) is collapsed to packed code 0 (= A on device). A separate
 *     1-bit mask bitmap records which positions were non-ACGT, so CPU
 *     domain workers can materialise true N (dsq code 15) when they
 *     reconstruct survivor-window dsq bytes.
 *   - The packed region and the mask region live in two separate
 *     page-aligned ranges in the file. Only the packed region is uploaded
 *     to GPU memory; the mask stays host-only.
 */
#ifndef p7NUCDB_INCLUDED
#define p7NUCDB_INCLUDED

#include <p7_config.h>
#include <stdint.h>
#include <stdio.h>

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_sq.h"
#include "esl_sqio.h"

#define P7_NUCDB_MAGIC     0x4E554344  /* "NUCD" */
#define P7_NUCDB_VERSION   2           /* v2 = 2-bit packed + mask, forward only */

#define P7_NUCDB_ENCODING_PACKED2BIT  1

typedef struct {
  uint32_t  magic;
  uint32_t  version;
  uint64_t  nseq;              /* number of original sequences */
  uint64_t  nres;              /* total residues (forward strand only) */
  uint64_t  max_seqlen;        /* longest original sequence */
  uint64_t  nchunks;           /* total chunks (forward only in v2) */
  uint64_t  chunk_size;        /* nominal chunk size used at build time */
  uint64_t  overlap;           /* overlap between adjacent chunks */
  uint64_t  data_offset;       /* page-aligned byte offset to PACKED region */
  uint64_t  mask_offset;       /* page-aligned byte offset to MASK region */
  uint64_t  chunk_idx_offset;  /* byte offset to chunk index */
  uint64_t  seq_idx_offset;    /* byte offset to sequence metadata */
  uint8_t   encoding;          /* P7_NUCDB_ENCODING_PACKED2BIT */
  uint8_t   reserved[7];
} P7_NUCDB_HEADER;              /* 96 bytes */

/* Per-chunk index entry (stored contiguously in file).
 * In v2, data_offset/mask_offset are in bytes relative to the respective
 * region bases. length is residue count (not byte count).
 */
typedef struct {
  int64_t  data_offset;        /* byte offset into packed region */
  int64_t  mask_offset;        /* byte offset into mask region */
  int32_t  length;             /* residues in this chunk */
  int32_t  seq_id;             /* parent sequence index */
  int64_t  seq_offset;         /* position in original forward sequence (0-based) */
  int8_t   pad[8];
} P7_NUCDB_CHUNK_IDX;

/* Per-sequence metadata (v2: forward chunks only). */
typedef struct {
  int64_t  name_offset;        /* byte offset into name blob */
  int64_t  length;             /* original sequence length */
  int32_t  chunk_start;        /* first chunk index */
  int32_t  chunk_count;        /* number of chunks */
} P7_NUCDB_SEQ_IDX;

typedef struct {
  char                *filename;
  int                  fd;
  void                *mmap_base;
  size_t               mmap_size;

  P7_NUCDB_HEADER      hdr;

  /* Pointers into mmap'd region */
  uint8_t             *chunk_data;     /* base of PACKED region (2-bit) */
  uint8_t             *mask_data;      /* base of MASK region (1-bit) */
  P7_NUCDB_CHUNK_IDX  *chunk_idx;      /* chunk index array */
  P7_NUCDB_SEQ_IDX    *seq_idx;        /* sequence metadata array */
  char                *name_blob;      /* sequence names (null-terminated strings) */
} P7_NUCDB;

extern int  p7_nucdb_Write(const char *basename, const ESL_ALPHABET *abc,
                           ESL_SQFILE *sqfp, int64_t chunk_size, int64_t overlap,
                           char *errbuf);
extern int  p7_nucdb_Open(const char *basename, P7_NUCDB **ret_ndb, char *errbuf);
extern void p7_nucdb_Close(P7_NUCDB *ndb);

/* Decode one residue from the packed buffer. Returns the 2-bit code
 * (A=0,C=1,G=2,T=3). Caller should inspect the mask separately if needed.
 */
static inline int
p7_nucdb_unpack2bit(const uint8_t *packed, int64_t pos)
{
  int shift = (int)((pos & 3) << 1);
  return (packed[pos >> 2] >> shift) & 0x3;
}

/* Test one mask bit. 1 => non-ACGT (N), 0 => canonical ACGT. */
static inline int
p7_nucdb_mask_bit(const uint8_t *mask, int64_t pos)
{
  return (mask[pos >> 3] >> (int)(pos & 7)) & 0x1;
}

#endif /* p7NUCDB_INCLUDED */
