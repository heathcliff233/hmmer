/* p7_nucdb.h
 * GPU-native nucleotide database format: pre-chunked, both strands, memory-mappable.
 */
#ifndef p7NUCDB_INCLUDED
#define p7NUCDB_INCLUDED

#include <p7_config.h>
#include <stdint.h>
#include <stdio.h>

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_sqio.h"

#define P7_NUCDB_MAGIC   0x4E554344   /* "NUCD" */
#define P7_NUCDB_VERSION 1

typedef struct {
  uint32_t  magic;
  uint32_t  version;
  uint64_t  nseq;              /* number of original sequences */
  uint64_t  nres;              /* total residues (forward strand only) */
  uint64_t  max_seqlen;        /* longest original sequence */
  uint64_t  nchunks;           /* total chunks (both strands) */
  uint64_t  chunk_size;        /* nominal chunk size used at build time */
  uint64_t  overlap;           /* overlap between adjacent chunks */
  uint64_t  data_offset;       /* byte offset to chunk data region (page-aligned) */
  uint64_t  chunk_idx_offset;  /* byte offset to chunk index */
  uint64_t  seq_idx_offset;    /* byte offset to sequence metadata */
  uint8_t   both_strands;      /* 1 = RC chunks stored, 0 = forward only */
  uint8_t   reserved[15];
} P7_NUCDB_HEADER;

/* Per-chunk index entry (stored contiguously in file) */
typedef struct {
  int64_t  data_offset;        /* byte offset from data region start to this chunk's sentinel */
  int32_t  length;             /* residues in this chunk (not counting sentinel) */
  int32_t  seq_id;             /* parent sequence index */
  int64_t  seq_offset;         /* position in original sequence (0-based) */
  int8_t   complementarity;    /* 0 = forward, 1 = reverse complement */
  int8_t   pad[3];
} P7_NUCDB_CHUNK_IDX;

/* Per-sequence metadata */
typedef struct {
  int64_t  name_offset;        /* byte offset into metadata blob */
  int64_t  length;             /* original sequence length */
  int32_t  fwd_chunk_start;    /* first chunk index for forward strand */
  int32_t  fwd_chunk_count;    /* number of forward chunks */
  int32_t  rc_chunk_start;     /* first chunk index for RC strand */
  int32_t  rc_chunk_count;     /* number of RC chunks */
} P7_NUCDB_SEQ_IDX;

typedef struct {
  char                *filename;
  int                  fd;
  void                *mmap_base;
  size_t               mmap_size;

  P7_NUCDB_HEADER      hdr;

  /* Pointers into mmap'd region */
  uint8_t             *chunk_data;     /* contiguous chunk residues */
  P7_NUCDB_CHUNK_IDX  *chunk_idx;      /* chunk index array */
  P7_NUCDB_SEQ_IDX    *seq_idx;        /* sequence metadata array */
  char                *name_blob;      /* sequence names (null-terminated strings) */
} P7_NUCDB;

extern int  p7_nucdb_Write(const char *basename, const ESL_ALPHABET *abc,
                           ESL_SQFILE *sqfp, int64_t chunk_size, int64_t overlap,
                           int both_strands, char *errbuf);
extern int  p7_nucdb_Open(const char *basename, P7_NUCDB **ret_ndb, char *errbuf);
extern void p7_nucdb_Close(P7_NUCDB *ndb);

#endif /* p7NUCDB_INCLUDED */
