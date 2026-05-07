/* p7_gpudb.h
 * GPU-native database format: memory-mappable, pre-unpacked sequences.
 */
#ifndef p7GPUDB_INCLUDED
#define p7GPUDB_INCLUDED

#include <p7_config.h>
#include <stdint.h>
#include <stdio.h>

#include "easel.h"
#include "esl_alphabet.h"

#define P7_GPUDB_MAGIC   0x47505544   /* "GPUD" */
#define P7_GPUDB_VERSION 2

typedef struct {
  uint32_t  magic;
  uint32_t  version;
  uint64_t  nseq;
  uint64_t  nres;
  uint64_t  max_seqlen;
  uint64_t  data_offset;      /* byte offset to sequence data region (page-aligned) */
  uint64_t  idx_offset;       /* byte offset to index region */
  uint64_t  meta_idx_offset;  /* byte offset to metadata index (int64 per seq) */
  uint64_t  meta_blob_offset; /* byte offset to metadata blob */
  uint64_t  meta_blob_size;   /* size of metadata blob in bytes */
  uint8_t   reserved[8];
} P7_GPUDB_HEADER;

typedef struct {
  char            *filename;
  int              fd;
  void            *mmap_base;
  size_t           mmap_size;

  P7_GPUDB_HEADER  hdr;

  /* Pointers into mmap'd region */
  uint8_t         *seq_data;     /* contiguous sequence residues (1 byte/residue) */
  int64_t         *offsets;      /* byte offset from seq_data to seq i's first residue */
  int32_t         *lengths;      /* residue count for each sequence */

  /* Metadata pointers (v2+) */
  int64_t         *meta_offsets; /* byte offset into meta_blob for seq i */
  char            *meta_blob;    /* packed metadata: name\0 acc\0 desc\0 taxid(4B) per seq */
  int              has_metadata; /* TRUE if metadata section is present */
} P7_GPUDB;

extern int  p7_gpudb_Write(const char *basename, const ESL_ALPHABET *abc,
                           ESL_SQ **sqarr, int64_t nseq, char *errbuf);
extern int  p7_gpudb_Open(const char *basename, P7_GPUDB **ret_gdb, char *errbuf);
extern void p7_gpudb_Close(P7_GPUDB *gdb);

extern int  p7_gpudb_GetMetadata(const P7_GPUDB *gdb, int64_t idx,
                                 const char **ret_name, const char **ret_acc,
                                 const char **ret_desc, int *ret_taxid);

#endif /* p7GPUDB_INCLUDED */
