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
#define P7_GPUDB_VERSION 1

typedef struct {
  uint32_t  magic;
  uint32_t  version;
  uint64_t  nseq;
  uint64_t  nres;
  uint64_t  max_seqlen;
  uint64_t  data_offset;    /* byte offset to sequence data region (page-aligned) */
  uint64_t  idx_offset;     /* byte offset to index region */
  uint8_t   reserved[16];
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
} P7_GPUDB;

extern int  p7_gpudb_Write(const char *basename, const ESL_ALPHABET *abc,
                           ESL_SQ **sqarr, int64_t nseq, char *errbuf);
extern int  p7_gpudb_Open(const char *basename, P7_GPUDB **ret_gdb, char *errbuf);
extern void p7_gpudb_Close(P7_GPUDB *gdb);

#endif /* p7GPUDB_INCLUDED */
