/* p7_gpudb.c
 * GPU-native database: memory-mappable, pre-unpacked sequence format.
 */
#include <p7_config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_sq.h"

#include "p7_gpudb.h"

#define PAGE_SIZE 4096

static size_t
align_up(size_t val, size_t alignment)
{
  return (val + alignment - 1) & ~(alignment - 1);
}

int
p7_gpudb_Write(const char *basename, const ESL_ALPHABET *abc,
               ESL_SQ **sqarr, int64_t nseq, char *errbuf)
{
  char         *fname = NULL;
  FILE         *fp    = NULL;
  int           status;
  int64_t       i;
  int64_t       nres = 0;
  int64_t       max_seqlen = 0;
  int64_t       data_size;
  int64_t       running_offset;
  P7_GPUDB_HEADER hdr;
  int64_t      *offsets  = NULL;
  int32_t      *lengths  = NULL;

  if (esl_sprintf(&fname, "%s.gpudb", basename) != eslOK)
    ESL_XFAIL(eslEMEM, errbuf, "allocation failed for gpudb filename");

  fp = fopen(fname, "wb");
  if (!fp) ESL_XFAIL(eslEWRITE, errbuf, "failed to open %s for writing", fname);

  for (i = 0; i < nseq; i++) {
    nres += sqarr[i]->n;
    if (sqarr[i]->n > max_seqlen) max_seqlen = sqarr[i]->n;
  }

  /* Compute layout: each sequence stored as [sentinel][residues], L+1 bytes */
  data_size = 0;
  for (i = 0; i < nseq; i++)
    data_size += (int64_t) sqarr[i]->n + 1;
  data_size += 1;  /* trailing sentinel */

  /* Header at offset 0 (64 bytes) */
  /* Index at offset 64 */
  size_t idx_offset = sizeof(P7_GPUDB_HEADER);
  size_t idx_size   = (size_t) nseq * (sizeof(int64_t) + sizeof(int32_t));
  size_t data_offset = align_up(idx_offset + idx_size, PAGE_SIZE);

  memset(&hdr, 0, sizeof(hdr));
  hdr.magic       = P7_GPUDB_MAGIC;
  hdr.version     = P7_GPUDB_VERSION;
  hdr.nseq        = (uint64_t) nseq;
  hdr.nres        = (uint64_t) nres;
  hdr.max_seqlen  = (uint64_t) max_seqlen;
  hdr.data_offset = (uint64_t) data_offset;
  hdr.idx_offset  = (uint64_t) idx_offset;

  if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1)
    ESL_XFAIL(eslEWRITE, errbuf, "failed to write gpudb header");

  /* Write index: offsets and lengths */
  ESL_ALLOC(offsets, sizeof(int64_t) * nseq);
  ESL_ALLOC(lengths, sizeof(int32_t) * nseq);

  running_offset = 0;
  for (i = 0; i < nseq; i++) {
    offsets[i] = running_offset;
    lengths[i] = (int32_t) sqarr[i]->n;
    running_offset += (int64_t) sqarr[i]->n + 1;
  }

  if (fwrite(offsets, sizeof(int64_t), nseq, fp) != (size_t) nseq)
    ESL_XFAIL(eslEWRITE, errbuf, "failed to write gpudb offsets");
  if (fwrite(lengths, sizeof(int32_t), nseq, fp) != (size_t) nseq)
    ESL_XFAIL(eslEWRITE, errbuf, "failed to write gpudb lengths");

  /* Pad to page boundary for data region */
  size_t current_pos = idx_offset + idx_size;
  if (current_pos < data_offset) {
    size_t pad_size = data_offset - current_pos;
    char  *pad = calloc(1, pad_size);
    if (!pad) ESL_XFAIL(eslEMEM, errbuf, "allocation failed for padding");
    if (fwrite(pad, 1, pad_size, fp) != pad_size) {
      free(pad);
      ESL_XFAIL(eslEWRITE, errbuf, "failed to write gpudb padding");
    }
    free(pad);
  }

  /* Write sequence data: contiguous, L+1 spacing per sequence.
   * Layout matches Easel dsq convention: [sentinel][residues_1..L] per sequence.
   * The CUDA kernel accesses sdsq[1..L] starting from offset, so leading sentinel is required. */
  for (i = 0; i < nseq; i++) {
    uint8_t sentinel = eslDSQ_SENTINEL;
    if (fwrite(&sentinel, 1, 1, fp) != 1)
      ESL_XFAIL(eslEWRITE, errbuf, "failed to write leading sentinel");
    if (fwrite(sqarr[i]->dsq + 1, 1, sqarr[i]->n, fp) != (size_t) sqarr[i]->n)
      ESL_XFAIL(eslEWRITE, errbuf, "failed to write sequence data");
  }
  /* Final trailing sentinel (for safety; not strictly needed) */
  { uint8_t sentinel = eslDSQ_SENTINEL;
    if (fwrite(&sentinel, 1, 1, fp) != 1)
      ESL_XFAIL(eslEWRITE, errbuf, "failed to write trailing sentinel");
  }

  fclose(fp); fp = NULL;
  free(offsets);
  free(lengths);
  free(fname);
  return eslOK;

 ERROR:
  if (fp) fclose(fp);
  free(offsets);
  free(lengths);
  free(fname);
  return status;
}

int
p7_gpudb_Open(const char *basename, P7_GPUDB **ret_gdb, char *errbuf)
{
  P7_GPUDB    *gdb   = NULL;
  char        *fname = NULL;
  struct stat  st;
  int          status;

  if (ret_gdb) *ret_gdb = NULL;

  ESL_ALLOC(gdb, sizeof(P7_GPUDB));
  memset(gdb, 0, sizeof(P7_GPUDB));
  gdb->fd = -1;

  {
    size_t blen = strlen(basename);
    if (blen > 6 && strcmp(basename + blen - 6, ".gpudb") == 0) {
      if (esl_sprintf(&fname, "%s", basename) != eslOK)
        ESL_XFAIL(eslEMEM, errbuf, "allocation failed for gpudb filename");
    } else {
      if (esl_sprintf(&fname, "%s.gpudb", basename) != eslOK)
        ESL_XFAIL(eslEMEM, errbuf, "allocation failed for gpudb filename");
    }
  }

  gdb->fd = open(fname, O_RDONLY);
  if (gdb->fd < 0) ESL_XFAIL(eslENOTFOUND, errbuf, "failed to open %s", fname);

  if (fstat(gdb->fd, &st) != 0)
    ESL_XFAIL(eslESYS, errbuf, "fstat failed on %s", fname);
  gdb->mmap_size = (size_t) st.st_size;

  gdb->mmap_base = mmap(NULL, gdb->mmap_size, PROT_READ, MAP_PRIVATE, gdb->fd, 0);
  if (gdb->mmap_base == MAP_FAILED) {
    gdb->mmap_base = NULL;
    ESL_XFAIL(eslESYS, errbuf, "mmap failed on %s", fname);
  }

  /* Parse header */
  if (gdb->mmap_size < sizeof(P7_GPUDB_HEADER))
    ESL_XFAIL(eslEFORMAT, errbuf, "gpudb file too small for header");

  memcpy(&gdb->hdr, gdb->mmap_base, sizeof(P7_GPUDB_HEADER));
  if (gdb->hdr.magic != P7_GPUDB_MAGIC)
    ESL_XFAIL(eslEFORMAT, errbuf, "gpudb file has wrong magic number");
  if (gdb->hdr.version != P7_GPUDB_VERSION)
    ESL_XFAIL(eslEFORMAT, errbuf, "gpudb file version %u not supported", gdb->hdr.version);

  /* Set up pointers into mmap'd region */
  gdb->offsets  = (int64_t *) ((uint8_t *) gdb->mmap_base + gdb->hdr.idx_offset);
  gdb->lengths  = (int32_t *) ((uint8_t *) gdb->mmap_base + gdb->hdr.idx_offset + sizeof(int64_t) * gdb->hdr.nseq);
  gdb->seq_data = (uint8_t *) gdb->mmap_base + gdb->hdr.data_offset;

  gdb->filename = fname;
  fname = NULL;

  *ret_gdb = gdb;
  return eslOK;

 ERROR:
  p7_gpudb_Close(gdb);
  free(fname);
  return status;
}

void
p7_gpudb_Close(P7_GPUDB *gdb)
{
  if (!gdb) return;
  if (gdb->mmap_base) munmap(gdb->mmap_base, gdb->mmap_size);
  if (gdb->fd >= 0)   close(gdb->fd);
  free(gdb->filename);
  free(gdb);
}
