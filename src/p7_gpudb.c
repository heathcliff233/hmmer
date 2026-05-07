/* p7_gpudb.c
 * GPU-native database: memory-mappable, pre-unpacked sequence format.
 * v2 adds embedded metadata (name, acc, desc, taxid) for zero-I/O survivor materialization.
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
  int64_t      *meta_offsets = NULL;
  char         *meta_blob = NULL;
  int64_t       meta_blob_size = 0;
  int64_t       meta_blob_alloc = 0;

  if (esl_sprintf(&fname, "%s.gpudb", basename) != eslOK)
    ESL_XFAIL(eslEMEM, errbuf, "allocation failed for gpudb filename");

  fp = fopen(fname, "wb");
  if (!fp) ESL_XFAIL(eslEWRITE, errbuf, "failed to open %s for writing", fname);

  for (i = 0; i < nseq; i++) {
    nres += sqarr[i]->n;
    if (sqarr[i]->n > max_seqlen) max_seqlen = sqarr[i]->n;
  }

  data_size = 0;
  for (i = 0; i < nseq; i++)
    data_size += (int64_t) sqarr[i]->n + 1;
  data_size += 1;

  size_t idx_offset = sizeof(P7_GPUDB_HEADER);
  size_t idx_size   = (size_t) nseq * (sizeof(int64_t) + sizeof(int32_t));
  size_t data_offset = align_up(idx_offset + idx_size, PAGE_SIZE);

  /* Build metadata blob: name\0 acc\0 desc\0 taxid(4B) per sequence */
  meta_blob_alloc = 64 * nseq;
  ESL_ALLOC(meta_blob, meta_blob_alloc);
  ESL_ALLOC(meta_offsets, sizeof(int64_t) * nseq);

  for (i = 0; i < nseq; i++) {
    const char *name = (sqarr[i]->name && sqarr[i]->name[0]) ? sqarr[i]->name : "";
    const char *acc  = (sqarr[i]->acc  && sqarr[i]->acc[0])  ? sqarr[i]->acc  : "";
    const char *desc = (sqarr[i]->desc && sqarr[i]->desc[0]) ? sqarr[i]->desc : "";
    int32_t taxid    = (int32_t) sqarr[i]->tax_id;
    size_t name_len  = strlen(name) + 1;
    size_t acc_len   = strlen(acc)  + 1;
    size_t desc_len  = strlen(desc) + 1;
    size_t entry_len = name_len + acc_len + desc_len + sizeof(int32_t);

    while (meta_blob_size + (int64_t)entry_len > meta_blob_alloc) {
      meta_blob_alloc *= 2;
      char *tmp;
      ESL_RALLOC(meta_blob, tmp, meta_blob_alloc);
    }
    meta_offsets[i] = meta_blob_size;
    memcpy(meta_blob + meta_blob_size, name, name_len);  meta_blob_size += name_len;
    memcpy(meta_blob + meta_blob_size, acc,  acc_len);   meta_blob_size += acc_len;
    memcpy(meta_blob + meta_blob_size, desc, desc_len);  meta_blob_size += desc_len;
    memcpy(meta_blob + meta_blob_size, &taxid, sizeof(int32_t)); meta_blob_size += sizeof(int32_t);
  }

  size_t meta_idx_offset  = align_up(data_offset + (size_t) data_size, PAGE_SIZE);
  size_t meta_blob_offset = meta_idx_offset + (size_t) nseq * sizeof(int64_t);

  memset(&hdr, 0, sizeof(hdr));
  hdr.magic            = P7_GPUDB_MAGIC;
  hdr.version          = P7_GPUDB_VERSION;
  hdr.nseq             = (uint64_t) nseq;
  hdr.nres             = (uint64_t) nres;
  hdr.max_seqlen       = (uint64_t) max_seqlen;
  hdr.data_offset      = (uint64_t) data_offset;
  hdr.idx_offset       = (uint64_t) idx_offset;
  hdr.meta_idx_offset  = (uint64_t) meta_idx_offset;
  hdr.meta_blob_offset = (uint64_t) meta_blob_offset;
  hdr.meta_blob_size   = (uint64_t) meta_blob_size;

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
  {
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
  }

  /* Write sequence data */
  for (i = 0; i < nseq; i++) {
    uint8_t sentinel = eslDSQ_SENTINEL;
    if (fwrite(&sentinel, 1, 1, fp) != 1)
      ESL_XFAIL(eslEWRITE, errbuf, "failed to write leading sentinel");
    if (fwrite(sqarr[i]->dsq + 1, 1, sqarr[i]->n, fp) != (size_t) sqarr[i]->n)
      ESL_XFAIL(eslEWRITE, errbuf, "failed to write sequence data");
  }
  { uint8_t sentinel = eslDSQ_SENTINEL;
    if (fwrite(&sentinel, 1, 1, fp) != 1)
      ESL_XFAIL(eslEWRITE, errbuf, "failed to write trailing sentinel");
  }

  /* Pad to page boundary for metadata region */
  {
    size_t current_pos = data_offset + (size_t) data_size;
    if (current_pos < meta_idx_offset) {
      size_t pad_size = meta_idx_offset - current_pos;
      char  *pad = calloc(1, pad_size);
      if (!pad) ESL_XFAIL(eslEMEM, errbuf, "allocation failed for metadata padding");
      if (fwrite(pad, 1, pad_size, fp) != pad_size) {
        free(pad);
        ESL_XFAIL(eslEWRITE, errbuf, "failed to write metadata padding");
      }
      free(pad);
    }
  }

  /* Write metadata index and blob */
  if (fwrite(meta_offsets, sizeof(int64_t), nseq, fp) != (size_t) nseq)
    ESL_XFAIL(eslEWRITE, errbuf, "failed to write metadata index");
  if (fwrite(meta_blob, 1, (size_t) meta_blob_size, fp) != (size_t) meta_blob_size)
    ESL_XFAIL(eslEWRITE, errbuf, "failed to write metadata blob");

  fclose(fp); fp = NULL;
  free(offsets);
  free(lengths);
  free(meta_offsets);
  free(meta_blob);
  free(fname);
  return eslOK;

 ERROR:
  if (fp) fclose(fp);
  free(offsets);
  free(lengths);
  free(meta_offsets);
  free(meta_blob);
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

  /* Sequential hint for initial GPU upload */
  madvise(gdb->mmap_base, gdb->mmap_size, MADV_SEQUENTIAL);

  if (gdb->mmap_size < sizeof(P7_GPUDB_HEADER))
    ESL_XFAIL(eslEFORMAT, errbuf, "gpudb file too small for header");

  memcpy(&gdb->hdr, gdb->mmap_base, sizeof(P7_GPUDB_HEADER));
  if (gdb->hdr.magic != P7_GPUDB_MAGIC)
    ESL_XFAIL(eslEFORMAT, errbuf, "gpudb file has wrong magic number");
  if (gdb->hdr.version != P7_GPUDB_VERSION && gdb->hdr.version != 1)
    ESL_XFAIL(eslEFORMAT, errbuf, "gpudb file version %u not supported", gdb->hdr.version);

  gdb->offsets  = (int64_t *) ((uint8_t *) gdb->mmap_base + gdb->hdr.idx_offset);
  gdb->lengths  = (int32_t *) ((uint8_t *) gdb->mmap_base + gdb->hdr.idx_offset + sizeof(int64_t) * gdb->hdr.nseq);
  gdb->seq_data = (uint8_t *) gdb->mmap_base + gdb->hdr.data_offset;

  /* Parse v2 metadata section */
  if (gdb->hdr.version >= 2 && gdb->hdr.meta_idx_offset > 0 && gdb->hdr.meta_blob_offset > 0) {
    gdb->meta_offsets = (int64_t *) ((uint8_t *) gdb->mmap_base + gdb->hdr.meta_idx_offset);
    gdb->meta_blob    = (char *)    ((uint8_t *) gdb->mmap_base + gdb->hdr.meta_blob_offset);
    gdb->has_metadata = TRUE;

    /* Random access hint for metadata (sparse survivor lookups) */
    size_t meta_region_start = (size_t) gdb->hdr.meta_idx_offset;
    size_t meta_region_end   = (size_t) gdb->hdr.meta_blob_offset + (size_t) gdb->hdr.meta_blob_size;
    madvise((uint8_t *) gdb->mmap_base + meta_region_start, meta_region_end - meta_region_start, MADV_RANDOM);
  } else {
    gdb->meta_offsets = NULL;
    gdb->meta_blob    = NULL;
    gdb->has_metadata = FALSE;
  }

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

int
p7_gpudb_GetMetadata(const P7_GPUDB *gdb, int64_t idx,
                     const char **ret_name, const char **ret_acc,
                     const char **ret_desc, int *ret_taxid)
{
  const char *p;
  const char *name;
  const char *acc;
  const char *desc;

  if (!gdb || !gdb->has_metadata) return eslEINVAL;
  if (idx < 0 || idx >= (int64_t) gdb->hdr.nseq) return eslEINVAL;

  p = gdb->meta_blob + gdb->meta_offsets[idx];
  name = p;
  p += strlen(p) + 1;
  acc = p;
  p += strlen(p) + 1;
  desc = p;
  p += strlen(p) + 1;

  if (ret_name)  *ret_name  = name;
  if (ret_acc)   *ret_acc   = acc;
  if (ret_desc)  *ret_desc  = desc;
  if (ret_taxid) {
    int32_t taxid;
    memcpy(&taxid, p, sizeof(int32_t));
    *ret_taxid = (int) taxid;
  }
  return eslOK;
}
