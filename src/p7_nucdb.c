/* p7_nucdb.c
 * GPU-native nucleotide database: pre-chunked, both strands, memory-mappable.
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
#include "esl_sqio.h"

#include "p7_nucdb.h"

#define PAGE_SIZE 4096

static size_t
align_up(size_t val, size_t alignment)
{
  return (val + alignment - 1) & ~(alignment - 1);
}

/* Compute number of chunks for a sequence of length L with given chunk_size and overlap. */
static int
compute_nchunks(int64_t L, int64_t chunk_size, int64_t overlap)
{
  if (L <= chunk_size) return 1;
  int64_t step = chunk_size - overlap;
  if (step < 1) step = 1;
  return 1 + (int)((L - chunk_size + step - 1) / step);
}

int
p7_nucdb_Write(const char *basename, const ESL_ALPHABET *abc,
               ESL_SQFILE *sqfp, int64_t chunk_size, int64_t overlap,
               int both_strands, char *errbuf)
{
  char    *fname = NULL;
  FILE    *fp    = NULL;
  int      status;

  /* First pass: read all sequences to compute layout */
  ESL_SQ **sqarr     = NULL;
  int64_t  nseq      = 0;
  int64_t  alloc_n   = 4096;
  int64_t  nres      = 0;
  int64_t  max_seqlen = 0;
  int64_t  total_chunks = 0;
  int64_t  total_data = 0;
  int64_t  name_blob_size = 0;

  ESL_SQ  *sq = esl_sq_CreateDigital(abc);
  if (!sq) ESL_XFAIL(eslEMEM, errbuf, "allocation failed");

  sqarr = (ESL_SQ **)malloc(sizeof(ESL_SQ *) * alloc_n);
  if (!sqarr) ESL_XFAIL(eslEMEM, errbuf, "allocation failed");

  while ((status = esl_sqio_Read(sqfp, sq)) == eslOK) {
    if (nseq >= alloc_n) {
      alloc_n *= 2;
      ESL_REALLOC(sqarr, sizeof(ESL_SQ *) * alloc_n);
    }
    sqarr[nseq] = sq;
    nres += sq->n;
    if (sq->n > max_seqlen) max_seqlen = sq->n;
    nseq++;
    sq = esl_sq_CreateDigital(abc);
    if (!sq) ESL_XFAIL(eslEMEM, errbuf, "allocation failed");
  }
  esl_sq_Destroy(sq);
  if (status != eslEOF) ESL_XFAIL(status, errbuf, "error reading sequences");

  /* Compute total chunks and data size */
  for (int64_t i = 0; i < nseq; i++) {
    int nc = compute_nchunks(sqarr[i]->n, chunk_size, overlap);
    total_chunks += nc;
    if (both_strands) total_chunks += nc;

    /* Each chunk: [sentinel][residues], plus trailing sentinel after all chunks of each strand */
    for (int c = 0; c < nc; c++) {
      int64_t seq_start = c * (chunk_size - overlap);
      int64_t clen = chunk_size;
      if (seq_start + clen > sqarr[i]->n) clen = sqarr[i]->n - seq_start;
      total_data += clen + 1;  /* sentinel + residues */
    }
    if (both_strands) {
      for (int c = 0; c < nc; c++) {
        int64_t seq_start = c * (chunk_size - overlap);
        int64_t clen = chunk_size;
        if (seq_start + clen > sqarr[i]->n) clen = sqarr[i]->n - seq_start;
        total_data += clen + 1;
      }
    }
    name_blob_size += strlen(sqarr[i]->name) + 1;
  }
  total_data += 1;  /* trailing sentinel */

  /* Layout */
  size_t hdr_size       = sizeof(P7_NUCDB_HEADER);
  size_t chunk_idx_off  = hdr_size;
  size_t chunk_idx_size = (size_t)total_chunks * sizeof(P7_NUCDB_CHUNK_IDX);
  size_t seq_idx_off    = chunk_idx_off + chunk_idx_size;
  size_t seq_idx_size   = (size_t)nseq * sizeof(P7_NUCDB_SEQ_IDX);
  size_t name_blob_off  = seq_idx_off + seq_idx_size;
  size_t data_off       = align_up(name_blob_off + name_blob_size, PAGE_SIZE);

  /* Write */
  if (esl_sprintf(&fname, "%s.nucdb", basename) != eslOK)
    ESL_XFAIL(eslEMEM, errbuf, "allocation failed");

  fp = fopen(fname, "wb");
  if (!fp) ESL_XFAIL(eslEWRITE, errbuf, "failed to open %s for writing", fname);

  P7_NUCDB_HEADER hdr;
  memset(&hdr, 0, sizeof(hdr));
  hdr.magic            = P7_NUCDB_MAGIC;
  hdr.version          = P7_NUCDB_VERSION;
  hdr.nseq             = (uint64_t)nseq;
  hdr.nres             = (uint64_t)nres;
  hdr.max_seqlen       = (uint64_t)max_seqlen;
  hdr.nchunks          = (uint64_t)total_chunks;
  hdr.chunk_size       = (uint64_t)chunk_size;
  hdr.overlap          = (uint64_t)overlap;
  hdr.data_offset      = (uint64_t)data_off;
  hdr.chunk_idx_offset = (uint64_t)chunk_idx_off;
  hdr.seq_idx_offset   = (uint64_t)seq_idx_off;
  hdr.both_strands     = both_strands ? 1 : 0;

  if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1)
    ESL_XFAIL(eslEWRITE, errbuf, "failed to write header");

  /* Write chunk index and sequence metadata */
  P7_NUCDB_CHUNK_IDX *cidx = NULL;
  P7_NUCDB_SEQ_IDX   *sidx = NULL;
  char               *names = NULL;
  ESL_ALLOC(cidx,  sizeof(P7_NUCDB_CHUNK_IDX) * total_chunks);
  ESL_ALLOC(sidx,  sizeof(P7_NUCDB_SEQ_IDX)   * nseq);
  ESL_ALLOC(names, name_blob_size);

  int64_t ci = 0;
  int64_t data_pos = 0;
  int64_t name_pos = 0;

  for (int64_t i = 0; i < nseq; i++) {
    int nc = compute_nchunks(sqarr[i]->n, chunk_size, overlap);
    int64_t step = chunk_size - overlap;
    if (step < 1) step = 1;

    sidx[i].name_offset    = name_pos;
    sidx[i].length         = sqarr[i]->n;
    sidx[i].fwd_chunk_start = (int32_t)ci;
    sidx[i].fwd_chunk_count = nc;

    /* Forward chunks */
    for (int c = 0; c < nc; c++) {
      int64_t seq_start = c * step;
      int64_t clen = chunk_size;
      if (seq_start + clen > sqarr[i]->n) clen = sqarr[i]->n - seq_start;

      cidx[ci].data_offset      = data_pos;
      cidx[ci].length           = (int32_t)clen;
      cidx[ci].seq_id           = (int32_t)i;
      cidx[ci].seq_offset       = seq_start;
      cidx[ci].complementarity  = 0;
      memset(cidx[ci].pad, 0, sizeof(cidx[ci].pad));
      ci++;
      data_pos += clen + 1;
    }

    sidx[i].rc_chunk_start = (int32_t)ci;
    sidx[i].rc_chunk_count = both_strands ? nc : 0;

    if (both_strands) {
      for (int c = 0; c < nc; c++) {
        int64_t seq_start = c * step;
        int64_t clen = chunk_size;
        if (seq_start + clen > sqarr[i]->n) clen = sqarr[i]->n - seq_start;

        cidx[ci].data_offset      = data_pos;
        cidx[ci].length           = (int32_t)clen;
        cidx[ci].seq_id           = (int32_t)i;
        cidx[ci].seq_offset       = seq_start;
        cidx[ci].complementarity  = 1;
        memset(cidx[ci].pad, 0, sizeof(cidx[ci].pad));
        ci++;
        data_pos += clen + 1;
      }
    }

    /* Copy name */
    size_t nlen = strlen(sqarr[i]->name) + 1;
    memcpy(names + name_pos, sqarr[i]->name, nlen);
    name_pos += nlen;
  }

  if (fwrite(cidx, sizeof(P7_NUCDB_CHUNK_IDX), total_chunks, fp) != (size_t)total_chunks)
    ESL_XFAIL(eslEWRITE, errbuf, "failed to write chunk index");
  if (fwrite(sidx, sizeof(P7_NUCDB_SEQ_IDX), nseq, fp) != (size_t)nseq)
    ESL_XFAIL(eslEWRITE, errbuf, "failed to write sequence index");
  if (fwrite(names, 1, name_blob_size, fp) != name_blob_size)
    ESL_XFAIL(eslEWRITE, errbuf, "failed to write name blob");

  /* Pad to page boundary */
  size_t current_pos = name_blob_off + name_blob_size;
  if (current_pos < data_off) {
    size_t pad_size = data_off - current_pos;
    char *pad = calloc(1, pad_size);
    if (!pad) ESL_XFAIL(eslEMEM, errbuf, "allocation failed for padding");
    if (fwrite(pad, 1, pad_size, fp) != pad_size) {
      free(pad);
      ESL_XFAIL(eslEWRITE, errbuf, "failed to write padding");
    }
    free(pad);
  }

  /* Write chunk data: forward chunks for each sequence, then RC chunks */
  for (int64_t i = 0; i < nseq; i++) {
    int nc = compute_nchunks(sqarr[i]->n, chunk_size, overlap);
    int64_t step = chunk_size - overlap;
    if (step < 1) step = 1;

    /* Forward chunks */
    for (int c = 0; c < nc; c++) {
      int64_t seq_start = c * step;
      int64_t clen = chunk_size;
      if (seq_start + clen > sqarr[i]->n) clen = sqarr[i]->n - seq_start;

      uint8_t sentinel = eslDSQ_SENTINEL;
      if (fwrite(&sentinel, 1, 1, fp) != 1) ESL_XFAIL(eslEWRITE, errbuf, "failed to write sentinel");
      if (fwrite(sqarr[i]->dsq + 1 + seq_start, 1, clen, fp) != (size_t)clen)
        ESL_XFAIL(eslEWRITE, errbuf, "failed to write chunk data");
    }

    /* Reverse complement chunks */
    if (both_strands) {
      ESL_SQ *rc = esl_sq_CreateDigital(abc);
      if (!rc) ESL_XFAIL(eslEMEM, errbuf, "allocation failed for RC");
      esl_sq_Copy(sqarr[i], rc);
      esl_sq_ReverseComplement(rc);

      for (int c = 0; c < nc; c++) {
        int64_t seq_start = c * step;
        int64_t clen = chunk_size;
        if (seq_start + clen > rc->n) clen = rc->n - seq_start;

        uint8_t sentinel = eslDSQ_SENTINEL;
        if (fwrite(&sentinel, 1, 1, fp) != 1) ESL_XFAIL(eslEWRITE, errbuf, "failed to write sentinel");
        if (fwrite(rc->dsq + 1 + seq_start, 1, clen, fp) != (size_t)clen)
          ESL_XFAIL(eslEWRITE, errbuf, "failed to write RC chunk data");
      }
      esl_sq_Destroy(rc);
    }
  }

  /* Trailing sentinel */
  { uint8_t sentinel = eslDSQ_SENTINEL;
    if (fwrite(&sentinel, 1, 1, fp) != 1)
      ESL_XFAIL(eslEWRITE, errbuf, "failed to write trailing sentinel");
  }

  fclose(fp); fp = NULL;
  free(cidx);
  free(sidx);
  free(names);
  for (int64_t i = 0; i < nseq; i++) esl_sq_Destroy(sqarr[i]);
  free(sqarr);
  free(fname);
  return eslOK;

 ERROR:
  if (fp) fclose(fp);
  free(cidx);
  free(sidx);
  free(names);
  if (sqarr) {
    for (int64_t i = 0; i < nseq; i++) if (sqarr[i]) esl_sq_Destroy(sqarr[i]);
    free(sqarr);
  }
  free(fname);
  return status;
}

int
p7_nucdb_Open(const char *basename, P7_NUCDB **ret_ndb, char *errbuf)
{
  P7_NUCDB    *ndb   = NULL;
  char        *fname = NULL;
  struct stat  st;
  int          status;

  if (ret_ndb) *ret_ndb = NULL;

  ESL_ALLOC(ndb, sizeof(P7_NUCDB));
  memset(ndb, 0, sizeof(P7_NUCDB));
  ndb->fd = -1;

  {
    size_t blen = strlen(basename);
    if (blen > 6 && strcmp(basename + blen - 6, ".nucdb") == 0) {
      if (esl_sprintf(&fname, "%s", basename) != eslOK)
        ESL_XFAIL(eslEMEM, errbuf, "allocation failed");
    } else {
      if (esl_sprintf(&fname, "%s.nucdb", basename) != eslOK)
        ESL_XFAIL(eslEMEM, errbuf, "allocation failed");
    }
  }

  ndb->fd = open(fname, O_RDONLY);
  if (ndb->fd < 0) ESL_XFAIL(eslENOTFOUND, errbuf, "failed to open %s", fname);

  if (fstat(ndb->fd, &st) != 0)
    ESL_XFAIL(eslESYS, errbuf, "fstat failed on %s", fname);
  ndb->mmap_size = (size_t)st.st_size;

  ndb->mmap_base = mmap(NULL, ndb->mmap_size, PROT_READ, MAP_PRIVATE, ndb->fd, 0);
  if (ndb->mmap_base == MAP_FAILED) {
    ndb->mmap_base = NULL;
    ESL_XFAIL(eslESYS, errbuf, "mmap failed on %s", fname);
  }

  if (ndb->mmap_size < sizeof(P7_NUCDB_HEADER))
    ESL_XFAIL(eslEFORMAT, errbuf, "nucdb file too small for header");

  memcpy(&ndb->hdr, ndb->mmap_base, sizeof(P7_NUCDB_HEADER));
  if (ndb->hdr.magic != P7_NUCDB_MAGIC)
    ESL_XFAIL(eslEFORMAT, errbuf, "nucdb file has wrong magic number");
  if (ndb->hdr.version != P7_NUCDB_VERSION)
    ESL_XFAIL(eslEFORMAT, errbuf, "nucdb file version %u not supported", ndb->hdr.version);

  uint8_t *base = (uint8_t *)ndb->mmap_base;
  ndb->chunk_idx  = (P7_NUCDB_CHUNK_IDX *)(base + ndb->hdr.chunk_idx_offset);
  ndb->seq_idx    = (P7_NUCDB_SEQ_IDX *)  (base + ndb->hdr.seq_idx_offset);
  ndb->name_blob  = (char *)              (base + ndb->hdr.seq_idx_offset +
                                           sizeof(P7_NUCDB_SEQ_IDX) * ndb->hdr.nseq);
  ndb->chunk_data = base + ndb->hdr.data_offset;

  ndb->filename = fname;
  fname = NULL;

  *ret_ndb = ndb;
  return eslOK;

 ERROR:
  p7_nucdb_Close(ndb);
  free(fname);
  return status;
}

void
p7_nucdb_Close(P7_NUCDB *ndb)
{
  if (!ndb) return;
  if (ndb->mmap_base) munmap(ndb->mmap_base, ndb->mmap_size);
  if (ndb->fd >= 0)   close(ndb->fd);
  free(ndb->filename);
  free(ndb);
}
