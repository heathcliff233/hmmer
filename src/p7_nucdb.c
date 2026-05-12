/* p7_nucdb.c
 * GPU-native nucleotide database (v2): 2-bit packed residues + mask bitmap,
 * forward-strand only. The reverse complement is generated on the fly at
 * query time by GPU kernels and by the CPU slice-fill helper.
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

static int64_t
packed_bytes_for(int64_t clen)
{
  return (clen + 3) >> 2;
}

static int64_t
mask_bytes_for(int64_t clen)
{
  return (clen + 7) >> 3;
}

/* Number of chunks for a sequence of length L given chunk_size and overlap. */
static int
compute_nchunks(int64_t L, int64_t chunk_size, int64_t overlap)
{
  if (L <= chunk_size) return 1;
  int64_t step = chunk_size - overlap;
  if (step < 1) step = 1;
  return 1 + (int)((L - chunk_size + step - 1) / step);
}

/* Build a dsq-code -> (2bit, is_mask) lookup table for DNA.
 * Canonical A/C/G/T (dsq codes 0..3) map to their 2-bit value, mask=0.
 * Everything else (gap, IUPAC degenerates, N, *, ~) -> packed=0, mask=1.
 */
static void
build_encode_tables(uint8_t packed_of_dsq[256], uint8_t mask_of_dsq[256])
{
  int i;
  for (i = 0; i < 256; i++) {
    packed_of_dsq[i] = 0;
    mask_of_dsq[i]   = 1;
  }
  packed_of_dsq[0] = 0; mask_of_dsq[0] = 0;  /* A */
  packed_of_dsq[1] = 1; mask_of_dsq[1] = 0;  /* C */
  packed_of_dsq[2] = 2; mask_of_dsq[2] = 0;  /* G */
  packed_of_dsq[3] = 3; mask_of_dsq[3] = 0;  /* T (eslDNA) */
}

/* Pack one chunk [seq_start .. seq_start+clen-1] of dsq into packed+mask
 * buffers of size packed_bytes_for(clen) / mask_bytes_for(clen).
 */
static void
pack_chunk(const ESL_DSQ *dsq, int64_t seq_start, int64_t clen,
           uint8_t *packed_out, uint8_t *mask_out,
           const uint8_t packed_of_dsq[256], const uint8_t mask_of_dsq[256])
{
  int64_t packed_bytes = packed_bytes_for(clen);
  int64_t mask_bytes   = mask_bytes_for(clen);
  if (packed_bytes > 0) memset(packed_out, 0, (size_t)packed_bytes);
  if (mask_bytes   > 0) memset(mask_out,   0, (size_t)mask_bytes);

  for (int64_t i = 0; i < clen; i++) {
    uint8_t x = dsq[1 + seq_start + i];  /* dsq is 1-indexed */
    uint8_t code = packed_of_dsq[x];
    uint8_t m    = mask_of_dsq[x];
    int     shift = (int)((i & 3) << 1);
    packed_out[i >> 2] |= (uint8_t)((code & 0x3) << shift);
    if (m) mask_out[i >> 3] |= (uint8_t)(1u << (i & 7));
  }
}

int
p7_nucdb_Write(const char *basename, const ESL_ALPHABET *abc,
               ESL_SQFILE *sqfp, int64_t chunk_size, int64_t overlap,
               char *errbuf)
{
  char    *fname = NULL;
  FILE    *fp    = NULL;
  int      status;

  (void) abc;  /* abc is implicit in the dsq codes; kept for API symmetry. */

  /* These are declared up-front so the ERROR label can free them even
   * when control jumps past their initialization (C goto semantics). */
  ESL_SQ **sqarr     = NULL;
  ESL_SQ  *sq        = NULL;
  P7_NUCDB_CHUNK_IDX *cidx = NULL;
  P7_NUCDB_SEQ_IDX   *sidx = NULL;
  char               *names = NULL;

  int64_t  nseq      = 0;
  int64_t  alloc_n   = 4096;
  int64_t  nres      = 0;
  int64_t  max_seqlen = 0;
  int64_t  total_chunks = 0;
  int64_t  name_blob_size = 0;

  sq = esl_sq_CreateDigital(abc);
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
  sq = NULL;
  if (status != eslEOF) ESL_XFAIL(status, errbuf, "error reading sequences");

  /* Compute total chunks, packed bytes, mask bytes */
  int64_t packed_region_bytes = 0;
  int64_t mask_region_bytes   = 0;

  for (int64_t i = 0; i < nseq; i++) {
    int nc = compute_nchunks(sqarr[i]->n, chunk_size, overlap);
    total_chunks += nc;

    int64_t step = chunk_size - overlap;
    if (step < 1) step = 1;
    for (int c = 0; c < nc; c++) {
      int64_t seq_start = (int64_t)c * step;
      int64_t clen = chunk_size;
      if (seq_start + clen > sqarr[i]->n) clen = sqarr[i]->n - seq_start;

      int64_t pb = packed_bytes_for(clen);
      int64_t mb = mask_bytes_for(clen);
      /* Pad each chunk's packed bytes to a 4-byte boundary so the next
       * chunk starts word-aligned; same for mask. */
      packed_region_bytes += ((pb + 3) & ~((int64_t)3));
      mask_region_bytes   += ((mb + 3) & ~((int64_t)3));
    }
    name_blob_size += (int64_t)strlen(sqarr[i]->name) + 1;
  }

  /* Layout */
  size_t hdr_size       = sizeof(P7_NUCDB_HEADER);
  size_t chunk_idx_off  = hdr_size;
  size_t chunk_idx_size = (size_t)total_chunks * sizeof(P7_NUCDB_CHUNK_IDX);
  size_t seq_idx_off    = chunk_idx_off + chunk_idx_size;
  size_t seq_idx_size   = (size_t)nseq * sizeof(P7_NUCDB_SEQ_IDX);
  size_t name_blob_off  = seq_idx_off + seq_idx_size;
  size_t data_off       = align_up(name_blob_off + name_blob_size, PAGE_SIZE);
  size_t mask_off       = align_up(data_off + (size_t)packed_region_bytes, PAGE_SIZE);

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
  hdr.mask_offset      = (uint64_t)mask_off;
  hdr.chunk_idx_offset = (uint64_t)chunk_idx_off;
  hdr.seq_idx_offset   = (uint64_t)seq_idx_off;
  hdr.encoding         = P7_NUCDB_ENCODING_PACKED2BIT;

  if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1)
    ESL_XFAIL(eslEWRITE, errbuf, "failed to write header");

  /* Build chunk/seq indices + name blob in memory */
  ESL_ALLOC(cidx,  sizeof(P7_NUCDB_CHUNK_IDX) * total_chunks);
  ESL_ALLOC(sidx,  sizeof(P7_NUCDB_SEQ_IDX)   * nseq);
  ESL_ALLOC(names, name_blob_size);

  {
    int64_t ci = 0;
    int64_t data_pos = 0;
    int64_t mask_pos = 0;
    int64_t name_pos = 0;

    for (int64_t i = 0; i < nseq; i++) {
      int nc = compute_nchunks(sqarr[i]->n, chunk_size, overlap);
      int64_t step = chunk_size - overlap;
      if (step < 1) step = 1;

      sidx[i].name_offset = name_pos;
      sidx[i].length      = sqarr[i]->n;
      sidx[i].chunk_start = (int32_t)ci;
      sidx[i].chunk_count = nc;

      for (int c = 0; c < nc; c++) {
        int64_t seq_start = (int64_t)c * step;
        int64_t clen = chunk_size;
        if (seq_start + clen > sqarr[i]->n) clen = sqarr[i]->n - seq_start;

        int64_t pb = packed_bytes_for(clen);
        int64_t mb = mask_bytes_for(clen);

        cidx[ci].data_offset = data_pos;
        cidx[ci].mask_offset = mask_pos;
        cidx[ci].length      = (int32_t)clen;
        cidx[ci].seq_id      = (int32_t)i;
        cidx[ci].seq_offset  = seq_start;
        memset(cidx[ci].pad, 0, sizeof(cidx[ci].pad));
        ci++;

        data_pos += ((pb + 3) & ~((int64_t)3));
        mask_pos += ((mb + 3) & ~((int64_t)3));
      }

      size_t nlen = strlen(sqarr[i]->name) + 1;
      memcpy(names + name_pos, sqarr[i]->name, nlen);
      name_pos += nlen;
    }
  }

  if (fwrite(cidx, sizeof(P7_NUCDB_CHUNK_IDX), total_chunks, fp) != (size_t)total_chunks)
    ESL_XFAIL(eslEWRITE, errbuf, "failed to write chunk index");
  if (fwrite(sidx, sizeof(P7_NUCDB_SEQ_IDX), nseq, fp) != (size_t)nseq)
    ESL_XFAIL(eslEWRITE, errbuf, "failed to write sequence index");
  if (fwrite(names, 1, name_blob_size, fp) != (size_t)name_blob_size)
    ESL_XFAIL(eslEWRITE, errbuf, "failed to write name blob");

  /* Pad to packed-region page boundary */
  {
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
  }

  /* Pass 2a: write packed region */
  {
    uint8_t packed_of_dsq[256];
    uint8_t mask_of_dsq[256];
    uint8_t *pack_buf = NULL;
    uint8_t *mask_buf = NULL;
    int64_t pack_buf_cap = 0;
    int64_t mask_buf_cap = 0;

    build_encode_tables(packed_of_dsq, mask_of_dsq);

    for (int64_t i = 0; i < nseq; i++) {
      int nc = compute_nchunks(sqarr[i]->n, chunk_size, overlap);
      int64_t step = chunk_size - overlap;
      if (step < 1) step = 1;
      for (int c = 0; c < nc; c++) {
        int64_t seq_start = (int64_t)c * step;
        int64_t clen = chunk_size;
        if (seq_start + clen > sqarr[i]->n) clen = sqarr[i]->n - seq_start;
        int64_t pb = packed_bytes_for(clen);
        int64_t mb = mask_bytes_for(clen);
        int64_t pb_padded = (pb + 3) & ~((int64_t)3);

        if (pack_buf_cap < pb_padded) {
          free(pack_buf);
          pack_buf = (uint8_t *)malloc((size_t)pb_padded);
          if (!pack_buf) { free(mask_buf); ESL_XFAIL(eslEMEM, errbuf, "allocation failed for pack buffer"); }
          pack_buf_cap = pb_padded;
        }
        if (mask_buf_cap < mb) {
          free(mask_buf);
          mask_buf = (uint8_t *)malloc((size_t)mb);
          if (!mask_buf) { free(pack_buf); ESL_XFAIL(eslEMEM, errbuf, "allocation failed for mask buffer"); }
          mask_buf_cap = mb;
        }
        memset(pack_buf, 0, (size_t)pb_padded);
        pack_chunk(sqarr[i]->dsq, seq_start, clen, pack_buf, mask_buf,
                   packed_of_dsq, mask_of_dsq);
        if (fwrite(pack_buf, 1, (size_t)pb_padded, fp) != (size_t)pb_padded) {
          free(pack_buf); free(mask_buf);
          ESL_XFAIL(eslEWRITE, errbuf, "failed to write packed chunk");
        }
      }
    }
    free(pack_buf);
    {
      size_t current_pos = data_off + (size_t)packed_region_bytes;
      if (current_pos < mask_off) {
        size_t pad_size = mask_off - current_pos;
        char *pad = calloc(1, pad_size);
        if (!pad) { free(mask_buf); ESL_XFAIL(eslEMEM, errbuf, "allocation failed for padding"); }
        if (fwrite(pad, 1, pad_size, fp) != pad_size) {
          free(pad); free(mask_buf);
          ESL_XFAIL(eslEWRITE, errbuf, "failed to write padding between packed and mask");
        }
        free(pad);
      }
    }

    /* Pass 2b: write mask region */
    for (int64_t i = 0; i < nseq; i++) {
      int nc = compute_nchunks(sqarr[i]->n, chunk_size, overlap);
      int64_t step = chunk_size - overlap;
      if (step < 1) step = 1;
      for (int c = 0; c < nc; c++) {
        int64_t seq_start = (int64_t)c * step;
        int64_t clen = chunk_size;
        if (seq_start + clen > sqarr[i]->n) clen = sqarr[i]->n - seq_start;
        int64_t pb = packed_bytes_for(clen);
        int64_t mb = mask_bytes_for(clen);
        int64_t pb_padded = (pb + 3) & ~((int64_t)3);
        int64_t mb_padded = (mb + 3) & ~((int64_t)3);
        (void)pb; (void)pb_padded;

        if (mask_buf_cap < mb_padded) {
          free(mask_buf);
          mask_buf = (uint8_t *)malloc((size_t)mb_padded);
          if (!mask_buf) ESL_XFAIL(eslEMEM, errbuf, "allocation failed for mask buffer");
          mask_buf_cap = mb_padded;
        }
        memset(mask_buf, 0, (size_t)mb_padded);
        {
          uint8_t *scratch_pack = (uint8_t *)malloc((size_t)packed_bytes_for(clen));
          if (!scratch_pack) { free(mask_buf); ESL_XFAIL(eslEMEM, errbuf, "allocation failed for scratch pack"); }
          pack_chunk(sqarr[i]->dsq, seq_start, clen, scratch_pack, mask_buf,
                     packed_of_dsq, mask_of_dsq);
          free(scratch_pack);
        }
        if (fwrite(mask_buf, 1, (size_t)mb_padded, fp) != (size_t)mb_padded) {
          free(mask_buf);
          ESL_XFAIL(eslEWRITE, errbuf, "failed to write mask chunk");
        }
      }
    }
    free(mask_buf);
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
  if (sq) esl_sq_Destroy(sq);
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
    ESL_XFAIL(eslEFORMAT, errbuf,
              "nucdb version %u is unsupported; rebuild with hmmnucdb",
              ndb->hdr.version);
  if (ndb->hdr.encoding != P7_NUCDB_ENCODING_PACKED2BIT)
    ESL_XFAIL(eslEFORMAT, errbuf,
              "nucdb encoding %u is unsupported; rebuild with hmmnucdb",
              (unsigned)ndb->hdr.encoding);

  uint8_t *base = (uint8_t *)ndb->mmap_base;
  ndb->chunk_idx  = (P7_NUCDB_CHUNK_IDX *)(base + ndb->hdr.chunk_idx_offset);
  ndb->seq_idx    = (P7_NUCDB_SEQ_IDX *)  (base + ndb->hdr.seq_idx_offset);
  ndb->name_blob  = (char *)              (base + ndb->hdr.seq_idx_offset +
                                           sizeof(P7_NUCDB_SEQ_IDX) * ndb->hdr.nseq);
  ndb->chunk_data = base + ndb->hdr.data_offset;
  ndb->mask_data  = base + ndb->hdr.mask_offset;

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


/*****************************************************************
 * Test driver.
 *****************************************************************/
#ifdef p7NUCDB_TESTDRIVE

#include <errno.h>
#include "esl_random.h"
#include "esl_getopts.h"
#include "hmmer.h"

static ESL_OPTIONS utest_options[] = {
  { "-h",     eslARG_NONE, FALSE, NULL, NULL, NULL, NULL, NULL, "show help",  0 },
  { "-s",     eslARG_INT,     "42", NULL, "n>=0", NULL, NULL, NULL, "seed",   0 },
  {  0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};
static char utest_usage[]  = "";
static char utest_banner[] = "unit tests for p7_nucdb (v2 packed+mask)";

/* Decode one residue from a .nucdb chunk. Returns 1 if masked (N),
 * otherwise stores canonical code 0..3 in *ret_code. */
static int
decode_chunk_residue(const P7_NUCDB *ndb, const P7_NUCDB_CHUNK_IDX *ci,
                     int64_t pos_in_chunk, int *ret_code)
{
  const uint8_t *packed = ndb->chunk_data + ci->data_offset;
  const uint8_t *mask   = ndb->mask_data  + ci->mask_offset;
  int code = p7_nucdb_unpack2bit(packed, pos_in_chunk);
  int m    = p7_nucdb_mask_bit(mask, pos_in_chunk);
  if (ret_code) *ret_code = code;
  return m;
}

static void
utest_roundtrip(ESL_RANDOMNESS *rng, const char *tmpbase)
{
  ESL_ALPHABET *abc = esl_alphabet_Create(eslDNA);
  /* IUPAC ambiguity codes only — no '*' or '~' (those are dsq-internal
   * sentinels, not valid FASTA residues). Lowercase is valid FASTA. */
  const char   *seqchars = "ACGTRYMKSWHBVDNacgtn";
  const int     nchar = (int)strlen(seqchars);
  char          fpath[1024];
  char          errbuf[eslERRBUFSIZE];
  int           status;

  /* Build a small FASTA input on disk: two sequences, one all-canonical,
   * one with every non-canonical code. */
  snprintf(fpath, sizeof(fpath), "%s.fa", tmpbase);
  FILE *ffa = fopen(fpath, "w");
  if (!ffa) esl_fatal("utest: cannot open %s", fpath);

  fprintf(ffa, ">seqA clean\n");
  int seqA_len = 200;
  char *seqA = malloc(seqA_len + 1);
  if (!seqA) esl_fatal("alloc failed");
  for (int i = 0; i < seqA_len; i++) seqA[i] = "ACGT"[esl_rnd_Roll(rng, 4)];
  seqA[seqA_len] = '\0';
  fprintf(ffa, "%s\n", seqA);

  fprintf(ffa, ">seqB mixed\n");
  int seqB_len = 100;
  char *seqB = malloc(seqB_len + 1);
  if (!seqB) esl_fatal("alloc failed");
  for (int i = 0; i < seqB_len; i++) seqB[i] = seqchars[esl_rnd_Roll(rng, nchar)];
  seqB[seqB_len] = '\0';
  fprintf(ffa, "%s\n", seqB);

  fclose(ffa);

  /* Open as ESL_SQFILE for the writer */
  ESL_SQFILE *sqfp = NULL;
  status = esl_sqfile_OpenDigital(abc, fpath, eslSQFILE_FASTA, NULL, &sqfp);
  if (status != eslOK) esl_fatal("utest: open %s failed: %d", fpath, status);

  /* Write with small chunk + overlap to exercise multi-chunk. */
  char baseout[1024];
  snprintf(baseout, sizeof(baseout), "%s", tmpbase);
  status = p7_nucdb_Write(baseout, abc, sqfp, /*chunk_size=*/64, /*overlap=*/8, errbuf);
  if (status != eslOK) esl_fatal("utest: p7_nucdb_Write failed: %s", errbuf);
  esl_sqfile_Close(sqfp);

  /* Reopen with p7_nucdb_Open */
  P7_NUCDB *ndb = NULL;
  status = p7_nucdb_Open(baseout, &ndb, errbuf);
  if (status != eslOK) esl_fatal("utest: p7_nucdb_Open failed: %s", errbuf);

  if (ndb->hdr.version != P7_NUCDB_VERSION)
    esl_fatal("utest: version = %u, expected %u", ndb->hdr.version, P7_NUCDB_VERSION);
  if (ndb->hdr.nseq != 2)
    esl_fatal("utest: nseq = %" PRIu64 ", expected 2", ndb->hdr.nseq);
  if (ndb->hdr.encoding != P7_NUCDB_ENCODING_PACKED2BIT)
    esl_fatal("utest: encoding = %u, expected packed2bit", (unsigned)ndb->hdr.encoding);

  /* Reconstruct each sequence via the chunk index + packed/mask decode, and
   * compare to the original. Non-canonical characters (and 'n','*','~','-')
   * must come back as mask=1. */
  const int seq_lens[2] = { seqA_len, seqB_len };
  const char *orig_seqs[2] = { seqA, seqB };

  for (int si = 0; si < 2; si++) {
    const P7_NUCDB_SEQ_IDX *sidx = &ndb->seq_idx[si];
    if ((int)sidx->length != seq_lens[si])
      esl_fatal("utest: seq %d length = %" PRId64 ", expected %d",
                si, sidx->length, seq_lens[si]);

    /* Track whether every position in the original is covered by at
     * least one chunk. */
    char *seen = calloc(seq_lens[si], 1);
    if (!seen) esl_fatal("alloc failed");

    for (int c = 0; c < sidx->chunk_count; c++) {
      const P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[sidx->chunk_start + c];
      if (ci->seq_id != si)
        esl_fatal("utest: chunk %d seq_id = %d, expected %d", c, ci->seq_id, si);
      for (int64_t p = 0; p < ci->length; p++) {
        int64_t abs_pos = ci->seq_offset + p;  /* 0-indexed into original */
        if (abs_pos < 0 || abs_pos >= seq_lens[si])
          esl_fatal("utest: chunk position %" PRId64 " out of range", abs_pos);
        int code = 0;
        int is_masked = decode_chunk_residue(ndb, ci, p, &code);
        char orig = orig_seqs[si][abs_pos];
        int expect_code = 0;
        int expect_mask = 1;
        switch (orig) {
          case 'A': case 'a': expect_code = 0; expect_mask = 0; break;
          case 'C': case 'c': expect_code = 1; expect_mask = 0; break;
          case 'G': case 'g': expect_code = 2; expect_mask = 0; break;
          case 'T': case 't': expect_code = 3; expect_mask = 0; break;
          default:            expect_code = 0; expect_mask = 1; break;
        }
        if (is_masked != expect_mask)
          esl_fatal("utest: seq %d pos %" PRId64 " orig='%c' mask=%d expect=%d",
                    si, abs_pos, orig, is_masked, expect_mask);
        if (!is_masked && code != expect_code)
          esl_fatal("utest: seq %d pos %" PRId64 " orig='%c' code=%d expect=%d",
                    si, abs_pos, orig, code, expect_code);
        seen[abs_pos] = 1;
      }
    }
    for (int p = 0; p < seq_lens[si]; p++)
      if (!seen[p]) esl_fatal("utest: seq %d pos %d not covered by any chunk", si, p);
    free(seen);
  }

  /* Verify overlap geometry: adjacent chunks within a sequence overlap by
   * ndb->hdr.overlap residues (except where the sequence is shorter than
   * a chunk). */
  for (uint64_t si = 0; si < ndb->hdr.nseq; si++) {
    const P7_NUCDB_SEQ_IDX *sidx = &ndb->seq_idx[si];
    for (int c = 1; c < sidx->chunk_count; c++) {
      const P7_NUCDB_CHUNK_IDX *prev = &ndb->chunk_idx[sidx->chunk_start + c - 1];
      const P7_NUCDB_CHUNK_IDX *curr = &ndb->chunk_idx[sidx->chunk_start + c];
      int64_t prev_end = prev->seq_offset + prev->length;
      int64_t overlap = prev_end - curr->seq_offset;
      if (overlap < (int64_t)ndb->hdr.overlap && curr->seq_offset + curr->length < (int64_t)sidx->length)
        esl_fatal("utest: seq %" PRIu64 " chunks %d-%d overlap=%" PRId64 " < expected %" PRIu64,
                  si, c-1, c, overlap, ndb->hdr.overlap);
    }
  }

  p7_nucdb_Close(ndb);

  /* Verify version rejection: mutate the version byte on disk and reopen. */
  char fullpath[1100];
  snprintf(fullpath, sizeof(fullpath), "%s.nucdb", baseout);
  int fd = open(fullpath, O_RDWR);
  if (fd < 0) esl_fatal("utest: cannot reopen %s rw: %s", fullpath, strerror(errno));
  uint32_t bad = 1;
  if (pwrite(fd, &bad, sizeof(bad), offsetof(P7_NUCDB_HEADER, version)) != (ssize_t)sizeof(bad))
    esl_fatal("utest: pwrite version failed");
  close(fd);
  status = p7_nucdb_Open(baseout, &ndb, errbuf);
  if (status == eslOK) esl_fatal("utest: v1 file should have been rejected");

  /* Cleanup temp files. */
  unlink(fullpath);
  unlink(fpath);

  free(seqA);
  free(seqB);
  esl_alphabet_Destroy(abc);
  printf("# p7_nucdb: round-trip + version-reject OK\n");
}

int
main(int argc, char **argv)
{
  ESL_GETOPTS *go = p7_CreateDefaultApp(utest_options, 0, argc, argv, utest_banner, utest_usage);
  ESL_RANDOMNESS *rng = esl_randomness_CreateFast(esl_opt_GetInteger(go, "-s"));

  utest_roundtrip(rng, "/tmp/p7_nucdb_utest");

  esl_randomness_Destroy(rng);
  esl_getopts_Destroy(go);
  return 0;
}

#endif /*p7NUCDB_TESTDRIVE*/
