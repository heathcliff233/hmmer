/* nhmmer GPU pipeline: nucleotide-database sequence reconstruction helpers.
 *
 * Builds full or windowed ESL_SQ objects from .nucdb chunk storage. Pure
 * host-side code; no CUDA calls. The whole body is wrapped in HMMER_CUDA
 * because it is only linked into the GPU pipeline; CPU-only builds compile
 * this translation unit to an empty object.
 */
#include <p7_config.h>

#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_sq.h"

#include "hmmer.h"
#include "p7_nucdb.h"
#include "nhmmer_internal.h"

#ifdef HMMER_CUDA
#include "cuda/p7_cuda.h"
#include "cuda/nhmmer_cuda_internal.h"

int
nhmmer_gpu_nucdb_reconstruct_sq(const P7_NUCDB *ndb, const ESL_ALPHABET *abc,
                                int64_t si, int complementarity, ESL_SQ **ret_sq)
{
  const P7_NUCDB_SEQ_IDX *sidx;
  const char *seqname;
  ESL_SQ *sq;
  int c;

  if (!ndb || !abc || !ret_sq) return eslEINVAL;
  if (si < 0 || si >= (int64_t)ndb->hdr.nseq) return eslEINVAL;

  sidx = &ndb->seq_idx[si];
  seqname = ndb->name_blob + sidx->name_offset;

  sq = esl_sq_CreateDigital(abc);
  if (!sq) return eslEMEM;
  esl_sq_SetName(sq, seqname);
  esl_sq_GrowTo(sq, sidx->length);

  if (complementarity == p7_NOCOMPLEMENT) {
    int64_t step = (int64_t)ndb->hdr.chunk_size - (int64_t)ndb->hdr.overlap;
    if (step < 1) step = 1;
    for (c = 0; c < sidx->fwd_chunk_count; c++) {
      P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[sidx->fwd_chunk_start + c];
      uint8_t *chunk_dsq = ndb->chunk_data + ci->data_offset;
      int64_t copy_start = 0;
      int64_t copy_len   = ci->length;
      if (c > 0) {
        int64_t already_copied = ci->seq_offset;
        int64_t prev_end = ndb->chunk_idx[sidx->fwd_chunk_start + c - 1].seq_offset +
                           ndb->chunk_idx[sidx->fwd_chunk_start + c - 1].length;
        if (prev_end > already_copied) {
          copy_start = prev_end - already_copied;
          copy_len  -= copy_start;
        }
      }
      if (copy_len > 0)
        memcpy(sq->dsq + 1 + ci->seq_offset + copy_start,
               chunk_dsq + 1 + copy_start, copy_len);
    }
    sq->start = 1;
    sq->end   = sq->n = sidx->length;
  } else {
    for (c = 0; c < sidx->rc_chunk_count; c++) {
      P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[sidx->rc_chunk_start + c];
      uint8_t *chunk_dsq = ndb->chunk_data + ci->data_offset;
      int64_t copy_start = 0;
      int64_t copy_len   = ci->length;
      if (c > 0) {
        int64_t prev_end = ndb->chunk_idx[sidx->rc_chunk_start + c - 1].seq_offset +
                           ndb->chunk_idx[sidx->rc_chunk_start + c - 1].length;
        if (prev_end > ci->seq_offset) {
          copy_start = prev_end - ci->seq_offset;
          copy_len  -= copy_start;
        }
      }
      if (copy_len > 0)
        memcpy(sq->dsq + 1 + ci->seq_offset + copy_start,
               chunk_dsq + 1 + copy_start, copy_len);
    }
    sq->start = sq->n = sidx->length;
    sq->end   = 1;
  }
  sq->L = sq->n;
  sq->dsq[0] = eslDSQ_SENTINEL;
  sq->dsq[sq->n + 1] = eslDSQ_SENTINEL;
  *ret_sq = sq;
  return eslOK;
}

int
nhmmer_gpu_nucdb_get_cached_sq(const P7_NUCDB *ndb, const ESL_ALPHABET *abc,
                               int64_t si, int complementarity,
                               ESL_SQ **ret_sq, int *ret_built)
{
  ESL_SQ **slot = NULL;
  int status;

  if (!ndb || !abc || !ret_sq) return eslEINVAL;
  if (ret_built) *ret_built = FALSE;
  if (si < 0 || si >= (int64_t)ndb->hdr.nseq) return eslEINVAL;

  if (complementarity == p7_NOCOMPLEMENT) slot = ndb->sq_cache_top;
  else                                    slot = ndb->sq_cache_rc;
  if (!slot) return eslEINVAL;

  if (slot[si] == NULL) {
    status = nhmmer_gpu_nucdb_reconstruct_sq(ndb, abc, si, complementarity, &slot[si]);
    if (status != eslOK) return status;
    if (ret_built) *ret_built = TRUE;
  }
  *ret_sq = slot[si];
  return eslOK;
}

int
nhmmer_gpu_nucdb_create_seq_shell(const P7_NUCDB *ndb, const ESL_ALPHABET *abc,
                                  int64_t si, ESL_SQ **ret_sq)
{
  const P7_NUCDB_SEQ_IDX *sidx;
  const char *seqname;
  ESL_SQ *sq;
  int status;

  if (!ndb || !abc || !ret_sq) return eslEINVAL;
  if (si < 0 || si >= (int64_t)ndb->hdr.nseq) return eslEINVAL;

  sidx = &ndb->seq_idx[si];
  seqname = ndb->name_blob + sidx->name_offset;
  sq = esl_sq_CreateDigital(abc);
  if (!sq) return eslEMEM;
  free(sq->dsq);
  sq->dsq = NULL;
  sq->salloc = 0;
  status = esl_sq_SetName(sq, seqname);
  if (status != eslOK) { esl_sq_Destroy(sq); return status; }
  sq->start = 1;
  sq->end   = sidx->length;
  sq->n     = sidx->length;
  sq->L     = sidx->length;
  sq->C     = 0;
  sq->W     = sidx->length;
  *ret_sq = sq;
  return eslOK;
}

int
nhmmer_gpu_nucdb_fill_slice(const P7_NUCDB *ndb, const ESL_ALPHABET *abc,
                            int64_t si, int complementarity,
                            uint64_t start, int length, const char *seqname,
                            ESL_SQ **ret_sq, ESL_DSQ **ret_dsq, int64_t *ret_alloc)
{
  const P7_NUCDB_SEQ_IDX *sidx;
  int chunk_start, chunk_count;
  int64_t need;
  int64_t seq_pos0;
  ESL_SQ *sq = *ret_sq;
  ESL_DSQ *dsq = *ret_dsq;
  int64_t alloc = *ret_alloc;

  if (!ndb || !abc || si < 0 || si >= (int64_t)ndb->hdr.nseq || start < 1 || length < 0)
    return eslEINVAL;

  sidx = &ndb->seq_idx[si];
  if (start + (uint64_t)length - 1 > (uint64_t)sidx->length) return eslEINVAL;

  if (sq == NULL) {
    sq = esl_sq_CreateDigital(abc);
    if (!sq) return eslEMEM;
    free(sq->dsq);
    sq->dsq = NULL;
    sq->salloc = 0;
    *ret_sq = sq;
  }
  if (sq->name == NULL || strcmp(sq->name, seqname) != 0) {
    int status = esl_sq_SetName(sq, seqname);
    if (status != eslOK) return status;
  }

  need = (int64_t)length + 2;
  if (alloc < need) {
    ESL_DSQ *tmp = realloc(dsq, sizeof(ESL_DSQ) * need);
    if (!tmp) return eslEMEM;
    dsq = tmp;
    alloc = need;
    *ret_dsq = dsq;
    *ret_alloc = alloc;
  }

  dsq[0] = eslDSQ_SENTINEL;
  dsq[length + 1] = eslDSQ_SENTINEL;
  memset(dsq + 1, eslDSQ_SENTINEL, sizeof(ESL_DSQ) * length);

  chunk_start = (complementarity == p7_NOCOMPLEMENT) ? sidx->fwd_chunk_start : sidx->rc_chunk_start;
  chunk_count = (complementarity == p7_NOCOMPLEMENT) ? sidx->fwd_chunk_count : sidx->rc_chunk_count;
  seq_pos0 = (int64_t)start - 1;

  for (int c = 0; c < chunk_count; c++) {
    P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[chunk_start + c];
    int64_t chunk_beg = ci->seq_offset;
    int64_t chunk_end = ci->seq_offset + ci->length;
    int64_t req_beg = seq_pos0;
    int64_t req_end = seq_pos0 + length;
    int64_t ov_beg = ESL_MAX(chunk_beg, req_beg);
    int64_t ov_end = ESL_MIN(chunk_end, req_end);
    if (ov_beg < ov_end) {
      int64_t dst = ov_beg - req_beg;
      int64_t src = ov_beg - chunk_beg;
      int64_t n   = ov_end - ov_beg;
      memcpy(dsq + 1 + dst, ndb->chunk_data + ci->data_offset + 1 + src, n);
    }
  }

  sq->dsq = dsq;
  sq->n = length;
  sq->L = sidx->length;
  sq->C = 0;
  sq->W = length;
  if (complementarity == p7_NOCOMPLEMENT) {
    sq->start = 1;
    sq->end = sidx->length;
  } else {
    sq->start = sidx->length;
    sq->end = 1;
  }
  sq->abc = abc;
  return eslOK;
}

#endif /* HMMER_CUDA */
