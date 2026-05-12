/* nhmmer GPU pipeline: nucleotide-database sequence slice helpers (v2).
 *
 * The .nucdb v2 on-disk layout stores forward-strand residues as 2-bit
 * packed codes (A=0,C=1,G=2,T=3) plus a 1-bit mask bitmap indicating
 * non-ACGT positions. The reverse-complement strand is not on disk; the
 * CPU domain workers and the GPU kernels reverse the index and
 * complement-XOR the 2-bit code on the fly.
 *
 * This file provides host-side helpers: a metadata-only shell ESL_SQ used
 * by the pipeline top level (so the genome is NEVER materialised as
 * byte-per-residue dsq), and a per-survivor-window slice fill that
 * *does* emit Easel dsq bytes (codes 0..4 + N=15) because the CPU
 * post-filter code (domain definition, null2, hit scoring) wants a
 * standard ESL_DSQ *. Slice sizes are hundreds to a few thousand
 * residues, so the total in-flight dsq residency is MB, not GB.
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
  /* Leave dsq unallocated — the shell is metadata-only. Workers later
   * materialise per-survivor-window dsq slices via nhmmer_gpu_nucdb_fill_slice.
   */
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

/* Fill `dsq[1..length]` with the unpacked residue bytes for the requested
 * window. On the forward strand the window is
 *   forward_positions [start .. start+length-1]   (1-indexed).
 * On the reverse-complement strand the nhmmer pipeline reports hit
 * coordinates in the RC frame with start > end, so here `start` is a
 * position in the RC sequence (1-indexed): it maps back to the forward
 * range [L-start-length+2 .. L-start+1] read out in reverse and complemented.
 */
int
nhmmer_gpu_nucdb_fill_slice(const P7_NUCDB *ndb, const ESL_ALPHABET *abc,
                            int64_t si, int complementarity,
                            uint64_t start, int length, const char *seqname,
                            ESL_SQ **ret_sq, ESL_DSQ **ret_dsq, int64_t *ret_alloc)
{
  const P7_NUCDB_SEQ_IDX *sidx;
  int64_t need;
  ESL_SQ *sq = *ret_sq;
  ESL_DSQ *dsq = *ret_dsq;
  int64_t alloc = *ret_alloc;
  int abc_N;
  int rc;

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

  abc_N = esl_abc_XGetUnknown(abc);     /* dsq code for N: 15 for DNA */
  rc    = (complementarity == p7_NOCOMPLEMENT) ? 0 : 1;

  dsq[0] = eslDSQ_SENTINEL;
  dsq[length + 1] = eslDSQ_SENTINEL;

  /* Translate to forward-coordinate range [fwd_begin .. fwd_begin+length-1],
   * 0-indexed into the forward sequence. */
  int64_t fwd_begin;
  int64_t L = sidx->length;
  if (!rc) {
    fwd_begin = (int64_t)start - 1;
  } else {
    /* RC window [start .. start+length-1] corresponds to forward
     * positions [L-start-length+2 .. L-start+1] (1-indexed). 0-indexed
     * start is L - start - length + 1. */
    fwd_begin = L - (int64_t)start - (int64_t)length + 1;
  }

  /* Zero-init; any position not covered by a chunk stays as N. */
  for (int i = 0; i < length; i++) dsq[1 + i] = (ESL_DSQ)abc_N;

  int chunk_start = sidx->chunk_start;
  int chunk_count = sidx->chunk_count;
  int64_t req_end = fwd_begin + (int64_t)length;

  for (int c = 0; c < chunk_count; c++) {
    const P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[chunk_start + c];
    int64_t chunk_beg = ci->seq_offset;
    int64_t chunk_end = chunk_beg + (int64_t)ci->length;
    int64_t ov_beg = fwd_begin > chunk_beg ? fwd_begin : chunk_beg;
    int64_t ov_end = req_end   < chunk_end ? req_end   : chunk_end;
    if (ov_beg >= ov_end) continue;

    const uint8_t *packed = ndb->chunk_data + ci->data_offset;
    const uint8_t *mask   = ndb->mask_data  + ci->mask_offset;

    /* For each covered forward position [ov_beg .. ov_end-1]: */
    for (int64_t f = ov_beg; f < ov_end; f++) {
      int64_t pos_in_chunk = f - chunk_beg;
      int code = p7_nucdb_unpack2bit(packed, pos_in_chunk);
      int m    = p7_nucdb_mask_bit(mask, pos_in_chunk);
      int val;
      if (m) {
        val = abc_N;
      } else if (rc) {
        /* Complement: A<->T (0<->3), C<->G (1<->2). */
        val = code ^ 0x3;
      } else {
        val = code;
      }

      int64_t win_index;
      if (!rc) {
        win_index = f - fwd_begin;
      } else {
        /* Reversed emission: forward position f maps to RC index
         * (length - 1 - (f - fwd_begin)). */
        win_index = (int64_t)length - 1 - (f - fwd_begin);
      }
      dsq[1 + win_index] = (ESL_DSQ)val;
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
