/* nhmmer GPU pipeline: public outer loops.
 *
 * Thin top-level drivers that iterate over the target database (FASTA via
 * ESL_SQFILE, or pre-digitised .nucdb chunks) and dispatch each sequence's
 * strands to the per-strand orchestration in cuda/p7_cuda_nhmmer_strand.c.
 * Also owns the one-time GPU nucdb upload helper. Stage-specific work
 * (SSV / F1 batch filter / Viterbi / Fwd / FB parser / CPU workers) lives
 * in sibling p7_cuda_nhmmer_*.c files.
 *
 * Plain C (no .cu); compiled by the host compiler. It #includes
 * cuda/p7_cuda.h and calls the kernel-dispatch APIs declared there.
 */
#include <p7_config.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

#ifdef HMMER_THREADS
#include <pthread.h>
#endif

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_dsqdata.h"
#include "esl_exponential.h"
#include "esl_gumbel.h"
#include "esl_sq.h"
#include "esl_sqio.h"
#include "esl_vectorops.h"

#include "hmmer.h"
#include "p7_nucdb.h"
#include "nhmmer_internal.h"

#ifdef HMMER_CUDA
#include "cuda/p7_cuda.h"
#include "cuda/nhmmer_cuda_internal.h"


char nhmmer_empty_str[] = "";


int
nhmmer_gpu_serial_loop(NHMMER_GPU_INFO *info, ESL_SQFILE *dbfp,
                       int strands, NHMMER_GPU_IDLEN_CB idlen_cb, void *idlen_data,
                       int *ret_nseqs, int64_t *ret_nres)
{
  int       status   = eslOK;
  int       wstatus  = eslOK;
  ESL_SQ   *dbsq     = NULL;
  ESL_SQ   *dbsq_rc  = NULL;
  int       seq_id   = 0;
  int64_t   nres     = 0;
  char      errbuf[eslERRBUFSIZE];
  int       chunk_size = info->gpu_chunk_size > 0 ? info->gpu_chunk_size : NHMMER_GPU_CHUNK_SIZE;
  int       overlap    = info->om->max_length;

  P7_OPROFILE *om = info->om;
  P7_BG       *bg = info->bg;

  dbsq = esl_sq_CreateDigital(om->abc);
  if (dbsq == NULL) return eslEMEM;
  if (om->abc->complement)
    dbsq_rc = esl_sq_CreateDigital(om->abc);

  float nullsc;
  p7_bg_SetLength(bg, om->max_length);
  p7_oprofile_ReconfigMSVLength(om, om->max_length);
  p7_bg_NullOne(bg, dbsq->dsq, om->max_length, &nullsc);

  p7_cuda_msvprofile_UpdateLength(info->cuda_msv, om, om->max_length, errbuf, sizeof(errbuf));

  float invP = esl_gumbel_invsurv(info->pli->F1, om->evparam[p7_MMU], om->evparam[p7_MLAMBDA]);
  uint8_t sc_thresh = (uint8_t)ceil(((nullsc + (invP * eslCONST_LOG2) + 3.0) * om->scale_b)
                                    + om->base_b + om->tec_b + om->tjb_b);

  wstatus = esl_sqio_ReadWindow(dbfp, 0, NHMMER_GPU_BLOCK_SIZE, dbsq);

  while (wstatus == eslOK) {
    dbsq->idx = seq_id;
    p7_pli_NewSeq(info->pli, dbsq);

    if (strands != p7_STRAND_BOTTOMONLY) {
      nres += dbsq->n - dbsq->C;
      status = nhmmer_gpu_process_strand(info, dbsq, p7_NOCOMPLEMENT,
                                         seq_id, sc_thresh, chunk_size, overlap,
                                         errbuf, sizeof(errbuf));
      if (status != eslOK) {
        fprintf(stderr, "GPU nhmmer forward strand failed: %s\n", errbuf);
        goto ERROR;
      }
    }

    if (strands != p7_STRAND_TOPONLY && om->abc->complement != NULL) {
      esl_sq_Copy(dbsq, dbsq_rc);
      esl_sq_ReverseComplement(dbsq_rc);
      nres += dbsq_rc->n;

      status = nhmmer_gpu_process_strand(info, dbsq_rc, p7_COMPLEMENT,
                                         seq_id, sc_thresh, chunk_size, overlap,
                                         errbuf, sizeof(errbuf));
      if (status != eslOK) {
        fprintf(stderr, "GPU nhmmer revcomp strand failed: %s\n", errbuf);
        goto ERROR;
      }
    }

    p7_pipeline_Reuse(info->pli);

    wstatus = esl_sqio_ReadWindow(dbfp, om->max_length, NHMMER_GPU_BLOCK_SIZE, dbsq);
    if (wstatus == eslEOD) {
      if (idlen_cb) idlen_cb(idlen_data, dbsq->idx, dbsq->L);
      info->pli->nseqs++;
      seq_id++;
      esl_sq_Reuse(dbsq);
      if (dbsq_rc) esl_sq_Reuse(dbsq_rc);
      wstatus = esl_sqio_ReadWindow(dbfp, 0, NHMMER_GPU_BLOCK_SIZE, dbsq);
    }
  }

  *ret_nseqs = seq_id;
  *ret_nres  = nres;

ERROR:
  if (dbsq)    esl_sq_Destroy(dbsq);
  if (dbsq_rc) esl_sq_Destroy(dbsq_rc);
  return (wstatus == eslEOF || wstatus == eslOK) ? eslOK : wstatus;
}

/* Main loop for nucdb format: iterate over sequences, build ESL_SQ from
 * pre-digitized chunks, process each strand. */
int
nhmmer_gpu_nucdb_upload(NHMMER_GPU_INFO *info, P7_NUCDB *ndb,
                        char *errbuf, int errbuf_size)
{
  int64_t nucdb_data_size;

  if (!info || !ndb) return eslEINVAL;
  if (p7_cuda_engine_NucdbDevPtr(info->cuda_engine) != NULL) {
    info->nucdb_resident = TRUE;
    return eslOK;
  }

  nucdb_data_size = (int64_t)(ndb->mmap_size - ndb->hdr.data_offset);
  int status = p7_cuda_engine_UploadNucdb(info->cuda_engine, ndb->chunk_data, nucdb_data_size,
                                          errbuf, errbuf_size);
  if (status == eslOK) info->nucdb_resident = TRUE;
  return status;
}

int
nhmmer_gpu_nucdb_loop(NHMMER_GPU_INFO *info, P7_NUCDB *ndb,
                      int strands, NHMMER_GPU_IDLEN_CB idlen_cb, void *idlen_data,
                      int *ret_nseqs, int64_t *ret_nres)
{
  int       status = eslOK;
  int64_t   nres   = 0;
  char      errbuf[eslERRBUFSIZE];
  P7_OPROFILE *om = info->om;
  P7_BG       *bg = info->bg;
  struct timespec ts0, ts1;

  p7_bg_SetLength(bg, om->max_length);
  p7_oprofile_ReconfigMSVLength(om, om->max_length);
  p7_cuda_msvprofile_UpdateLength(info->cuda_msv, om, om->max_length, errbuf, sizeof(errbuf));

  status = nhmmer_gpu_nucdb_upload(info, ndb, errbuf, sizeof(errbuf));
  if (status != eslOK) {
    fprintf(stderr, "GPU nhmmer: failed to upload nucdb: %s\n", errbuf);
    return status;
  }

  for (int64_t si = 0; si < (int64_t)ndb->hdr.nseq; si++) {
    P7_NUCDB_SEQ_IDX *sidx = &ndb->seq_idx[si];
    ESL_SQ *sq = NULL;
    int chunk_size = info->gpu_chunk_size > 0 ? info->gpu_chunk_size : NHMMER_GPU_CHUNK_SIZE;
    int use_shell = (!info->do_compare &&
                     !info->do_cpu_postmsv &&
                     info->do_gpu_fwd &&
                     p7_cuda_engine_NucdbDevPtr(info->cuda_engine) != NULL &&
                     (int64_t)ndb->hdr.overlap >= (int64_t)om->max_length &&
                     (int64_t)ndb->hdr.chunk_size == (int64_t)chunk_size);

    clock_gettime(CLOCK_MONOTONIC, &ts0);
    if (use_shell) {
      status = nhmmer_gpu_nucdb_create_seq_shell(ndb, om->abc, si, &sq);
    } else {
      int sq_built = FALSE;
      status = nhmmer_gpu_nucdb_get_cached_sq(ndb, om->abc, si, p7_NOCOMPLEMENT, &sq, &sq_built);
      if (sq_built) {
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        info->t_nucdb_reconstruct += nhmmer_gpu_elapsed(&ts0, &ts1);
      }
    }
    if (status != eslOK) return status;

    p7_pli_NewSeq(info->pli, sq);

    /* Forward strand */
    if (strands != p7_STRAND_BOTTOMONLY) {
      nres += sq->n;
      status = nhmmer_gpu_process_nucdb_strand(info, ndb,
                                                sidx->fwd_chunk_start, sidx->fwd_chunk_count,
                                                sq, p7_NOCOMPLEMENT, si,
                                                errbuf, sizeof(errbuf));
      if (status != eslOK) {
        fprintf(stderr, "GPU nhmmer nucdb forward strand failed: %s\n", errbuf);
        goto ERROR;
      }
    }

    /* Reverse complement strand — reconstruct directly from nucdb RC chunks */
    if (strands != p7_STRAND_TOPONLY && sidx->rc_chunk_count > 0) {
      ESL_SQ *sq_rc = NULL;
      if (use_shell) {
        sq_rc = sq;
      } else {
        int sq_rc_built = FALSE;
        clock_gettime(CLOCK_MONOTONIC, &ts0);
        status = nhmmer_gpu_nucdb_get_cached_sq(ndb, om->abc, si, p7_COMPLEMENT, &sq_rc, &sq_rc_built);
        if (status != eslOK) {
          fprintf(stderr, "GPU nhmmer nucdb revcomp cache failed\n");
          goto ERROR;
        }
        if (sq_rc_built) {
          clock_gettime(CLOCK_MONOTONIC, &ts1);
          info->t_nucdb_reconstruct += nhmmer_gpu_elapsed(&ts0, &ts1);
        }
      }
      nres += sq_rc->n;

      status = nhmmer_gpu_process_nucdb_strand(info, ndb,
                                                sidx->rc_chunk_start, sidx->rc_chunk_count,
                                                sq_rc, p7_COMPLEMENT, si,
                                                errbuf, sizeof(errbuf));
      if (status != eslOK) {
        fprintf(stderr, "GPU nhmmer nucdb revcomp strand failed: %s\n", errbuf);
        goto ERROR;
      }
    }

    p7_pipeline_Reuse(info->pli);
    if (idlen_cb) idlen_cb(idlen_data, si, sidx->length);
    info->pli->nseqs++;
    if (use_shell) esl_sq_Destroy(sq);
  }

  *ret_nseqs = (int)ndb->hdr.nseq;
  *ret_nres  = nres;
  return eslOK;

ERROR:
  return status;
}


#endif /* HMMER_CUDA */
