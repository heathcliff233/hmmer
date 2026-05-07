#include <p7_config.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HMMER_THREADS
#include <pthread.h>
#endif

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_gumbel.h"
#include "esl_sq.h"
#include "esl_sqio.h"
#include "esl_vectorops.h"

#include "hmmer.h"
#include "nhmmer_internal.h"

#ifdef HMMER_CUDA
#include "cuda/p7_cuda.h"
#endif

#define NHMMER_GPU_CHUNK_SIZE  65536
#define NHMMER_GPU_BLOCK_SIZE  (512 * 1024 * 1024)

#ifdef HMMER_CUDA

typedef struct {
  P7_OPROFILE      *om;
  P7_BG            *bg;
  P7_PIPELINE      *pli;
  P7_TOPHITS       *th;
  P7_SCOREDATA     *scoredata;
  P7_PIPELINE_LONGTARGET_OBJS *pli_tmp;
  P7_HMM_WINDOWLIST vit_windowlist;

  const ESL_SQ     *sq;
  P7_HMM_WINDOW   *windows;
  int               nwindows;
  int64_t           seq_id;
  int               complementarity;
  int               status;
} NHMMER_GPU_WORKER;

static int
nhmmer_gpu_worker_init(NHMMER_GPU_WORKER *w, const NHMMER_GPU_INFO *info)
{
  int status = eslOK;
  P7_OPROFILE *om = info->om;
  P7_BG       *bg = info->bg;

  w->om        = p7_oprofile_Copy(om);
  w->bg        = p7_bg_Clone(bg);
  w->th        = p7_tophits_Create();
  w->scoredata = p7_hmm_ScoreDataClone(info->scoredata, om->abc->Kp);
  w->pli       = p7_pipeline_Create(info->go, om->M, 100, TRUE, p7_SEARCH_SEQS);

  if (!w->om || !w->bg || !w->th || !w->scoredata || !w->pli)
    { status = eslEMEM; goto ERROR; }

  w->pli->F1 = info->pli->F1;
  status = p7_pli_NewModel(w->pli, w->om, w->bg);
  if (status != eslOK) goto ERROR;

  w->pli->strands = info->pli->strands;

  ESL_ALLOC(w->pli_tmp, sizeof(P7_PIPELINE_LONGTARGET_OBJS));
  w->pli_tmp->tmpseq = NULL;
  w->pli_tmp->bg     = p7_bg_Clone(bg);
  w->pli_tmp->om     = p7_oprofile_Create(om->M, om->abc);
  ESL_ALLOC(w->pli_tmp->scores, sizeof(float) * om->abc->Kp * 4);
  ESL_ALLOC(w->pli_tmp->fwd_emissions_arr, sizeof(float) * om->abc->Kp * (om->M + 1));
  p7_oprofile_GetFwdEmissionArray(om, bg, w->pli_tmp->fwd_emissions_arr);

  p7_omx_GrowTo(w->pli->oxf, om->M, 0, om->max_length);
  p7_hmmwindow_init(&w->vit_windowlist);
  w->pli_tmp->tmpseq = esl_sq_CreateDigital(om->abc);
  free(w->pli_tmp->tmpseq->dsq);
  w->pli_tmp->tmpseq->dsq = NULL;

  w->sq              = NULL;
  w->windows         = NULL;
  w->nwindows        = 0;
  w->seq_id          = 0;
  w->complementarity = 0;
  w->status          = eslOK;
  return eslOK;

ERROR:
  return status;
}

static void
nhmmer_gpu_worker_destroy(NHMMER_GPU_WORKER *w)
{
  if (w->pli_tmp) {
    if (w->pli_tmp->tmpseq) { w->pli_tmp->tmpseq->dsq = NULL; esl_sq_Destroy(w->pli_tmp->tmpseq); }
    if (w->pli_tmp->bg)     p7_bg_Destroy(w->pli_tmp->bg);
    if (w->pli_tmp->om)     p7_oprofile_Destroy(w->pli_tmp->om);
    if (w->pli_tmp->scores) free(w->pli_tmp->scores);
    if (w->pli_tmp->fwd_emissions_arr) free(w->pli_tmp->fwd_emissions_arr);
    free(w->pli_tmp);
  }
  if (w->vit_windowlist.windows) free(w->vit_windowlist.windows);
  if (w->scoredata) p7_hmm_ScoreDataDestroy(w->scoredata);
  if (w->th)        p7_tophits_Destroy(w->th);
  if (w->pli)       p7_pipeline_Destroy(w->pli);
  if (w->om)        p7_oprofile_Destroy(w->om);
  if (w->bg)        p7_bg_Destroy(w->bg);
}

static void
nhmmer_gpu_worker_process(NHMMER_GPU_WORKER *w)
{
  int          i;
  float        nullsc, usc, P;
  ESL_DSQ     *subseq;
  P7_HMM_WINDOW *window;

  for (i = 0; i < w->nwindows; i++) {
    window = w->windows + i;
    subseq = w->sq->dsq + window->n - 1;

    p7_bg_SetLength(w->bg, window->length);
    p7_bg_NullOne(w->bg, subseq, window->length, &nullsc);

    p7_oprofile_ReconfigMSVLength(w->om, window->length);
    p7_MSVFilter(subseq, window->length, w->om, w->pli->oxf, &usc);
    P = esl_gumbel_surv((usc - nullsc) / eslCONST_LOG2,
                        w->om->evparam[p7_MMU], w->om->evparam[p7_MLAMBDA]);

    if (P > w->pli->F1) continue;
    w->pli->pos_past_msv += window->length;

    w->status = p7_pli_postSSV_LongTarget(w->pli, w->om, w->bg, w->th, w->scoredata,
                                          w->seq_id, window->n, window->length, subseq,
                                          w->sq->start, w->sq->name, w->sq->source,
                                          w->sq->acc, w->sq->desc,
                                          -1, nullsc, usc, w->complementarity,
                                          &w->vit_windowlist, w->pli_tmp);
    if (w->status != eslOK) return;
  }
}

#ifdef HMMER_THREADS
static void *
nhmmer_gpu_thread_func(void *arg)
{
  NHMMER_GPU_WORKER *w = (NHMMER_GPU_WORKER *)arg;
  nhmmer_gpu_worker_process(w);
  return NULL;
}
#endif

static int
nhmmer_gpu_process_strand(NHMMER_GPU_INFO *info, const ESL_SQ *sq, int complementarity,
                          int64_t seq_id, uint8_t sc_thresh, int chunk_size, int overlap,
                          char *errbuf, int errbuf_size)
{
  int status = eslOK;
  P7_OPROFILE *om = info->om;
  int          i;
  int          L = sq->n;

  P7_CUDA_LT_WINDOW *gpu_windows = NULL;
  int gpu_nwindows = 0;

  P7_HMM_WINDOWLIST  msv_windowlist;
  msv_windowlist.windows = NULL;

  status = p7_cuda_SSVLongtarget(info->cuda_engine, info->cuda_msv,
                                 sq->dsq, L,
                                 info->scoredata->ssv_scores, om->abc->Kp,
                                 sc_thresh, om->scale_b,
                                 chunk_size, overlap,
                                 &gpu_windows, &gpu_nwindows,
                                 errbuf, errbuf_size);
  if (status != eslOK) return status;

  if (gpu_nwindows == 0) { free(gpu_windows); return eslOK; }

  p7_hmmwindow_init(&msv_windowlist);
  for (i = 0; i < gpu_nwindows; i++) {
    p7_hmmwindow_new(&msv_windowlist,
                     0,
                     gpu_windows[i].target_start,
                     0,
                     (uint16_t)gpu_windows[i].model_end,
                     gpu_windows[i].model_end - gpu_windows[i].model_start + 1,
                     gpu_windows[i].score,
                     p7_NOCOMPLEMENT,
                     L);
  }
  free(gpu_windows);
  gpu_windows = NULL;

  if (info->scoredata->prefix_lengths == NULL)
    p7_hmm_ScoreDataComputeRest(om, info->scoredata);

  p7_pli_ExtendAndMergeWindows(om, info->scoredata, &msv_windowlist, 0);

  if (msv_windowlist.count == 0) {
    free(msv_windowlist.windows);
    return eslOK;
  }

  int ncpus = info->ncpus;
  if (ncpus < 1) ncpus = 1;
  if (ncpus > msv_windowlist.count) ncpus = msv_windowlist.count;

#ifdef HMMER_THREADS
  if (ncpus > 1) {
    NHMMER_GPU_WORKER *workers = NULL;
    pthread_t         *threads = NULL;
    int                nworkers = ncpus;
    int                windows_per_thread = msv_windowlist.count / nworkers;
    int                remainder = msv_windowlist.count % nworkers;
    int                offset = 0;

    ESL_ALLOC(workers, sizeof(NHMMER_GPU_WORKER) * nworkers);
    ESL_ALLOC(threads, sizeof(pthread_t) * nworkers);
    memset(workers, 0, sizeof(NHMMER_GPU_WORKER) * nworkers);

    for (i = 0; i < nworkers; i++) {
      status = nhmmer_gpu_worker_init(&workers[i], info);
      if (status != eslOK) { nworkers = i; goto THREAD_ERROR; }

      workers[i].sq              = sq;
      workers[i].seq_id          = seq_id;
      workers[i].complementarity = complementarity;
      workers[i].nwindows        = windows_per_thread + (i < remainder ? 1 : 0);
      workers[i].windows         = msv_windowlist.windows + offset;
      offset += workers[i].nwindows;
    }

    for (i = 1; i < nworkers; i++)
      pthread_create(&threads[i], NULL, nhmmer_gpu_thread_func, &workers[i]);

    nhmmer_gpu_worker_process(&workers[0]);

    for (i = 1; i < nworkers; i++)
      pthread_join(threads[i], NULL);

    status = workers[0].status;
    for (i = 1; i < nworkers && status == eslOK; i++)
      status = workers[i].status;

    for (i = 1; i < nworkers && status == eslOK; i++) {
      p7_tophits_Merge(info->th, workers[i].th);
      p7_pipeline_Merge(info->pli, workers[i].pli);
    }
    p7_tophits_Merge(info->th, workers[0].th);
    p7_pipeline_Merge(info->pli, workers[0].pli);

  THREAD_ERROR:
    for (i = 0; i < nworkers; i++)
      nhmmer_gpu_worker_destroy(&workers[i]);
    free(workers);
    free(threads);
    free(msv_windowlist.windows);
    return status;
  }
#endif /* HMMER_THREADS */

  /* Single-threaded fallback */
  {
    NHMMER_GPU_WORKER w;
    memset(&w, 0, sizeof(w));
    status = nhmmer_gpu_worker_init(&w, info);
    if (status != eslOK) { free(msv_windowlist.windows); return status; }

    w.sq              = sq;
    w.seq_id          = seq_id;
    w.complementarity = complementarity;
    w.nwindows        = msv_windowlist.count;
    w.windows         = msv_windowlist.windows;

    nhmmer_gpu_worker_process(&w);
    status = w.status;

    if (status == eslOK) {
      p7_tophits_Merge(info->th, w.th);
      p7_pipeline_Merge(info->pli, w.pli);
    }

    nhmmer_gpu_worker_destroy(&w);
    free(msv_windowlist.windows);
    return status;
  }

ERROR:
  free(gpu_windows);
  if (msv_windowlist.windows) free(msv_windowlist.windows);
  return status;
}

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

#else /* !HMMER_CUDA */

int
nhmmer_gpu_serial_loop(NHMMER_GPU_INFO *info, ESL_SQFILE *dbfp,
                       int strands, NHMMER_GPU_IDLEN_CB idlen_cb, void *idlen_data,
                       int *ret_nseqs, int64_t *ret_nres)
{
  ESL_UNUSED(info); ESL_UNUSED(dbfp); ESL_UNUSED(strands);
  ESL_UNUSED(idlen_cb); ESL_UNUSED(idlen_data);
  ESL_UNUSED(ret_nseqs); ESL_UNUSED(ret_nres);
  return eslENORESULT;
}

#endif /* HMMER_CUDA */
