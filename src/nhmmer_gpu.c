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
#include "esl_dsqdata.h"
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
#define NHMMER_GPU_BATCH_MIN   32

#ifdef HMMER_CUDA

static char nhmmer_empty_str[] = "";

static int
nhmmer_gpu_window_batch_init(NHMMER_GPU_WINDOW_BATCH *wb, int alloc)
{
  int status;
  wb->nwindows = 0;
  wb->alloc    = alloc;

  ESL_ALLOC(wb->dsq_ptrs, sizeof(ESL_DSQ *)  * alloc);
  ESL_ALLOC(wb->lengths,  sizeof(int64_t)    * alloc);
  ESL_ALLOC(wb->names,    sizeof(char *)     * alloc);
  ESL_ALLOC(wb->accs,     sizeof(char *)     * alloc);
  ESL_ALLOC(wb->descs,    sizeof(char *)     * alloc);
  ESL_ALLOC(wb->taxids,   sizeof(int32_t)    * alloc);
  ESL_ALLOC(wb->win_idx,  sizeof(int)        * alloc);

  memset(&wb->chu, 0, sizeof(ESL_DSQDATA_CHUNK));
  return eslOK;

ERROR:
  return eslEMEM;
}

static void
nhmmer_gpu_window_batch_free(NHMMER_GPU_WINDOW_BATCH *wb)
{
  if (wb->dsq_ptrs) free(wb->dsq_ptrs);
  if (wb->lengths)  free(wb->lengths);
  if (wb->names)    free(wb->names);
  if (wb->accs)     free(wb->accs);
  if (wb->descs)    free(wb->descs);
  if (wb->taxids)   free(wb->taxids);
  if (wb->win_idx)  free(wb->win_idx);
}

/* Pack merged windows into a synthetic ESL_DSQDATA_CHUNK for GPU batch APIs.
 * Zero-copy: dsq pointers point into the parent sequence. */
static int
nhmmer_gpu_window_batch_pack(NHMMER_GPU_WINDOW_BATCH *wb, const ESL_SQ *sq,
                             P7_HMM_WINDOWLIST *wl)
{
  int i;
  int status;

  if (wl->count > wb->alloc) {
    wb->alloc = wl->count;
    ESL_REALLOC(wb->dsq_ptrs, sizeof(ESL_DSQ *)  * wb->alloc);
    ESL_REALLOC(wb->lengths,  sizeof(int64_t)    * wb->alloc);
    ESL_REALLOC(wb->names,    sizeof(char *)     * wb->alloc);
    ESL_REALLOC(wb->accs,     sizeof(char *)     * wb->alloc);
    ESL_REALLOC(wb->descs,    sizeof(char *)     * wb->alloc);
    ESL_REALLOC(wb->taxids,   sizeof(int32_t)    * wb->alloc);
    ESL_REALLOC(wb->win_idx,  sizeof(int)        * wb->alloc);
  }

  wb->nwindows = wl->count;
  for (i = 0; i < wl->count; i++) {
    wb->dsq_ptrs[i] = sq->dsq + wl->windows[i].n - 1;
    wb->lengths[i]  = wl->windows[i].length;
    wb->names[i]    = nhmmer_empty_str;
    wb->accs[i]     = nhmmer_empty_str;
    wb->descs[i]    = nhmmer_empty_str;
    wb->taxids[i]   = -1;
    wb->win_idx[i]  = i;
  }

  wb->chu.i0       = 0;
  wb->chu.N        = wl->count;
  wb->chu.dsq      = wb->dsq_ptrs;
  wb->chu.name     = wb->names;
  wb->chu.acc      = wb->accs;
  wb->chu.desc     = wb->descs;
  wb->chu.taxid    = wb->taxids;
  wb->chu.L        = wb->lengths;
  wb->chu.smem     = NULL;
  wb->chu.psq      = NULL;
  wb->chu.pn       = 0;
  wb->chu.metadata = NULL;
  wb->chu.mdalloc  = 0;
  wb->chu.nxt      = NULL;

  return eslOK;

ERROR:
  return eslEMEM;
}

/* Ensure persistent scratch arrays in info are large enough for N elements. */
static int
nhmmer_gpu_ensure_scratch(NHMMER_GPU_INFO *info, int N)
{
  int status;
  if (N <= info->h_filter_alloc) return eslOK;

  ESL_REALLOC(info->h_ssv_scores,  sizeof(float) * N);
  ESL_REALLOC(info->h_ssv_status,  sizeof(int)   * N);
  ESL_REALLOC(info->h_null_scores, sizeof(float) * N);
  ESL_REALLOC(info->h_bias_scores, sizeof(float) * N);
  info->h_filter_alloc = N;
  return eslOK;

ERROR:
  return eslEMEM;
}

/* GPU batch SSV + null + bias + F1 gating.
 * Filters the merged window list in-place: survivors are compacted to the front.
 * Returns the number of survivors in *ret_nsurv.
 * Bias scores are retained in info->h_bias_scores for downstream reuse. */
static int
nhmmer_gpu_batch_filter(NHMMER_GPU_INFO *info, NHMMER_GPU_WINDOW_BATCH *wb,
                        P7_HMM_WINDOWLIST *wl, int *ret_nsurv,
                        char *errbuf, int errbuf_size)
{
  int       status;
  int       N = wb->nwindows;
  int       nsurv = 0;
  int       i;
  double    P;
  float     seq_score;
  P7_OPROFILE *om = info->om;
  P7_PIPELINE *pli = info->pli;

  status = nhmmer_gpu_ensure_scratch(info, N);
  if (status != eslOK) return status;

  float *ssv_scores  = info->h_ssv_scores;
  int   *ssv_status  = info->h_ssv_status;
  float *null_scores = info->h_null_scores;
  float *bias_scores = info->h_bias_scores;

  status = p7_cuda_MSVFilterDsqdataChunk(info->cuda_engine, info->cuda_msv,
                                         &wb->chu, ssv_scores, ssv_status,
                                         errbuf, errbuf_size);
  if (status != eslOK) return status;

  status = p7_cuda_NullScoreDsqdataChunk(info->cuda_engine, info->bg,
                                         &wb->chu, null_scores,
                                         errbuf, errbuf_size);
  if (status != eslOK) return status;

  if (pli->do_biasfilter) {
    status = p7_cuda_BiasFilterDsqdataChunk(info->cuda_engine, info->bg,
                                            &wb->chu, bias_scores,
                                            errbuf, errbuf_size);
    if (status != eslOK) return status;
  }

  for (i = 0; i < N; i++) {
    if (ssv_status[i] == eslERANGE) {
      if (nsurv != i)
        wl->windows[nsurv] = wl->windows[i];
      nsurv++;
      continue;
    }
    if (ssv_status[i] != eslOK) continue;

    int    window_len = (int)wb->lengths[i];
    float  nullsc     = null_scores[i];
    float  usc        = ssv_scores[i];
    int    F1_L       = ESL_MIN(window_len, pli->B1);
    float  filtersc;

    if (pli->do_biasfilter) {
      float bias_filtersc = bias_scores[i] - nullsc;
      filtersc = nullsc + (bias_filtersc * ((F1_L > window_len) ? 1.0f : (float)F1_L / window_len));
      seq_score = (usc - filtersc) / eslCONST_LOG2;
    } else {
      seq_score = (usc - nullsc) / eslCONST_LOG2;
    }

    P = esl_gumbel_surv(seq_score, om->evparam[p7_MMU], om->evparam[p7_MLAMBDA]);
    if (P > pli->F1) continue;

    if (nsurv != i)
      wl->windows[nsurv] = wl->windows[i];
    nsurv++;
  }

  wl->count = nsurv;
  *ret_nsurv = nsurv;
  return eslOK;
}

/* GPU Viterbi pre-filter on a subset of windows.
 * Windows that fail GPU Viterbi (single-score < threshold) cannot produce
 * sub-windows in scanning Viterbi, so they are removed.
 * Filters wl in-place, returns survivor count in *ret_nsurv. */
static int
nhmmer_gpu_viterbi_prefilter(NHMMER_GPU_INFO *info, const ESL_SQ *sq,
                             NHMMER_GPU_WINDOW_BATCH *wb,
                             P7_HMM_WINDOWLIST *wl, int *ret_nsurv,
                             char *errbuf, int errbuf_size)
{
  int       status;
  int       N = wl->count;
  float    *vit_scores  = NULL;
  int      *vit_status  = NULL;
  float    *filtersc    = NULL;
  int      *passed      = NULL;
  int      *seqidx      = NULL;
  float    *bias_scores = NULL;
  int       nsurv = 0;
  int       i;
  P7_OPROFILE *om = info->om;
  P7_PIPELINE *pli = info->pli;

  nhmmer_gpu_window_batch_pack(wb, sq, wl);

  ESL_ALLOC(vit_scores, sizeof(float) * N);
  ESL_ALLOC(vit_status, sizeof(int)   * N);
  ESL_ALLOC(filtersc,   sizeof(float) * N);
  ESL_ALLOC(passed,     sizeof(int)   * N);
  ESL_ALLOC(seqidx,     sizeof(int)   * N);
  ESL_ALLOC(bias_scores, sizeof(float) * N);

  /* GPU null scores */
  float *null_scores = NULL;
  ESL_ALLOC(null_scores, sizeof(float) * N);
  status = p7_cuda_NullScoreDsqdataChunk(info->cuda_engine, info->bg, &wb->chu, null_scores, errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;

  /* GPU bias scores */
  if (pli->do_biasfilter) {
    status = p7_cuda_BiasFilterDsqdataChunk(info->cuda_engine, info->bg, &wb->chu, bias_scores, errbuf, errbuf_size);
    if (status != eslOK) goto ERROR;
  }

  for (i = 0; i < N; i++) {
    seqidx[i] = i;

    int   window_len = (int)wb->lengths[i];
    int   F2_L      = ESL_MIN(window_len, pli->B2);
    float nullsc    = null_scores[i];

    if (pli->do_biasfilter) {
      float bias_filtersc_val = bias_scores[i] - nullsc;
      filtersc[i] = nullsc + (bias_filtersc_val * ((F2_L > window_len) ? 1.0f : (float)F2_L / window_len));
    } else {
      filtersc[i] = nullsc;
    }
  }

  status = p7_cuda_ViterbiFilterDsqdataSubset(info->cuda_engine, info->cuda_msv,
                                              &wb->chu, seqidx, N,
                                              filtersc,
                                              om->evparam[p7_VMU], om->evparam[p7_VLAMBDA],
                                              pli->F2,
                                              vit_scores, vit_status, passed,
                                              errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;

  for (i = 0; i < N; i++) {
    if (passed[i]) {
      if (nsurv != i)
        wl->windows[nsurv] = wl->windows[i];
      nsurv++;
    }
  }

  wl->count = nsurv;
  *ret_nsurv = nsurv;

  free(vit_scores);
  free(vit_status);
  free(filtersc);
  free(passed);
  free(seqidx);
  free(bias_scores);
  free(null_scores);
  return eslOK;

ERROR:
  if (vit_scores)  free(vit_scores);
  if (vit_status)  free(vit_status);
  if (filtersc)    free(filtersc);
  if (passed)      free(passed);
  if (seqidx)      free(seqidx);
  if (bias_scores) free(bias_scores);
  if (null_scores) free(null_scores);
  *ret_nsurv = 0;
  return status;
}

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

/* Post-Viterbi worker: processes sub-windows that already passed GPU scanning Viterbi.
 * Only runs Forward/Backward and domain definition (skips MSV and scanning Viterbi). */
static void
nhmmer_gpu_worker_process_post_vit(NHMMER_GPU_WORKER *w)
{
  int          i;
  int          overlap = 0;
  P7_HMM_WINDOW *window;

  for (i = 0; i < w->nwindows; i++) {
    window = w->windows + i;
    ESL_DSQ *subseq = w->sq->dsq + window->n - 1;

    w->pli->pos_past_msv += window->length;
    w->pli->pos_past_bias += window->length;
    w->pli->pos_past_vit += window->length;
    if (i > 0)
      w->pli->pos_past_vit -= ESL_MAX(0, (int)(w->windows[i-1].n + w->windows[i-1].length) - (int)window->n);

    p7_omx_GrowTo(w->pli->oxf, w->om->M, 0, window->length);
    p7_oprofile_ReconfigRestLength(w->om, ESL_MIN((int)window->length, w->om->max_length));

    w->status = p7_pli_postViterbi_LongTarget(w->pli, w->om, w->bg, w->th, w->scoredata,
                                              w->seq_id, (int)window->n, window->length, subseq,
                                              (int64_t)w->sq->start, w->sq->name, w->sq->source,
                                              w->sq->acc, w->sq->desc,
                                              -1, w->complementarity, &overlap, w->pli_tmp);
    if (w->status != eslOK) return;

    if (overlap == -1 && i < w->nwindows - 1) {
      overlap = ESL_MAX(0, (int)(window->n + window->length) - (int)w->windows[i+1].n);
    } else {
      overlap = 0;
    }

    w->pli->ddef->ndom = 0;
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

static void *
nhmmer_gpu_thread_func_post_vit(void *arg)
{
  NHMMER_GPU_WORKER *w = (NHMMER_GPU_WORKER *)arg;
  nhmmer_gpu_worker_process_post_vit(w);
  return NULL;
}
#endif

/* GPU scanning Viterbi longtarget: replaces CPU p7_ViterbiFilter_longtarget.
 * Computes per-window thresholds, runs GPU kernel, converts output to window list.
 * Returns sub-windows in *ret_vit_wl (caller must free windows). */
static int
nhmmer_gpu_viterbi_longtarget(NHMMER_GPU_INFO *info, const ESL_SQ *sq,
                              P7_HMM_WINDOWLIST *input_wl,
                              P7_HMM_WINDOWLIST *ret_vit_wl,
                              char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int N = input_wl->count;
  P7_OPROFILE *om = info->om;
  P7_PIPELINE *pli = info->pli;
  float       *bias_scores = NULL;
  P7_CUDA_VIT_LT_WINDOW *gpu_windows = NULL;
  int          gpu_nwindows = 0;
  int          i;
  int          max_window_len = 80000;
  int          overlap_len = ESL_MIN(40000, om->max_length);

  p7_hmmwindow_init(ret_vit_wl);

  if (N == 0) return eslOK;

  /* GPU bias filter scores (if enabled) — reuse batch infrastructure */
  if (pli->do_biasfilter) {
    NHMMER_GPU_WINDOW_BATCH wb;
    memset(&wb, 0, sizeof(wb));
    status = nhmmer_gpu_window_batch_init(&wb, N);
    if (status != eslOK) goto ERROR;
    status = nhmmer_gpu_window_batch_pack(&wb, sq, input_wl);
    if (status != eslOK) { nhmmer_gpu_window_batch_free(&wb); goto ERROR; }

    ESL_ALLOC(bias_scores, sizeof(float) * N);
    status = p7_cuda_BiasFilterDsqdataChunk(info->cuda_engine, info->bg, &wb.chu, bias_scores,
                                             errbuf, errbuf_size);
    nhmmer_gpu_window_batch_free(&wb);
    if (status != eslOK) goto ERROR;
  }

  status = p7_cuda_ViterbiLongtarget(info->cuda_engine, info->cuda_msv,
                                     sq->dsq, sq->n,
                                     input_wl->windows, N,
                                     bias_scores, pli->do_biasfilter,
                                     pli->B2, pli->F2,
                                     om->evparam[p7_VMU], om->evparam[p7_VLAMBDA],
                                     om->scale_w,
                                     (float)om->xw[p7O_E][p7O_MOVE],
                                     (float)om->xw[p7O_C][p7O_MOVE],
                                     (float)om->base_w, om->max_length,
                                     &gpu_windows, &gpu_nwindows,
                                     errbuf, errbuf_size);
  if (bias_scores) { free(bias_scores); bias_scores = NULL; }
  if (status != eslOK) goto ERROR;

  if (gpu_nwindows == 0) {
    return eslOK;
  }

  for (i = 0; i < gpu_nwindows; i++) {
    int win_id = gpu_windows[i].window_id;
    int64_t base_n = input_wl->windows[win_id].n;
    int pos = gpu_windows[i].position;
    int k = gpu_windows[i].model_k;

    p7_hmmwindow_new(ret_vit_wl, 0,
                     (uint32_t)(base_n + pos - 1),
                     0, (uint16_t)k, 1, 0.0f,
                     p7_NOCOMPLEMENT,
                     (uint32_t)sq->n);
  }
  free(gpu_windows);
  gpu_windows = NULL;

  p7_pli_ExtendAndMergeWindows(om, info->scoredata, ret_vit_wl, 0.5);

  for (i = 0; i < ret_vit_wl->count; i++) {
    if ((int)ret_vit_wl->windows[i].length > max_window_len) {
      uint64_t new_n   = ret_vit_wl->windows[i].n;
      uint32_t new_len = ret_vit_wl->windows[i].length;
      ret_vit_wl->windows[i].length = max_window_len;
      do {
        int shift = max_window_len - overlap_len;
        new_n   += shift;
        new_len -= shift;
        p7_hmmwindow_new(ret_vit_wl, 0, new_n, 0, 0,
                         ESL_MIN(max_window_len, new_len), 0.0f,
                         p7_NOCOMPLEMENT, new_len);
      } while ((int)new_len > max_window_len);
    }
  }

  return eslOK;

ERROR:
  if (bias_scores) free(bias_scores);
  free(gpu_windows);
  return status;
}

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

  /* GPU batch SSV/bias/F1 gating: filter merged windows before thread dispatch */
  NHMMER_GPU_WINDOW_BATCH wb;
  memset(&wb, 0, sizeof(wb));

  if (info->do_gpu_batch && msv_windowlist.count >= NHMMER_GPU_BATCH_MIN) {
    status = nhmmer_gpu_window_batch_init(&wb, msv_windowlist.count);
    if (status != eslOK) goto ERROR;

    status = nhmmer_gpu_window_batch_pack(&wb, sq, &msv_windowlist);
    if (status != eslOK) { nhmmer_gpu_window_batch_free(&wb); goto ERROR; }

    int nsurv = 0;
    status = nhmmer_gpu_batch_filter(info, &wb, &msv_windowlist, &nsurv, errbuf, errbuf_size);
    if (status != eslOK) {
      fprintf(stderr, "GPU batch SSV/bias filter failed: %s\n", errbuf);
      nhmmer_gpu_window_batch_free(&wb);
      goto ERROR;
    }

    if (msv_windowlist.count == 0) {
      nhmmer_gpu_window_batch_free(&wb);
      free(msv_windowlist.windows);
      return eslOK;
    }

    /* GPU Viterbi pre-filter: remove windows whose single-score Viterbi < F2 */
    if (info->do_gpu_vit && msv_windowlist.count >= NHMMER_GPU_BATCH_MIN) {
      status = nhmmer_gpu_viterbi_prefilter(info, sq, &wb, &msv_windowlist, &nsurv,
                                            errbuf, errbuf_size);
      if (status != eslOK) {
        fprintf(stderr, "GPU Viterbi pre-filter failed: %s\n", errbuf);
        nhmmer_gpu_window_batch_free(&wb);
        goto ERROR;
      }

      if (msv_windowlist.count == 0) {
        nhmmer_gpu_window_batch_free(&wb);
        free(msv_windowlist.windows);
        return eslOK;
      }
    }

    nhmmer_gpu_window_batch_free(&wb);
  }

  /* GPU scanning Viterbi longtarget: replaces CPU scanning Viterbi in workers */
  if (info->do_gpu_vit_lt && msv_windowlist.count > 0) {
    P7_HMM_WINDOWLIST vit_wl;
    vit_wl.windows = NULL;

    status = nhmmer_gpu_viterbi_longtarget(info, sq, &msv_windowlist, &vit_wl,
                                           errbuf, errbuf_size);
    if (status != eslOK) {
      fprintf(stderr, "GPU scanning Viterbi failed: %s\n", errbuf);
      if (vit_wl.windows) free(vit_wl.windows);
      goto ERROR;
    }

    free(msv_windowlist.windows);
    msv_windowlist.windows = NULL;

    if (vit_wl.count == 0) {
      if (vit_wl.windows) free(vit_wl.windows);
      return eslOK;
    }

    /* Dispatch sub-windows to post-Viterbi workers (Forward/Backward only) */
    int ncpus_vlt = info->ncpus;
    if (ncpus_vlt < 1) ncpus_vlt = 1;
    if (ncpus_vlt > vit_wl.count) ncpus_vlt = vit_wl.count;

#ifdef HMMER_THREADS
    if (ncpus_vlt > 1) {
      NHMMER_GPU_WORKER *workers = NULL;
      pthread_t         *threads = NULL;
      int                nworkers = ncpus_vlt;
      int                windows_per_thread = vit_wl.count / nworkers;
      int                remainder = vit_wl.count % nworkers;
      int                offset = 0;

      ESL_ALLOC(workers, sizeof(NHMMER_GPU_WORKER) * nworkers);
      ESL_ALLOC(threads, sizeof(pthread_t) * nworkers);
      memset(workers, 0, sizeof(NHMMER_GPU_WORKER) * nworkers);

      for (i = 0; i < nworkers; i++) {
        status = nhmmer_gpu_worker_init(&workers[i], info);
        if (status != eslOK) { nworkers = i; goto VLT_THREAD_ERROR; }

        workers[i].sq              = sq;
        workers[i].seq_id          = seq_id;
        workers[i].complementarity = complementarity;
        workers[i].nwindows        = windows_per_thread + (i < remainder ? 1 : 0);
        workers[i].windows         = vit_wl.windows + offset;
        offset += workers[i].nwindows;
      }

      for (i = 1; i < nworkers; i++)
        pthread_create(&threads[i], NULL, nhmmer_gpu_thread_func_post_vit, &workers[i]);

      nhmmer_gpu_worker_process_post_vit(&workers[0]);

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

    VLT_THREAD_ERROR:
      for (i = 0; i < nworkers; i++)
        nhmmer_gpu_worker_destroy(&workers[i]);
      free(workers);
      free(threads);
      free(vit_wl.windows);
      return status;
    }
#endif /* HMMER_THREADS */

    /* Single-threaded fallback for post-Viterbi */
    {
      NHMMER_GPU_WORKER w;
      memset(&w, 0, sizeof(w));
      status = nhmmer_gpu_worker_init(&w, info);
      if (status != eslOK) { free(vit_wl.windows); return status; }

      w.sq              = sq;
      w.seq_id          = seq_id;
      w.complementarity = complementarity;
      w.nwindows        = vit_wl.count;
      w.windows         = vit_wl.windows;

      nhmmer_gpu_worker_process_post_vit(&w);
      status = w.status;

      if (status == eslOK) {
        p7_tophits_Merge(info->th, w.th);
        p7_pipeline_Merge(info->pli, w.pli);
      }

      nhmmer_gpu_worker_destroy(&w);
      free(vit_wl.windows);
      return status;
    }
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

/* Process one strand's worth of pre-built chunks from nucdb.
 * Chunks are packed as synthetic batch and run through GPU SSV+bias+F1
 * then GPU Viterbi pre-filter, then threaded CPU downstream. */
static int
nhmmer_gpu_process_nucdb_strand(NHMMER_GPU_INFO *info,
                                const P7_NUCDB *ndb,
                                int chunk_start, int chunk_count,
                                const ESL_SQ *sq, int complementarity,
                                int64_t seq_id,
                                char *errbuf, int errbuf_size)
{
  int       status = eslOK;
  int       i;
  P7_OPROFILE *om = info->om;

  /* Build an HMM window list from the pre-chunked data.
   * Each chunk becomes a window. Overlapping regions will be
   * merged by p7_pli_ExtendAndMergeWindows below. */
  P7_HMM_WINDOWLIST msv_windowlist;
  p7_hmmwindow_init(&msv_windowlist);

  /* We run SSV on each chunk as a "window" and then do batch filtering.
   * First, compute SSV threshold (same as in nhmmer_gpu_serial_loop). */
  float nullsc;
  p7_bg_SetLength(info->bg, om->max_length);
  p7_bg_NullOne(info->bg, sq->dsq, om->max_length, &nullsc);

  float invP = esl_gumbel_invsurv(info->pli->F1, om->evparam[p7_MMU], om->evparam[p7_MLAMBDA]);
  uint8_t sc_thresh = (uint8_t)ceil(((nullsc + (invP * eslCONST_LOG2) + 3.0) * om->scale_b)
                                    + om->base_b + om->tec_b + om->tjb_b);

  int chunk_size = info->gpu_chunk_size > 0 ? info->gpu_chunk_size : NHMMER_GPU_CHUNK_SIZE;
  int overlap    = om->max_length;

  P7_CUDA_LT_WINDOW *gpu_windows = NULL;
  int gpu_nwindows = 0;

  /* Use GPU-resident path when nucdb is uploaded and chunks have sufficient overlap */
  const uint8_t *d_nucdb = p7_cuda_engine_NucdbDevPtr(info->cuda_engine);
  if (d_nucdb &&
      (int64_t)ndb->hdr.overlap >= (int64_t)om->max_length &&
      (int64_t)ndb->hdr.chunk_size == (int64_t)chunk_size)
  {
    int *h_offsets = (int *)malloc(sizeof(int) * chunk_count);
    int *h_lengths = (int *)malloc(sizeof(int) * chunk_count);
    if (!h_offsets || !h_lengths) { free(h_offsets); free(h_lengths); return eslEMEM; }

    int ndb_step = (int)ndb->hdr.chunk_size - (int)ndb->hdr.overlap;
    if (ndb_step < 1) ndb_step = 1;

    for (int c = 0; c < chunk_count; c++) {
      P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[chunk_start + c];
      h_offsets[c] = (int)ci->data_offset;
      h_lengths[c] = ci->length;
    }

    status = p7_cuda_SSVLongtargetResident(info->cuda_engine, info->cuda_msv,
                                            d_nucdb, chunk_count,
                                            h_offsets, h_lengths,
                                            info->scoredata->ssv_scores, om->abc->Kp,
                                            sc_thresh, om->scale_b,
                                            ndb_step,
                                            &gpu_windows, &gpu_nwindows,
                                            errbuf, errbuf_size);
    free(h_offsets);
    free(h_lengths);
  }
  else
  {
    /* Fallback: use host dsq (kernel does its own chunking with overlap) */
    status = p7_cuda_SSVLongtarget(info->cuda_engine, info->cuda_msv,
                                   sq->dsq, sq->n,
                                   info->scoredata->ssv_scores, om->abc->Kp,
                                   sc_thresh, om->scale_b,
                                   chunk_size, overlap,
                                   &gpu_windows, &gpu_nwindows,
                                   errbuf, errbuf_size);
  }
  if (status != eslOK) return status;

  if (gpu_nwindows == 0) { free(gpu_windows); return eslOK; }

  for (i = 0; i < gpu_nwindows; i++) {
    p7_hmmwindow_new(&msv_windowlist,
                     0,
                     gpu_windows[i].target_start,
                     0,
                     (uint16_t)gpu_windows[i].model_end,
                     gpu_windows[i].model_end - gpu_windows[i].model_start + 1,
                     gpu_windows[i].score,
                     p7_NOCOMPLEMENT,
                     sq->n);
  }
  free(gpu_windows);

  if (info->scoredata->prefix_lengths == NULL)
    p7_hmm_ScoreDataComputeRest(om, info->scoredata);

  p7_pli_ExtendAndMergeWindows(om, info->scoredata, &msv_windowlist, 0);

  if (msv_windowlist.count == 0) {
    free(msv_windowlist.windows);
    return eslOK;
  }

  /* GPU batch SSV/bias/F1 gating */
  NHMMER_GPU_WINDOW_BATCH wb;
  memset(&wb, 0, sizeof(wb));

  if (info->do_gpu_batch && msv_windowlist.count >= NHMMER_GPU_BATCH_MIN) {
    status = nhmmer_gpu_window_batch_init(&wb, msv_windowlist.count);
    if (status != eslOK) goto ERROR;

    status = nhmmer_gpu_window_batch_pack(&wb, sq, &msv_windowlist);
    if (status != eslOK) { nhmmer_gpu_window_batch_free(&wb); goto ERROR; }

    int nsurv = 0;
    status = nhmmer_gpu_batch_filter(info, &wb, &msv_windowlist, &nsurv, errbuf, errbuf_size);
    if (status != eslOK) { nhmmer_gpu_window_batch_free(&wb); goto ERROR; }

    if (msv_windowlist.count == 0) {
      nhmmer_gpu_window_batch_free(&wb);
      free(msv_windowlist.windows);
      return eslOK;
    }

    if (info->do_gpu_vit && msv_windowlist.count >= NHMMER_GPU_BATCH_MIN) {
      status = nhmmer_gpu_viterbi_prefilter(info, sq, &wb, &msv_windowlist, &nsurv,
                                            errbuf, errbuf_size);
      if (status != eslOK) { nhmmer_gpu_window_batch_free(&wb); goto ERROR; }

      if (msv_windowlist.count == 0) {
        nhmmer_gpu_window_batch_free(&wb);
        free(msv_windowlist.windows);
        return eslOK;
      }
    }

    nhmmer_gpu_window_batch_free(&wb);
  }

  /* GPU scanning Viterbi longtarget */
  if (info->do_gpu_vit_lt && msv_windowlist.count > 0) {
    P7_HMM_WINDOWLIST vit_wl;
    vit_wl.windows = NULL;

    status = nhmmer_gpu_viterbi_longtarget(info, sq, &msv_windowlist, &vit_wl,
                                           errbuf, errbuf_size);
    if (status != eslOK) {
      if (vit_wl.windows) free(vit_wl.windows);
      goto ERROR;
    }

    free(msv_windowlist.windows);
    msv_windowlist.windows = NULL;

    if (vit_wl.count == 0) {
      if (vit_wl.windows) free(vit_wl.windows);
      return eslOK;
    }

    /* Dispatch to post-Viterbi workers */
    int ncpus_vlt = info->ncpus;
    if (ncpus_vlt < 1) ncpus_vlt = 1;
    if (ncpus_vlt > vit_wl.count) ncpus_vlt = vit_wl.count;

#ifdef HMMER_THREADS
    if (ncpus_vlt > 1) {
      NHMMER_GPU_WORKER *workers = NULL;
      pthread_t         *threads = NULL;
      int                nworkers = ncpus_vlt;
      int                windows_per_thread = vit_wl.count / nworkers;
      int                remainder = vit_wl.count % nworkers;
      int                offset = 0;

      ESL_ALLOC(workers, sizeof(NHMMER_GPU_WORKER) * nworkers);
      ESL_ALLOC(threads, sizeof(pthread_t) * nworkers);
      memset(workers, 0, sizeof(NHMMER_GPU_WORKER) * nworkers);

      for (i = 0; i < nworkers; i++) {
        status = nhmmer_gpu_worker_init(&workers[i], info);
        if (status != eslOK) { nworkers = i; goto VLT2_THREAD_ERROR; }

        workers[i].sq              = sq;
        workers[i].seq_id          = seq_id;
        workers[i].complementarity = complementarity;
        workers[i].nwindows        = windows_per_thread + (i < remainder ? 1 : 0);
        workers[i].windows         = vit_wl.windows + offset;
        offset += workers[i].nwindows;
      }

      for (i = 1; i < nworkers; i++)
        pthread_create(&threads[i], NULL, nhmmer_gpu_thread_func_post_vit, &workers[i]);
      nhmmer_gpu_worker_process_post_vit(&workers[0]);
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

    VLT2_THREAD_ERROR:
      for (i = 0; i < nworkers; i++)
        nhmmer_gpu_worker_destroy(&workers[i]);
      free(workers);
      free(threads);
      free(vit_wl.windows);
      return status;
    }
#endif

    {
      NHMMER_GPU_WORKER w;
      memset(&w, 0, sizeof(w));
      status = nhmmer_gpu_worker_init(&w, info);
      if (status != eslOK) { free(vit_wl.windows); return status; }

      w.sq              = sq;
      w.seq_id          = seq_id;
      w.complementarity = complementarity;
      w.nwindows        = vit_wl.count;
      w.windows         = vit_wl.windows;

      nhmmer_gpu_worker_process_post_vit(&w);
      status = w.status;

      if (status == eslOK) {
        p7_tophits_Merge(info->th, w.th);
        p7_pipeline_Merge(info->pli, w.pli);
      }

      nhmmer_gpu_worker_destroy(&w);
      free(vit_wl.windows);
      return status;
    }
  }

  /* Fallback: dispatch windows to full CPU workers */
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
      if (status != eslOK) { nworkers = i; goto NDB_THREAD_ERROR; }

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

  NDB_THREAD_ERROR:
    for (i = 0; i < nworkers; i++)
      nhmmer_gpu_worker_destroy(&workers[i]);
    free(workers);
    free(threads);
    free(msv_windowlist.windows);
    return status;
  }
#endif

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
  if (msv_windowlist.windows) free(msv_windowlist.windows);
  return status;
}

/* Main loop for nucdb format: iterate over sequences, build ESL_SQ from
 * pre-digitized chunks, process each strand. */
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

  p7_bg_SetLength(bg, om->max_length);
  p7_oprofile_ReconfigMSVLength(om, om->max_length);
  p7_cuda_msvprofile_UpdateLength(info->cuda_msv, om, om->max_length, errbuf, sizeof(errbuf));

  /* Upload nucdb data to GPU once for all sequences */
  int64_t nucdb_data_size = (int64_t)(ndb->mmap_size - ndb->hdr.data_offset);
  status = p7_cuda_engine_UploadNucdb(info->cuda_engine, ndb->chunk_data, nucdb_data_size,
                                       errbuf, sizeof(errbuf));
  if (status != eslOK) {
    fprintf(stderr, "GPU nhmmer: failed to upload nucdb: %s\n", errbuf);
    goto ERROR;
  }

  for (int64_t si = 0; si < (int64_t)ndb->hdr.nseq; si++) {
    P7_NUCDB_SEQ_IDX *sidx = &ndb->seq_idx[si];
    const char       *seqname = ndb->name_blob + sidx->name_offset;

    /* Build forward-strand ESL_SQ from nucdb data.
     * The full sequence is reconstructed from its chunks (stripping overlaps). */
    ESL_SQ *sq = esl_sq_CreateDigital(om->abc);
    if (!sq) return eslEMEM;
    esl_sq_SetName(sq, seqname);
    esl_sq_GrowTo(sq, sidx->length);

    /* Reconstruct the full digitized sequence from forward chunks */
    int64_t step = (int64_t)ndb->hdr.chunk_size - (int64_t)ndb->hdr.overlap;
    if (step < 1) step = 1;

    for (int c = 0; c < sidx->fwd_chunk_count; c++) {
      P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[sidx->fwd_chunk_start + c];
      uint8_t *chunk_dsq = ndb->chunk_data + ci->data_offset;
      /* chunk_dsq[0] = sentinel, chunk_dsq[1..clen] = residues */

      int64_t copy_start = 0;
      int64_t copy_len   = ci->length;
      if (c > 0) {
        /* Skip the overlap region (already copied from previous chunk) */
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
    sq->n = sidx->length;
    sq->dsq[0] = eslDSQ_SENTINEL;
    sq->dsq[sq->n + 1] = eslDSQ_SENTINEL;
    sq->start = 1;
    sq->end   = sq->n;
    sq->L     = sq->n;

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
        esl_sq_Destroy(sq);
        goto ERROR;
      }
    }

    /* Reverse complement strand — reconstruct directly from nucdb RC chunks */
    if (strands != p7_STRAND_TOPONLY && sidx->rc_chunk_count > 0) {
      ESL_SQ *sq_rc = esl_sq_CreateDigital(om->abc);
      if (!sq_rc) { esl_sq_Destroy(sq); return eslEMEM; }
      esl_sq_SetName(sq_rc, seqname);
      esl_sq_GrowTo(sq_rc, sidx->length);

      for (int c = 0; c < sidx->rc_chunk_count; c++) {
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
          memcpy(sq_rc->dsq + 1 + ci->seq_offset + copy_start,
                 chunk_dsq + 1 + copy_start, copy_len);
      }
      sq_rc->n = sidx->length;
      sq_rc->dsq[0] = eslDSQ_SENTINEL;
      sq_rc->dsq[sq_rc->n + 1] = eslDSQ_SENTINEL;
      sq_rc->start = 1;
      sq_rc->end   = sq_rc->n;
      sq_rc->L     = sq_rc->n;
      nres += sq_rc->n;

      status = nhmmer_gpu_process_nucdb_strand(info, ndb,
                                                sidx->rc_chunk_start, sidx->rc_chunk_count,
                                                sq_rc, p7_COMPLEMENT, si,
                                                errbuf, sizeof(errbuf));
      if (status != eslOK) {
        fprintf(stderr, "GPU nhmmer nucdb revcomp strand failed: %s\n", errbuf);
        esl_sq_Destroy(sq_rc);
        esl_sq_Destroy(sq);
        goto ERROR;
      }
      esl_sq_Destroy(sq_rc);
    }

    p7_pipeline_Reuse(info->pli);
    if (idlen_cb) idlen_cb(idlen_data, si, sidx->length);
    info->pli->nseqs++;

    esl_sq_Destroy(sq);
  }

  *ret_nseqs = (int)ndb->hdr.nseq;
  *ret_nres  = nres;
  p7_cuda_engine_ReleaseNucdb(info->cuda_engine);
  return eslOK;

ERROR:
  p7_cuda_engine_ReleaseNucdb(info->cuda_engine);
  return status;
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

int
nhmmer_gpu_nucdb_loop(NHMMER_GPU_INFO *info, P7_NUCDB *ndb,
                      int strands, NHMMER_GPU_IDLEN_CB idlen_cb, void *idlen_data,
                      int *ret_nseqs, int64_t *ret_nres)
{
  ESL_UNUSED(info); ESL_UNUSED(ndb); ESL_UNUSED(strands);
  ESL_UNUSED(idlen_cb); ESL_UNUSED(idlen_data);
  ESL_UNUSED(ret_nseqs); ESL_UNUSED(ret_nres);
  return eslENORESULT;
}

#endif /* HMMER_CUDA */
