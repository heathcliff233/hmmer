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
  /* GPU FB parser results (non-owning pointers into shared buffers) */
  float            *gpu_xf;
  float            *gpu_xb;
  float            *gpu_scores;
  int              *gpu_statuses;
  size_t           *gpu_x_offsets;
  int              *gpu_L_eff;
  int               use_gpu_fb;
  P7_CUDA_ENGINE   *cuda_engine;
  P7_CUDA_MSVPROFILE *cuda_msv;
#ifdef HMMER_THREADS
  pthread_mutex_t  *gpu_domain_mutex;
#endif
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
  w->cuda_engine     = info->cuda_engine;
  w->cuda_msv        = info->cuda_msv;
#ifdef HMMER_THREADS
  w->gpu_domain_mutex = &((NHMMER_GPU_INFO *)info)->gpu_domain_mutex;
#endif
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

typedef struct {
  float *xmx;
  int    allocXR;
  int    M;
  int    L;
  int    has_own_scales;
  float  totscale;
} NHMMER_OMX_BINDING;

static void
nhmmer_gpu_BindOmxXmx(P7_OMX *ox, float *xmx, int M, int L, int has_own_scales, float totscale, NHMMER_OMX_BINDING *saved)
{
  saved->xmx            = ox->xmx;
  saved->allocXR        = ox->allocXR;
  saved->M              = ox->M;
  saved->L              = ox->L;
  saved->has_own_scales = ox->has_own_scales;
  saved->totscale       = ox->totscale;
  ox->xmx            = xmx;
  ox->allocXR        = L + 1;
  ox->M              = M;
  ox->L              = L;
  ox->has_own_scales = has_own_scales;
  ox->totscale       = totscale;
}

static void
nhmmer_gpu_RestoreOmxXmx(P7_OMX *ox, const NHMMER_OMX_BINDING *saved)
{
  ox->xmx            = saved->xmx;
  ox->allocXR        = saved->allocXR;
  ox->M              = saved->M;
  ox->L              = saved->L;
  ox->has_own_scales = saved->has_own_scales;
  ox->totscale       = saved->totscale;
}

static int
nhmmer_gpu_computeAliScores(P7_DOMAIN *dom, ESL_DSQ *seq, const P7_SCOREDATA *data, int K)
{
  int status;
  int i, j, k;
  float sc;

  ESL_ALLOC( dom->scores_per_pos, sizeof(float) * dom->ad->N );
  for (i=0; i<dom->ad->N; i++)  dom->scores_per_pos[i] = 0.0;

  i = dom->iali - 1;
  j = dom->ad->hmmfrom - 1;
  k = 0;
  while ( k<dom->ad->N) {
    if (dom->ad->model[k] != '.' && dom->ad->aseq[k] != '-') {
      i++;  j++;
      dom->scores_per_pos[k] = data->fwd_scores[K * j + seq[i]]
                             +  (j==1 ? 0 : log(data->fwd_transitions[p7O_MM][j]) );
      k++;
    } else if (dom->ad->model[k] == '.' ) {
      dom->scores_per_pos[k] = -eslINFINITY;
      sc = log(data->fwd_transitions[p7O_MI][j]);
      i++; k++;
      while (k<dom->ad->N && dom->ad->model[k] == '.') {
        dom->scores_per_pos[k] = -eslINFINITY;
        sc += log(data->fwd_transitions[p7O_II][j]);
        i++; k++;
      }
      sc += log(data->fwd_transitions[p7O_IM][j+1]) - log(data->fwd_transitions[p7O_MM][j+1]);
      dom->scores_per_pos[k-1] = sc;
    } else if (dom->ad->aseq[k] == '-' ) {
      dom->scores_per_pos[k] = -eslINFINITY;
      sc = log(data->fwd_transitions[p7O_MD][j]);
      j++; k++;
      while (k<dom->ad->N && dom->ad->aseq[k] == '-')  {
        dom->scores_per_pos[k] = -eslINFINITY;
        sc += log(data->fwd_transitions[p7O_DD][j]);
        j++; k++;
      }
      sc += log(data->fwd_transitions[p7O_DM][j+1]) - log(data->fwd_transitions[p7O_MM][j+1]);
      dom->scores_per_pos[k-1] = sc;
    }
  }

  return eslOK;

ERROR:
  return eslEMEM;
}

static const char gpu_to_p7_state[] = {
  p7T_S,   /* GPU_p7T_S = 0 */
  p7T_N,   /* GPU_p7T_N = 1 */
  p7T_B,   /* GPU_p7T_B = 2 */
  p7T_M,   /* GPU_p7T_M = 3 */
  p7T_D,   /* GPU_p7T_D = 4 */
  p7T_I,   /* GPU_p7T_I = 5 */
  p7T_E,   /* GPU_p7T_E = 6 */
  p7T_C,   /* GPU_p7T_C = 7 */
  p7T_T,   /* GPU_p7T_T = 8 */
  p7T_J,   /* GPU_p7T_J = 9 */
};

static int
nhmmer_gpu_trace_from_gpu(P7_TRACE *tr, int8_t *st, int *tk, int *ti, float *tp,
                          int tN, int M, int L)
{
  int z, status;

  p7_trace_Reuse(tr);
  for (z = tN - 1; z >= 0; z--) {
    int state = gpu_to_p7_state[(int)st[z]];
    int k_val = tk[z];
    int i_val = ti[z];
    float pp_val = tp[z];
    if ((status = p7_trace_AppendWithPP(tr, state, k_val, i_val, pp_val)) != eslOK) return status;
  }
  tr->M = M;
  tr->L = L;
  return eslOK;
}

static int
nhmmer_gpu_rescore_domains(NHMMER_GPU_WORKER *w, ESL_DSQ *subseq, int window_len,
                           int ienv_base)
{
  P7_DOMAINDEF *ddef = w->pli->ddef;
  P7_OPROFILE  *om   = w->om;
  P7_BG        *bg   = w->bg;
  int           ndom = ddef->ndom;
  int           Q    = p7O_NQF(om->M);
  int           Kp   = om->abc->Kp;
  float         nj   = om->nj;
  int           d, status;
  char          errbuf[256];

  const uint8_t **h_dsq_ptrs = NULL;
  int            *h_lengths  = NULL;
  const float   **h_rfv_ptrs = NULL;
  float          *h_orig_rfv = NULL;
  float          *h_envsc = NULL, *h_domcorr = NULL, *h_oasc = NULL;
  int8_t         *h_trace_st = NULL;
  int            *h_trace_k = NULL, *h_trace_i = NULL, *h_trace_N = NULL;
  float          *h_trace_pp = NULL;
  int            *h_statuses = NULL;
  float         **rfv_copies = NULL;
  int             max_trace_len;
  int             max_Ld = 0;
  int             orig_L = om->L;
  size_t          rfv_size = (size_t)Kp * Q * 4 * sizeof(float);

  if (ndom == 0) return eslOK;

  for (d = 0; d < ndom; d++) {
    int Ld = ddef->dcl[d].jenv - ddef->dcl[d].ienv + 1;
    if (Ld > max_Ld) max_Ld = Ld;
  }
  max_trace_len = max_Ld * 4 + 20;

  ESL_ALLOC(h_dsq_ptrs, sizeof(uint8_t *) * ndom);
  ESL_ALLOC(h_lengths,  sizeof(int)       * ndom);
  ESL_ALLOC(h_rfv_ptrs, sizeof(float *)   * ndom);
  ESL_ALLOC(rfv_copies, sizeof(float *)   * ndom);
  ESL_ALLOC(h_envsc,    sizeof(float)     * ndom);
  ESL_ALLOC(h_domcorr,  sizeof(float)     * ndom);
  ESL_ALLOC(h_oasc,     sizeof(float)     * ndom);
  ESL_ALLOC(h_statuses, sizeof(int)       * ndom);
  ESL_ALLOC(h_trace_st, sizeof(int8_t)    * ndom * max_trace_len);
  ESL_ALLOC(h_trace_k,  sizeof(int)       * ndom * max_trace_len);
  ESL_ALLOC(h_trace_i,  sizeof(int)       * ndom * max_trace_len);
  ESL_ALLOC(h_trace_pp, sizeof(float)     * ndom * max_trace_len);
  ESL_ALLOC(h_trace_N,  sizeof(int)       * ndom);
  ESL_ALLOC(h_orig_rfv, rfv_size);

  for (d = 0; d < ndom; d++) rfv_copies[d] = NULL;

  memcpy(h_orig_rfv, om->rfv[0], rfv_size);

  for (d = 0; d < ndom; d++) {
    int ienv = ddef->dcl[d].ienv;
    int jenv = ddef->dcl[d].jenv;
    int Ld   = jenv - ienv + 1;

    p7_oprofile_ReconfigRestLength(om, Ld);

    if (w->pli_tmp->scores != NULL) {
      reparameterize_model(bg, om, w->pli_tmp->tmpseq, ienv, Ld,
                           w->pli_tmp->fwd_emissions_arr, w->pli_tmp->bg->f, w->pli_tmp->scores);
    }

    ESL_ALLOC(rfv_copies[d], rfv_size);
    memcpy(rfv_copies[d], om->rfv[0], rfv_size);

    h_dsq_ptrs[d] = subseq + ienv - 1;
    h_lengths[d]  = Ld;
    h_rfv_ptrs[d] = rfv_copies[d];

    if (w->pli_tmp->scores != NULL) {
      reparameterize_model(bg, om, NULL, 0, 0,
                           w->pli_tmp->fwd_emissions_arr, w->pli_tmp->bg->f, w->pli_tmp->scores);
    }
  }

#ifdef HMMER_THREADS
  if (w->gpu_domain_mutex) pthread_mutex_lock(w->gpu_domain_mutex);
#endif
  status = p7_cuda_DomainRescoreBatch(w->cuda_engine, w->cuda_msv,
                                       ndom, h_dsq_ptrs, h_lengths,
                                       h_rfv_ptrs, h_orig_rfv,
                                       Q, Kp, nj,
                                       h_envsc, h_domcorr, h_oasc,
                                       h_trace_st, h_trace_k, h_trace_i,
                                       h_trace_pp, h_trace_N, max_trace_len,
                                       h_statuses, errbuf, sizeof(errbuf));
#ifdef HMMER_THREADS
  if (w->gpu_domain_mutex) pthread_mutex_unlock(w->gpu_domain_mutex);
#endif
  if (status != eslOK) goto ERROR;

  for (d = 0; d < ndom; d++) {
    P7_DOMAIN *dom = &ddef->dcl[d];
    int ienv = dom->ienv;
    int jenv = dom->jenv;
    int Ld   = jenv - ienv + 1;

    if (Ld <= 0 || ienv <= 0 || jenv <= 0) {
      dom->ad = NULL; dom->scores_per_pos = NULL;
      dom->envsc = 0.0; dom->domcorrection = 0.0; dom->oasc = 0.0;
      continue;
    }

    if (h_statuses[d] != eslOK) {
      /* GPU overflow (eslERANGE): domain is dropped, matching CPU behavior
       * where p7_Decoding overflow causes rescore_isolated_domain to return eslFAIL */
      dom->ad             = NULL;
      dom->scores_per_pos = NULL;
      dom->envsc          = 0.0;
      dom->domcorrection  = 0.0;
      dom->oasc           = 0.0;
      continue;
    }

    if (h_trace_N[d] <= 0 || h_trace_N[d] > max_trace_len) {
      dom->ad = NULL; dom->scores_per_pos = NULL;
      dom->envsc = 0.0; dom->domcorrection = 0.0; dom->oasc = 0.0;
      continue;
    }

    p7_trace_Reuse(ddef->tr);
    nhmmer_gpu_trace_from_gpu(ddef->tr,
                              h_trace_st + (size_t)d * max_trace_len,
                              h_trace_k  + (size_t)d * max_trace_len,
                              h_trace_i  + (size_t)d * max_trace_len,
                              h_trace_pp + (size_t)d * max_trace_len,
                              h_trace_N[d], om->M, Ld);

    /* Point tmpseq at envelope subsequence for alidisplay creation (envelope-relative trace) */
    ESL_DSQ *saved_dsq = w->pli_tmp->tmpseq->dsq;
    int64_t  saved_n   = w->pli_tmp->tmpseq->n;
    w->pli_tmp->tmpseq->dsq = subseq + ienv - 1;
    w->pli_tmp->tmpseq->n = Ld;
    p7_oprofile_ReconfigRestLength(om, Ld);
    dom->ad = p7_alidisplay_Create(ddef->tr, 0, om, w->pli_tmp->tmpseq, NULL);
    w->pli_tmp->tmpseq->dsq = saved_dsq;
    w->pli_tmp->tmpseq->n = saved_n;
    if (dom->ad == NULL) {
      continue;
    }
    dom->scores_per_pos = NULL;

    float envsc = h_envsc[d];
    float domcorrection;

    if (dom->ad->sqfrom <= 0 || dom->ad->sqto <= 0 || dom->ad->sqfrom > Ld || dom->ad->sqto > Ld) {
      p7_alidisplay_Destroy(dom->ad);
      dom->ad = NULL;
      continue;
    }

    /* Envelope trimming: if alignment is much smaller than envelope, re-run on tighter bounds.
     * sqfrom/sqto are envelope-relative (1..Ld). Trim if envelope extends >20 past alignment. */
    if (1 < dom->ad->sqfrom - 20 || Ld > dom->ad->sqto + 20) {
      /* Compute new envelope in envelope-relative coords, then convert to window-relative */
      int new_env_start = ESL_MAX(1, (int)dom->ad->sqfrom - 20);
      int new_env_end   = ESL_MIN(Ld, (int)dom->ad->sqto + 20);
      int new_ienv = ienv + new_env_start - 1;  /* window-relative */
      int new_jenv = ienv + new_env_end - 1;     /* window-relative */
      int new_Ld = new_jenv - new_ienv + 1;
      const uint8_t *trim_dsq = subseq + new_ienv - 1;
      int trim_len = new_Ld;
      float trim_envsc, trim_domcorr, trim_oasc;
      int   trim_status;
      int8_t *trim_st;
      int    *trim_k, *trim_i, *trim_N_val;
      float  *trim_pp_val;
      int     trim_max_trace = new_Ld * 4 + 20;
      float  *trim_rfv = NULL;

      ESL_ALLOC(trim_st,    sizeof(int8_t) * trim_max_trace);
      ESL_ALLOC(trim_k,     sizeof(int)    * trim_max_trace);
      ESL_ALLOC(trim_i,     sizeof(int)    * trim_max_trace);
      ESL_ALLOC(trim_pp_val,sizeof(float)  * trim_max_trace);
      ESL_ALLOC(trim_N_val, sizeof(int));
      ESL_ALLOC(trim_rfv,   rfv_size);

      p7_oprofile_ReconfigRestLength(om, new_Ld);
      if (w->pli_tmp->scores != NULL) {
        reparameterize_model(bg, om, w->pli_tmp->tmpseq, new_ienv, new_Ld,
                             w->pli_tmp->fwd_emissions_arr, w->pli_tmp->bg->f, w->pli_tmp->scores);
      }
      memcpy(trim_rfv, om->rfv[0], rfv_size);
      if (w->pli_tmp->scores != NULL) {
        reparameterize_model(bg, om, NULL, 0, 0,
                             w->pli_tmp->fwd_emissions_arr, w->pli_tmp->bg->f, w->pli_tmp->scores);
      }

      const float *trim_rfv_const = trim_rfv;
#ifdef HMMER_THREADS
      if (w->gpu_domain_mutex) pthread_mutex_lock(w->gpu_domain_mutex);
#endif
      status = p7_cuda_DomainRescoreBatch(w->cuda_engine, w->cuda_msv,
                                           1, &trim_dsq, &trim_len,
                                           &trim_rfv_const, h_orig_rfv,
                                           Q, Kp, nj,
                                           &trim_envsc, &trim_domcorr, &trim_oasc,
                                           trim_st, trim_k, trim_i,
                                           trim_pp_val, trim_N_val, trim_max_trace,
                                           &trim_status, errbuf, sizeof(errbuf));
#ifdef HMMER_THREADS
      if (w->gpu_domain_mutex) pthread_mutex_unlock(w->gpu_domain_mutex);
#endif

      if (status == eslOK && trim_status == eslOK) {
        envsc = trim_envsc;
        p7_trace_Reuse(ddef->tr);
        nhmmer_gpu_trace_from_gpu(ddef->tr, trim_st, trim_k, trim_i,
                                  trim_pp_val, *trim_N_val, om->M, new_Ld);
        w->pli_tmp->tmpseq->dsq = subseq + new_ienv - 1;
        w->pli_tmp->tmpseq->n = new_Ld;
        p7_oprofile_ReconfigRestLength(om, new_Ld);
        P7_ALIDISPLAY *trim_ad = p7_alidisplay_Create(ddef->tr, 0, om, w->pli_tmp->tmpseq, NULL);
        w->pli_tmp->tmpseq->dsq = saved_dsq;
        w->pli_tmp->tmpseq->n = saved_n;
        if (trim_ad != NULL) {
          p7_alidisplay_Destroy(dom->ad);
          dom->ad = trim_ad;
          dom->ienv = new_ienv;
          dom->jenv = new_jenv;
          domcorrection = trim_domcorr;
        } else {
          domcorrection = h_domcorr[d];
        }
      } else {
        domcorrection = h_domcorr[d];
      }

      free(trim_st); free(trim_k); free(trim_i); free(trim_pp_val);
      free(trim_N_val); free(trim_rfv);
    } else {
      domcorrection = h_domcorr[d];
    }

    if (domcorrection < envsc)
      envsc = domcorrection;

    if (dom->ad == NULL) continue;

    dom->domcorrection = domcorrection - envsc;
    dom->envsc         = envsc;
    dom->oasc          = h_oasc[d];
    dom->iali          = dom->ad->sqfrom;
    dom->jali          = dom->ad->sqto;
    dom->dombias       = 0.0;
    dom->bitscore      = 0.0;
    dom->lnP           = 0.0;
    dom->is_reported   = FALSE;
    dom->is_included   = FALSE;
  }

  p7_oprofile_ReconfigRestLength(om, orig_L);
  p7_trace_Reuse(ddef->tr);
  status = eslOK;

 ERROR:
  if (h_dsq_ptrs)  free(h_dsq_ptrs);
  if (h_lengths)   free(h_lengths);
  if (h_rfv_ptrs)  free(h_rfv_ptrs);
  if (h_orig_rfv)  free(h_orig_rfv);
  if (h_envsc)     free(h_envsc);
  if (h_domcorr)   free(h_domcorr);
  if (h_oasc)      free(h_oasc);
  if (h_statuses)  free(h_statuses);
  if (h_trace_st)  free(h_trace_st);
  if (h_trace_k)   free(h_trace_k);
  if (h_trace_i)   free(h_trace_i);
  if (h_trace_pp)  free(h_trace_pp);
  if (h_trace_N)   free(h_trace_N);
  if (rfv_copies) {
    for (d = 0; d < ndom; d++) if (rfv_copies[d]) free(rfv_copies[d]);
    free(rfv_copies);
  }
  return status;
}

/* Per-domain context saved during Phase 1 (envelope collection) */
typedef struct {
  int       win_ctx_idx;    /* index into win_ctx array */
  ESL_DSQ  *subseq;         /* window start pointer (for coordinate fixups) */
  int       ienv;           /* envelope start (window-relative) */
  int       jenv;           /* envelope end (window-relative) */
  int       Ld;
  float    *rfv_copy;       /* reparameterized rfv (owned) */
} NHMMER_GPU_PDOM;

/* Per-window context saved during Phase 1 for hit reporting in Phase 3 */
typedef struct {
  P7_HMM_WINDOW *window;
  ESL_DSQ       *subseq;
  int            window_len;
  int            ndom;
  int            first_dom;   /* index into pdom array */
  int            L_eff;
} NHMMER_GPU_WCTX;

static void
nhmmer_gpu_worker_process_post_vit_gpu(NHMMER_GPU_WORKER *w)
{
  int          status = eslOK;
  int          i, d;
  int          overlap = 0;
  P7_HMM_WINDOW *window;
  NHMMER_OMX_BINDING saved_oxf, saved_oxb;

  /* Phase 1 collection arrays (grow as needed) */
  NHMMER_GPU_PDOM *pdoms = NULL;
  NHMMER_GPU_WCTX *wctxs = NULL;
  int  npdoms = 0, pdom_alloc = 0;
  int  nwctxs = 0, wctx_alloc = 0;
  int  orig_L = w->om->L;
  int  Q  = p7O_NQF(w->om->M);
  int  Kp = w->om->abc->Kp;
  float nj = w->om->nj;
  size_t rfv_size = (size_t)Kp * Q * 4 * sizeof(float);
  float *orig_rfv = NULL;
  ESL_ALLOC(orig_rfv, rfv_size);
  memcpy(orig_rfv, w->om->rfv[0], rfv_size);

  /* ==== PHASE 1: Envelope collection ==== */
  for (i = 0; i < w->nwindows; i++) {
    window = w->windows + i;
    ESL_DSQ *subseq = w->sq->dsq + window->n - 1;
    int window_len = (int)window->length;
    int L_eff = w->gpu_L_eff[i];

    w->pli->pos_past_msv  += window_len;
    w->pli->pos_past_bias += window_len;
    w->pli->pos_past_vit  += window_len;
    if (i > 0)
      w->pli->pos_past_vit -= ESL_MAX(0, (int)(w->windows[i-1].n + w->windows[i-1].length) - (int)window->n);

    int fwd_status = w->gpu_statuses[i * 2 + 0];
    if (fwd_status != eslOK) {
      p7_omx_GrowTo(w->pli->oxf, w->om->M, 0, window_len);
      w->status = p7_pli_postViterbi_LongTarget(w->pli, w->om, w->bg, w->th, w->scoredata,
                      w->seq_id, (int)window->n, window_len, subseq,
                      (int64_t)w->sq->start, w->sq->name, w->sq->source,
                      w->sq->acc, w->sq->desc, -1, w->complementarity, &overlap, w->pli_tmp);
      if (w->status != eslOK) goto CLEANUP;
      if (overlap == -1 && i < w->nwindows - 1)
        overlap = ESL_MAX(0, (int)(window->n + window->length) - (int)w->windows[i+1].n);
      else
        overlap = 0;
      w->pli->ddef->ndom = 0;
      continue;
    }

    /* F3 gate */
    float fwdsc = w->gpu_scores[i * 2 + 0];
    float nullsc, bias_filtersc, filtersc, seq_score;
    double P;
    int F3_L = ESL_MIN(window_len, w->pli->B3);

    p7_bg_SetLength(w->bg, window_len);
    p7_bg_NullOne(w->bg, subseq, window_len, &nullsc);
    if (w->pli->do_biasfilter) {
      p7_bg_FilterScore(w->bg, subseq, window_len, &bias_filtersc);
      bias_filtersc -= nullsc;
    } else {
      bias_filtersc = 0.0f;
    }
    filtersc = nullsc + bias_filtersc * (F3_L > window_len ? 1.0f : (float)F3_L / window_len);
    seq_score = (fwdsc - filtersc) / eslCONST_LOG2;
    P = esl_exp_surv(seq_score, w->om->evparam[p7_FTAU], w->om->evparam[p7_FLAMBDA]);
    if (P > w->pli->F3) {
      overlap = 0;
      continue;
    }

    w->pli->pos_past_fwd += window_len - (overlap == -1 ? 0 : overlap);
    overlap = -1;

    p7_oprofile_ReconfigRestLength(w->om, window_len);

    ESL_DSQ *dsq_holder = w->pli_tmp->tmpseq->dsq;
    esl_sq_SetName(w->pli_tmp->tmpseq, w->sq->name);
    esl_sq_SetSource(w->pli_tmp->tmpseq, w->sq->source);
    esl_sq_SetAccession(w->pli_tmp->tmpseq, w->sq->acc);
    esl_sq_SetDesc(w->pli_tmp->tmpseq, w->sq->desc);
    w->pli_tmp->tmpseq->L = -1;
    w->pli_tmp->tmpseq->n = window_len;
    w->pli_tmp->tmpseq->dsq = subseq;

    float *xf_ptr = w->gpu_xf + w->gpu_x_offsets[i];
    float *xb_ptr = w->gpu_xb + w->gpu_x_offsets[i];
    float  bcksc  = w->gpu_scores[i * 2 + 1];

    nhmmer_gpu_BindOmxXmx(w->pli->oxf, xf_ptr, w->om->M, L_eff, 1, fwdsc, &saved_oxf);
    nhmmer_gpu_BindOmxXmx(w->pli->oxb, xb_ptr, w->om->M, L_eff, 0, bcksc, &saved_oxb);

    w->status = p7_domaindef_ByPosteriorHeuristics(w->pli_tmp->tmpseq, NULL, w->om,
                    w->pli->oxf, w->pli->oxb, w->pli->fwd, w->pli->bck, w->pli->ddef,
                    w->bg, TRUE,
                    w->pli_tmp->bg,
                    (w->pli->do_null2 ? w->pli_tmp->scores : NULL),
                    w->pli_tmp->fwd_emissions_arr, TRUE);

    nhmmer_gpu_RestoreOmxXmx(w->pli->oxf, &saved_oxf);
    nhmmer_gpu_RestoreOmxXmx(w->pli->oxb, &saved_oxb);
    w->pli_tmp->tmpseq->dsq = dsq_holder;

    if (w->status != eslOK) goto CLEANUP;
    if (w->pli->ddef->nregions == 0 || w->pli->ddef->nenvelopes == 0 || w->pli->ddef->ndom == 0) {
      if (overlap == -1 && i < w->nwindows - 1)
        overlap = ESL_MAX(0, (int)(window->n + window->length) - (int)w->windows[i+1].n);
      else
        overlap = 0;
      w->pli->ddef->ndom = 0;
      continue;
    }

    /* Save window context */
    if (nwctxs >= wctx_alloc) {
      wctx_alloc = wctx_alloc ? wctx_alloc * 2 : 256;
      ESL_REALLOC(wctxs, sizeof(NHMMER_GPU_WCTX) * wctx_alloc);
    }
    NHMMER_GPU_WCTX *wc = &wctxs[nwctxs];
    wc->window     = window;
    wc->subseq     = subseq;
    wc->window_len = window_len;
    wc->ndom       = w->pli->ddef->ndom;
    wc->first_dom  = npdoms;
    wc->L_eff      = L_eff;
    nwctxs++;

    /* Collect domains from this window */
    w->pli_tmp->tmpseq->dsq = subseq;  /* reparameterize_model needs window-relative dsq */
    w->pli_tmp->tmpseq->n = window_len;
    for (d = 0; d < w->pli->ddef->ndom; d++) {
      int ienv = w->pli->ddef->dcl[d].ienv;
      int jenv = w->pli->ddef->dcl[d].jenv;
      int Ld = jenv - ienv + 1;
      if (Ld <= 0 || ienv <= 0 || jenv <= 0 || ienv > window_len || jenv > window_len) continue;

      if (npdoms >= pdom_alloc) {
        pdom_alloc = pdom_alloc ? pdom_alloc * 2 : 1024;
        ESL_REALLOC(pdoms, sizeof(NHMMER_GPU_PDOM) * pdom_alloc);
      }

      NHMMER_GPU_PDOM *pd = &pdoms[npdoms];
      pd->win_ctx_idx = nwctxs - 1;
      pd->subseq  = subseq;
      pd->ienv    = ienv;
      pd->jenv    = jenv;
      pd->Ld      = Ld;
      pd->rfv_copy = NULL;

      p7_oprofile_ReconfigRestLength(w->om, Ld);
      if (w->pli_tmp->scores != NULL) {
        reparameterize_model(w->bg, w->om, w->pli_tmp->tmpseq, ienv, Ld,
                             w->pli_tmp->fwd_emissions_arr, w->pli_tmp->bg->f, w->pli_tmp->scores);
      }
      ESL_ALLOC(pd->rfv_copy, rfv_size);
      memcpy(pd->rfv_copy, w->om->rfv[0], rfv_size);

      if (w->pli_tmp->scores != NULL) {
        reparameterize_model(w->bg, w->om, NULL, 0, 0,
                             w->pli_tmp->fwd_emissions_arr, w->pli_tmp->bg->f, w->pli_tmp->scores);
      }
      npdoms++;
    }
    w->pli_tmp->tmpseq->dsq = dsq_holder;  /* restore after reparameterize loop */
    /* Update wctx ndom to actual count (after filtering Ld<=0) */
    wctxs[nwctxs-1].ndom = npdoms - wctxs[nwctxs-1].first_dom;

    if (overlap == -1 && i < w->nwindows - 1)
      overlap = ESL_MAX(0, (int)(window->n + window->length) - (int)w->windows[i+1].n);
    else
      overlap = 0;
    w->pli->ddef->ndom = 0;
  }

  if (npdoms == 0) goto CLEANUP;

  /* ==== PHASE 2: One GPU batch call ==== */
  {
    const uint8_t **h_dsq_ptrs = NULL;
    int            *h_lengths = NULL;
    const float   **h_rfv_ptrs = NULL;
    float          *h_envsc = NULL, *h_domcorr = NULL, *h_oasc = NULL;
    int8_t         *h_trace_st = NULL;
    int            *h_trace_k = NULL, *h_trace_i = NULL, *h_trace_N = NULL;
    float          *h_trace_pp = NULL;
    int            *h_statuses = NULL;
    int             max_Ld = 0, max_trace_len;
    char            errbuf[256];

    for (d = 0; d < npdoms; d++)
      if (pdoms[d].Ld > max_Ld) max_Ld = pdoms[d].Ld;
    max_trace_len = max_Ld * 4 + 20;

    ESL_ALLOC(h_dsq_ptrs, sizeof(uint8_t *) * npdoms);
    ESL_ALLOC(h_lengths,  sizeof(int)       * npdoms);
    ESL_ALLOC(h_rfv_ptrs, sizeof(float *)   * npdoms);
    ESL_ALLOC(h_envsc,    sizeof(float)     * npdoms);
    ESL_ALLOC(h_domcorr,  sizeof(float)     * npdoms);
    ESL_ALLOC(h_oasc,     sizeof(float)     * npdoms);
    ESL_ALLOC(h_statuses, sizeof(int)       * npdoms);
    ESL_ALLOC(h_trace_st, sizeof(int8_t)    * npdoms * max_trace_len);
    ESL_ALLOC(h_trace_k,  sizeof(int)       * npdoms * max_trace_len);
    ESL_ALLOC(h_trace_i,  sizeof(int)       * npdoms * max_trace_len);
    ESL_ALLOC(h_trace_pp, sizeof(float)     * npdoms * max_trace_len);
    ESL_ALLOC(h_trace_N,  sizeof(int)       * npdoms);

    for (d = 0; d < npdoms; d++) {
      h_dsq_ptrs[d] = pdoms[d].subseq + pdoms[d].ienv - 1;
      h_lengths[d]  = pdoms[d].Ld;
      h_rfv_ptrs[d] = pdoms[d].rfv_copy;
    }

#ifdef HMMER_THREADS
    if (w->gpu_domain_mutex) pthread_mutex_lock(w->gpu_domain_mutex);
#endif
    status = p7_cuda_DomainRescoreBatch(w->cuda_engine, w->cuda_msv,
                                         npdoms, h_dsq_ptrs, h_lengths,
                                         h_rfv_ptrs, orig_rfv,
                                         Q, Kp, nj,
                                         h_envsc, h_domcorr, h_oasc,
                                         h_trace_st, h_trace_k, h_trace_i,
                                         h_trace_pp, h_trace_N, max_trace_len,
                                         h_statuses, errbuf, sizeof(errbuf));
#ifdef HMMER_THREADS
    if (w->gpu_domain_mutex) pthread_mutex_unlock(w->gpu_domain_mutex);
#endif
    if (status != eslOK) {
      free(h_dsq_ptrs); free(h_lengths); free(h_rfv_ptrs);
      free(h_envsc); free(h_domcorr); free(h_oasc); free(h_statuses);
      free(h_trace_st); free(h_trace_k); free(h_trace_i); free(h_trace_pp); free(h_trace_N);
      goto CLEANUP;
    }

    /* ==== PHASE 3: Unpack results, batch trim, report hits ==== */

    /* Phase 3a: Build traces/alidisplays, identify trim candidates */
    typedef struct {
      P7_ALIDISPLAY *ad;
      int            ienv, jenv, Ld;
      float          envsc, domcorr;
      int            wi_idx;           /* index into wctxs */
      int            gd;               /* global domain index */
      int            needs_trim;
      int            trim_ienv, trim_jenv, trim_Ld;
      float         *trim_rfv;         /* reparameterized rfv for trim envelope */
    } NHMMER_GPU_DOM_RESULT;

    NHMMER_GPU_DOM_RESULT *dom_results = NULL;
    int nresults = 0, result_alloc = 0;

    for (int wi = 0; wi < nwctxs; wi++) {
      NHMMER_GPU_WCTX *wc = &wctxs[wi];
      ESL_DSQ *subseq2 = wc->subseq;
      int window_len2 = wc->window_len;

      for (int dd = 0; dd < wc->ndom; dd++) {
        int gd = wc->first_dom + dd;
        NHMMER_GPU_PDOM *pd = &pdoms[gd];
        int Ld = pd->Ld;
        int ienv = pd->ienv;
        int jenv = pd->jenv;

        if (h_statuses[gd] != eslOK) continue;
        if (h_trace_N[gd] <= 0 || h_trace_N[gd] > max_trace_len) continue;

        p7_trace_Reuse(w->pli->ddef->tr);
        nhmmer_gpu_trace_from_gpu(w->pli->ddef->tr,
                                  h_trace_st + (size_t)gd * max_trace_len,
                                  h_trace_k  + (size_t)gd * max_trace_len,
                                  h_trace_i  + (size_t)gd * max_trace_len,
                                  h_trace_pp + (size_t)gd * max_trace_len,
                                  h_trace_N[gd], w->om->M, Ld);

        ESL_DSQ *dsq_save = w->pli_tmp->tmpseq->dsq;
        int64_t  n_save   = w->pli_tmp->tmpseq->n;
        w->pli_tmp->tmpseq->dsq = subseq2 + ienv - 1;
        w->pli_tmp->tmpseq->n = Ld;
        p7_oprofile_ReconfigRestLength(w->om, Ld);
        P7_ALIDISPLAY *ad = p7_alidisplay_Create(w->pli->ddef->tr, 0, w->om, w->pli_tmp->tmpseq, NULL);
        w->pli_tmp->tmpseq->dsq = dsq_save;
        w->pli_tmp->tmpseq->n = n_save;

        if (ad == NULL) continue;
        if (ad->sqfrom <= 0 || ad->sqto <= 0 || ad->sqfrom > Ld || ad->sqto > Ld) {
          p7_alidisplay_Destroy(ad);
          continue;
        }

        if (nresults >= result_alloc) {
          result_alloc = result_alloc ? result_alloc * 2 : 1024;
          ESL_REALLOC(dom_results, sizeof(NHMMER_GPU_DOM_RESULT) * result_alloc);
        }
        NHMMER_GPU_DOM_RESULT *dr = &dom_results[nresults];
        dr->ad       = ad;
        dr->ienv     = ienv;
        dr->jenv     = jenv;
        dr->Ld       = Ld;
        dr->envsc    = h_envsc[gd];
        dr->domcorr  = h_domcorr[gd];
        dr->wi_idx   = wi;
        dr->gd       = gd;
        dr->needs_trim = 0;
        dr->trim_rfv = NULL;

        if (1 < (int)ad->sqfrom - 20 || Ld > (int)ad->sqto + 20) {
          int new_env_start = ESL_MAX(1, (int)ad->sqfrom - 20);
          int new_env_end   = ESL_MIN(Ld, (int)ad->sqto + 20);
          dr->trim_ienv = ienv + new_env_start - 1;
          dr->trim_jenv = ienv + new_env_end - 1;
          dr->trim_Ld   = dr->trim_jenv - dr->trim_ienv + 1;
          dr->needs_trim = 1;

          p7_oprofile_ReconfigRestLength(w->om, dr->trim_Ld);
          if (w->pli_tmp->scores != NULL) {
            w->pli_tmp->tmpseq->dsq = subseq2;
            w->pli_tmp->tmpseq->n   = window_len2;
            reparameterize_model(w->bg, w->om, w->pli_tmp->tmpseq, dr->trim_ienv, dr->trim_Ld,
                                 w->pli_tmp->fwd_emissions_arr, w->pli_tmp->bg->f, w->pli_tmp->scores);
            w->pli_tmp->tmpseq->dsq = dsq_save;
          }
          ESL_ALLOC(dr->trim_rfv, rfv_size);
          memcpy(dr->trim_rfv, w->om->rfv[0], rfv_size);
          if (w->pli_tmp->scores != NULL) {
            reparameterize_model(w->bg, w->om, NULL, 0, 0,
                                 w->pli_tmp->fwd_emissions_arr, w->pli_tmp->bg->f, w->pli_tmp->scores);
          }
        }
        nresults++;
      }
    }

    /* Phase 3b: Single GPU batch call for all trim domains */
    int ntrim = 0;
    for (int r = 0; r < nresults; r++) if (dom_results[r].needs_trim) ntrim++;

    if (ntrim > 0) {
      const uint8_t **trim_dsq_ptrs = NULL;
      int            *trim_lengths  = NULL;
      const float   **trim_rfv_ptrs = NULL;
      float          *trim_envsc    = NULL;
      float          *trim_domcorr  = NULL;
      float          *trim_oasc     = NULL;
      int            *trim_statuses = NULL;
      int8_t         *trim_trace_st = NULL;
      int            *trim_trace_k  = NULL;
      int            *trim_trace_i  = NULL;
      float          *trim_trace_pp = NULL;
      int            *trim_trace_N  = NULL;
      int            *trim_map      = NULL;  /* maps trim index -> dom_results index */

      int trim_max_Ld = 0;
      for (int r = 0; r < nresults; r++)
        if (dom_results[r].needs_trim && dom_results[r].trim_Ld > trim_max_Ld)
          trim_max_Ld = dom_results[r].trim_Ld;
      int trim_max_trace = trim_max_Ld * 4 + 20;

      ESL_ALLOC(trim_dsq_ptrs, sizeof(uint8_t *) * ntrim);
      ESL_ALLOC(trim_lengths,  sizeof(int)       * ntrim);
      ESL_ALLOC(trim_rfv_ptrs, sizeof(float *)   * ntrim);
      ESL_ALLOC(trim_envsc,    sizeof(float)     * ntrim);
      ESL_ALLOC(trim_domcorr,  sizeof(float)     * ntrim);
      ESL_ALLOC(trim_oasc,     sizeof(float)     * ntrim);
      ESL_ALLOC(trim_statuses, sizeof(int)       * ntrim);
      ESL_ALLOC(trim_trace_st, sizeof(int8_t)    * (size_t)ntrim * trim_max_trace);
      ESL_ALLOC(trim_trace_k,  sizeof(int)       * (size_t)ntrim * trim_max_trace);
      ESL_ALLOC(trim_trace_i,  sizeof(int)       * (size_t)ntrim * trim_max_trace);
      ESL_ALLOC(trim_trace_pp, sizeof(float)     * (size_t)ntrim * trim_max_trace);
      ESL_ALLOC(trim_trace_N,  sizeof(int)       * ntrim);
      ESL_ALLOC(trim_map,      sizeof(int)       * ntrim);

      int ti = 0;
      for (int r = 0; r < nresults; r++) {
        if (!dom_results[r].needs_trim) continue;
        NHMMER_GPU_DOM_RESULT *dr = &dom_results[r];
        NHMMER_GPU_WCTX *wc = &wctxs[dr->wi_idx];
        trim_dsq_ptrs[ti] = wc->subseq + dr->trim_ienv - 1;
        trim_lengths[ti]  = dr->trim_Ld;
        trim_rfv_ptrs[ti] = dr->trim_rfv;
        trim_map[ti]      = r;
        ti++;
      }

#ifdef HMMER_THREADS
      if (w->gpu_domain_mutex) pthread_mutex_lock(w->gpu_domain_mutex);
#endif
      int trim_status = p7_cuda_DomainRescoreBatch(w->cuda_engine, w->cuda_msv,
                                                    ntrim, trim_dsq_ptrs, trim_lengths,
                                                    trim_rfv_ptrs, orig_rfv,
                                                    Q, Kp, nj,
                                                    trim_envsc, trim_domcorr, trim_oasc,
                                                    trim_trace_st, trim_trace_k, trim_trace_i,
                                                    trim_trace_pp, trim_trace_N, trim_max_trace,
                                                    trim_statuses, errbuf, sizeof(errbuf));
#ifdef HMMER_THREADS
      if (w->gpu_domain_mutex) pthread_mutex_unlock(w->gpu_domain_mutex);
#endif

      /* Apply trim results */
      if (trim_status == eslOK) {
        for (ti = 0; ti < ntrim; ti++) {
          if (trim_statuses[ti] != eslOK) continue;
          if (trim_trace_N[ti] <= 0 || trim_trace_N[ti] > trim_max_trace) continue;

          int r = trim_map[ti];
          NHMMER_GPU_DOM_RESULT *dr = &dom_results[r];
          NHMMER_GPU_WCTX *wc = &wctxs[dr->wi_idx];

          p7_trace_Reuse(w->pli->ddef->tr);
          nhmmer_gpu_trace_from_gpu(w->pli->ddef->tr,
                                    trim_trace_st + (size_t)ti * trim_max_trace,
                                    trim_trace_k  + (size_t)ti * trim_max_trace,
                                    trim_trace_i  + (size_t)ti * trim_max_trace,
                                    trim_trace_pp + (size_t)ti * trim_max_trace,
                                    trim_trace_N[ti], w->om->M, dr->trim_Ld);

          ESL_DSQ *dsq_save = w->pli_tmp->tmpseq->dsq;
          int64_t  n_save   = w->pli_tmp->tmpseq->n;
          w->pli_tmp->tmpseq->dsq = wc->subseq + dr->trim_ienv - 1;
          w->pli_tmp->tmpseq->n = dr->trim_Ld;
          p7_oprofile_ReconfigRestLength(w->om, dr->trim_Ld);
          P7_ALIDISPLAY *trim_ad = p7_alidisplay_Create(w->pli->ddef->tr, 0, w->om, w->pli_tmp->tmpseq, NULL);
          w->pli_tmp->tmpseq->dsq = dsq_save;
          w->pli_tmp->tmpseq->n = n_save;

          if (trim_ad != NULL) {
            p7_alidisplay_Destroy(dr->ad);
            dr->ad    = trim_ad;
            dr->ienv  = dr->trim_ienv;
            dr->jenv  = dr->trim_jenv;
            dr->Ld    = dr->trim_Ld;
            dr->envsc = trim_envsc[ti];
            dr->domcorr = trim_domcorr[ti];
          }
        }
      }

      free(trim_dsq_ptrs); free(trim_lengths); free(trim_rfv_ptrs);
      free(trim_envsc); free(trim_domcorr); free(trim_oasc); free(trim_statuses);
      free(trim_trace_st); free(trim_trace_k); free(trim_trace_i);
      free(trim_trace_pp); free(trim_trace_N); free(trim_map);
    }

    /* Phase 3c: Report hits */
    for (int r = 0; r < nresults; r++) {
      NHMMER_GPU_DOM_RESULT *dr = &dom_results[r];
      NHMMER_GPU_WCTX *wc = &wctxs[dr->wi_idx];
      P7_ALIDISPLAY *ad = dr->ad;
      int ienv = dr->ienv;
      int jenv = dr->jenv;
      int Ld   = dr->Ld;
      float envsc = dr->envsc;
      float domcorrection = dr->domcorr;
      window = wc->window;
      ESL_DSQ *subseq2 = wc->subseq;

      if (domcorrection < envsc) envsc = domcorrection;

      float dom_envsc = envsc;
      float dom_domcorr = domcorrection - envsc;
      float dom_oasc = h_oasc[dr->gd];
      int   env_len = jenv - ienv + 1;
      int   ali_len = (int)ad->sqto - (int)ad->sqfrom + 1;
      float bitscore = dom_envsc;

      if (ali_len < 8) { p7_alidisplay_Destroy(ad); continue; }

      bitscore -= 2 * log(2. / (env_len + 2));
      bitscore += 2 * log(2. / (w->om->max_length + 2));
      bitscore -= (env_len - ali_len) * log((float)env_len / (env_len + 2));
      bitscore += (ESL_MAX(w->om->max_length, env_len) - ali_len) * log((float)w->om->max_length / (float)(w->om->max_length + 2));

      float nullsc2;
      p7_bg_SetLength(w->bg, ESL_MAX(w->om->max_length, env_len));
      p7_bg_NullOne(w->bg, subseq2, ESL_MAX(w->om->max_length, env_len), &nullsc2);
      float dom_score = (bitscore - nullsc2) / eslCONST_LOG2;
      double dom_lnP = esl_exp_logsurv(dom_score, w->om->evparam[p7_FTAU], w->om->evparam[p7_FLAMBDA]);

      float *scores_per_pos = NULL;
      if (w->pli->do_alignment_score_calc) {
        P7_DOMAIN tmp_dom;
        memset(&tmp_dom, 0, sizeof(tmp_dom));
        tmp_dom.ad = ad;
        tmp_dom.iali = (int)ad->sqfrom;
        tmp_dom.jali = (int)ad->sqto;
        nhmmer_gpu_computeAliScores(&tmp_dom, subseq2 + ienv - 1, w->scoredata, Kp);
        scores_per_pos = tmp_dom.scores_per_pos;
      }

      P7_HIT *hit = NULL;
      p7_tophits_CreateNextHit(w->th, &hit);
      hit->ndom        = 1;
      hit->best_domain = 0;
      hit->window_length = w->om->max_length;
      hit->seqidx = w->seq_id;
      hit->subseq_start = (int64_t)w->sq->start;

      ESL_ALLOC(hit->dcl, sizeof(P7_DOMAIN));
      memset(&hit->dcl[0], 0, sizeof(P7_DOMAIN));
      hit->dcl[0].ad       = ad;
      hit->dcl[0].envsc    = dom_envsc;
      hit->dcl[0].domcorrection = dom_domcorr;
      hit->dcl[0].oasc     = dom_oasc;
      hit->dcl[0].ienv     = ienv;
      hit->dcl[0].jenv     = jenv;
      hit->dcl[0].iali     = (int)ad->sqfrom;
      hit->dcl[0].jali     = (int)ad->sqto;
      hit->dcl[0].dombias  = dom_domcorr;
      hit->dcl[0].bitscore = dom_score;
      hit->dcl[0].lnP      = dom_lnP;
      hit->dcl[0].is_reported = FALSE;
      hit->dcl[0].is_included = FALSE;
      hit->dcl[0].scores_per_pos = scores_per_pos;
      ad->L = -1;

      int ienv_off = ienv - 1;
      hit->dcl[0].iali += ienv_off;
      hit->dcl[0].jali += ienv_off;
      hit->dcl[0].ad->sqfrom += ienv_off;
      hit->dcl[0].ad->sqto   += ienv_off;

      if (w->complementarity == p7_NOCOMPLEMENT) {
        hit->dcl[0].ienv       += (int64_t)w->sq->start + (int)window->n - 2;
        hit->dcl[0].jenv       += (int64_t)w->sq->start + (int)window->n - 2;
        hit->dcl[0].iali       += (int64_t)w->sq->start + (int)window->n - 2;
        hit->dcl[0].jali       += (int64_t)w->sq->start + (int)window->n - 2;
        hit->dcl[0].ad->sqfrom += (int64_t)w->sq->start + (int)window->n - 2;
        hit->dcl[0].ad->sqto   += (int64_t)w->sq->start + (int)window->n - 2;
      } else {
        hit->dcl[0].ienv       = (int64_t)w->sq->start - ((int)window->n + hit->dcl[0].ienv) + 2;
        hit->dcl[0].jenv       = (int64_t)w->sq->start - ((int)window->n + hit->dcl[0].jenv) + 2;
        hit->dcl[0].iali       = (int64_t)w->sq->start - ((int)window->n + hit->dcl[0].iali) + 2;
        hit->dcl[0].jali       = (int64_t)w->sq->start - ((int)window->n + hit->dcl[0].jali) + 2;
        hit->dcl[0].ad->sqfrom = (int64_t)w->sq->start - ((int)window->n + hit->dcl[0].ad->sqfrom) + 2;
        hit->dcl[0].ad->sqto   = (int64_t)w->sq->start - ((int)window->n + hit->dcl[0].ad->sqto) + 2;
      }

      float pre_bitscore = bitscore / eslCONST_LOG2;
      hit->pre_score = pre_bitscore;
      hit->pre_lnP   = esl_exp_logsurv(pre_bitscore, w->om->evparam[p7_FTAU], w->om->evparam[p7_FLAMBDA]);
      hit->sum_score = hit->score = dom_score;
      hit->sum_lnP   = hit->lnP   = dom_lnP;

      if (w->pli->mode == p7_SEARCH_SEQS) {
        esl_strdup(w->sq->name, -1, &(hit->name));
        if (w->sq->acc[0]  != '\0') esl_strdup(w->sq->acc,  -1, &(hit->acc));
        if (w->sq->desc[0] != '\0') esl_strdup(w->sq->desc, -1, &(hit->desc));
      } else {
        esl_strdup(w->om->name, -1, &(hit->name));
        esl_strdup(w->om->acc,  -1, &(hit->acc));
        esl_strdup(w->om->desc, -1, &(hit->desc));
      }

      if (w->pli->use_bit_cutoffs) {
        if (p7_pli_TargetReportable(w->pli, hit->score, hit->lnP)) {
          hit->flags |= p7_IS_REPORTED;
          if (p7_pli_TargetIncludable(w->pli, hit->score, hit->lnP))
            hit->flags |= p7_IS_INCLUDED;
        }
        if (p7_pli_DomainReportable(w->pli, hit->dcl[0].bitscore, hit->dcl[0].lnP)) {
          hit->dcl[0].is_reported = TRUE;
          if (p7_pli_DomainIncludable(w->pli, hit->dcl[0].bitscore, hit->dcl[0].lnP))
            hit->dcl[0].is_included = TRUE;
        }
      }
    }

    if (dom_results) {
      for (int r = 0; r < nresults; r++) if (dom_results[r].trim_rfv) free(dom_results[r].trim_rfv);
      free(dom_results);
    }

    free(h_dsq_ptrs); free(h_lengths); free(h_rfv_ptrs);
    free(h_envsc); free(h_domcorr); free(h_oasc); free(h_statuses);
    free(h_trace_st); free(h_trace_k); free(h_trace_i); free(h_trace_pp); free(h_trace_N);
  }

CLEANUP:
  p7_oprofile_ReconfigRestLength(w->om, orig_L);
  if (orig_rfv) free(orig_rfv);
  if (pdoms) {
    for (d = 0; d < npdoms; d++) if (pdoms[d].rfv_copy) free(pdoms[d].rfv_copy);
    free(pdoms);
  }
  if (wctxs) free(wctxs);
  return;

ERROR:
  w->status = eslEMEM;
  goto CLEANUP;
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
  if (w->use_gpu_fb)
    nhmmer_gpu_worker_process_post_vit_gpu(w);
  else
    nhmmer_gpu_worker_process_post_vit(w);
  return NULL;
}
#endif

/* GPU Forward pre-filter: uses GPU Forward score for F3 gating, then dispatches
 * surviving windows to CPU worker threads for ForwardParser + domaindef.
 * Eliminates 40-60% of windows before CPU work, while preserving correct
 * overlap tracking and hit reporting from the existing worker path. */
static int
nhmmer_gpu_forward_prefilter(NHMMER_GPU_INFO *info, const ESL_SQ *sq, int64_t seq_id,
                             int complementarity, P7_HMM_WINDOW *windows, int nwindows,
                             P7_HMM_WINDOW **ret_survivors, int *ret_nsurv,
                             float **ret_surv_xf, float **ret_surv_fwdsc,
                             char *errbuf, int errbuf_size)
{
  int status = eslOK;
  P7_OPROFILE *om = info->om;
  P7_BG *bg = info->bg;
  P7_PIPELINE *pli = info->pli;
  int i;

  NHMMER_GPU_WINDOW_BATCH wb;
  float *scores = NULL;
  int *statuses = NULL;
  int *seqidx = NULL;
  float *xf = NULL;
  size_t *x_offsets = NULL;
  size_t total_xcells = 0;
  P7_HMM_WINDOW *survivors = NULL;
  float *surv_xf = NULL;
  float *surv_fwdsc = NULL;
  size_t *surv_x_offsets = NULL;
  int nsurv = 0;

  *ret_survivors = NULL;
  *ret_nsurv = 0;
  *ret_surv_xf = NULL;
  *ret_surv_fwdsc = NULL;

  if (nwindows == 0) return eslOK;

  memset(&wb, 0, sizeof(wb));
  status = nhmmer_gpu_window_batch_init(&wb, nwindows);
  if (status != eslOK) goto ERROR;

  P7_HMM_WINDOWLIST tmp_wl;
  tmp_wl.windows = windows;
  tmp_wl.count = nwindows;
  tmp_wl.size = nwindows;
  status = nhmmer_gpu_window_batch_pack(&wb, sq, &tmp_wl);
  if (status != eslOK) { nhmmer_gpu_window_batch_free(&wb); goto ERROR; }

  ESL_ALLOC(scores, sizeof(float) * 2 * nwindows);
  ESL_ALLOC(statuses, sizeof(int) * 2 * nwindows);
  ESL_ALLOC(seqidx, sizeof(int) * nwindows);
  ESL_ALLOC(x_offsets, sizeof(size_t) * nwindows);

  for (i = 0; i < nwindows; i++) {
    seqidx[i] = i;
    x_offsets[i] = total_xcells;
    total_xcells += (size_t)(windows[i].length + 1) * p7X_NXCELLS;
  }

  ESL_ALLOC(xf, sizeof(float) * total_xcells);

  /* GPU Forward parser: compute Forward scores AND save xmx for all windows */
  status = p7_cuda_ForwardParserDsqdataSubset(info->cuda_engine, info->cuda_msv,
                                               &wb.chu, seqidx, nwindows,
                                               x_offsets, total_xcells,
                                               xf, scores, statuses,
                                               errbuf, errbuf_size);
  nhmmer_gpu_window_batch_free(&wb);
  if (status != eslOK) goto ERROR;

  /* F3 gating: keep windows where GPU Forward score passes threshold.
   * Track survivor indices for xf compaction. */
  ESL_ALLOC(survivors, sizeof(P7_HMM_WINDOW) * nwindows);
  ESL_ALLOC(surv_x_offsets, sizeof(size_t) * nwindows);
  ESL_ALLOC(surv_fwdsc, sizeof(float) * nwindows);
  int *surv_src_idx = NULL;
  ESL_ALLOC(surv_src_idx, sizeof(int) * nwindows);
  size_t surv_total_xcells = 0;

  for (i = 0; i < nwindows; i++) {
    if (statuses[2*i] != eslOK) continue;

    int window_len = windows[i].length;
    ESL_DSQ *subseq = sq->dsq + windows[i].n - 1;
    float fwdsc = scores[2*i];
    float nullsc, bias_filtersc, filtersc, seq_score;
    double P_val;

    int F3_L = ESL_MIN(window_len, pli->B3);
    p7_bg_SetLength(bg, window_len);
    p7_bg_NullOne(bg, subseq, window_len, &nullsc);
    if (pli->do_biasfilter) {
      p7_bg_FilterScore(bg, subseq, window_len, &bias_filtersc);
      bias_filtersc -= nullsc;
    } else {
      bias_filtersc = 0;
    }
    filtersc = nullsc + (bias_filtersc * (F3_L > window_len ? 1.0f : (float)F3_L / window_len));
    seq_score = (fwdsc - filtersc) / eslCONST_LOG2;
    P_val = esl_exp_surv(seq_score, om->evparam[p7_FTAU], om->evparam[p7_FLAMBDA]);
    if (P_val > pli->F3 * 2.0) continue;

    surv_src_idx[nsurv] = i;
    surv_fwdsc[nsurv] = fwdsc;
    survivors[nsurv] = windows[i];
    surv_x_offsets[nsurv] = surv_total_xcells;
    surv_total_xcells += (size_t)(window_len + 1) * p7X_NXCELLS;
    nsurv++;
  }

  /* Compact xf: copy survivor rows into contiguous surv_xf buffer */
  if (nsurv > 0 && surv_total_xcells > 0) {
    ESL_ALLOC(surv_xf, sizeof(float) * surv_total_xcells);
    for (int si = 0; si < nsurv; si++) {
      int src = surv_src_idx[si];
      size_t xcells = (size_t)(windows[src].length + 1) * p7X_NXCELLS;
      memcpy(surv_xf + surv_x_offsets[si], xf + x_offsets[src], sizeof(float) * xcells);
    }
  }

  *ret_survivors = survivors;
  *ret_nsurv = nsurv;
  *ret_surv_xf = surv_xf;
  *ret_surv_fwdsc = surv_fwdsc;
  free(surv_x_offsets);
  free(surv_src_idx);
  free(xf);
  free(x_offsets);
  free(scores);
  free(statuses);
  free(seqidx);
  return eslOK;

ERROR:
  free(survivors);
  free(surv_xf);
  free(surv_fwdsc);
  free(surv_x_offsets);
  free(surv_src_idx);
  free(xf);
  free(x_offsets);
  free(scores);
  free(statuses);
  free(seqidx);
  return status;
}

#define NHMMER_GPU_FB_MAX_XCELLS (128UL * 1024 * 1024)

static int
nhmmer_gpu_run_fb_parser_batch(NHMMER_GPU_INFO *info, const ESL_SQ *sq,
                               P7_HMM_WINDOW *windows, int nwindows,
                               float *precomputed_xf, const float *precomputed_fwdsc,
                               float **ret_xf, float **ret_xb,
                               float **ret_scores, int **ret_statuses,
                               size_t **ret_x_offsets, int **ret_L_eff,
                               char *errbuf, int errbuf_size)
{
  int         status = eslOK;
  P7_OPROFILE *om = info->om;
  int         i;
  size_t      total_xcells = 0;
  int        *L_eff     = NULL;
  size_t     *x_offsets = NULL;
  float      *xf        = NULL;
  float      *xb        = NULL;
  float      *fb_scores = NULL;
  int        *fb_statuses = NULL;
  int         backward_only = (precomputed_xf != NULL);

  *ret_xf = *ret_xb = NULL;
  *ret_scores = NULL;
  *ret_statuses = NULL;
  *ret_x_offsets = NULL;
  *ret_L_eff = NULL;

  if (nwindows == 0) return eslOK;

  ESL_ALLOC(L_eff,       sizeof(int)    * nwindows);
  ESL_ALLOC(x_offsets,   sizeof(size_t) * nwindows);
  ESL_ALLOC(fb_scores,   sizeof(float)  * 2 * nwindows);
  ESL_ALLOC(fb_statuses, sizeof(int)    * 2 * nwindows);

  for (i = 0; i < nwindows; i++) {
    L_eff[i]     = (int)windows[i].length;
    x_offsets[i] = total_xcells;
    total_xcells += (size_t)(L_eff[i] + 1) * p7X_NXCELLS;
  }

  if (backward_only) {
    xf = precomputed_xf;
    for (i = 0; i < nwindows; i++)
      fb_scores[i * 2] = precomputed_fwdsc[i];
  } else {
    ESL_ALLOC(xf, sizeof(float) * total_xcells);
  }
  ESL_ALLOC(xb, sizeof(float) * total_xcells);

  /* Sub-batch loop */
  int sub_start = 0;
  while (sub_start < nwindows) {
    int sub_end = sub_start;
    size_t sub_xcells = 0;
    while (sub_end < nwindows) {
      size_t w_xcells = (size_t)(L_eff[sub_end] + 1) * p7X_NXCELLS;
      if (sub_end > sub_start && sub_xcells + w_xcells > NHMMER_GPU_FB_MAX_XCELLS) break;
      sub_xcells += w_xcells;
      sub_end++;
    }
    int sub_n = sub_end - sub_start;

    NHMMER_GPU_WINDOW_BATCH wb;
    memset(&wb, 0, sizeof(wb));
    status = nhmmer_gpu_window_batch_init(&wb, sub_n);
    if (status != eslOK) goto ERROR;

    for (int k = 0; k < sub_n; k++) {
      int wi = sub_start + k;
      wb.dsq_ptrs[k] = sq->dsq + windows[wi].n - 1;
      wb.lengths[k]  = L_eff[wi];
      wb.names[k]    = nhmmer_empty_str;
      wb.accs[k]     = nhmmer_empty_str;
      wb.descs[k]    = nhmmer_empty_str;
      wb.taxids[k]   = -1;
      wb.win_idx[k]  = k;
    }
    wb.nwindows    = sub_n;
    wb.chu.i0      = 0;
    wb.chu.N       = sub_n;
    wb.chu.dsq     = wb.dsq_ptrs;
    wb.chu.L       = wb.lengths;
    wb.chu.name    = wb.names;
    wb.chu.acc     = wb.accs;
    wb.chu.desc    = wb.descs;
    wb.chu.taxid   = wb.taxids;
    wb.chu.smem    = NULL;
    wb.chu.psq     = NULL;
    wb.chu.pn      = 0;
    wb.chu.metadata = NULL;
    wb.chu.mdalloc  = 0;

    int *local_seqidx = NULL;
    size_t *local_offsets = NULL;
    ESL_ALLOC(local_seqidx,  sizeof(int)    * sub_n);
    ESL_ALLOC(local_offsets,  sizeof(size_t) * sub_n);
    for (int k = 0; k < sub_n; k++) {
      local_seqidx[k]  = k;
      local_offsets[k] = x_offsets[sub_start + k] - x_offsets[sub_start];
    }

    size_t sub_total = (sub_end < nwindows)
                     ? x_offsets[sub_end] - x_offsets[sub_start]
                     : total_xcells       - x_offsets[sub_start];

    if (backward_only) {
      status = p7_cuda_BackwardParserDsqdataSubset(
          info->cuda_engine, info->cuda_msv,
          &wb.chu, local_seqidx, sub_n,
          local_offsets, sub_total,
          xf + x_offsets[sub_start],
          xb + x_offsets[sub_start],
          fb_scores  + sub_start * 2,
          fb_statuses + sub_start * 2,
          errbuf, errbuf_size);
    } else {
      status = p7_cuda_ForwardBackwardParserDsqdataSubset(
          info->cuda_engine, info->cuda_msv,
          &wb.chu, local_seqidx, sub_n,
          local_offsets, sub_total,
          xf + x_offsets[sub_start],
          xb + x_offsets[sub_start],
          fb_scores  + sub_start * 2,
          fb_statuses + sub_start * 2,
          errbuf, errbuf_size);
    }

    free(local_seqidx);
    free(local_offsets);
    nhmmer_gpu_window_batch_free(&wb);
    if (status != eslOK) goto ERROR;

    sub_start = sub_end;
  }

  *ret_xf        = xf;
  *ret_xb        = xb;
  *ret_scores    = fb_scores;
  *ret_statuses  = fb_statuses;
  *ret_x_offsets = x_offsets;
  *ret_L_eff     = L_eff;
  return eslOK;

ERROR:
  if (!backward_only) free(xf);
  free(xb); free(fb_scores); free(fb_statuses);
  free(x_offsets); free(L_eff);
  return status;
}

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
                                     om->nj,
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

  struct timespec ts0, ts1;

  P7_CUDA_LT_WINDOW *gpu_windows = NULL;
  int gpu_nwindows = 0;

  P7_HMM_WINDOWLIST  msv_windowlist;
  msv_windowlist.windows = NULL;

  clock_gettime(CLOCK_MONOTONIC, &ts0);
  status = p7_cuda_SSVLongtarget(info->cuda_engine, info->cuda_msv,
                                 sq->dsq, L,
                                 info->scoredata->ssv_scores, om->abc->Kp,
                                 sc_thresh, om->scale_b,
                                 chunk_size, overlap,
                                 &gpu_windows, &gpu_nwindows,
                                 errbuf, errbuf_size);
  if (status != eslOK) return status;
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_ssv += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  if (gpu_nwindows == 0) { free(gpu_windows); return eslOK; }

  clock_gettime(CLOCK_MONOTONIC, &ts0);
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
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_merge += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  if (msv_windowlist.count == 0) {
    free(msv_windowlist.windows);
    return eslOK;
  }

  /* GPU batch SSV/bias/F1 gating: filter merged windows before thread dispatch */
  NHMMER_GPU_WINDOW_BATCH wb;
  memset(&wb, 0, sizeof(wb));

  clock_gettime(CLOCK_MONOTONIC, &ts0);
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
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_batch_filter += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  /* GPU scanning Viterbi longtarget: replaces CPU scanning Viterbi in workers */
  if (info->do_gpu_vit_lt && msv_windowlist.count > 0) {
    P7_HMM_WINDOWLIST vit_wl;
    vit_wl.windows = NULL;

    clock_gettime(CLOCK_MONOTONIC, &ts0);
    status = nhmmer_gpu_viterbi_longtarget(info, sq, &msv_windowlist, &vit_wl,
                                           errbuf, errbuf_size);
    if (status != eslOK) {
      fprintf(stderr, "GPU scanning Viterbi failed: %s\n", errbuf);
      if (vit_wl.windows) free(vit_wl.windows);
      goto ERROR;
    }

    free(msv_windowlist.windows);
    msv_windowlist.windows = NULL;
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    info->t_vit_lt += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

    if (vit_wl.count == 0) {
      if (vit_wl.windows) free(vit_wl.windows);
      return eslOK;
    }

    /* Dispatch sub-windows to post-Viterbi workers (Forward/Backward only) */
    info->n_vit_lt_windows_out += vit_wl.count;
    info->n_post_vit_windows   += vit_wl.count;

    /* GPU Forward pre-filter: eliminate windows failing F3 before CPU dispatch */
    P7_HMM_WINDOW *fwd_survivors = NULL;
    int nfwd_surv = 0;
    int nvit_windows = vit_wl.count;
    float  *prefilter_xf = NULL;
    float  *prefilter_fwdsc = NULL;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    status = nhmmer_gpu_forward_prefilter(info, sq, seq_id, complementarity,
                                          vit_wl.windows, vit_wl.count,
                                          &fwd_survivors, &nfwd_surv,
                                          &prefilter_xf, &prefilter_fwdsc,
                                          errbuf, errbuf_size);
    free(vit_wl.windows);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    info->t_fwd_prefilter += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    if (status != eslOK) { free(fwd_survivors); free(prefilter_xf); free(prefilter_fwdsc); goto ERROR; }

    /* Log window stats for profiling */
    {
      int64_t total_len = 0;
      int max_len = 0;
      for (int wi = 0; wi < nfwd_surv; wi++) {
        total_len += fwd_survivors[wi].length;
        if ((int)fwd_survivors[wi].length > max_len) max_len = (int)fwd_survivors[wi].length;
      }
    }

    if (nfwd_surv == 0) {
      free(fwd_survivors);
      free(prefilter_xf);
      free(prefilter_fwdsc);
      return eslOK;
    }

    /* GPU FB parser batch: Backward-only using prefilter xf, or full FB if no prefilter xf */
    float   *gpu_xf = NULL, *gpu_xb = NULL, *gpu_fb_scores = NULL;
    int     *gpu_fb_statuses = NULL, *gpu_fb_L_eff = NULL;
    size_t  *gpu_fb_x_offsets = NULL;
    int      gpu_fb_ok = FALSE;

    clock_gettime(CLOCK_MONOTONIC, &ts0);
    status = nhmmer_gpu_run_fb_parser_batch(info, sq, fwd_survivors, nfwd_surv,
                                             prefilter_xf, prefilter_fwdsc,
                                             &gpu_xf, &gpu_xb, &gpu_fb_scores, &gpu_fb_statuses,
                                             &gpu_fb_x_offsets, &gpu_fb_L_eff, errbuf, errbuf_size);
    free(prefilter_fwdsc); prefilter_fwdsc = NULL;
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    info->t_gpu_fb_parser += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    if (status == eslOK && gpu_xf != NULL)
      gpu_fb_ok = TRUE;
    else
      status = eslOK;  /* non-fatal: fall back to CPU FB */

    /* Dispatch survivors to worker threads */
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    int ncpus_vlt = info->ncpus;
    if (ncpus_vlt < 1) ncpus_vlt = 1;
    if (ncpus_vlt > nfwd_surv) ncpus_vlt = nfwd_surv;

#ifdef HMMER_THREADS
    if (ncpus_vlt > 1) {
      NHMMER_GPU_WORKER *workers = NULL;
      pthread_t         *threads = NULL;
      int                nworkers = ncpus_vlt;
      int                windows_per_thread = nfwd_surv / nworkers;
      int                remainder = nfwd_surv % nworkers;
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
        workers[i].windows         = fwd_survivors + offset;
        workers[i].use_gpu_fb      = gpu_fb_ok;
        workers[i].gpu_xf          = gpu_xf;
        workers[i].gpu_xb          = gpu_xb;
        workers[i].gpu_scores      = gpu_fb_ok ? gpu_fb_scores + offset * 2 : NULL;
        workers[i].gpu_statuses    = gpu_fb_ok ? gpu_fb_statuses + offset * 2 : NULL;
        workers[i].gpu_x_offsets   = gpu_fb_ok ? gpu_fb_x_offsets + offset : NULL;
        workers[i].gpu_L_eff       = gpu_fb_ok ? gpu_fb_L_eff + offset : NULL;
        offset += workers[i].nwindows;
      }

      for (i = 1; i < nworkers; i++)
        pthread_create(&threads[i], NULL, nhmmer_gpu_thread_func_post_vit, &workers[i]);

      if (workers[0].use_gpu_fb)
        nhmmer_gpu_worker_process_post_vit_gpu(&workers[0]);
      else
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
      free(fwd_survivors);
      free(gpu_xf); free(gpu_xb); free(gpu_fb_scores);
      free(gpu_fb_statuses); free(gpu_fb_x_offsets); free(gpu_fb_L_eff);
      clock_gettime(CLOCK_MONOTONIC, &ts1);
      info->t_cpu_workers += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
      return status;
    }
#endif /* HMMER_THREADS */

    /* Single-threaded fallback */
    {
      NHMMER_GPU_WORKER w;
      memset(&w, 0, sizeof(w));
      status = nhmmer_gpu_worker_init(&w, info);
      if (status != eslOK) { free(fwd_survivors); free(gpu_xf); free(gpu_xb); free(gpu_fb_scores); free(gpu_fb_statuses); free(gpu_fb_x_offsets); free(gpu_fb_L_eff); return status; }

      w.sq              = sq;
      w.seq_id          = seq_id;
      w.complementarity = complementarity;
      w.nwindows        = nfwd_surv;
      w.windows         = fwd_survivors;
      w.use_gpu_fb      = gpu_fb_ok;
      w.gpu_xf          = gpu_xf;
      w.gpu_xb          = gpu_xb;
      w.gpu_scores      = gpu_fb_scores;
      w.gpu_statuses    = gpu_fb_statuses;
      w.gpu_x_offsets   = gpu_fb_x_offsets;
      w.gpu_L_eff       = gpu_fb_L_eff;

      if (w.use_gpu_fb)
        nhmmer_gpu_worker_process_post_vit_gpu(&w);
      else
        nhmmer_gpu_worker_process_post_vit(&w);
      status = w.status;

      if (status == eslOK) {
        p7_tophits_Merge(info->th, w.th);
        p7_pipeline_Merge(info->pli, w.pli);
      }

      nhmmer_gpu_worker_destroy(&w);
      free(fwd_survivors);
      free(gpu_xf); free(gpu_xb); free(gpu_fb_scores);
      free(gpu_fb_statuses); free(gpu_fb_x_offsets); free(gpu_fb_L_eff);
      clock_gettime(CLOCK_MONOTONIC, &ts1);
      info->t_cpu_workers += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
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

  struct timespec ts0, ts1;

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
  clock_gettime(CLOCK_MONOTONIC, &ts0);
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
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_ssv += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  if (gpu_nwindows == 0) { free(gpu_windows); return eslOK; }

  clock_gettime(CLOCK_MONOTONIC, &ts0);
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
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_merge += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  if (msv_windowlist.count == 0) {
    free(msv_windowlist.windows);
    return eslOK;
  }

  /* GPU batch SSV/bias/F1 gating */
  NHMMER_GPU_WINDOW_BATCH wb;
  memset(&wb, 0, sizeof(wb));

  clock_gettime(CLOCK_MONOTONIC, &ts0);
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
      clock_gettime(CLOCK_MONOTONIC, &ts1);
      info->t_batch_filter += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
      return eslOK;
    }

    if (info->do_gpu_vit && msv_windowlist.count >= NHMMER_GPU_BATCH_MIN) {
      status = nhmmer_gpu_viterbi_prefilter(info, sq, &wb, &msv_windowlist, &nsurv,
                                            errbuf, errbuf_size);
      if (status != eslOK) { nhmmer_gpu_window_batch_free(&wb); goto ERROR; }

      if (msv_windowlist.count == 0) {
        nhmmer_gpu_window_batch_free(&wb);
        free(msv_windowlist.windows);
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        info->t_batch_filter += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
        return eslOK;
      }
    }

    nhmmer_gpu_window_batch_free(&wb);
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_batch_filter += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  /* GPU scanning Viterbi longtarget */
  if (info->do_gpu_vit_lt && msv_windowlist.count > 0) {
    P7_HMM_WINDOWLIST vit_wl;
    vit_wl.windows = NULL;

    clock_gettime(CLOCK_MONOTONIC, &ts0);
    status = nhmmer_gpu_viterbi_longtarget(info, sq, &msv_windowlist, &vit_wl,
                                           errbuf, errbuf_size);
    if (status != eslOK) {
      if (vit_wl.windows) free(vit_wl.windows);
      goto ERROR;
    }

    free(msv_windowlist.windows);
    msv_windowlist.windows = NULL;
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    info->t_vit_lt += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

    if (vit_wl.count == 0) {
      if (vit_wl.windows) free(vit_wl.windows);
      return eslOK;
    }

    /* Dispatch to post-Viterbi workers */
    info->n_vit_lt_windows_in  += msv_windowlist.count;
    info->n_vit_lt_windows_out += vit_wl.count;
    info->n_post_vit_windows   += vit_wl.count;
    int ncpus_vlt = info->ncpus;
    if (ncpus_vlt < 1) ncpus_vlt = 1;
    if (ncpus_vlt > vit_wl.count) ncpus_vlt = vit_wl.count;

    clock_gettime(CLOCK_MONOTONIC, &ts0);
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
      clock_gettime(CLOCK_MONOTONIC, &ts1);
      info->t_cpu_workers += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
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
      clock_gettime(CLOCK_MONOTONIC, &ts1);
      info->t_cpu_workers += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
      return status;
    }
  }

  /* Fallback: dispatch windows to full CPU workers */
  int ncpus = info->ncpus;
  if (ncpus < 1) ncpus = 1;
  if (ncpus > msv_windowlist.count) ncpus = msv_windowlist.count;

  clock_gettime(CLOCK_MONOTONIC, &ts0);
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
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    info->t_cpu_workers += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
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
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    info->t_cpu_workers += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
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
