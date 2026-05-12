/* nhmmer GPU pipeline: CPU worker pool, OMX special-state binding, and
 * supporting helpers.
 *
 * After the GPU finishes its filter/parser stages, surviving windows are
 * dispatched to a small pool of CPU worker threads here for the remaining
 * domain definition + hit reporting work. This file owns:
 *   - The NHMMER_GPU_WORKER lifecycle (init/destroy, dynamic work-queue
 *     fetch).
 *   - Three post-* worker bodies (post-Viterbi, post-Forward, post-FB),
 *     each consuming GPU-precomputed special-state matrices.
 *   - OMX XMX binding helpers that swap GPU-produced xf/xb buffers into a
 *     P7_OMX without copying.
 *   - Trace/score helpers used downstream of the GPU pipeline.
 *   - A scalar Viterbi reimplementation kept as a debug oracle for the
 *     CUDA scanning Viterbi kernel.
 *
 * Pure host code; no CUDA calls.
 */
#include <p7_config.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>

#ifdef HMMER_THREADS
#include <pthread.h>
#endif

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_gumbel.h"
#include "esl_sq.h"
#include "esl_vectorops.h"

#include "hmmer.h"
#include "nhmmer_internal.h"

#ifdef HMMER_CUDA
#include "cuda/p7_cuda.h"
#include "cuda/nhmmer_cuda_internal.h"

int
nhmmer_gpu_worker_next_window(NHMMER_GPU_WORKER *w)
{
  int idx;

  if (w->next_window == NULL) return -1;
#ifdef HMMER_THREADS
  if (w->work_mutex) pthread_mutex_lock(w->work_mutex);
#endif
  idx = *(w->next_window);
  if (idx < w->global_nwindows) (*(w->next_window))++;
#ifdef HMMER_THREADS
  if (w->work_mutex) pthread_mutex_unlock(w->work_mutex);
#endif
  return (idx < w->global_nwindows) ? idx : -1;
}

void
nhmmer_gpu_trace_domain_window(const NHMMER_GPU_WORKER *w, const char *path,
                               int local_idx, const P7_HMM_WINDOW *window,
                               double domain_before, int status)
{
  if (!w->do_domain_trace) return;

  double domain_delta = w->pli->time_domain - domain_before;
  fprintf(stderr,
          "NHMMER_GPU_DOMAIN_TRACE path=%s worker=%d local=%d seq=%" PRId64
          " strand=%d n=%" PRIu64 " len=%u status=%d domain_sec=%.6f"
          " nregions=%d nenvelopes=%d ndom=%d hits=%" PRIu64 "\n",
          path, w->worker_id, local_idx, w->seq_id, w->complementarity,
          (uint64_t) window->n, (uint32_t) window->length, status,
          domain_delta,
          w->pli->ddef ? w->pli->ddef->nregions : -1,
          w->pli->ddef ? w->pli->ddef->nenvelopes : -1,
          w->pli->ddef ? w->pli->ddef->ndom : -1,
          w->th ? w->th->N : -1);
}

int
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
  w->ndb             = NULL;
  w->slice_sq        = NULL;
  w->slice_dsq       = NULL;
  w->slice_alloc     = 0;
  w->windows         = NULL;
  w->nwindows        = 0;
  w->seq_id          = 0;
  w->complementarity = 0;
  w->status          = eslOK;
  w->cuda_engine     = info->cuda_engine;
  w->cuda_msv        = info->cuda_msv;
  w->worker_id       = -1;
  w->do_domain_trace = info->do_domain_trace;
  w->next_window     = NULL;
  w->global_nwindows = 0;
  w->prefilter_x_offsets = NULL;
  w->domcorr_pending.entries = NULL;
  w->domcorr_pending.n       = 0;
  w->domcorr_pending.nalloc  = 0;
#ifdef HMMER_THREADS
  w->work_mutex      = NULL;
#endif
  return eslOK;

ERROR:
  return status;
}

void
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
  if (w->slice_sq) {
    w->slice_sq->dsq = NULL;
    esl_sq_Destroy(w->slice_sq);
  }
  free(w->slice_dsq);
  free(w->domcorr_pending.entries);
  if (w->scoredata) p7_hmm_ScoreDataDestroy(w->scoredata);
  if (w->th)        p7_tophits_Destroy(w->th);
  if (w->pli)       p7_pipeline_Destroy(w->pli);
  if (w->om)        p7_oprofile_Destroy(w->om);
  if (w->bg)        p7_bg_Destroy(w->bg);
}

void
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

/* Post-Viterbi worker: processes sub-windows that already passed GPU scanning
 * Viterbi. Only runs Forward/Backward and domain definition (skips MSV and
 * scanning Viterbi). */
void
nhmmer_gpu_worker_process_post_vit(NHMMER_GPU_WORKER *w)
{
  int          i;
  int          overlap = 0;
  int          use_dynamic = (w->next_window != NULL);
  P7_HMM_WINDOW *window;

  for (i = use_dynamic ? nhmmer_gpu_worker_next_window(w) : 0;
       use_dynamic ? (i >= 0) : (i < w->nwindows);
       i = use_dynamic ? nhmmer_gpu_worker_next_window(w) : i + 1) {
    window = w->windows + i;
    ESL_DSQ *subseq = w->sq->dsq + window->n - 1;

    w->pli->pos_past_msv += window->length;
    w->pli->pos_past_bias += window->length;
    w->pli->pos_past_vit += window->length;
    if (i > 0)
      w->pli->pos_past_vit -= ESL_MIN((int)window->length,
                                      ESL_MAX(0, (int)(w->windows[i-1].n + w->windows[i-1].length) - (int)window->n));

    p7_omx_GrowTo(w->pli->oxf, w->om->M, 0, window->length);
    p7_oprofile_ReconfigRestLength(w->om, ESL_MIN((int)window->length, w->om->max_length));

    double domain_before = w->pli->time_domain;
    w->status = p7_pli_postViterbi_LongTarget(w->pli, w->om, w->bg, w->th, w->scoredata,
                                              w->seq_id, (int)window->n, window->length, subseq,
                                              (int64_t)w->sq->start, w->sq->name, w->sq->source,
                                              w->sq->acc, w->sq->desc,
                                              -1, w->complementarity, &overlap, w->pli_tmp);
    nhmmer_gpu_trace_domain_window(w, "post_vit", i, window, domain_before, w->status);
    if (w->status != eslOK) return;

    if (overlap == -1 && i < w->global_nwindows - 1) {
      overlap = ESL_MAX(0, (int)(window->n + window->length) - (int)w->windows[i+1].n);
    } else {
      overlap = 0;
    }

    w->pli->ddef->ndom = 0;
  }
}

/* Worker function that skips Forward recomputation by using precomputed xf
 * from the GPU Forward prefilter. Injects xf into pli->oxf, then calls
 * p7_pli_postViterbi_LongTarget with Forward pre-filled. The F3 check inside
 * will recompute fwdsc from oxf but should pass since GPU already filtered. */
void
nhmmer_gpu_worker_process_post_fwd(NHMMER_GPU_WORKER *w)
{
  int          i;
  int          overlap = 0;
  int          use_dynamic = (w->next_window != NULL);
  P7_HMM_WINDOW *window;

  for (i = use_dynamic ? nhmmer_gpu_worker_next_window(w) : 0;
       use_dynamic ? (i >= 0) : (i < w->nwindows);
       i = use_dynamic ? nhmmer_gpu_worker_next_window(w) : i + 1) {
    window = w->windows + i;
    ESL_DSQ *subseq = w->sq->dsq + window->n - 1;
    int window_len = window->length;
    NHMMER_OMX_BINDING saved_oxf;
    float *win_xf;
    float totscale = 0.0f;

    w->pli->pos_past_msv += window_len;
    w->pli->pos_past_bias += window_len;
    w->pli->pos_past_vit += window_len;
    if (i > 0)
      w->pli->pos_past_vit -= ESL_MIN(window_len,
                                      ESL_MAX(0, (int)(w->windows[i-1].n + w->windows[i-1].length) - (int)window->n));

    p7_oprofile_ReconfigRestLength(w->om, window_len);

    win_xf = w->prefilter_xf + (w->prefilter_x_offsets ? w->prefilter_x_offsets[i] : w->prefilter_xf_offset);
    for (int r = 1; r <= window_len; r++) {
      float s = win_xf[r * p7X_NXCELLS + p7X_SCALE];
      if (s > 1.0f) totscale += logf(s);
    }
    nhmmer_gpu_BindOmxXmx(w->pli->oxf, win_xf, w->om->M, window_len, TRUE, totscale, &saved_oxf);

    /* Call existing postViterbi but with oxf already filled.
     * p7_pli_postViterbi_LongTarget will call p7_ForwardParser which
     * overwrites oxf->xmx with equivalent CPU values. We can't avoid
     * that without modifying shared code. Instead, use the dedicated
     * bypass: call p7_pli_postFwd_LongTarget which skips Forward. */
    double domain_before = w->pli->time_domain;
    w->status = p7_pli_postFwd_LongTarget(w->pli, w->om, w->bg, w->th, w->scoredata,
                                           w->seq_id, (int)window->n, window_len, subseq,
                                           (int64_t)w->sq->start, w->sq->name, w->sq->source,
                                           w->sq->acc, w->sq->desc,
                                           -1, w->complementarity, &overlap, w->pli_tmp,
                                           w->prefilter_fwdsc[i]);
    nhmmer_gpu_RestoreOmxXmx(w->pli->oxf, &saved_oxf);
    nhmmer_gpu_trace_domain_window(w, "post_fwd", i, window, domain_before, w->status);
    if (w->status != eslOK) return;

    if (overlap == -1 && i < w->global_nwindows - 1) {
      overlap = ESL_MAX(0, (int)(window->n + window->length) - (int)w->windows[i+1].n);
    } else {
      overlap = 0;
    }

    w->pli->ddef->ndom = 0;
  }
}

/* Worker function that consumes GPU Forward+Backward parser special-state
 * matrices. CPU domain definition still owns envelope/domain logic, but it
 * starts from GPU-computed oxf/oxb xmx instead of rerunning CPU Fwd/Bwd. */
void
nhmmer_gpu_worker_process_post_fb(NHMMER_GPU_WORKER *w)
{
  int          i;
  int          overlap = 0;
  int          use_dynamic = (w->next_window != NULL);
  P7_HMM_WINDOW *window;

  /* P1: connect the pending-list pointer to the worker's pipeline; the
   * strand orchestrator owns the do_gpu_domcorr flag and consumes the list
   * once after this thread joins. */
  w->pli->gpu_domcorr_pending = &w->domcorr_pending;
  w->domcorr_pending.n        = 0;  /* reuse across strands; do not free entries */

  for (i = use_dynamic ? nhmmer_gpu_worker_next_window(w) : 0;
       use_dynamic ? (i >= 0) : (i < w->nwindows);
       i = use_dynamic ? nhmmer_gpu_worker_next_window(w) : i + 1) {
    window = w->windows + i;
    const ESL_SQ *seq_for_window = w->sq;
    ESL_DSQ *subseq;
    int window_len = window->length;
    size_t xoff = w->gpu_x_offsets[i];
    NHMMER_OMX_BINDING saved_oxf;
    NHMMER_OMX_BINDING saved_oxb;

    if (w->gpu_statuses && w->gpu_statuses[i * 2 + 1] != eslOK) {
      w->status = w->gpu_statuses[i * 2 + 1];
      return;
    }

    w->pli->pos_past_msv += window_len;
    w->pli->pos_past_bias += window_len;
    w->pli->pos_past_vit += window_len;
    if (i > 0)
      w->pli->pos_past_vit -= ESL_MIN(window_len,
                                      ESL_MAX(0, (int)(w->windows[i-1].n + w->windows[i-1].length) - (int)window->n));

    p7_oprofile_ReconfigRestLength(w->om, window_len);

    if (w->ndb != NULL && w->sq->dsq == NULL) {
      w->status = nhmmer_gpu_nucdb_fill_slice(w->ndb, w->om->abc, w->seq_id,
                                              w->complementarity, window->n,
                                              window_len, w->sq->name,
                                              &w->slice_sq, &w->slice_dsq,
                                              &w->slice_alloc);
      if (w->status != eslOK) return;
      seq_for_window = w->slice_sq;
      subseq = w->slice_sq->dsq;
    } else {
      subseq = w->sq->dsq + window->n - 1;
    }
    nhmmer_gpu_BindOmxXmx(w->pli->oxf, w->gpu_xf + xoff, w->om->M, window_len, TRUE, 0.0f, &saved_oxf);
    nhmmer_gpu_BindOmxXmx(w->pli->oxb, w->gpu_xb + xoff, w->om->M, window_len, FALSE, 0.0f, &saved_oxb);

    double domain_before = w->pli->time_domain;
    w->status = p7_pli_postFwdBwdNoF3_LongTarget(w->pli, w->om, w->bg, w->th, w->scoredata,
                                                 w->seq_id, (int)window->n, window_len, subseq,
                                                 (int64_t)seq_for_window->start, seq_for_window->name, seq_for_window->source,
                                                 seq_for_window->acc, seq_for_window->desc,
                                                 -1, w->complementarity, &overlap, w->pli_tmp,
                                                 w->gpu_scores[i * 2]);
    nhmmer_gpu_RestoreOmxXmx(w->pli->oxb, &saved_oxb);
    nhmmer_gpu_RestoreOmxXmx(w->pli->oxf, &saved_oxf);
    nhmmer_gpu_trace_domain_window(w, "post_fb", i, window, domain_before, w->status);
    if (w->status != eslOK) return;

    if (overlap == -1 && i < w->global_nwindows - 1) {
      overlap = ESL_MAX(0, (int)(window->n + window->length) - (int)w->windows[i+1].n);
    } else {
      overlap = 0;
    }

    w->pli->ddef->ndom = 0;
  }
}

void
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

void
nhmmer_gpu_RestoreOmxXmx(P7_OMX *ox, const NHMMER_OMX_BINDING *saved)
{
  ox->xmx            = saved->xmx;
  ox->allocXR        = saved->allocXR;
  ox->M              = saved->M;
  ox->L              = saved->L;
  ox->has_own_scales = saved->has_own_scales;
  ox->totscale       = saved->totscale;
}

/* P1: after all workers in a strand have joined, run the deferred 2nd-pass
 * Forwards as a single batched GPU call and patch hit->dcl[0].dombias /
 * .domcorrection for every queued entry. No-op if no entry is queued. The
 * pending lists are owned by workers; this function consumes them in place
 * (sets pend->n = 0 on return) so the workers can be reused on the next
 * strand without reallocation. */
int
nhmmer_gpu_flush_domcorr(NHMMER_GPU_INFO *info, NHMMER_GPU_WORKER *workers, int nworkers,
                         char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int total = 0;
  int w, k;
  const uint8_t **dsq_ptrs   = NULL;
  int            *lengths    = NULL;
  float          *gpu_scores = NULL;
  int            *gpu_statuses = NULL;
  struct timespec ts0, ts1;
  double h2d_sec = 0.0, kernel_sec = 0.0, d2h_sec = 0.0;

  if (info == NULL || workers == NULL || nworkers <= 0) return eslOK;
  if (info->cuda_engine == NULL || info->cuda_msv == NULL) return eslOK;

  for (w = 0; w < nworkers; w++) total += workers[w].domcorr_pending.n;
  if (total == 0) return eslOK;

  clock_gettime(CLOCK_MONOTONIC, &ts0);

  dsq_ptrs     = (const uint8_t **)malloc(sizeof(uint8_t *) * total);
  lengths      = (int *)           malloc(sizeof(int)       * total);
  gpu_scores   = (float *)         malloc(sizeof(float)     * total);
  gpu_statuses = (int *)           malloc(sizeof(int)       * total);
  if (!dsq_ptrs || !lengths || !gpu_scores || !gpu_statuses) {
    status = eslEMEM;
    if (errbuf && errbuf_size > 0)
      snprintf(errbuf, errbuf_size, "domcorr flush: out of memory (total=%d)", total);
    goto DONE;
  }

  {
    int off = 0;
    for (w = 0; w < nworkers; w++) {
      P7_GPU_DOMCORR_PENDING *pend = &workers[w].domcorr_pending;
      for (k = 0; k < pend->n; k++) {
        dsq_ptrs[off] = pend->entries[k].dsq;
        lengths[off]  = pend->entries[k].n;
        off++;
      }
    }
  }

  status = p7_cuda_DomainScoreOnlyFwdBatch(info->cuda_engine, info->cuda_msv,
                                           total, dsq_ptrs, lengths,
                                           gpu_scores, gpu_statuses,
                                           &h2d_sec, &kernel_sec, &d2h_sec,
                                           errbuf, errbuf_size);
  if (status != eslOK) goto DONE;

  /* Patch hit fields per pending entry. p7_FLogsum mirrors the formula used
   * by p7_pli_postViterbi_LongTarget's tail (p7_pipeline.c:1058) so the
   * GPU dombias is computed identically to that path. The
   * postFwd_LongTarget_Core path itself sets hit->dcl[0].dombias = dom_bias
   * = dom->domcorrection, so we follow that convention here. */
  {
    int off = 0;
    for (w = 0; w < nworkers; w++) {
      P7_GPU_DOMCORR_PENDING *pend = &workers[w].domcorr_pending;
      for (k = 0; k < pend->n; k++, off++) {
        P7_HIT *hit = pend->entries[k].hit;
        float   envsc = pend->entries[k].envsc;
        if (hit == NULL) continue;  /* domain skipped by ali_len<8 in hit-reporting */
        if (gpu_statuses[off] != eslOK) {
          /* Kernel rejected this envelope (eslEINVAL/eslERANGE). Leave the
           * sentinel zero so the hit reports as if domcorrection had not
           * fired; matches the CPU eslERANGE-skipped behavior in
           * rescore_isolated_domain. */
          continue;
        }
        float dom_corr = gpu_scores[off] - envsc;
        hit->dcl[0].domcorrection = dom_corr;
        hit->dcl[0].dombias       = dom_corr;
      }
      pend->n = 0;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_gpu_domcorr        += nhmmer_gpu_elapsed(&ts0, &ts1);
  info->t_gpu_domcorr_h2d    += h2d_sec;
  info->t_gpu_domcorr_kernel += kernel_sec;
  info->t_gpu_domcorr_d2h    += d2h_sec;
  info->gpu_domcorr_envelopes += total;
  info->gpu_domcorr_launches += 1;

 DONE:
  free(dsq_ptrs);
  free(lengths);
  free(gpu_scores);
  free(gpu_statuses);
  return status;
}

int
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

const char gpu_to_p7_state[] = {
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

int
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

#ifdef HMMER_THREADS
void *
nhmmer_gpu_thread_func(void *arg)
{
  NHMMER_GPU_WORKER *w = (NHMMER_GPU_WORKER *)arg;
  nhmmer_gpu_worker_process(w);
  return NULL;
}

void *
nhmmer_gpu_thread_func_post_vit(void *arg)
{
  NHMMER_GPU_WORKER *w = (NHMMER_GPU_WORKER *)arg;
  nhmmer_gpu_worker_process_post_vit(w);
  return NULL;
}

void *
nhmmer_gpu_thread_func_post_fwd(void *arg)
{
  NHMMER_GPU_WORKER *w = (NHMMER_GPU_WORKER *)arg;
  nhmmer_gpu_worker_process_post_fwd(w);
  return NULL;
}

void *
nhmmer_gpu_thread_func_post_fb(void *arg)
{
  NHMMER_GPU_WORKER *w = (NHMMER_GPU_WORKER *)arg;
  nhmmer_gpu_worker_process_post_fb(w);
  return NULL;
}
#endif

/* Scalar reference for the CUDA scanning Viterbi kernel. Kept out of the GPU
 * dispatch file because it is purely host code and only invoked under the
 * --gpu-compare debug path; co-locating it with the worker pool keeps the
 * GPU dispatch unit focused on CUDA orchestration. */
int
nhmmer_gpu_scalar_viterbi_debug(P7_OPROFILE *om, const ESL_DSQ *dsq, int L,
                                float filtersc, float F2, int16_t override_thresh,
                                int *ret_nseeds, int16_t *xE_trace, int trace_len)
{
  int Q = p7O_NQW(om->M);
  int M = om->M;
  int Kp = om->abc->Kp;
  int N = Q * 8;

  int16_t *prev_m = (int16_t *)calloc(N, sizeof(int16_t));
  int16_t *prev_d = (int16_t *)calloc(N, sizeof(int16_t));
  int16_t *prev_i = (int16_t *)calloc(N, sizeof(int16_t));
  int16_t *curr_m = (int16_t *)calloc(N, sizeof(int16_t));
  int16_t *curr_d = (int16_t *)calloc(N, sizeof(int16_t));
  int16_t *curr_i = (int16_t *)calloc(N, sizeof(int16_t));
  if (!prev_m || !prev_d || !prev_i || !curr_m || !curr_d || !curr_i) {
    free(prev_m); free(prev_d); free(prev_i); free(curr_m); free(curr_d); free(curr_i);
    return eslEMEM;
  }

  int16_t *twv = (int16_t *)om->twv;
  int16_t *rwv_base = (int16_t *)om->rwv[0];
  #define TWV(t, q, z) ((t) == 7 ? twv[((7*Q)+(q))*8+(z)] : twv[((q)*7+(t))*8+(z)])
  #define RWV(x, q, z) rwv_base[((int)(x)*Q+(q))*8+(z)]

  float invP = esl_gumbel_invsurv(F2, om->evparam[p7_VMU], om->evparam[p7_VLAMBDA]);
  int loc_L = ESL_MIN(L, om->max_length);
  float pmove = (2.0f + om->nj) / ((float)loc_L + 2.0f + om->nj);
  int16_t xw_move = (int16_t)roundf(om->scale_w * logf(pmove));
  float xw_c_move_f = roundf(om->scale_w * logf(pmove));
  int16_t sc_thresh;
  if (override_thresh != 0) {
    sc_thresh = override_thresh;
  } else {
    sc_thresh = (int16_t)ceilf(((filtersc + 0.69314718f * invP + 3.0f) * om->scale_w)
                       - (float)om->xw[p7O_E][p7O_MOVE] - xw_c_move_f + (float)om->base_w);
  }
  int16_t xw_e_loop = om->xw[p7O_E][p7O_LOOP];
  int16_t xw_e_move = om->xw[p7O_E][p7O_MOVE];
  int16_t base_w    = om->base_w;
  int16_t ddbound_w = om->ddbound_w;

  #define SAT16(a,b) ({ int _v=(int)(a)+(int)(b); (int16_t)(_v>32767?32767:(_v<-32768?-32768:_v)); })

  for (int k = 0; k < N; k++) prev_m[k] = prev_d[k] = prev_i[k] = -32768;

  int16_t xN = base_w, xB = SAT16(xN, xw_move), xJ = -32768, xC = -32768;
  int nseeds = 0;

  for (int i = 1; i <= L; i++) {
    uint8_t x = dsq[i];
    if (x >= Kp) { if (xE_trace && i <= trace_len) xE_trace[i] = -32768; continue; }

    int16_t dcv_lanes[8];
    int16_t xE = -32768;
    int16_t dmax_all = -32768;

    for (int z = 0; z < 8; z++) {
      int16_t mpv = (z == 0) ? (int16_t)-32768 : prev_m[(z-1)*Q + Q-1];
      int16_t dpv = (z == 0) ? (int16_t)-32768 : prev_d[(z-1)*Q + Q-1];
      int16_t ipv = (z == 0) ? (int16_t)-32768 : prev_i[(z-1)*Q + Q-1];
      int16_t dcv = -32768;
      int16_t xE_lane = -32768;
      int16_t dmax_lane = -32768;

      for (int q = 0; q < Q; q++) {
        int c = z * Q + q;
        int16_t sv = SAT16(xB, TWV(0,q,z));
        int16_t cand;
        cand = SAT16(mpv, TWV(1,q,z)); if (cand > sv) sv = cand;
        cand = SAT16(ipv, TWV(2,q,z)); if (cand > sv) sv = cand;
        cand = SAT16(dpv, TWV(3,q,z)); if (cand > sv) sv = cand;
        sv = SAT16(sv, RWV(x,q,z));
        if (sv > xE_lane) xE_lane = sv;

        mpv = prev_m[c]; dpv = prev_d[c]; ipv = prev_i[c];
        curr_m[c] = sv;
        curr_d[c] = dcv;
        dcv = SAT16(sv, TWV(4,q,z));
        if (dcv > dmax_lane) dmax_lane = dcv;
        cand = SAT16(mpv, TWV(5,q,z));
        int16_t isv = SAT16(ipv, TWV(6,q,z));
        curr_i[c] = cand > isv ? cand : isv;
      }
      if (xE_lane > xE) xE = xE_lane;
      if (dmax_lane > dmax_all) dmax_all = dmax_lane;
      dcv_lanes[z] = dcv;
    }

    if (xE_trace && i <= trace_len) xE_trace[i] = xE;

    if (xE >= sc_thresh) {
      nseeds++;
      for (int k = 0; k < N; k++) curr_m[k] = curr_d[k] = curr_i[k] = -32768;
      xN = base_w; xB = SAT16(xN, xw_move); xJ = -32768; xC = -32768;
    } else {
      int16_t c2 = SAT16(xE, xw_e_move);
      xC = xC > c2 ? xC : c2;
      int16_t j2 = SAT16(xE, xw_e_loop);
      xJ = xJ > j2 ? xJ : j2;
      xB = ESL_MAX(SAT16(xJ, xw_move), SAT16(xN, xw_move));

      if ((int)dmax_all + (int)ddbound_w > (int)xB) {
        /* Full lazy-F: propagate D->D across all lanes */
        for (int pass = 0; pass < 8; pass++) {
          int any_update = 0;
          for (int z = 0; z < 8; z++) {
            int16_t dcv = (z == 0) ? (int16_t)-32768 : dcv_lanes[z-1];
            for (int q = 0; q < Q; q++) {
              int c = z * Q + q;
              if (dcv > curr_d[c]) { curr_d[c] = dcv; any_update = 1; }
              dcv = SAT16(curr_d[c], TWV(7,q,z));
            }
            dcv_lanes[z] = dcv;
          }
          if (!any_update) break;
        }
      } else {
        for (int z = 0; z < 8; z++)
          curr_d[z * Q] = (z == 0) ? (int16_t)-32768 : dcv_lanes[z-1];
      }
    }

    int16_t *tmp;
    tmp = prev_m; prev_m = curr_m; curr_m = tmp;
    tmp = prev_d; prev_d = curr_d; curr_d = tmp;
    tmp = prev_i; prev_i = curr_i; curr_i = tmp;
  }

  #undef TWV
  #undef RWV
  #undef SAT16

  *ret_nseeds = nseeds;
  free(prev_m); free(prev_d); free(prev_i);
  free(curr_m); free(curr_d); free(curr_i);
  return eslOK;
}

#endif /* HMMER_CUDA */
