/* nhmmer GPU pipeline: Forward prefilter + Forward/Backward parser dispatch.
 *
 * Stage that runs after the scanning Viterbi narrows the candidate window
 * list. Two complementary entry points feed the CPU worker pool:
 *   - forward_prefilter: GPU Forward score + F3 gate, returns survivor xf.
 *   - forward_backward_compact: GPU Forward then Backward in one shot,
 *     compacting survivors and returning xf/xb special-state matrices.
 *   - run_fb_parser_batch: chunked driver used when the prefilter result
 *     is reused as the Forward input.
 *
 * Plain C (no .cu); calls the kernel-dispatch APIs declared in cuda/p7_cuda.h.
 */
#include <p7_config.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_dsqdata.h"
#include "esl_exponential.h"
#include "esl_gumbel.h"
#include "esl_sq.h"
#include "esl_vectorops.h"

#include "hmmer.h"
#include "p7_nucdb.h"
#include "nhmmer_internal.h"

#ifdef HMMER_CUDA
#include "cuda/p7_cuda.h"
#include "cuda/nhmmer_cuda_internal.h"

/* GPU Forward pre-filter: uses GPU Forward score for F3 gating, then dispatches
 * surviving windows to CPU worker threads for ForwardParser + domaindef.
 * Eliminates 40-60% of windows before CPU work, while preserving correct
 * overlap tracking and hit reporting from the existing worker path. */
int
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
  double fwd_gate = pli->F3;
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
    if (P_val > fwd_gate) continue;

    surv_src_idx[nsurv] = i;
    surv_fwdsc[nsurv] = fwdsc;
    survivors[nsurv] = windows[i];
    surv_x_offsets[nsurv] = surv_total_xcells;
    surv_total_xcells += (size_t)(window_len + 1) * p7X_NXCELLS;
    nsurv++;
  }

  /* --gpu-compare Stage C: compare GPU Forward scores to CPU ForwardParser */
  if (info->do_compare) {
    int gpu_pass_count = nsurv;
    int cpu_pass_count = 0;
    p7_omx_GrowTo(pli->oxf, om->M, 0, om->max_length);

    for (i = 0; i < nwindows; i++) {
      if (statuses[2*i] != eslOK) continue;

      int window_len = windows[i].length;
      ESL_DSQ *subseq = sq->dsq + windows[i].n - 1;
      float gpu_fwdsc = scores[2*i];
      float cpu_fwdsc, nullsc, bias_filtersc, filtersc, cpu_seq_score, gpu_seq_score;
      double cpu_P, gpu_P;

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

      p7_oprofile_ReconfigRestLength(om, window_len);
      p7_omx_GrowTo(pli->oxf, om->M, 0, window_len);
      p7_ForwardParser(subseq, window_len, om, pli->oxf, &cpu_fwdsc);

      gpu_seq_score = (gpu_fwdsc - filtersc) / eslCONST_LOG2;
      cpu_seq_score = (cpu_fwdsc - filtersc) / eslCONST_LOG2;
      gpu_P = esl_exp_surv(gpu_seq_score, om->evparam[p7_FTAU], om->evparam[p7_FLAMBDA]);
      cpu_P = esl_exp_surv(cpu_seq_score, om->evparam[p7_FTAU], om->evparam[p7_FLAMBDA]);

      int gpu_pass = (gpu_P <= pli->F3) ? 1 : 0;
      int cpu_pass = (cpu_P <= pli->F3) ? 1 : 0;
      cpu_pass_count += cpu_pass;

      if (gpu_pass != cpu_pass || fabsf(gpu_fwdsc - cpu_fwdsc) > 0.1f)
        fprintf(stderr, "NHMMER_GPU_COMPARE_FWD win=%d n=%" PRId64 " len=%d gpu_fwdsc=%.4f cpu_fwdsc=%.4f delta=%.6f gpu_P=%.2e cpu_P=%.2e gpu_pass=%d cpu_pass=%d\n",
                i, (int64_t)windows[i].n, window_len, gpu_fwdsc, cpu_fwdsc, gpu_fwdsc - cpu_fwdsc, gpu_P, cpu_P, gpu_pass, cpu_pass);
    }
    fprintf(stderr, "NHMMER_GPU_COMPARE_FWD_SUMMARY total=%d gpu_pass=%d cpu_pass=%d\n",
            nwindows, gpu_pass_count, cpu_pass_count);
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

int
nhmmer_gpu_forward_backward_compact(NHMMER_GPU_INFO *info, const ESL_SQ *sq,
                                    int64_t seq_id, int complementarity,
                                    const P7_NUCDB *ndb, int chunk_start, int chunk_count,
                                    const uint8_t *d_nucdb,
                                    P7_HMM_WINDOW *windows, int nwindows,
                                    P7_HMM_WINDOW **ret_survivors, int *ret_nsurv,
                                    float **ret_xf, float **ret_xb,
                                    float **ret_scores, int **ret_statuses,
                                    size_t **ret_x_offsets, int **ret_L_eff,
                                    char *errbuf, int errbuf_size)
{
  int status = eslOK;
  P7_OPROFILE *om = info->om;
  P7_BG *bg = info->bg;
  P7_PIPELINE *pli = info->pli;
  NHMMER_GPU_WINDOW_BATCH wb;
  int *seqidx = NULL;
  size_t *x_offsets = NULL;
  size_t total_xcells = 0;
  P7_HMM_WINDOW *survivors = NULL;
  int *surv_src_idx = NULL;
  size_t *surv_x_offsets = NULL;
  int *surv_L_eff = NULL;
  size_t surv_total_xcells = 0;
  size_t host_surv_total_xcells = 0;
  int nsurv = 0;
  float *gpu_xf = NULL, *gpu_xb = NULL, *gpu_scores = NULL;
  int *gpu_statuses = NULL;
  struct timespec pack_ts0, pack_ts1;

  (void)seq_id;
  (void)complementarity;

  *ret_survivors = NULL;
  *ret_nsurv = 0;
  *ret_xf = NULL;
  *ret_xb = NULL;
  *ret_scores = NULL;
  *ret_statuses = NULL;
  *ret_x_offsets = NULL;
  *ret_L_eff = NULL;

  if (nwindows == 0) return eslOK;

  memset(&wb, 0, sizeof(wb));
  status = nhmmer_gpu_window_batch_init(&wb, nwindows);
  if (status != eslOK) goto ERROR;

  clock_gettime(CLOCK_MONOTONIC, &pack_ts0);
  P7_HMM_WINDOWLIST tmp_wl;
  tmp_wl.windows = windows;
  tmp_wl.count = nwindows;
  tmp_wl.size = nwindows;
  if (ndb != NULL && d_nucdb != NULL) {
    status = nhmmer_gpu_window_batch_describe(&wb, &tmp_wl);
    if (status != eslOK) goto ERROR;
    status = nhmmer_gpu_prepare_parser_resident_batch(info, ndb, chunk_start, chunk_count,
                                                      complementarity, d_nucdb,
                                                      windows, nwindows,
                                                      &wb.chu, errbuf, errbuf_size);
    if (status != eslOK) goto ERROR;
  } else {
    status = nhmmer_gpu_window_batch_pack(&wb, sq, &tmp_wl);
    if (status != eslOK) goto ERROR;
  }

  status = nhmmer_gpu_ensure_parser_scratch(info, nwindows);
  if (status != eslOK) goto ERROR;
  seqidx       = info->h_parser_seqidx;
  x_offsets    = info->h_parser_x_offsets;
  surv_src_idx = info->h_parser_surv_src_idx;

  for (int i = 0; i < nwindows; i++) {
    seqidx[i] = i;
    x_offsets[i] = total_xcells;
    total_xcells += (size_t)(windows[i].length + 1) * p7X_NXCELLS;
  }
  clock_gettime(CLOCK_MONOTONIC, &pack_ts1);
  info->t_gpu_fb_pack += nhmmer_gpu_elapsed(&pack_ts0, &pack_ts1);

  ESL_ALLOC(survivors, sizeof(P7_HMM_WINDOW) * nwindows);
  ESL_ALLOC(surv_x_offsets, sizeof(size_t) * nwindows);
  ESL_ALLOC(surv_L_eff, sizeof(int) * nwindows);

  status = p7_cuda_ForwardParserDsqdataSubsetF3SurvivorsDevice(info->cuda_engine, info->cuda_msv,
                                                               bg, &wb.chu, seqidx, nwindows,
                                                               x_offsets, total_xcells,
                                                               pli->do_biasfilter, pli->B3,
                                                               om->evparam[p7_FTAU], om->evparam[p7_FLAMBDA], pli->F3,
                                                               NULL, &nsurv, &surv_total_xcells,
                                                               errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;

  if (nsurv == 0) {
    *ret_survivors = survivors;
    *ret_nsurv = 0;
    nhmmer_gpu_window_batch_free(&wb);
    free(surv_x_offsets); free(surv_L_eff);
    return eslOK;
  }

  ESL_ALLOC(gpu_xf, sizeof(float) * surv_total_xcells);
  ESL_ALLOC(gpu_xb, sizeof(float) * surv_total_xcells);
  ESL_ALLOC(gpu_scores, sizeof(float) * 2 * nsurv);
  ESL_ALLOC(gpu_statuses, sizeof(int) * 2 * nsurv);

  status = p7_cuda_BackwardParserDsqdataSubsetStoredForwardDeviceSurvivors(info->cuda_engine, info->cuda_msv,
                                                                           &wb.chu, nsurv,
                                                                           surv_total_xcells,
                                                                           gpu_xf, gpu_xb,
                                                                           gpu_scores, gpu_statuses,
                                                                           errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;

  status = p7_cuda_ForwardParserGetF3SurvivorIdx(info->cuda_engine, surv_src_idx, nsurv,
                                                 errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;

  for (int si = 0; si < nsurv; si++) {
    int src = surv_src_idx[si];
    int window_len = windows[src].length;
    survivors[si] = windows[src];
    surv_x_offsets[si] = host_surv_total_xcells;
    surv_L_eff[si] = window_len;
    host_surv_total_xcells += (size_t)(window_len + 1) * p7X_NXCELLS;
  }
  if (host_surv_total_xcells != surv_total_xcells) { status = eslFAIL; goto ERROR; }

  *ret_survivors = survivors;
  *ret_nsurv = nsurv;
  *ret_xf = gpu_xf;
  *ret_xb = gpu_xb;
  *ret_scores = gpu_scores;
  *ret_statuses = gpu_statuses;
  *ret_x_offsets = surv_x_offsets;
  *ret_L_eff = surv_L_eff;

  {
    P7_CUDA_MSV_STATS stats;
    p7_cuda_engine_GetStats(info->cuda_engine, &stats);
    info->t_gpu_fb_h2d    = stats.fwd_h2d_seconds + stats.bck_h2d_seconds;
    info->t_gpu_fb_kernel = stats.fwd_kernel_seconds + stats.bck_kernel_seconds;
    info->t_gpu_fb_d2h    = stats.fwd_d2h_seconds + stats.bck_d2h_seconds;
    if (ndb == NULL || d_nucdb == NULL) {
      info->gpu_fb_packed_bytes += 1;
      for (int pi = 0; pi < wb.chu.N; pi++)
        info->gpu_fb_packed_bytes += wb.chu.L[pi] + 1;
    }
  }

  nhmmer_gpu_window_batch_free(&wb);
  return eslOK;

ERROR:
  nhmmer_gpu_window_batch_free(&wb);
  free(survivors);
  free(surv_x_offsets); free(surv_L_eff);
  free(gpu_xf); free(gpu_xb); free(gpu_scores); free(gpu_statuses);
  return status;
}

#define NHMMER_GPU_FB_MAX_XCELLS (128UL * 1024 * 1024)

int
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

    if (backward_only) {
      for (int k = 0; k < sub_n; k++)
        fb_scores[(sub_start + k) * 2] = precomputed_fwdsc[sub_start + k];
    }

    sub_start = sub_end;
  }

  if (info->do_compare) {
    P7_OMX *cpu_xf = p7_omx_Create(om->M, 0, 0);
    P7_OMX *cpu_xb = p7_omx_Create(om->M, 0, 0);
    int nprinted = 0;
    if (cpu_xf && cpu_xb) {
      for (i = 0; i < nwindows && nprinted < 12; i++) {
        int L = (int)windows[i].length;
        ESL_DSQ *subseq = sq->dsq + windows[i].n - 1;
        float cpu_fwdsc = 0.0f;
        float cpu_bcksc = 0.0f;
        float max_fwd = 0.0f;
        float max_bck = 0.0f;
        int max_fwd_row = 0, max_fwd_cell = 0;
        int max_bck_row = 0, max_bck_cell = 0;
        size_t xcells = (size_t)(L + 1) * p7X_NXCELLS;

        p7_oprofile_ReconfigRestLength(om, L);
        p7_omx_GrowTo(cpu_xf, om->M, 0, L);
        p7_omx_GrowTo(cpu_xb, om->M, 0, L);
        p7_ForwardParser(subseq, L, om, cpu_xf, &cpu_fwdsc);
        p7_BackwardParser(subseq, L, om, cpu_xf, cpu_xb, &cpu_bcksc);

        for (size_t xc = 0; xc < xcells; xc++) {
          float df = fabsf(cpu_xf->xmx[xc] - xf[x_offsets[i] + xc]);
          float db = fabsf(cpu_xb->xmx[xc] - xb[x_offsets[i] + xc]);
          if (df > max_fwd) {
            max_fwd = df;
            max_fwd_row = (int)(xc / p7X_NXCELLS);
            max_fwd_cell = (int)(xc % p7X_NXCELLS);
          }
          if (db > max_bck) {
            max_bck = db;
            max_bck_row = (int)(xc / p7X_NXCELLS);
            max_bck_cell = (int)(xc % p7X_NXCELLS);
          }
        }

        if (max_fwd > 1e-3f || max_bck > 1e-3f ||
            fabsf(cpu_fwdsc - fb_scores[i * 2]) > 1e-3f ||
            fabsf(cpu_bcksc - fb_scores[i * 2 + 1]) > 1e-3f) {
          fprintf(stderr,
                  "NHMMER_GPU_COMPARE_FB win=%d n=%" PRId64 " len=%d "
                  "fwd_cpu=%.6f fwd_gpu=%.6f df=%.6f maxxf=%.6g@%d/%d "
                  "bck_cpu=%.6f bck_gpu=%.6f db=%.6f maxxb=%.6g@%d/%d "
                  "cpu_bck_own=%d\n",
                  i, (int64_t)windows[i].n, L,
                  cpu_fwdsc, fb_scores[i * 2], cpu_fwdsc - fb_scores[i * 2],
                  max_fwd, max_fwd_row, max_fwd_cell,
                  cpu_bcksc, fb_scores[i * 2 + 1], cpu_bcksc - fb_scores[i * 2 + 1],
                  max_bck, max_bck_row, max_bck_cell,
                  cpu_xb->has_own_scales);
          nprinted++;
        }
      }
    }
    p7_omx_Destroy(cpu_xf);
    p7_omx_Destroy(cpu_xb);
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


#endif /* HMMER_CUDA */
