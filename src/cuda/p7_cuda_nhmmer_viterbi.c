/* nhmmer GPU pipeline: scanning Viterbi longtarget orchestration.
 *
 * Drives p7_cuda_ViterbiLongtarget* from the survivors of the F1 batch
 * filter. Computes per-window thresholds, dispatches the CUDA scanning
 * Viterbi kernel (with optional reuse of F1-resident device buffers),
 * sorts/extends/merges the resulting seeds, and returns sub-windows in
 * target coordinates for the Forward stage.
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
#include "esl_gumbel.h"
#include "esl_sq.h"
#include "esl_vectorops.h"

#include "hmmer.h"
#include "nhmmer_internal.h"

#ifdef HMMER_CUDA
#include "cuda/p7_cuda.h"
#include "cuda/nhmmer_cuda_internal.h"

/* GPU scanning Viterbi longtarget: replaces CPU p7_ViterbiFilter_longtarget.
 * Computes per-window thresholds, runs GPU kernel, converts output to window list.
 * Returns sub-windows in *ret_vit_wl (caller must free windows). */
int
nhmmer_gpu_viterbi_longtarget(NHMMER_GPU_INFO *info, const ESL_SQ *sq,
                              P7_HMM_WINDOWLIST *input_wl,
                              const float *precomputed_bias_scores,
                              int use_f1_resident,
                              P7_HMM_WINDOWLIST *ret_vit_wl,
                              char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int N = input_wl->count;
  P7_OPROFILE *om = info->om;
  P7_PIPELINE *pli = info->pli;
  float       *bias_scores = NULL;
  P7_CUDA_VIT_LT_WINDOW *gpu_windows = NULL;
  P7_HMM_WINDOW *gpu_merged_windows = NULL;
  P7_CUDA_VIT_LT_STATS cuda_stats;
  int          gpu_nwindows = 0;
  int          i;
  int          max_window_len = 80000;
  int          overlap_len = ESL_MIN(40000, om->max_length);
  int          used_f1_resident = FALSE;
  struct timespec ts0, ts1;

  ret_vit_wl->windows = NULL;
  ret_vit_wl->count   = 0;
  ret_vit_wl->size    = 0;
  memset(&cuda_stats, 0, sizeof(cuda_stats));

  if (N == 0) return eslOK;

  /* GPU bias filter scores (if enabled) — reuse batch infrastructure */
  if (pli->do_biasfilter && precomputed_bias_scores != NULL) {
    bias_scores = (float *) precomputed_bias_scores;
  } else if (pli->do_biasfilter) {
    NHMMER_GPU_WINDOW_BATCH wb;
    memset(&wb, 0, sizeof(wb));
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    status = nhmmer_gpu_window_batch_init(&wb, N);
    if (status != eslOK) goto ERROR;
    status = nhmmer_gpu_window_batch_pack(&wb, sq, input_wl);
    if (status != eslOK) { nhmmer_gpu_window_batch_free(&wb); goto ERROR; }

    ESL_ALLOC(bias_scores, sizeof(float) * N);
    status = p7_cuda_BiasFilterDsqdataChunk(info->cuda_engine, info->bg, &wb.chu, bias_scores,
                                             errbuf, errbuf_size);
    nhmmer_gpu_window_batch_free(&wb);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    info->t_vit_bias += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    if (status != eslOK) goto ERROR;
  }

  if (info->do_compare) {
    fprintf(stderr, "NHMMER_GPU_COMPARE_VIT_PARAMS scale_w=%.1f xw_e_move=%d base_w=%d nj=%.1f max_length=%d Q=%d M=%d F2=%.4g vmu=%.6f vlambda=%.6f B2=%d\n",
            om->scale_w, (int)om->xw[p7O_E][p7O_MOVE], (int)om->base_w, om->nj, om->max_length,
            p7O_NQW(om->M), om->M, pli->F2, om->evparam[p7_VMU], om->evparam[p7_VLAMBDA], pli->B2);
  }
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (use_f1_resident && pli->do_biasfilter && bias_scores == precomputed_bias_scores) {
    if (info->do_compare) {
      status = p7_cuda_ViterbiLongtargetFromF1(info->cuda_engine, info->cuda_msv,
                                               N,
                                               pli->B2, pli->F2,
                                               om->evparam[p7_VMU], om->evparam[p7_VLAMBDA],
                                               om->scale_w,
                                               (float)om->xw[p7O_E][p7O_MOVE],
                                               om->nj,
                                               (float)om->base_w, om->max_length,
                                               &gpu_windows, &gpu_nwindows,
                                               &cuda_stats,
                                               errbuf, errbuf_size);
    } else {
      status = p7_cuda_ViterbiLongtargetFromF1Windows(info->cuda_engine, info->cuda_msv,
                                                      info->scoredata,
                                                      input_wl->windows, N,
                                                      pli->B2, pli->F2,
                                                      om->evparam[p7_VMU], om->evparam[p7_VLAMBDA],
                                                      om->scale_w,
                                                      (float)om->xw[p7O_E][p7O_MOVE],
                                                      om->nj,
                                                      (float)om->base_w, om->max_length,
                                                      &gpu_merged_windows, &gpu_nwindows,
                                                      &cuda_stats,
                                                      errbuf, errbuf_size);
    }
    if (status != eslOK) {
      if (info->do_compare) {
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
                                           &cuda_stats,
                                           errbuf, errbuf_size);
      } else {
        status = p7_cuda_ViterbiLongtargetWindows(info->cuda_engine, info->cuda_msv,
                                                  info->scoredata,
                                                  sq->dsq, sq->n,
                                                  input_wl->windows, N,
                                                  bias_scores, pli->do_biasfilter,
                                                  pli->B2, pli->F2,
                                                  om->evparam[p7_VMU], om->evparam[p7_VLAMBDA],
                                                  om->scale_w,
                                                  (float)om->xw[p7O_E][p7O_MOVE],
                                                  om->nj,
                                                  (float)om->base_w, om->max_length,
                                                  &gpu_merged_windows, &gpu_nwindows,
                                                  &cuda_stats,
                                                  errbuf, errbuf_size);
      }
    } else {
      used_f1_resident = TRUE;
    }
  } else {
    if (info->do_compare) {
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
                                         &cuda_stats,
                                         errbuf, errbuf_size);
    } else {
      status = p7_cuda_ViterbiLongtargetWindows(info->cuda_engine, info->cuda_msv,
                                                info->scoredata,
                                                sq->dsq, sq->n,
                                                input_wl->windows, N,
                                                bias_scores, pli->do_biasfilter,
                                                pli->B2, pli->F2,
                                                om->evparam[p7_VMU], om->evparam[p7_VLAMBDA],
                                                om->scale_w,
                                                (float)om->xw[p7O_E][p7O_MOVE],
                                                om->nj,
                                                (float)om->base_w, om->max_length,
                                                &gpu_merged_windows, &gpu_nwindows,
                                                &cuda_stats,
                                                errbuf, errbuf_size);
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  if (used_f1_resident && status == eslOK)
    info->t_vit_f1_resident += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
  info->t_vit_cuda   += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
  info->t_vit_pack   += cuda_stats.pack_seconds;
  info->t_vit_h2d    += cuda_stats.h2d_seconds;
  info->t_vit_thresh += cuda_stats.threshold_kernel_seconds;
  info->t_vit_kernel += cuda_stats.kernel_seconds;
  info->t_vit_d2h    += cuda_stats.d2h_seconds;
  info->t_vit_alloc  += cuda_stats.alloc_seconds;
  info->t_vit_stream += cuda_stats.stream_seconds;
  info->vit_packed_bytes += cuda_stats.packed_bytes;
  nhmmer_gpu_record_vit_launch(info, &cuda_stats);
  /* Save GPU bias scores for comparison before freeing */
  float *saved_bias_scores = NULL;
  int16_t *gpu_thresholds = NULL;
  if (info->do_compare && bias_scores) {
    saved_bias_scores = (float *)malloc(sizeof(float) * N);
    if (saved_bias_scores) memcpy(saved_bias_scores, bias_scores, sizeof(float) * N);
  }
  if (info->do_compare) {
    gpu_thresholds = (int16_t *)malloc(sizeof(int16_t) * N);
    if (gpu_thresholds) {
      p7_cuda_ViterbiLongtarget_GetThresholds(info->cuda_engine, gpu_thresholds, N);
    }
  }
  if (bias_scores && bias_scores != precomputed_bias_scores) { free(bias_scores); bias_scores = NULL; }
  if (status != eslOK) { free(saved_bias_scores); free(gpu_thresholds); goto ERROR; }

  if (!info->do_compare) {
    if (gpu_nwindows == 0) return eslOK;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    free(ret_vit_wl->windows);
    status = nhmmer_gpu_copy_windows(ret_vit_wl, gpu_merged_windows, gpu_nwindows);
    if (status != eslOK) goto ERROR;
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    info->t_vit_extend += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    return eslOK;
  }

  if (gpu_nwindows == 0) {
    /* Still run comparison even when GPU produces 0 windows */
    if (info->do_compare) goto COMPARE_STAGE_B;
    return eslOK;
  }

  /* Count per-input-window GPU seeds for comparison */
  int *gpu_seeds_per_win = NULL;
  P7_CUDA_VIT_LT_WINDOW *saved_gpu_windows = NULL;
  int saved_gpu_nwindows = 0;
  if (info->do_compare) {
    gpu_seeds_per_win = (int *)calloc(N, sizeof(int));
    if (gpu_seeds_per_win) {
      for (i = 0; i < gpu_nwindows; i++)
        gpu_seeds_per_win[gpu_windows[i].window_id]++;
    }
    /* Save a copy for detailed position printing */
    saved_gpu_windows = (P7_CUDA_VIT_LT_WINDOW *)malloc(sizeof(P7_CUDA_VIT_LT_WINDOW) * gpu_nwindows);
    if (saved_gpu_windows) {
      memcpy(saved_gpu_windows, gpu_windows, sizeof(P7_CUDA_VIT_LT_WINDOW) * gpu_nwindows);
      saved_gpu_nwindows = gpu_nwindows;
    }
  }

  /* Viterbi CUDA now returns seeds compacted in parent-window order, with
   * per-window emission order preserved by the single DP group handling that
   * parent. Keep the CPU sort only for diagnostic comparison snapshots. */
  if (info->do_compare) {
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    qsort(gpu_windows, gpu_nwindows, sizeof(P7_CUDA_VIT_LT_WINDOW),
          nhmmer_gpu_vit_window_compare);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    info->t_vit_sort += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
  }

  clock_gettime(CLOCK_MONOTONIC, &ts0);
  free(ret_vit_wl->windows);
  status = nhmmer_gpu_windowlist_alloc(ret_vit_wl, gpu_nwindows);
  if (status != eslOK) goto ERROR;
  nhmmer_gpu_fill_vit_seed_windows(ret_vit_wl, gpu_windows, gpu_nwindows, input_wl);
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
        p7_hmmwindow_new(ret_vit_wl, ret_vit_wl->windows[i].id, new_n, 0, 0,
                         ESL_MIN(max_window_len, new_len), 0.0f,
                         p7_NOCOMPLEMENT, new_len);
      } while ((int)new_len > max_window_len);
    }
  }

  /* CPU p7_pli_postSSV_LongTarget() runs scanning Viterbi and window
   * extension in coordinates local to each parent MSV window, then converts
   * surviving windows back to target coordinates when dispatching them.
   * The GPU batch path must do the same; otherwise ExtendAndMerge can extend
   * each Viterbi seed against the whole chromosome chunk and massively inflate
   * downstream domain-definition work.
   */
  for (i = 0; i < ret_vit_wl->count; i++) {
    int win_id = ret_vit_wl->windows[i].id;
    ret_vit_wl->windows[i].n += input_wl->windows[win_id].n - 1;
    ret_vit_wl->windows[i].target_len = sq->n;
    ret_vit_wl->windows[i].id = 0;
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_vit_extend += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  /* --gpu-compare Stage B: compare GPU Viterbi window set to CPU */
  COMPARE_STAGE_B:
  if (info->do_compare) {
    P7_BG *bg = info->bg;
    int cpu_vit_total = 0;
    int gpu_raw_total = gpu_nwindows;  /* raw seeds before ExtendAndMerge */
    int n_printed = 0;

    P7_OMX *cmp_ox = p7_omx_Create(om->M, 0, om->max_length);
    if (cmp_ox) {
      for (i = 0; i < N; i++) {
        P7_HMM_WINDOWLIST cpu_vit_wl;
        p7_hmmwindow_init(&cpu_vit_wl);

        int window_len = (int)input_wl->windows[i].length;
        ESL_DSQ *subseq = sq->dsq + input_wl->windows[i].n - 1;
        float nullsc, bias_filtersc, filtersc;

        int loc_window_len = ESL_MIN(window_len, om->max_length);
        p7_bg_SetLength(bg, loc_window_len);
        p7_bg_NullOne(bg, subseq, loc_window_len, &nullsc);

        bias_filtersc = 0;
        if (pli->do_biasfilter) {
          float nullsc_win;
          p7_bg_SetLength(bg, window_len);
          p7_bg_NullOne(bg, subseq, window_len, &nullsc_win);
          p7_bg_FilterScore(bg, subseq, window_len, &bias_filtersc);
          bias_filtersc -= nullsc_win;
        }
        int F2_L = ESL_MIN(window_len, pli->B2);
        filtersc = nullsc + (bias_filtersc * ((F2_L > window_len) ? 1.0f : (float)F2_L / window_len));
        p7_oprofile_ReconfigRestLength(om, loc_window_len);
        p7_omx_GrowTo(cmp_ox, om->M, 0, window_len);
        p7_ViterbiFilter_longtarget(subseq, window_len, om, cmp_ox, filtersc, pli->F2, &cpu_vit_wl);

        /* Print per-window comparison for divergent windows */
        int cpu_raw_seeds = cpu_vit_wl.count;
        int gpu_raw_this = (gpu_seeds_per_win ? gpu_seeds_per_win[i] : 0);
        p7_pli_ExtendAndMergeWindows(om, info->scoredata, &cpu_vit_wl, 0.5);

        if (n_printed < 5 && (cpu_raw_seeds != gpu_raw_this)) {
          /* Print bias score comparison */
          float cpu_bias_raw = 0;
          if (pli->do_biasfilter) {
            p7_bg_SetLength(info->bg, window_len);
            p7_bg_FilterScore(info->bg, subseq, window_len, &cpu_bias_raw);
          }
          float gpu_bias_raw = saved_bias_scores ? saved_bias_scores[i] : 0;
          fprintf(stderr, "  bias_scores: cpu=%.4f gpu=%.4f delta=%.4f nullsc=%.4f filtersc=%.4f\n",
                  cpu_bias_raw, gpu_bias_raw, cpu_bias_raw - gpu_bias_raw, nullsc, filtersc);
          /* Compute CPU threshold */
          float cpu_invP = esl_gumbel_invsurv(pli->F2, om->evparam[p7_VMU], om->evparam[p7_VLAMBDA]);
          int16_t cpu_sc_thresh = (int16_t) ceil( ((filtersc + (eslCONST_LOG2 * cpu_invP) + 3.0) * om->scale_w)
                                   - (float)om->xw[p7O_E][p7O_MOVE] - (float)om->xw[p7O_C][p7O_MOVE] + (float)om->base_w);

          /* Compute actual GPU threshold using GPU bias scores (matches GPU kernel logic) */
          int16_t gpu_actual_thresh = cpu_sc_thresh;
          if (saved_bias_scores && pli->do_biasfilter) {
            float gpu_p1_loc = (float)loc_window_len / (float)(loc_window_len + 1);
            float gpu_nullsc_loc = (float)((double)loc_window_len * log((double)gpu_p1_loc)
                               + log(1.0 - (double)gpu_p1_loc));
            float gpu_p1_win = (float)window_len / (float)(window_len + 1);
            float gpu_nullsc_win = (float)((double)window_len * log((double)gpu_p1_win)
                               + log(1.0 - (double)gpu_p1_win));
            float gpu_bias_filtersc = saved_bias_scores[i] - gpu_nullsc_win;
            int F2_L_gpu = ESL_MIN(window_len, pli->B2);
            float gpu_ratio = (F2_L_gpu > window_len) ? 1.0f : (float)F2_L_gpu / (float)window_len;
            float gpu_filtersc = gpu_nullsc_loc + gpu_bias_filtersc * gpu_ratio;
            float gpu_pmove = (2.0f + om->nj) / ((float)loc_window_len + 2.0f + om->nj);
            float gpu_xw_c_move = roundf(om->scale_w * logf(gpu_pmove));
            double gpu_invP = om->evparam[p7_VMU] - log(-log(1.0 - (double)pli->F2)) / om->evparam[p7_VLAMBDA];
            gpu_actual_thresh = (int16_t)ceil(((double)gpu_filtersc + 0.69314718055994530942 * gpu_invP + 3.0) * (double)om->scale_w
                                   - (double)om->xw[p7O_E][p7O_MOVE] - (double)gpu_xw_c_move + (double)om->base_w);
          }

          fprintf(stderr, "NHMMER_GPU_COMPARE_VIT_WIN win=%d n=%ld len=%d gpu_raw=%d cpu_raw=%d"
                          " cpu_thresh=%d gpu_actual_thresh=%d delta=%d\n",
                  i, (long)input_wl->windows[i].n, window_len, gpu_raw_this, cpu_raw_seeds,
                  (int)cpu_sc_thresh, (int)gpu_actual_thresh, (int)cpu_sc_thresh - (int)gpu_actual_thresh);
          /* Print GPU seed positions for this window */
          if (saved_gpu_windows) {
            fprintf(stderr, "  GPU seeds:");
            for (int g = 0; g < saved_gpu_nwindows; g++) {
              if (saved_gpu_windows[g].window_id == i)
                fprintf(stderr, " pos=%d,k=%d", saved_gpu_windows[g].position, (int)saved_gpu_windows[g].model_k);
            }
            fprintf(stderr, "\n");
          }
          /* Print CPU seed positions */
          {
            P7_HMM_WINDOWLIST cpu_raw_wl;
            p7_hmmwindow_init(&cpu_raw_wl);
            p7_oprofile_ReconfigRestLength(om, loc_window_len);
            p7_omx_GrowTo(cmp_ox, om->M, 0, window_len);
            p7_ViterbiFilter_longtarget(subseq, window_len, om, cmp_ox, filtersc, pli->F2, &cpu_raw_wl);
            fprintf(stderr, "  CPU seeds:");
            for (int g = 0; g < cpu_raw_wl.count; g++)
              fprintf(stderr, " pos=%ld,k=%d", (long)cpu_raw_wl.windows[g].n, (int)cpu_raw_wl.windows[g].k);
            fprintf(stderr, "\n");
            if (cpu_raw_wl.windows) free(cpu_raw_wl.windows);
          }
          /* Run scalar GPU-algorithm emulation with CPU threshold and actual GPU threshold */
          {
            int scalar_seeds_cpu = 0, scalar_seeds_gpu = 0;
            int16_t actual_gpu_thresh = (gpu_thresholds ? gpu_thresholds[i] : 0);
            p7_oprofile_ReconfigRestLength(om, loc_window_len);
            nhmmer_gpu_scalar_viterbi_debug(om, subseq, window_len, filtersc, pli->F2, 0,
                                             &scalar_seeds_cpu, NULL, 0);
            if (actual_gpu_thresh != 0)
              nhmmer_gpu_scalar_viterbi_debug(om, subseq, window_len, filtersc, pli->F2, actual_gpu_thresh,
                                               &scalar_seeds_gpu, NULL, 0);
            fprintf(stderr, "  scalar_cpu_thresh=%d scalar_gpu_thresh=%d actual_gpu_thresh=%d (vs cpu=%d gpu=%d)\n",
                    scalar_seeds_cpu, scalar_seeds_gpu, (int)actual_gpu_thresh, cpu_raw_seeds, gpu_raw_this);
          }
          n_printed++;
        }

        cpu_vit_total += cpu_vit_wl.count;
        if (cpu_vit_wl.windows) free(cpu_vit_wl.windows);
      }

      fprintf(stderr, "NHMMER_GPU_COMPARE_VIT gpu_raw_seeds=%d gpu_merged=%d cpu_count=%d input_windows=%d\n",
              gpu_raw_total, ret_vit_wl->count, cpu_vit_total, N);

      p7_omx_Destroy(cmp_ox);
    }
  }
  if (gpu_seeds_per_win) { free(gpu_seeds_per_win); gpu_seeds_per_win = NULL; }
  if (saved_gpu_windows) { free(saved_gpu_windows); saved_gpu_windows = NULL; }
  if (saved_bias_scores) { free(saved_bias_scores); saved_bias_scores = NULL; }
  if (gpu_thresholds) { free(gpu_thresholds); gpu_thresholds = NULL; }

  return eslOK;

ERROR:
  if (bias_scores && bias_scores != precomputed_bias_scores) free(bias_scores);
  return status;
}


#endif /* HMMER_CUDA */
