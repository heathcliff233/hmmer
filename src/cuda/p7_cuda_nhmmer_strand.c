/* nhmmer GPU pipeline: per-strand orchestration.
 *
 * Drives the GPU pipeline for one strand of one target sequence. Two entry
 * points share most of the stage sequence (SSV longtarget -> batch SSV/F1
 * filter -> scanning Viterbi -> Forward prefilter or Forward/Backward parser
 * -> CPU worker pool) but differ in where the sequence comes from:
 *
 *   - process_strand: host-loaded ESL_SQ (FASTA path).
 *   - process_nucdb_strand: pre-digitised P7_NUCDB chunks, with a
 *     GPU-resident fast path when chunk geometry permits.
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

#ifdef HMMER_THREADS
#include <pthread.h>
#endif

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_dsqdata.h"
#include "esl_gumbel.h"
#include "esl_sq.h"
#include "esl_vectorops.h"

#include "hmmer.h"
#include "p7_nucdb.h"
#include "nhmmer_internal.h"

#ifdef HMMER_CUDA
#include "cuda/p7_cuda.h"
#include "cuda/nhmmer_cuda_internal.h"

int
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
  P7_HMM_WINDOW *gpu_merged_windows = NULL;
  P7_CUDA_SSV_LT_STATS ssv_stats;
  int gpu_nwindows = 0;

  P7_HMM_WINDOWLIST  msv_windowlist;
  msv_windowlist.windows = NULL;
  memset(&ssv_stats, 0, sizeof(ssv_stats));

  clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (info->scoredata->prefix_lengths == NULL)
    p7_hmm_ScoreDataComputeRest(om, info->scoredata);
  if (info->do_compare) {
    status = p7_cuda_SSVLongtarget(info->cuda_engine, info->cuda_msv,
                                   sq->dsq, L,
                                   info->scoredata->ssv_scores, om->abc->Kp,
                                   sc_thresh, om->scale_b,
                                   chunk_size, overlap,
                                   &gpu_windows, &gpu_nwindows,
                                   &ssv_stats,
                                   errbuf, errbuf_size);
  } else {
    status = p7_cuda_SSVLongtargetWindows(info->cuda_engine, info->cuda_msv,
                                          info->scoredata,
                                          sq->dsq, L,
                                          info->scoredata->ssv_scores, om->abc->Kp,
                                          sc_thresh, om->scale_b,
                                          om->max_length, chunk_size, overlap,
                                          &gpu_merged_windows, &gpu_nwindows,
                                          &ssv_stats,
                                          errbuf, errbuf_size);
  }
  if (status != eslOK) return status;
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_ssv += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
  nhmmer_gpu_record_ssv_launch(info, &ssv_stats);

  if (gpu_nwindows == 0) return eslOK;

  clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (info->do_compare) {
    status = nhmmer_gpu_windowlist_alloc(&msv_windowlist, gpu_nwindows);
    if (status != eslOK) return status;
    nhmmer_gpu_fill_ssv_windows(&msv_windowlist, gpu_windows, gpu_nwindows, (uint32_t) L);
    gpu_windows = NULL;
    p7_pli_ExtendAndMergeWindows(om, info->scoredata, &msv_windowlist, 0);
  } else {
    status = nhmmer_gpu_copy_windows(&msv_windowlist, gpu_merged_windows, gpu_nwindows);
    if (status != eslOK) return status;
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_merge += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  if (msv_windowlist.count == 0) {
    free(msv_windowlist.windows);
    return eslOK;
  }

  /* GPU batch SSV/bias/F1 gating: filter merged windows before thread dispatch */
  NHMMER_GPU_WINDOW_BATCH wb;
  const float *batch_bias_scores = NULL;
  memset(&wb, 0, sizeof(wb));

  clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (!info->do_cpu_postmsv && info->do_gpu_batch && msv_windowlist.count >= NHMMER_GPU_BATCH_MIN) {
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

    if (info->pli->do_biasfilter && nsurv > 0)
      batch_bias_scores = info->h_bias_scores;

    nhmmer_gpu_window_batch_free(&wb);
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_batch_filter += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  /* GPU scanning Viterbi longtarget: replaces CPU scanning Viterbi in workers */
  if (!info->do_cpu_postmsv && info->do_gpu_vit_lt && msv_windowlist.count > 0) {
    P7_HMM_WINDOWLIST vit_wl;
    vit_wl.windows = NULL;

    clock_gettime(CLOCK_MONOTONIC, &ts0);
    status = nhmmer_gpu_viterbi_longtarget(info, sq, &msv_windowlist, batch_bias_scores,
                                           batch_bias_scores != NULL, &vit_wl,
                                           errbuf, errbuf_size);
    if (status != eslOK) {
      fprintf(stderr, "GPU scanning Viterbi failed: %s\n", errbuf);
      if (vit_wl.windows) free(vit_wl.windows);
      goto ERROR;
    }

    free(msv_windowlist.windows);
    msv_windowlist.windows = NULL;
    batch_bias_scores = NULL;
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
    float   *gpu_xf = NULL, *gpu_xb = NULL, *gpu_fb_scores = NULL;
    int     *gpu_fb_statuses = NULL, *gpu_fb_L_eff = NULL;
    size_t  *gpu_fb_x_offsets = NULL;
    int      gpu_fb_ok = FALSE;
    int      use_skip_fwd = FALSE;
    int      use_compact_fb = info->do_gpu_fwd;
    if (use_compact_fb) {
      clock_gettime(CLOCK_MONOTONIC, &ts0);
      status = nhmmer_gpu_forward_backward_compact(info, sq, seq_id, complementarity,
                                                   NULL, 0, 0, NULL,
                                                   vit_wl.windows, vit_wl.count,
                                                   &fwd_survivors, &nfwd_surv,
                                                   &gpu_xf, &gpu_xb, &gpu_fb_scores,
                                                   &gpu_fb_statuses, &gpu_fb_x_offsets,
                                                   &gpu_fb_L_eff, errbuf, errbuf_size);
      free(vit_wl.windows);
      clock_gettime(CLOCK_MONOTONIC, &ts1);
      info->t_gpu_fb_parser += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
      if (status != eslOK) { free(fwd_survivors); goto ERROR; }
      gpu_fb_ok = TRUE;
      use_skip_fwd = TRUE;
    } else {
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
      use_skip_fwd = info->do_gpu_fwd && prefilter_xf != NULL;
    }
    info->n_fwd_survivor_windows += nfwd_surv;

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

    /* GPU Backward xmx handoff is still too sensitive at domain-envelope
     * boundaries. Keep production output on the exact CPU Backward parser
     * after the GPU Forward prefilter; leave the GPU FB batch path disabled
     * until exact coordinate parity is demonstrated.
     */
    int use_gpu_fb = use_skip_fwd && !gpu_fb_ok;
    if (use_gpu_fb && use_skip_fwd) {
      clock_gettime(CLOCK_MONOTONIC, &ts0);
      status = nhmmer_gpu_run_fb_parser_batch(info, sq, fwd_survivors, nfwd_surv,
                                              prefilter_xf, prefilter_fwdsc,
                                              &gpu_xf, &gpu_xb, &gpu_fb_scores,
                                              &gpu_fb_statuses, &gpu_fb_x_offsets,
                                              &gpu_fb_L_eff, errbuf, errbuf_size);
      clock_gettime(CLOCK_MONOTONIC, &ts1);
      info->t_gpu_fb_parser += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
      if (status != eslOK) { free(fwd_survivors); free(prefilter_xf); free(prefilter_fwdsc); goto ERROR; }
      gpu_fb_ok = TRUE;
      if (gpu_xf == prefilter_xf) prefilter_xf = NULL;
    }

    /* Compute per-window xf offsets for partitioning among workers */
    size_t *surv_xf_offsets = NULL;
    if (use_skip_fwd && !gpu_fb_ok) {
      ESL_ALLOC(surv_xf_offsets, sizeof(size_t) * (nfwd_surv + 1));
      surv_xf_offsets[0] = 0;
      for (int wi = 0; wi < nfwd_surv; wi++)
        surv_xf_offsets[wi + 1] = surv_xf_offsets[wi] + (size_t)(fwd_survivors[wi].length + 1) * p7X_NXCELLS;
    }

    /* Dispatch survivors to worker threads */
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    int ncpus_vlt = info->ncpus;
    if (ncpus_vlt < 1) ncpus_vlt = 1;
    if (ncpus_vlt > nfwd_surv) ncpus_vlt = nfwd_surv;

#ifdef HMMER_THREADS
    if (ncpus_vlt > 1) {
      NHMMER_GPU_WORKER *workers = NULL;
      pthread_t         *threads = NULL;
      pthread_mutex_t    work_mutex;
      int                nworkers = ncpus_vlt;
      int                next_window = 0;

      ESL_ALLOC(workers, sizeof(NHMMER_GPU_WORKER) * nworkers);
      ESL_ALLOC(threads, sizeof(pthread_t) * nworkers);
      memset(workers, 0, sizeof(NHMMER_GPU_WORKER) * nworkers);
      pthread_mutex_init(&work_mutex, NULL);

      for (i = 0; i < nworkers; i++) {
        status = nhmmer_gpu_worker_init(&workers[i], info);
        if (status != eslOK) { nworkers = i; goto VLT_THREAD_ERROR; }

        workers[i].sq              = sq;
        workers[i].worker_id       = i;
        workers[i].seq_id          = seq_id;
        workers[i].complementarity = complementarity;
        workers[i].nwindows        = nfwd_surv;
        workers[i].global_nwindows = nfwd_surv;
        workers[i].windows         = fwd_survivors;
        workers[i].work_mutex      = &work_mutex;
        workers[i].next_window     = &next_window;
        workers[i].use_gpu_fb      = gpu_fb_ok;
        workers[i].gpu_xf          = gpu_xf;
        workers[i].gpu_xb          = gpu_xb;
        workers[i].gpu_scores      = gpu_fb_ok ? gpu_fb_scores : NULL;
        workers[i].gpu_statuses    = gpu_fb_ok ? gpu_fb_statuses : NULL;
        workers[i].gpu_x_offsets   = gpu_fb_ok ? gpu_fb_x_offsets : NULL;
        workers[i].gpu_L_eff       = gpu_fb_ok ? gpu_fb_L_eff : NULL;
        workers[i].prefilter_xf       = (use_skip_fwd && !gpu_fb_ok) ? prefilter_xf : NULL;
        workers[i].prefilter_fwdsc    = (use_skip_fwd && !gpu_fb_ok) ? prefilter_fwdsc : NULL;
        workers[i].prefilter_x_offsets = (use_skip_fwd && !gpu_fb_ok) ? surv_xf_offsets : NULL;
        workers[i].prefilter_xf_offset = 0;
        /* P1: enable GPU domcorrection deferral only on the GPU FB parser
         * continuation path; other continuations stay on CPU. */
        workers[i].pli->do_gpu_domcorr = (gpu_fb_ok && info->do_gpu_domcorr) ? 1 : 0;
      }

      for (i = 1; i < nworkers; i++) {
        if (gpu_fb_ok)
          pthread_create(&threads[i], NULL, nhmmer_gpu_thread_func_post_fb, &workers[i]);
        else if (use_skip_fwd)
          pthread_create(&threads[i], NULL, nhmmer_gpu_thread_func_post_fwd, &workers[i]);
        else
          pthread_create(&threads[i], NULL, nhmmer_gpu_thread_func_post_vit, &workers[i]);
      }

      if (gpu_fb_ok)
        nhmmer_gpu_worker_process_post_fb(&workers[0]);
      else if (use_skip_fwd)
        nhmmer_gpu_worker_process_post_fwd(&workers[0]);
      else
        nhmmer_gpu_worker_process_post_vit(&workers[0]);

      for (i = 1; i < nworkers; i++)
        pthread_join(threads[i], NULL);

      status = workers[0].status;
      for (i = 1; i < nworkers && status == eslOK; i++)
        status = workers[i].status;

      /* P1: flush all workers' deferred 2nd-pass Forwards as one GPU batch
       * and patch hit fields. No-op when do_gpu_domcorr is off. */
      if (status == eslOK && gpu_fb_ok && info->do_gpu_domcorr) {
        status = nhmmer_gpu_flush_domcorr(info, workers, nworkers, errbuf, errbuf_size);
      }

      {
        double w_null = 0, w_bias = 0, w_bck = 0, w_domain = 0, w_output = 0;
        for (i = 0; i < nworkers; i++) {
          if (workers[i].pli->time_null   > w_null)   w_null   = workers[i].pli->time_null;
          if (workers[i].pli->time_bias   > w_bias)   w_bias   = workers[i].pli->time_bias;
          if (workers[i].pli->time_bck    > w_bck)    w_bck    = workers[i].pli->time_bck;
          if (workers[i].pli->time_domain > w_domain) w_domain = workers[i].pli->time_domain;
          if (workers[i].pli->time_output > w_output) w_output = workers[i].pli->time_output;
        }
        info->t_worker_null   += w_null;
        info->t_worker_bias   += w_bias;
        info->t_worker_bck    += w_bck;
        info->t_worker_domain += w_domain;
        info->t_worker_output += w_output;
      }

      for (i = 1; i < nworkers && status == eslOK; i++) {
        p7_tophits_Merge(info->th, workers[i].th);
        p7_pipeline_Merge(info->pli, workers[i].pli);
      }
      p7_tophits_Merge(info->th, workers[0].th);
      p7_pipeline_Merge(info->pli, workers[0].pli);

    VLT_THREAD_ERROR:
      for (i = 0; i < nworkers; i++)
        nhmmer_gpu_worker_destroy(&workers[i]);
      pthread_mutex_destroy(&work_mutex);
      free(workers);
      free(threads);
      free(fwd_survivors);
      free(prefilter_xf); free(prefilter_fwdsc); free(surv_xf_offsets);
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
      if (status != eslOK) { free(fwd_survivors); free(prefilter_xf); free(prefilter_fwdsc); free(surv_xf_offsets); free(gpu_xf); free(gpu_xb); free(gpu_fb_scores); free(gpu_fb_statuses); free(gpu_fb_x_offsets); free(gpu_fb_L_eff); return status; }

      w.sq              = sq;
      w.worker_id       = 0;
      w.seq_id          = seq_id;
      w.complementarity = complementarity;
      w.nwindows        = nfwd_surv;
      w.global_nwindows = nfwd_surv;
      w.windows         = fwd_survivors;
      w.use_gpu_fb      = gpu_fb_ok;
      w.gpu_xf          = gpu_xf;
      w.gpu_xb          = gpu_xb;
      w.gpu_scores      = gpu_fb_scores;
      w.gpu_statuses    = gpu_fb_statuses;
      w.gpu_x_offsets   = gpu_fb_x_offsets;
      w.gpu_L_eff       = gpu_fb_L_eff;
      w.prefilter_xf       = (use_skip_fwd && !gpu_fb_ok) ? prefilter_xf : NULL;
      w.prefilter_fwdsc    = (use_skip_fwd && !gpu_fb_ok) ? prefilter_fwdsc : NULL;
      w.prefilter_x_offsets = NULL;
      w.prefilter_xf_offset = 0;
      w.pli->do_gpu_domcorr = (gpu_fb_ok && info->do_gpu_domcorr) ? 1 : 0;

      if (gpu_fb_ok)
        nhmmer_gpu_worker_process_post_fb(&w);
      else if (use_skip_fwd)
        nhmmer_gpu_worker_process_post_fwd(&w);
      else
        nhmmer_gpu_worker_process_post_vit(&w);
      status = w.status;

      /* P1: flush deferred 2nd-pass Forwards. */
      if (status == eslOK && gpu_fb_ok && info->do_gpu_domcorr) {
        status = nhmmer_gpu_flush_domcorr(info, &w, 1, errbuf, errbuf_size);
      }

      if (status == eslOK) {
        p7_tophits_Merge(info->th, w.th);
        p7_pipeline_Merge(info->pli, w.pli);
      }

      info->t_worker_null   += w.pli->time_null;
      info->t_worker_bias   += w.pli->time_bias;
      info->t_worker_bck    += w.pli->time_bck;
      info->t_worker_domain += w.pli->time_domain;
      info->t_worker_output += w.pli->time_output;

      nhmmer_gpu_worker_destroy(&w);
      free(fwd_survivors);
      free(prefilter_xf); free(prefilter_fwdsc); free(surv_xf_offsets);
      free(gpu_xf); free(gpu_xb); free(gpu_fb_scores);
      free(gpu_fb_statuses); free(gpu_fb_x_offsets); free(gpu_fb_L_eff);
      clock_gettime(CLOCK_MONOTONIC, &ts1);
      info->t_cpu_workers += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
      return status;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &ts0);
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
      workers[i].worker_id       = i;
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
    if (status != eslOK) { free(msv_windowlist.windows); return status; }

    w.sq              = sq;
    w.worker_id       = 0;
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

/* Process one strand's worth of pre-built chunks from nucdb.
 * The default path keeps SSV windowing, F1, scanning Viterbi, and
 * Forward/Backward parser work on GPU before CPU domain processing. */
int
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

  /* Build an HMM window list from GPU-merged SSV results. In compare mode only,
   * raw GPU hits are converted to CPU windows and merged by the reference path. */
  P7_HMM_WINDOWLIST msv_windowlist;
  msv_windowlist.windows = NULL;
  msv_windowlist.count   = 0;
  msv_windowlist.size    = 0;

  /* We run SSV on each chunk as a "window" and then do batch filtering.
   * First, compute SSV threshold (same as in nhmmer_gpu_serial_loop). */
  float nullsc;
  p7_bg_SetLength(info->bg, om->max_length);
  p7_bg_NullOne(info->bg, NULL, om->max_length, &nullsc);

  float invP = esl_gumbel_invsurv(info->pli->F1, om->evparam[p7_MMU], om->evparam[p7_MLAMBDA]);
  uint8_t sc_thresh = (uint8_t)ceil(((nullsc + (invP * eslCONST_LOG2) + 3.0) * om->scale_b)
                                    + om->base_b + om->tec_b + om->tjb_b);

  int chunk_size = info->gpu_chunk_size > 0 ? info->gpu_chunk_size : NHMMER_GPU_CHUNK_SIZE;
  int overlap    = om->max_length;

  P7_CUDA_LT_WINDOW *gpu_windows = NULL;
  P7_HMM_WINDOW *gpu_merged_windows = NULL;
  P7_CUDA_SSV_LT_STATS ssv_stats;
  int gpu_nwindows = 0;
  memset(&ssv_stats, 0, sizeof(ssv_stats));

  /* Use GPU-resident path when nucdb is uploaded and chunks have sufficient overlap */
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  const uint8_t *d_nucdb = p7_cuda_engine_NucdbDevPtr(info->cuda_engine);
  if (info->scoredata->prefix_lengths == NULL)
    p7_hmm_ScoreDataComputeRest(om, info->scoredata);
  if (d_nucdb &&
      (int64_t)ndb->hdr.overlap >= (int64_t)om->max_length &&
      (int64_t)ndb->hdr.chunk_size == (int64_t)chunk_size)
  {
    status = nhmmer_gpu_ensure_nucdb_chunk_scratch(info, chunk_count);
    if (status != eslOK) return status;
    int *h_offsets = info->h_nucdb_chunk_offsets;
    int *h_lengths = info->h_nucdb_chunk_lengths;

    int ndb_step = (int)ndb->hdr.chunk_size - (int)ndb->hdr.overlap;
    if (ndb_step < 1) ndb_step = 1;

    for (int c = 0; c < chunk_count; c++) {
      P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[chunk_start + c];
      h_offsets[c] = (int)ci->data_offset;
      h_lengths[c] = ci->length;
    }

    if (info->do_compare) {
      status = p7_cuda_SSVLongtargetResident(info->cuda_engine, info->cuda_msv,
                                             d_nucdb, chunk_count,
                                             h_offsets, h_lengths,
                                             info->scoredata->ssv_scores, om->abc->Kp,
                                             sc_thresh, om->scale_b,
                                             ndb_step,
                                             &gpu_windows, &gpu_nwindows,
                                             &ssv_stats,
                                             errbuf, errbuf_size);
    } else {
      status = p7_cuda_SSVLongtargetResidentWindows(info->cuda_engine, info->cuda_msv,
                                                    info->scoredata,
                                                    d_nucdb, chunk_count,
                                                    h_offsets, h_lengths,
                                                    info->scoredata->ssv_scores, om->abc->Kp,
                                                    sc_thresh, om->scale_b,
                                                    om->max_length, ndb_step, sq->n,
                                                    &gpu_merged_windows, &gpu_nwindows,
                                                    &ssv_stats,
                                                    errbuf, errbuf_size);
    }
  }
  else
  {
    /* Fallback: use host dsq (kernel does its own chunking with overlap) */
    if (info->do_compare) {
      status = p7_cuda_SSVLongtarget(info->cuda_engine, info->cuda_msv,
                                     sq->dsq, sq->n,
                                     info->scoredata->ssv_scores, om->abc->Kp,
                                     sc_thresh, om->scale_b,
                                     chunk_size, overlap,
                                     &gpu_windows, &gpu_nwindows,
                                     &ssv_stats,
                                     errbuf, errbuf_size);
    } else {
      status = p7_cuda_SSVLongtargetWindows(info->cuda_engine, info->cuda_msv,
                                            info->scoredata,
                                            sq->dsq, sq->n,
                                            info->scoredata->ssv_scores, om->abc->Kp,
                                            sc_thresh, om->scale_b,
                                            om->max_length, chunk_size, overlap,
                                            &gpu_merged_windows, &gpu_nwindows,
                                            &ssv_stats,
                                            errbuf, errbuf_size);
    }
  }
  if (status != eslOK) return status;
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_ssv += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
  nhmmer_gpu_record_ssv_launch(info, &ssv_stats);

  if (gpu_nwindows == 0) return eslOK;

  clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (info->do_compare) {
    status = nhmmer_gpu_windowlist_alloc(&msv_windowlist, gpu_nwindows);
    if (status != eslOK) return status;
    nhmmer_gpu_fill_ssv_windows(&msv_windowlist, gpu_windows, gpu_nwindows, (uint32_t) sq->n);
    gpu_windows = NULL;
    p7_pli_ExtendAndMergeWindows(om, info->scoredata, &msv_windowlist, 0);
  } else {
    status = nhmmer_gpu_copy_windows(&msv_windowlist, gpu_merged_windows, gpu_nwindows);
    if (status != eslOK) return status;
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_merge += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  if (msv_windowlist.count == 0) {
    free(msv_windowlist.windows);
    return eslOK;
  }

  /* GPU batch SSV/bias/F1 gating */
  NHMMER_GPU_WINDOW_BATCH wb;
  const float *batch_bias_scores = NULL;
  int *resident_f1_offsets = NULL;
  int *resident_f1_lengths = NULL;
  int *resident_f1_src1_lengths = NULL;
  int *resident_f1_src2_offsets = NULL;
  int use_resident_f1 = FALSE;
  int use_resident_f1_gather = FALSE;
  memset(&wb, 0, sizeof(wb));

  clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (!info->do_cpu_postmsv && info->do_gpu_batch && msv_windowlist.count >= NHMMER_GPU_BATCH_MIN) {
    int nsurv = 0;
    if (d_nucdb != NULL) {
      status = nhmmer_gpu_try_map_nucdb_windows(info, ndb, chunk_start, chunk_count,
                                                complementarity,
                                                &msv_windowlist,
                                                &resident_f1_offsets,
                                                &resident_f1_lengths,
                                                &resident_f1_src1_lengths,
                                                &resident_f1_src2_offsets,
                                                &use_resident_f1_gather);
      if (status == eslOK)
        use_resident_f1 = TRUE;
      else {
        if (getenv("HMMER_NHMMER_GPU_TRACE_RESIDENT_F1") != NULL)
          fprintf(stderr, "NHMMER_GPU_RESIDENT_F1 fallback seq=%" PRId64 " comp=%d windows=%d status=%d\n",
                  (int64_t)seq_id, complementarity, msv_windowlist.count, status);
        status = eslOK;
      }
    }

    if (use_resident_f1) {
      if (use_resident_f1_gather)
        status = nhmmer_gpu_batch_filter_resident_gather(info, d_nucdb,
                                                         resident_f1_offsets,
                                                         resident_f1_src1_lengths,
                                                         resident_f1_src2_offsets,
                                                         resident_f1_lengths,
                                                         &msv_windowlist, &nsurv,
                                                         errbuf, errbuf_size);
      else {
        for (int wi = 0; wi < msv_windowlist.count; wi++)
          resident_f1_offsets[wi]--;
        status = nhmmer_gpu_batch_filter_resident(info, d_nucdb,
                                                  resident_f1_offsets, resident_f1_lengths,
                                                  &msv_windowlist, &nsurv,
                                                  errbuf, errbuf_size);
      }
      if (status != eslOK) goto ERROR;
    } else {
      status = nhmmer_gpu_window_batch_init(&wb, msv_windowlist.count);
      if (status != eslOK) goto ERROR;

      status = nhmmer_gpu_window_batch_pack(&wb, sq, &msv_windowlist);
      if (status != eslOK) { nhmmer_gpu_window_batch_free(&wb); goto ERROR; }

      status = nhmmer_gpu_batch_filter(info, &wb, &msv_windowlist, &nsurv, errbuf, errbuf_size);
      if (status != eslOK) { nhmmer_gpu_window_batch_free(&wb); goto ERROR; }
    }

    if (msv_windowlist.count == 0) {
      nhmmer_gpu_window_batch_free(&wb);
      free(msv_windowlist.windows);
      clock_gettime(CLOCK_MONOTONIC, &ts1);
      info->t_batch_filter += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
      return eslOK;
    }

    if (info->pli->do_biasfilter && nsurv > 0)
      batch_bias_scores = info->h_bias_scores;

    nhmmer_gpu_window_batch_free(&wb);
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_batch_filter += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  /* GPU scanning Viterbi longtarget */
  if (!info->do_cpu_postmsv && info->do_gpu_vit_lt && msv_windowlist.count > 0) {
    P7_HMM_WINDOWLIST vit_wl;
    vit_wl.windows = NULL;

    clock_gettime(CLOCK_MONOTONIC, &ts0);
    status = nhmmer_gpu_viterbi_longtarget(info, sq, &msv_windowlist, batch_bias_scores,
                                           batch_bias_scores != NULL, &vit_wl,
                                           errbuf, errbuf_size);
    if (status != eslOK) {
      if (vit_wl.windows) free(vit_wl.windows);
      goto ERROR;
    }

    free(msv_windowlist.windows);
    msv_windowlist.windows = NULL;
    batch_bias_scores = NULL;
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    info->t_vit_lt += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

    if (vit_wl.count == 0) {
      if (vit_wl.windows) free(vit_wl.windows);
      return eslOK;
    }

    /* GPU Forward prefilter + Backward parser before CPU domain workers. */
    info->n_vit_lt_windows_in  += msv_windowlist.count;
    info->n_vit_lt_windows_out += vit_wl.count;
    info->n_post_vit_windows   += vit_wl.count;
    P7_HMM_WINDOW *fwd_survivors = NULL;
    int nfwd_surv = 0;
    float *prefilter_xf = NULL;
    float *prefilter_fwdsc = NULL;
    float *gpu_xf = NULL, *gpu_xb = NULL, *gpu_fb_scores = NULL;
    int   *gpu_fb_statuses = NULL, *gpu_fb_L_eff = NULL;
    size_t *gpu_fb_x_offsets = NULL;
    int gpu_fb_ok = FALSE;
    int use_skip_fwd = FALSE;
    int use_compact_fb = info->do_gpu_fwd;

    if (use_compact_fb) {
      clock_gettime(CLOCK_MONOTONIC, &ts0);
      status = nhmmer_gpu_forward_backward_compact(info, sq, seq_id, complementarity,
                                                   ndb, chunk_start, chunk_count, d_nucdb,
                                                   vit_wl.windows, vit_wl.count,
                                                   &fwd_survivors, &nfwd_surv,
                                                   &gpu_xf, &gpu_xb, &gpu_fb_scores,
                                                   &gpu_fb_statuses, &gpu_fb_x_offsets,
                                                   &gpu_fb_L_eff, errbuf, errbuf_size);
      free(vit_wl.windows);
      clock_gettime(CLOCK_MONOTONIC, &ts1);
      info->t_gpu_fb_parser += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
      if (status != eslOK) { free(fwd_survivors); goto ERROR; }
      gpu_fb_ok = TRUE;
      use_skip_fwd = TRUE;
    } else {
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
      use_skip_fwd = info->do_gpu_fwd && prefilter_xf != NULL;
    }
    info->n_fwd_survivor_windows += nfwd_surv;

    if (nfwd_surv == 0) {
      free(fwd_survivors);
      free(prefilter_xf);
      free(prefilter_fwdsc);
      return eslOK;
    }

    int use_gpu_fb = use_skip_fwd && !gpu_fb_ok;

    if (use_gpu_fb && use_skip_fwd) {
      clock_gettime(CLOCK_MONOTONIC, &ts0);
      status = nhmmer_gpu_run_fb_parser_batch(info, sq, fwd_survivors, nfwd_surv,
                                              prefilter_xf, prefilter_fwdsc,
                                              &gpu_xf, &gpu_xb, &gpu_fb_scores,
                                              &gpu_fb_statuses, &gpu_fb_x_offsets,
                                              &gpu_fb_L_eff, errbuf, errbuf_size);
      clock_gettime(CLOCK_MONOTONIC, &ts1);
      info->t_gpu_fb_parser += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
      if (status != eslOK) {
        free(fwd_survivors); free(prefilter_xf); free(prefilter_fwdsc);
        goto ERROR;
      }
      gpu_fb_ok = TRUE;
      if (gpu_xf == prefilter_xf) prefilter_xf = NULL;
    }

    size_t *surv_xf_offsets = NULL;
    if (use_skip_fwd && !gpu_fb_ok) {
      ESL_ALLOC(surv_xf_offsets, sizeof(size_t) * (nfwd_surv + 1));
      surv_xf_offsets[0] = 0;
      for (int wi = 0; wi < nfwd_surv; wi++)
        surv_xf_offsets[wi + 1] = surv_xf_offsets[wi] + (size_t)(fwd_survivors[wi].length + 1) * p7X_NXCELLS;
    }

    int ncpus_vlt = info->ncpus;
    if (ncpus_vlt < 1) ncpus_vlt = 1;
    if (ncpus_vlt > nfwd_surv) ncpus_vlt = nfwd_surv;

    clock_gettime(CLOCK_MONOTONIC, &ts0);
#ifdef HMMER_THREADS
    if (ncpus_vlt > 1) {
      NHMMER_GPU_WORKER *workers = NULL;
      pthread_t         *threads = NULL;
      pthread_mutex_t    work_mutex;
      int                nworkers = ncpus_vlt;
      int                next_window = 0;

      ESL_ALLOC(workers, sizeof(NHMMER_GPU_WORKER) * nworkers);
      ESL_ALLOC(threads, sizeof(pthread_t) * nworkers);
      memset(workers, 0, sizeof(NHMMER_GPU_WORKER) * nworkers);
      pthread_mutex_init(&work_mutex, NULL);

      for (i = 0; i < nworkers; i++) {
        status = nhmmer_gpu_worker_init(&workers[i], info);
        if (status != eslOK) { nworkers = i; goto VLT2_THREAD_ERROR; }

        workers[i].sq              = sq;
        workers[i].ndb             = ndb;
        workers[i].worker_id       = i;
        workers[i].seq_id          = seq_id;
        workers[i].complementarity = complementarity;
        workers[i].nwindows        = nfwd_surv;
        workers[i].global_nwindows = nfwd_surv;
        workers[i].windows         = fwd_survivors;
        workers[i].work_mutex      = &work_mutex;
        workers[i].next_window     = &next_window;
        workers[i].use_gpu_fb      = gpu_fb_ok;
        workers[i].gpu_xf          = gpu_xf;
        workers[i].gpu_xb          = gpu_xb;
        workers[i].gpu_scores      = gpu_fb_ok ? gpu_fb_scores : NULL;
        workers[i].gpu_statuses    = gpu_fb_ok ? gpu_fb_statuses : NULL;
        workers[i].gpu_x_offsets   = gpu_fb_ok ? gpu_fb_x_offsets : NULL;
        workers[i].gpu_L_eff       = gpu_fb_ok ? gpu_fb_L_eff : NULL;
        workers[i].prefilter_xf       = (use_skip_fwd && !gpu_fb_ok) ? prefilter_xf : NULL;
        workers[i].prefilter_fwdsc    = (use_skip_fwd && !gpu_fb_ok) ? prefilter_fwdsc : NULL;
        workers[i].prefilter_x_offsets = (use_skip_fwd && !gpu_fb_ok) ? surv_xf_offsets : NULL;
        workers[i].prefilter_xf_offset = 0;
        workers[i].pli->do_gpu_domcorr = (gpu_fb_ok && info->do_gpu_domcorr) ? 1 : 0;
      }

      for (i = 1; i < nworkers; i++) {
        if (gpu_fb_ok)
          pthread_create(&threads[i], NULL, nhmmer_gpu_thread_func_post_fb, &workers[i]);
        else if (use_skip_fwd)
          pthread_create(&threads[i], NULL, nhmmer_gpu_thread_func_post_fwd, &workers[i]);
        else
          pthread_create(&threads[i], NULL, nhmmer_gpu_thread_func_post_vit, &workers[i]);
      }
      if (gpu_fb_ok)
        nhmmer_gpu_worker_process_post_fb(&workers[0]);
      else if (use_skip_fwd)
        nhmmer_gpu_worker_process_post_fwd(&workers[0]);
      else
        nhmmer_gpu_worker_process_post_vit(&workers[0]);
      for (i = 1; i < nworkers; i++)
        pthread_join(threads[i], NULL);

      status = workers[0].status;
      for (i = 1; i < nworkers && status == eslOK; i++)
        status = workers[i].status;

      /* P1: flush all workers' deferred 2nd-pass Forwards. */
      if (status == eslOK && gpu_fb_ok && info->do_gpu_domcorr) {
        status = nhmmer_gpu_flush_domcorr(info, workers, nworkers, errbuf, errbuf_size);
      }

      {
        double w_null = 0, w_bias = 0, w_bck = 0, w_domain = 0, w_output = 0;
        for (i = 0; i < nworkers; i++) {
          if (workers[i].pli->time_null   > w_null)   w_null   = workers[i].pli->time_null;
          if (workers[i].pli->time_bias   > w_bias)   w_bias   = workers[i].pli->time_bias;
          if (workers[i].pli->time_bck    > w_bck)    w_bck    = workers[i].pli->time_bck;
          if (workers[i].pli->time_domain > w_domain) w_domain = workers[i].pli->time_domain;
          if (workers[i].pli->time_output > w_output) w_output = workers[i].pli->time_output;
        }
        info->t_worker_null   += w_null;
        info->t_worker_bias   += w_bias;
        info->t_worker_bck    += w_bck;
        info->t_worker_domain += w_domain;
        info->t_worker_output += w_output;
      }

      for (i = 1; i < nworkers && status == eslOK; i++) {
        p7_tophits_Merge(info->th, workers[i].th);
        p7_pipeline_Merge(info->pli, workers[i].pli);
      }
      p7_tophits_Merge(info->th, workers[0].th);
      p7_pipeline_Merge(info->pli, workers[0].pli);

    VLT2_THREAD_ERROR:
      for (i = 0; i < nworkers; i++)
        nhmmer_gpu_worker_destroy(&workers[i]);
      pthread_mutex_destroy(&work_mutex);
      free(workers);
      free(threads);
      free(fwd_survivors);
      free(prefilter_xf); free(prefilter_fwdsc); free(surv_xf_offsets);
      free(gpu_xf); free(gpu_xb); free(gpu_fb_scores);
      free(gpu_fb_statuses); free(gpu_fb_x_offsets); free(gpu_fb_L_eff);
      clock_gettime(CLOCK_MONOTONIC, &ts1);
      info->t_cpu_workers += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
      return status;
    }
#endif

    {
      NHMMER_GPU_WORKER w;
      memset(&w, 0, sizeof(w));
      status = nhmmer_gpu_worker_init(&w, info);
      if (status != eslOK) {
        free(fwd_survivors); free(prefilter_xf); free(prefilter_fwdsc); free(surv_xf_offsets);
        free(gpu_xf); free(gpu_xb); free(gpu_fb_scores);
        free(gpu_fb_statuses); free(gpu_fb_x_offsets); free(gpu_fb_L_eff);
        return status;
      }

      w.sq              = sq;
      w.ndb             = ndb;
      w.worker_id       = 0;
      w.seq_id          = seq_id;
      w.complementarity = complementarity;
      w.nwindows        = nfwd_surv;
      w.global_nwindows = nfwd_surv;
      w.windows         = fwd_survivors;
      w.use_gpu_fb      = gpu_fb_ok;
      w.gpu_xf          = gpu_xf;
      w.gpu_xb          = gpu_xb;
      w.gpu_scores      = gpu_fb_scores;
      w.gpu_statuses    = gpu_fb_statuses;
      w.gpu_x_offsets   = gpu_fb_x_offsets;
      w.gpu_L_eff       = gpu_fb_L_eff;
      w.prefilter_xf       = (use_skip_fwd && !gpu_fb_ok) ? prefilter_xf : NULL;
      w.prefilter_fwdsc    = (use_skip_fwd && !gpu_fb_ok) ? prefilter_fwdsc : NULL;
      w.prefilter_x_offsets = NULL;
      w.prefilter_xf_offset = 0;
      w.pli->do_gpu_domcorr = (gpu_fb_ok && info->do_gpu_domcorr) ? 1 : 0;

      if (gpu_fb_ok)
        nhmmer_gpu_worker_process_post_fb(&w);
      else if (use_skip_fwd)
        nhmmer_gpu_worker_process_post_fwd(&w);
      else
        nhmmer_gpu_worker_process_post_vit(&w);
      status = w.status;

      /* P1: flush deferred 2nd-pass Forwards. */
      if (status == eslOK && gpu_fb_ok && info->do_gpu_domcorr) {
        status = nhmmer_gpu_flush_domcorr(info, &w, 1, errbuf, errbuf_size);
      }

      if (status == eslOK) {
        info->t_worker_null   += w.pli->time_null;
        info->t_worker_bias   += w.pli->time_bias;
        info->t_worker_bck    += w.pli->time_bck;
        info->t_worker_domain += w.pli->time_domain;
        info->t_worker_output += w.pli->time_output;
        p7_tophits_Merge(info->th, w.th);
        p7_pipeline_Merge(info->pli, w.pli);
      }

      nhmmer_gpu_worker_destroy(&w);
      free(fwd_survivors);
      free(prefilter_xf); free(prefilter_fwdsc); free(surv_xf_offsets);
      free(gpu_xf); free(gpu_xb); free(gpu_fb_scores);
      free(gpu_fb_statuses); free(gpu_fb_x_offsets); free(gpu_fb_L_eff);
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
      workers[i].worker_id       = i;
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
    w.worker_id       = 0;
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


#endif /* HMMER_CUDA */
