/* nhmmer GPU pipeline: per-strand orchestration (resident overlap nucdb only).
 *
 * Provides two entry points that share the same stage sequence:
 *   - nhmmer_gpu_run_strand_gpu_phase: runs SSV -> merge -> batch F1 gate ->
 *     scanning Viterbi -> FB parser, and stashes the FB parser outputs into
 *     a slot. Does NOT dispatch CPU workers; that's the caller's job via
 *     nhmmer_gpu_slot_launch_workers.
 *   - nhmmer_gpu_process_nucdb_strand: synchronous wrapper that creates a
 *     slot, runs the GPU phase, then runs and joins workers in one call.
 *     Only used for the CPU=1 single-threaded fallback (no async benefit).
 *
 * The 2-slot ring in cuda/p7_cuda_nhmmer_search.c drives the async path;
 * this file never retains slot state across calls.
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
#include "esl_gumbel.h"
#include "esl_sq.h"
#include "esl_vectorops.h"

#include "hmmer.h"
#include "p7_nucdb.h"
#include "nhmmer_internal.h"

#ifdef HMMER_CUDA
#include "cuda/p7_cuda.h"
#include "cuda/nhmmer_cuda_internal.h"

/* Drive the GPU pipeline for one strand on the resident overlap nucdb.
 * Stashes FB-parser outputs into the slot's owned buffers and sets
 * slot->has_work. Returns eslOK (even when no survivors — has_work == FALSE
 * in that case). The caller must already have slot->sq_shell populated.
 */
int
nhmmer_gpu_run_strand_gpu_phase(NHMMER_GPU_INFO *info, NHMMER_GPU_SLOT *slot,
                                const P7_NUCDB *ndb,
                                int chunk_start, int chunk_count,
                                int complementarity,
                                char *errbuf, int errbuf_size)
{
  int status = eslOK;
  P7_OPROFILE *om = info->om;
  const ESL_SQ *sq = slot->sq_shell;
  int64_t seq_id = slot->seq_id;

  struct timespec ts0, ts1;

  slot->has_work = FALSE;

  P7_HMM_WINDOWLIST msv_windowlist;
  msv_windowlist.windows = NULL;
  msv_windowlist.count   = 0;
  msv_windowlist.size    = 0;

  float nullsc;
  p7_bg_SetLength(info->bg, om->max_length);
  p7_bg_NullOne(info->bg, NULL, om->max_length, &nullsc);

  float invP = esl_gumbel_invsurv(info->pli->F1, om->evparam[p7_MMU], om->evparam[p7_MLAMBDA]);
  uint8_t sc_thresh = (uint8_t)ceil(((nullsc + (invP * eslCONST_LOG2) + 3.0) * om->scale_b)
                                    + om->base_b + om->tec_b + om->tjb_b);

  P7_HMM_WINDOW *gpu_merged_windows = NULL;
  P7_CUDA_SSV_LT_STATS ssv_stats;
  int gpu_nwindows = 0;
  memset(&ssv_stats, 0, sizeof(ssv_stats));

  const uint8_t *d_nucdb = p7_cuda_engine_NucdbDevPtr(info->cuda_engine);

  clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (info->scoredata->prefix_lengths == NULL)
    p7_hmm_ScoreDataComputeRest(om, info->scoredata);

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

  status = p7_cuda_SSVLongtargetResidentWindows(info->cuda_engine, info->cuda_msv,
                                                info->scoredata,
                                                d_nucdb, chunk_count,
                                                h_offsets, h_lengths,
                                                info->scoredata->ssv_scores, om->abc->Kp,
                                                sc_thresh, om->scale_b,
                                                om->max_length, ndb_step, sq->n,
                                                (complementarity == p7_COMPLEMENT) ? 1 : 0,
                                                &gpu_merged_windows, &gpu_nwindows,
                                                &ssv_stats,
                                                errbuf, errbuf_size);
  if (status != eslOK) return status;
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_ssv += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
  nhmmer_gpu_record_ssv_launch(info, &ssv_stats);

  if (gpu_nwindows == 0) return eslOK;

  clock_gettime(CLOCK_MONOTONIC, &ts0);
  status = nhmmer_gpu_copy_windows(&msv_windowlist, gpu_merged_windows, gpu_nwindows);
  if (status != eslOK) return status;
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_merge += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  if (msv_windowlist.count == 0) { free(msv_windowlist.windows); return eslOK; }

  /* GPU batch SSV/bias/F1 gating on resident sequence bytes */
  const float *batch_bias_scores = NULL;
  int *resident_f1_offsets = NULL;
  int *resident_f1_lengths = NULL;
  int *resident_f1_src1_lengths = NULL;
  int *resident_f1_src2_offsets = NULL;
  int use_resident_f1_gather = FALSE;

  info->n_f1_in_windows += msv_windowlist.count;

  clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (msv_windowlist.count >= NHMMER_GPU_BATCH_MIN) {
    int nsurv = 0;
    status = nhmmer_gpu_try_map_nucdb_windows(info, ndb, chunk_start, chunk_count,
                                              complementarity,
                                              &msv_windowlist,
                                              &resident_f1_offsets,
                                              &resident_f1_lengths,
                                              &resident_f1_src1_lengths,
                                              &resident_f1_src2_offsets,
                                              &use_resident_f1_gather);
    if (status != eslOK) {
      free(msv_windowlist.windows);
      strncpy(errbuf,
              "GPU batch F1 failed to map windows to resident nucdb (overlap too small or chunk geometry mismatch)",
              errbuf_size - 1);
      errbuf[errbuf_size - 1] = '\0';
      return status;
    }

    if (use_resident_f1_gather) {
      status = nhmmer_gpu_batch_filter_resident_gather(info, d_nucdb,
                                                       resident_f1_offsets,
                                                       resident_f1_src1_lengths,
                                                       resident_f1_src2_offsets,
                                                       resident_f1_lengths,
                                                       (complementarity == p7_COMPLEMENT) ? 1 : 0,
                                                       &msv_windowlist, &nsurv,
                                                       errbuf, errbuf_size);
    } else {
      for (int wi = 0; wi < msv_windowlist.count; wi++)
        resident_f1_offsets[wi]--;
      status = nhmmer_gpu_batch_filter_resident(info, d_nucdb,
                                                resident_f1_offsets, resident_f1_lengths,
                                                &msv_windowlist, &nsurv,
                                                errbuf, errbuf_size);
    }
    if (status != eslOK) { free(msv_windowlist.windows); return status; }

    if (msv_windowlist.count == 0) {
      free(msv_windowlist.windows);
      clock_gettime(CLOCK_MONOTONIC, &ts1);
      info->t_batch_filter += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
      return eslOK;
    }

    if (info->pli->do_biasfilter && nsurv > 0)
      batch_bias_scores = info->h_bias_scores;
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_batch_filter += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  /* GPU scanning Viterbi longtarget (FromF1: reuses resident device bytes) */
  P7_HMM_WINDOWLIST vit_wl;
  vit_wl.windows = NULL;

  clock_gettime(CLOCK_MONOTONIC, &ts0);
  status = nhmmer_gpu_viterbi_longtarget(info, sq, &msv_windowlist, batch_bias_scores,
                                         batch_bias_scores != NULL, &vit_wl,
                                         errbuf, errbuf_size);
  if (status != eslOK) {
    free(msv_windowlist.windows);
    if (vit_wl.windows) free(vit_wl.windows);
    return status;
  }

  int msv_count_in = msv_windowlist.count;
  free(msv_windowlist.windows);
  msv_windowlist.windows = NULL;
  batch_bias_scores = NULL;
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  info->t_vit_lt += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  if (vit_wl.count == 0) {
    if (vit_wl.windows) free(vit_wl.windows);
    return eslOK;
  }

  info->n_vit_lt_windows_in  += msv_count_in;
  info->n_vit_lt_windows_out += vit_wl.count;
  info->n_post_vit_windows   += vit_wl.count;

  /* GPU Forward+Backward parser (F3 gate on device; D2H compact xf/xb for survivors) */
  P7_HMM_WINDOW *fwd_survivors = NULL;
  int nfwd_surv = 0;
  float *gpu_xf = NULL, *gpu_xb = NULL, *gpu_fb_scores = NULL;
  int   *gpu_fb_statuses = NULL, *gpu_fb_L_eff = NULL;
  size_t *gpu_fb_x_offsets = NULL;

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
  if (status != eslOK) { free(fwd_survivors); return status; }
  info->n_fwd_survivor_windows += nfwd_surv;

  if (nfwd_surv == 0) {
    free(fwd_survivors);
    free(gpu_xf); free(gpu_xb); free(gpu_fb_scores);
    free(gpu_fb_statuses); free(gpu_fb_x_offsets); free(gpu_fb_L_eff);
    return eslOK;
  }

  /* Transfer ownership to the slot */
  slot->fwd_survivors    = fwd_survivors;
  slot->nfwd_surv        = nfwd_surv;
  slot->gpu_xf           = gpu_xf;
  slot->gpu_xb           = gpu_xb;
  slot->gpu_fb_scores    = gpu_fb_scores;
  slot->gpu_fb_statuses  = gpu_fb_statuses;
  slot->gpu_fb_x_offsets = gpu_fb_x_offsets;
  slot->gpu_fb_L_eff     = gpu_fb_L_eff;
  slot->complementarity  = complementarity;
  slot->has_work         = TRUE;
  return eslOK;
}

/* Synchronous wrapper: create a one-shot slot, run GPU, launch workers, join.
 * Kept because the top-level outer loop still uses this entry point for the
 * CPU=1 single-threaded path (no async benefit with zero background workers).
 */
int
nhmmer_gpu_process_nucdb_strand(NHMMER_GPU_INFO *info,
                                const P7_NUCDB *ndb,
                                int chunk_start, int chunk_count,
                                const ESL_SQ *sq, int complementarity,
                                int64_t seq_id,
                                char *errbuf, int errbuf_size)
{
  NHMMER_GPU_SLOT slot;
  memset(&slot, 0, sizeof(slot));
  int ncpus_vlt = info->ncpus;
  if (ncpus_vlt < 1) ncpus_vlt = 1;
  int status = nhmmer_gpu_slot_init(&slot, info, ncpus_vlt);
  if (status != eslOK) return status;

  slot.sq_shell        = (ESL_SQ *)sq;  /* borrowed; we won't let slot free it */
  int sq_was_external  = TRUE;
  slot.seq_id          = seq_id;
  slot.complementarity = complementarity;

  status = nhmmer_gpu_run_strand_gpu_phase(info, &slot, ndb, chunk_start, chunk_count,
                                           complementarity, errbuf, errbuf_size);
  if (status != eslOK) {
    if (sq_was_external) slot.sq_shell = NULL;
    nhmmer_gpu_slot_destroy(&slot);
    return status;
  }
  if (!slot.has_work) {
    if (sq_was_external) slot.sq_shell = NULL;
    nhmmer_gpu_slot_destroy(&slot);
    return eslOK;
  }

  status = nhmmer_gpu_slot_launch_workers(&slot, info);
  if (status != eslOK) {
    if (sq_was_external) slot.sq_shell = NULL;
    nhmmer_gpu_slot_destroy(&slot);
    return status;
  }
  status = nhmmer_gpu_slot_retire(&slot, info);

  if (sq_was_external) slot.sq_shell = NULL;  /* caller owns sq */
  nhmmer_gpu_slot_destroy(&slot);
  return status;
}

#endif /* HMMER_CUDA */
