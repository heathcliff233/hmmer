/* nhmmer GPU pipeline: async 2-slot strand pipeline.
 *
 * A slot holds the per-strand state that must outlive the main thread's
 * return from "run GPU pipeline": the FB parser D2H output buffers, the
 * CPU worker pool, the shell ESL_SQ, and error/timing state. The outer
 * loop in cuda/p7_cuda_nhmmer_search.c alternates between two slots so
 * that strand N+1's GPU pipeline can run while strand N's CPU workers
 * are still finishing.
 *
 * Lifecycle:
 *   init(slot)                    once per outer loop
 *   loop:
 *     retire(slot)                join background workers of whatever
 *                                 strand last used this slot; merge
 *                                 hits/pli into the shared info structs
 *     reset(slot)                 zero out per-strand fields
 *     run_strand_gpu_phase(...)   drive GPU, stash outputs into slot
 *     launch_workers(slot)        spawn background workers, return
 *   drain: retire each slot
 *   destroy(slot)
 *
 * Only one slot runs on the GPU at a time (the main thread driver serializes
 * that), so the single P7_CUDA_ENGINE and NHMMER_GPU_INFO host scratch
 * arrays are safely reused across slots. What IS duplicated per slot:
 *   - fwd_survivors / gpu_xf / gpu_xb / gpu_fb_* arrays (owned by slot)
 *   - worker pool (pthread array, mutex, per-worker P7_OMX etc.)
 *   - shell ESL_SQ (for CPU-side dsq slice materialization in workers)
 */
#include <p7_config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef HMMER_THREADS
#include <pthread.h>
#endif

#include "easel.h"
#include "esl_sq.h"

#include "hmmer.h"
#include "p7_nucdb.h"
#include "nhmmer_internal.h"

#ifdef HMMER_CUDA
#include "cuda/p7_cuda.h"
#include "cuda/nhmmer_cuda_internal.h"


static double
slot_elapsed(const struct timespec *ts0, const struct timespec *ts1)
{
  return (ts1->tv_sec - ts0->tv_sec) + (ts1->tv_nsec - ts0->tv_nsec) * 1e-9;
}


int
nhmmer_gpu_slot_init(NHMMER_GPU_SLOT *slot, NHMMER_GPU_INFO *info, int nworkers_max)
{
  int status = eslOK;
  memset(slot, 0, sizeof(*slot));
  if (nworkers_max < 1) nworkers_max = 1;

  ESL_ALLOC(slot->workers, sizeof(NHMMER_GPU_WORKER) * nworkers_max);
  memset(slot->workers, 0, sizeof(NHMMER_GPU_WORKER) * nworkers_max);
#ifdef HMMER_THREADS
  ESL_ALLOC(slot->threads, sizeof(pthread_t) * nworkers_max);
  memset(slot->threads, 0, sizeof(pthread_t) * nworkers_max);
  if (pthread_mutex_init(&slot->work_mutex, NULL) != 0) {
    status = eslESYS;
    goto ERROR;
  }
  slot->mutex_initialized = TRUE;
#endif
  slot->nworkers  = 0;
  slot->in_flight = FALSE;
  slot->has_work  = FALSE;
  slot->status    = eslOK;
  return eslOK;

ERROR:
  nhmmer_gpu_slot_destroy(slot);
  return status;
}


void
nhmmer_gpu_slot_destroy(NHMMER_GPU_SLOT *slot)
{
  if (slot == NULL) return;
  if (slot->in_flight) {
    fprintf(stderr, "nhmmer_gpu_slot_destroy: slot still in flight (call retire first)\n");
    abort();
  }
#ifdef HMMER_THREADS
  if (slot->mutex_initialized) {
    pthread_mutex_destroy(&slot->work_mutex);
    slot->mutex_initialized = FALSE;
  }
  free(slot->threads); slot->threads = NULL;
#endif
  if (slot->sq_shell) { esl_sq_Destroy(slot->sq_shell); slot->sq_shell = NULL; }
  free(slot->workers); slot->workers = NULL;
  free(slot->fwd_survivors);
  free(slot->gpu_xf); free(slot->gpu_xb); free(slot->gpu_fb_scores);
  free(slot->gpu_fb_statuses); free(slot->gpu_fb_x_offsets); free(slot->gpu_fb_L_eff);
  memset(slot, 0, sizeof(*slot));
}


/* Zero out per-strand fields between uses. Keeps the worker array and mutex. */
void
nhmmer_gpu_slot_reset(NHMMER_GPU_SLOT *slot)
{
  if (slot->in_flight) {
    fprintf(stderr, "nhmmer_gpu_slot_reset: slot still in flight (call retire first)\n");
    abort();
  }
  if (slot->sq_shell) { esl_sq_Destroy(slot->sq_shell); slot->sq_shell = NULL; }
  free(slot->fwd_survivors);  slot->fwd_survivors  = NULL;
  free(slot->gpu_xf);         slot->gpu_xf         = NULL;
  free(slot->gpu_xb);         slot->gpu_xb         = NULL;
  free(slot->gpu_fb_scores);  slot->gpu_fb_scores  = NULL;
  free(slot->gpu_fb_statuses);slot->gpu_fb_statuses= NULL;
  free(slot->gpu_fb_x_offsets);slot->gpu_fb_x_offsets= NULL;
  free(slot->gpu_fb_L_eff);   slot->gpu_fb_L_eff   = NULL;
  slot->nfwd_surv = 0;
  slot->has_work  = FALSE;
  slot->status    = eslOK;
  slot->next_window = 0;
  slot->nworkers    = 0;
  slot->seq_id         = 0;
  slot->complementarity = 0;
}


/* Spawn background worker threads for the FB post-processing. Does NOT run
 * any worker on the main thread — the main thread is reserved for the GPU
 * driver. Returns after pthread_create for every worker is issued.
 */
int
nhmmer_gpu_slot_launch_workers(NHMMER_GPU_SLOT *slot, NHMMER_GPU_INFO *info)
{
  int status = eslOK;

  if (!slot->has_work || slot->nfwd_surv <= 0) return eslOK;

  int nworkers = info->ncpus;
  if (nworkers < 1) nworkers = 1;
  if (nworkers > slot->nfwd_surv) nworkers = slot->nfwd_surv;
  slot->nworkers    = nworkers;
  slot->next_window = 0;

  for (int i = 0; i < nworkers; i++) {
    status = nhmmer_gpu_worker_init(&slot->workers[i], info);
    if (status != eslOK) {
      /* clean up the workers we already initialized */
      for (int j = 0; j < i; j++) nhmmer_gpu_worker_destroy(&slot->workers[j]);
      slot->nworkers = 0;
      return status;
    }

    slot->workers[i].sq              = slot->sq_shell;
    slot->workers[i].ndb             = slot->ndb;
    slot->workers[i].worker_id       = i;
    slot->workers[i].seq_id          = slot->seq_id;
    slot->workers[i].complementarity = slot->complementarity;
    slot->workers[i].nwindows        = slot->nfwd_surv;
    slot->workers[i].global_nwindows = slot->nfwd_surv;
    slot->workers[i].windows         = slot->fwd_survivors;
    slot->workers[i].use_gpu_fb      = TRUE;
    slot->workers[i].gpu_xf          = slot->gpu_xf;
    slot->workers[i].gpu_xb          = slot->gpu_xb;
    slot->workers[i].gpu_scores      = slot->gpu_fb_scores;
    slot->workers[i].gpu_statuses    = slot->gpu_fb_statuses;
    slot->workers[i].gpu_x_offsets   = slot->gpu_fb_x_offsets;
    slot->workers[i].gpu_L_eff       = slot->gpu_fb_L_eff;
#ifdef HMMER_THREADS
    slot->workers[i].work_mutex      = &slot->work_mutex;
    slot->workers[i].next_window     = &slot->next_window;
#endif
  }

  clock_gettime(CLOCK_MONOTONIC, &slot->ts_launch);

#ifdef HMMER_THREADS
  if (nworkers > 1) {
    for (int i = 0; i < nworkers; i++) {
      int rc = pthread_create(&slot->threads[i], NULL,
                              nhmmer_gpu_thread_func_post_fb, &slot->workers[i]);
      if (rc != 0) {
        /* join whatever we launched, then fall back to inline processing */
        for (int j = 0; j < i; j++) pthread_join(slot->threads[j], NULL);
        for (int j = 0; j < nworkers; j++) nhmmer_gpu_worker_destroy(&slot->workers[j]);
        slot->nworkers = 0;
        return eslESYS;
      }
    }
    slot->in_flight = TRUE;
    return eslOK;
  }
#endif

  /* Single worker: run it inline on the main thread. The pipeline is now
   * fully synchronous for this strand. */
  nhmmer_gpu_worker_process_post_fb(&slot->workers[0]);
  slot->in_flight = TRUE;  /* retire() will do the merge + teardown */
  return eslOK;
}


/* Join all background workers, accumulate worker timings + hits into the
 * shared info structs, tear down per-worker state, and free the slot's
 * owned FB parser outputs. Idempotent: safe to call on a slot that is not
 * in flight (returns immediately).
 */
int
nhmmer_gpu_slot_retire(NHMMER_GPU_SLOT *slot, NHMMER_GPU_INFO *info)
{
  int status = eslOK;
  if (!slot->in_flight) return eslOK;

  struct timespec ts_join0, ts_join1;
  clock_gettime(CLOCK_MONOTONIC, &ts_join0);

#ifdef HMMER_THREADS
  if (slot->nworkers > 1) {
    for (int i = 0; i < slot->nworkers; i++) {
      pthread_join(slot->threads[i], NULL);
    }
  }
#endif

  clock_gettime(CLOCK_MONOTONIC, &ts_join1);

  double wall_total = slot_elapsed(&slot->ts_launch, &ts_join1);
  double wall_wait  = slot_elapsed(&ts_join0,        &ts_join1);
  double overlap    = wall_total - wall_wait;
  if (overlap < 0) overlap = 0;

  info->t_cpu_workers   += wall_total;
  info->t_worker_wait   += wall_wait;
  info->t_overlap_saved += overlap;

  /* Merge worker status + timing + hits */
  status = slot->workers[0].status;
  for (int i = 1; i < slot->nworkers; i++)
    if (status == eslOK) status = slot->workers[i].status;

  {
    double w_null = 0, w_bias = 0, w_bck = 0, w_domain = 0, w_output = 0;
    for (int i = 0; i < slot->nworkers; i++) {
      if (slot->workers[i].pli->time_null   > w_null)   w_null   = slot->workers[i].pli->time_null;
      if (slot->workers[i].pli->time_bias   > w_bias)   w_bias   = slot->workers[i].pli->time_bias;
      if (slot->workers[i].pli->time_bck    > w_bck)    w_bck    = slot->workers[i].pli->time_bck;
      if (slot->workers[i].pli->time_domain > w_domain) w_domain = slot->workers[i].pli->time_domain;
      if (slot->workers[i].pli->time_output > w_output) w_output = slot->workers[i].pli->time_output;
    }
    info->t_worker_null   += w_null;
    info->t_worker_bias   += w_bias;
    info->t_worker_bck    += w_bck;
    info->t_worker_domain += w_domain;
    info->t_worker_output += w_output;
  }

  if (status == eslOK) {
    for (int i = 0; i < slot->nworkers; i++) {
      p7_tophits_Merge(info->th, slot->workers[i].th);
      p7_pipeline_Merge(info->pli, slot->workers[i].pli);
    }
  }

  for (int i = 0; i < slot->nworkers; i++)
    nhmmer_gpu_worker_destroy(&slot->workers[i]);

  /* Free per-strand owned buffers */
  if (slot->sq_shell) { esl_sq_Destroy(slot->sq_shell); slot->sq_shell = NULL; }
  free(slot->fwd_survivors);  slot->fwd_survivors  = NULL;
  free(slot->gpu_xf);         slot->gpu_xf         = NULL;
  free(slot->gpu_xb);         slot->gpu_xb         = NULL;
  free(slot->gpu_fb_scores);  slot->gpu_fb_scores  = NULL;
  free(slot->gpu_fb_statuses);slot->gpu_fb_statuses= NULL;
  free(slot->gpu_fb_x_offsets);slot->gpu_fb_x_offsets= NULL;
  free(slot->gpu_fb_L_eff);   slot->gpu_fb_L_eff   = NULL;
  slot->nfwd_surv = 0;
  slot->has_work  = FALSE;
  slot->in_flight = FALSE;
  slot->nworkers  = 0;
  slot->next_window = 0;
  slot->status    = status;
  return status;
}


#endif /* HMMER_CUDA */
