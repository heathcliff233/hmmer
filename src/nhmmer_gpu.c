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

static double
nhmmer_gpu_elapsed(const struct timespec *ts0, const struct timespec *ts1)
{
  return (ts1->tv_sec - ts0->tv_sec) + (ts1->tv_nsec - ts0->tv_nsec) * 1e-9;
}

static int
nhmmer_gpu_windowlist_alloc(P7_HMM_WINDOWLIST *wl, int n)
{
  int status;
  wl->count = 0;
  wl->size  = ESL_MAX(n, 1);
  ESL_ALLOC(wl->windows, sizeof(P7_HMM_WINDOW) * wl->size);
  return eslOK;

ERROR:
  wl->windows = NULL;
  wl->size = wl->count = 0;
  return eslEMEM;
}

static void
nhmmer_gpu_fill_ssv_windows(P7_HMM_WINDOWLIST *wl, const P7_CUDA_LT_WINDOW *gpu_windows,
                            int n, uint32_t target_len)
{
  for (int i = 0; i < n; i++) {
    P7_HMM_WINDOW *w = wl->windows + i;
    w->id              = 0;
    w->n               = gpu_windows[i].target_start;
    w->fm_n            = 0;
    w->k               = (uint16_t) gpu_windows[i].model_end;
    w->length          = gpu_windows[i].model_end - gpu_windows[i].model_start + 1;
    w->score           = gpu_windows[i].score;
    w->complementarity = p7_NOCOMPLEMENT;
    w->target_len      = target_len;
  }
  wl->count = n;
}

static int
nhmmer_gpu_copy_windows(P7_HMM_WINDOWLIST *wl, const P7_HMM_WINDOW *windows, int n)
{
  int status;
  status = nhmmer_gpu_windowlist_alloc(wl, n);
  if (status != eslOK) return status;
  if (n > 0) memcpy(wl->windows, windows, sizeof(P7_HMM_WINDOW) * n);
  wl->count = n;
  return eslOK;
}

static void
nhmmer_gpu_fill_vit_seed_windows(P7_HMM_WINDOWLIST *wl, const P7_CUDA_VIT_LT_WINDOW *gpu_windows,
                                 int n, const P7_HMM_WINDOWLIST *input_wl)
{
  for (int i = 0; i < n; i++) {
    int win_id = gpu_windows[i].window_id;
    P7_HMM_WINDOW *w = wl->windows + i;
    w->id              = (uint32_t) win_id;
    w->n               = (uint32_t) gpu_windows[i].position;
    w->fm_n            = 0;
    w->k               = (uint16_t) gpu_windows[i].model_k;
    w->length          = 1;
    w->score           = 0.0f;
    w->complementarity = p7_NOCOMPLEMENT;
    w->target_len      = (uint32_t) input_wl->windows[win_id].length;
  }
  wl->count = n;
}

static int
nhmmer_gpu_nucdb_reconstruct_sq(const P7_NUCDB *ndb, const ESL_ALPHABET *abc,
                                int64_t si, int complementarity, ESL_SQ **ret_sq)
{
  const P7_NUCDB_SEQ_IDX *sidx;
  const char *seqname;
  ESL_SQ *sq;
  int c;

  if (!ndb || !abc || !ret_sq) return eslEINVAL;
  if (si < 0 || si >= (int64_t)ndb->hdr.nseq) return eslEINVAL;

  sidx = &ndb->seq_idx[si];
  seqname = ndb->name_blob + sidx->name_offset;

  sq = esl_sq_CreateDigital(abc);
  if (!sq) return eslEMEM;
  esl_sq_SetName(sq, seqname);
  esl_sq_GrowTo(sq, sidx->length);

  if (complementarity == p7_NOCOMPLEMENT) {
    int64_t step = (int64_t)ndb->hdr.chunk_size - (int64_t)ndb->hdr.overlap;
    if (step < 1) step = 1;
    for (c = 0; c < sidx->fwd_chunk_count; c++) {
      P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[sidx->fwd_chunk_start + c];
      uint8_t *chunk_dsq = ndb->chunk_data + ci->data_offset;
      int64_t copy_start = 0;
      int64_t copy_len   = ci->length;
      if (c > 0) {
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
    sq->start = 1;
    sq->end   = sq->n = sidx->length;
  } else {
    for (c = 0; c < sidx->rc_chunk_count; c++) {
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
        memcpy(sq->dsq + 1 + ci->seq_offset + copy_start,
               chunk_dsq + 1 + copy_start, copy_len);
    }
    sq->start = sq->n = sidx->length;
    sq->end   = 1;
  }
  sq->L = sq->n;
  sq->dsq[0] = eslDSQ_SENTINEL;
  sq->dsq[sq->n + 1] = eslDSQ_SENTINEL;
  *ret_sq = sq;
  return eslOK;
}

static int
nhmmer_gpu_nucdb_get_cached_sq(const P7_NUCDB *ndb, const ESL_ALPHABET *abc,
                               int64_t si, int complementarity,
                               ESL_SQ **ret_sq, int *ret_built)
{
  ESL_SQ **slot = NULL;
  int status;

  if (!ndb || !abc || !ret_sq) return eslEINVAL;
  if (ret_built) *ret_built = FALSE;
  if (si < 0 || si >= (int64_t)ndb->hdr.nseq) return eslEINVAL;

  if (complementarity == p7_NOCOMPLEMENT) slot = ndb->sq_cache_top;
  else                                    slot = ndb->sq_cache_rc;
  if (!slot) return eslEINVAL;

  if (slot[si] == NULL) {
    status = nhmmer_gpu_nucdb_reconstruct_sq(ndb, abc, si, complementarity, &slot[si]);
    if (status != eslOK) return status;
    if (ret_built) *ret_built = TRUE;
  }
  *ret_sq = slot[si];
  return eslOK;
}

static int
nhmmer_gpu_nucdb_create_seq_shell(const P7_NUCDB *ndb, const ESL_ALPHABET *abc,
                                  int64_t si, ESL_SQ **ret_sq)
{
  const P7_NUCDB_SEQ_IDX *sidx;
  const char *seqname;
  ESL_SQ *sq;
  int status;

  if (!ndb || !abc || !ret_sq) return eslEINVAL;
  if (si < 0 || si >= (int64_t)ndb->hdr.nseq) return eslEINVAL;

  sidx = &ndb->seq_idx[si];
  seqname = ndb->name_blob + sidx->name_offset;
  sq = esl_sq_CreateDigital(abc);
  if (!sq) return eslEMEM;
  free(sq->dsq);
  sq->dsq = NULL;
  sq->salloc = 0;
  status = esl_sq_SetName(sq, seqname);
  if (status != eslOK) { esl_sq_Destroy(sq); return status; }
  sq->start = 1;
  sq->end   = sidx->length;
  sq->n     = sidx->length;
  sq->L     = sidx->length;
  sq->C     = 0;
  sq->W     = sidx->length;
  *ret_sq = sq;
  return eslOK;
}

static int
nhmmer_gpu_vit_window_compare(const void *a, const void *b)
{
  const P7_CUDA_VIT_LT_WINDOW *wa = (const P7_CUDA_VIT_LT_WINDOW *)a;
  const P7_CUDA_VIT_LT_WINDOW *wb = (const P7_CUDA_VIT_LT_WINDOW *)b;

  if (wa->window_id != wb->window_id) return (wa->window_id < wb->window_id) ? -1 : 1;
  if (wa->position  != wb->position)  return (wa->position  < wb->position)  ? -1 : 1;
  if (wa->model_k   != wb->model_k)   return (wa->model_k   < wb->model_k)   ? -1 : 1;
  return 0;
}

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

/* Describe merged windows as a synthetic chunk without binding host sequence
 * pointers. Resident .nucdb parser batches gather sequence bytes on the GPU;
 * the parser API still needs the chunk's lengths/metadata for allocation,
 * reuse checks, and statistics. */
static int
nhmmer_gpu_window_batch_describe(NHMMER_GPU_WINDOW_BATCH *wb, P7_HMM_WINDOWLIST *wl)
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
    wb->dsq_ptrs[i] = NULL;
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

static void
nhmmer_gpu_record_ssv_launch(NHMMER_GPU_INFO *info, const P7_CUDA_SSV_LT_STATS *stats)
{
  if (stats == NULL || stats->grid_blocks <= 0) return;

  info->ssv_launches++;
  info->ssv_grid_blocks         = stats->grid_blocks;
  info->ssv_block_threads       = stats->block_threads;
  info->ssv_dynamic_smem        = stats->dynamic_smem_bytes;
  info->ssv_active_blocks_per_sm = stats->active_blocks_per_sm;
  info->ssv_active_warps_per_sm = stats->active_warps_per_sm;
  info->ssv_max_warps_per_sm    = stats->max_warps_per_sm;
  info->ssv_sm_count            = stats->sm_count;
  info->ssv_nchunks             = stats->nchunks;
  info->ssv_chunk_size          = stats->chunk_size;
  info->ssv_theoretical_occupancy += stats->theoretical_occupancy;
  info->ssv_grid_sm_coverage      += stats->grid_sm_coverage;
  info->ssv_kernel_seconds        += stats->kernel_seconds;
}

static void
nhmmer_gpu_record_vit_launch(NHMMER_GPU_INFO *info, const P7_CUDA_VIT_LT_STATS *stats)
{
  if (stats == NULL || stats->grid_blocks <= 0) return;

  info->vit_launches++;
  info->vit_grid_blocks         = stats->grid_blocks;
  info->vit_block_threads       = stats->block_threads;
  info->vit_dynamic_smem        = stats->dynamic_smem_bytes;
  info->vit_active_blocks_per_sm = stats->active_blocks_per_sm;
  info->vit_active_warps_per_sm = stats->active_warps_per_sm;
  info->vit_max_warps_per_sm    = stats->max_warps_per_sm;
  info->vit_sm_count            = stats->sm_count;
  info->vit_theoretical_occupancy += stats->theoretical_occupancy;
  info->vit_grid_sm_coverage      += stats->grid_sm_coverage;
  info->vit_device_active_seconds += stats->device_active_seconds;
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
  ESL_REALLOC(info->h_f1_survivor_idx,  sizeof(int)   * N);
  ESL_REALLOC(info->h_f1_survivor_bias, sizeof(float) * N);
  info->h_filter_alloc = N;
  return eslOK;

ERROR:
  return eslEMEM;
}

static int
nhmmer_gpu_ensure_nucdb_chunk_scratch(NHMMER_GPU_INFO *info, int n)
{
  int *new_offsets = NULL;
  int *new_lengths = NULL;

  if (n <= info->h_nucdb_chunk_alloc) return eslOK;

  new_offsets = (int *)realloc(info->h_nucdb_chunk_offsets, sizeof(int) * n);
  new_lengths = (int *)realloc(info->h_nucdb_chunk_lengths, sizeof(int) * n);
  if (!new_offsets || !new_lengths) {
    if (new_offsets) info->h_nucdb_chunk_offsets = new_offsets;
    if (new_lengths) info->h_nucdb_chunk_lengths = new_lengths;
    return eslEMEM;
  }
  info->h_nucdb_chunk_offsets = new_offsets;
  info->h_nucdb_chunk_lengths = new_lengths;
  info->h_nucdb_chunk_alloc = n;
  return eslOK;
}

static int
nhmmer_gpu_ensure_nucdb_window_scratch(NHMMER_GPU_INFO *info, int n)
{
  int *new_offsets = NULL;
  int *new_lengths = NULL;
  int *new_src1_lengths = NULL;
  int *new_src2_offsets = NULL;

  if (n <= info->h_nucdb_window_alloc) return eslOK;

  new_offsets = (int *)realloc(info->h_nucdb_window_offsets, sizeof(int) * n);
  new_lengths = (int *)realloc(info->h_nucdb_window_lengths, sizeof(int) * n);
  new_src1_lengths = (int *)realloc(info->h_nucdb_window_src1_lengths, sizeof(int) * n);
  new_src2_offsets = (int *)realloc(info->h_nucdb_window_src2_offsets, sizeof(int) * n);
  if (!new_offsets || !new_lengths || !new_src1_lengths || !new_src2_offsets) {
    if (new_offsets) info->h_nucdb_window_offsets = new_offsets;
    if (new_lengths) info->h_nucdb_window_lengths = new_lengths;
    if (new_src1_lengths) info->h_nucdb_window_src1_lengths = new_src1_lengths;
    if (new_src2_offsets) info->h_nucdb_window_src2_offsets = new_src2_offsets;
    return eslEMEM;
  }
  info->h_nucdb_window_offsets = new_offsets;
  info->h_nucdb_window_lengths = new_lengths;
  info->h_nucdb_window_src1_lengths = new_src1_lengths;
  info->h_nucdb_window_src2_offsets = new_src2_offsets;
  info->h_nucdb_window_alloc = n;
  return eslOK;
}

static int
nhmmer_gpu_ensure_parser_scratch(NHMMER_GPU_INFO *info, int n)
{
  int *new_seqidx = NULL;
  size_t *new_x_offsets = NULL;
  int *new_surv_src_idx = NULL;

  if (n <= info->h_parser_alloc) return eslOK;

  new_seqidx       = (int *)realloc(info->h_parser_seqidx, sizeof(int) * n);
  new_x_offsets    = (size_t *)realloc(info->h_parser_x_offsets, sizeof(size_t) * n);
  new_surv_src_idx = (int *)realloc(info->h_parser_surv_src_idx, sizeof(int) * n);
  if (!new_seqidx || !new_x_offsets || !new_surv_src_idx) {
    if (new_seqidx)       info->h_parser_seqidx = new_seqidx;
    if (new_x_offsets)    info->h_parser_x_offsets = new_x_offsets;
    if (new_surv_src_idx) info->h_parser_surv_src_idx = new_surv_src_idx;
    return eslEMEM;
  }
  info->h_parser_seqidx       = new_seqidx;
  info->h_parser_x_offsets    = new_x_offsets;
  info->h_parser_surv_src_idx = new_surv_src_idx;
  info->h_parser_alloc        = n;
  return eslOK;
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
  P7_OPROFILE *om = info->om;
  P7_PIPELINE *pli = info->pli;

  status = nhmmer_gpu_ensure_scratch(info, N);
  if (status != eslOK) return status;

  int   *survivor_idx  = info->h_f1_survivor_idx;
  float *survivor_bias = info->h_f1_survivor_bias;
  float *bias_scores   = info->h_bias_scores;

  status = p7_cuda_NhmmerF1GateDsqdataChunk(info->cuda_engine, info->cuda_msv,
                                            info->bg, &wb->chu, pli->do_biasfilter,
                                            pli->B1,
                                            om->evparam[p7_MMU], om->evparam[p7_MLAMBDA], pli->F1,
                                            survivor_idx, &nsurv,
                                            survivor_bias, NULL,
                                            4, errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;

  {
	    P7_CUDA_MSV_STATS stats;
	    p7_cuda_engine_GetStats(info->cuda_engine, &stats);
	    info->t_batch_h2d    = stats.h2d_seconds;
	    info->t_batch_kernel = stats.kernel_seconds;
	    info->t_batch_f1_gate = stats.f1_gate_kernel_seconds;
	    info->t_batch_compact = stats.f1_compact_kernel_seconds;
	    info->t_batch_d2h    = stats.d2h_seconds;
    info->batch_packed_bytes += 1;
    for (i = 0; i < wb->chu.N; i++)
      info->batch_packed_bytes += wb->chu.L[i] + 1;
  }

  /* --gpu-compare Stage A: compare GPU batch MSV/null/bias scores to CPU */
  if (info->do_compare) {
    P7_BG *bg = info->bg;
    for (i = 0; i < N; i++) {
      int    window_len = (int)wb->lengths[i];
      ESL_DSQ *subseq   = wb->chu.dsq[i];
      float  cpu_usc, cpu_nullsc, cpu_bias_filtersc, cpu_filtersc, cpu_seq_score;
      double cpu_P;

      p7_omx_GrowTo(pli->oxf, om->M, 0, window_len);
      p7_oprofile_ReconfigMSVLength(om, window_len);
      p7_MSVFilter(subseq, window_len, om, pli->oxf, &cpu_usc);
      p7_bg_SetLength(bg, window_len);
      p7_bg_NullOne(bg, subseq, window_len, &cpu_nullsc);

      int gpu_pass = 0;
      int cpu_pass = 0;

      for (int sj = 0; sj < nsurv; sj++) {
        if (survivor_idx[sj] == i) { gpu_pass = 1; break; }
      }

      if (pli->do_biasfilter) {
        p7_bg_FilterScore(bg, subseq, window_len, &cpu_bias_filtersc);
        cpu_bias_filtersc -= cpu_nullsc;
        int F1_L = ESL_MIN(window_len, pli->B1);
        cpu_filtersc = cpu_nullsc + (cpu_bias_filtersc * ((F1_L > window_len) ? 1.0f : (float)F1_L / window_len));
        cpu_seq_score = (cpu_usc - cpu_filtersc) / eslCONST_LOG2;
      } else {
        cpu_seq_score = (cpu_usc - cpu_nullsc) / eslCONST_LOG2;
      }
      cpu_P = esl_gumbel_surv(cpu_seq_score, om->evparam[p7_MMU], om->evparam[p7_MLAMBDA]);
      cpu_pass = (cpu_P <= pli->F1) ? 1 : 0;

      if (gpu_pass != cpu_pass)
        fprintf(stderr, "NHMMER_GPU_COMPARE_BATCH win=%d n=%" PRId64 " len=%d cpu_usc=%.4f cpu_null=%.4f cpu_ss=%.4f gpu_pass=%d cpu_pass=%d\n",
                i, (int64_t)wl->windows[i].n, window_len, cpu_usc, cpu_nullsc, cpu_seq_score, gpu_pass, cpu_pass);
    }
  }

  for (i = 0; i < nsurv; i++) {
    int src = survivor_idx[i];
    if (src < 0 || src >= N) { status = eslEINVAL; goto ERROR; }
    if (i != src)
      wl->windows[i] = wl->windows[src];
    if (pli->do_biasfilter) bias_scores[i] = survivor_bias[i];
  }

  wl->count = nsurv;
  *ret_nsurv = nsurv;
  return eslOK;

ERROR:
  *ret_nsurv = 0;
  return status;
}

static int
nhmmer_gpu_batch_filter_resident(NHMMER_GPU_INFO *info, const uint8_t *d_dsq_base,
                                 const int *h_offsets, const int *h_lengths,
                                 P7_HMM_WINDOWLIST *wl, int *ret_nsurv,
                                 char *errbuf, int errbuf_size)
{
  int       status;
  int       N = wl->count;
  int       nsurv = 0;
  int       i;
  P7_OPROFILE *om = info->om;
  P7_PIPELINE *pli = info->pli;

  status = nhmmer_gpu_ensure_scratch(info, N);
  if (status != eslOK) return status;

  int   *survivor_idx  = info->h_f1_survivor_idx;
  float *survivor_bias = info->h_f1_survivor_bias;
  float *bias_scores   = info->h_bias_scores;

  status = p7_cuda_NhmmerF1GateResident(info->cuda_engine, info->cuda_msv,
                                        info->bg, d_dsq_base,
                                        h_offsets, h_lengths, N,
                                        pli->do_biasfilter, pli->B1,
                                        om->evparam[p7_MMU], om->evparam[p7_MLAMBDA], pli->F1,
                                        survivor_idx, &nsurv,
                                        survivor_bias, NULL,
                                        4, errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;

  {
    P7_CUDA_MSV_STATS stats;
    p7_cuda_engine_GetStats(info->cuda_engine, &stats);
    info->t_batch_h2d     = stats.h2d_seconds;
    info->t_batch_kernel  = stats.kernel_seconds;
    info->t_batch_f1_gate = stats.f1_gate_kernel_seconds;
    info->t_batch_compact = stats.f1_compact_kernel_seconds;
    info->t_batch_d2h     = stats.d2h_seconds;
  }

  for (i = 0; i < nsurv; i++) {
    int src = survivor_idx[i];
    if (src < 0 || src >= N) { status = eslEINVAL; goto ERROR; }
    if (i != src)
      wl->windows[i] = wl->windows[src];
    if (pli->do_biasfilter) bias_scores[i] = survivor_bias[i];
  }

  wl->count = nsurv;
  *ret_nsurv = nsurv;
  return eslOK;

ERROR:
  *ret_nsurv = 0;
  return status;
}

static int
nhmmer_gpu_batch_filter_resident_gather(NHMMER_GPU_INFO *info, const uint8_t *d_dsq_base,
                                        const int *h_offsets, const int *h_src1_lengths,
                                        const int *h_src2_offsets, const int *h_lengths,
                                        P7_HMM_WINDOWLIST *wl, int *ret_nsurv,
                                        char *errbuf, int errbuf_size)
{
  int       status;
  int       N = wl->count;
  int       nsurv = 0;
  int       i;
  P7_OPROFILE *om = info->om;
  P7_PIPELINE *pli = info->pli;

  status = nhmmer_gpu_ensure_scratch(info, N);
  if (status != eslOK) return status;

  int   *survivor_idx  = info->h_f1_survivor_idx;
  float *survivor_bias = info->h_f1_survivor_bias;
  float *bias_scores   = info->h_bias_scores;

  status = p7_cuda_NhmmerF1GateResidentGather(info->cuda_engine, info->cuda_msv,
                                              info->bg, d_dsq_base,
                                              h_offsets, h_src1_lengths,
                                              h_src2_offsets, h_lengths, N,
                                              pli->do_biasfilter, pli->B1,
                                              om->evparam[p7_MMU], om->evparam[p7_MLAMBDA], pli->F1,
                                              survivor_idx, &nsurv,
                                              survivor_bias, NULL,
                                              4, errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;

  {
    P7_CUDA_MSV_STATS stats;
    p7_cuda_engine_GetStats(info->cuda_engine, &stats);
    info->t_batch_h2d     = stats.h2d_seconds;
    info->t_batch_kernel  = stats.kernel_seconds;
    info->t_batch_f1_gate = stats.f1_gate_kernel_seconds;
    info->t_batch_compact = stats.f1_compact_kernel_seconds;
    info->t_batch_d2h     = stats.d2h_seconds;
  }

  for (i = 0; i < nsurv; i++) {
    int src = survivor_idx[i];
    if (src < 0 || src >= N) { status = eslEINVAL; goto ERROR; }
    if (i != src)
      wl->windows[i] = wl->windows[src];
    if (pli->do_biasfilter) bias_scores[i] = survivor_bias[i];
  }

  wl->count = nsurv;
  *ret_nsurv = nsurv;
  return eslOK;

ERROR:
  *ret_nsurv = 0;
  return status;
}

static int
nhmmer_gpu_try_map_nucdb_windows(NHMMER_GPU_INFO *info, const P7_NUCDB *ndb,
                                 int chunk_start, int chunk_count,
                                 int complementarity,
                                 const P7_HMM_WINDOWLIST *wl,
                                 int **ret_offsets, int **ret_lengths,
                                 int **ret_src1_lengths, int **ret_src2_offsets,
                                 int *ret_needs_gather)
{
  int status;
  int64_t step;
  int needs_gather = FALSE;

  if (!info || !ndb || !wl || !ret_offsets || !ret_lengths || !ret_src1_lengths ||
      !ret_src2_offsets || !ret_needs_gather) return eslEINVAL;
  *ret_offsets = NULL;
  *ret_lengths = NULL;
  *ret_src1_lengths = NULL;
  *ret_src2_offsets = NULL;
  *ret_needs_gather = FALSE;
  if (wl->count <= 0) return eslOK;
  if (chunk_count <= 0 || ndb->hdr.chunk_size <= 0) return eslFAIL;

  status = nhmmer_gpu_ensure_nucdb_window_scratch(info, wl->count);
  if (status != eslOK) return status;

  step = (int64_t)ndb->hdr.chunk_size - (int64_t)ndb->hdr.overlap;
  if (step < 1) step = 1;

  for (int i = 0; i < wl->count; i++) {
    const P7_HMM_WINDOW *w = &wl->windows[i];
    int64_t start0 = (int64_t)w->n - 1;
    int64_t end0;
    int64_t guess;
    int found = FALSE;

    if (start0 < 0 || w->length <= 0) return eslFAIL;
    end0 = start0 + (int64_t)w->length;
    guess = start0 / step;
    if (guess < 0) guess = 0;
    if (guess >= chunk_count) guess = chunk_count - 1;

    for (int delta = 0; delta <= 2 && !found; delta++) {
      for (int sign = -1; sign <= 1; sign += 2) {
        int64_t c = guess + sign * delta;
        if (delta == 0 && sign > -1) continue;
        if (c < 0 || c >= chunk_count) continue;

        const P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[chunk_start + c];
        int64_t ci_start = ci->seq_offset;
        int64_t ci_end   = ci_start + ci->length;
        int64_t rel;

        if (start0 < ci_start || end0 > ci_end) continue;
        rel = start0 - ci_start;
        if (ci->data_offset + 1 + rel > INT32_MAX || w->length > INT32_MAX)
          return eslFAIL;
        info->h_nucdb_window_offsets[i] = (int)(ci->data_offset + 1 + rel);
        info->h_nucdb_window_lengths[i] = (int)w->length;
        info->h_nucdb_window_src1_lengths[i] = (int)w->length;
        info->h_nucdb_window_src2_offsets[i] = 0;
        found = TRUE;
        break;
      }
    }
    for (int64_t c = 0; c < chunk_count && !found; c++) {
      const P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[chunk_start + c];
      int64_t ci_start = ci->seq_offset;
      int64_t ci_end   = ci_start + ci->length;
      int64_t rel;

      if (start0 < ci_start || end0 > ci_end) continue;
      rel = start0 - ci_start;
      if (ci->data_offset + 1 + rel > INT32_MAX || w->length > INT32_MAX)
        return eslFAIL;
      info->h_nucdb_window_offsets[i] = (int)(ci->data_offset + 1 + rel);
      info->h_nucdb_window_lengths[i] = (int)w->length;
      info->h_nucdb_window_src1_lengths[i] = (int)w->length;
      info->h_nucdb_window_src2_offsets[i] = 0;
      found = TRUE;
    }
    if (!found) {
      for (int64_t c = 0; c < chunk_count - 1 && !found; c++) {
        const P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[chunk_start + c];
        const P7_NUCDB_CHUNK_IDX *cj = &ndb->chunk_idx[chunk_start + c + 1];
        int64_t ci_start = ci->seq_offset;
        int64_t ci_end   = ci_start + ci->length;
        int64_t cj_start = cj->seq_offset;
        int64_t cj_end   = cj_start + cj->length;
        int64_t rel1, len1, rel2;

        if (start0 < ci_start || start0 >= ci_end) continue;
        if (end0 <= ci_end) continue;
        if (end0 > cj_end || ci_end < cj_start) continue;
        rel1 = start0 - ci_start;
        len1 = ci_end - start0;
        rel2 = ci_end - cj_start;
        if (len1 <= 0 || len1 >= (int64_t)w->length || rel2 < 0) continue;
        if (ci->data_offset + 1 + rel1 > INT32_MAX ||
            cj->data_offset + 1 + rel2 > INT32_MAX ||
            w->length > INT32_MAX || len1 > INT32_MAX)
          return eslFAIL;
        info->h_nucdb_window_offsets[i] = (int)(ci->data_offset + 1 + rel1);
        info->h_nucdb_window_lengths[i] = (int)w->length;
        info->h_nucdb_window_src1_lengths[i] = (int)len1;
        info->h_nucdb_window_src2_offsets[i] = (int)(cj->data_offset + 1 + rel2);
        found = TRUE;
        needs_gather = TRUE;
      }
      if (!found) {
        if (getenv("HMMER_NHMMER_GPU_TRACE_RESIDENT_F1") != NULL) {
          fprintf(stderr,
                  "NHMMER_GPU_RESIDENT_F1 map_fail comp=%d logical_n=%" PRId64 " len=%" PRIu32
                  " mapped_start=%" PRId64 " mapped_end=%" PRId64 " step=%" PRId64
                  " guess=%" PRId64 " chunks=%d\n",
                  complementarity, (int64_t)w->n, w->length,
                  start0, end0, step, guess, chunk_count);
        }
        return eslFAIL;
      }
    }
  }

  *ret_offsets = info->h_nucdb_window_offsets;
  *ret_lengths = info->h_nucdb_window_lengths;
  *ret_src1_lengths = info->h_nucdb_window_src1_lengths;
  *ret_src2_offsets = info->h_nucdb_window_src2_offsets;
  *ret_needs_gather = needs_gather;
  return eslOK;
}

static int
nhmmer_gpu_prepare_parser_resident_batch(NHMMER_GPU_INFO *info, const P7_NUCDB *ndb,
                                         int chunk_start, int chunk_count,
                                         int complementarity,
                                         const uint8_t *d_nucdb,
                                         P7_HMM_WINDOW *windows, int nwindows,
                                         const void *batch_owner,
                                         char *errbuf, int errbuf_size)
{
  P7_HMM_WINDOWLIST tmp_wl;
  int *offsets = NULL;
  int *lengths = NULL;
  int *src1_lengths = NULL;
  int *src2_offsets = NULL;
  int needs_gather = FALSE;
  int status;

  if (!info || !ndb || !d_nucdb || !windows || nwindows <= 0) return eslEINVAL;

  tmp_wl.windows = windows;
  tmp_wl.count   = nwindows;
  tmp_wl.size    = nwindows;

  status = nhmmer_gpu_try_map_nucdb_windows(info, ndb, chunk_start, chunk_count,
                                            complementarity, &tmp_wl,
                                            &offsets, &lengths,
                                            &src1_lengths, &src2_offsets,
                                            &needs_gather);
  if (status != eslOK) return status;

  /* The parser batch always uses a gathered active batch. That keeps one
   * sentinel-prefixed sequence layout for both chunk-contained and
   * boundary-spanning windows and lets the existing parser API reuse it. */
  return p7_cuda_PrepareResidentWindowBatch(info->cuda_engine, d_nucdb,
                                            offsets, src1_lengths,
                                            src2_offsets, lengths, nwindows,
                                            batch_owner,
                                            errbuf, errbuf_size);
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
  const P7_NUCDB   *ndb;
  ESL_SQ           *slice_sq;
  ESL_DSQ          *slice_dsq;
  int64_t           slice_alloc;
  P7_HMM_WINDOW   *windows;
  int               nwindows;
  int64_t           seq_id;
  int               complementarity;
  int               status;
  /* GPU Forward prefilter results (non-owning pointer into shared xf buffer) */
  float            *prefilter_xf;       /* precomputed forward special cells from GPU */
  float            *prefilter_fwdsc;    /* precomputed forward scores from GPU */
  size_t            prefilter_xf_offset; /* byte offset into prefilter_xf for this worker's first window */
  size_t           *prefilter_x_offsets; /* per-window offsets for dynamic scheduling */
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
  int               worker_id;
  int               do_domain_trace;
#ifdef HMMER_THREADS
  pthread_mutex_t  *work_mutex;
#endif
  int              *next_window;
  int               global_nwindows;
} NHMMER_GPU_WORKER;

static int
nhmmer_gpu_nucdb_fill_slice(const P7_NUCDB *ndb, const ESL_ALPHABET *abc,
                            int64_t si, int complementarity,
                            uint64_t start, int length, const char *seqname,
                            ESL_SQ **ret_sq, ESL_DSQ **ret_dsq, int64_t *ret_alloc)
{
  const P7_NUCDB_SEQ_IDX *sidx;
  int chunk_start, chunk_count;
  int64_t need;
  int64_t seq_pos0;
  ESL_SQ *sq = *ret_sq;
  ESL_DSQ *dsq = *ret_dsq;
  int64_t alloc = *ret_alloc;

  if (!ndb || !abc || si < 0 || si >= (int64_t)ndb->hdr.nseq || start < 1 || length < 0)
    return eslEINVAL;

  sidx = &ndb->seq_idx[si];
  if (start + (uint64_t)length - 1 > (uint64_t)sidx->length) return eslEINVAL;

  if (sq == NULL) {
    sq = esl_sq_CreateDigital(abc);
    if (!sq) return eslEMEM;
    free(sq->dsq);
    sq->dsq = NULL;
    sq->salloc = 0;
    *ret_sq = sq;
  }
  if (sq->name == NULL || strcmp(sq->name, seqname) != 0) {
    int status = esl_sq_SetName(sq, seqname);
    if (status != eslOK) return status;
  }

  need = (int64_t)length + 2;
  if (alloc < need) {
    ESL_DSQ *tmp = realloc(dsq, sizeof(ESL_DSQ) * need);
    if (!tmp) return eslEMEM;
    dsq = tmp;
    alloc = need;
    *ret_dsq = dsq;
    *ret_alloc = alloc;
  }

  dsq[0] = eslDSQ_SENTINEL;
  dsq[length + 1] = eslDSQ_SENTINEL;
  memset(dsq + 1, eslDSQ_SENTINEL, sizeof(ESL_DSQ) * length);

  chunk_start = (complementarity == p7_NOCOMPLEMENT) ? sidx->fwd_chunk_start : sidx->rc_chunk_start;
  chunk_count = (complementarity == p7_NOCOMPLEMENT) ? sidx->fwd_chunk_count : sidx->rc_chunk_count;
  seq_pos0 = (int64_t)start - 1;

  for (int c = 0; c < chunk_count; c++) {
    P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[chunk_start + c];
    int64_t chunk_beg = ci->seq_offset;
    int64_t chunk_end = ci->seq_offset + ci->length;
    int64_t req_beg = seq_pos0;
    int64_t req_end = seq_pos0 + length;
    int64_t ov_beg = ESL_MAX(chunk_beg, req_beg);
    int64_t ov_end = ESL_MIN(chunk_end, req_end);
    if (ov_beg < ov_end) {
      int64_t dst = ov_beg - req_beg;
      int64_t src = ov_beg - chunk_beg;
      int64_t n   = ov_end - ov_beg;
      memcpy(dsq + 1 + dst, ndb->chunk_data + ci->data_offset + 1 + src, n);
    }
  }

  sq->dsq = dsq;
  sq->n = length;
  sq->L = sidx->length;
  sq->C = 0;
  sq->W = length;
  if (complementarity == p7_NOCOMPLEMENT) {
    sq->start = 1;
    sq->end = sidx->length;
  } else {
    sq->start = sidx->length;
    sq->end = 1;
  }
  sq->abc = abc;
  return eslOK;
}

static int
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

static void
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
#ifdef HMMER_THREADS
  w->work_mutex      = NULL;
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
  if (w->slice_sq) {
    w->slice_sq->dsq = NULL;
    esl_sq_Destroy(w->slice_sq);
  }
  free(w->slice_dsq);
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
typedef struct {
  float *xmx;
  int    allocXR;
  int    M;
  int    L;
  int    has_own_scales;
  float  totscale;
} NHMMER_OMX_BINDING;

static void nhmmer_gpu_BindOmxXmx(P7_OMX *ox, float *xmx, int M, int L,
                                  int has_own_scales, float totscale,
                                  NHMMER_OMX_BINDING *saved);
static void nhmmer_gpu_RestoreOmxXmx(P7_OMX *ox, const NHMMER_OMX_BINDING *saved);

static void
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
static void
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
static void
nhmmer_gpu_worker_process_post_fb(NHMMER_GPU_WORKER *w)
{
  int          i;
  int          overlap = 0;
  int          use_dynamic = (w->next_window != NULL);
  P7_HMM_WINDOW *window;

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

static void *
nhmmer_gpu_thread_func_post_fwd(void *arg)
{
  NHMMER_GPU_WORKER *w = (NHMMER_GPU_WORKER *)arg;
  nhmmer_gpu_worker_process_post_fwd(w);
  return NULL;
}

static void *
nhmmer_gpu_thread_func_post_fb(void *arg)
{
  NHMMER_GPU_WORKER *w = (NHMMER_GPU_WORKER *)arg;
  nhmmer_gpu_worker_process_post_fb(w);
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

static int
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

/* Debug: scalar reimplementation of cuda_viterbi_longtarget_kernel's DP logic.
 * Runs GPU algorithm on CPU, returns the number of seeds and optionally xE trace.
 * If override_thresh != 0, use that as the threshold instead of computing from filtersc.
 * NOTE: lazy-F additional passes simplified (runs full propagation without early exit). */
static int
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
        /* Full lazy-F: propagate D→D across all lanes */
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

/* GPU scanning Viterbi longtarget: replaces CPU p7_ViterbiFilter_longtarget.
 * Computes per-window thresholds, runs GPU kernel, converts output to window list.
 * Returns sub-windows in *ret_vit_wl (caller must free windows). */
static int
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

      if (gpu_fb_ok)
        nhmmer_gpu_worker_process_post_fb(&w);
      else if (use_skip_fwd)
        nhmmer_gpu_worker_process_post_fwd(&w);
      else
        nhmmer_gpu_worker_process_post_vit(&w);
      status = w.status;

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
 * The default path keeps SSV windowing, F1, scanning Viterbi, and
 * Forward/Backward parser work on GPU before CPU domain processing. */
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

      if (gpu_fb_ok)
        nhmmer_gpu_worker_process_post_fb(&w);
      else if (use_skip_fwd)
        nhmmer_gpu_worker_process_post_fwd(&w);
      else
        nhmmer_gpu_worker_process_post_vit(&w);
      status = w.status;

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
