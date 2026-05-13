/* nhmmer GPU pipeline: scratch arena lifecycle + GPU batch SSV/bias/F1 filter
 * + nucdb resident-batch preparation + launch-stat plumbing.
 *
 * This file owns the "first half" of the GPU search loop: everything that
 * happens after the SSV longtarget kernel produces merged windows but before
 * the scanning Viterbi / Forward-Backward parser stages take over. It is
 * plain C (no .cu) and is compiled by the host compiler; it includes
 * cuda/p7_cuda.h and calls the kernel-dispatch APIs declared there.
 */
#include <p7_config.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

#include "easel.h"
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

double
nhmmer_gpu_elapsed(const struct timespec *ts0, const struct timespec *ts1)
{
  return (ts1->tv_sec - ts0->tv_sec) + (ts1->tv_nsec - ts0->tv_nsec) * 1e-9;
}

void
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

void
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
int
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

int
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

int
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

int
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
int
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
	    info->t_batch_gather  = stats.gather_kernel_seconds;
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

int
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
                                        4, errbuf, errbuf_size,
                                        NULL, NULL, 0);
  if (status != eslOK) goto ERROR;

  {
    P7_CUDA_MSV_STATS stats;
    p7_cuda_engine_GetStats(info->cuda_engine, &stats);
    info->t_batch_gather  = stats.gather_kernel_seconds;
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

int
nhmmer_gpu_batch_filter_resident_gather(NHMMER_GPU_INFO *info, const uint8_t *d_dsq_base,
                                        const int *h_offsets, const int *h_src1_lengths,
                                        const int *h_src2_offsets, const int *h_lengths,
                                        int rc_flag,
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
                                              4, rc_flag, errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;

  {
    P7_CUDA_MSV_STATS stats;
    p7_cuda_engine_GetStats(info->cuda_engine, &stats);
    info->t_batch_gather  = stats.gather_kernel_seconds;
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

int
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
  int needs_gather = TRUE; /* v2 packed: always gather to unpack 2-bit → byte */

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
    int64_t start0;
    int64_t end0;
    int64_t guess;
    int found = FALSE;

    if (w->length <= 0) return eslFAIL;

    if (complementarity == p7_COMPLEMENT) {
      /* RC scan: global RC position w->n came from chunk c where
       * c*step < w->n <= c*step + chunk_len. Local position = w->n - c*step.
       * Forward position of scan pos p in chunk c = ci->seq_offset + ci->length - local_p.
       * Forward start of window = ci->seq_offset + ci->length - (w->n + w->length - 1 - c*step).
       * We must find the chunk first, then compute start0. */
      guess = ((int64_t)w->n - 1) / step;
      if (guess < 0) guess = 0;
      if (guess >= chunk_count) guess = chunk_count - 1;

      for (int delta = 0; delta <= 2 && !found; delta++) {
        for (int sign = -1; sign <= 1; sign += 2) {
          int64_t c = guess + sign * delta;
          if (delta == 0 && sign > -1) continue;
          if (c < 0 || c >= chunk_count) continue;

          const P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[chunk_start + c];
          int64_t local_start = (int64_t)w->n - c * step;
          int64_t local_end   = local_start + (int64_t)w->length - 1;
          if (local_start < 1 || local_end > ci->length) continue;

          /* Forward position (0-based) of the window region: */
          start0 = ci->seq_offset + (ci->length - local_end);
          end0   = start0 + (int64_t)w->length;
          int64_t rel = start0 - ci->seq_offset;

          if ((int64_t)ci->data_offset * 4 + rel > INT32_MAX || w->length > INT32_MAX)
            return eslFAIL;
          info->h_nucdb_window_offsets[i] = (int)((int64_t)ci->data_offset * 4 + rel);
          info->h_nucdb_window_lengths[i] = (int)w->length;
          info->h_nucdb_window_src1_lengths[i] = (int)w->length;
          info->h_nucdb_window_src2_offsets[i] = 0;
          found = TRUE;
          break;
        }
      }
      /* Fallback: linear scan for RC */
      for (int64_t c = 0; c < chunk_count && !found; c++) {
        const P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[chunk_start + c];
        int64_t local_start = (int64_t)w->n - c * step;
        int64_t local_end   = local_start + (int64_t)w->length - 1;
        if (local_start < 1 || local_end > ci->length) continue;

        start0 = ci->seq_offset + (ci->length - local_end);
        int64_t rel = start0 - ci->seq_offset;

        if ((int64_t)ci->data_offset * 4 + rel > INT32_MAX || w->length > INT32_MAX)
          return eslFAIL;
        info->h_nucdb_window_offsets[i] = (int)((int64_t)ci->data_offset * 4 + rel);
        info->h_nucdb_window_lengths[i] = (int)w->length;
        info->h_nucdb_window_src1_lengths[i] = (int)w->length;
        info->h_nucdb_window_src2_offsets[i] = 0;
        found = TRUE;
      }
      /* Two-chunk span for RC */
      if (!found) {
        for (int64_t c = 0; c < chunk_count - 1; c++) {
          const P7_NUCDB_CHUNK_IDX *ci = &ndb->chunk_idx[chunk_start + c];
          const P7_NUCDB_CHUNK_IDX *cj = &ndb->chunk_idx[chunk_start + c + 1];
          int64_t local_start = (int64_t)w->n - c * step;
          int64_t local_end   = local_start + (int64_t)w->length - 1;
          if (local_start < 1 || local_start > ci->length) continue;
          if (local_end <= ci->length) continue;

          /* Window spans chunk boundary in RC scan space.
           * Part 1 covers local RC positions [local_start .. ci->length] in chunk c.
           *   Forward range: ci->seq_offset + [ci->length - ci->length .. ci->length - local_start]
           *                = ci->seq_offset + [0 .. ci->length - local_start]
           * Part 2 covers local RC positions [1 .. local_end - ci->length] in chunk c+1.
           *   Forward range: cj->seq_offset + [cj->length - (local_end - ci->length) .. cj->length - 1]
           *
           * The gather kernel reverses: output[0] should read the highest RC scan position
           * (part 2's first forward position). Output[len_in_cj..] reads part 1.
           * src1 = part 2 forward range (it's the "earlier" in output after reversal)
           * src2 = part 1 forward range
           */
          int64_t len_in_c  = ci->length - local_start + 1;
          int64_t len_in_cj = (int64_t)w->length - len_in_c;

          /* Part 2: chunk c+1, local RC positions 1..len_in_cj
           * Forward start: cj->seq_offset + cj->length - len_in_cj */
          int64_t fwd_start_cj = cj->seq_offset + cj->length - len_in_cj;
          int64_t rel_cj = fwd_start_cj - cj->seq_offset;

          /* Part 1: chunk c, local RC positions local_start..ci->length
           * Forward start: ci->seq_offset + 0 = ci->seq_offset */
          int64_t fwd_start_c = ci->seq_offset;
          int64_t rel_c = 0;

          if ((int64_t)cj->data_offset * 4 + rel_cj > INT32_MAX ||
              (int64_t)ci->data_offset * 4 + rel_c > INT32_MAX ||
              w->length > INT32_MAX)
            return eslFAIL;

          /* For RC gather: output[k] reads position (L-1-k) from the logical
           * concatenation [src1 || src2]. src1 has len_in_cj elements (from part 2),
           * src2 has len_in_c elements (from part 1). */
          info->h_nucdb_window_offsets[i] = (int)((int64_t)cj->data_offset * 4 + rel_cj);
          info->h_nucdb_window_lengths[i] = (int)w->length;
          info->h_nucdb_window_src1_lengths[i] = (int)len_in_cj;
          info->h_nucdb_window_src2_offsets[i] = (int)((int64_t)ci->data_offset * 4 + rel_c);
          found = TRUE;
          needs_gather = TRUE;
          break;
        }
      }

      if (!found) return eslFAIL;
      continue;  /* skip the forward path below */
    }

    /* Forward strand: w->n is 1-based forward position. */
    start0 = (int64_t)w->n - 1;
    end0 = start0 + (int64_t)w->length;
    if (start0 < 0) return eslFAIL;
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
        if ((int64_t)ci->data_offset * 4 + rel > INT32_MAX || w->length > INT32_MAX)
          return eslFAIL;
        info->h_nucdb_window_offsets[i] = (int)((int64_t)ci->data_offset * 4 + rel);
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
      if ((int64_t)ci->data_offset * 4 + rel > INT32_MAX || w->length > INT32_MAX)
        return eslFAIL;
      info->h_nucdb_window_offsets[i] = (int)((int64_t)ci->data_offset * 4 + rel);
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
        if ((int64_t)ci->data_offset * 4 + rel1 > INT32_MAX ||
            (int64_t)cj->data_offset * 4 + rel2 > INT32_MAX ||
            w->length > INT32_MAX || len1 > INT32_MAX)
          return eslFAIL;
        info->h_nucdb_window_offsets[i] = (int)((int64_t)ci->data_offset * 4 + rel1);
        info->h_nucdb_window_lengths[i] = (int)w->length;
        info->h_nucdb_window_src1_lengths[i] = (int)len1;
        info->h_nucdb_window_src2_offsets[i] = (int)((int64_t)cj->data_offset * 4 + rel2);
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

int
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
  int needs_gather = TRUE; /* v2 packed: always gather */
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
                                            (complementarity == p7_COMPLEMENT) ? 1 : 0,
                                            errbuf, errbuf_size);
}


#endif /* HMMER_CUDA */
