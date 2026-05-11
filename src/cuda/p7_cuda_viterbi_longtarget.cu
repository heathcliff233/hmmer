#include "p7_cuda_internal.h"

#define VIT_LT_MAX_WINDOWS_PER_INPUT 64
#define VIT_LT_LANES                 8
#define VIT_LT_GROUPS_PER_WARP       (32 / VIT_LT_LANES)
#define VIT_LT_GROUPS_PER_BLOCK      4

__device__ static inline double
esl_gumbel_invsurv_device(double p, double mu, double lambda)
{
  return mu - log(-log(1.0 - p)) / lambda;
}

__global__ static void
cuda_compute_viterbi_thresholds_kernel(
    const float *null_scores, const float *bias_scores,
    const int *seqidx, const int *lengths, int nwindows, int do_biasfilter,
    int B2, float F2, float vmu, float vlambda,
    float scale_w, float xw_e_move, float nj, float base_w,
    int max_length, int16_t *sc_thresholds)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nwindows) return;

  int src_i = seqidx ? seqidx[i] : i;
  int window_len = lengths[src_i];
  int loc_window_len = (window_len < max_length) ? window_len : max_length;
  float p1_loc = (float)loc_window_len / (float)(loc_window_len + 1);
  float nullsc_loc = (float)((double)loc_window_len * log((double)p1_loc) + log(1.0 - (double)p1_loc));

  /* bias_scores[i] was computed using the full window_len, so we must subtract
   * null(window_len) — not null(loc_window_len) — to isolate the composition bias,
   * matching what the CPU does in p7_pli_postSSV_LongTarget(). */
  float p1_win = (float)window_len / (float)(window_len + 1);
  float nullsc_win = (float)((double)window_len * log((double)p1_win) + log(1.0 - (double)p1_win));

  int F2_L = (window_len < B2) ? window_len : B2;
  float filtersc;

  if (do_biasfilter) {
    float bias_filtersc = bias_scores[i] - nullsc_win;
    float ratio = (F2_L > window_len) ? 1.0f : (float)F2_L / (float)window_len;
    filtersc = nullsc_loc + bias_filtersc * ratio;
  } else {
    filtersc = nullsc_loc;
  }

  float pmove = (2.0f + nj) / ((float)loc_window_len + 2.0f + nj);
  float xw_c_move = roundf(scale_w * logf(pmove));

  double invP = esl_gumbel_invsurv_device((double)F2, (double)vmu, (double)vlambda);
  sc_thresholds[i] = (int16_t)ceil(((double)filtersc + 0.69314718055994530942 * invP + 3.0) * (double)scale_w
                                    - (double)xw_e_move - (double)xw_c_move + (double)base_w);
}

__device__ static inline int16_t
vlt_i16_add_sat(int16_t a, int16_t b)
{
  int v = (int) a + (int) b;
  if (v > 32767) return 32767;
  if (v < -32768) return -32768;
  return (int16_t) v;
}

__device__ static inline int
vlt_twv_idx(int t, int q, int lane, int Q)
{
  return (t == p7O_DD) ? (((7 * Q) + q) * 8 + lane) : (((q * 7) + t) * 8 + lane);
}

__device__ static inline int64_t
vlt_i64_min(int64_t a, int64_t b)
{
  return (a < b) ? a : b;
}

__device__ static inline int64_t
vlt_i64_max(int64_t a, int64_t b)
{
  return (a > b) ? a : b;
}

__global__ static void
cuda_viterbi_longtarget_kernel(
    const uint8_t *dsq, const int *offsets, const int *lengths,
    const int *seqidx, int nwindows,
    const int16_t *rwv, const int16_t *twv, int M, int Q, int Kp,
    float nj, float scale_w, int max_length,
    int16_t xw_e_loop, int16_t xw_e_move,
    int16_t base_w, int16_t ddbound_w,
    const int16_t *sc_thresholds,
    P7_CUDA_VIT_LT_WINDOW *d_windows, int *d_win_count, int max_windows)
{
  extern __shared__ int16_t vlt_mem[];
  int groups_per_block = blockDim.x / VIT_LT_LANES;
  int group = threadIdx.x / VIT_LT_LANES;
  int lane = threadIdx.x & (VIT_LT_LANES - 1);
  int warp_lane = threadIdx.x & 31;
  unsigned int mask = 0xffu << (warp_lane & ~(VIT_LT_LANES - 1));
  int bi = blockIdx.x * groups_per_block + group;
  int N = Q * VIT_LT_LANES;
  size_t group_stride = (size_t) N * 3 * 2;
  if (group >= groups_per_block || bi >= nwindows) return;
  int16_t *group_mem = vlt_mem + ((size_t) group * group_stride);
  int16_t *prev = group_mem;
  int16_t *curr = prev + (size_t) N * 3;

  int src_i = seqidx ? seqidx[bi] : bi;
  int L = lengths[src_i];
  const uint8_t *s = dsq + offsets[src_i];
  int16_t my_thresh = sc_thresholds[bi];

  /* Per-window length reconfiguration (matches CPU ReconfigRestLength) */
  int loc_L = (L < max_length) ? L : max_length;
  float pmove = (2.0f + nj) / ((float)loc_L + 2.0f + nj);
  int16_t xw_move = (int16_t)roundf(scale_w * logf(pmove));

  for (int q = 0; q < Q; q++) {
    int cell = (q + lane * Q) * 3;
    prev[cell + 0] = -32768;
    prev[cell + 1] = -32768;
    prev[cell + 2] = -32768;
  }

  int16_t xN = base_w;
  int16_t xB = vlt_i16_add_sat(xN, xw_move);
  int16_t xJ = -32768;
  int16_t xC = -32768;

  for (int i = 1; i <= L; i++) {
    uint8_t x = s[i];
    if (x >= Kp) continue;

    xB = (int16_t) __shfl_sync(mask, (int) xB, 0, VIT_LT_LANES);
    int16_t mpv = prev[((Q - 1) + lane * Q) * 3 + 0];
    int16_t dpv = prev[((Q - 1) + lane * Q) * 3 + 1];
    int16_t ipv = prev[((Q - 1) + lane * Q) * 3 + 2];
    mpv = (int16_t) __shfl_up_sync(mask, (int) mpv, 1, VIT_LT_LANES);
    dpv = (int16_t) __shfl_up_sync(mask, (int) dpv, 1, VIT_LT_LANES);
    ipv = (int16_t) __shfl_up_sync(mask, (int) ipv, 1, VIT_LT_LANES);
    if (lane == 0) mpv = dpv = ipv = -32768;

    int16_t dcv = -32768;
    int16_t xE_lane = -32768;
    int16_t dmax_lane = -32768;
    for (int q = 0; q < Q; q++) {
      int cell = (q + lane * Q) * 3;
      int16_t sv = vlt_i16_add_sat(xB, twv[vlt_twv_idx(p7O_BM, q, lane, Q)]);
      int16_t cand = vlt_i16_add_sat(mpv, twv[vlt_twv_idx(p7O_MM, q, lane, Q)]);
      if (cand > sv) sv = cand;
      cand = vlt_i16_add_sat(ipv, twv[vlt_twv_idx(p7O_IM, q, lane, Q)]);
      if (cand > sv) sv = cand;
      cand = vlt_i16_add_sat(dpv, twv[vlt_twv_idx(p7O_DM, q, lane, Q)]);
      if (cand > sv) sv = cand;
      sv = vlt_i16_add_sat(sv, rwv[((int) x * Q + q) * 8 + lane]);
      if (sv > xE_lane) xE_lane = sv;

      mpv = prev[cell + 0];
      dpv = prev[cell + 1];
      ipv = prev[cell + 2];

      curr[cell + 0] = sv;
      curr[cell + 1] = dcv;
      dcv = vlt_i16_add_sat(sv, twv[vlt_twv_idx(p7O_MD, q, lane, Q)]);
      if (dcv > dmax_lane) dmax_lane = dcv;
      cand = vlt_i16_add_sat(mpv, twv[vlt_twv_idx(p7O_MI, q, lane, Q)]);
      sv = vlt_i16_add_sat(ipv, twv[vlt_twv_idx(p7O_II, q, lane, Q)]);
      curr[cell + 2] = cand > sv ? cand : sv;
    }

    int xEi = (int) xE_lane;
    int dmaxi = (int) dmax_lane;
    for (int off = 4; off > 0; off >>= 1) {
      int other = __shfl_down_sync(mask, xEi, off, VIT_LT_LANES);
      if (lane + off < VIT_LT_LANES && other > xEi) xEi = other;
      other = __shfl_down_sync(mask, dmaxi, off, VIT_LT_LANES);
      if (lane + off < VIT_LT_LANES && other > dmaxi) dmaxi = other;
    }

    int16_t xE = (int16_t) __shfl_sync(mask, xEi, 0, VIT_LT_LANES);

    if (xE >= my_thresh) {
      /* Emit windows for ALL model positions k where M[k] == xE */
      for (int q = 0; q < Q; q++) {
        int cell = (q + lane * Q) * 3;
        if (curr[cell + 0] == xE) {
          int k = q + Q * lane + 1;
          if (k <= M) {
            int widx = atomicAdd(d_win_count + bi, 1);
            if (widx < VIT_LT_MAX_WINDOWS_PER_INPUT) {
              int out = bi * VIT_LT_MAX_WINDOWS_PER_INPUT + widx;
              if (out < max_windows) {
                d_windows[out].window_id = bi;
                d_windows[out].position  = i;
                d_windows[out].model_k   = (int16_t) k;
                d_windows[out].pad       = 0;
              }
            }
          }
        }
      }

      /* Reset DP state */
      for (int q = 0; q < Q; q++) {
        int cell = (q + lane * Q) * 3;
        curr[cell + 0] = -32768;
        curr[cell + 1] = -32768;
        curr[cell + 2] = -32768;
      }
      if (lane == 0) {
        xN = base_w;
        xB = vlt_i16_add_sat(xN, xw_move);
        xJ = -32768;
        xC = -32768;
      }
    } else {
      if (lane == 0) {
        xN = xN;  /* xw_n_loop = 0 (3nat approx) */
        int16_t c2 = vlt_i16_add_sat(xE, xw_e_move);
        xC = xC > c2 ? xC : c2;  /* xw_c_loop = 0 */
        int16_t j2 = vlt_i16_add_sat(xE, xw_e_loop);
        xJ = xJ > j2 ? xJ : j2;  /* xw_j_loop = 0 */
        int16_t b1 = vlt_i16_add_sat(xJ, xw_move);
        int16_t b2 = vlt_i16_add_sat(xN, xw_move);
        xB = b1 > b2 ? b1 : b2;
      }
      xB = (int16_t) __shfl_sync(mask, (int) xB, 0, VIT_LT_LANES);

      /* Lazy-F D-state propagation */
      int dmax_all = __shfl_sync(mask, dmaxi, 0, VIT_LT_LANES);
      if (dmax_all + ddbound_w > xB) {
        dcv = (int16_t) __shfl_up_sync(mask, (int) dcv, 1, VIT_LT_LANES);
        if (lane == 0) dcv = -32768;
        for (int q = 0; q < Q; q++) {
          int cell = (q + lane * Q) * 3;
          if (dcv > curr[cell + 1]) curr[cell + 1] = dcv;
          dcv = vlt_i16_add_sat(curr[cell + 1], twv[vlt_twv_idx(p7O_DD, q, lane, Q)]);
        }
        int completed;
        do {
          completed = 1;
          dcv = (int16_t) __shfl_up_sync(mask, (int) dcv, 1, VIT_LT_LANES);
          if (lane == 0) dcv = -32768;
          for (int q = 0; q < Q; q++) {
            int cell = (q + lane * Q) * 3;
            int gt = (dcv > curr[cell + 1]);
            if (!__any_sync(mask, gt)) {
              completed = 0;
              break;
            }
            if (gt) curr[cell + 1] = dcv;
            dcv = vlt_i16_add_sat(curr[cell + 1], twv[vlt_twv_idx(p7O_DD, q, lane, Q)]);
          }
        } while (__all_sync(mask, completed));
      } else {
        dcv = (int16_t) __shfl_up_sync(mask, (int) dcv, 1, VIT_LT_LANES);
        if (lane == 0) dcv = -32768;
        curr[lane * Q * 3 + 1] = dcv;
      }
    }

    int16_t *tmp = prev;
    prev = curr;
    curr = tmp;
	}
}

__global__ static void
cuda_viterbi_longtarget_prefix_kernel(const int *counts, int nwindows, int *offsets)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  int total = 0;
  for (int w = 0; w < nwindows; w++) {
    int cnt = counts[w];
    if (cnt > VIT_LT_MAX_WINDOWS_PER_INPUT) {
      offsets[nwindows] = -1;
      return;
    }
    offsets[w] = total;
    total += cnt;
  }
  offsets[nwindows] = total;
}

__global__ static void
cuda_viterbi_longtarget_compact_kernel(const P7_CUDA_VIT_LT_WINDOW *src,
                                       const int *counts, const int *offsets,
                                       int nwindows, P7_CUDA_VIT_LT_WINDOW *dst)
{
  int w = blockIdx.x;
  if (w >= nwindows) return;

  int cnt = counts[w];
  int out0 = offsets[w];
  int src0 = w * VIT_LT_MAX_WINDOWS_PER_INPUT;
  for (int j = threadIdx.x; j < cnt; j += blockDim.x)
    dst[out0 + j] = src[src0 + j];
}

__global__ static void
cuda_viterbi_windows_extend_merge_kernel(const P7_CUDA_VIT_LT_WINDOW *src, int nsrc,
                                         const P7_HMM_WINDOW *input_wl,
                                         const float *prefix_lengths,
                                         const float *suffix_lengths,
                                         int model_max_length,
                                         int max_window_len,
                                         int overlap_len,
                                         int dst_cap,
                                         P7_HMM_WINDOW *tmp,
                                         P7_HMM_WINDOW *dst,
                                         int *dst_count)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  int tmp_count = 0;
  for (int i = 0; i < nsrc; i++) {
    P7_CUDA_VIT_LT_WINDOW gw = src[i];
    int win_id = gw.window_id;
    P7_HMM_WINDOW parent = input_wl[win_id];
    int k = (int)gw.model_k;
    int64_t window_start = vlt_i64_max((int64_t)1,
                                       (int64_t)((double)gw.position -
                                                 ((double)model_max_length * (0.1 + (double)prefix_lengths[k]))));
    int64_t window_end = vlt_i64_min((int64_t)parent.length,
                                     (int64_t)((double)gw.position + 1.0 +
                                               ((double)model_max_length * (0.1 + (double)suffix_lengths[k]))));
    P7_HMM_WINDOW curr;
    curr.score = 0.0f;
    curr.null_sc = 0.0f;
    curr.id = win_id;
    curr.n = window_start;
    curr.fm_n = 0;
    curr.length = (int32_t)(window_end - window_start + 1);
    curr.k = (int16_t)k;
    curr.target_len = parent.length;
    curr.complementarity = p7_NOCOMPLEMENT;
    curr.used_to_extend = 0;

    if (tmp_count > 0) {
      P7_HMM_WINDOW *prev = &tmp[tmp_count - 1];
      int64_t ov_start = vlt_i64_max(prev->n, curr.n);
      int64_t ov_end = vlt_i64_min(prev->n + prev->length - 1, curr.n + curr.length - 1);
      int64_t ov_len = ov_end - ov_start + 1;
      if (prev->complementarity == curr.complementarity &&
          prev->id == curr.id &&
          (float)ov_len / (float)((prev->length < curr.length) ? prev->length : curr.length) > 0.5f) {
        int64_t merged_start = vlt_i64_min(prev->n, curr.n);
        int64_t merged_end = vlt_i64_max(prev->n + prev->length - 1, curr.n + curr.length - 1);
        prev->fm_n -= (prev->n - merged_start);
        prev->n = merged_start;
        prev->length = (int32_t)(merged_end - merged_start + 1);
        continue;
      }
    }
    tmp[tmp_count++] = curr;
  }

  int out = 0;
  for (int i = 0; i < tmp_count; i++) {
    P7_HMM_WINDOW merged = tmp[i];
    int parent_id = merged.id;
    int64_t local_n = merged.n;
    int64_t remaining_len = merged.length;

    P7_HMM_WINDOW w = merged;
    w.n = local_n + input_wl[parent_id].n - 1;
    w.target_len = input_wl[parent_id].target_len;
    w.id = 0;
    if (w.length > max_window_len) w.length = max_window_len;
    if (out >= dst_cap) { *dst_count = -1; return; }
    dst[out++] = w;

    while (remaining_len > max_window_len) {
      int shift = max_window_len - overlap_len;
      local_n += shift;
      remaining_len -= shift;
      P7_HMM_WINDOW split = merged;
      split.n = local_n + input_wl[parent_id].n - 1;
      split.length = (int32_t)((remaining_len < max_window_len) ? remaining_len : max_window_len);
      split.target_len = input_wl[parent_id].target_len;
      split.id = 0;
      split.k = 0;
      split.score = 0.0f;
      split.null_sc = 0.0f;
      split.fm_n = 0;
      split.complementarity = p7_NOCOMPLEMENT;
      split.used_to_extend = 0;
      if (out >= dst_cap) { *dst_count = -1; return; }
      dst[out++] = split;
    }
  }
  *dst_count = out;
}

static int
p7_cuda_viterbi_longtarget_prepare_common(P7_CUDA_ENGINE *engine, int nwindows,
                                          int max_out, int need_dsq, int need_host_dsq,
                                          int need_meta,
                                          P7_CUDA_VIT_LT_STATS *local_stats,
                                          char *errbuf, int errbuf_size)
{
  int status = eslOK;
  double t0, t1;

  t0 = host_seconds();
  if (need_dsq && engine->vlt_dsq_alloc < need_dsq) {
    if (engine->d_vlt_dsq) cudaFree(engine->d_vlt_dsq);
    engine->d_vlt_dsq = NULL;
    engine->vlt_dsq_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_vlt_dsq, need_dsq), errbuf, errbuf_size, "cudaMalloc(vlt dsq)")) != eslOK) goto ERROR;
    engine->vlt_dsq_alloc = need_dsq;
  }
  if (need_host_dsq && engine->h_vlt_dsq_alloc < need_host_dsq) {
    if (engine->h_vlt_dsq) cudaFreeHost(engine->h_vlt_dsq);
    engine->h_vlt_dsq = NULL;
    engine->h_vlt_dsq_alloc = 0;
    if ((status = cuda_status(cudaMallocHost((void **)&engine->h_vlt_dsq, need_host_dsq), errbuf, errbuf_size, "cudaMallocHost(vlt dsq)")) != eslOK) goto ERROR;
    engine->h_vlt_dsq_alloc = need_host_dsq;
  }
  if (need_meta) {
    if (engine->vlt_meta_alloc < nwindows) {
      if (engine->d_vlt_offsets) cudaFree(engine->d_vlt_offsets);
      if (engine->d_vlt_lengths) cudaFree(engine->d_vlt_lengths);
      engine->d_vlt_offsets = NULL;
      engine->d_vlt_lengths = NULL;
      engine->vlt_meta_alloc = 0;
      if ((status = cuda_status(cudaMalloc((void **)&engine->d_vlt_offsets, sizeof(int) * nwindows), errbuf, errbuf_size, "cudaMalloc(vlt offsets)")) != eslOK) goto ERROR;
      if ((status = cuda_status(cudaMalloc((void **)&engine->d_vlt_lengths, sizeof(int) * nwindows), errbuf, errbuf_size, "cudaMalloc(vlt lengths)")) != eslOK) goto ERROR;
      engine->vlt_meta_alloc = nwindows;
    }
  }
  if (engine->vlt_thresh_alloc < nwindows) {
    if (engine->d_vlt_thresholds) cudaFree(engine->d_vlt_thresholds);
    engine->d_vlt_thresholds = NULL;
    engine->vlt_thresh_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_vlt_thresholds, sizeof(int16_t) * nwindows), errbuf, errbuf_size, "cudaMalloc(vlt thresholds)")) != eslOK) goto ERROR;
    engine->vlt_thresh_alloc = nwindows;
	  }
	  if (engine->vlt_win_alloc < max_out) {
	    if (engine->d_vlt_windows) cudaFree(engine->d_vlt_windows);
	    engine->d_vlt_windows = NULL;
	    engine->vlt_win_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_vlt_windows, sizeof(P7_CUDA_VIT_LT_WINDOW) * max_out), errbuf, errbuf_size, "cudaMalloc(vlt windows)")) != eslOK) goto ERROR;
    free(engine->h_vlt_windows);
    engine->h_vlt_windows = (P7_CUDA_VIT_LT_WINDOW *)malloc(sizeof(P7_CUDA_VIT_LT_WINDOW) * max_out);
    if (!engine->h_vlt_windows) { status = eslEMEM; goto ERROR; }
    engine->vlt_win_alloc = max_out;
  }
  if (engine->vlt_count_alloc < nwindows) {
    if (engine->d_vlt_win_count) cudaFree(engine->d_vlt_win_count);
    if (engine->d_vlt_win_offsets) cudaFree(engine->d_vlt_win_offsets);
    engine->d_vlt_win_count = NULL;
    engine->d_vlt_win_offsets = NULL;
    engine->vlt_count_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_vlt_win_count, sizeof(int) * nwindows), errbuf, errbuf_size, "cudaMalloc(vlt win_count)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_vlt_win_offsets, sizeof(int) * (nwindows + 1)), errbuf, errbuf_size, "cudaMalloc(vlt win offsets)")) != eslOK) goto ERROR;
    engine->vlt_count_alloc = nwindows;
  }
  t1 = host_seconds();
  local_stats->alloc_seconds += t1 - t0;

ERROR:
  return status;
}

static int
p7_cuda_viterbi_longtarget_launch(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                  const uint8_t *d_dsq, const int *d_offsets,
                                  const int *d_lengths,
                                  const int *d_seqidx,
                                  int nwindows, int max_length,
                                  float nj, float scale_w,
                                  P7_CUDA_VIT_LT_WINDOW **ret_windows, int *ret_nwindows,
                                  P7_CUDA_VIT_LT_STATS *local_stats,
                                  char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int Q = cuom->Qw;
  int M = cuom->M;
  int h_win_count = 0;
  P7_CUDA_VIT_LT_WINDOW *h_windows = NULL;

  {
    size_t shmem_per_group = (size_t) Q * VIT_LT_LANES * 3 * 2 * sizeof(int16_t);
    int groups_per_block = VIT_LT_GROUPS_PER_BLOCK;
    while (groups_per_block > 1 && shmem_per_group * groups_per_block > 24 * 1024)
      groups_per_block >>= 1;
    int block_threads = groups_per_block * VIT_LT_LANES;
    int physical_warps_per_block = (block_threads + 31) / 32;
    int nblocks = (nwindows + groups_per_block - 1) / groups_per_block;
    size_t shmem = shmem_per_group * groups_per_block;
    local_stats->warps_per_block = physical_warps_per_block;
    local_stats->grid_blocks        = nblocks;
    local_stats->block_threads      = block_threads;
    local_stats->dynamic_smem_bytes = (int) shmem;
    {
      cudaDeviceProp prop;
      int dev = 0;
      int active_blocks = 0;
      if (cudaGetDevice(&dev) == cudaSuccess &&
          cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                                        cuda_viterbi_longtarget_kernel,
                                                        block_threads, shmem) == cudaSuccess) {
        local_stats->active_blocks_per_sm  = active_blocks;
        local_stats->active_warps_per_sm   = active_blocks * physical_warps_per_block;
        local_stats->max_warps_per_sm      = prop.maxThreadsPerMultiProcessor / 32;
        local_stats->sm_count              = prop.multiProcessorCount;
        local_stats->theoretical_occupancy = local_stats->max_warps_per_sm > 0
                                            ? (double)local_stats->active_warps_per_sm / (double)local_stats->max_warps_per_sm
                                            : 0.0;
        local_stats->grid_sm_coverage      = local_stats->sm_count > 0
                                            ? (double)nblocks / (double)local_stats->sm_count
                                            : 0.0;
      }
    }
    cudaEventRecord(engine->evt_k0);
    cuda_viterbi_longtarget_kernel<<<nblocks, block_threads, shmem>>>(
      d_dsq, d_offsets, d_lengths, d_seqidx,
      nwindows,
      cuom->d_rwv, cuom->d_twv, M, Q, cuom->Kp,
      nj, scale_w, max_length,
      cuom->xw_e_loop, cuom->xw_e_move,
      cuom->base_w, cuom->ddbound_w,
      engine->d_vlt_thresholds,
      engine->d_vlt_windows, engine->d_vlt_win_count, engine->vlt_win_alloc);
    cudaEventRecord(engine->evt_k1);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_viterbi_longtarget_kernel launch")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaEventSynchronize(engine->evt_k1), errbuf, errbuf_size, "cudaEventSynchronize(vlt kernel)")) != eslOK) goto ERROR;
    local_stats->kernel_seconds += elapsed_seconds(engine->evt_k0, engine->evt_k1);
  }

  /* Host-side prefix sum: D2H win_count, compute prefix, H2D offsets */
  if (engine->h_vlt_prefix_alloc < nwindows) {
    free(engine->h_vlt_win_counts);
    engine->h_vlt_win_counts = NULL;
    engine->h_vlt_prefix_alloc = 0;
    engine->h_vlt_win_counts = (int *)malloc(sizeof(int) * (nwindows + 1));
    if (!engine->h_vlt_win_counts) { status = eslEMEM; goto ERROR; }
    engine->h_vlt_prefix_alloc = nwindows;
  }
  cudaEventRecord(engine->evt_d2h0);
  if ((status = cuda_status(cudaMemcpy(engine->h_vlt_win_counts, engine->d_vlt_win_count,
                                       sizeof(int) * nwindows, cudaMemcpyDeviceToHost),
                            errbuf, errbuf_size, "cudaMemcpy(vlt win_count D2H)")) != eslOK) goto ERROR;
  {
    int total = 0;
    int *h_offsets = engine->h_vlt_win_counts;
    for (int w = 0; w < nwindows; w++) {
      int cnt = h_offsets[w];
      if (cnt > VIT_LT_MAX_WINDOWS_PER_INPUT) {
        if (errbuf && errbuf_size > 0)
          snprintf(errbuf, errbuf_size, "CUDA Viterbi longtarget emitted more than %d seeds in at least one input window",
                   VIT_LT_MAX_WINDOWS_PER_INPUT);
        status = eslERANGE;
        goto ERROR;
      }
      h_offsets[w] = total;
      total += cnt;
    }
    h_offsets[nwindows] = total;
    h_win_count = total;
  }
  if (h_win_count > engine->vlt_win_alloc) h_win_count = engine->vlt_win_alloc;
  if (h_win_count > 0) {
    if ((status = cuda_status(cudaMemcpy(engine->d_vlt_win_offsets, engine->h_vlt_win_counts,
                                         sizeof(int) * (nwindows + 1), cudaMemcpyHostToDevice),
                              errbuf, errbuf_size, "cudaMemcpy(vlt win offsets H2D)")) != eslOK) goto ERROR;
  }

  if (h_win_count > 0) {
    if (engine->vlt_compact_alloc < h_win_count) {
      if (engine->d_vlt_windows_compact) cudaFree(engine->d_vlt_windows_compact);
      engine->d_vlt_windows_compact = NULL;
      engine->vlt_compact_alloc = 0;
      if ((status = cuda_status(cudaMalloc((void **)&engine->d_vlt_windows_compact, sizeof(P7_CUDA_VIT_LT_WINDOW) * h_win_count), errbuf, errbuf_size, "cudaMalloc(vlt compact windows)")) != eslOK) goto ERROR;
      engine->vlt_compact_alloc = h_win_count;
    }
    cuda_viterbi_longtarget_compact_kernel<<<nwindows, 128>>>(engine->d_vlt_windows,
                                                              engine->d_vlt_win_count,
                                                              engine->d_vlt_win_offsets,
                                                              nwindows,
                                                              engine->d_vlt_windows_compact);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_viterbi_longtarget_compact_kernel launch")) != eslOK) goto ERROR;

    h_windows = engine->h_vlt_windows;
    if (!h_windows) { status = eslEMEM; goto ERROR; }
    if ((status = cuda_status(cudaMemcpy(h_windows, engine->d_vlt_windows_compact, sizeof(P7_CUDA_VIT_LT_WINDOW) * h_win_count, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(vlt compact windows)")) != eslOK) goto ERROR;
  }
  cudaEventRecord(engine->evt_d2h1);
  cudaEventSynchronize(engine->evt_d2h1);
  local_stats->d2h_seconds += elapsed_seconds(engine->evt_d2h0, engine->evt_d2h1);
  local_stats->nwindows_out = h_win_count;
  local_stats->device_active_seconds = local_stats->threshold_kernel_seconds + local_stats->kernel_seconds;

  *ret_windows  = h_windows;
  *ret_nwindows = h_win_count;
  return eslOK;

ERROR:
  *ret_windows  = NULL;
  *ret_nwindows = 0;
  return status;
}

static int
p7_cuda_viterbi_longtarget_launch_windows(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                          const P7_SCOREDATA *scoredata,
                                          const P7_HMM_WINDOW *input_windows,
                                          const uint8_t *d_dsq, const int *d_offsets,
                                          const int *d_lengths, const int *d_seqidx,
                                          int nwindows, int max_length,
                                          float nj, float scale_w,
                                          P7_HMM_WINDOW **ret_windows, int *ret_nwindows,
                                          P7_CUDA_VIT_LT_STATS *local_stats,
                                          char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int raw_count = 0;
  int merged_count = 0;
  int max_window_len = 80000;
  int overlap_len = (max_length < 40000) ? max_length : 40000;
  P7_CUDA_VIT_LT_WINDOW *raw_windows = NULL;
  int out_cap;

  status = p7_cuda_viterbi_longtarget_launch(engine, cuom, d_dsq, d_offsets, d_lengths,
                                             d_seqidx, nwindows, max_length, nj, scale_w,
                                             &raw_windows, &raw_count, local_stats,
                                             errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;
  if (raw_count == 0) {
    *ret_windows = NULL;
    *ret_nwindows = 0;
    return eslOK;
  }

  out_cap = raw_count * 4;
  if (out_cap < raw_count) out_cap = raw_count;
  if (engine->vlt_hmm_alloc < out_cap) {
    free(engine->h_vlt_hmm_windows);
    engine->h_vlt_hmm_windows = NULL;
    engine->vlt_hmm_alloc = 0;
    engine->h_vlt_hmm_windows = (P7_HMM_WINDOW *)malloc(sizeof(P7_HMM_WINDOW) * out_cap);
    if (!engine->h_vlt_hmm_windows) { status = eslEMEM; goto ERROR; }
    engine->vlt_hmm_alloc = out_cap;
  }

  /* Host-side extend/merge/split — raw_windows and input_windows are already on host */
  {
    const P7_CUDA_VIT_LT_WINDOW *src = raw_windows;
    const float *prefix_lengths = scoredata->prefix_lengths;
    const float *suffix_lengths = scoredata->suffix_lengths;
    P7_HMM_WINDOW *tmp = (P7_HMM_WINDOW *)malloc(sizeof(P7_HMM_WINDOW) * raw_count);
    P7_HMM_WINDOW *dst = engine->h_vlt_hmm_windows;
    int tmp_count = 0;
    int out = 0;

    if (!tmp) { status = eslEMEM; goto ERROR; }

    /* Phase 1: Extend and merge with >50% overlap threshold */
    for (int i = 0; i < raw_count; i++) {
      P7_CUDA_VIT_LT_WINDOW gw = src[i];
      int win_id = gw.window_id;
      P7_HMM_WINDOW parent = input_windows[win_id];
      int k = (int)gw.model_k;
      int64_t window_start = (int64_t)((double)gw.position -
                             ((double)max_length * (0.1 + (double)prefix_lengths[k])));
      int64_t window_end   = (int64_t)((double)gw.position + 1.0 +
                             ((double)max_length * (0.1 + (double)suffix_lengths[k])));
      if (window_start < 1) window_start = 1;
      if (window_end > (int64_t)parent.length) window_end = (int64_t)parent.length;

      P7_HMM_WINDOW curr;
      curr.score           = 0.0f;
      curr.null_sc         = 0.0f;
      curr.id              = win_id;
      curr.n               = window_start;
      curr.fm_n            = 0;
      curr.length          = (int32_t)(window_end - window_start + 1);
      curr.k               = (int16_t)k;
      curr.target_len      = parent.length;
      curr.complementarity = p7_NOCOMPLEMENT;
      curr.used_to_extend  = 0;

      if (tmp_count > 0) {
        P7_HMM_WINDOW *prev = &tmp[tmp_count - 1];
        int64_t ov_start = (prev->n > curr.n) ? prev->n : curr.n;
        int64_t ov_end   = ((prev->n + prev->length - 1) < (curr.n + curr.length - 1))
                            ? (prev->n + prev->length - 1) : (curr.n + curr.length - 1);
        int64_t ov_len   = ov_end - ov_start + 1;
        int32_t min_len  = (prev->length < curr.length) ? prev->length : curr.length;
        if (prev->complementarity == curr.complementarity &&
            prev->id == curr.id &&
            (float)ov_len / (float)min_len > 0.5f) {
          int64_t merged_start = (prev->n < curr.n) ? prev->n : curr.n;
          int64_t merged_end   = ((prev->n + prev->length - 1) > (curr.n + curr.length - 1))
                                  ? (prev->n + prev->length - 1) : (curr.n + curr.length - 1);
          prev->fm_n -= (int32_t)(prev->n - merged_start);
          prev->n     = merged_start;
          prev->length = (int32_t)(merged_end - merged_start + 1);
          continue;
        }
      }
      tmp[tmp_count++] = curr;
    }

    /* Phase 2: Translate to global coords and split oversized windows */
    for (int i = 0; i < tmp_count; i++) {
      P7_HMM_WINDOW merged = tmp[i];
      int parent_id       = merged.id;
      int64_t local_n     = merged.n;
      int64_t remaining_len = merged.length;

      P7_HMM_WINDOW w = merged;
      w.n          = local_n + input_windows[parent_id].n - 1;
      w.target_len = input_windows[parent_id].target_len;
      w.id         = 0;
      if (w.length > max_window_len) w.length = max_window_len;
      if (out >= out_cap) { free(tmp); merged_count = -1; goto VLT_OVERFLOW; }
      dst[out++] = w;

      while (remaining_len > max_window_len) {
        int shift   = max_window_len - overlap_len;
        local_n    += shift;
        remaining_len -= shift;
        P7_HMM_WINDOW split = merged;
        split.n           = local_n + input_windows[parent_id].n - 1;
        split.length      = (int32_t)((remaining_len < max_window_len) ? remaining_len : max_window_len);
        split.target_len  = input_windows[parent_id].target_len;
        split.id          = 0;
        split.k           = 0;
        split.score       = 0.0f;
        split.null_sc     = 0.0f;
        split.fm_n        = 0;
        split.complementarity = p7_NOCOMPLEMENT;
        split.used_to_extend  = 0;
        if (out >= out_cap) { free(tmp); merged_count = -1; goto VLT_OVERFLOW; }
        dst[out++] = split;
      }
    }
    free(tmp);
    merged_count = out;
  }

VLT_OVERFLOW:
  if (merged_count < 0) {
    if (errbuf && errbuf_size > 0)
      snprintf(errbuf, errbuf_size, "CUDA Viterbi longtarget merged window output exceeded capacity");
    status = eslERANGE;
    goto ERROR;
  }
  if (merged_count == 0) {
    *ret_windows = NULL;
    *ret_nwindows = 0;
    return eslOK;
  }

  *ret_windows = engine->h_vlt_hmm_windows;
  *ret_nwindows = merged_count;
  return eslOK;

ERROR:
  *ret_windows = NULL;
  *ret_nwindows = 0;
  return status;
}

extern "C" int
p7_cuda_ViterbiLongtarget(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                          const ESL_DSQ *dsq, int L,
                          const P7_HMM_WINDOW *windows, int nwindows,
                          const float *bias_scores, int do_biasfilter,
                          int B2, float F2, float vmu, float vlambda,
                          float scale_w, float xw_e_move, float nj,
                          float base_w, int max_length,
                          P7_CUDA_VIT_LT_WINDOW **ret_windows, int *ret_nwindows,
                          P7_CUDA_VIT_LT_STATS *stats,
                          char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int max_out;
  int *h_offsets = NULL;
  int *h_lengths = NULL;
  int total_packed = 0;
  P7_CUDA_VIT_LT_WINDOW *h_windows = NULL;
  P7_CUDA_VIT_LT_STATS local_stats;
  double t0, t1, t_stream0, t_stream1;

  memset(&local_stats, 0, sizeof(local_stats));
  local_stats.nwindows_in = nwindows;

  if (nwindows == 0) {
    *ret_windows  = NULL;
    *ret_nwindows = 0;
    if (stats) *stats = local_stats;
    return eslOK;
  }

  max_out = nwindows * VIT_LT_MAX_WINDOWS_PER_INPUT;

  if (engine->h_meta_alloc < nwindows) {
    int     *new_offsets = (int *)     realloc(engine->h_meta_offsets,    sizeof(int)     * nwindows);
    int     *new_lengths = (int *)     realloc(engine->h_meta_lengths,    sizeof(int)     * nwindows);
    uint8_t *new_tjb     = (uint8_t *) realloc(engine->h_meta_tjb_by_seq, sizeof(uint8_t) * nwindows);
    if (!new_offsets || !new_lengths || !new_tjb) {
      if (new_offsets) engine->h_meta_offsets = new_offsets;
      if (new_lengths) engine->h_meta_lengths = new_lengths;
      if (new_tjb)     engine->h_meta_tjb_by_seq = new_tjb;
      status = eslEMEM; goto ERROR;
    }
    engine->h_meta_offsets    = new_offsets;
    engine->h_meta_lengths    = new_lengths;
    engine->h_meta_tjb_by_seq = new_tjb;
    engine->h_meta_alloc      = nwindows;
  }
  h_offsets = engine->h_meta_offsets;
  h_lengths = engine->h_meta_lengths;

  for (int w = 0; w < nwindows; w++) {
    h_offsets[w] = total_packed;
    h_lengths[w] = windows[w].length;
    total_packed += windows[w].length + 1;
  }
  total_packed += 1;

  status = p7_cuda_viterbi_longtarget_prepare_common(engine, nwindows, max_out,
                                                     total_packed, total_packed, 1,
                                                     &local_stats, errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;
  if (do_biasfilter && bias_scores && engine->vlt_bias_alloc < nwindows) {
    if (engine->d_vlt_bias) cudaFree(engine->d_vlt_bias);
    engine->d_vlt_bias = NULL;
    engine->vlt_bias_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_vlt_bias, sizeof(float) * nwindows), errbuf, errbuf_size, "cudaMalloc(vlt bias)")) != eslOK) goto ERROR;
    engine->vlt_bias_alloc = nwindows;
  }

  /* Pack window subsequences into pinned host buffer */
  t0 = host_seconds();
  for (int w = 0; w < nwindows; w++) {
    uint8_t *dest = engine->h_vlt_dsq + h_offsets[w];
    int64_t seq_start = windows[w].n;
    int wlen = windows[w].length;
    dest[0] = eslDSQ_SENTINEL;
    memcpy(dest + 1, dsq + seq_start, wlen);
  }
  t1 = host_seconds();
  local_stats.pack_seconds += t1 - t0;
  local_stats.packed_bytes += total_packed;

  /* Upload and compute thresholds concurrently using streams */
  {
    cudaStream_t stream_copy   = engine->vlt_stream_copy;
    cudaStream_t stream_thresh = engine->vlt_stream_thresh;

    /* Stream 1: upload DSQ data (large transfer) */
    cudaEventRecord(engine->evt_h2d0, stream_copy);
    if ((status = cuda_status(cudaMemcpyAsync(engine->d_vlt_dsq, engine->h_vlt_dsq, total_packed, cudaMemcpyHostToDevice, stream_copy), errbuf, errbuf_size, "cudaMemcpyAsync(vlt dsq)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMemcpyAsync(engine->d_vlt_offsets, h_offsets, sizeof(int) * nwindows, cudaMemcpyHostToDevice, stream_copy), errbuf, errbuf_size, "cudaMemcpyAsync(vlt offsets)")) != eslOK) goto ERROR;
    cudaEventRecord(engine->evt_h2d1, stream_copy);

    /* Stream 2: upload metadata + bias, then compute thresholds. */
    if ((status = cuda_status(cudaMemcpyAsync(engine->d_vlt_lengths, h_lengths, sizeof(int) * nwindows, cudaMemcpyHostToDevice, stream_thresh), errbuf, errbuf_size, "cudaMemcpyAsync(vlt lengths)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMemsetAsync(engine->d_vlt_win_count, 0, sizeof(int) * nwindows, stream_thresh), errbuf, errbuf_size, "cudaMemsetAsync(vlt win_count)")) != eslOK) goto ERROR;

    float *d_bias = NULL;
    if (do_biasfilter && bias_scores) {
      d_bias = engine->d_vlt_bias;
      if ((status = cuda_status(cudaMemcpyAsync(d_bias, bias_scores, sizeof(float) * nwindows, cudaMemcpyHostToDevice, stream_thresh), errbuf, errbuf_size, "cudaMemcpyAsync(vlt bias)")) != eslOK) goto ERROR;
    }
    cudaEventRecord(engine->evt_d2h0, stream_thresh);
    cuda_compute_viterbi_thresholds_kernel<<<(nwindows + 127) / 128, 128, 0, stream_thresh>>>(
        NULL, d_bias, NULL, engine->d_vlt_lengths, nwindows, do_biasfilter,
        B2, F2, vmu, vlambda, scale_w, xw_e_move, nj, base_w,
        max_length, engine->d_vlt_thresholds);
    cudaEventRecord(engine->evt_d2h1, stream_thresh);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "vlt threshold kernel launch")) != eslOK) goto ERROR;

    /* Wait for both streams to complete before launching main kernel */
    t_stream0 = host_seconds();
    if ((status = cuda_status(cudaStreamSynchronize(stream_copy), errbuf, errbuf_size, "cudaStreamSynchronize(vlt copy)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaStreamSynchronize(stream_thresh), errbuf, errbuf_size, "cudaStreamSynchronize(vlt thresh)")) != eslOK) goto ERROR;
    t_stream1 = host_seconds();
    local_stats.stream_seconds += t_stream1 - t_stream0;
    local_stats.threshold_kernel_seconds += elapsed_seconds(engine->evt_h2d0, engine->evt_h2d1);
    local_stats.threshold_kernel_seconds += elapsed_seconds(engine->evt_d2h0, engine->evt_d2h1);
  }

  status = p7_cuda_viterbi_longtarget_launch(engine, cuom,
                                             engine->d_vlt_dsq, engine->d_vlt_offsets, engine->d_vlt_lengths, NULL,
                                             nwindows, max_length, nj, scale_w,
                                             &h_windows, ret_nwindows, &local_stats,
                                             errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;

  engine->stats.vit_h2d_seconds    += local_stats.h2d_seconds;
  engine->stats.vit_kernel_seconds += local_stats.threshold_kernel_seconds + local_stats.kernel_seconds;
  engine->stats.vit_d2h_seconds    += local_stats.d2h_seconds;
  engine->stats.vit_nseqs          += nwindows;
  engine->stats.vit_nres           += total_packed > nwindows ? (uint64_t)(total_packed - nwindows - 1) : 0;
  engine->stats.vit_nbatches       += 1;

  *ret_windows  = h_windows;
  if (stats) *stats = local_stats;
  return eslOK;

ERROR:
  *ret_windows  = NULL;
  *ret_nwindows = 0;
  if (stats) *stats = local_stats;
  return status;
}

extern "C" int
p7_cuda_ViterbiLongtargetWindows(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                 const P7_SCOREDATA *scoredata,
                                 const ESL_DSQ *dsq, int L,
                                 const P7_HMM_WINDOW *windows, int nwindows,
                                 const float *bias_scores, int do_biasfilter,
                                 int B2, float F2, float vmu, float vlambda,
                                 float scale_w, float xw_e_move, float nj,
                                 float base_w, int max_length,
                                 P7_HMM_WINDOW **ret_windows, int *ret_nwindows,
                                 P7_CUDA_VIT_LT_STATS *stats,
                                 char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int max_out;
  int *h_offsets = NULL;
  int *h_lengths = NULL;
  int total_packed = 0;
  P7_CUDA_VIT_LT_STATS local_stats;
  double t0, t1, t_stream0, t_stream1;

  (void)L;
  memset(&local_stats, 0, sizeof(local_stats));
  local_stats.nwindows_in = nwindows;

  if (nwindows == 0) {
    *ret_windows  = NULL;
    *ret_nwindows = 0;
    if (stats) *stats = local_stats;
    return eslOK;
  }

  max_out = nwindows * VIT_LT_MAX_WINDOWS_PER_INPUT;

  if (engine->h_meta_alloc < nwindows) {
    int     *new_offsets = (int *)     realloc(engine->h_meta_offsets,    sizeof(int)     * nwindows);
    int     *new_lengths = (int *)     realloc(engine->h_meta_lengths,    sizeof(int)     * nwindows);
    uint8_t *new_tjb     = (uint8_t *) realloc(engine->h_meta_tjb_by_seq, sizeof(uint8_t) * nwindows);
    if (!new_offsets || !new_lengths || !new_tjb) {
      if (new_offsets) engine->h_meta_offsets = new_offsets;
      if (new_lengths) engine->h_meta_lengths = new_lengths;
      if (new_tjb)     engine->h_meta_tjb_by_seq = new_tjb;
      status = eslEMEM; goto ERROR;
    }
    engine->h_meta_offsets    = new_offsets;
    engine->h_meta_lengths    = new_lengths;
    engine->h_meta_tjb_by_seq = new_tjb;
    engine->h_meta_alloc      = nwindows;
  }
  h_offsets = engine->h_meta_offsets;
  h_lengths = engine->h_meta_lengths;

  for (int w = 0; w < nwindows; w++) {
    h_offsets[w] = total_packed;
    h_lengths[w] = windows[w].length;
    total_packed += windows[w].length + 1;
  }
  total_packed += 1;

  status = p7_cuda_viterbi_longtarget_prepare_common(engine, nwindows, max_out,
                                                     total_packed, total_packed, 1,
                                                     &local_stats, errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;
  if (do_biasfilter && bias_scores && engine->vlt_bias_alloc < nwindows) {
    if (engine->d_vlt_bias) cudaFree(engine->d_vlt_bias);
    engine->d_vlt_bias = NULL;
    engine->vlt_bias_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_vlt_bias, sizeof(float) * nwindows), errbuf, errbuf_size, "cudaMalloc(vlt bias)")) != eslOK) goto ERROR;
    engine->vlt_bias_alloc = nwindows;
  }

  t0 = host_seconds();
  for (int w = 0; w < nwindows; w++) {
    uint8_t *dest = engine->h_vlt_dsq + h_offsets[w];
    int64_t seq_start = windows[w].n;
    int wlen = windows[w].length;
    dest[0] = eslDSQ_SENTINEL;
    memcpy(dest + 1, dsq + seq_start, wlen);
  }
  t1 = host_seconds();
  local_stats.pack_seconds += t1 - t0;
  local_stats.packed_bytes += total_packed;

  {
    cudaStream_t stream_copy   = engine->vlt_stream_copy;
    cudaStream_t stream_thresh = engine->vlt_stream_thresh;

    cudaEventRecord(engine->evt_h2d0, stream_copy);
    if ((status = cuda_status(cudaMemcpyAsync(engine->d_vlt_dsq, engine->h_vlt_dsq, total_packed, cudaMemcpyHostToDevice, stream_copy), errbuf, errbuf_size, "cudaMemcpyAsync(vlt dsq)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMemcpyAsync(engine->d_vlt_offsets, h_offsets, sizeof(int) * nwindows, cudaMemcpyHostToDevice, stream_copy), errbuf, errbuf_size, "cudaMemcpyAsync(vlt offsets)")) != eslOK) goto ERROR;
    cudaEventRecord(engine->evt_h2d1, stream_copy);

    if ((status = cuda_status(cudaMemcpyAsync(engine->d_vlt_lengths, h_lengths, sizeof(int) * nwindows, cudaMemcpyHostToDevice, stream_thresh), errbuf, errbuf_size, "cudaMemcpyAsync(vlt lengths)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMemsetAsync(engine->d_vlt_win_count, 0, sizeof(int) * nwindows, stream_thresh), errbuf, errbuf_size, "cudaMemsetAsync(vlt win_count)")) != eslOK) goto ERROR;

    float *d_bias = NULL;
    if (do_biasfilter && bias_scores) {
      d_bias = engine->d_vlt_bias;
      if ((status = cuda_status(cudaMemcpyAsync(d_bias, bias_scores, sizeof(float) * nwindows, cudaMemcpyHostToDevice, stream_thresh), errbuf, errbuf_size, "cudaMemcpyAsync(vlt bias)")) != eslOK) goto ERROR;
    }
    cudaEventRecord(engine->evt_d2h0, stream_thresh);
    cuda_compute_viterbi_thresholds_kernel<<<(nwindows + 127) / 128, 128, 0, stream_thresh>>>(
        NULL, d_bias, NULL, engine->d_vlt_lengths, nwindows, do_biasfilter,
        B2, F2, vmu, vlambda, scale_w, xw_e_move, nj, base_w,
        max_length, engine->d_vlt_thresholds);
    cudaEventRecord(engine->evt_d2h1, stream_thresh);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "vlt threshold kernel launch")) != eslOK) goto ERROR;

    t_stream0 = host_seconds();
    if ((status = cuda_status(cudaStreamSynchronize(stream_copy), errbuf, errbuf_size, "cudaStreamSynchronize(vlt copy)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaStreamSynchronize(stream_thresh), errbuf, errbuf_size, "cudaStreamSynchronize(vlt thresh)")) != eslOK) goto ERROR;
    t_stream1 = host_seconds();
    local_stats.stream_seconds += t_stream1 - t_stream0;
    local_stats.threshold_kernel_seconds += elapsed_seconds(engine->evt_h2d0, engine->evt_h2d1);
    local_stats.threshold_kernel_seconds += elapsed_seconds(engine->evt_d2h0, engine->evt_d2h1);
  }

  status = p7_cuda_viterbi_longtarget_launch_windows(engine, cuom, scoredata, windows,
                                                     engine->d_vlt_dsq, engine->d_vlt_offsets,
                                                     engine->d_vlt_lengths, NULL,
                                                     nwindows, max_length, nj, scale_w,
                                                     ret_windows, ret_nwindows, &local_stats,
                                                     errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;

  engine->stats.vit_h2d_seconds    += local_stats.h2d_seconds;
  engine->stats.vit_kernel_seconds += local_stats.threshold_kernel_seconds + local_stats.kernel_seconds;
  engine->stats.vit_d2h_seconds    += local_stats.d2h_seconds;
  engine->stats.vit_nseqs          += nwindows;
  engine->stats.vit_nres           += total_packed > nwindows ? (uint64_t)(total_packed - nwindows - 1) : 0;
  engine->stats.vit_nbatches       += 1;

  if (stats) *stats = local_stats;
  return eslOK;

ERROR:
  *ret_windows  = NULL;
  *ret_nwindows = 0;
  if (stats) *stats = local_stats;
  return status;
}

extern "C" int
p7_cuda_ViterbiLongtargetFromF1(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                int nwindows,
                                int B2, float F2, float vmu, float vlambda,
                                float scale_w, float xw_e_move, float nj,
                                float base_w, int max_length,
                                P7_CUDA_VIT_LT_WINDOW **ret_windows, int *ret_nwindows,
                                P7_CUDA_VIT_LT_STATS *stats,
                                char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int max_out;
  P7_CUDA_VIT_LT_WINDOW *h_windows = NULL;
  P7_CUDA_VIT_LT_STATS local_stats;
  double t_stream0, t_stream1;

  memset(&local_stats, 0, sizeof(local_stats));
  local_stats.nwindows_in = nwindows;

  if (nwindows == 0) {
    *ret_windows  = NULL;
    *ret_nwindows = 0;
    if (stats) *stats = local_stats;
    return eslOK;
  }
  if (!engine || !cuom || !engine->d_offsets || !engine->d_lengths ||
      !engine->d_f1_survivor_idx || !engine->d_f1_survivor_filtersc ||
      engine->batch_nseq < nwindows) {
    *ret_windows  = NULL;
    *ret_nwindows = 0;
    if (stats) *stats = local_stats;
    return eslEINVAL;
  }
  const uint8_t *d_batch_dsq = engine->d_batch_dsq ? engine->d_batch_dsq : engine->d_dsq;
  if (!d_batch_dsq) {
    *ret_windows  = NULL;
    *ret_nwindows = 0;
    if (stats) *stats = local_stats;
    return eslEINVAL;
  }

  max_out = nwindows * VIT_LT_MAX_WINDOWS_PER_INPUT;

  status = p7_cuda_viterbi_longtarget_prepare_common(engine, nwindows, max_out,
                                                     0, 0, 0,
                                                     &local_stats, errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;

  {
    cudaStream_t stream_meta   = engine->vlt_stream_copy;
    cudaStream_t stream_thresh = engine->vlt_stream_thresh;

    cudaEventRecord(engine->evt_h2d0, stream_meta);
    if ((status = cuda_status(cudaMemsetAsync(engine->d_vlt_win_count, 0, sizeof(int) * nwindows, stream_meta), errbuf, errbuf_size, "cudaMemsetAsync(vlt win_count from f1)")) != eslOK) goto ERROR;
    cudaEventRecord(engine->evt_h2d1, stream_meta);

    if ((status = cuda_status(cudaStreamWaitEvent(stream_thresh, engine->evt_h2d1, 0), errbuf, errbuf_size, "cudaStreamWaitEvent(vlt f1 metadata)")) != eslOK) goto ERROR;
    cudaEventRecord(engine->evt_d2h0, stream_thresh);
    cuda_compute_viterbi_thresholds_kernel<<<(nwindows + 127) / 128, 128, 0, stream_thresh>>>(
        NULL, engine->d_f1_survivor_filtersc, engine->d_f1_survivor_idx, engine->d_lengths, nwindows, TRUE,
        B2, F2, vmu, vlambda, scale_w, xw_e_move, nj, base_w,
        max_length, engine->d_vlt_thresholds);
    cudaEventRecord(engine->evt_d2h1, stream_thresh);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "vlt threshold kernel from f1 launch")) != eslOK) goto ERROR;

    t_stream0 = host_seconds();
    if ((status = cuda_status(cudaStreamSynchronize(stream_meta), errbuf, errbuf_size, "cudaStreamSynchronize(vlt f1 metadata)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaStreamSynchronize(stream_thresh), errbuf, errbuf_size, "cudaStreamSynchronize(vlt f1 thresh)")) != eslOK) goto ERROR;
    t_stream1 = host_seconds();
    local_stats.stream_seconds += t_stream1 - t_stream0;
    local_stats.h2d_seconds += elapsed_seconds(engine->evt_h2d0, engine->evt_h2d1);
    local_stats.threshold_kernel_seconds += elapsed_seconds(engine->evt_d2h0, engine->evt_d2h1);
  }

  status = p7_cuda_viterbi_longtarget_launch(engine, cuom,
                                             d_batch_dsq, engine->d_offsets, engine->d_lengths, engine->d_f1_survivor_idx,
                                             nwindows, max_length, nj, scale_w,
                                             &h_windows, ret_nwindows, &local_stats,
                                             errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;

  engine->stats.vit_h2d_seconds    += local_stats.h2d_seconds;
  engine->stats.vit_kernel_seconds += local_stats.threshold_kernel_seconds + local_stats.kernel_seconds;
  engine->stats.vit_d2h_seconds    += local_stats.d2h_seconds;
  engine->stats.vit_nseqs          += nwindows;
  engine->stats.vit_nres           += 0;
  engine->stats.vit_nbatches       += 1;

  *ret_windows = h_windows;
  if (stats) *stats = local_stats;
  return eslOK;

ERROR:
  *ret_windows  = NULL;
  *ret_nwindows = 0;
  if (stats) *stats = local_stats;
  return status;
}

extern "C" int
p7_cuda_ViterbiLongtargetFromF1Windows(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                       const P7_SCOREDATA *scoredata,
                                       const P7_HMM_WINDOW *windows, int nwindows,
                                       int B2, float F2, float vmu, float vlambda,
                                       float scale_w, float xw_e_move, float nj,
                                       float base_w, int max_length,
                                       P7_HMM_WINDOW **ret_windows, int *ret_nwindows,
                                       P7_CUDA_VIT_LT_STATS *stats,
                                       char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int max_out;
  P7_CUDA_VIT_LT_STATS local_stats;
  double t_stream0, t_stream1;

  memset(&local_stats, 0, sizeof(local_stats));
  local_stats.nwindows_in = nwindows;

  if (nwindows == 0) {
    *ret_windows  = NULL;
    *ret_nwindows = 0;
    if (stats) *stats = local_stats;
    return eslOK;
  }
  if (!engine || !cuom || !engine->d_offsets || !engine->d_lengths ||
      !engine->d_f1_survivor_idx || !engine->d_f1_survivor_filtersc ||
      engine->batch_nseq < nwindows) {
    *ret_windows  = NULL;
    *ret_nwindows = 0;
    if (stats) *stats = local_stats;
    return eslEINVAL;
  }
  const uint8_t *d_batch_dsq = engine->d_batch_dsq ? engine->d_batch_dsq : engine->d_dsq;
  if (!d_batch_dsq) {
    *ret_windows  = NULL;
    *ret_nwindows = 0;
    if (stats) *stats = local_stats;
    return eslEINVAL;
  }

  max_out = nwindows * VIT_LT_MAX_WINDOWS_PER_INPUT;

  status = p7_cuda_viterbi_longtarget_prepare_common(engine, nwindows, max_out,
                                                     0, 0, 0,
                                                     &local_stats, errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;

  {
    cudaStream_t stream_meta   = engine->vlt_stream_copy;
    cudaStream_t stream_thresh = engine->vlt_stream_thresh;

    cudaEventRecord(engine->evt_h2d0, stream_meta);
    if ((status = cuda_status(cudaMemsetAsync(engine->d_vlt_win_count, 0, sizeof(int) * nwindows, stream_meta), errbuf, errbuf_size, "cudaMemsetAsync(vlt win_count from f1)")) != eslOK) goto ERROR;
    cudaEventRecord(engine->evt_h2d1, stream_meta);

    if ((status = cuda_status(cudaStreamWaitEvent(stream_thresh, engine->evt_h2d1, 0), errbuf, errbuf_size, "cudaStreamWaitEvent(vlt f1 metadata)")) != eslOK) goto ERROR;
    cudaEventRecord(engine->evt_d2h0, stream_thresh);
    cuda_compute_viterbi_thresholds_kernel<<<(nwindows + 127) / 128, 128, 0, stream_thresh>>>(
        NULL, engine->d_f1_survivor_filtersc, engine->d_f1_survivor_idx, engine->d_lengths, nwindows, TRUE,
        B2, F2, vmu, vlambda, scale_w, xw_e_move, nj, base_w,
        max_length, engine->d_vlt_thresholds);
    cudaEventRecord(engine->evt_d2h1, stream_thresh);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "vlt threshold kernel from f1 launch")) != eslOK) goto ERROR;

    t_stream0 = host_seconds();
    if ((status = cuda_status(cudaStreamSynchronize(stream_meta), errbuf, errbuf_size, "cudaStreamSynchronize(vlt f1 metadata)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaStreamSynchronize(stream_thresh), errbuf, errbuf_size, "cudaStreamSynchronize(vlt f1 thresh)")) != eslOK) goto ERROR;
    t_stream1 = host_seconds();
    local_stats.stream_seconds += t_stream1 - t_stream0;
    local_stats.h2d_seconds += elapsed_seconds(engine->evt_h2d0, engine->evt_h2d1);
    local_stats.threshold_kernel_seconds += elapsed_seconds(engine->evt_d2h0, engine->evt_d2h1);
  }

  status = p7_cuda_viterbi_longtarget_launch_windows(engine, cuom, scoredata, windows,
                                                     d_batch_dsq, engine->d_offsets, engine->d_lengths,
                                                     engine->d_f1_survivor_idx,
                                                     nwindows, max_length, nj, scale_w,
                                                     ret_windows, ret_nwindows, &local_stats,
                                                     errbuf, errbuf_size);
  if (status != eslOK) goto ERROR;

  engine->stats.vit_h2d_seconds    += local_stats.h2d_seconds;
  engine->stats.vit_kernel_seconds += local_stats.threshold_kernel_seconds + local_stats.kernel_seconds;
  engine->stats.vit_d2h_seconds    += local_stats.d2h_seconds;
  engine->stats.vit_nseqs          += nwindows;
  engine->stats.vit_nres           += 0;
  engine->stats.vit_nbatches       += 1;

  if (stats) *stats = local_stats;
  return eslOK;

ERROR:
  *ret_windows  = NULL;
  *ret_nwindows = 0;
  if (stats) *stats = local_stats;
  return status;
}

extern "C" int
p7_cuda_ViterbiLongtarget_GetThresholds(P7_CUDA_ENGINE *engine, int16_t *h_thresholds, int nwindows)
{
  if (!engine->d_vlt_thresholds || nwindows <= 0) return eslOK;
  cudaError_t err = cudaMemcpy(h_thresholds, engine->d_vlt_thresholds,
                               sizeof(int16_t) * nwindows, cudaMemcpyDeviceToHost);
  return (err == cudaSuccess) ? eslOK : eslFAIL;
}
