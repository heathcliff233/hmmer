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
    const int *lengths, int nwindows, int do_biasfilter,
    int B2, float F2, float vmu, float vlambda,
    float scale_w, float xw_e_move, float nj, float base_w,
    int max_length, int16_t *sc_thresholds)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nwindows) return;

  int window_len = lengths[i];
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

__global__ static void
cuda_viterbi_longtarget_kernel(
    const uint8_t *dsq, const int *offsets, const int *lengths,
    int nwindows,
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

  int L = lengths[bi];
  const uint8_t *s = dsq + offsets[bi];
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
            int widx = atomicAdd(d_win_count, 1);
            if (widx < max_windows) {
              d_windows[widx].window_id = bi;
              d_windows[widx].position  = i;
              d_windows[widx].model_k   = (int16_t) k;
              d_windows[widx].pad       = 0;
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
  int Q = cuom->Qw;
  int M = cuom->M;
  int max_out;
  int *h_offsets = NULL;
  int *h_lengths = NULL;
  int total_packed = 0;
  int h_win_count = 0;
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

  h_offsets = (int *)malloc(sizeof(int) * nwindows);
  h_lengths = (int *)malloc(sizeof(int) * nwindows);
  if (!h_offsets || !h_lengths) { status = eslEMEM; goto ERROR; }

  for (int w = 0; w < nwindows; w++) {
    h_offsets[w] = total_packed;
    h_lengths[w] = windows[w].length;
    total_packed += windows[w].length + 1;
  }
  total_packed += 1;

  t0 = host_seconds();
  /* Grow engine-persistent buffers */
  if (engine->vlt_dsq_alloc < total_packed) {
    if (engine->d_vlt_dsq) cudaFree(engine->d_vlt_dsq);
    engine->d_vlt_dsq = NULL;
    engine->vlt_dsq_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_vlt_dsq, total_packed), errbuf, errbuf_size, "cudaMalloc(vlt dsq)")) != eslOK) goto ERROR;
    engine->vlt_dsq_alloc = total_packed;
  }
  if (engine->h_vlt_dsq_alloc < total_packed) {
    if (engine->h_vlt_dsq) cudaFreeHost(engine->h_vlt_dsq);
    engine->h_vlt_dsq = NULL;
    engine->h_vlt_dsq_alloc = 0;
    if ((status = cuda_status(cudaMallocHost((void **)&engine->h_vlt_dsq, total_packed), errbuf, errbuf_size, "cudaMallocHost(vlt dsq)")) != eslOK) goto ERROR;
    engine->h_vlt_dsq_alloc = total_packed;
  }
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
  if (engine->vlt_thresh_alloc < nwindows) {
    if (engine->d_vlt_thresholds) cudaFree(engine->d_vlt_thresholds);
    engine->d_vlt_thresholds = NULL;
    engine->vlt_thresh_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_vlt_thresholds, sizeof(int16_t) * nwindows), errbuf, errbuf_size, "cudaMalloc(vlt thresholds)")) != eslOK) goto ERROR;
    engine->vlt_thresh_alloc = nwindows;
  }
  if (do_biasfilter && bias_scores && engine->vlt_bias_alloc < nwindows) {
    if (engine->d_vlt_bias) cudaFree(engine->d_vlt_bias);
    engine->d_vlt_bias = NULL;
    engine->vlt_bias_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_vlt_bias, sizeof(float) * nwindows), errbuf, errbuf_size, "cudaMalloc(vlt bias)")) != eslOK) goto ERROR;
    engine->vlt_bias_alloc = nwindows;
  }
  if (engine->vlt_win_alloc < max_out) {
    if (engine->d_vlt_windows) cudaFree(engine->d_vlt_windows);
    if (engine->d_vlt_win_count) cudaFree(engine->d_vlt_win_count);
    engine->d_vlt_windows = NULL;
    engine->d_vlt_win_count = NULL;
    engine->vlt_win_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_vlt_windows, sizeof(P7_CUDA_VIT_LT_WINDOW) * max_out), errbuf, errbuf_size, "cudaMalloc(vlt windows)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_vlt_win_count, sizeof(int)), errbuf, errbuf_size, "cudaMalloc(vlt win_count)")) != eslOK) goto ERROR;
    engine->vlt_win_alloc = max_out;
  }
  t1 = host_seconds();
  local_stats.alloc_seconds += t1 - t0;

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
    cudaStream_t stream_copy, stream_thresh;
    t_stream0 = host_seconds();
    cudaStreamCreate(&stream_copy);
    cudaStreamCreate(&stream_thresh);
    t_stream1 = host_seconds();
    local_stats.stream_seconds += t_stream1 - t_stream0;

    /* Stream 1: upload DSQ data (large transfer) */
    cudaEventRecord(engine->evt_h2d0, stream_copy);
    cudaMemcpyAsync(engine->d_vlt_dsq, engine->h_vlt_dsq, total_packed, cudaMemcpyHostToDevice, stream_copy);
    cudaMemcpyAsync(engine->d_vlt_offsets, h_offsets, sizeof(int) * nwindows, cudaMemcpyHostToDevice, stream_copy);
    cudaEventRecord(engine->evt_h2d1, stream_copy);

    /* Stream 2 (default): upload lengths + bias + compute thresholds */
    if ((status = cuda_status(cudaMemcpy(engine->d_vlt_lengths, h_lengths, sizeof(int) * nwindows, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(vlt lengths)")) != eslOK) { cudaStreamDestroy(stream_copy); cudaStreamDestroy(stream_thresh); goto ERROR; }
    if ((status = cuda_status(cudaMemset(engine->d_vlt_win_count, 0, sizeof(int)), errbuf, errbuf_size, "cudaMemset(vlt win_count)")) != eslOK) { cudaStreamDestroy(stream_copy); cudaStreamDestroy(stream_thresh); goto ERROR; }

    float *d_bias = NULL;
    if (do_biasfilter && bias_scores) {
      d_bias = engine->d_vlt_bias;
      if ((status = cuda_status(cudaMemcpy(d_bias, bias_scores, sizeof(float) * nwindows, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(vlt bias)")) != eslOK) { cudaStreamDestroy(stream_copy); cudaStreamDestroy(stream_thresh); goto ERROR; }
    }
    cudaEventRecord(engine->evt_d2h0, stream_thresh);
    cuda_compute_viterbi_thresholds_kernel<<<(nwindows + 127) / 128, 128, 0, stream_thresh>>>(
        NULL, d_bias, engine->d_vlt_lengths, nwindows, do_biasfilter,
        B2, F2, vmu, vlambda, scale_w, xw_e_move, nj, base_w,
        max_length, engine->d_vlt_thresholds);
    cudaEventRecord(engine->evt_d2h1, stream_thresh);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "vlt threshold kernel launch")) != eslOK) { cudaStreamDestroy(stream_copy); cudaStreamDestroy(stream_thresh); goto ERROR; }

    /* Wait for both streams to complete before launching main kernel */
    t_stream0 = host_seconds();
    cudaStreamSynchronize(stream_copy);
    cudaStreamSynchronize(stream_thresh);
    t_stream1 = host_seconds();
    local_stats.stream_seconds += t_stream1 - t_stream0;
    local_stats.h2d_seconds += elapsed_seconds(engine->evt_h2d0, engine->evt_h2d1);
    local_stats.threshold_kernel_seconds += elapsed_seconds(engine->evt_d2h0, engine->evt_d2h1);
    t_stream0 = host_seconds();
    cudaStreamDestroy(stream_copy);
    cudaStreamDestroy(stream_thresh);
    t_stream1 = host_seconds();
    local_stats.stream_seconds += t_stream1 - t_stream0;
  }

  /* Launch kernel: dynamically choose warps-per-block for occupancy */
  {
    size_t shmem_per_group = (size_t) Q * VIT_LT_LANES * 3 * 2 * sizeof(int16_t);
    int groups_per_block = VIT_LT_GROUPS_PER_BLOCK;
    while (groups_per_block > 1 && shmem_per_group * groups_per_block > 24 * 1024)
      groups_per_block >>= 1;
    int block_threads = groups_per_block * VIT_LT_LANES;
    int physical_warps_per_block = (block_threads + 31) / 32;
    int nblocks = (nwindows + groups_per_block - 1) / groups_per_block;
    size_t shmem = shmem_per_group * groups_per_block;
    local_stats.warps_per_block = physical_warps_per_block;
    local_stats.grid_blocks        = nblocks;
    local_stats.block_threads      = block_threads;
    local_stats.dynamic_smem_bytes = (int) shmem;
    {
      cudaDeviceProp prop;
      int dev = 0;
      int active_blocks = 0;
      if (cudaGetDevice(&dev) == cudaSuccess &&
          cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                                        cuda_viterbi_longtarget_kernel,
                                                        block_threads, shmem) == cudaSuccess) {
        local_stats.active_blocks_per_sm  = active_blocks;
        local_stats.active_warps_per_sm   = active_blocks * physical_warps_per_block;
        local_stats.max_warps_per_sm      = prop.maxThreadsPerMultiProcessor / 32;
        local_stats.sm_count              = prop.multiProcessorCount;
        local_stats.theoretical_occupancy = local_stats.max_warps_per_sm > 0
                                            ? (double)local_stats.active_warps_per_sm / (double)local_stats.max_warps_per_sm
                                            : 0.0;
        local_stats.grid_sm_coverage      = local_stats.sm_count > 0
                                            ? (double)nblocks / (double)local_stats.sm_count
                                            : 0.0;
      }
    }
    cudaEventRecord(engine->evt_k0);
    cuda_viterbi_longtarget_kernel<<<nblocks, block_threads, shmem>>>(
      engine->d_vlt_dsq, engine->d_vlt_offsets, engine->d_vlt_lengths,
      nwindows,
      cuom->d_rwv, cuom->d_twv, M, Q, cuom->Kp,
      nj, scale_w, max_length,
      cuom->xw_e_loop, cuom->xw_e_move,
      cuom->base_w, cuom->ddbound_w,
      engine->d_vlt_thresholds,
      engine->d_vlt_windows, engine->d_vlt_win_count, engine->vlt_win_alloc);
    cudaEventRecord(engine->evt_k1);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_viterbi_longtarget_kernel launch")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaDeviceSynchronize(), errbuf, errbuf_size, "cuda_viterbi_longtarget sync")) != eslOK) goto ERROR;
    local_stats.kernel_seconds += elapsed_seconds(engine->evt_k0, engine->evt_k1);
  }

  /* Download results */
  cudaEventRecord(engine->evt_d2h0);
  if ((status = cuda_status(cudaMemcpy(&h_win_count, engine->d_vlt_win_count, sizeof(int), cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(vlt win_count)")) != eslOK) goto ERROR;
  if (h_win_count > engine->vlt_win_alloc) h_win_count = engine->vlt_win_alloc;

  if (h_win_count > 0) {
    h_windows = (P7_CUDA_VIT_LT_WINDOW *)malloc(sizeof(P7_CUDA_VIT_LT_WINDOW) * h_win_count);
    if (!h_windows) { status = eslEMEM; goto ERROR; }
    if ((status = cuda_status(cudaMemcpy(h_windows, engine->d_vlt_windows, sizeof(P7_CUDA_VIT_LT_WINDOW) * h_win_count, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(vlt windows)")) != eslOK) goto ERROR;
  }
  cudaEventRecord(engine->evt_d2h1);
  cudaEventSynchronize(engine->evt_d2h1);
  local_stats.d2h_seconds += elapsed_seconds(engine->evt_d2h0, engine->evt_d2h1);
  local_stats.nwindows_out = h_win_count;
  local_stats.device_active_seconds = local_stats.threshold_kernel_seconds + local_stats.kernel_seconds;

  engine->stats.vit_h2d_seconds    += local_stats.h2d_seconds;
  engine->stats.vit_kernel_seconds += local_stats.threshold_kernel_seconds + local_stats.kernel_seconds;
  engine->stats.vit_d2h_seconds    += local_stats.d2h_seconds;
  engine->stats.vit_nseqs          += nwindows;
  engine->stats.vit_nres           += total_packed > nwindows ? (uint64_t)(total_packed - nwindows - 1) : 0;
  engine->stats.vit_nbatches       += 1;

  *ret_windows  = h_windows;
  *ret_nwindows = h_win_count;
  if (stats) *stats = local_stats;
  free(h_offsets);
  free(h_lengths);
  return eslOK;

ERROR:
  free(h_offsets);
  free(h_lengths);
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
