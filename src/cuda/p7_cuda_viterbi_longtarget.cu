#include "p7_cuda_internal.h"
#include "p7_cuda_nucdb_pack.cuh"
#include "nhmmer_cuda_internal.h"
#include <stdlib.h>

#define VIT_LT_MAX_WINDOWS_PER_INPUT 64
#define VIT_LT_LANES                 8
#define VIT_LT_GROUPS_PER_WARP       (32 / VIT_LT_LANES)
#define VIT_LT_GROUPS_PER_BLOCK      4
#define VLT_CHUNK_MAX_DEFAULT        1
#define VLT_REG_STRIDE               24   /* register-based kernel: max M = 768 */

static int vlt_get_chunk_max(void) {
  const char *env = getenv("HMMER_VLT_CHUNK_MAX");
  if (env) { int v = atoi(env); return (v >= 1) ? v : 1; }
  return VLT_CHUNK_MAX_DEFAULT;
}

static inline int vlt_chunk_count(int L, int M, int C_max) {
  if (M <= 0 || L < 2 * M) return 1;
  int c = L / M;
  if (c < 2) return 1;
  return (c > C_max) ? C_max : c;
}

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

/* Register-based scanning Viterbi: one full warp (32 lanes) per window.
 * Each lane holds stride=ceil(M/32) consecutive model positions in registers.
 * This eliminates shared memory latency from the inner loop — the only
 * cross-lane synchronisation is one shfl_up per position for the M/D/I carry.
 * Template parameter STRIDE is the max model positions per lane. */
template<int STRIDE>
__global__ static void
cuda_viterbi_longtarget_reg_kernel(
    const uint8_t *dsq, const int *offsets, const int *lengths,
    const int *seqidx, int nwindows,
    const int16_t *rwv, const int16_t *twv, int M, int Q, int Kp,
    float nj, float scale_w, int max_length,
    int16_t xw_e_loop, int16_t xw_e_move,
    int16_t base_w, int16_t ddbound_w,
    const int16_t *sc_thresholds,
    P7_CUDA_VIT_LT_WINDOW *d_windows, int *d_win_count, int max_windows,
    const int *src1_lengths = NULL, const int *src2_offsets = NULL, int rc_flag = 0)
{
  extern __shared__ uint8_t vlt_reg_mem[];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int warps_per_block = blockDim.x >> 5;
  int bi = blockIdx.x * warps_per_block + warp;
  if (bi >= nwindows) return;

  int stride = (M + 31) / 32;
  int my_start = lane * stride;
  int my_count = stride;
  if (my_start + my_count > M) my_count = M - my_start;
  if (my_count < 0) my_count = 0;

  /* s_q[k] = k % Q, s_z[k] = k / Q — shared across all warps in block */
  uint8_t *s_q = vlt_reg_mem;
  uint8_t *s_z = vlt_reg_mem + M;
  for (int k = threadIdx.x; k < M; k += blockDim.x) {
    s_q[k] = (uint8_t)(k % Q);
    s_z[k] = (uint8_t)(k / Q);
  }
  __syncthreads();

  int src_i = seqidx ? seqidx[bi] : bi;
  int L = lengths[src_i];
  int packed_2bit = (src1_lengths != NULL);
  const uint8_t *s = packed_2bit ? dsq : (dsq + offsets[src_i]);
  int s_offset   = packed_2bit ? offsets[src_i]       : 0;
  int s_src1_len = packed_2bit ? src1_lengths[src_i]  : 0;
  int s_src2_off = packed_2bit ? src2_offsets[src_i]  : 0;
  int16_t my_thresh = sc_thresholds[bi];

  int loc_L = (L < max_length) ? L : max_length;
  float pmove = (2.0f + nj) / ((float)loc_L + 2.0f + nj);
  int16_t xw_move = (int16_t)roundf(scale_w * logf(pmove));

  /* Precompute packed traversal */
  int sp_cur = 0, sp_step = 0, sp_split = 0, sp_jump = 0;
  if (packed_2bit) {
    if (rc_flag) {
      sp_cur = s_offset + L - 1; sp_step = -1;
      if (s_src1_len < L) { sp_split = s_src1_len; sp_jump = s_src2_off - (s_offset - 1); }
    } else {
      sp_cur = s_offset; sp_step = 1;
      if (s_src1_len < L) { sp_split = s_src1_len; sp_jump = s_src2_off - (s_offset + s_src1_len); }
    }
  }

  int16_t reg_M[STRIDE];
  int16_t reg_D[STRIDE];
  int16_t reg_I[STRIDE];
  #pragma unroll
  for (int j = 0; j < STRIDE; j++) reg_M[j] = reg_D[j] = reg_I[j] = -32768;

  int16_t xN = base_w;
  int16_t xB = vlt_i16_add_sat(xN, xw_move);
  int16_t xJ = -32768;
  int16_t xC = -32768;

  for (int i = 1; i <= L; i++) {
    uint8_t x;
    if (packed_2bit) {
      if (__builtin_expect(sp_split && i - 1 == sp_split, 0))
        sp_cur += sp_jump;
      int b = sp_cur >> 2;
      int sh = (sp_cur & 3) << 1;
      x = ((s[b] >> sh) & 0x3);
      if (rc_flag) x ^= 0x3;
      sp_cur += sp_step;
    } else {
      x = s[i];
    }
    xB = (int16_t) __shfl_sync(0xffffffff, (int) xB, 0);

    int16_t km1_M = (my_count > 0) ? reg_M[my_count - 1] : (int16_t)-32768;
    int16_t km1_D = (my_count > 0) ? reg_D[my_count - 1] : (int16_t)-32768;
    int16_t km1_I = (my_count > 0) ? reg_I[my_count - 1] : (int16_t)-32768;
    km1_M = (int16_t) __shfl_up_sync(0xffffffff, (int) km1_M, 1);
    km1_D = (int16_t) __shfl_up_sync(0xffffffff, (int) km1_D, 1);
    km1_I = (int16_t) __shfl_up_sync(0xffffffff, (int) km1_I, 1);
    if (lane == 0) { km1_M = -32768; km1_D = -32768; km1_I = -32768; }

    int16_t dcv = -32768;
    int16_t xE_local = -32768;
    int16_t dmax_local = -32768;

    #pragma unroll
    for (int j = 0; j < STRIDE; j++) {
      if (j >= my_count) break;
      int k = my_start + j;
      int q_k = (int) s_q[k];
      int z_k = (int) s_z[k];
      int16_t old_M = reg_M[j], old_D = reg_D[j], old_I = reg_I[j];

      int16_t sv = vlt_i16_add_sat(xB, twv[(q_k * 7 + p7O_BM) * 8 + z_k]);
      int16_t cand;
      cand = vlt_i16_add_sat(km1_M, twv[(q_k * 7 + p7O_MM) * 8 + z_k]); if (cand > sv) sv = cand;
      cand = vlt_i16_add_sat(km1_I, twv[(q_k * 7 + p7O_IM) * 8 + z_k]); if (cand > sv) sv = cand;
      cand = vlt_i16_add_sat(km1_D, twv[(q_k * 7 + p7O_DM) * 8 + z_k]); if (cand > sv) sv = cand;
      sv = vlt_i16_add_sat(sv, rwv[((int) x * Q + q_k) * 8 + z_k]);
      if (sv > xE_local) xE_local = sv;

      int16_t sv_i = vlt_i16_add_sat(old_M, twv[(q_k * 7 + p7O_MI) * 8 + z_k]);
      cand = vlt_i16_add_sat(old_I, twv[(q_k * 7 + p7O_II) * 8 + z_k]);
      if (cand > sv_i) sv_i = cand;

      int16_t sv_d = dcv;
      dcv = vlt_i16_add_sat(sv, twv[(q_k * 7 + p7O_MD) * 8 + z_k]);
      if (dcv > dmax_local) dmax_local = dcv;

      reg_M[j] = sv;
      reg_I[j] = sv_i;
      reg_D[j] = sv_d;
      km1_M = old_M; km1_D = old_D; km1_I = old_I;
    }

    int xEi = (int) xE_local, dmaxi = (int) dmax_local;
    for (int off = 16; off > 0; off >>= 1) {
      int o;
      o = __shfl_down_sync(0xffffffff, xEi, off);   if (o > xEi) xEi = o;
      o = __shfl_down_sync(0xffffffff, dmaxi, off); if (o > dmaxi) dmaxi = o;
    }
    int16_t xE = (int16_t) __shfl_sync(0xffffffff, xEi, 0);

    if (xE >= my_thresh) {
      /* Emit seed windows for all model positions k where reg_M[j] == xE */
      for (int j = 0; j < my_count; j++) {
        if (reg_M[j] == xE) {
          int k = my_start + j + 1;
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
      #pragma unroll
      for (int j = 0; j < STRIDE; j++) reg_M[j] = reg_D[j] = reg_I[j] = -32768;
      if (lane == 0) {
        xN = base_w;
        xB = vlt_i16_add_sat(xN, xw_move);
        xJ = -32768;
        xC = -32768;
      }
    } else {
      if (lane == 0) {
        xN = xN;
        int16_t c2 = vlt_i16_add_sat(xE, xw_e_move);
        xC = xC > c2 ? xC : c2;
        int16_t j2 = vlt_i16_add_sat(xE, xw_e_loop);
        xJ = xJ > j2 ? xJ : j2;
        int16_t b1 = vlt_i16_add_sat(xJ, xw_move);
        int16_t b2 = vlt_i16_add_sat(xN, xw_move);
        xB = b1 > b2 ? b1 : b2;
      }

      /* Lazy-F D-state propagation */
      int dmax_all = __shfl_sync(0xffffffff, dmaxi, 0);
      int xB_bcast = __shfl_sync(0xffffffff, (int) xB, 0);
      if (dmax_all + (int) ddbound_w > xB_bcast) {
        int16_t from_dcv = (int16_t) __shfl_up_sync(0xffffffff, (int) dcv, 1);
        if (lane == 0) from_dcv = -32768;
        #pragma unroll
        for (int j = 0; j < STRIDE; j++) {
          if (j >= my_count) break;
          int k = my_start + j;
          int q_k = (int) s_q[k];
          int z_k = (int) s_z[k];
          if (from_dcv > reg_D[j]) reg_D[j] = from_dcv;
          from_dcv = vlt_i16_add_sat(reg_D[j], twv[(7 * Q + q_k) * 8 + z_k]);
        }
        int any_improved;
        do {
          from_dcv = (int16_t) __shfl_up_sync(0xffffffff, (int) from_dcv, 1);
          if (lane == 0) from_dcv = -32768;
          int improved = 0;
          #pragma unroll
          for (int j = 0; j < STRIDE; j++) {
            if (j >= my_count) break;
            int k = my_start + j;
            int q_k = (int) s_q[k];
            int z_k = (int) s_z[k];
            if (from_dcv > reg_D[j]) { reg_D[j] = from_dcv; improved = 1; }
            from_dcv = vlt_i16_add_sat(reg_D[j], twv[(7 * Q + q_k) * 8 + z_k]);
          }
          any_improved = __any_sync(0xffffffff, improved);
        } while (any_improved);
      } else {
        int16_t from_dcv = (int16_t) __shfl_up_sync(0xffffffff, (int) dcv, 1);
        if (lane == 0) from_dcv = -32768;
        if (my_count > 0 && from_dcv > reg_D[0]) reg_D[0] = from_dcv;
      }
    }
  }
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
    P7_CUDA_VIT_LT_WINDOW *d_windows, int *d_win_count, int max_windows,
    const int *src1_lengths = NULL, const int *src2_offsets = NULL, int rc_flag = 0)
{
  extern __shared__ int16_t vlt_mem[];
  int groups_per_block = blockDim.x / VIT_LT_LANES;
  int group = threadIdx.x / VIT_LT_LANES;
  int lane = threadIdx.x & (VIT_LT_LANES - 1);
  int warp_lane = threadIdx.x & 31;
  unsigned int mask = 0xffu << (warp_lane & ~(VIT_LT_LANES - 1));
  int bi = blockIdx.x * groups_per_block + group;
  int N = Q * VIT_LT_LANES;
  size_t group_stride = (size_t) N * 3;
  if (group >= groups_per_block || bi >= nwindows) return;
  int16_t *dp = vlt_mem + ((size_t) group * group_stride);

  int src_i = seqidx ? seqidx[bi] : bi;
  int L = lengths[src_i];
  int packed_2bit = (src1_lengths != NULL);
  const uint8_t *s = packed_2bit ? dsq : (dsq + offsets[src_i]);
  int s_offset   = packed_2bit ? offsets[src_i]       : 0;
  int s_src1_len = packed_2bit ? src1_lengths[src_i]  : 0;
  int s_src2_off = packed_2bit ? src2_offsets[src_i]  : 0;
  int16_t my_thresh = sc_thresholds[bi];

  /* Per-window length reconfiguration (matches CPU ReconfigRestLength) */
  int loc_L = (L < max_length) ? L : max_length;
  float pmove = (2.0f + nj) / ((float)loc_L + 2.0f + nj);
  int16_t xw_move = (int16_t)roundf(scale_w * logf(pmove));

  /* Precompute packed traversal */
  int sp_cur = 0, sp_step = 0, sp_split = 0, sp_jump = 0;
  if (packed_2bit) {
    if (rc_flag) {
      sp_cur = s_offset + L - 1; sp_step = -1;
      if (s_src1_len < L) { sp_split = s_src1_len; sp_jump = s_src2_off - (s_offset - 1); }
    } else {
      sp_cur = s_offset; sp_step = 1;
      if (s_src1_len < L) { sp_split = s_src1_len; sp_jump = s_src2_off - (s_offset + s_src1_len); }
    }
  }

  for (int q = 0; q < Q; q++) {
    int cell = (q + lane * Q) * 3;
    dp[cell + 0] = -32768;
    dp[cell + 1] = -32768;
    dp[cell + 2] = -32768;
  }

  int16_t xN = base_w;
  int16_t xB = vlt_i16_add_sat(xN, xw_move);
  int16_t xJ = -32768;
  int16_t xC = -32768;

  for (int i = 1; i <= L; i++) {
    uint8_t x;
    if (packed_2bit) {
      if (__builtin_expect(sp_split && i - 1 == sp_split, 0))
        sp_cur += sp_jump;
      int b = sp_cur >> 2;
      int sh = (sp_cur & 3) << 1;
      x = ((s[b] >> sh) & 0x3);
      if (rc_flag) x ^= 0x3;
      sp_cur += sp_step;
    } else {
      x = s[i];
    }

    xB = (int16_t) __shfl_sync(mask, (int) xB, 0, VIT_LT_LANES);
    int16_t mpv = dp[((Q - 1) + lane * Q) * 3 + 0];
    int16_t dpv = dp[((Q - 1) + lane * Q) * 3 + 1];
    int16_t ipv = dp[((Q - 1) + lane * Q) * 3 + 2];
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

      mpv = dp[cell + 0];
      dpv = dp[cell + 1];
      ipv = dp[cell + 2];

      dp[cell + 0] = sv;
      dp[cell + 1] = dcv;
      dcv = vlt_i16_add_sat(sv, twv[vlt_twv_idx(p7O_MD, q, lane, Q)]);
      if (dcv > dmax_lane) dmax_lane = dcv;
      cand = vlt_i16_add_sat(mpv, twv[vlt_twv_idx(p7O_MI, q, lane, Q)]);
      sv = vlt_i16_add_sat(ipv, twv[vlt_twv_idx(p7O_II, q, lane, Q)]);
      dp[cell + 2] = cand > sv ? cand : sv;
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
        if (dp[cell + 0] == xE) {
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
        dp[cell + 0] = -32768;
        dp[cell + 1] = -32768;
        dp[cell + 2] = -32768;
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
          if (dcv > dp[cell + 1]) dp[cell + 1] = dcv;
          dcv = vlt_i16_add_sat(dp[cell + 1], twv[vlt_twv_idx(p7O_DD, q, lane, Q)]);
        }
        int completed;
        do {
          completed = 1;
          dcv = (int16_t) __shfl_up_sync(mask, (int) dcv, 1, VIT_LT_LANES);
          if (lane == 0) dcv = -32768;
          for (int q = 0; q < Q; q++) {
            int cell = (q + lane * Q) * 3;
            int gt = (dcv > dp[cell + 1]);
            if (!__any_sync(mask, gt)) {
              completed = 0;
              break;
            }
            if (gt) dp[cell + 1] = dcv;
            dcv = vlt_i16_add_sat(dp[cell + 1], twv[vlt_twv_idx(p7O_DD, q, lane, Q)]);
          }
        } while (__all_sync(mask, completed));
      } else {
        dcv = (int16_t) __shfl_up_sync(mask, (int) dcv, 1, VIT_LT_LANES);
        if (lane == 0) dcv = -32768;
        dp[lane * Q * 3 + 1] = dcv;
      }
    }
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
                                  char *errbuf, int errbuf_size,
                                  const int *d_src1_lengths = NULL, const int *d_src2_offsets = NULL, int vlt_rc_flag = 0)
{
  int status = eslOK;
  int Q = cuom->Qw;
  int M = cuom->M;
  int h_win_count = 0;
  P7_CUDA_VIT_LT_WINDOW *h_windows = NULL;
  int use_chunking = 0;

  {
    size_t shmem_per_group = (size_t) Q * VIT_LT_LANES * 3 * sizeof(int16_t);
    int groups_per_block = VIT_LT_GROUPS_PER_BLOCK;

    /* Dynamic configuration for shared-memory kernel (fallback for M>768):
     * maximize concurrent windows/SM = active_blocks * groups_per_block. */
    {
      cudaDeviceProp prop;
      int dev = 0;
      if (cudaGetDevice(&dev) == cudaSuccess &&
          cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {

        int best_groups = 1;
        int best_concurrent = 0;
        for (int g = 8; g >= 1; g >>= 1) {
          size_t try_shmem = shmem_per_group * g;
          if (try_shmem > (size_t)prop.sharedMemPerMultiprocessor) continue;
          int try_threads = g * VIT_LT_LANES;
          int active = 0;
          if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
                cuda_viterbi_longtarget_kernel, try_threads, try_shmem) == cudaSuccess && active > 0) {
            int concurrent = active * g;
            if (concurrent > best_concurrent) {
              best_concurrent = concurrent;
              best_groups = g;
            }
          }
        }
        groups_per_block = best_groups;
      } else {
        while (groups_per_block > 1 && shmem_per_group * groups_per_block > 24 * 1024)
          groups_per_block >>= 1;
      }
    }

    int block_threads = groups_per_block * VIT_LT_LANES;
    int physical_warps_per_block = (block_threads + 31) / 32;
    size_t shmem = shmem_per_group * groups_per_block;

    /* --- Chunked launch path --- */
    int chunk_max = vlt_get_chunk_max();
    use_chunking = (chunk_max > 1 && d_seqidx != NULL && engine->batch_nseq > 0);
    int nvirtual = nwindows;
    int nwindows_orig = nwindows;

    if (use_chunking) {
      int *h_seqidx   = (int *)malloc(nwindows * sizeof(int));
      int *h_all_off  = (int *)malloc(engine->batch_nseq * sizeof(int));
      int *h_all_len  = (int *)malloc(engine->batch_nseq * sizeof(int));
      int16_t *h_thr  = (int16_t *)malloc(nwindows * sizeof(int16_t));
      if (!h_seqidx || !h_all_off || !h_all_len || !h_thr) { status = eslEMEM; goto ERROR; }

      cudaMemcpy(h_seqidx, d_seqidx, nwindows * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_all_off, d_offsets, engine->batch_nseq * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_all_len, d_lengths, engine->batch_nseq * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_thr, engine->d_vlt_thresholds, nwindows * sizeof(int16_t), cudaMemcpyDeviceToHost);

      nvirtual = 0;
      for (int w = 0; w < nwindows; w++) {
        int L = h_all_len[h_seqidx[w]];
        nvirtual += vlt_chunk_count(L, M, chunk_max);
      }

      if (nvirtual > engine->h_vlt_virt_remap_alloc) {
        free(engine->h_vlt_virt_parent_id);
        free(engine->h_vlt_virt_chunk_start);
        engine->h_vlt_virt_parent_id  = (int *)malloc(nvirtual * sizeof(int));
        engine->h_vlt_virt_chunk_start = (int *)malloc(nvirtual * sizeof(int));
        engine->h_vlt_virt_remap_alloc = nvirtual;
        if (!engine->h_vlt_virt_parent_id || !engine->h_vlt_virt_chunk_start) { status = eslEMEM; goto ERROR; }
      }

      int *h_voff = (int *)malloc(nvirtual * sizeof(int));
      int *h_vlen = (int *)malloc(nvirtual * sizeof(int));
      int16_t *h_vthr = (int16_t *)malloc(nvirtual * sizeof(int16_t));
      if (!h_voff || !h_vlen || !h_vthr) { status = eslEMEM; goto ERROR; }

      int v = 0;
      for (int w = 0; w < nwindows; w++) {
        int src_i = h_seqidx[w];
        int base_off = h_all_off[src_i];
        int L = h_all_len[src_i];
        int C = vlt_chunk_count(L, M, chunk_max);
        int useful = (C > 1) ? (L - M) / C : L;
        for (int c = 0; c < C; c++) {
          int cstart = c * useful;
          int clen = (c == C - 1) ? (L - cstart) : (useful + M);
          h_voff[v] = base_off + cstart;
          h_vlen[v] = clen;
          h_vthr[v] = h_thr[w];
          engine->h_vlt_virt_parent_id[v] = w;
          engine->h_vlt_virt_chunk_start[v] = cstart;
          v++;
        }
      }

      if (nvirtual > engine->vlt_virt_alloc) {
        if (engine->d_vlt_virt_offsets) cudaFree(engine->d_vlt_virt_offsets);
        if (engine->d_vlt_virt_lengths) cudaFree(engine->d_vlt_virt_lengths);
        if (engine->d_vlt_virt_thresholds) cudaFree(engine->d_vlt_virt_thresholds);
        engine->d_vlt_virt_offsets = NULL;
        engine->d_vlt_virt_lengths = NULL;
        engine->d_vlt_virt_thresholds = NULL;
        engine->vlt_virt_alloc = 0;
        if (cuda_status(cudaMalloc((void **)&engine->d_vlt_virt_offsets, nvirtual * sizeof(int)), errbuf, errbuf_size, "cudaMalloc virt offsets") != eslOK) goto ERROR;
        if (cuda_status(cudaMalloc((void **)&engine->d_vlt_virt_lengths, nvirtual * sizeof(int)), errbuf, errbuf_size, "cudaMalloc virt lengths") != eslOK) goto ERROR;
        if (cuda_status(cudaMalloc((void **)&engine->d_vlt_virt_thresholds, nvirtual * sizeof(int16_t)), errbuf, errbuf_size, "cudaMalloc virt thresh") != eslOK) goto ERROR;
        engine->vlt_virt_alloc = nvirtual;
      }
      cudaMemcpy(engine->d_vlt_virt_offsets, h_voff, nvirtual * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(engine->d_vlt_virt_lengths, h_vlen, nvirtual * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(engine->d_vlt_virt_thresholds, h_vthr, nvirtual * sizeof(int16_t), cudaMemcpyHostToDevice);

      /* Grow kernel output buffers for nvirtual */
      int max_out_v = nvirtual * VIT_LT_MAX_WINDOWS_PER_INPUT;
      if (engine->vlt_win_alloc < max_out_v) {
        if (engine->d_vlt_windows) cudaFree(engine->d_vlt_windows);
        free(engine->h_vlt_windows);
        engine->d_vlt_windows = NULL;
        engine->h_vlt_windows = NULL;
        engine->vlt_win_alloc = 0;
        if (cuda_status(cudaMalloc((void **)&engine->d_vlt_windows, sizeof(P7_CUDA_VIT_LT_WINDOW) * max_out_v), errbuf, errbuf_size, "cudaMalloc vlt windows chunked") != eslOK) goto ERROR;
        engine->h_vlt_windows = (P7_CUDA_VIT_LT_WINDOW *)malloc(sizeof(P7_CUDA_VIT_LT_WINDOW) * max_out_v);
        if (!engine->h_vlt_windows) { status = eslEMEM; goto ERROR; }
        engine->vlt_win_alloc = max_out_v;
      }
      if (engine->vlt_count_alloc < nvirtual) {
        if (engine->d_vlt_win_count) cudaFree(engine->d_vlt_win_count);
        if (engine->d_vlt_win_offsets) cudaFree(engine->d_vlt_win_offsets);
        engine->d_vlt_win_count = NULL;
        engine->d_vlt_win_offsets = NULL;
        engine->vlt_count_alloc = 0;
        if (cuda_status(cudaMalloc((void **)&engine->d_vlt_win_count, sizeof(int) * nvirtual), errbuf, errbuf_size, "cudaMalloc vlt win_count chunked") != eslOK) goto ERROR;
        if (cuda_status(cudaMalloc((void **)&engine->d_vlt_win_offsets, sizeof(int) * (nvirtual + 1)), errbuf, errbuf_size, "cudaMalloc vlt win_offsets chunked") != eslOK) goto ERROR;
        engine->vlt_count_alloc = nvirtual;
      }
      cudaMemset(engine->d_vlt_win_count, 0, nvirtual * sizeof(int));

      free(h_seqidx); free(h_all_off); free(h_all_len); free(h_thr);
      free(h_voff); free(h_vlen); free(h_vthr);

      nwindows = nvirtual;
    }

    /* Kernel launch — use virtual arrays when chunking */
    const int     *launch_offsets    = use_chunking ? engine->d_vlt_virt_offsets    : d_offsets;
    const int     *launch_lengths    = use_chunking ? engine->d_vlt_virt_lengths    : d_lengths;
    const int     *launch_seqidx     = use_chunking ? NULL                          : d_seqidx;
    const int16_t *launch_thresholds = use_chunking ? engine->d_vlt_virt_thresholds : engine->d_vlt_thresholds;
    const int     *launch_src1_lengths = use_chunking ? NULL : d_src1_lengths;
    const int     *launch_src2_offsets = use_chunking ? NULL : d_src2_offsets;
    int            launch_rc_flag      = use_chunking ? 0    : vlt_rc_flag;

    const int16_t *launch_rwv = cuom->d_rwv_nuc ? cuom->d_rwv_nuc : cuom->d_rwv;
    const int16_t *launch_twv = cuom->d_twv;
    int launch_Kp = cuom->Kp_nuc ? cuom->Kp_nuc : cuom->Kp;

    /* Select register-based kernel for 256 <= M <= VLT_REG_STRIDE*32 = 768.
     * Models below M=256 have short serial chains (stride<8) where the shmem
     * kernel performs well, and the register kernel has edge-case issues with
     * very short models. */
    int use_reg_kernel = (M >= 256 && M <= VLT_REG_STRIDE * 32);
    const char *env_vlt_reg = getenv("HMMER_VLT_REG");
    if (env_vlt_reg) use_reg_kernel = atoi(env_vlt_reg);

    if (use_reg_kernel) {
      int reg_warps_per_block = 4;
      const char *env_vlt_warps = getenv("HMMER_VLT_REG_WARPS");
      if (env_vlt_warps) { int v = atoi(env_vlt_warps); if (v >= 1 && v <= 16) reg_warps_per_block = v; }
      int reg_block_threads = reg_warps_per_block * 32;
      size_t reg_shmem = 2 * M;
      int nblocks = (nwindows + reg_warps_per_block - 1) / reg_warps_per_block;

      local_stats->warps_per_block    = reg_warps_per_block;
      local_stats->grid_blocks        = nblocks;
      local_stats->block_threads      = reg_block_threads;
      local_stats->dynamic_smem_bytes = (int) reg_shmem;

      {
        cudaDeviceProp prop;
        int dev = 0;
        int active_blocks = 0;
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
            cuda_viterbi_longtarget_reg_kernel<VLT_REG_STRIDE>,
            reg_block_threads, reg_shmem);
          if (active_blocks > 0) {
            local_stats->active_blocks_per_sm  = active_blocks;
            local_stats->active_warps_per_sm   = active_blocks * reg_warps_per_block;
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
      }

      cudaEventRecord(engine->evt_k0);
      cuda_viterbi_longtarget_reg_kernel<VLT_REG_STRIDE><<<nblocks, reg_block_threads, reg_shmem>>>(
        d_dsq, launch_offsets, launch_lengths, launch_seqidx,
        nwindows, launch_rwv, launch_twv, M, Q, launch_Kp,
        nj, scale_w, max_length,
        cuom->xw_e_loop, cuom->xw_e_move,
        cuom->base_w, cuom->ddbound_w,
        launch_thresholds,
        engine->d_vlt_windows, engine->d_vlt_win_count, engine->vlt_win_alloc,
        launch_src1_lengths, launch_src2_offsets, launch_rc_flag);
      cudaEventRecord(engine->evt_k1);
      if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_viterbi_longtarget_reg_kernel launch")) != eslOK) goto ERROR;
      if ((status = cuda_status(cudaEventSynchronize(engine->evt_k1), errbuf, errbuf_size, "cudaEventSynchronize(vlt reg kernel)")) != eslOK) goto ERROR;
    } else {
      int nblocks = (nwindows + groups_per_block - 1) / groups_per_block;
      local_stats->warps_per_block    = physical_warps_per_block;
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
        d_dsq, launch_offsets, launch_lengths, launch_seqidx,
        nwindows, launch_rwv, launch_twv, M, Q, launch_Kp,
        nj, scale_w, max_length,
        cuom->xw_e_loop, cuom->xw_e_move,
        cuom->base_w, cuom->ddbound_w,
        launch_thresholds,
        engine->d_vlt_windows, engine->d_vlt_win_count, engine->vlt_win_alloc,
        launch_src1_lengths, launch_src2_offsets, launch_rc_flag);
      cudaEventRecord(engine->evt_k1);
      if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_viterbi_longtarget_kernel launch")) != eslOK) goto ERROR;
      if ((status = cuda_status(cudaEventSynchronize(engine->evt_k1), errbuf, errbuf_size, "cudaEventSynchronize(vlt kernel)")) != eslOK) goto ERROR;
    }
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

  /* Remap chunked seeds back to parent coordinates */
  if (use_chunking && h_win_count > 0) {
    for (int i = 0; i < h_win_count; i++) {
      int vid = h_windows[i].window_id;
      h_windows[i].window_id = engine->h_vlt_virt_parent_id[vid];
      h_windows[i].position += engine->h_vlt_virt_chunk_start[vid];
    }
    qsort(h_windows, h_win_count, sizeof(P7_CUDA_VIT_LT_WINDOW),
          nhmmer_gpu_vit_window_compare);
  }

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
                                          char *errbuf, int errbuf_size,
                                          const int *d_src1_lengths = NULL, const int *d_src2_offsets = NULL, int vlt_rc_flag = 0)
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
                                             errbuf, errbuf_size,
                                             d_src1_lengths, d_src2_offsets, vlt_rc_flag);
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
                                             errbuf, errbuf_size,
                                             engine->batch_packed_2bit ? engine->d_gather_src1_lengths : NULL,
                                             engine->batch_packed_2bit ? engine->d_gather_src2_offsets : NULL,
                                             engine->batch_packed_2bit ? engine->batch_rc_flag : 0);
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
                                                     errbuf, errbuf_size,
                                                     engine->batch_packed_2bit ? engine->d_gather_src1_lengths : NULL,
                                                     engine->batch_packed_2bit ? engine->d_gather_src2_offsets : NULL,
                                                     engine->batch_packed_2bit ? engine->batch_rc_flag : 0);
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
