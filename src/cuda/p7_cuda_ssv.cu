#include "p7_cuda_internal.h"
#include "p7_cuda_nucdb_pack.cuh"

__device__ static inline uint8_t
u8_sub_sat(uint8_t a, uint8_t b)
{
  return (a > b) ? (uint8_t) (a - b) : 0;
}

__device__ static inline uint8_t
u8_add_sat(uint8_t a, uint8_t b)
{
  unsigned int v = (unsigned int) a + (unsigned int) b;
  return (v > 255) ? 255 : (uint8_t) v;
}

/* Maximum model length for register-based SSV. stride = ceil(M/32) ≤ 10 for M ≤ 320. */
#define SSV_OPT_MAX_STRIDE 10

__global__ static void
cuda_ssv_opt_kernel(const uint8_t *dsq, const int *offsets, const int *lengths,
                    const uint8_t *tjb_by_seq, int nseq,
                    const uint8_t *rbv, int M, int Q, int Kp,
                    uint8_t tbm_b, uint8_t tec_b, uint8_t tjb_b,
                    uint8_t base_b, uint8_t bias_b,
                    int *raw_sc, int *overflow)
{
  extern __shared__ uint8_t mem[];
  int seq = blockIdx.x;
  int tid = threadIdx.x;
  if (seq >= nseq) return;

  const uint8_t *sdsq = dsq + offsets[seq];
  int L = lengths[seq];
  uint8_t seq_tjb_b = tjb_by_seq[seq];

  int stride = (M + 31) / 32;
  int my_start = tid * stride;
  int my_count = stride;
  if (my_start + my_count > M) my_count = M - my_start;
  if (my_count < 0) my_count = 0;

  /* Precompute q/z lookup in shared memory — eliminates integer division from inner loop */
  uint8_t *s_q = mem;
  uint8_t *s_z = mem + M;
  for (int k = tid; k < M; k += 32) {
    s_q[k] = (uint8_t)(k % Q);
    s_z[k] = (uint8_t)(k / Q);
  }
  __syncthreads();

  __shared__ int use_full_msv;

  if ((int) seq_tjb_b + (int) tbm_b + (int) tec_b + (int) bias_b < 127) {
    uint8_t xB_ssv = u8_sub_sat(base_b, (uint8_t)(seq_tjb_b + tbm_b));
    uint8_t xE_local = 0;

    /* Register-based DP state — each thread owns a contiguous stripe */
    uint8_t prev_reg[SSV_OPT_MAX_STRIDE];
    for (int j = 0; j < my_count; j++) prev_reg[j] = 0;

    if (tid == 0) {
      overflow[seq] = 0;
      use_full_msv = 0;
    }

    uint8_t last_prev = 0;

    for (int i = 1; i <= L; i++) {
      uint8_t x = sdsq[i];

      /* Warp shuffle: thread tid gets dp[i-1][last_node_of_thread_tid-1].
       * Since last_prev = dp[i-1][last_node_of_this_thread] (set at end of prev iteration),
       * thread tid receives dp[i-1][my_start] from thread tid-1. */
      uint8_t from_left = __shfl_up_sync(0xffffffff, last_prev, 1);
      if (tid == 0) from_left = 0;

      for (int j = 0; j < my_count; j++) {
        int km1 = my_start + j;
        uint8_t old_prev = prev_reg[j];
        uint8_t rsc = rbv[((int) x * Q + (int) s_q[km1]) * 16 + (int) s_z[km1]];
        uint8_t v = (from_left > xB_ssv) ? from_left : xB_ssv;
        v = u8_add_sat(v, bias_b);
        v = u8_sub_sat(v, rsc);
        prev_reg[j] = v;
        if (v > xE_local) xE_local = v;
        from_left = old_prev;
      }

      last_prev = (my_count > 0) ? prev_reg[my_count - 1] : 0;
    }

    /* Single warp reduction at end */
    for (int s = 16; s > 0; s >>= 1) {
      uint8_t other = __shfl_down_sync(0xffffffff, xE_local, s);
      if (other > xE_local) xE_local = other;
    }

    if (tid == 0) {
      int shifted_xE = (int) xE_local - (int) base_b + (int) seq_tjb_b + (int) tbm_b + 128;
      if (shifted_xE >= 255 - (int) bias_b) {
        if ((int) base_b - (int) seq_tjb_b - (int) tbm_b < 128) {
          use_full_msv = 1;
        } else {
          raw_sc[seq] = 0;
          overflow[seq] = 1;
        }
      } else if ((int) xE_local >= 255 - (int) bias_b) {
        raw_sc[seq] = 0;
        overflow[seq] = 1;
      } else {
        int xJ = (int) xE_local - (int) tec_b;
        if (xJ > (int) base_b) {
          use_full_msv = 1;
        } else {
          raw_sc[seq] = xJ;
          overflow[seq] = 0;
        }
      }
    }
    __syncthreads();
    if (!use_full_msv) return;
  }

  /* Full MSV fallback — only ~0.3% of sequences reach here.
   * Reuse shared memory for prev/curr DP arrays. */
  uint8_t *prev = mem;
  uint8_t *curr = prev + M + 1;

  for (int k = tid; k <= M; k += 32) prev[k] = 0;
  if (tid == 0) overflow[seq] = 0;
  __syncthreads();

  uint8_t xJ = 0;
  uint8_t tjbm_b = (uint8_t)(seq_tjb_b + tbm_b);
  uint8_t xB = u8_sub_sat(base_b, tjbm_b);

  for (int i = 1; i <= L; i++) {
    uint8_t xE_pos = 0;
    uint8_t x = sdsq[i];

    for (int k = tid + 1; k <= M; k += 32) {
      int km1 = k - 1;
      int q   = km1 % Q;
      int z   = km1 / Q;
      uint8_t rsc = rbv[((int) x * Q + q) * 16 + z];
      uint8_t v = prev[k-1] > xB ? prev[k-1] : xB;
      v = u8_add_sat(v, bias_b);
      v = u8_sub_sat(v, rsc);
      curr[k] = v;
      if (v > xE_pos) xE_pos = v;
      if (u8_add_sat(v, bias_b) == 255) overflow[seq] = 1;
    }
    if (tid == 0) curr[0] = 0;
    __syncthreads();

    for (int s = 16; s > 0; s >>= 1) {
      uint8_t other = __shfl_down_sync(0xffffffff, xE_pos, s);
      if (other > xE_pos) xE_pos = other;
    }
    __shared__ uint8_t xE_shared;
    if (tid == 0) xE_shared = u8_sub_sat(xE_pos, tec_b);
    __syncthreads();

    if (tid == 0) {
      if (xE_shared > xJ) xJ = xE_shared;
      uint8_t maxBJ = (base_b > xJ) ? base_b : xJ;
      xB = u8_sub_sat(maxBJ, tjbm_b);
    }
    __syncthreads();

    uint8_t *tmp = prev;
    prev = curr;
    curr = tmp;
  }

  if (tid == 0) raw_sc[seq] = (int) xJ;
}

__global__ static void
cuda_ssv_fused_kernel(const uint8_t *dsq, const int *offsets, const int *lengths, const uint8_t *tjb_by_seq, int nseq,
                      const uint8_t *rbv, int M, int Q, int Kp,
                      uint8_t tbm_b, uint8_t tec_b, uint8_t tjb_b, uint8_t base_b, uint8_t bias_b,
                      int *raw_sc, int *overflow)
{
  extern __shared__ uint8_t mem[];
  int seq = blockIdx.x;
  int tid = threadIdx.x;
  if (seq >= nseq) return;

  uint8_t *prev = mem;
  uint8_t *curr = prev + M + 1;
  const uint8_t *sdsq = dsq + offsets[seq];
  int L = lengths[seq];
  uint8_t seq_tjb_b = tjb_by_seq[seq];
  __shared__ int use_full_msv;

  if ((int) seq_tjb_b + (int) tbm_b + (int) tec_b + (int) bias_b < 127) {
    uint8_t xE_global_local = 0;
    uint8_t xB_ssv = u8_sub_sat(base_b, (uint8_t) (seq_tjb_b + tbm_b));

    for (int k = tid; k <= M; k += blockDim.x) prev[k] = 0;
    if (tid == 0) {
      overflow[seq] = 0;
      use_full_msv = 0;
    }
    __syncthreads();

    for (int i = 1; i <= L; i++) {
      uint8_t x = sdsq[i];

      for (int k = tid + 1; k <= M; k += blockDim.x) {
        int km1 = k - 1;
        int q   = km1 % Q;
        int z   = km1 / Q;
        uint8_t rsc = rbv[((int) x * Q + q) * 16 + z];
        uint8_t v = prev[k-1] > xB_ssv ? prev[k-1] : xB_ssv;
        v = u8_add_sat(v, bias_b);
        v = u8_sub_sat(v, rsc);
        curr[k] = v;
        if (v > xE_global_local) xE_global_local = v;
      }
      if (tid == 0) curr[0] = 0;
      __syncthreads();

      uint8_t *tmp = prev;
      prev = curr;
      curr = tmp;
    }

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      uint8_t other = __shfl_down_sync(0xffffffff, xE_global_local, stride);
      if (other > xE_global_local) xE_global_local = other;
    }

    if (tid == 0) {
      int shifted_xE = (int) xE_global_local - (int) base_b + (int) seq_tjb_b + (int) tbm_b + 128;
      if (shifted_xE >= 255 - (int) bias_b) {
        if ((int) base_b - (int) seq_tjb_b - (int) tbm_b < 128) {
          use_full_msv = 1;
        } else {
          raw_sc[seq] = 0;
          overflow[seq] = 1;
        }
      } else if ((int) xE_global_local >= 255 - (int) bias_b) {
        raw_sc[seq] = 0;
        overflow[seq] = 1;
      } else {
        int xJ = (int) xE_global_local - (int) tec_b;
        if (xJ > (int) base_b) {
          use_full_msv = 1;
        } else {
          raw_sc[seq] = xJ;
          overflow[seq] = 0;
        }
      }
    }
    __syncthreads();
    if (!use_full_msv) return;
  }

  /* Full MSV fallback — only ~0.3% of sequences reach here */
  for (int k = tid; k <= M; k += blockDim.x) prev[k] = 0;
  if (tid == 0) overflow[seq] = 0;
  __syncthreads();

  uint8_t xJ = 0;
  uint8_t tjbm_b = (uint8_t) (seq_tjb_b + tbm_b);
  uint8_t xB = u8_sub_sat(base_b, tjbm_b);

  for (int i = 1; i <= L; i++) {
    uint8_t xE_local = 0;
    uint8_t x = sdsq[i];

    for (int k = tid + 1; k <= M; k += blockDim.x) {
      int km1 = k - 1;
      int q   = km1 % Q;
      int z   = km1 / Q;
      uint8_t rsc = rbv[((int) x * Q + q) * 16 + z];
      uint8_t v = prev[k-1] > xB ? prev[k-1] : xB;
      v = u8_add_sat(v, bias_b);
      v = u8_sub_sat(v, rsc);
      curr[k] = v;
      if (v > xE_local) xE_local = v;
      if (u8_add_sat(v, bias_b) == 255) overflow[seq] = 1;
    }
    if (tid == 0) curr[0] = 0;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      uint8_t other = __shfl_down_sync(0xffffffff, xE_local, stride);
      if (other > xE_local) xE_local = other;
    }
    __shared__ uint8_t xE_shared;
    if (tid == 0) xE_shared = u8_sub_sat(xE_local, tec_b);
    __syncthreads();

    if (tid == 0) {
      if (xE_shared > xJ) xJ = xE_shared;
      uint8_t maxBJ = (base_b > xJ) ? base_b : xJ;
      xB = u8_sub_sat(maxBJ, tjbm_b);
    }
    __syncthreads();

    uint8_t *tmp = prev;
    prev = curr;
    curr = tmp;
  }

  if (tid == 0) raw_sc[seq] = (int) xJ;
}

/* ================================================================
 * Fused SSV + Null + Bias + F1 Gate kernel
 *
 * Eliminates 3 separate kernel launches and sync barriers by computing
 * null score, bias filter, and Gumbel P-value gating in-line after
 * the SSV DP, all within the same warp/block.
 * ================================================================ */

#define SSV_FUSED_MAX_STRIDE 20  /* register-based fast path for M <= 640 */

template <int STRIDE, int WARPS>
__global__ __launch_bounds__(32 * WARPS, ((1536) / (32 * WARPS) < 24 ? (1536) / (32 * WARPS) : 24))
static void
cuda_ssv_null_bias_gate_kernel(
    const uint8_t *dsq, const int *offsets, const int *lengths,
    const uint8_t *tjb_by_seq, int nseq,
    const uint8_t *rbv, const uint8_t *rbv_lin, int M, int Q, int Kp,
    uint8_t tbm_b, uint8_t tec_b, uint8_t tjb_b,
    uint8_t base_b, uint8_t bias_b,
    float scale_b,
    /* bias filter params */
    const float *bias_pi, const float *bias_t, const float *bias_eo,
    int do_biasfilter,
    int bias_gate_B,
    /* F1 Gumbel gate params */
    float ev_mu, float ev_lambda, float F1, float log2_inv,
    /* outputs: raw scores still stored for --gpu-ssv-compare compatibility */
    int *raw_sc, int *overflow_out,
    /* outputs: null/bias/survivors for downstream stages */
    float *nullsc_out, float *filtersc_out,
    int *survivor_idx, int *survivor_counter,
    float *survivor_usc, float *survivor_filtersc, int *survivor_status,
    /* packed 2-bit mode: when src1_lengths != NULL, offsets are residue-positions
     * into the 2-bit packed buffer and residues are fetched via p7_nucdb_fetch1 */
    const int *src1_lengths = NULL, const int *src2_offsets = NULL, int rc_flag = 0)
{
  /* Block layout: WARPS warps per block, one sequence per warp.
   * Shared memory layout (extern):
   *   [0 ..   2*M)            : s_q, s_z lookup (block-shared, written once)
   *   [2*M .. 2*M + WARPS*2*(M+1)) : per-warp prev/curr buffers for the
   *                                  full-MSV fallback path (rare).
   */
  extern __shared__ uint8_t mem[];
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int seq  = blockIdx.x * WARPS + warp;
  if (seq >= nseq) return;

  int packed_2bit = (src1_lengths != NULL);
  const uint8_t *sdsq = packed_2bit ? dsq : (dsq + offsets[seq]);
  int s_offset   = packed_2bit ? offsets[seq]       : 0;
  int s_src1_len = packed_2bit ? src1_lengths[seq]  : 0;
  int s_src2_off = packed_2bit ? src2_offsets[seq]  : 0;
  int L = lengths[seq];
  uint8_t seq_tjb_b = tjb_by_seq[seq];

  int stride = (M + 31) / 32;
  int my_start = lane * stride;
  int my_count = stride;
  if (my_start + my_count > M) my_count = M - my_start;
  if (my_count < 0) my_count = 0;

  /* Precompute q/z lookup in shared memory (only needed for MSV fallback).
   * Block-wide cooperative prefill so all warps share the table. */
  uint8_t *s_q = mem;
  uint8_t *s_z = mem + M;
  for (int k = threadIdx.x; k < M; k += 32 * WARPS) {
    s_q[k] = (uint8_t)(k % Q);
    s_z[k] = (uint8_t)(k / Q);
  }
  __syncthreads();

  __shared__ int use_full_msv[WARPS];
  int raw_result = 0;
  int overflow_result = 0;

  if (stride <= STRIDE &&
      (int) seq_tjb_b + (int) tbm_b + (int) tec_b + (int) bias_b < 127) {
    uint8_t xB_ssv = u8_sub_sat(base_b, (uint8_t)(seq_tjb_b + tbm_b));
    uint8_t xE_local = 0;

    uint8_t prev_reg[STRIDE];
    #pragma unroll
    for (int j = 0; j < STRIDE; j++) prev_reg[j] = 0;

    if (lane == 0) {
      use_full_msv[warp] = 0;
    }

    uint8_t last_prev = 0;

    for (int i = 1; i <= L; i++) {
      uint8_t x;
      if (packed_2bit) {
        int k = i - 1;
        int pos = rc_flag ? (L - 1 - k) : k;
        int sp  = (pos < s_src1_len) ? (s_offset + pos) : (s_src2_off + (pos - s_src1_len));
        x = (uint8_t)p7_nucdb_fetch1(sdsq, sp, rc_flag);
      } else {
        x = sdsq[i];
      }
      uint8_t from_left = __shfl_up_sync(0xffffffff, last_prev, 1);
      if (lane == 0) from_left = 0;

      const uint8_t *rsc_row = rbv_lin + (int)x * M;
      for (int j = 0; j < my_count; j++) {
        int km1 = my_start + j;
        uint8_t old_prev = prev_reg[j];
        uint8_t rsc = rsc_row[km1];
        uint8_t v = (from_left > xB_ssv) ? from_left : xB_ssv;
        v = u8_add_sat(v, bias_b);
        v = u8_sub_sat(v, rsc);
        prev_reg[j] = v;
        if (v > xE_local) xE_local = v;
        from_left = old_prev;
      }

      last_prev = (my_count > 0) ? prev_reg[my_count - 1] : 0;
    }

    /* Warp reduction for max score */
    for (int s = 16; s > 0; s >>= 1) {
      uint8_t other = __shfl_down_sync(0xffffffff, xE_local, s);
      if (other > xE_local) xE_local = other;
    }

    if (lane == 0) {
      int shifted_xE = (int) xE_local - (int) base_b + (int) seq_tjb_b + (int) tbm_b + 128;
      if (shifted_xE >= 255 - (int) bias_b) {
        if ((int) base_b - (int) seq_tjb_b - (int) tbm_b < 128) {
          use_full_msv[warp] = 1;
        } else {
          raw_result = 0;
          overflow_result = 1;
        }
      } else if ((int) xE_local >= 255 - (int) bias_b) {
        raw_result = 0;
        overflow_result = 1;
      } else {
        int xJ = (int) xE_local - (int) tec_b;
        if (xJ > (int) base_b) {
          use_full_msv[warp] = 1;
        } else {
          raw_result = xJ;
          overflow_result = 0;
        }
      }
    }
    __syncwarp();
    if (!use_full_msv[warp]) goto SSV_DONE;
  }

  /* Full MSV fallback (shared-memory path for ~0.3% of sequences or stride > STRIDE) */
  {
    /* Per-warp prev/curr buffers, placed after the shared s_q/s_z table. */
    uint8_t *warp_mem = mem + 2 * M + warp * 2 * (M + 1);
    uint8_t *prev = warp_mem;
    uint8_t *curr = prev + M + 1;

    for (int k = lane; k <= M; k += 32) prev[k] = 0;
    if (lane == 0) overflow_result = 0;
    __syncwarp();

    uint8_t xJ = 0;
    uint8_t tjbm_b = (uint8_t)(seq_tjb_b + tbm_b);
    uint8_t xB = u8_sub_sat(base_b, tjbm_b);

    for (int i = 1; i <= L; i++) {
      uint8_t xE_pos = 0;
      uint8_t x;
      if (packed_2bit) {
        int k2 = i - 1;
        int pos = rc_flag ? (L - 1 - k2) : k2;
        int sp  = (pos < s_src1_len) ? (s_offset + pos) : (s_src2_off + (pos - s_src1_len));
        x = (uint8_t)p7_nucdb_fetch1(sdsq, sp, rc_flag);
      } else {
        x = sdsq[i];
      }

      for (int k = lane + 1; k <= M; k += 32) {
        int km1 = k - 1;
        int q   = km1 % Q;
        int z   = km1 / Q;
        uint8_t rsc = rbv[((int) x * Q + q) * 16 + z];
        uint8_t v = prev[k-1] > xB ? prev[k-1] : xB;
        v = u8_add_sat(v, bias_b);
        v = u8_sub_sat(v, rsc);
        curr[k] = v;
        if (v > xE_pos) xE_pos = v;
        if (u8_add_sat(v, bias_b) == 255) overflow_result = 1;
      }
      if (lane == 0) curr[0] = 0;
      __syncwarp();

      for (int s = 16; s > 0; s >>= 1) {
        uint8_t other = __shfl_down_sync(0xffffffff, xE_pos, s);
        if (other > xE_pos) xE_pos = other;
      }
      uint8_t xE_after = u8_sub_sat(xE_pos, tec_b);   /* valid on lane 0; only used by lane 0 */

      if (lane == 0) {
        if (xE_after > xJ) xJ = xE_after;
        uint8_t maxBJ = (base_b > xJ) ? base_b : xJ;
        xB = u8_sub_sat(maxBJ, tjbm_b);
      }
      __syncwarp();

      uint8_t *tmp = prev;
      prev = curr;
      curr = tmp;
    }

    if (lane == 0) raw_result = (int) xJ;
  }

SSV_DONE:
  /* Store raw SSV results (needed for --gpu-ssv-compare and downstream stats) */
  if (lane == 0) {
    raw_sc[seq] = raw_result;
    overflow_out[seq] = overflow_result;
  }

  /* ---- Null + Bias + F1 Gate (lane 0 of each warp only) ---- */
  if (lane == 0) {
    /* Step 2: Null score */
    float nullsc;
    if (L <= 0) {
      nullsc = 0.0f;
    } else {
      float p1 = (float) L / (float) (L + 1);
      nullsc = (float) L * logf(p1) + logf(1.0f - p1);
    }
    nullsc_out[seq] = nullsc;

    /* Step 3: Bias filter (length-correct per sequence) */
    float filtersc;
    if (!do_biasfilter || L <= 0) {
      filtersc = nullsc;
    } else {
      float t00 = (float) L / (float) (L + 1);
      float t01 = 1.0f / (float) (L + 1);
      float t10 = bias_t[3];  /* fhmm->t[1][0] */
      float t11 = bias_t[4];  /* fhmm->t[1][1] */
      float t02 = bias_t[2];  /* fhmm->t[0][2] (end) */
      float t12 = bias_t[5];  /* fhmm->t[1][2] (end) */

      uint8_t x1;
      if (packed_2bit) {
        int pos = rc_flag ? (L - 1) : 0;
        int sp  = (pos < s_src1_len) ? (s_offset + pos) : (s_src2_off + (pos - s_src1_len));
        x1 = (uint8_t)p7_nucdb_fetch1(sdsq, sp, rc_flag);
      } else {
        x1 = sdsq[1];
      }
      float p0  = bias_eo[(int) x1 * 2 + 0] * bias_pi[0];
      float p1s = bias_eo[(int) x1 * 2 + 1] * bias_pi[1];
      float maxv = fmaxf(fmaxf(p0, p1s), 0.0f);
      p0  /= maxv;
      p1s /= maxv;
      float logsc = logf(maxv);

      for (int i = 2; i <= L; i++) {
        uint8_t x;
        if (packed_2bit) {
          int k2 = i - 1;
          int pos = rc_flag ? (L - 1 - k2) : k2;
          int sp  = (pos < s_src1_len) ? (s_offset + pos) : (s_src2_off + (pos - s_src1_len));
          x = (uint8_t)p7_nucdb_fetch1(sdsq, sp, rc_flag);
        } else {
          x = sdsq[i];
        }
        float n0  = (p0 * t00 + p1s * t10) * bias_eo[(int) x * 2 + 0];
        float n1  = (p0 * t01 + p1s * t11) * bias_eo[(int) x * 2 + 1];
        maxv = fmaxf(fmaxf(n0, n1), 0.0f);
        p0  = n0 / maxv;
        p1s = n1 / maxv;
        logsc += logf(maxv);
      }

      logsc += logf(p0 * t02 + p1s * t12);
      float len_p1 = (float) L / (float) (L + 1);
      filtersc = logsc + (float) L * logf(len_p1) + logf(1.0f - len_p1);
    }
    filtersc_out[seq] = filtersc;

    /* Step 4: F1 Gumbel P-value gate */
    float usc;
    if (overflow_result) {
      usc = 1e38f;
    } else {
      usc = ((float)(raw_result - (int) seq_tjb_b) - (float) base_b) / scale_b - 3.0f;
    }

    float gate_filtersc = nullsc;
    if (do_biasfilter && bias_gate_B > 0) {
      int F1_L = (L < bias_gate_B) ? L : bias_gate_B;
      float ratio = (F1_L > L) ? 1.0f : (float)F1_L / (float)L;
      gate_filtersc = nullsc + (filtersc - nullsc) * ratio;
    }

    float seq_score = (usc - gate_filtersc) * log2_inv;
    float y = ev_lambda * (seq_score - ev_mu);
    float ey = -expf(-y);
    float P = (fabsf(ey) < 1e-7f) ? -ey : 1.0f - expf(ey);

    if (overflow_result) P = 0.0f;
    if (P > F1) {
      if (survivor_status && survivor_idx == NULL && survivor_counter == NULL) survivor_status[seq] = 0;
      return;
    }

    if (do_biasfilter && bias_gate_B <= 0) {
      float bias_score = (usc - filtersc) * log2_inv;
      float by = ev_lambda * (bias_score - ev_mu);
      float bey = -expf(-by);
      float bP = (fabsf(bey) < 1e-7f) ? -bey : 1.0f - expf(bey);
      if (bP > F1) {
        if (survivor_status && survivor_idx == NULL && survivor_counter == NULL) survivor_status[seq] = 0;
        return;
      }
    }

    if (survivor_status && survivor_idx == NULL && survivor_counter == NULL) {
      survivor_status[seq] = 1;
      return;
    }

    int idx = atomicAdd(survivor_counter, 1);
    survivor_idx[idx] = seq;
    survivor_usc[idx] = usc;
    if (survivor_filtersc) survivor_filtersc[idx] = filtersc;
    if (survivor_status) survivor_status[idx] = overflow_result ? 4 : 0;  /* eslERANGE=4, eslOK=0 */
  }
}

__global__ static void
cuda_f1_compact_ordered_kernel(const int *mask, const float *filtersc, int nseq,
                               int *survivor_idx, float *survivor_filtersc,
                               int *survivor_status, int *survivor_count)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  int total = 0;
  for (int i = 0; i < nseq; i++) {
    if (mask[i]) {
      survivor_idx[total] = i;
      survivor_filtersc[total] = filtersc[i];
      if (survivor_status) survivor_status[total] = 0;
      total++;
    }
  }
  *survivor_count = total;
}

__global__ static void
cuda_nhmmer_gather_windows_kernel(const uint8_t *src_dsq,
                                  const int *src1_offsets, const int *src1_lengths,
                                  const int *src2_offsets, const int *dst_offsets,
                                  const int *lengths, int nseq,
                                  uint8_t *dst_dsq,
                                  int packed_2bit = 0, int rc_flag = 0)
{
  int seq = blockIdx.x;
  if (seq >= nseq) return;

  int dst0 = dst_offsets[seq];
  int len1 = src1_lengths[seq];
  int L    = lengths[seq];
  int src1 = src1_offsets[seq];
  int src2 = src2_offsets[seq];

  if (threadIdx.x == 0) dst_dsq[dst0] = eslDSQ_SENTINEL;
  for (int k = threadIdx.x; k < L; k += blockDim.x) {
    int sp = (k < len1) ? (src1 + k) : (src2 + (k - len1));
    uint8_t val;
    if (packed_2bit) {
      int pos = rc_flag ? (L - 1 - k) : k;
      int actual_sp = (pos < len1) ? (src1 + pos) : (src2 + (pos - len1));
      val = (uint8_t)p7_nucdb_fetch1(src_dsq, actual_sp, rc_flag);
    } else {
      val = src_dsq[sp];
    }
    dst_dsq[dst0 + 1 + k] = val;
  }
}

/* Dispatch macro for fused kernel template instantiation.
 * Block dim = 32*WARPS_VAL (one warp = one sequence).
 * Grid    = ceil(nseq / WARPS_VAL). */
#define SSV_FUSED_LAUNCH(STRIDE_VAL, WARPS_VAL) \
  cuda_ssv_null_bias_gate_kernel<STRIDE_VAL, WARPS_VAL><<<(nseq + (WARPS_VAL) - 1) / (WARPS_VAL), 32 * (WARPS_VAL), shmem>>>( \
    d_dsq_ptr, d_off_ptr, d_len_ptr, d_tjb_ptr, nseq, \
    cuom->d_rbv, cuom->d_rbv_lin, cuom->M, cuom->Q, cuom->Kp, \
    cuom->tbm_b, cuom->tec_b, cuom->tjb_b, \
    cuom->base_b, cuom->bias_b, cuom->scale_b, \
    engine->d_bias_pi, engine->d_bias_t, engine->d_bias_eo, \
    do_biasfilter, 0, \
    (float) ev_mu, (float) ev_lambda, (float) F1, log2_inv, \
    engine->d_raw, engine->d_overflow, \
    engine->d_null_scores, engine->d_bias_filtersc, \
    engine->d_f1_survivor_idx, engine->d_f1_counter, \
    engine->d_f1_survivor_usc, engine->d_f1_survivor_filtersc, survivor_statuses ? engine->d_f1_survivor_status : NULL)

#define SSV_NHMMER_FUSED_LAUNCH(STRIDE_VAL, WARPS_VAL) \
  cuda_ssv_null_bias_gate_kernel<STRIDE_VAL, WARPS_VAL><<<(nseq + (WARPS_VAL) - 1) / (WARPS_VAL), 32 * (WARPS_VAL), shmem>>>( \
    d_dsq_ptr, d_off_ptr, d_len_ptr, d_tjb_ptr, nseq, \
    cuom->d_rbv, (cuom->d_rbv_lin_nuc ? cuom->d_rbv_lin_nuc : cuom->d_rbv_lin), cuom->M, cuom->Q, (cuom->Kp_nuc ? cuom->Kp_nuc : cuom->Kp), \
    cuom->tbm_b, cuom->tec_b, cuom->tjb_b, \
    cuom->base_b, cuom->bias_b, cuom->scale_b, \
    engine->d_bias_pi, engine->d_bias_t, engine->d_bias_eo, \
    do_biasfilter, B1, \
    (float) ev_mu, (float) ev_lambda, (float) F1, log2_inv, \
    engine->d_raw, engine->d_overflow, \
    engine->d_null_scores, engine->d_bias_filtersc, \
    engine->d_f1_survivor_idx, engine->d_f1_counter, \
    engine->d_f1_survivor_usc, engine->d_f1_survivor_filtersc, engine->d_f1_survivor_status, \
    d_src1_len_ptr, d_src2_off_ptr, f1_rc_flag)

#define SSV_NHMMER_MASK_LAUNCH(STRIDE_VAL, WARPS_VAL) \
  cuda_ssv_null_bias_gate_kernel<STRIDE_VAL, WARPS_VAL><<<(nseq + (WARPS_VAL) - 1) / (WARPS_VAL), 32 * (WARPS_VAL), shmem>>>( \
    d_dsq_ptr, d_off_ptr, d_len_ptr, d_tjb_ptr, nseq, \
    cuom->d_rbv, (cuom->d_rbv_lin_nuc ? cuom->d_rbv_lin_nuc : cuom->d_rbv_lin), cuom->M, cuom->Q, (cuom->Kp_nuc ? cuom->Kp_nuc : cuom->Kp), \
    cuom->tbm_b, cuom->tec_b, cuom->tjb_b, \
    cuom->base_b, cuom->bias_b, cuom->scale_b, \
    engine->d_bias_pi, engine->d_bias_t, engine->d_bias_eo, \
    do_biasfilter, B1, \
    (float) ev_mu, (float) ev_lambda, (float) F1, log2_inv, \
    engine->d_raw, engine->d_overflow, \
    engine->d_null_scores, engine->d_bias_filtersc, \
    NULL, NULL, \
    NULL, NULL, engine->d_f1_pass_mask, \
    d_src1_len_ptr, d_src2_off_ptr, f1_rc_flag)

/* 2D switch: one branch per (STRIDE, WARPS) value pair. */
#define SSV_FUSED_DISPATCH_STRIDES(WARPS_VAL) \
  switch (stride) { \
    case  1: SSV_FUSED_LAUNCH( 1, WARPS_VAL); break; \
    case  2: SSV_FUSED_LAUNCH( 2, WARPS_VAL); break; \
    case  3: SSV_FUSED_LAUNCH( 3, WARPS_VAL); break; \
    case  4: SSV_FUSED_LAUNCH( 4, WARPS_VAL); break; \
    case  5: SSV_FUSED_LAUNCH( 5, WARPS_VAL); break; \
    case  6: SSV_FUSED_LAUNCH( 6, WARPS_VAL); break; \
    case  7: SSV_FUSED_LAUNCH( 7, WARPS_VAL); break; \
    case  8: SSV_FUSED_LAUNCH( 8, WARPS_VAL); break; \
    case  9: SSV_FUSED_LAUNCH( 9, WARPS_VAL); break; \
    case 10: SSV_FUSED_LAUNCH(10, WARPS_VAL); break; \
    case 11: SSV_FUSED_LAUNCH(11, WARPS_VAL); break; \
    case 12: SSV_FUSED_LAUNCH(12, WARPS_VAL); break; \
    case 13: SSV_FUSED_LAUNCH(13, WARPS_VAL); break; \
    case 14: SSV_FUSED_LAUNCH(14, WARPS_VAL); break; \
    case 15: SSV_FUSED_LAUNCH(15, WARPS_VAL); break; \
    case 16: SSV_FUSED_LAUNCH(16, WARPS_VAL); break; \
    case 17: SSV_FUSED_LAUNCH(17, WARPS_VAL); break; \
    case 18: SSV_FUSED_LAUNCH(18, WARPS_VAL); break; \
    case 19: SSV_FUSED_LAUNCH(19, WARPS_VAL); break; \
    case 20: SSV_FUSED_LAUNCH(20, WARPS_VAL); break; \
    default: SSV_FUSED_LAUNCH(20, WARPS_VAL); break; \
  }

#define SSV_NHMMER_FUSED_DISPATCH_STRIDES(WARPS_VAL) \
  switch (stride) { \
    case  1: SSV_NHMMER_FUSED_LAUNCH( 1, WARPS_VAL); break; \
    case  2: SSV_NHMMER_FUSED_LAUNCH( 2, WARPS_VAL); break; \
    case  3: SSV_NHMMER_FUSED_LAUNCH( 3, WARPS_VAL); break; \
    case  4: SSV_NHMMER_FUSED_LAUNCH( 4, WARPS_VAL); break; \
    case  5: SSV_NHMMER_FUSED_LAUNCH( 5, WARPS_VAL); break; \
    case  6: SSV_NHMMER_FUSED_LAUNCH( 6, WARPS_VAL); break; \
    case  7: SSV_NHMMER_FUSED_LAUNCH( 7, WARPS_VAL); break; \
    case  8: SSV_NHMMER_FUSED_LAUNCH( 8, WARPS_VAL); break; \
    case  9: SSV_NHMMER_FUSED_LAUNCH( 9, WARPS_VAL); break; \
    case 10: SSV_NHMMER_FUSED_LAUNCH(10, WARPS_VAL); break; \
    case 11: SSV_NHMMER_FUSED_LAUNCH(11, WARPS_VAL); break; \
    case 12: SSV_NHMMER_FUSED_LAUNCH(12, WARPS_VAL); break; \
    case 13: SSV_NHMMER_FUSED_LAUNCH(13, WARPS_VAL); break; \
    case 14: SSV_NHMMER_FUSED_LAUNCH(14, WARPS_VAL); break; \
    case 15: SSV_NHMMER_FUSED_LAUNCH(15, WARPS_VAL); break; \
    case 16: SSV_NHMMER_FUSED_LAUNCH(16, WARPS_VAL); break; \
    case 17: SSV_NHMMER_FUSED_LAUNCH(17, WARPS_VAL); break; \
    case 18: SSV_NHMMER_FUSED_LAUNCH(18, WARPS_VAL); break; \
    case 19: SSV_NHMMER_FUSED_LAUNCH(19, WARPS_VAL); break; \
    case 20: SSV_NHMMER_FUSED_LAUNCH(20, WARPS_VAL); break; \
    default: SSV_NHMMER_FUSED_LAUNCH(20, WARPS_VAL); break; \
  }

#define SSV_NHMMER_MASK_DISPATCH_STRIDES(WARPS_VAL) \
  switch (stride) { \
    case  1: SSV_NHMMER_MASK_LAUNCH( 1, WARPS_VAL); break; \
    case  2: SSV_NHMMER_MASK_LAUNCH( 2, WARPS_VAL); break; \
    case  3: SSV_NHMMER_MASK_LAUNCH( 3, WARPS_VAL); break; \
    case  4: SSV_NHMMER_MASK_LAUNCH( 4, WARPS_VAL); break; \
    case  5: SSV_NHMMER_MASK_LAUNCH( 5, WARPS_VAL); break; \
    case  6: SSV_NHMMER_MASK_LAUNCH( 6, WARPS_VAL); break; \
    case  7: SSV_NHMMER_MASK_LAUNCH( 7, WARPS_VAL); break; \
    case  8: SSV_NHMMER_MASK_LAUNCH( 8, WARPS_VAL); break; \
    case  9: SSV_NHMMER_MASK_LAUNCH( 9, WARPS_VAL); break; \
    case 10: SSV_NHMMER_MASK_LAUNCH(10, WARPS_VAL); break; \
    case 11: SSV_NHMMER_MASK_LAUNCH(11, WARPS_VAL); break; \
    case 12: SSV_NHMMER_MASK_LAUNCH(12, WARPS_VAL); break; \
    case 13: SSV_NHMMER_MASK_LAUNCH(13, WARPS_VAL); break; \
    case 14: SSV_NHMMER_MASK_LAUNCH(14, WARPS_VAL); break; \
    case 15: SSV_NHMMER_MASK_LAUNCH(15, WARPS_VAL); break; \
    case 16: SSV_NHMMER_MASK_LAUNCH(16, WARPS_VAL); break; \
    case 17: SSV_NHMMER_MASK_LAUNCH(17, WARPS_VAL); break; \
    case 18: SSV_NHMMER_MASK_LAUNCH(18, WARPS_VAL); break; \
    case 19: SSV_NHMMER_MASK_LAUNCH(19, WARPS_VAL); break; \
    case 20: SSV_NHMMER_MASK_LAUNCH(20, WARPS_VAL); break; \
    default: SSV_NHMMER_MASK_LAUNCH(20, WARPS_VAL); break; \
  }

extern "C" int
p7_cuda_SSVNullBiasGateResident(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                const P7_BG *bg, int64_t seq0, int nseq, int do_biasfilter,
                                double ev_mu, double ev_lambda, double F1,
                                int *survivor_idx, int *ret_nsurv,
                                float *nullsc, float *filtersc,
                                float *survivor_scores, int *survivor_statuses,
                                int warps_per_block,
                                char *errbuf, int errbuf_size)
{
  int        status = eslOK;
  uint8_t   *h_tjb_by_seq = NULL;
  int        WARPS = warps_per_block;
  /* Per-block shmem: 2*M for the block-shared s_q/s_z lookup +
   *                  WARPS*2*(M+1) for per-warp prev/curr in the rare full-MSV fallback. */
  size_t     shmem;
  cudaEvent_t h2d0 = engine->evt_h2d0, h2d1 = engine->evt_h2d1;
  cudaEvent_t k0 = engine->evt_k0, k1 = engine->evt_k1;
  cudaEvent_t d2h0 = engine->evt_d2h0, d2h1 = engine->evt_d2h1;
  double     ht0;
  int        h_counter = 0;
  float      log2_inv = 1.0f / 0.693147180559945f;

  if (!engine || !cuom || !bg || !survivor_idx || !ret_nsurv) return eslEINVAL;
  if (!engine->resident_active) return eslEINVAL;
  if (nseq <= 0) { *ret_nsurv = 0; return eslOK; }
  if (seq0 < 0 || seq0 + nseq > engine->resident_nseq) return eslEINVAL;
  if (WARPS != 1 && WARPS != 2 && WARPS != 3 && WARPS != 4 && WARPS != 6 && WARPS != 8) WARPS = 4;
  shmem = (size_t) (2 * cuom->M) + (size_t) WARPS * 2 * (cuom->M + 1);
  if (shmem > 96 * 1024) {
    /* Drop W until it fits, falling back to W=1 if even that's too big. */
    while (WARPS > 1 && shmem > 96 * 1024) {
      WARPS = (WARPS == 8 ? 6 : (WARPS == 6 ? 4 : (WARPS == 4 ? 3 : (WARPS == 3 ? 2 : 1))));
      shmem = (size_t) (2 * cuom->M) + (size_t) WARPS * 2 * (cuom->M + 1);
    }
    if (shmem > 96 * 1024) {
      if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "CUDA SSV fused profile M=%d exceeds shared-memory limit", cuom->M);
      return eslERANGE;
    }
  }

  ht0 = host_seconds();
  h_tjb_by_seq = (uint8_t *) malloc(sizeof(uint8_t) * nseq);
  if (!h_tjb_by_seq) { status = eslEMEM; goto ERROR; }
  engine->stats.host_malloc_free_seconds += host_seconds() - ht0;

  ht0 = host_seconds();
  {
    int *h_lengths_tmp = (int *) malloc(sizeof(int) * nseq);
    if (!h_lengths_tmp) { status = eslEMEM; goto ERROR; }
    if ((status = cuda_status(cudaMemcpy(h_lengths_tmp, engine->d_resident_lengths + seq0, sizeof(int) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(resident lengths->host)")) != eslOK) { free(h_lengths_tmp); goto ERROR; }
    for (int i = 0; i < nseq; i++) {
      if ((status = cuda_msvprofile_GrowLengthLookup((P7_CUDA_MSVPROFILE *) cuom, h_lengths_tmp[i])) != eslOK) { free(h_lengths_tmp); goto ERROR; }
      h_tjb_by_seq[i] = cuom->h_tjb_by_len[h_lengths_tmp[i]];
    }
    free(h_lengths_tmp);
  }
  engine->stats.host_metadata_loop_seconds += host_seconds() - ht0;

  /* Ensure metadata buffers allocated */
  if (engine->meta_alloc < nseq) {
    if (engine->d_tjb_by_seq) cudaFree(engine->d_tjb_by_seq);
    engine->d_tjb_by_seq = NULL;
    engine->meta_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_tjb_by_seq, sizeof(uint8_t) * nseq), errbuf, errbuf_size, "cudaMalloc(fused tjb_by_seq)")) != eslOK) goto ERROR;
    engine->meta_alloc = nseq;
  }
  if (engine->result_alloc < nseq) {
    if (engine->d_raw) cudaFree(engine->d_raw);
    if (engine->d_overflow) cudaFree(engine->d_overflow);
    engine->d_raw = NULL;
    engine->d_overflow = NULL;
    engine->result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_raw, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(fused raw)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_overflow, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(fused overflow)")) != eslOK) goto ERROR;
    engine->result_alloc = nseq;
  }
  if (engine->null_result_alloc < nseq) {
    if (engine->d_null_scores) cudaFree(engine->d_null_scores);
    engine->d_null_scores = NULL;
    engine->null_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_null_scores, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(fused null)")) != eslOK) goto ERROR;
    engine->null_result_alloc = nseq;
  }
  if (engine->bias_result_alloc < nseq) {
    if (engine->d_bias_filtersc) cudaFree(engine->d_bias_filtersc);
    engine->d_bias_filtersc = NULL;
    engine->bias_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_filtersc, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(fused bias)")) != eslOK) goto ERROR;
    engine->bias_result_alloc = nseq;
  }
  if (engine->f1_result_alloc < nseq ||
      engine->d_f1_survivor_idx == NULL || engine->d_f1_counter == NULL ||
      engine->d_f1_survivor_usc == NULL || engine->d_f1_survivor_filtersc == NULL ||
      engine->d_f1_survivor_status == NULL) {
    if (engine->d_f1_survivor_idx) cudaFree(engine->d_f1_survivor_idx);
    if (engine->d_f1_counter) cudaFree(engine->d_f1_counter);
    if (engine->d_f1_survivor_usc) cudaFree(engine->d_f1_survivor_usc);
    if (engine->d_f1_survivor_filtersc) cudaFree(engine->d_f1_survivor_filtersc);
    if (engine->d_f1_survivor_status) cudaFree(engine->d_f1_survivor_status);
    engine->d_f1_survivor_idx = NULL;
    engine->d_f1_counter = NULL;
    engine->d_f1_survivor_usc = NULL;
    engine->d_f1_survivor_filtersc = NULL;
    engine->d_f1_survivor_status = NULL;
    engine->f1_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_idx, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(fused f1 idx)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_counter, sizeof(int)), errbuf, errbuf_size, "cudaMalloc(fused f1 ctr)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_usc, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(fused f1 usc)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_filtersc, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(fused f1 filtersc)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_status, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(fused f1 status)")) != eslOK) goto ERROR;
    engine->f1_result_alloc = nseq;
  }

  /* Upload bias params if needed */
  if (!engine->bias_params_uploaded && do_biasfilter && bg->fhmm) {
    float h_pi[3], h_t[6];
    size_t eo_bytes = (size_t) bg->fhmm->abc->Kp * 2 * sizeof(float);
    h_pi[0] = bg->fhmm->pi[0]; h_pi[1] = bg->fhmm->pi[1]; h_pi[2] = bg->fhmm->pi[2];
    h_t[0] = bg->fhmm->t[0][0]; h_t[1] = bg->fhmm->t[0][1]; h_t[2] = bg->fhmm->t[0][2];
    h_t[3] = bg->fhmm->t[1][0]; h_t[4] = bg->fhmm->t[1][1]; h_t[5] = bg->fhmm->t[1][2];
    if (engine->d_bias_pi == NULL) {
      if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_pi, sizeof(float) * 3), errbuf, errbuf_size, "cudaMalloc(bias pi)")) != eslOK) goto ERROR;
      if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_t, sizeof(float) * 6), errbuf, errbuf_size, "cudaMalloc(bias t)")) != eslOK) goto ERROR;
    }
    if (engine->d_bias_eo == NULL) {
      if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_eo, eo_bytes), errbuf, errbuf_size, "cudaMalloc(bias eo)")) != eslOK) goto ERROR;
    }
    if ((status = cuda_status(cudaMemcpy(engine->d_bias_pi, h_pi, sizeof(h_pi), cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(bias pi)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_bias_t, h_t, sizeof(h_t), cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(bias t)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_bias_eo, bg->fhmm->eo[0], eo_bytes, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(bias eo)")) != eslOK) goto CUDA_ERROR;
    engine->bias_params_uploaded = 1;
  }

  /* Reset survivor counter */
  if ((status = cuda_status(cudaMemcpy(engine->d_f1_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(fused f1 counter reset)")) != eslOK) goto CUDA_ERROR;

  /* Upload tjb_by_seq */
  cudaEventRecord(h2d0);
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(engine->d_tjb_by_seq, h_tjb_by_seq, sizeof(uint8_t) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(fused tjb)")) != eslOK) goto CUDA_ERROR;
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(h2d1);

  /* Launch fused kernel with (STRIDE, WARPS) dispatch */
  {
    const uint8_t *d_dsq_ptr = engine->d_resident_dsq;
    const int     *d_off_ptr = engine->d_resident_offsets + seq0;
    const int     *d_len_ptr = engine->d_resident_lengths + seq0;
    const uint8_t *d_tjb_ptr = engine->d_tjb_by_seq;
    int stride = (cuom->M + 31) / 32;
    /* Some kernels with W*WARP*WARPS large dynamic shmem need explicit opt-in on Ada */
    if (shmem > 48 * 1024) {
      switch (WARPS) {
        case 1: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 2: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 3: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 4: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 6: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 8: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
      }
    }

    cudaEventRecord(k0);
    switch (WARPS) {
      case 1: SSV_FUSED_DISPATCH_STRIDES(1); break;
      case 2: SSV_FUSED_DISPATCH_STRIDES(2); break;
      case 3: SSV_FUSED_DISPATCH_STRIDES(3); break;
      case 4: SSV_FUSED_DISPATCH_STRIDES(4); break;
      case 6: SSV_FUSED_DISPATCH_STRIDES(6); break;
      case 8: SSV_FUSED_DISPATCH_STRIDES(8); break;
      default: SSV_FUSED_DISPATCH_STRIDES(4); break;
    }
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_ssv_null_bias_gate_kernel launch")) != eslOK) goto CUDA_ERROR;
    cudaEventRecord(k1);
  }
  ht0 = host_seconds();
  cudaEventSynchronize(k1);
  engine->stats.host_sync_seconds += host_seconds() - ht0;

  /* D2H: survivor list, counter, nullsc, filtersc, survivor scores */
  cudaEventRecord(d2h0);
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(&h_counter, engine->d_f1_counter, sizeof(int), cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fused f1 counter)")) != eslOK) goto CUDA_ERROR;
  if (h_counter > 0) {
    if ((status = cuda_status(cudaMemcpy(survivor_idx, engine->d_f1_survivor_idx, sizeof(int) * h_counter, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fused f1 survivors)")) != eslOK) goto CUDA_ERROR;
    if (survivor_scores)
      if ((status = cuda_status(cudaMemcpy(survivor_scores, engine->d_f1_survivor_usc, sizeof(float) * h_counter, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fused f1 usc)")) != eslOK) goto CUDA_ERROR;
    if (survivor_statuses)
      if ((status = cuda_status(cudaMemcpy(survivor_statuses, engine->d_f1_survivor_status, sizeof(int) * h_counter, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fused f1 status)")) != eslOK) goto CUDA_ERROR;
  }
  if ((status = cuda_status(cudaMemcpy(nullsc, engine->d_null_scores, sizeof(float) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fused nullsc)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(filtersc, engine->d_bias_filtersc, sizeof(float) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fused filtersc)")) != eslOK) goto CUDA_ERROR;
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(d2h1);

  *ret_nsurv = h_counter;
  engine->stats.h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.ssv_kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.d2h_seconds    += elapsed_seconds(d2h0, d2h1);
  engine->stats.nseqs          += nseq;
  engine->stats.nbatches       += 1;
  engine->last_cuom             = cuom;
  engine->batch_owner           = NULL;
  engine->batch_nseq            = nseq;
  engine->resident_batch_seq0   = seq0;
  engine->resident_batch_nseq   = nseq;

CUDA_ERROR:
ERROR:
  ht0 = host_seconds();
  free(h_tjb_by_seq);
  engine->stats.host_malloc_free_seconds += host_seconds() - ht0;
  return status;
}

extern "C" int
p7_cuda_SSVNullBiasGateDsqdataChunk(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                    const P7_BG *bg, ESL_DSQDATA_CHUNK *chu, int do_biasfilter,
                                    double ev_mu, double ev_lambda, double F1,
                                    int *survivor_idx, int *ret_nsurv,
                                    float *nullsc, float *filtersc,
                                    float *survivor_scores, int *survivor_statuses,
                                    int warps_per_block,
                                    char *errbuf, int errbuf_size)
{
  int        status = eslOK;
  int        nseq;
  int        total = 0;
  int       *h_offsets = NULL;
  int       *h_lengths = NULL;
  uint8_t   *h_tjb_by_seq = NULL;
  int        WARPS = warps_per_block;
  size_t     shmem;
  cudaEvent_t h2d0 = engine->evt_h2d0, h2d1 = engine->evt_h2d1;
  cudaEvent_t k0 = engine->evt_k0, k1 = engine->evt_k1;
  cudaEvent_t d2h0 = engine->evt_d2h0, d2h1 = engine->evt_d2h1;
  double     ht0;
  int        h_counter = 0;
  float      log2_inv = 1.0f / 0.693147180559945f;

  if (!engine || !cuom || !bg || !chu || !survivor_idx || !ret_nsurv) return eslEINVAL;
  nseq = chu->N;
  if (nseq <= 0) { *ret_nsurv = 0; return eslOK; }
  if (WARPS != 1 && WARPS != 2 && WARPS != 3 && WARPS != 4 && WARPS != 6 && WARPS != 8) WARPS = 4;
  shmem = (size_t) (2 * cuom->M) + (size_t) WARPS * 2 * (cuom->M + 1);
  if (shmem > 96 * 1024) {
    while (WARPS > 1 && shmem > 96 * 1024) {
      WARPS = (WARPS == 8 ? 6 : (WARPS == 6 ? 4 : (WARPS == 4 ? 3 : (WARPS == 3 ? 2 : 1))));
      shmem = (size_t) (2 * cuom->M) + (size_t) WARPS * 2 * (cuom->M + 1);
    }
    if (shmem > 96 * 1024) {
      if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "CUDA SSV fused profile M=%d exceeds shared-memory limit", cuom->M);
      return eslERANGE;
    }
  }

  if (engine->h_meta_alloc < nseq) {
    ht0 = host_seconds();
    int     *new_offsets = (int *)     realloc(engine->h_meta_offsets,    sizeof(int)     * nseq);
    int     *new_lengths = (int *)     realloc(engine->h_meta_lengths,    sizeof(int)     * nseq);
    uint8_t *new_tjb     = (uint8_t *) realloc(engine->h_meta_tjb_by_seq, sizeof(uint8_t) * nseq);
    if (!new_offsets || !new_lengths || !new_tjb) {
      if (new_offsets) engine->h_meta_offsets = new_offsets;
      if (new_lengths) engine->h_meta_lengths = new_lengths;
      if (new_tjb)     engine->h_meta_tjb_by_seq = new_tjb;
      status = eslEMEM; goto ERROR;
    }
    engine->h_meta_offsets    = new_offsets;
    engine->h_meta_lengths    = new_lengths;
    engine->h_meta_tjb_by_seq = new_tjb;
    engine->h_meta_alloc      = nseq;
    engine->stats.host_malloc_free_seconds += host_seconds() - ht0;
  }
  h_offsets    = engine->h_meta_offsets;
  h_lengths    = engine->h_meta_lengths;
  h_tjb_by_seq = engine->h_meta_tjb_by_seq;

  ht0 = host_seconds();
  for (int i = 0; i < nseq; i++) {
    h_offsets[i] = total;
    if (chu->L[i] > INT32_MAX) {
      if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "dsqdata sequence length exceeds CUDA fused SSV limit");
      status = eslERANGE; goto ERROR;
    }
    h_lengths[i] = (int) chu->L[i];
    if ((status = cuda_msvprofile_GrowLengthLookup((P7_CUDA_MSVPROFILE *) cuom, h_lengths[i])) != eslOK) goto ERROR;
    h_tjb_by_seq[i] = cuom->h_tjb_by_len[h_lengths[i]];
    total += h_lengths[i] + 1;
  }
  total += 1;
  engine->stats.host_metadata_loop_seconds += host_seconds() - ht0;

  /* Allocate device buffers */
  if (engine->dsq_alloc < total) {
    if (engine->d_dsq) cudaFree(engine->d_dsq);
    if (engine->h_dsq) cudaFreeHost(engine->h_dsq);
    engine->d_dsq = NULL; engine->h_dsq = NULL;
    engine->dsq_alloc = 0; engine->h_dsq_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_dsq, total), errbuf, errbuf_size, "cudaMalloc(fused dsq)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMallocHost((void **) &engine->h_dsq, total), errbuf, errbuf_size, "cudaMallocHost(fused dsq)")) != eslOK) goto ERROR;
    engine->dsq_alloc = total; engine->h_dsq_alloc = total;
  }
  if (engine->meta_alloc < nseq) {
    if (engine->d_offsets) cudaFree(engine->d_offsets);
    if (engine->d_lengths) cudaFree(engine->d_lengths);
    if (engine->d_tjb_by_seq) cudaFree(engine->d_tjb_by_seq);
    engine->d_offsets = NULL; engine->d_lengths = NULL; engine->d_tjb_by_seq = NULL;
    engine->meta_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(fused offsets)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_lengths, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(fused lengths)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_tjb_by_seq, sizeof(uint8_t) * nseq), errbuf, errbuf_size, "cudaMalloc(fused tjb)")) != eslOK) goto ERROR;
    engine->meta_alloc = nseq;
  }
  if (engine->result_alloc < nseq) {
    if (engine->d_raw) cudaFree(engine->d_raw);
    if (engine->d_overflow) cudaFree(engine->d_overflow);
    engine->d_raw = NULL; engine->d_overflow = NULL;
    engine->result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_raw, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(fused raw)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_overflow, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(fused overflow)")) != eslOK) goto ERROR;
    engine->result_alloc = nseq;
  }
  if (engine->null_result_alloc < nseq) {
    if (engine->d_null_scores) cudaFree(engine->d_null_scores);
    engine->d_null_scores = NULL; engine->null_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_null_scores, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(fused null)")) != eslOK) goto ERROR;
    engine->null_result_alloc = nseq;
  }
  if (engine->bias_result_alloc < nseq) {
    if (engine->d_bias_filtersc) cudaFree(engine->d_bias_filtersc);
    engine->d_bias_filtersc = NULL; engine->bias_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_filtersc, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(fused bias)")) != eslOK) goto ERROR;
    engine->bias_result_alloc = nseq;
  }
  if (engine->f1_result_alloc < nseq ||
      engine->d_f1_survivor_idx == NULL || engine->d_f1_counter == NULL ||
      engine->d_f1_survivor_usc == NULL || engine->d_f1_survivor_filtersc == NULL ||
      engine->d_f1_survivor_status == NULL) {
    if (engine->d_f1_survivor_idx) cudaFree(engine->d_f1_survivor_idx);
    if (engine->d_f1_counter) cudaFree(engine->d_f1_counter);
    if (engine->d_f1_survivor_usc) cudaFree(engine->d_f1_survivor_usc);
    if (engine->d_f1_survivor_filtersc) cudaFree(engine->d_f1_survivor_filtersc);
    if (engine->d_f1_survivor_status) cudaFree(engine->d_f1_survivor_status);
    engine->d_f1_survivor_idx = NULL;
    engine->d_f1_counter = NULL;
    engine->d_f1_survivor_usc = NULL;
    engine->d_f1_survivor_filtersc = NULL;
    engine->d_f1_survivor_status = NULL;
    engine->f1_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_idx, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(fused f1 idx)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_counter, sizeof(int)), errbuf, errbuf_size, "cudaMalloc(fused f1 ctr)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_usc, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(fused f1 usc)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_filtersc, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(fused f1 filtersc)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_status, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(fused f1 status)")) != eslOK) goto ERROR;
    engine->f1_result_alloc = nseq;
  }

  /* Upload bias params if needed */
  if (!engine->bias_params_uploaded && do_biasfilter && bg->fhmm) {
    float h_pi[3], h_t[6];
    size_t eo_bytes = (size_t) bg->fhmm->abc->Kp * 2 * sizeof(float);
    h_pi[0] = bg->fhmm->pi[0]; h_pi[1] = bg->fhmm->pi[1]; h_pi[2] = bg->fhmm->pi[2];
    h_t[0] = bg->fhmm->t[0][0]; h_t[1] = bg->fhmm->t[0][1]; h_t[2] = bg->fhmm->t[0][2];
    h_t[3] = bg->fhmm->t[1][0]; h_t[4] = bg->fhmm->t[1][1]; h_t[5] = bg->fhmm->t[1][2];
    if (engine->d_bias_pi == NULL) {
      if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_pi, sizeof(float) * 3), errbuf, errbuf_size, "cudaMalloc(bias pi)")) != eslOK) goto ERROR;
      if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_t, sizeof(float) * 6), errbuf, errbuf_size, "cudaMalloc(bias t)")) != eslOK) goto ERROR;
    }
    if (engine->d_bias_eo == NULL) {
      if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_eo, eo_bytes), errbuf, errbuf_size, "cudaMalloc(bias eo)")) != eslOK) goto ERROR;
    }
    if ((status = cuda_status(cudaMemcpy(engine->d_bias_pi, h_pi, sizeof(h_pi), cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(bias pi)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_bias_t, h_t, sizeof(h_t), cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(bias t)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_bias_eo, bg->fhmm->eo[0], eo_bytes, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(bias eo)")) != eslOK) goto CUDA_ERROR;
    engine->bias_params_uploaded = 1;
  }

  /* Reset survivor counter */
  if ((status = cuda_status(cudaMemcpy(engine->d_f1_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(fused f1 counter reset)")) != eslOK) goto CUDA_ERROR;

  /* Pack and upload DSQ */
  ht0 = host_seconds();
  if (chu->smem != NULL) {
    memcpy(engine->h_dsq, chu->smem, total);
  } else {
    for (int i = 0; i < nseq; i++)
      memcpy(engine->h_dsq + h_offsets[i], chu->dsq[i], h_lengths[i] + 1);
  }
  engine->stats.host_pack_loop_seconds += host_seconds() - ht0;

  cudaEventRecord(h2d0);
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(engine->d_dsq, engine->h_dsq, total, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(fused dsq)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_offsets, h_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(fused offsets)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_lengths, h_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(fused lengths)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_tjb_by_seq, h_tjb_by_seq, sizeof(uint8_t) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(fused tjb)")) != eslOK) goto CUDA_ERROR;
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(h2d1);

  /* Launch fused kernel with (STRIDE, WARPS) dispatch */
  {
    const uint8_t *d_dsq_ptr = engine->d_dsq;
    const int     *d_off_ptr = engine->d_offsets;
    const int     *d_len_ptr = engine->d_lengths;
    const uint8_t *d_tjb_ptr = engine->d_tjb_by_seq;
    int stride = (cuom->M + 31) / 32;
    if (shmem > 48 * 1024) {
      switch (WARPS) {
        case 1: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 2: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 3: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 4: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 6: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 8: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
      }
    }

    cudaEventRecord(k0);
    switch (WARPS) {
      case 1: SSV_FUSED_DISPATCH_STRIDES(1); break;
      case 2: SSV_FUSED_DISPATCH_STRIDES(2); break;
      case 3: SSV_FUSED_DISPATCH_STRIDES(3); break;
      case 4: SSV_FUSED_DISPATCH_STRIDES(4); break;
      case 6: SSV_FUSED_DISPATCH_STRIDES(6); break;
      case 8: SSV_FUSED_DISPATCH_STRIDES(8); break;
      default: SSV_FUSED_DISPATCH_STRIDES(4); break;
    }
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_ssv_null_bias_gate_kernel launch")) != eslOK) goto CUDA_ERROR;
    cudaEventRecord(k1);
  }
  ht0 = host_seconds();
  cudaEventSynchronize(k1);
  engine->stats.host_sync_seconds += host_seconds() - ht0;

  /* D2H: survivor list, counter, nullsc, filtersc, survivor scores */
  cudaEventRecord(d2h0);
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(&h_counter, engine->d_f1_counter, sizeof(int), cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fused f1 counter)")) != eslOK) goto CUDA_ERROR;
  if (h_counter > 0) {
    if ((status = cuda_status(cudaMemcpy(survivor_idx, engine->d_f1_survivor_idx, sizeof(int) * h_counter, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fused f1 survivors)")) != eslOK) goto CUDA_ERROR;
    if (survivor_scores)
      if ((status = cuda_status(cudaMemcpy(survivor_scores, engine->d_f1_survivor_usc, sizeof(float) * h_counter, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fused f1 usc)")) != eslOK) goto CUDA_ERROR;
    if (survivor_statuses)
      if ((status = cuda_status(cudaMemcpy(survivor_statuses, engine->d_f1_survivor_status, sizeof(int) * h_counter, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fused f1 status)")) != eslOK) goto CUDA_ERROR;
  }
  if ((status = cuda_status(cudaMemcpy(nullsc, engine->d_null_scores, sizeof(float) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fused nullsc)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(filtersc, engine->d_bias_filtersc, sizeof(float) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fused filtersc)")) != eslOK) goto CUDA_ERROR;
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(d2h1);

  *ret_nsurv = h_counter;
  engine->stats.h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.ssv_kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.d2h_seconds    += elapsed_seconds(d2h0, d2h1);
  engine->stats.nseqs          += nseq;
  engine->stats.nres           += total - (2 * nseq);
  engine->stats.nbatches       += 1;
  engine->batch_owner           = chu;
  engine->batch_nseq            = nseq;
  engine->batch_total           = total;
  engine->last_cuom             = cuom;

CUDA_ERROR:
ERROR:
  return status;
}

extern "C" int
p7_cuda_NhmmerF1GateDsqdataChunk(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                 const P7_BG *bg, ESL_DSQDATA_CHUNK *chu, int do_biasfilter,
                                 int B1, double ev_mu, double ev_lambda, double F1,
                                 int *survivor_idx, int *ret_nsurv,
                                 float *survivor_filtersc, int *survivor_statuses,
                                 int warps_per_block,
                                 char *errbuf, int errbuf_size)
{
  int        status = eslOK;
  int        nseq;
  int        total = 0;
  int       *h_offsets = NULL;
  int       *h_lengths = NULL;
  uint8_t   *h_tjb_by_seq = NULL;
  int        WARPS = warps_per_block;
  size_t     shmem;
  cudaEvent_t h2d0 = engine->evt_h2d0, h2d1 = engine->evt_h2d1;
  cudaEvent_t k0 = engine->evt_k0, k1 = engine->evt_k1;
  cudaEvent_t d2h0 = engine->evt_d2h0, d2h1 = engine->evt_d2h1;
  double     ht0;
  double     gate_seconds = 0.0;
  int        h_counter = 0;
  float      log2_inv = 1.0f / 0.693147180559945f;

  if (!engine || !cuom || !bg || !chu || !survivor_idx || !ret_nsurv || !survivor_filtersc) return eslEINVAL;
  nseq = chu->N;
  if (nseq <= 0) { *ret_nsurv = 0; return eslOK; }
  if (WARPS != 1 && WARPS != 2 && WARPS != 3 && WARPS != 4 && WARPS != 6 && WARPS != 8) WARPS = 4;
  shmem = (size_t) (2 * cuom->M) + (size_t) WARPS * 2 * (cuom->M + 1);
  if (shmem > 96 * 1024) {
    while (WARPS > 1 && shmem > 96 * 1024) {
      WARPS = (WARPS == 8 ? 6 : (WARPS == 6 ? 4 : (WARPS == 4 ? 3 : (WARPS == 3 ? 2 : 1))));
      shmem = (size_t) (2 * cuom->M) + (size_t) WARPS * 2 * (cuom->M + 1);
    }
    if (shmem > 96 * 1024) {
      if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "CUDA nhmmer F1 profile M=%d exceeds shared-memory limit", cuom->M);
      return eslERANGE;
    }
  }

  if (engine->h_meta_alloc < nseq) {
    ht0 = host_seconds();
    int     *new_offsets = (int *)     realloc(engine->h_meta_offsets,    sizeof(int)     * nseq);
    int     *new_lengths = (int *)     realloc(engine->h_meta_lengths,    sizeof(int)     * nseq);
    uint8_t *new_tjb     = (uint8_t *) realloc(engine->h_meta_tjb_by_seq, sizeof(uint8_t) * nseq);
    if (!new_offsets || !new_lengths || !new_tjb) {
      if (new_offsets) engine->h_meta_offsets = new_offsets;
      if (new_lengths) engine->h_meta_lengths = new_lengths;
      if (new_tjb)     engine->h_meta_tjb_by_seq = new_tjb;
      status = eslEMEM; goto ERROR;
    }
    engine->h_meta_offsets    = new_offsets;
    engine->h_meta_lengths    = new_lengths;
    engine->h_meta_tjb_by_seq = new_tjb;
    engine->h_meta_alloc      = nseq;
    engine->stats.host_malloc_free_seconds += host_seconds() - ht0;
  }
  h_offsets    = engine->h_meta_offsets;
  h_lengths    = engine->h_meta_lengths;
  h_tjb_by_seq = engine->h_meta_tjb_by_seq;

  ht0 = host_seconds();
  for (int i = 0; i < nseq; i++) {
    h_offsets[i] = total;
    if (chu->L[i] > INT32_MAX) {
      if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "dsqdata sequence length exceeds CUDA nhmmer F1 limit");
      status = eslERANGE; goto ERROR;
    }
    h_lengths[i] = (int) chu->L[i];
    if ((status = cuda_msvprofile_GrowLengthLookup((P7_CUDA_MSVPROFILE *) cuom, h_lengths[i])) != eslOK) goto ERROR;
    h_tjb_by_seq[i] = cuom->h_tjb_by_len[h_lengths[i]];
    total += h_lengths[i] + 1;
  }
  total += 1;
  engine->stats.host_metadata_loop_seconds += host_seconds() - ht0;

  if (engine->dsq_alloc < total) {
    if (engine->d_dsq) cudaFree(engine->d_dsq);
    if (engine->h_dsq) cudaFreeHost(engine->h_dsq);
    engine->d_dsq = NULL; engine->h_dsq = NULL;
    engine->dsq_alloc = 0; engine->h_dsq_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_dsq, total), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 dsq)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMallocHost((void **) &engine->h_dsq, total), errbuf, errbuf_size, "cudaMallocHost(nhmmer f1 dsq)")) != eslOK) goto ERROR;
    engine->dsq_alloc = total; engine->h_dsq_alloc = total;
  }
  if (engine->meta_alloc < nseq) {
    if (engine->d_offsets) cudaFree(engine->d_offsets);
    if (engine->d_lengths) cudaFree(engine->d_lengths);
    if (engine->d_tjb_by_seq) cudaFree(engine->d_tjb_by_seq);
    engine->d_offsets = NULL; engine->d_lengths = NULL; engine->d_tjb_by_seq = NULL;
    engine->meta_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 offsets)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_lengths, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 lengths)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_tjb_by_seq, sizeof(uint8_t) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 tjb)")) != eslOK) goto ERROR;
    engine->meta_alloc = nseq;
  }
  if (engine->result_alloc < nseq) {
    if (engine->d_raw) cudaFree(engine->d_raw);
    if (engine->d_overflow) cudaFree(engine->d_overflow);
    engine->d_raw = NULL; engine->d_overflow = NULL;
    engine->result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_raw, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 raw)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_overflow, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 overflow)")) != eslOK) goto ERROR;
    engine->result_alloc = nseq;
  }
  if (engine->null_result_alloc < nseq) {
    if (engine->d_null_scores) cudaFree(engine->d_null_scores);
    engine->d_null_scores = NULL; engine->null_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_null_scores, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 null)")) != eslOK) goto ERROR;
    engine->null_result_alloc = nseq;
  }
  if (engine->bias_result_alloc < nseq) {
    if (engine->d_bias_filtersc) cudaFree(engine->d_bias_filtersc);
    engine->d_bias_filtersc = NULL; engine->bias_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_filtersc, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 bias)")) != eslOK) goto ERROR;
    engine->bias_result_alloc = nseq;
  }
  if (engine->f1_result_alloc < nseq ||
      engine->d_f1_survivor_idx == NULL || engine->d_f1_counter == NULL ||
      engine->d_f1_survivor_usc == NULL || engine->d_f1_survivor_filtersc == NULL ||
      engine->d_f1_survivor_status == NULL) {
    if (engine->d_f1_survivor_idx) cudaFree(engine->d_f1_survivor_idx);
    if (engine->d_f1_counter) cudaFree(engine->d_f1_counter);
    if (engine->d_f1_survivor_usc) cudaFree(engine->d_f1_survivor_usc);
    if (engine->d_f1_survivor_filtersc) cudaFree(engine->d_f1_survivor_filtersc);
    if (engine->d_f1_survivor_status) cudaFree(engine->d_f1_survivor_status);
    engine->d_f1_survivor_idx = NULL;
    engine->d_f1_counter = NULL;
    engine->d_f1_survivor_usc = NULL;
    engine->d_f1_survivor_filtersc = NULL;
    engine->d_f1_survivor_status = NULL;
    engine->f1_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_idx, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 idx)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_counter, sizeof(int)), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 ctr)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_usc, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 usc)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_filtersc, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 filtersc)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_status, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 status)")) != eslOK) goto ERROR;
    engine->f1_result_alloc = nseq;
  }
  if (engine->f1_order_alloc < nseq || engine->d_f1_pass_mask == NULL) {
    if (engine->d_f1_pass_mask) cudaFree(engine->d_f1_pass_mask);
    engine->d_f1_pass_mask = NULL;
    engine->f1_order_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_pass_mask, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 pass mask)")) != eslOK) goto ERROR;
    engine->f1_order_alloc = nseq;
  }
  if (engine->h_f1_order_alloc < nseq) {
    free(engine->h_f1_pass_mask);
    free(engine->h_f1_filtersc);
    engine->h_f1_pass_mask = (int *)   malloc(sizeof(int)   * nseq);
    engine->h_f1_filtersc  = (float *) malloc(sizeof(float) * nseq);
    if (!engine->h_f1_pass_mask || !engine->h_f1_filtersc) { status = eslEMEM; goto ERROR; }
    engine->h_f1_order_alloc = nseq;
  }

  if (!engine->bias_params_uploaded && do_biasfilter && bg->fhmm) {
    float h_pi[3], h_t[6];
    size_t eo_bytes = (size_t) bg->fhmm->abc->Kp * 2 * sizeof(float);
    h_pi[0] = bg->fhmm->pi[0]; h_pi[1] = bg->fhmm->pi[1]; h_pi[2] = bg->fhmm->pi[2];
    h_t[0] = bg->fhmm->t[0][0]; h_t[1] = bg->fhmm->t[0][1]; h_t[2] = bg->fhmm->t[0][2];
    h_t[3] = bg->fhmm->t[1][0]; h_t[4] = bg->fhmm->t[1][1]; h_t[5] = bg->fhmm->t[1][2];
    if (engine->d_bias_pi == NULL) {
      if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_pi, sizeof(float) * 3), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 bias pi)")) != eslOK) goto ERROR;
      if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_t, sizeof(float) * 6), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 bias t)")) != eslOK) goto ERROR;
    }
    if (engine->d_bias_eo == NULL) {
      if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_eo, eo_bytes), errbuf, errbuf_size, "cudaMalloc(nhmmer f1 bias eo)")) != eslOK) goto ERROR;
    }
    if ((status = cuda_status(cudaMemcpy(engine->d_bias_pi, h_pi, sizeof(h_pi), cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer f1 bias pi)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_bias_t, h_t, sizeof(h_t), cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer f1 bias t)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_bias_eo, bg->fhmm->eo[0], eo_bytes, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer f1 bias eo)")) != eslOK) goto CUDA_ERROR;
    engine->bias_params_uploaded = 1;
  }

  if ((status = cuda_status(cudaMemset(engine->d_f1_pass_mask, 0, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMemset(nhmmer f1 pass mask)")) != eslOK) goto CUDA_ERROR;

  ht0 = host_seconds();
  if (chu->smem != NULL) {
    memcpy(engine->h_dsq, chu->smem, total);
  } else {
    for (int i = 0; i < nseq; i++)
      memcpy(engine->h_dsq + h_offsets[i], chu->dsq[i], h_lengths[i] + 1);
  }
  engine->stats.host_pack_loop_seconds += host_seconds() - ht0;

  cudaEventRecord(h2d0);
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(engine->d_dsq, engine->h_dsq, total, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer f1 dsq)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_offsets, h_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer f1 offsets)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_lengths, h_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer f1 lengths)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_tjb_by_seq, h_tjb_by_seq, sizeof(uint8_t) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer f1 tjb)")) != eslOK) goto CUDA_ERROR;
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(h2d1);

  {
    const uint8_t *d_dsq_ptr = engine->d_dsq;
    const int     *d_off_ptr = engine->d_offsets;
    const int     *d_len_ptr = engine->d_lengths;
    const uint8_t *d_tjb_ptr = engine->d_tjb_by_seq;
    const int     *d_src1_len_ptr = NULL;
    const int     *d_src2_off_ptr = NULL;
    int            f1_rc_flag     = 0;
    int stride = (cuom->M + 31) / 32;
    if (shmem > 48 * 1024) {
      switch (WARPS) {
        case 1: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 2: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 3: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 4: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 6: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 8: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
      }
    }

    cudaEventRecord(k0);
    switch (WARPS) {
      case 1: SSV_NHMMER_MASK_DISPATCH_STRIDES(1); break;
      case 2: SSV_NHMMER_MASK_DISPATCH_STRIDES(2); break;
      case 3: SSV_NHMMER_MASK_DISPATCH_STRIDES(3); break;
      case 4: SSV_NHMMER_MASK_DISPATCH_STRIDES(4); break;
      case 6: SSV_NHMMER_MASK_DISPATCH_STRIDES(6); break;
      case 8: SSV_NHMMER_MASK_DISPATCH_STRIDES(8); break;
      default: SSV_NHMMER_MASK_DISPATCH_STRIDES(4); break;
    }
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_ssv_nhmmer_f1_gate launch")) != eslOK) goto CUDA_ERROR;
    cudaEventRecord(k1);
  }
  ht0 = host_seconds();
  cudaEventSynchronize(k1);
  engine->stats.host_sync_seconds += host_seconds() - ht0;
  gate_seconds = elapsed_seconds(k0, k1);
  engine->stats.f1_gate_kernel_seconds += gate_seconds;

  cudaEventRecord(d2h0);
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(engine->h_f1_pass_mask, engine->d_f1_pass_mask, sizeof(int) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(nhmmer f1 pass mask D2H)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->h_f1_filtersc, engine->d_bias_filtersc, sizeof(float) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(nhmmer f1 filtersc D2H)")) != eslOK) goto CUDA_ERROR;
  {
    int total_surv = 0;
    for (int i = 0; i < nseq; i++) {
      if (engine->h_f1_pass_mask[i]) {
        survivor_idx[total_surv]      = i;
        survivor_filtersc[total_surv] = engine->h_f1_filtersc[i];
        if (survivor_statuses) survivor_statuses[total_surv] = 0;
        total_surv++;
      }
    }
    h_counter = total_surv;
  }
  if (h_counter > 0) {
    if ((status = cuda_status(cudaMemcpy(engine->d_f1_survivor_idx, survivor_idx, sizeof(int) * h_counter, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer f1 surv idx H2D)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_f1_survivor_filtersc, survivor_filtersc, sizeof(float) * h_counter, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer f1 surv filtersc H2D)")) != eslOK) goto CUDA_ERROR;
  }
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(d2h1);

  *ret_nsurv = h_counter;
  engine->stats.h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.kernel_seconds += gate_seconds;
  engine->stats.ssv_kernel_seconds += gate_seconds;
  engine->stats.d2h_seconds    += elapsed_seconds(d2h0, d2h1);
  engine->stats.nseqs          += nseq;
  engine->stats.nres           += total - (2 * nseq);
  engine->stats.nbatches       += 1;
  engine->batch_owner           = chu;
  engine->batch_nseq            = nseq;
  engine->batch_total           = total;
  engine->d_batch_dsq           = engine->d_dsq;
  engine->last_cuom             = cuom;

CUDA_ERROR:
ERROR:
  return status;
}

extern "C" int
p7_cuda_NhmmerF1GateResident(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                             const P7_BG *bg, const uint8_t *d_dsq_base,
                             const int *h_offsets_in, const int *h_lengths_in, int nseq,
                             int do_biasfilter, int B1,
                             double ev_mu, double ev_lambda, double F1,
                             int *survivor_idx, int *ret_nsurv,
                             float *survivor_filtersc, int *survivor_statuses,
                             int warps_per_block,
                             char *errbuf, int errbuf_size,
                             const int *h_src1_lengths_in, const int *h_src2_offsets_in,
                             int packed_rc_flag)
{
  int        status = eslOK;
  int       *h_offsets = NULL;
  int       *h_lengths = NULL;
  uint8_t   *h_tjb_by_seq = NULL;
  int        WARPS = warps_per_block;
  size_t     shmem;
  cudaEvent_t h2d0 = engine->evt_h2d0, h2d1 = engine->evt_h2d1;
  cudaEvent_t k0 = engine->evt_k0, k1 = engine->evt_k1;
  cudaEvent_t d2h0 = engine->evt_d2h0, d2h1 = engine->evt_d2h1;
  double     ht0;
  double     gate_seconds = 0.0;
  int        h_counter = 0;
  int64_t    total_res = 0;
  float      log2_inv = 1.0f / 0.693147180559945f;

  if (!engine || !cuom || !bg || !d_dsq_base || !h_offsets_in || !h_lengths_in ||
      !survivor_idx || !ret_nsurv || !survivor_filtersc) return eslEINVAL;
  if (nseq <= 0) { *ret_nsurv = 0; return eslOK; }
  if (WARPS != 1 && WARPS != 2 && WARPS != 3 && WARPS != 4 && WARPS != 6 && WARPS != 8) WARPS = 4;
  shmem = (size_t) (2 * cuom->M) + (size_t) WARPS * 2 * (cuom->M + 1);
  if (shmem > 96 * 1024) {
    while (WARPS > 1 && shmem > 96 * 1024) {
      WARPS = (WARPS == 8 ? 6 : (WARPS == 6 ? 4 : (WARPS == 4 ? 3 : (WARPS == 3 ? 2 : 1))));
      shmem = (size_t) (2 * cuom->M) + (size_t) WARPS * 2 * (cuom->M + 1);
    }
    if (shmem > 96 * 1024) {
      if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "CUDA nhmmer resident F1 profile M=%d exceeds shared-memory limit", cuom->M);
      return eslERANGE;
    }
  }

  if (engine->h_meta_alloc < nseq) {
    ht0 = host_seconds();
    int     *new_offsets = (int *)     realloc(engine->h_meta_offsets,    sizeof(int)     * nseq);
    int     *new_lengths = (int *)     realloc(engine->h_meta_lengths,    sizeof(int)     * nseq);
    uint8_t *new_tjb     = (uint8_t *) realloc(engine->h_meta_tjb_by_seq, sizeof(uint8_t) * nseq);
    if (!new_offsets || !new_lengths || !new_tjb) {
      if (new_offsets) engine->h_meta_offsets = new_offsets;
      if (new_lengths) engine->h_meta_lengths = new_lengths;
      if (new_tjb)     engine->h_meta_tjb_by_seq = new_tjb;
      status = eslEMEM; goto ERROR;
    }
    engine->h_meta_offsets    = new_offsets;
    engine->h_meta_lengths    = new_lengths;
    engine->h_meta_tjb_by_seq = new_tjb;
    engine->h_meta_alloc      = nseq;
    engine->stats.host_malloc_free_seconds += host_seconds() - ht0;
  }
  h_offsets    = engine->h_meta_offsets;
  h_lengths    = engine->h_meta_lengths;
  h_tjb_by_seq = engine->h_meta_tjb_by_seq;

  ht0 = host_seconds();
  for (int i = 0; i < nseq; i++) {
    if (h_offsets_in[i] < 0 || h_lengths_in[i] < 0) {
      status = eslERANGE; goto ERROR;
    }
    h_offsets[i] = h_offsets_in[i];
    h_lengths[i] = h_lengths_in[i];
    if ((status = cuda_msvprofile_GrowLengthLookup((P7_CUDA_MSVPROFILE *) cuom, h_lengths[i])) != eslOK) goto ERROR;
    h_tjb_by_seq[i] = cuom->h_tjb_by_len[h_lengths[i]];
    total_res += h_lengths[i];
  }
  engine->stats.host_metadata_loop_seconds += host_seconds() - ht0;

  if (engine->meta_alloc < nseq) {
    if (engine->d_offsets) cudaFree(engine->d_offsets);
    if (engine->d_lengths) cudaFree(engine->d_lengths);
    if (engine->d_tjb_by_seq) cudaFree(engine->d_tjb_by_seq);
    engine->d_offsets = NULL; engine->d_lengths = NULL; engine->d_tjb_by_seq = NULL;
    engine->meta_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 offsets)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_lengths, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 lengths)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_tjb_by_seq, sizeof(uint8_t) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 tjb)")) != eslOK) goto ERROR;
    engine->meta_alloc = nseq;
  }
  if (engine->result_alloc < nseq) {
    if (engine->d_raw) cudaFree(engine->d_raw);
    if (engine->d_overflow) cudaFree(engine->d_overflow);
    engine->d_raw = NULL; engine->d_overflow = NULL;
    engine->result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_raw, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 raw)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_overflow, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 overflow)")) != eslOK) goto ERROR;
    engine->result_alloc = nseq;
  }
  if (engine->null_result_alloc < nseq) {
    if (engine->d_null_scores) cudaFree(engine->d_null_scores);
    engine->d_null_scores = NULL; engine->null_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_null_scores, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 null)")) != eslOK) goto ERROR;
    engine->null_result_alloc = nseq;
  }
  if (engine->bias_result_alloc < nseq) {
    if (engine->d_bias_filtersc) cudaFree(engine->d_bias_filtersc);
    engine->d_bias_filtersc = NULL; engine->bias_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_filtersc, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 bias)")) != eslOK) goto ERROR;
    engine->bias_result_alloc = nseq;
  }
  if (engine->f1_result_alloc < nseq ||
      engine->d_f1_survivor_idx == NULL || engine->d_f1_counter == NULL ||
      engine->d_f1_survivor_usc == NULL || engine->d_f1_survivor_filtersc == NULL ||
      engine->d_f1_survivor_status == NULL) {
    if (engine->d_f1_survivor_idx) cudaFree(engine->d_f1_survivor_idx);
    if (engine->d_f1_counter) cudaFree(engine->d_f1_counter);
    if (engine->d_f1_survivor_usc) cudaFree(engine->d_f1_survivor_usc);
    if (engine->d_f1_survivor_filtersc) cudaFree(engine->d_f1_survivor_filtersc);
    if (engine->d_f1_survivor_status) cudaFree(engine->d_f1_survivor_status);
    engine->d_f1_survivor_idx = NULL;
    engine->d_f1_counter = NULL;
    engine->d_f1_survivor_usc = NULL;
    engine->d_f1_survivor_filtersc = NULL;
    engine->d_f1_survivor_status = NULL;
    engine->f1_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_idx, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 idx)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_counter, sizeof(int)), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 ctr)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_usc, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 usc)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_filtersc, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 filtersc)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_status, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 status)")) != eslOK) goto ERROR;
    engine->f1_result_alloc = nseq;
  }
  if (engine->f1_order_alloc < nseq || engine->d_f1_pass_mask == NULL) {
    if (engine->d_f1_pass_mask) cudaFree(engine->d_f1_pass_mask);
    engine->d_f1_pass_mask = NULL;
    engine->f1_order_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_pass_mask, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 pass mask)")) != eslOK) goto ERROR;
    engine->f1_order_alloc = nseq;
  }
  if (engine->h_f1_order_alloc < nseq) {
    free(engine->h_f1_pass_mask);
    free(engine->h_f1_filtersc);
    engine->h_f1_pass_mask = (int *)   malloc(sizeof(int)   * nseq);
    engine->h_f1_filtersc  = (float *) malloc(sizeof(float) * nseq);
    if (!engine->h_f1_pass_mask || !engine->h_f1_filtersc) { status = eslEMEM; goto ERROR; }
    engine->h_f1_order_alloc = nseq;
  }

  if (!engine->bias_params_uploaded && do_biasfilter && bg->fhmm) {
    float h_pi[3], h_t[6];
    size_t eo_bytes = (size_t) bg->fhmm->abc->Kp * 2 * sizeof(float);
    h_pi[0] = bg->fhmm->pi[0]; h_pi[1] = bg->fhmm->pi[1]; h_pi[2] = bg->fhmm->pi[2];
    h_t[0] = bg->fhmm->t[0][0]; h_t[1] = bg->fhmm->t[0][1]; h_t[2] = bg->fhmm->t[0][2];
    h_t[3] = bg->fhmm->t[1][0]; h_t[4] = bg->fhmm->t[1][1]; h_t[5] = bg->fhmm->t[1][2];
    if (engine->d_bias_pi == NULL) {
      if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_pi, sizeof(float) * 3), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 bias pi)")) != eslOK) goto ERROR;
      if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_t, sizeof(float) * 6), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 bias t)")) != eslOK) goto ERROR;
    }
    if (engine->d_bias_eo == NULL) {
      if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_eo, eo_bytes), errbuf, errbuf_size, "cudaMalloc(nhmmer resident f1 bias eo)")) != eslOK) goto ERROR;
    }
    if ((status = cuda_status(cudaMemcpy(engine->d_bias_pi, h_pi, sizeof(h_pi), cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer resident f1 bias pi)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_bias_t, h_t, sizeof(h_t), cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer resident f1 bias t)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_bias_eo, bg->fhmm->eo[0], eo_bytes, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer resident f1 bias eo)")) != eslOK) goto CUDA_ERROR;
    engine->bias_params_uploaded = 1;
  }

  if (h_src1_lengths_in) {
    if (engine->gather_alloc < nseq) {
      if (engine->d_gather_src1_offsets) cudaFree(engine->d_gather_src1_offsets);
      if (engine->d_gather_src1_lengths) cudaFree(engine->d_gather_src1_lengths);
      if (engine->d_gather_src2_offsets) cudaFree(engine->d_gather_src2_offsets);
      if (engine->d_gather_lengths)      cudaFree(engine->d_gather_lengths);
      if (engine->d_gather_dst_offsets)  cudaFree(engine->d_gather_dst_offsets);
      engine->d_gather_src1_offsets = NULL;
      engine->d_gather_src1_lengths = NULL;
      engine->d_gather_src2_offsets = NULL;
      engine->d_gather_lengths      = NULL;
      engine->d_gather_dst_offsets  = NULL;
      engine->gather_alloc = 0;
      if ((status = cuda_status(cudaMalloc((void **)&engine->d_gather_src1_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(f1 packed src1 offsets)")) != eslOK) goto ERROR;
      if ((status = cuda_status(cudaMalloc((void **)&engine->d_gather_src1_lengths, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(f1 packed src1 lengths)")) != eslOK) goto ERROR;
      if ((status = cuda_status(cudaMalloc((void **)&engine->d_gather_src2_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(f1 packed src2 offsets)")) != eslOK) goto ERROR;
      if ((status = cuda_status(cudaMalloc((void **)&engine->d_gather_lengths, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(f1 packed lengths)")) != eslOK) goto ERROR;
      if ((status = cuda_status(cudaMalloc((void **)&engine->d_gather_dst_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(f1 packed dst offsets)")) != eslOK) goto ERROR;
      engine->gather_alloc = nseq;
    }
  }

  cudaEventRecord(h2d0);
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemset(engine->d_f1_pass_mask, 0, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMemset(nhmmer resident f1 pass mask)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_offsets, h_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer resident f1 offsets)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_lengths, h_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer resident f1 lengths)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_tjb_by_seq, h_tjb_by_seq, sizeof(uint8_t) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer resident f1 tjb)")) != eslOK) goto CUDA_ERROR;
  if (h_src1_lengths_in) {
    if ((status = cuda_status(cudaMemcpy(engine->d_gather_src1_lengths, h_src1_lengths_in, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(f1 packed src1 lengths)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_gather_src2_offsets, h_src2_offsets_in, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(f1 packed src2 offsets)")) != eslOK) goto CUDA_ERROR;
  }
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(h2d1);

  {
    const uint8_t *d_dsq_ptr = d_dsq_base;
    const int     *d_off_ptr = engine->d_offsets;
    const int     *d_len_ptr = engine->d_lengths;
    const uint8_t *d_tjb_ptr = engine->d_tjb_by_seq;
    const int     *d_src1_len_ptr = h_src1_lengths_in ? engine->d_gather_src1_lengths : NULL;
    const int     *d_src2_off_ptr = h_src1_lengths_in ? engine->d_gather_src2_offsets : NULL;
    int            f1_rc_flag     = h_src1_lengths_in ? packed_rc_flag : 0;
    int stride = (cuom->M + 31) / 32;
    if (shmem > 48 * 1024) {
      switch (WARPS) {
        case 1: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 2: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 3: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 4: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 6: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
        case 8: cudaFuncSetAttribute((const void *) cuda_ssv_null_bias_gate_kernel<20, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); break;
      }
    }

    cudaEventRecord(k0);
    switch (WARPS) {
      case 1: SSV_NHMMER_MASK_DISPATCH_STRIDES(1); break;
      case 2: SSV_NHMMER_MASK_DISPATCH_STRIDES(2); break;
      case 3: SSV_NHMMER_MASK_DISPATCH_STRIDES(3); break;
      case 4: SSV_NHMMER_MASK_DISPATCH_STRIDES(4); break;
      case 6: SSV_NHMMER_MASK_DISPATCH_STRIDES(6); break;
      case 8: SSV_NHMMER_MASK_DISPATCH_STRIDES(8); break;
      default: SSV_NHMMER_MASK_DISPATCH_STRIDES(4); break;
    }
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_ssv_nhmmer_resident_f1_gate launch")) != eslOK) goto CUDA_ERROR;
    cudaEventRecord(k1);
  }
  ht0 = host_seconds();
  cudaEventSynchronize(k1);
  engine->stats.host_sync_seconds += host_seconds() - ht0;
  gate_seconds = elapsed_seconds(k0, k1);
  engine->stats.f1_gate_kernel_seconds += gate_seconds;

  cudaEventRecord(d2h0);
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(engine->h_f1_pass_mask, engine->d_f1_pass_mask, sizeof(int) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(nhmmer resident f1 pass mask D2H)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->h_f1_filtersc, engine->d_bias_filtersc, sizeof(float) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(nhmmer resident f1 filtersc D2H)")) != eslOK) goto CUDA_ERROR;
  {
    int total_surv = 0;
    for (int i = 0; i < nseq; i++) {
      if (engine->h_f1_pass_mask[i]) {
        survivor_idx[total_surv]      = i;
        survivor_filtersc[total_surv] = engine->h_f1_filtersc[i];
        if (survivor_statuses) survivor_statuses[total_surv] = 0;
        total_surv++;
      }
    }
    h_counter = total_surv;
  }
  if (h_counter > 0) {
    if ((status = cuda_status(cudaMemcpy(engine->d_f1_survivor_idx, survivor_idx, sizeof(int) * h_counter, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer resident f1 surv idx H2D)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_f1_survivor_filtersc, survivor_filtersc, sizeof(float) * h_counter, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nhmmer resident f1 surv filtersc H2D)")) != eslOK) goto CUDA_ERROR;
  }
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(d2h1);

  *ret_nsurv = h_counter;
  engine->stats.h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.kernel_seconds += gate_seconds;
  engine->stats.ssv_kernel_seconds += gate_seconds;
  engine->stats.d2h_seconds    += elapsed_seconds(d2h0, d2h1);
  engine->stats.nseqs          += nseq;
  engine->stats.nres           += (uint64_t) total_res;
  engine->stats.nbatches       += 1;
  engine->batch_owner           = d_dsq_base;
  engine->batch_nseq            = nseq;
  engine->batch_total           = 0;
  engine->d_batch_dsq           = d_dsq_base;
  engine->batch_packed_2bit     = h_src1_lengths_in ? 1 : 0;
  engine->batch_rc_flag         = h_src1_lengths_in ? packed_rc_flag : 0;
  engine->d_batch_packed_base   = h_src1_lengths_in ? d_dsq_base : NULL;
  engine->last_cuom             = cuom;

CUDA_ERROR:
ERROR:
  return status;
}

extern "C" int
p7_cuda_NhmmerF1GateResidentGather(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                   const P7_BG *bg, const uint8_t *d_dsq_base,
                                   const int *h_src1_offsets, const int *h_src1_lengths,
                                   const int *h_src2_offsets, const int *h_lengths, int nseq,
                                   int do_biasfilter, int B1,
                                   double ev_mu, double ev_lambda, double F1,
                                   int *survivor_idx, int *ret_nsurv,
                                   float *survivor_filtersc, int *survivor_statuses,
                                   int warps_per_block,
                                   int rc_flag,
                                   char *errbuf, int errbuf_size)
{
  if (!engine || !cuom || !bg || !d_dsq_base || !h_src1_offsets || !h_src1_lengths ||
      !h_src2_offsets || !h_lengths || !survivor_idx || !ret_nsurv || !survivor_filtersc)
    return eslEINVAL;
  if (nseq <= 0) { *ret_nsurv = 0; return eslOK; }

  return p7_cuda_NhmmerF1GateResident(engine, cuom, bg, d_dsq_base,
                                      h_src1_offsets, h_lengths, nseq,
                                      do_biasfilter, B1, ev_mu, ev_lambda, F1,
                                      survivor_idx, ret_nsurv,
                                      survivor_filtersc, survivor_statuses,
                                      warps_per_block, errbuf, errbuf_size,
                                      h_src1_lengths, h_src2_offsets, rc_flag);
}

extern "C" int
p7_cuda_PrepareResidentWindowBatch(P7_CUDA_ENGINE *engine, const uint8_t *d_dsq_base,
                                   const int *h_src1_offsets, const int *h_src1_lengths,
                                   const int *h_src2_offsets, const int *h_lengths,
                                   int nseq, const void *batch_owner, int rc_flag,
                                   char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int total = 1;

  if (!engine || !d_dsq_base || !h_src1_offsets || !h_src1_lengths ||
      !h_src2_offsets || !h_lengths || nseq < 0)
    return eslEINVAL;
  if (nseq == 0) return eslOK;

  if (engine->h_meta_alloc < nseq) {
    double ht0 = host_seconds();
    int     *new_offsets = (int *)     realloc(engine->h_meta_offsets,    sizeof(int)     * nseq);
    int     *new_lengths = (int *)     realloc(engine->h_meta_lengths,    sizeof(int)     * nseq);
    uint8_t *new_tjb     = (uint8_t *) realloc(engine->h_meta_tjb_by_seq, sizeof(uint8_t) * nseq);
    if (!new_offsets || !new_lengths || !new_tjb) {
      if (new_offsets) engine->h_meta_offsets = new_offsets;
      if (new_lengths) engine->h_meta_lengths = new_lengths;
      if (new_tjb)     engine->h_meta_tjb_by_seq = new_tjb;
      return eslEMEM;
    }
    engine->h_meta_offsets    = new_offsets;
    engine->h_meta_lengths    = new_lengths;
    engine->h_meta_tjb_by_seq = new_tjb;
    engine->h_meta_alloc      = nseq;
    engine->stats.host_malloc_free_seconds += host_seconds() - ht0;
  }

  for (int i = 0; i < nseq; i++) {
    if (h_lengths[i] < 0 || h_src1_lengths[i] < 0 || h_src1_lengths[i] > h_lengths[i])
      return eslERANGE;
    engine->h_meta_offsets[i] = total;
    engine->h_meta_lengths[i] = h_lengths[i];
    total += h_lengths[i] + 1;
  }

  if (engine->dsq_alloc < total) {
    if (engine->d_dsq) cudaFree(engine->d_dsq);
    if (engine->h_dsq) cudaFreeHost(engine->h_dsq);
    engine->d_dsq = NULL; engine->h_dsq = NULL;
    engine->dsq_alloc = 0; engine->h_dsq_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_dsq, total), errbuf, errbuf_size, "cudaMalloc(resident window batch dsq)")) != eslOK) return status;
    if ((status = cuda_status(cudaMallocHost((void **) &engine->h_dsq, total), errbuf, errbuf_size, "cudaMallocHost(resident window batch dsq scratch)")) != eslOK) return status;
    engine->dsq_alloc = total;
    engine->h_dsq_alloc = total;
  } else if (engine->h_dsq_alloc < total || engine->h_dsq == NULL) {
    if (engine->h_dsq) cudaFreeHost(engine->h_dsq);
    engine->h_dsq = NULL;
    engine->h_dsq_alloc = 0;
    if ((status = cuda_status(cudaMallocHost((void **) &engine->h_dsq, total), errbuf, errbuf_size, "cudaMallocHost(resident window batch dsq scratch)")) != eslOK) return status;
    engine->h_dsq_alloc = total;
  }

  if (engine->meta_alloc < nseq) {
    if (engine->d_offsets) cudaFree(engine->d_offsets);
    if (engine->d_lengths) cudaFree(engine->d_lengths);
    if (engine->d_tjb_by_seq) cudaFree(engine->d_tjb_by_seq);
    engine->d_offsets = NULL; engine->d_lengths = NULL; engine->d_tjb_by_seq = NULL;
    engine->meta_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(resident window batch offsets)")) != eslOK) return status;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_lengths, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(resident window batch lengths)")) != eslOK) return status;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_tjb_by_seq, sizeof(uint8_t) * nseq), errbuf, errbuf_size, "cudaMalloc(resident window batch tjb)")) != eslOK) return status;
    engine->meta_alloc = nseq;
  }

  if (engine->gather_alloc < nseq) {
    if (engine->d_gather_src1_offsets) cudaFree(engine->d_gather_src1_offsets);
    if (engine->d_gather_src1_lengths) cudaFree(engine->d_gather_src1_lengths);
    if (engine->d_gather_src2_offsets) cudaFree(engine->d_gather_src2_offsets);
    if (engine->d_gather_lengths)      cudaFree(engine->d_gather_lengths);
    if (engine->d_gather_dst_offsets)  cudaFree(engine->d_gather_dst_offsets);
    engine->d_gather_src1_offsets = NULL;
    engine->d_gather_src1_lengths = NULL;
    engine->d_gather_src2_offsets = NULL;
    engine->d_gather_lengths      = NULL;
    engine->d_gather_dst_offsets  = NULL;
    engine->gather_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_gather_src1_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(resident window gather src1 offsets)")) != eslOK) return status;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_gather_src1_lengths, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(resident window gather src1 lengths)")) != eslOK) return status;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_gather_src2_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(resident window gather src2 offsets)")) != eslOK) return status;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_gather_lengths, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(resident window gather lengths)")) != eslOK) return status;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_gather_dst_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(resident window gather dst offsets)")) != eslOK) return status;
    engine->gather_alloc = nseq;
  }

  cudaEventRecord(engine->evt_h2d0);
  if ((status = cuda_status(cudaMemcpy(engine->d_gather_src1_offsets, h_src1_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(resident window gather src1 offsets)")) != eslOK) return status;
  if ((status = cuda_status(cudaMemcpy(engine->d_gather_src1_lengths, h_src1_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(resident window gather src1 lengths)")) != eslOK) return status;
  if ((status = cuda_status(cudaMemcpy(engine->d_gather_src2_offsets, h_src2_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(resident window gather src2 offsets)")) != eslOK) return status;
  if ((status = cuda_status(cudaMemcpy(engine->d_gather_lengths, h_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(resident window gather lengths)")) != eslOK) return status;
  if ((status = cuda_status(cudaMemcpy(engine->d_offsets, h_src1_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(resident window batch offsets)")) != eslOK) return status;
  if ((status = cuda_status(cudaMemcpy(engine->d_lengths, engine->h_meta_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(resident window batch lengths)")) != eslOK) return status;
  cudaEventRecord(engine->evt_h2d1);
  engine->stats.h2d_seconds += elapsed_seconds(engine->evt_h2d0, engine->evt_h2d1);

  engine->batch_owner = batch_owner;
  engine->batch_nseq  = nseq;
  engine->batch_total = total;
  engine->d_batch_dsq       = (const uint8_t *)d_dsq_base;
  engine->batch_packed_2bit = 1;
  engine->batch_rc_flag     = rc_flag;
  engine->d_batch_packed_base = d_dsq_base;
  return eslOK;
}

/* ================================================================ */

extern "C" int
p7_cuda_SSVFilterDsqdataChunk(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                              ESL_DSQDATA_CHUNK *chu, float *scores, int *statuses,
                              char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int nseq;
  int total = 0;
  int *h_offsets = NULL;
  int *h_lengths = NULL;
  uint8_t *h_tjb_by_seq = NULL;
  int *h_raw = NULL;
  int *h_overflow = NULL;
  size_t shmem = (size_t) (cuom->M + 1) * 2;
  cudaEvent_t h2d0 = engine->evt_h2d0, h2d1 = engine->evt_h2d1;
  cudaEvent_t k0 = engine->evt_k0, k1 = engine->evt_k1;
  cudaEvent_t d2h0 = engine->evt_d2h0, d2h1 = engine->evt_d2h1;
  double ht0;

  if (!engine || !cuom || !chu || !scores || !statuses) return eslEINVAL;
  nseq = chu->N;
  if (nseq <= 0) return eslOK;
  if (shmem > 96 * 1024) {
    if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "CUDA SSV profile M=%d exceeds shared-memory limit", cuom->M);
    return eslERANGE;
  }

  ht0 = host_seconds();
  h_offsets    = (int *)     malloc(sizeof(int) * nseq);
  h_lengths    = (int *)     malloc(sizeof(int) * nseq);
  h_tjb_by_seq = (uint8_t *) malloc(sizeof(uint8_t) * nseq);
  h_raw        = (int *)     malloc(sizeof(int) * nseq);
  h_overflow   = (int *)     malloc(sizeof(int) * nseq);
  if (!h_offsets || !h_lengths || !h_tjb_by_seq || !h_raw || !h_overflow)
    { status = eslEMEM; goto ERROR; }
  engine->stats.host_malloc_free_seconds += host_seconds() - ht0;

  ht0 = host_seconds();
  for (int i = 0; i < nseq; i++) {
    h_offsets[i] = total;
    if (chu->L[i] > INT32_MAX) {
      if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "dsqdata sequence length exceeds CUDA SSV limit");
      status = eslERANGE;
      goto ERROR;
    }
    h_lengths[i] = (int) chu->L[i];
    if ((status = cuda_msvprofile_GrowLengthLookup((P7_CUDA_MSVPROFILE *) cuom, h_lengths[i])) != eslOK) goto ERROR;
    h_tjb_by_seq[i] = cuom->h_tjb_by_len[h_lengths[i]];
    total += h_lengths[i] + 1;
  }
  total += 1;
  engine->stats.host_metadata_loop_seconds += host_seconds() - ht0;

  if (engine->dsq_alloc < total) {
    if (engine->d_dsq) cudaFree(engine->d_dsq);
    if (engine->h_dsq) cudaFreeHost(engine->h_dsq);
    engine->d_dsq = NULL;
    engine->h_dsq = NULL;
    engine->dsq_alloc = 0;
    engine->h_dsq_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_dsq, total), errbuf, errbuf_size, "cudaMalloc(ssv dsq)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMallocHost((void **) &engine->h_dsq, total), errbuf, errbuf_size, "cudaMallocHost(ssv dsq)")) != eslOK) goto ERROR;
    engine->dsq_alloc = total;
    engine->h_dsq_alloc = total;
  }
  if (engine->meta_alloc < nseq) {
    if (engine->d_offsets) cudaFree(engine->d_offsets);
    if (engine->d_lengths) cudaFree(engine->d_lengths);
    if (engine->d_tjb_by_seq) cudaFree(engine->d_tjb_by_seq);
    engine->d_offsets = NULL;
    engine->d_lengths = NULL;
    engine->d_tjb_by_seq = NULL;
    engine->meta_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(ssv offsets)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_lengths, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(ssv lengths)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_tjb_by_seq, sizeof(uint8_t) * nseq), errbuf, errbuf_size, "cudaMalloc(ssv tjb)")) != eslOK) goto ERROR;
    engine->meta_alloc = nseq;
  }
  if (engine->result_alloc < nseq) {
    if (engine->d_raw) cudaFree(engine->d_raw);
    if (engine->d_overflow) cudaFree(engine->d_overflow);
    engine->d_raw = NULL;
    engine->d_overflow = NULL;
    engine->result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_raw, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(ssv raw)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_overflow, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(ssv overflow)")) != eslOK) goto ERROR;
    engine->result_alloc = nseq;
  }

  ht0 = host_seconds();
  if (chu->smem != NULL) {
    memcpy(engine->h_dsq, chu->smem, total);
  } else {
    for (int i = 0; i < nseq; i++)
      memcpy(engine->h_dsq + h_offsets[i], chu->dsq[i], h_lengths[i] + 1);
  }
  engine->stats.host_pack_loop_seconds += host_seconds() - ht0;

  cudaEventRecord(h2d0);
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(engine->d_dsq, engine->h_dsq, total, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(ssv dsq)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_offsets, h_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(ssv offsets)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_lengths, h_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(ssv lengths)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_tjb_by_seq, h_tjb_by_seq, sizeof(uint8_t) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(ssv tjb)")) != eslOK) goto CUDA_ERROR;
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(h2d1);

  cudaEventRecord(k0);
  cuda_ssv_opt_kernel<<<nseq, P7_CUDA_MSV_BLOCK_THREADS, shmem>>>(engine->d_dsq, engine->d_offsets, engine->d_lengths, engine->d_tjb_by_seq, nseq,
                                                                   cuom->d_rbv, cuom->M, cuom->Q, cuom->Kp,
                                                                   cuom->tbm_b, cuom->tec_b, cuom->tjb_b,
                                                                   cuom->base_b, cuom->bias_b,
                                                                   engine->d_raw, engine->d_overflow);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_ssv_opt_kernel launch")) != eslOK) goto CUDA_ERROR;
  cudaEventRecord(k1);
  ht0 = host_seconds();
  cudaEventSynchronize(k1);
  engine->stats.host_sync_seconds += host_seconds() - ht0;

  cudaEventRecord(d2h0);
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(h_raw, engine->d_raw, sizeof(int) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(ssv raw)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(h_overflow, engine->d_overflow, sizeof(int) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(ssv overflow)")) != eslOK) goto CUDA_ERROR;
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(d2h1);

  engine->batch_owner = chu;
  engine->batch_nseq  = nseq;
  engine->batch_total = total;
  engine->last_cuom   = cuom;
  engine->stats.h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.ssv_kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.d2h_seconds    += elapsed_seconds(d2h0, d2h1);
  engine->stats.nseqs          += nseq;
  engine->stats.nres           += total - (2 * nseq);
  engine->stats.nbatches       += 1;

  ht0 = host_seconds();
  for (int i = 0; i < nseq; i++) {
    if (h_overflow[i]) {
      scores[i] = eslINFINITY;
      statuses[i] = eslERANGE;
    } else {
      scores[i] = ((float) (h_raw[i] - h_tjb_by_seq[i]) - (float) cuom->base_b);
      scores[i] /= cuom->scale_b;
      scores[i] -= 3.0f;
      statuses[i] = eslOK;
    }
  }
  engine->stats.host_score_convert_seconds += host_seconds() - ht0;

CUDA_ERROR:
ERROR:
  ht0 = host_seconds();
  free(h_offsets);
  free(h_lengths);
  free(h_tjb_by_seq);
  free(h_raw);
  free(h_overflow);
  engine->stats.host_malloc_free_seconds += host_seconds() - ht0;
  return status;
}

extern "C" int
p7_cuda_SSVFilterResident(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                          int64_t seq0, int nseq, float *scores, int *statuses,
                          char *errbuf, int errbuf_size)
{
  int        status = eslOK;
  uint8_t   *h_tjb_by_seq = NULL;
  int       *h_raw = NULL;
  int       *h_overflow = NULL;
  size_t     shmem = (size_t) (cuom->M + 1) * 2;
  cudaEvent_t h2d0 = engine->evt_h2d0, h2d1 = engine->evt_h2d1;
  cudaEvent_t k0 = engine->evt_k0, k1 = engine->evt_k1;
  cudaEvent_t d2h0 = engine->evt_d2h0, d2h1 = engine->evt_d2h1;
  double     ht0;

  if (!engine || !cuom || !scores || !statuses) return eslEINVAL;
  if (!engine->resident_active) return eslEINVAL;
  if (nseq <= 0) return eslOK;
  if (seq0 < 0 || seq0 + nseq > engine->resident_nseq) return eslEINVAL;
  if (shmem > 96 * 1024) {
    if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "CUDA SSV profile M=%d exceeds shared-memory limit", cuom->M);
    return eslERANGE;
  }

  ht0 = host_seconds();
  h_tjb_by_seq = (uint8_t *) malloc(sizeof(uint8_t) * nseq);
  h_raw        = (int *)     malloc(sizeof(int) * nseq);
  h_overflow   = (int *)     malloc(sizeof(int) * nseq);
  if (!h_tjb_by_seq || !h_raw || !h_overflow) { status = eslEMEM; goto ERROR; }
  engine->stats.host_malloc_free_seconds += host_seconds() - ht0;

  ht0 = host_seconds();
  {
    int *h_lengths_tmp = (int *) malloc(sizeof(int) * nseq);
    if (!h_lengths_tmp) { status = eslEMEM; goto ERROR; }
    if ((status = cuda_status(cudaMemcpy(h_lengths_tmp, engine->d_resident_lengths + seq0, sizeof(int) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(resident lengths->host)")) != eslOK) { free(h_lengths_tmp); goto ERROR; }
    for (int i = 0; i < nseq; i++) {
      if ((status = cuda_msvprofile_GrowLengthLookup((P7_CUDA_MSVPROFILE *) cuom, h_lengths_tmp[i])) != eslOK) { free(h_lengths_tmp); goto ERROR; }
      h_tjb_by_seq[i] = cuom->h_tjb_by_len[h_lengths_tmp[i]];
    }
    free(h_lengths_tmp);
  }
  engine->stats.host_metadata_loop_seconds += host_seconds() - ht0;

  if (engine->meta_alloc < nseq) {
    if (engine->d_tjb_by_seq) cudaFree(engine->d_tjb_by_seq);
    engine->d_tjb_by_seq = NULL;
    engine->meta_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_tjb_by_seq, sizeof(uint8_t) * nseq), errbuf, errbuf_size, "cudaMalloc(tjb_by_seq)")) != eslOK) goto ERROR;
    engine->meta_alloc = nseq;
  }
  if (engine->result_alloc < nseq) {
    if (engine->d_raw) cudaFree(engine->d_raw);
    if (engine->d_overflow) cudaFree(engine->d_overflow);
    engine->d_raw = NULL;
    engine->d_overflow = NULL;
    engine->result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_raw, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(raw)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_overflow, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(overflow)")) != eslOK) goto ERROR;
    engine->result_alloc = nseq;
  }

  cudaEventRecord(h2d0);
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(engine->d_tjb_by_seq, h_tjb_by_seq, sizeof(uint8_t) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(tjb_by_seq)")) != eslOK) goto CUDA_ERROR;
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(h2d1);

  cudaEventRecord(k0);
  cuda_ssv_opt_kernel<<<nseq, P7_CUDA_MSV_BLOCK_THREADS, shmem>>>(
    engine->d_resident_dsq, engine->d_resident_offsets + seq0, engine->d_resident_lengths + seq0,
    engine->d_tjb_by_seq, nseq,
    cuom->d_rbv, cuom->M, cuom->Q, cuom->Kp,
    cuom->tbm_b, cuom->tec_b, cuom->tjb_b,
    cuom->base_b, cuom->bias_b,
    engine->d_raw, engine->d_overflow);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_ssv_opt_kernel(resident) launch")) != eslOK) goto CUDA_ERROR;
  cudaEventRecord(k1);
  ht0 = host_seconds();
  cudaEventSynchronize(k1);
  engine->stats.host_sync_seconds += host_seconds() - ht0;

  cudaEventRecord(d2h0);
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(h_raw, engine->d_raw, sizeof(int) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(raw)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(h_overflow, engine->d_overflow, sizeof(int) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(overflow)")) != eslOK) goto CUDA_ERROR;
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(d2h1);

  engine->stats.h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.ssv_kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.d2h_seconds    += elapsed_seconds(d2h0, d2h1);
  engine->stats.nseqs          += nseq;
  engine->stats.nbatches       += 1;

  ht0 = host_seconds();
  for (int i = 0; i < nseq; i++) {
    if (h_overflow[i]) {
      scores[i] = eslINFINITY;
      statuses[i] = eslERANGE;
    } else {
      scores[i] = ((float) (h_raw[i] - h_tjb_by_seq[i]) - (float) cuom->base_b);
      scores[i] /= cuom->scale_b;
      scores[i] -= 3.0f;
      statuses[i] = eslOK;
    }
  }
  engine->stats.host_score_convert_seconds += host_seconds() - ht0;

  engine->last_cuom           = cuom;
  engine->batch_owner         = NULL;
  engine->batch_nseq          = nseq;
  engine->resident_batch_seq0 = seq0;
  engine->resident_batch_nseq = nseq;

CUDA_ERROR:
ERROR:
  ht0 = host_seconds();
  free(h_tjb_by_seq);
  free(h_raw);
  free(h_overflow);
  engine->stats.host_malloc_free_seconds += host_seconds() - ht0;
  return status;
}
