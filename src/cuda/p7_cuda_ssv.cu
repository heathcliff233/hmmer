#include "p7_cuda_internal.h"

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

template <int STRIDE>
__global__ __launch_bounds__(32, 16)
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
    /* F1 Gumbel gate params */
    float ev_mu, float ev_lambda, float F1, float log2_inv,
    /* outputs: raw scores still stored for --gpu-ssv-compare compatibility */
    int *raw_sc, int *overflow_out,
    /* outputs: null/bias/survivors for downstream stages */
    float *nullsc_out, float *filtersc_out,
    int *survivor_idx, int *survivor_counter,
    float *survivor_usc, int *survivor_status)
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

  /* Precompute q/z lookup in shared memory (only needed for MSV fallback) */
  uint8_t *s_q = mem;
  uint8_t *s_z = mem + M;
  for (int k = tid; k < M; k += 32) {
    s_q[k] = (uint8_t)(k % Q);
    s_z[k] = (uint8_t)(k / Q);
  }
  __syncthreads();

  __shared__ int use_full_msv;
  int raw_result = 0;
  int overflow_result = 0;

  if (stride <= STRIDE &&
      (int) seq_tjb_b + (int) tbm_b + (int) tec_b + (int) bias_b < 127) {
    uint8_t xB_ssv = u8_sub_sat(base_b, (uint8_t)(seq_tjb_b + tbm_b));
    uint8_t xE_local = 0;

    uint8_t prev_reg[STRIDE];
    #pragma unroll
    for (int j = 0; j < STRIDE; j++) prev_reg[j] = 0;

    if (tid == 0) {
      use_full_msv = 0;
    }

    uint8_t last_prev = 0;

    for (int i = 1; i <= L; i++) {
      uint8_t x = sdsq[i];
      uint8_t from_left = __shfl_up_sync(0xffffffff, last_prev, 1);
      if (tid == 0) from_left = 0;

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

    if (tid == 0) {
      int shifted_xE = (int) xE_local - (int) base_b + (int) seq_tjb_b + (int) tbm_b + 128;
      if (shifted_xE >= 255 - (int) bias_b) {
        if ((int) base_b - (int) seq_tjb_b - (int) tbm_b < 128) {
          use_full_msv = 1;
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
          use_full_msv = 1;
        } else {
          raw_result = xJ;
          overflow_result = 0;
        }
      }
    }
    __syncthreads();
    if (!use_full_msv) goto SSV_DONE;
  }

  /* Full MSV fallback (shared-memory path for ~0.3% of sequences or stride > STRIDE) */
  {
    uint8_t *prev = mem;
    uint8_t *curr = prev + M + 1;

    for (int k = tid; k <= M; k += 32) prev[k] = 0;
    if (tid == 0) overflow_result = 0;
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
        if (u8_add_sat(v, bias_b) == 255) overflow_result = 1;
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

    if (tid == 0) raw_result = (int) xJ;
  }

SSV_DONE:
  /* Store raw SSV results (needed for --gpu-ssv-compare and downstream stats) */
  if (tid == 0) {
    raw_sc[seq] = raw_result;
    overflow_out[seq] = overflow_result;
  }

  /* ---- Null + Bias + F1 Gate (thread 0 only) ---- */
  if (tid == 0) {
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

      float p0  = bias_eo[(int) sdsq[1] * 2 + 0] * bias_pi[0];
      float p1s = bias_eo[(int) sdsq[1] * 2 + 1] * bias_pi[1];
      float maxv = fmaxf(fmaxf(p0, p1s), 0.0f);
      p0  /= maxv;
      p1s /= maxv;
      float logsc = logf(maxv);

      for (int i = 2; i <= L; i++) {
        uint8_t x = sdsq[i];
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

    float seq_score = (usc - nullsc) * log2_inv;
    float y = ev_lambda * (seq_score - ev_mu);
    float ey = -expf(-y);
    float P = (fabsf(ey) < 1e-7f) ? -ey : 1.0f - expf(ey);

    if (overflow_result) P = 0.0f;
    if (P > F1) return;

    if (do_biasfilter) {
      float bias_score = (usc - filtersc) * log2_inv;
      float by = ev_lambda * (bias_score - ev_mu);
      float bey = -expf(-by);
      float bP = (fabsf(bey) < 1e-7f) ? -bey : 1.0f - expf(bey);
      if (bP > F1) return;
    }

    int idx = atomicAdd(survivor_counter, 1);
    survivor_idx[idx] = seq;
    survivor_usc[idx] = usc;
    survivor_status[idx] = overflow_result ? 4 : 0;  /* eslERANGE=4, eslOK=0 */
  }
}

/* Dispatch macro for fused kernel template instantiation */
#define SSV_FUSED_LAUNCH(STRIDE_VAL) \
  cuda_ssv_null_bias_gate_kernel<STRIDE_VAL><<<nseq, P7_CUDA_MSV_BLOCK_THREADS, shmem>>>( \
    d_dsq_ptr, d_off_ptr, d_len_ptr, d_tjb_ptr, nseq, \
    cuom->d_rbv, cuom->d_rbv_lin, cuom->M, cuom->Q, cuom->Kp, \
    cuom->tbm_b, cuom->tec_b, cuom->tjb_b, \
    cuom->base_b, cuom->bias_b, cuom->scale_b, \
    engine->d_bias_pi, engine->d_bias_t, engine->d_bias_eo, \
    do_biasfilter, \
    (float) ev_mu, (float) ev_lambda, (float) F1, log2_inv, \
    engine->d_raw, engine->d_overflow, \
    engine->d_null_scores, engine->d_bias_filtersc, \
    engine->d_f1_survivor_idx, engine->d_f1_counter, \
    engine->d_f1_survivor_usc, engine->d_f1_survivor_status)

extern "C" int
p7_cuda_SSVNullBiasGateResident(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                const P7_BG *bg, int64_t seq0, int nseq, int do_biasfilter,
                                double ev_mu, double ev_lambda, double F1,
                                int *survivor_idx, int *ret_nsurv,
                                float *nullsc, float *filtersc,
                                float *survivor_scores, int *survivor_statuses,
                                char *errbuf, int errbuf_size)
{
  int        status = eslOK;
  uint8_t   *h_tjb_by_seq = NULL;
  size_t     shmem = (size_t) (cuom->M + 1) * 2;
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
  if (shmem > 96 * 1024) {
    if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "CUDA SSV fused profile M=%d exceeds shared-memory limit", cuom->M);
    return eslERANGE;
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
  if (engine->f1_result_alloc < nseq) {
    if (engine->d_f1_survivor_idx) cudaFree(engine->d_f1_survivor_idx);
    if (engine->d_f1_counter) cudaFree(engine->d_f1_counter);
    if (engine->d_f1_survivor_usc) cudaFree(engine->d_f1_survivor_usc);
    if (engine->d_f1_survivor_status) cudaFree(engine->d_f1_survivor_status);
    engine->d_f1_survivor_idx = NULL;
    engine->d_f1_counter = NULL;
    engine->d_f1_survivor_usc = NULL;
    engine->d_f1_survivor_status = NULL;
    engine->f1_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_idx, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(fused f1 idx)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_counter, sizeof(int)), errbuf, errbuf_size, "cudaMalloc(fused f1 ctr)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_usc, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(fused f1 usc)")) != eslOK) goto ERROR;
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

  /* Launch fused kernel with STRIDE dispatch */
  {
    const uint8_t *d_dsq_ptr = engine->d_resident_dsq;
    const int     *d_off_ptr = engine->d_resident_offsets + seq0;
    const int     *d_len_ptr = engine->d_resident_lengths + seq0;
    const uint8_t *d_tjb_ptr = engine->d_tjb_by_seq;
    int stride = (cuom->M + 31) / 32;

    cudaEventRecord(k0);
    switch (stride) {
      case  1: SSV_FUSED_LAUNCH(1);  break;
      case  2: SSV_FUSED_LAUNCH(2);  break;
      case  3: SSV_FUSED_LAUNCH(3);  break;
      case  4: SSV_FUSED_LAUNCH(4);  break;
      case  5: SSV_FUSED_LAUNCH(5);  break;
      case  6: SSV_FUSED_LAUNCH(6);  break;
      case  7: SSV_FUSED_LAUNCH(7);  break;
      case  8: SSV_FUSED_LAUNCH(8);  break;
      case  9: SSV_FUSED_LAUNCH(9);  break;
      case 10: SSV_FUSED_LAUNCH(10); break;
      case 11: SSV_FUSED_LAUNCH(11); break;
      case 12: SSV_FUSED_LAUNCH(12); break;
      case 13: SSV_FUSED_LAUNCH(13); break;
      case 14: SSV_FUSED_LAUNCH(14); break;
      case 15: SSV_FUSED_LAUNCH(15); break;
      case 16: SSV_FUSED_LAUNCH(16); break;
      case 17: SSV_FUSED_LAUNCH(17); break;
      case 18: SSV_FUSED_LAUNCH(18); break;
      case 19: SSV_FUSED_LAUNCH(19); break;
      case 20: SSV_FUSED_LAUNCH(20); break;
      default: SSV_FUSED_LAUNCH(20); break;  /* fallback uses shared-mem path inside kernel */
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
                                    char *errbuf, int errbuf_size)
{
  int        status = eslOK;
  int        nseq;
  int        total = 0;
  int       *h_offsets = NULL;
  int       *h_lengths = NULL;
  uint8_t   *h_tjb_by_seq = NULL;
  size_t     shmem = (size_t) (cuom->M + 1) * 2;
  cudaEvent_t h2d0 = engine->evt_h2d0, h2d1 = engine->evt_h2d1;
  cudaEvent_t k0 = engine->evt_k0, k1 = engine->evt_k1;
  cudaEvent_t d2h0 = engine->evt_d2h0, d2h1 = engine->evt_d2h1;
  double     ht0;
  int        h_counter = 0;
  float      log2_inv = 1.0f / 0.693147180559945f;

  if (!engine || !cuom || !bg || !chu || !survivor_idx || !ret_nsurv) return eslEINVAL;
  nseq = chu->N;
  if (nseq <= 0) { *ret_nsurv = 0; return eslOK; }
  if (shmem > 96 * 1024) {
    if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "CUDA SSV fused profile M=%d exceeds shared-memory limit", cuom->M);
    return eslERANGE;
  }

  ht0 = host_seconds();
  h_offsets    = (int *)     malloc(sizeof(int) * nseq);
  h_lengths    = (int *)     malloc(sizeof(int) * nseq);
  h_tjb_by_seq = (uint8_t *) malloc(sizeof(uint8_t) * nseq);
  if (!h_offsets || !h_lengths || !h_tjb_by_seq) { status = eslEMEM; goto ERROR; }
  engine->stats.host_malloc_free_seconds += host_seconds() - ht0;

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
  if (engine->f1_result_alloc < nseq) {
    if (engine->d_f1_survivor_idx) cudaFree(engine->d_f1_survivor_idx);
    if (engine->d_f1_counter) cudaFree(engine->d_f1_counter);
    if (engine->d_f1_survivor_usc) cudaFree(engine->d_f1_survivor_usc);
    if (engine->d_f1_survivor_status) cudaFree(engine->d_f1_survivor_status);
    engine->d_f1_survivor_idx = NULL;
    engine->d_f1_counter = NULL;
    engine->d_f1_survivor_usc = NULL;
    engine->d_f1_survivor_status = NULL;
    engine->f1_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_idx, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(fused f1 idx)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_counter, sizeof(int)), errbuf, errbuf_size, "cudaMalloc(fused f1 ctr)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_usc, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(fused f1 usc)")) != eslOK) goto ERROR;
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

  /* Launch fused kernel with STRIDE dispatch */
  {
    const uint8_t *d_dsq_ptr = engine->d_dsq;
    const int     *d_off_ptr = engine->d_offsets;
    const int     *d_len_ptr = engine->d_lengths;
    const uint8_t *d_tjb_ptr = engine->d_tjb_by_seq;
    int stride = (cuom->M + 31) / 32;

    cudaEventRecord(k0);
    switch (stride) {
      case  1: SSV_FUSED_LAUNCH(1);  break;
      case  2: SSV_FUSED_LAUNCH(2);  break;
      case  3: SSV_FUSED_LAUNCH(3);  break;
      case  4: SSV_FUSED_LAUNCH(4);  break;
      case  5: SSV_FUSED_LAUNCH(5);  break;
      case  6: SSV_FUSED_LAUNCH(6);  break;
      case  7: SSV_FUSED_LAUNCH(7);  break;
      case  8: SSV_FUSED_LAUNCH(8);  break;
      case  9: SSV_FUSED_LAUNCH(9);  break;
      case 10: SSV_FUSED_LAUNCH(10); break;
      case 11: SSV_FUSED_LAUNCH(11); break;
      case 12: SSV_FUSED_LAUNCH(12); break;
      case 13: SSV_FUSED_LAUNCH(13); break;
      case 14: SSV_FUSED_LAUNCH(14); break;
      case 15: SSV_FUSED_LAUNCH(15); break;
      case 16: SSV_FUSED_LAUNCH(16); break;
      case 17: SSV_FUSED_LAUNCH(17); break;
      case 18: SSV_FUSED_LAUNCH(18); break;
      case 19: SSV_FUSED_LAUNCH(19); break;
      case 20: SSV_FUSED_LAUNCH(20); break;
      default: SSV_FUSED_LAUNCH(20); break;
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
  ht0 = host_seconds();
  free(h_offsets);
  free(h_lengths);
  free(h_tjb_by_seq);
  engine->stats.host_malloc_free_seconds += host_seconds() - ht0;
  return status;
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
