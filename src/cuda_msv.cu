#include <p7_config.h>

#include <cuda_runtime.h>

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "cuda_msv.h"
}

#define P7_CUDA_MSV_BLOCK_THREADS 32

struct p7_cuda_msv_engine_s {
  int                 device_id;
  uint8_t            *d_dsq;
  int                 dsq_alloc;
  uint8_t            *h_dsq;
  int                 h_dsq_alloc;
  const void         *batch_owner;
  int                 batch_nseq;
  int                 batch_total;
  int                *d_offsets;
  int                *d_lengths;
  uint8_t            *d_tjb_by_seq;
  int                 meta_alloc;
  int                *d_raw;
  int                *d_overflow;
  int                 result_alloc;
  float              *d_bias_filtersc;
  int                 bias_result_alloc;
  float              *d_bias_pi;
  float              *d_bias_t;
  float              *d_bias_eo;
  float              *d_fwd_scores;
  int                *d_fwd_statuses;
  int                *d_fwd_seqidx;
  int                 fwd_result_alloc;
  float              *d_vit_scores;
  int                *d_vit_statuses;
  int                *d_vit_seqidx;
  int                 vit_result_alloc;
  float              *d_fwd_prev;
  float              *d_fwd_curr;
  size_t              fwd_dp_alloc;
  P7_CUDA_MSV_STATS   stats;
};

struct p7_cuda_msv_profile_s {
  int      M;
  int      Q;
  int      Kp;
  uint8_t  tbm_b;
  uint8_t  tec_b;
  uint8_t  tjb_b;
  uint8_t *h_tjb_by_len;
  int      tjb_len_alloc;
  float    scale_b;
  uint8_t  base_b;
  uint8_t  bias_b;
  uint8_t *d_rbv;
  int      Qf;
  float   *d_rfv;
  float   *d_tfv;
  float    xf_e_loop;
  float    xf_e_move;
  float    xf_n_loop;
  float    xf_n_move;
  float    xf_c_loop;
  float    xf_c_move;
  float    xf_j_loop;
  float    xf_j_move;
  float    nj;
  int      Qw;
  int16_t *d_rwv;
  int16_t *d_twv;
  int16_t  xw_n_loop;
  int16_t  xw_n_move;
  int16_t  xw_c_loop;
  int16_t  xw_c_move;
  int16_t  xw_j_loop;
  int16_t  xw_j_move;
  int16_t  xw_e_loop;
  int16_t  xw_e_move;
  float    scale_w;
  int16_t  base_w;
  int16_t  ddbound_w;
};

static int
cuda_status(cudaError_t cerr, char *errbuf, int errbuf_size, const char *where)
{
  if (cerr == cudaSuccess) return eslOK;
  if (errbuf && errbuf_size > 0)
    snprintf(errbuf, errbuf_size, "%s: %s", where, cudaGetErrorString(cerr));
  return eslFAIL;
}

static double
elapsed_seconds(cudaEvent_t start, cudaEvent_t stop)
{
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  return (double) ms / 1000.0;
}

static int
next_pow2_at_least(int n, int min_n)
{
  int p = min_n;
  while (p < n) p <<= 1;
  return p;
}

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

__device__ static inline int16_t
i16_add_sat(int16_t a, int16_t b)
{
  int v = (int) a + (int) b;
  if (v > 32767) return 32767;
  if (v < -32768) return -32768;
  return (int16_t) v;
}

__device__ static inline int16_t
i16_wordify(float scale, float sc)
{
  float v = roundf(scale * sc);
  if (v >= 32767.0f) return 32767;
  if (v <= -32768.0f) return -32768;
  return (int16_t) v;
}

__device__ static inline int
vit_twv_idx(int t, int q, int lane, int Q)
{
  return (t == p7O_DD) ? (((7 * Q) + q) * 8 + lane) : (((q * 7) + t) * 8 + lane);
}

__device__ static inline int
fwd_tfv_idx(int t, int q, int lane, int Q)
{
  return (t == p7O_DD) ? (((7 * Q) + q) * 4 + lane) : (((q * 7) + t) * 4 + lane);
}

__global__ static void
cuda_viterbi_score_kernel(const uint8_t *dsq, const int *offsets, const int *lengths,
                          const int *seqidx, int nidx,
                          const int16_t *rwv, const int16_t *twv, int M, int Q, int Kp,
                          int16_t xw_n_loop, int16_t xw_n_move, int16_t xw_c_loop, int16_t xw_c_move,
                          int16_t xw_j_loop, int16_t xw_j_move, int16_t xw_e_loop, int16_t xw_e_move,
                          float scale_w, int16_t base_w, int16_t ddbound_w, float nj,
                          float *scores, int *statuses)
{
  extern __shared__ int16_t vit_mem[];
  int N = Q * 8;
  int16_t *prev = vit_mem;
  int16_t *curr = prev + (size_t) N * 3;
  int bi = blockIdx.x;
  int lane = threadIdx.x & 31;
  unsigned int mask = 0xff;

  if (bi >= nidx || lane >= 8) return;

  int si = seqidx ? seqidx[bi] : bi;
  int L = lengths[si];
  const uint8_t *s = dsq + offsets[si];
  float pmove = (2.0f + nj) / ((float) L + 2.0f + nj);
  int16_t xw_move = i16_wordify(scale_w, logf(pmove));

  for (int q = 0; q < Q; q++) {
    int cell = (q + lane * Q) * 3;
    prev[cell + 0] = -32768;
    prev[cell + 1] = -32768;
    prev[cell + 2] = -32768;
  }

  int16_t xN = base_w;
  int16_t xB = i16_add_sat(xN, xw_move);
  int16_t xJ = -32768;
  int16_t xC = -32768;
  int overflow = 0;

  for (int i = 1; i <= L; i++) {
    uint8_t x = s[i];
    if (x >= Kp) {
      if (lane == 0) {
        scores[bi] = 0.0f;
        statuses[bi] = eslEINVAL;
      }
      return;
    }

    xB = (int16_t) __shfl_sync(mask, (int) xB, 0);
    int16_t mpv = prev[((Q - 1) + lane * Q) * 3 + 0];
    int16_t dpv = prev[((Q - 1) + lane * Q) * 3 + 1];
    int16_t ipv = prev[((Q - 1) + lane * Q) * 3 + 2];
    mpv = (int16_t) __shfl_up_sync(mask, (int) mpv, 1);
    dpv = (int16_t) __shfl_up_sync(mask, (int) dpv, 1);
    ipv = (int16_t) __shfl_up_sync(mask, (int) ipv, 1);
    if (lane == 0) mpv = dpv = ipv = -32768;

    int16_t dcv = -32768;
    int16_t xE_lane = -32768;
    int16_t dmax_lane = -32768;
    for (int q = 0; q < Q; q++) {
      int cell = (q + lane * Q) * 3;
      int16_t sv = i16_add_sat(xB, twv[vit_twv_idx(p7O_BM, q, lane, Q)]);
      int16_t cand = i16_add_sat(mpv, twv[vit_twv_idx(p7O_MM, q, lane, Q)]);
      if (cand > sv) sv = cand;
      cand = i16_add_sat(ipv, twv[vit_twv_idx(p7O_IM, q, lane, Q)]);
      if (cand > sv) sv = cand;
      cand = i16_add_sat(dpv, twv[vit_twv_idx(p7O_DM, q, lane, Q)]);
      if (cand > sv) sv = cand;
      sv = i16_add_sat(sv, rwv[((int) x * Q + q) * 8 + lane]);
      if (sv > xE_lane) xE_lane = sv;

      mpv = prev[cell + 0];
      dpv = prev[cell + 1];
      ipv = prev[cell + 2];

      curr[cell + 0] = sv;
      curr[cell + 1] = dcv;
      dcv = i16_add_sat(sv, twv[vit_twv_idx(p7O_MD, q, lane, Q)]);
      if (dcv > dmax_lane) dmax_lane = dcv;
      cand = i16_add_sat(mpv, twv[vit_twv_idx(p7O_MI, q, lane, Q)]);
      sv = i16_add_sat(ipv, twv[vit_twv_idx(p7O_II, q, lane, Q)]);
      curr[cell + 2] = cand > sv ? cand : sv;
    }

    int xEi = (int) xE_lane;
    int dmaxi = (int) dmax_lane;
    for (int off = 4; off > 0; off >>= 1) {
      int other = __shfl_down_sync(mask, xEi, off);
      if (lane + off < 8 && other > xEi) xEi = other;
      other = __shfl_down_sync(mask, dmaxi, off);
      if (lane + off < 8 && other > dmaxi) dmaxi = other;
    }

    if (lane == 0) {
      int16_t xE = (int16_t) xEi;
      if (xE >= 32767) overflow = 1;
      xN = i16_add_sat(xN, xw_n_loop);
      int16_t c1 = i16_add_sat(xC, xw_c_loop);
      int16_t c2 = i16_add_sat(xE, xw_e_move);
      xC = c1 > c2 ? c1 : c2;
      int16_t j1 = i16_add_sat(xJ, xw_j_loop);
      int16_t j2 = i16_add_sat(xE, xw_e_loop);
      xJ = j1 > j2 ? j1 : j2;
      int16_t b1 = i16_add_sat(xJ, xw_move);
      int16_t b2 = i16_add_sat(xN, xw_move);
      xB = b1 > b2 ? b1 : b2;
    }
    xB = (int16_t) __shfl_sync(mask, (int) xB, 0);

    int dmax_all = __shfl_sync(mask, dmaxi, 0);
    if (dmax_all + ddbound_w > xB) {
      dcv = (int16_t) __shfl_up_sync(mask, (int) dcv, 1);
      if (lane == 0) dcv = -32768;
      for (int q = 0; q < Q; q++) {
        int cell = (q + lane * Q) * 3;
        if (dcv > curr[cell + 1]) curr[cell + 1] = dcv;
        dcv = i16_add_sat(curr[cell + 1], twv[vit_twv_idx(p7O_DD, q, lane, Q)]);
      }
      int completed;
      do {
        completed = 1;
        dcv = (int16_t) __shfl_up_sync(mask, (int) dcv, 1);
        if (lane == 0) dcv = -32768;
        for (int q = 0; q < Q; q++) {
          int cell = (q + lane * Q) * 3;
          int gt = (dcv > curr[cell + 1]);
          if (!__any_sync(mask, gt)) {
            completed = 0;
            break;
          }
          if (gt) {
            curr[cell + 1] = dcv;
          }
          dcv = i16_add_sat(curr[cell + 1], twv[vit_twv_idx(p7O_DD, q, lane, Q)]);
        }
      } while (__all_sync(mask, completed));
    } else {
      dcv = (int16_t) __shfl_up_sync(mask, (int) dcv, 1);
      if (lane == 0) dcv = -32768;
      curr[lane * Q * 3 + 1] = dcv;
    }

    int16_t *tmp = prev;
    prev = curr;
    curr = tmp;
  }

  if (lane == 0) {
    if (overflow) {
      scores[bi] = eslINFINITY;
      statuses[bi] = eslERANGE;
    } else if (xC > -32768) {
      scores[bi] = ((float) xC + (float) xw_move - (float) base_w) / scale_w - 3.0f;
      statuses[bi] = eslOK;
    } else {
      scores[bi] = -eslINFINITY;
      statuses[bi] = eslOK;
    }
  }
}

__global__ static void
cuda_forward_score_kernel(const uint8_t *dsq, const int *offsets, const int *lengths,
                          const int *seqidx, int nidx,
                          const float *rfv, const float *tfv, int M, int Q, int Kp,
                          float xf_e_loop, float xf_e_move, float xf_n_loop_base, float xf_n_move_base,
                          float xf_c_loop_base, float xf_c_move_base, float xf_j_loop_base, float xf_j_move_base,
                          float nj, float *scores, int *statuses)
{
  extern __shared__ float fwd_mem[];
  int N = Q * 4;
  float *prev = fwd_mem;
  float *curr = prev + (size_t) N * 3;
  int bi = blockIdx.x;
  int lane = threadIdx.x & 31;
  unsigned int mask = 0x0f;

  if (bi >= nidx || lane >= 4) return;

  int si = seqidx ? seqidx[bi] : bi;
  int L = lengths[si];
  const uint8_t *s = dsq + offsets[si];
  float pmove = (2.0f + nj) / ((float) L + 2.0f + nj);
  float ploop = 1.0f - pmove;
  float xf_n_loop = xf_n_loop_base >= 0.0f ? xf_n_loop_base : ploop;
  float xf_n_move = xf_n_move_base >= 0.0f ? xf_n_move_base : pmove;
  float xf_c_loop = xf_c_loop_base >= 0.0f ? xf_c_loop_base : ploop;
  float xf_c_move = xf_c_move_base >= 0.0f ? xf_c_move_base : pmove;
  float xf_j_loop = xf_j_loop_base >= 0.0f ? xf_j_loop_base : ploop;
  float xf_j_move = xf_j_move_base >= 0.0f ? xf_j_move_base : pmove;

  for (int q = 0; q < Q; q++) {
    int cell = (q + lane * Q) * 3;
    prev[cell + 0] = 0.0f;
    prev[cell + 1] = 0.0f;
    prev[cell + 2] = 0.0f;
  }

  float xN = 1.0f;
  float xJ = 0.0f;
  float xB = xf_n_move;
  float xC = 0.0f;
  float totscale = 0.0f;

  for (int i = 1; i <= L; i++) {
    uint8_t x = s[i];

    if (x >= Kp) {
      if (lane == 0) {
        scores[bi] = 0.0f;
        statuses[bi] = eslEINVAL;
      }
      return;
    }

    xB = __shfl_sync(mask, xB, 0);
    float mpv = prev[((Q - 1) + lane * Q) * 3 + 0];
    float dpv = prev[((Q - 1) + lane * Q) * 3 + 1];
    float ipv = prev[((Q - 1) + lane * Q) * 3 + 2];
    mpv = __shfl_up_sync(mask, mpv, 1);
    dpv = __shfl_up_sync(mask, dpv, 1);
    ipv = __shfl_up_sync(mask, ipv, 1);
    if (lane == 0) mpv = dpv = ipv = 0.0f;

    float dcv = 0.0f;
    float xE_lane = 0.0f;
    for (int q = 0; q < Q; q++) {
      int cell = (q + lane * Q) * 3;
      float m = xB * tfv[fwd_tfv_idx(p7O_BM, q, lane, Q)];
      m += mpv * tfv[fwd_tfv_idx(p7O_MM, q, lane, Q)];
      m += ipv * tfv[fwd_tfv_idx(p7O_IM, q, lane, Q)];
      m += dpv * tfv[fwd_tfv_idx(p7O_DM, q, lane, Q)];
      m *= rfv[((x * Q + q) * 4) + lane];
      xE_lane += m;

      mpv = prev[cell + 0];
      dpv = prev[cell + 1];
      ipv = prev[cell + 2];

      curr[cell + 0] = m;
      curr[cell + 1] = dcv;
      dcv = m * tfv[fwd_tfv_idx(p7O_MD, q, lane, Q)];
      curr[cell + 2] = mpv * tfv[fwd_tfv_idx(p7O_MI, q, lane, Q)]
                     + ipv * tfv[fwd_tfv_idx(p7O_II, q, lane, Q)];
    }

    dcv = __shfl_up_sync(mask, dcv, 1);
    if (lane == 0) dcv = 0.0f;
    for (int q = 0; q < Q; q++) {
      int cell = (q + lane * Q) * 3;
      float d = curr[cell + 1] + dcv;
      curr[cell + 1] = d;
      dcv = d * tfv[fwd_tfv_idx(p7O_DD, q, lane, Q)];
    }

    for (int j = 1; j < 4; j++) {
      int any = 0;
      dcv = __shfl_up_sync(mask, dcv, 1);
      if (lane == 0) dcv = 0.0f;
      for (int q = 0; q < Q; q++) {
        int cell = (q + lane * Q) * 3;
        float old = curr[cell + 1];
        float d = old + dcv;
        curr[cell + 1] = d;
        if (d > old) any = 1;
        dcv = dcv * tfv[fwd_tfv_idx(p7O_DD, q, lane, Q)];
      }
      if (M >= 100 && !__any_sync(mask, any)) break;
    }

    for (int q = 0; q < Q; q++) {
      int cell = (q + lane * Q) * 3;
      xE_lane += curr[cell + 1];
    }
    float xE = xE_lane;
    xE += __shfl_down_sync(mask, xE, 2);
    xE += __shfl_down_sync(mask, xE, 1);

    if (lane == 0) {
      xN = xN * xf_n_loop;
      xC = (xC * xf_c_loop) + (xE * xf_e_move);
      xJ = (xJ * xf_j_loop) + (xE * xf_e_loop);
      xB = (xJ * xf_j_move) + (xN * xf_n_move);

      if (xE > 1.0e4f) {
        float scale_inv = 1.0f / xE;
        xN *= scale_inv;
        xC *= scale_inv;
        xJ *= scale_inv;
        xB *= scale_inv;
        totscale += logf(xE);
      }
    }

    float scale_inv = 1.0f;
    int scale_row = 0;
    if (lane == 0 && xE > 1.0e4f) {
      scale_inv = 1.0f / xE;
      scale_row = 1;
    }
    scale_inv = __shfl_sync(mask, scale_inv, 0);
    scale_row = __shfl_sync(mask, scale_row, 0);
    if (scale_row) {
      for (int q = 0; q < Q; q++) {
        int cell = (q + lane * Q) * 3;
        curr[cell + 0] *= scale_inv;
        curr[cell + 1] *= scale_inv;
        curr[cell + 2] *= scale_inv;
      }
    }

    float *tmp = prev;
    prev = curr;
    curr = tmp;
  }

  if (lane == 0) {
    if (isnan(xC) || (L > 0 && xC == 0.0f) || isinf(xC)) {
      scores[bi] = 0.0f;
      statuses[bi] = eslERANGE;
    } else {
      scores[bi] = totscale + logf(xC * xf_c_move);
      statuses[bi] = eslOK;
    }
  }
}

__global__ static void
cuda_forward_score_prefix_kernel(const uint8_t *dsq, const int *offsets, const int *lengths,
                                 const int *seqidx, int nidx,
                                 const float *rfv, const float *tfv, int M, int Q, int Kp,
                                 float xf_e_loop, float xf_e_move, float xf_n_loop_base, float xf_n_move_base,
                                 float xf_c_loop_base, float xf_c_move_base, float xf_j_loop_base, float xf_j_move_base,
                                 float nj, float *scores, int *statuses)
{
  extern __shared__ float fwd_scan_mem[];
  int N = Q * 4;
  int T = blockDim.x;
  int tid = threadIdx.x;
  int bi = blockIdx.x;
  float *prev  = fwd_scan_mem;
  float *curr  = prev + (size_t) N * 3;
  float *bcoef = curr + (size_t) N * 3;
  float *scanA = bcoef + N;
  float *scanB = scanA + T;
  __shared__ float sxB;
  __shared__ float sxN;
  __shared__ float sxJ;
  __shared__ float sxC;
  __shared__ float stotscale;
  __shared__ float sscale_inv;
  __shared__ int sscale_row;

  if (bi >= nidx) return;

  int si = seqidx ? seqidx[bi] : bi;
  int L = lengths[si];
  const uint8_t *s = dsq + offsets[si];
  float pmove = (2.0f + nj) / ((float) L + 2.0f + nj);
  float ploop = 1.0f - pmove;
  float xf_n_loop = xf_n_loop_base >= 0.0f ? xf_n_loop_base : ploop;
  float xf_n_move = xf_n_move_base >= 0.0f ? xf_n_move_base : pmove;
  float xf_c_loop = xf_c_loop_base >= 0.0f ? xf_c_loop_base : ploop;
  float xf_c_move = xf_c_move_base >= 0.0f ? xf_c_move_base : pmove;
  float xf_j_loop = xf_j_loop_base >= 0.0f ? xf_j_loop_base : ploop;
  float xf_j_move = xf_j_move_base >= 0.0f ? xf_j_move_base : pmove;

  for (int cell = tid; cell < N * 3; cell += blockDim.x) prev[cell] = 0.0f;
  if (tid == 0) {
    sxN = 1.0f;
    sxJ = 0.0f;
    sxB = xf_n_move;
    sxC = 0.0f;
    stotscale = 0.0f;
  }
  __syncthreads();

  for (int i = 1; i <= L; i++) {
    uint8_t x = s[i];
    if (x >= Kp) {
      if (tid == 0) {
        scores[bi] = 0.0f;
        statuses[bi] = eslEINVAL;
      }
      return;
    }

    float xB = sxB;
    for (int c = tid; c < N; c += blockDim.x) {
      int q = c % Q;
      int lane = c / Q;
      int cell = c * 3;
      float mpv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 0];
      float dpv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 1];
      float ipv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 2];
      float m = xB * tfv[fwd_tfv_idx(p7O_BM, q, lane, Q)];
      m += mpv * tfv[fwd_tfv_idx(p7O_MM, q, lane, Q)];
      m += ipv * tfv[fwd_tfv_idx(p7O_IM, q, lane, Q)];
      m += dpv * tfv[fwd_tfv_idx(p7O_DM, q, lane, Q)];
      m *= rfv[((x * Q + q) * 4) + lane];
      curr[cell + 0] = m;
      curr[cell + 2] = prev[cell + 0] * tfv[fwd_tfv_idx(p7O_MI, q, lane, Q)]
                     + prev[cell + 2] * tfv[fwd_tfv_idx(p7O_II, q, lane, Q)];
      if (c == 0) {
        curr[cell + 1] = 0.0f;
        bcoef[c] = 0.0f;
      } else {
        int pq = (c - 1) % Q;
        int plane = (c - 1) / Q;
        curr[cell + 1] = curr[(c - 1) * 3 + 0] * tfv[fwd_tfv_idx(p7O_MD, pq, plane, Q)];
        bcoef[c] = tfv[fwd_tfv_idx(p7O_DD, pq, plane, Q)];
      }
    }
    __syncthreads();

    if (tid < T) {
      int c0 = tid * 2;
      int c1 = c0 + 1;
      if (c0 < N) {
        float a0 = curr[c0 * 3 + 1];
        float b0 = bcoef[c0];
        float a1 = (c1 < N) ? curr[c1 * 3 + 1] : 0.0f;
        float b1 = (c1 < N) ? bcoef[c1] : 1.0f;
        scanA[tid] = a1 + b1 * a0;
        scanB[tid] = b1 * b0;
      } else {
        scanA[tid] = 0.0f;
        scanB[tid] = 1.0f;
      }
    }
    __syncthreads();

    for (int off = 1; off < T; off <<= 1) {
      float a_prev = 0.0f;
      float b_prev = 1.0f;
      float a_cur = 0.0f;
      float b_cur = 1.0f;
      if (tid < T) {
        a_cur = scanA[tid];
        b_cur = scanB[tid];
        if (tid >= off) {
          a_prev = scanA[tid - off];
          b_prev = scanB[tid - off];
        }
      }
      __syncthreads();
      if (tid < T && tid >= off) {
        scanA[tid] = a_cur + b_cur * a_prev;
        scanB[tid] = b_cur * b_prev;
      }
      __syncthreads();
    }

    if (tid < T) {
      int c0 = tid * 2;
      int c1 = c0 + 1;
      if (c0 < N) {
        float prefixA = (tid == 0) ? 0.0f : scanA[tid - 1];
        float a0 = curr[c0 * 3 + 1];
        float d0 = a0 + bcoef[c0] * prefixA;
        curr[c0 * 3 + 1] = d0;
        if (c1 < N) {
          float a1 = curr[c1 * 3 + 1];
          curr[c1 * 3 + 1] = a1 + bcoef[c1] * d0;
        }
      }
    }
    __syncthreads();

    float partial = 0.0f;
    for (int c = tid; c < N; c += blockDim.x) partial += curr[c * 3 + 0] + curr[c * 3 + 1];
    if (tid < T) scanA[tid] = partial;
    __syncthreads();
    for (int off = T >> 1; off > 0; off >>= 1) {
      if (tid < off) scanA[tid] += scanA[tid + off];
      __syncthreads();
    }

    if (tid == 0) {
      float xE = scanA[0];
      sxN = sxN * xf_n_loop;
      sxC = (sxC * xf_c_loop) + (xE * xf_e_move);
      sxJ = (sxJ * xf_j_loop) + (xE * xf_e_loop);
      sxB = (sxJ * xf_j_move) + (sxN * xf_n_move);
      if (xE > 1.0e4f) {
        sscale_inv = 1.0f / xE;
        sxN *= sscale_inv;
        sxC *= sscale_inv;
        sxJ *= sscale_inv;
        sxB *= sscale_inv;
        stotscale += logf(xE);
        sscale_row = 1;
      } else {
        sscale_row = 0;
      }
    }
    __syncthreads();

    if (sscale_row) {
      for (int cell = tid; cell < N * 3; cell += blockDim.x) curr[cell] *= sscale_inv;
    }
    __syncthreads();

    float *tmp = prev;
    prev = curr;
    curr = tmp;
  }

  if (tid == 0) {
    if (isnan(sxC) || (L > 0 && sxC == 0.0f) || isinf(sxC)) {
      scores[bi] = 0.0f;
      statuses[bi] = eslERANGE;
    } else {
      scores[bi] = stotscale + logf(sxC * xf_c_move);
      statuses[bi] = eslOK;
    }
  }
}

__global__ static void
cuda_msv_kernel(const uint8_t *dsq, int L, const uint8_t *rbv, int M, int Q, int Kp,
                uint8_t tbm_b, uint8_t tec_b, uint8_t tjb_b, uint8_t base_b, uint8_t bias_b,
                int *raw_sc, int *overflow)
{
  extern __shared__ uint8_t mem[];
  uint8_t *prev = mem;
  uint8_t *curr = mem + M + 1;

  int tid = threadIdx.x;
  for (int k = tid; k <= M; k += blockDim.x) prev[k] = 0;
  if (tid == 0) *overflow = 0;
  __syncthreads();

  uint8_t xJ = 0;
  uint8_t tjbm_b = (uint8_t) (tjb_b + tbm_b);
  uint8_t xB = u8_sub_sat(base_b, tjbm_b);

  for (int i = 1; i <= L; i++) {
    uint8_t xE_local = 0;
    uint8_t x = dsq[i];

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
      if (u8_add_sat(v, bias_b) == 255) *overflow = 1;
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

  if (tid == 0) *raw_sc = (int) xJ;
}

__global__ static void
cuda_msv_batch_kernel(const uint8_t *dsq, const int *offsets, const int *lengths, const uint8_t *tjb_by_seq, int nseq,
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

__global__ static void
cuda_bias_filter_kernel(const uint8_t *dsq, const int *offsets, const int *lengths, int nseq,
                        const float *pi, const float *t, const float *eo,
                        float *filtersc)
{
  int seq = blockIdx.x * blockDim.x + threadIdx.x;
  if (seq >= nseq) return;

  const uint8_t *sdsq = dsq + offsets[seq];
  int L = lengths[seq];
  float p0, p1s, n0, n1, maxv;
  float len_p1 = (float) L / (float) (L + 1);
  float sc = 0.0f;

  if (L == 0) {
    filtersc[seq] = logf(pi[2]);
    return;
  }

  p0 = eo[(int) sdsq[1] * 2 + 0] * pi[0];
  p1s = eo[(int) sdsq[1] * 2 + 1] * pi[1];
  maxv = fmaxf(fmaxf(p0, p1s), 0.0f);
  p0 /= maxv;
  p1s /= maxv;
  sc += logf(maxv);

  for (int i = 2; i <= L; i++) {
    uint8_t x = sdsq[i];
    n0 = (p0 * len_p1 + p1s * t[3]) * eo[(int) x * 2 + 0];
    n1 = (p0 * (1.0f - len_p1) + p1s * t[4]) * eo[(int) x * 2 + 1];
    maxv = fmaxf(fmaxf(n0, n1), 0.0f);
    p0 = n0 / maxv;
    p1s = n1 / maxv;
    sc += logf(maxv);
  }

  sc += logf(p0 * t[2] + p1s * t[5]);
  filtersc[seq] = sc + (float) L * logf(len_p1) + logf(1.0f - len_p1);
}

extern "C" int
p7_cuda_Available(char *errbuf, int errbuf_size)
{
  int n = 0;
  cudaError_t cerr = cudaGetDeviceCount(&n);
  if (cerr != cudaSuccess)
    return cuda_status(cerr, errbuf, errbuf_size, "cudaGetDeviceCount");
  if (n <= 0) {
    if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "no usable CUDA device found");
    return eslENOTFOUND;
  }
  return eslOK;
}

extern "C" int
p7_cuda_engine_Create(int device_id, P7_CUDA_ENGINE **ret_engine, char *errbuf, int errbuf_size)
{
  P7_CUDA_ENGINE *engine = NULL;
  int n = 0;
  int status;

  if (ret_engine) *ret_engine = NULL;
  if ((status = p7_cuda_Available(errbuf, errbuf_size)) != eslOK) return status;
  if ((status = cuda_status(cudaGetDeviceCount(&n), errbuf, errbuf_size, "cudaGetDeviceCount")) != eslOK) return status;
  if (device_id < 0 || device_id >= n) {
    if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "CUDA device %d is not available", device_id);
    return eslEINVAL;
  }
  if ((status = cuda_status(cudaSetDevice(device_id), errbuf, errbuf_size, "cudaSetDevice")) != eslOK) return status;

  engine = (P7_CUDA_ENGINE *) calloc(1, sizeof(*engine));
  if (!engine) return eslEMEM;
  engine->device_id = device_id;

  if ((status = cuda_status(cudaMalloc((void **) &engine->d_raw, sizeof(int)), errbuf, errbuf_size, "cudaMalloc(raw)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMalloc((void **) &engine->d_overflow, sizeof(int)), errbuf, errbuf_size, "cudaMalloc(overflow)")) != eslOK) goto ERROR;

  *ret_engine = engine;
  return eslOK;

ERROR:
  p7_cuda_engine_Destroy(engine);
  return status;
}

extern "C" void
p7_cuda_engine_Destroy(P7_CUDA_ENGINE *engine)
{
  if (!engine) return;
  cudaSetDevice(engine->device_id);
  if (engine->d_dsq)      cudaFree(engine->d_dsq);
  if (engine->h_dsq)      cudaFreeHost(engine->h_dsq);
  if (engine->d_offsets)  cudaFree(engine->d_offsets);
  if (engine->d_lengths)  cudaFree(engine->d_lengths);
  if (engine->d_tjb_by_seq) cudaFree(engine->d_tjb_by_seq);
  if (engine->d_raw)      cudaFree(engine->d_raw);
  if (engine->d_overflow) cudaFree(engine->d_overflow);
  if (engine->d_bias_filtersc) cudaFree(engine->d_bias_filtersc);
  if (engine->d_bias_pi)       cudaFree(engine->d_bias_pi);
  if (engine->d_bias_t)        cudaFree(engine->d_bias_t);
  if (engine->d_bias_eo)       cudaFree(engine->d_bias_eo);
  if (engine->d_fwd_scores)    cudaFree(engine->d_fwd_scores);
  if (engine->d_fwd_statuses)  cudaFree(engine->d_fwd_statuses);
  if (engine->d_fwd_seqidx)    cudaFree(engine->d_fwd_seqidx);
  if (engine->d_vit_scores)    cudaFree(engine->d_vit_scores);
  if (engine->d_vit_statuses)  cudaFree(engine->d_vit_statuses);
  if (engine->d_vit_seqidx)    cudaFree(engine->d_vit_seqidx);
  if (engine->d_fwd_prev)      cudaFree(engine->d_fwd_prev);
  if (engine->d_fwd_curr)      cudaFree(engine->d_fwd_curr);
  free(engine);
}

extern "C" void
p7_cuda_engine_GetStats(const P7_CUDA_ENGINE *engine, P7_CUDA_MSV_STATS *stats)
{
  if (!engine || !stats) return;
  *stats = engine->stats;
}

extern "C" int
p7_cuda_msvprofile_Create(const P7_OPROFILE *om, P7_CUDA_MSVPROFILE **ret_cuom, char *errbuf, int errbuf_size)
{
  P7_CUDA_MSVPROFILE *cuom = NULL;
  int status;
  size_t nbytes;
  size_t fbytes;

  if (ret_cuom) *ret_cuom = NULL;
  cuom = (P7_CUDA_MSVPROFILE *) calloc(1, sizeof(*cuom));
  if (!cuom) return eslEMEM;

  cuom->M       = om->M;
  cuom->Q       = p7O_NQB(om->M);
  cuom->Qf      = p7O_NQF(om->M);
  cuom->Qw      = p7O_NQW(om->M);
  cuom->Kp      = om->abc->Kp;
  cuom->tbm_b   = om->tbm_b;
  cuom->tec_b   = om->tec_b;
  cuom->tjb_b   = om->tjb_b;
  cuom->scale_b = om->scale_b;
  cuom->base_b  = om->base_b;
  cuom->bias_b  = om->bias_b;
  cuom->xf_e_loop = om->xf[p7O_E][p7O_LOOP];
  cuom->xf_e_move = om->xf[p7O_E][p7O_MOVE];
  cuom->xf_n_loop = -1.0f;
  cuom->xf_n_move = -1.0f;
  cuom->xf_c_loop = -1.0f;
  cuom->xf_c_move = -1.0f;
  cuom->xf_j_loop = -1.0f;
  cuom->xf_j_move = -1.0f;
  cuom->nj        = om->nj;
  cuom->xw_n_loop = om->xw[p7O_N][p7O_LOOP];
  cuom->xw_n_move = om->xw[p7O_N][p7O_MOVE];
  cuom->xw_c_loop = om->xw[p7O_C][p7O_LOOP];
  cuom->xw_c_move = om->xw[p7O_C][p7O_MOVE];
  cuom->xw_j_loop = om->xw[p7O_J][p7O_LOOP];
  cuom->xw_j_move = om->xw[p7O_J][p7O_MOVE];
  cuom->xw_e_loop = om->xw[p7O_E][p7O_LOOP];
  cuom->xw_e_move = om->xw[p7O_E][p7O_MOVE];
  cuom->scale_w   = om->scale_w;
  cuom->base_w    = om->base_w;
  cuom->ddbound_w = om->ddbound_w;

  nbytes = (size_t) cuom->Kp * (size_t) cuom->Q * 16;
  if ((status = cuda_status(cudaMalloc((void **) &cuom->d_rbv, nbytes), errbuf, errbuf_size, "cudaMalloc(profile)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(cuom->d_rbv, (const void *) om->rbv[0], nbytes, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(profile)")) != eslOK) goto ERROR;

  fbytes = sizeof(float) * (size_t) cuom->Kp * (size_t) cuom->Qf * 4;
  if ((status = cuda_status(cudaMalloc((void **) &cuom->d_rfv, fbytes), errbuf, errbuf_size, "cudaMalloc(fwd emissions)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(cuom->d_rfv, (const void *) om->rfv[0], fbytes, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(fwd emissions)")) != eslOK) goto ERROR;
  fbytes = sizeof(float) * (size_t) p7O_NTRANS * (size_t) cuom->Qf * 4;
  if ((status = cuda_status(cudaMalloc((void **) &cuom->d_tfv, fbytes), errbuf, errbuf_size, "cudaMalloc(fwd transitions)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(cuom->d_tfv, (const void *) om->tfv, fbytes, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(fwd transitions)")) != eslOK) goto ERROR;
  fbytes = sizeof(int16_t) * (size_t) cuom->Kp * (size_t) cuom->Qw * 8;
  if ((status = cuda_status(cudaMalloc((void **) &cuom->d_rwv, fbytes), errbuf, errbuf_size, "cudaMalloc(vit emissions)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(cuom->d_rwv, (const void *) om->rwv[0], fbytes, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(vit emissions)")) != eslOK) goto ERROR;
  fbytes = sizeof(int16_t) * (size_t) p7O_NTRANS * (size_t) cuom->Qw * 8;
  if ((status = cuda_status(cudaMalloc((void **) &cuom->d_twv, fbytes), errbuf, errbuf_size, "cudaMalloc(vit transitions)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(cuom->d_twv, (const void *) om->twv, fbytes, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(vit transitions)")) != eslOK) goto ERROR;

  *ret_cuom = cuom;
  return eslOK;

ERROR:
  p7_cuda_msvprofile_Destroy(cuom);
  return status;
}

static uint8_t
cuda_unbiased_byteify(const P7_CUDA_MSVPROFILE *cuom, float sc)
{
  sc = -1.0f * roundf(cuom->scale_b * sc);
  return (sc > 255.0f) ? 255 : (uint8_t) sc;
}

static int
cuda_msvprofile_GrowLengthLookup(P7_CUDA_MSVPROFILE *cuom, int L)
{
  uint8_t *tmp;
  int old_alloc;

  if (L < cuom->tjb_len_alloc) return eslOK;
  old_alloc = cuom->tjb_len_alloc;
  tmp = (uint8_t *) realloc(cuom->h_tjb_by_len, sizeof(uint8_t) * (L + 1));
  if (!tmp) return eslEMEM;
  cuom->h_tjb_by_len = tmp;
  cuom->tjb_len_alloc = L + 1;
  for (int i = old_alloc; i <= L; i++)
    cuom->h_tjb_by_len[i] = cuda_unbiased_byteify(cuom, logf(3.0f / (float) (i + 3)));
  return eslOK;
}

extern "C" int
p7_cuda_MSVFilterDsqdataChunk(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
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
  cudaEvent_t h2d0, h2d1, k0, k1, d2h0, d2h1;

  if (!engine || !cuom || !chu || !scores || !statuses) return eslEINVAL;
  nseq = chu->N;
  if (nseq <= 0) return eslOK;
  if (shmem > 96 * 1024) {
    if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "CUDA MSV profile M=%d exceeds v1 shared-memory limit", cuom->M);
    return eslERANGE;
  }

  h_offsets = (int *) malloc(sizeof(int) * nseq);
  h_lengths = (int *) malloc(sizeof(int) * nseq);
  h_tjb_by_seq = (uint8_t *) malloc(sizeof(uint8_t) * nseq);
  h_raw = (int *) malloc(sizeof(int) * nseq);
  h_overflow = (int *) malloc(sizeof(int) * nseq);
  if (!h_offsets || !h_lengths || !h_tjb_by_seq || !h_raw || !h_overflow) { status = eslEMEM; goto ERROR; }

  for (int i = 0; i < nseq; i++) {
    h_offsets[i] = total;
    if (chu->L[i] > INT32_MAX) {
      if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "dsqdata sequence length exceeds CUDA MSV v1 limit");
      status = eslERANGE;
      goto ERROR;
    }
    h_lengths[i] = (int) chu->L[i];
    if ((status = cuda_msvprofile_GrowLengthLookup((P7_CUDA_MSVPROFILE *) cuom, h_lengths[i])) != eslOK) goto ERROR;
    h_tjb_by_seq[i] = cuom->h_tjb_by_len[h_lengths[i]];
    total += h_lengths[i] + 2;
  }

  if (engine->dsq_alloc < total) {
    if (engine->d_dsq) cudaFree(engine->d_dsq);
    if (engine->h_dsq) cudaFreeHost(engine->h_dsq);
    engine->d_dsq = NULL;
    engine->h_dsq = NULL;
    engine->dsq_alloc = 0;
    engine->h_dsq_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_dsq, total), errbuf, errbuf_size, "cudaMalloc(batch dsq)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMallocHost((void **) &engine->h_dsq, total), errbuf, errbuf_size, "cudaMallocHost(batch dsq)")) != eslOK) goto ERROR;
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
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(offsets)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_lengths, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(lengths)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_tjb_by_seq, sizeof(uint8_t) * nseq), errbuf, errbuf_size, "cudaMalloc(tjb_by_seq)")) != eslOK) goto ERROR;
    engine->meta_alloc = nseq;
  }
  if (engine->result_alloc < nseq) {
    if (engine->d_raw) cudaFree(engine->d_raw);
    if (engine->d_overflow) cudaFree(engine->d_overflow);
    engine->d_raw = NULL;
    engine->d_overflow = NULL;
    engine->result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_raw, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(raw batch)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_overflow, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(overflow batch)")) != eslOK) goto ERROR;
    engine->result_alloc = nseq;
  }

  cudaEventCreate(&h2d0);
  cudaEventCreate(&h2d1);
  cudaEventCreate(&k0);
  cudaEventCreate(&k1);
  cudaEventCreate(&d2h0);
  cudaEventCreate(&d2h1);

  cudaEventRecord(h2d0);
  for (int i = 0; i < nseq; i++) {
    memcpy(engine->h_dsq + h_offsets[i], chu->dsq[i], h_lengths[i] + 2);
  }
  if ((status = cuda_status(cudaMemcpy(engine->d_dsq, engine->h_dsq, total, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(batch dsq)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_offsets, h_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(offsets)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_lengths, h_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(lengths)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_tjb_by_seq, h_tjb_by_seq, sizeof(uint8_t) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(tjb_by_seq)")) != eslOK) goto CUDA_ERROR;
  cudaEventRecord(h2d1);
  cudaEventSynchronize(h2d1);

  cudaEventRecord(k0);
  cuda_msv_batch_kernel<<<nseq, P7_CUDA_MSV_BLOCK_THREADS, shmem>>>(engine->d_dsq, engine->d_offsets, engine->d_lengths, engine->d_tjb_by_seq, nseq,
                                                                    cuom->d_rbv, cuom->M, cuom->Q, cuom->Kp,
                                                                    cuom->tbm_b, cuom->tec_b, cuom->tjb_b,
                                                                    cuom->base_b, cuom->bias_b,
                                                                    engine->d_raw, engine->d_overflow);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_msv_batch_kernel launch")) != eslOK) goto CUDA_ERROR;
  cudaEventRecord(k1);
  cudaEventSynchronize(k1);

  cudaEventRecord(d2h0);
  if ((status = cuda_status(cudaMemcpy(h_raw, engine->d_raw, sizeof(int) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(raw batch)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(h_overflow, engine->d_overflow, sizeof(int) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(overflow batch)")) != eslOK) goto CUDA_ERROR;
  cudaEventRecord(d2h1);
  cudaEventSynchronize(d2h1);

  engine->batch_owner = chu;
  engine->batch_nseq  = nseq;
  engine->batch_total = total;
  engine->stats.h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.d2h_seconds    += elapsed_seconds(d2h0, d2h1);
  engine->stats.nseqs          += nseq;
  engine->stats.nres           += total - (2 * nseq);
  engine->stats.nbatches       += 1;

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

CUDA_ERROR:
  cudaEventDestroy(h2d0);
  cudaEventDestroy(h2d1);
  cudaEventDestroy(k0);
  cudaEventDestroy(k1);
  cudaEventDestroy(d2h0);
  cudaEventDestroy(d2h1);
ERROR:
  free(h_offsets);
  free(h_lengths);
  free(h_tjb_by_seq);
  free(h_raw);
  free(h_overflow);
  return status;
}

extern "C" int
p7_cuda_BiasFilterDsqdataChunk(P7_CUDA_ENGINE *engine, const P7_BG *bg,
                               ESL_DSQDATA_CHUNK *chu, float *filtersc,
                               char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int nseq;
  int total = 0;
  int *h_offsets = NULL;
  int *h_lengths = NULL;
  float h_pi[3];
  float h_t[6];
  size_t eo_bytes;
  int reuse_batch = FALSE;
  cudaEvent_t h2d0, h2d1, k0, k1, d2h0, d2h1;

  if (!engine || !bg || !bg->fhmm || !chu || !filtersc) return eslEINVAL;
  nseq = chu->N;
  if (nseq <= 0) return eslOK;

  h_offsets = (int *) malloc(sizeof(int) * nseq);
  h_lengths = (int *) malloc(sizeof(int) * nseq);
  if (!h_offsets || !h_lengths) { status = eslEMEM; goto ERROR; }

  for (int i = 0; i < nseq; i++) {
    h_offsets[i] = total;
    if (chu->L[i] > INT32_MAX) {
      if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "dsqdata sequence length exceeds CUDA bias v1 limit");
      status = eslERANGE;
      goto ERROR;
    }
    h_lengths[i] = (int) chu->L[i];
    total += h_lengths[i] + 2;
  }
  reuse_batch = (engine->batch_owner == chu && engine->batch_nseq == nseq && engine->batch_total == total);
  engine->batch_owner = NULL;
  engine->batch_nseq  = 0;
  engine->batch_total = 0;

  if (engine->dsq_alloc < total) {
    if (engine->d_dsq) cudaFree(engine->d_dsq);
    if (engine->h_dsq) cudaFreeHost(engine->h_dsq);
    engine->d_dsq = NULL;
    engine->h_dsq = NULL;
    engine->dsq_alloc = 0;
    engine->h_dsq_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_dsq, total), errbuf, errbuf_size, "cudaMalloc(batch dsq)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMallocHost((void **) &engine->h_dsq, total), errbuf, errbuf_size, "cudaMallocHost(batch dsq)")) != eslOK) goto ERROR;
    engine->dsq_alloc = total;
    engine->h_dsq_alloc = total;
    reuse_batch = FALSE;
  }
  if (engine->meta_alloc < nseq) {
    if (engine->d_offsets) cudaFree(engine->d_offsets);
    if (engine->d_lengths) cudaFree(engine->d_lengths);
    if (engine->d_tjb_by_seq) cudaFree(engine->d_tjb_by_seq);
    engine->d_offsets = NULL;
    engine->d_lengths = NULL;
    engine->d_tjb_by_seq = NULL;
    engine->meta_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(offsets)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_lengths, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(lengths)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_tjb_by_seq, sizeof(uint8_t) * nseq), errbuf, errbuf_size, "cudaMalloc(tjb_by_seq)")) != eslOK) goto ERROR;
    engine->meta_alloc = nseq;
    reuse_batch = FALSE;
  }
  if (engine->bias_result_alloc < nseq) {
    if (engine->d_bias_filtersc) cudaFree(engine->d_bias_filtersc);
    engine->d_bias_filtersc = NULL;
    engine->bias_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_filtersc, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(bias filtersc)")) != eslOK) goto ERROR;
    engine->bias_result_alloc = nseq;
  }
  if (engine->d_bias_pi == NULL) {
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_pi, sizeof(float) * 3), errbuf, errbuf_size, "cudaMalloc(bias pi)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_t, sizeof(float) * 6), errbuf, errbuf_size, "cudaMalloc(bias t)")) != eslOK) goto ERROR;
  }
  eo_bytes = (size_t) bg->fhmm->abc->Kp * 2 * sizeof(float);
  if (engine->d_bias_eo == NULL) {
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_bias_eo, eo_bytes), errbuf, errbuf_size, "cudaMalloc(bias eo)")) != eslOK) goto ERROR;
  }

  h_pi[0] = bg->fhmm->pi[0];
  h_pi[1] = bg->fhmm->pi[1];
  h_pi[2] = bg->fhmm->pi[2];
  h_t[0] = bg->fhmm->t[0][0];
  h_t[1] = bg->fhmm->t[0][1];
  h_t[2] = bg->fhmm->t[0][2];
  h_t[3] = bg->fhmm->t[1][0];
  h_t[4] = bg->fhmm->t[1][1];
  h_t[5] = bg->fhmm->t[1][2];

  cudaEventCreate(&h2d0);
  cudaEventCreate(&h2d1);
  cudaEventCreate(&k0);
  cudaEventCreate(&k1);
  cudaEventCreate(&d2h0);
  cudaEventCreate(&d2h1);

  cudaEventRecord(h2d0);
  if (!reuse_batch) {
    for (int i = 0; i < nseq; i++) {
      memcpy(engine->h_dsq + h_offsets[i], chu->dsq[i], h_lengths[i] + 2);
    }
    if ((status = cuda_status(cudaMemcpy(engine->d_dsq, engine->h_dsq, total, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(batch dsq)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_offsets, h_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(offsets)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_lengths, h_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(lengths)")) != eslOK) goto CUDA_ERROR;
  }
  if ((status = cuda_status(cudaMemcpy(engine->d_bias_pi, h_pi, sizeof(h_pi), cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(bias pi)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_bias_t, h_t, sizeof(h_t), cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(bias t)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_bias_eo, bg->fhmm->eo[0], eo_bytes, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(bias eo)")) != eslOK) goto CUDA_ERROR;
  cudaEventRecord(h2d1);
  cudaEventSynchronize(h2d1);

  cudaEventRecord(k0);
  cuda_bias_filter_kernel<<<(nseq + 127) / 128, 128>>>(engine->d_dsq, engine->d_offsets, engine->d_lengths, nseq,
                                                       engine->d_bias_pi, engine->d_bias_t, engine->d_bias_eo,
                                                       engine->d_bias_filtersc);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_bias_filter_kernel launch")) != eslOK) goto CUDA_ERROR;
  cudaEventRecord(k1);
  cudaEventSynchronize(k1);

  cudaEventRecord(d2h0);
  if ((status = cuda_status(cudaMemcpy(filtersc, engine->d_bias_filtersc, sizeof(float) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(bias filtersc)")) != eslOK) goto CUDA_ERROR;
  cudaEventRecord(d2h1);
  cudaEventSynchronize(d2h1);

  engine->stats.bias_h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.bias_kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.bias_d2h_seconds    += elapsed_seconds(d2h0, d2h1);

CUDA_ERROR:
  cudaEventDestroy(h2d0);
  cudaEventDestroy(h2d1);
  cudaEventDestroy(k0);
  cudaEventDestroy(k1);
  cudaEventDestroy(d2h0);
  cudaEventDestroy(d2h1);
ERROR:
  free(h_offsets);
  free(h_lengths);
  return status;
}

extern "C" int
p7_cuda_ForwardScoreDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                  ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                  float *scores, int *statuses,
                                  char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int nseq;
  int total = 0;
  int *h_offsets = NULL;
  int *h_lengths = NULL;
  int reuse_batch = FALSE;
  int use_prefix = FALSE;
  int prefix_threads = 0;
  size_t shmem;
  cudaEvent_t h2d0, h2d1, k0, k1, d2h0, d2h1;

  if (!engine || !cuom || !chu || !scores || !statuses) return eslEINVAL;
  nseq = chu->N;
  if (nidx <= 0) return eslOK;
  if (nseq <= 0) return eslOK;

  h_offsets = (int *) malloc(sizeof(int) * nseq);
  h_lengths = (int *) malloc(sizeof(int) * nseq);
  if (!h_offsets || !h_lengths) { status = eslEMEM; goto ERROR; }

  for (int i = 0; i < nseq; i++) {
    h_offsets[i] = total;
    if (chu->L[i] > INT32_MAX) {
      if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "dsqdata sequence length exceeds CUDA Forward v1 limit");
      status = eslERANGE;
      goto ERROR;
    }
    h_lengths[i] = (int) chu->L[i];
    total += h_lengths[i] + 2;
  }
  reuse_batch = (engine->batch_owner == chu && engine->batch_nseq == nseq && engine->batch_total == total);

  if (engine->dsq_alloc < total) {
    if (engine->d_dsq) cudaFree(engine->d_dsq);
    if (engine->h_dsq) cudaFreeHost(engine->h_dsq);
    engine->d_dsq = NULL;
    engine->h_dsq = NULL;
    engine->dsq_alloc = 0;
    engine->h_dsq_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_dsq, total), errbuf, errbuf_size, "cudaMalloc(batch dsq)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMallocHost((void **) &engine->h_dsq, total), errbuf, errbuf_size, "cudaMallocHost(batch dsq)")) != eslOK) goto ERROR;
    engine->dsq_alloc = total;
    engine->h_dsq_alloc = total;
    reuse_batch = FALSE;
  }
  if (engine->meta_alloc < nseq) {
    if (engine->d_offsets) cudaFree(engine->d_offsets);
    if (engine->d_lengths) cudaFree(engine->d_lengths);
    if (engine->d_tjb_by_seq) cudaFree(engine->d_tjb_by_seq);
    engine->d_offsets = NULL;
    engine->d_lengths = NULL;
    engine->d_tjb_by_seq = NULL;
    engine->meta_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(offsets)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_lengths, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(lengths)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_tjb_by_seq, sizeof(uint8_t) * nseq), errbuf, errbuf_size, "cudaMalloc(tjb_by_seq)")) != eslOK) goto ERROR;
    engine->meta_alloc = nseq;
    reuse_batch = FALSE;
  }
  if (engine->fwd_result_alloc < nidx) {
    if (engine->d_fwd_scores) cudaFree(engine->d_fwd_scores);
    if (engine->d_fwd_statuses) cudaFree(engine->d_fwd_statuses);
    if (engine->d_fwd_seqidx) cudaFree(engine->d_fwd_seqidx);
    engine->d_fwd_scores = NULL;
    engine->d_fwd_statuses = NULL;
    engine->d_fwd_seqidx = NULL;
    engine->fwd_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_fwd_scores, sizeof(float) * nidx), errbuf, errbuf_size, "cudaMalloc(fwd scores)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_fwd_statuses, sizeof(int) * nidx), errbuf, errbuf_size, "cudaMalloc(fwd statuses)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_fwd_seqidx, sizeof(int) * nidx), errbuf, errbuf_size, "cudaMalloc(fwd seqidx)")) != eslOK) goto ERROR;
    engine->fwd_result_alloc = nidx;
  }
  if (cuom->Qf * 4 <= 1024) {
    prefix_threads = next_pow2_at_least((cuom->Qf * 4 + 1) / 2, 32);
    if (prefix_threads <= 512) use_prefix = TRUE;
  }
  shmem = use_prefix ? (sizeof(float) * ((size_t) cuom->Qf * 4 * 3 * 2 + (size_t) cuom->Qf * 4 + (size_t) prefix_threads * 2))
                     : (sizeof(float) * (size_t) cuom->Qf * 4 * 3 * 2);

  cudaEventCreate(&h2d0);
  cudaEventCreate(&h2d1);
  cudaEventCreate(&k0);
  cudaEventCreate(&k1);
  cudaEventCreate(&d2h0);
  cudaEventCreate(&d2h1);

  cudaEventRecord(h2d0);
  if (!reuse_batch) {
    for (int i = 0; i < nseq; i++) memcpy(engine->h_dsq + h_offsets[i], chu->dsq[i], h_lengths[i] + 2);
    if ((status = cuda_status(cudaMemcpy(engine->d_dsq, engine->h_dsq, total, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(batch dsq)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_offsets, h_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(offsets)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_lengths, h_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(lengths)")) != eslOK) goto CUDA_ERROR;
  }
  if (seqidx) {
    if ((status = cuda_status(cudaMemcpy(engine->d_fwd_seqidx, seqidx, sizeof(int) * nidx, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(fwd seqidx)")) != eslOK) goto CUDA_ERROR;
  }
  cudaEventRecord(h2d1);
  cudaEventSynchronize(h2d1);

  cudaEventRecord(k0);
  if (use_prefix) {
    cuda_forward_score_prefix_kernel<<<nidx, prefix_threads, shmem>>>(engine->d_dsq, engine->d_offsets, engine->d_lengths,
                                                    seqidx ? engine->d_fwd_seqidx : NULL, nidx,
                                                    cuom->d_rfv, cuom->d_tfv, cuom->M, cuom->Qf, cuom->Kp,
                                                    cuom->xf_e_loop, cuom->xf_e_move,
                                                    cuom->xf_n_loop, cuom->xf_n_move,
                                                    cuom->xf_c_loop, cuom->xf_c_move,
                                                    cuom->xf_j_loop, cuom->xf_j_move,
                                                    cuom->nj, engine->d_fwd_scores, engine->d_fwd_statuses);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_forward_score_prefix_kernel launch")) != eslOK) goto CUDA_ERROR;
  } else {
    cuda_forward_score_kernel<<<nidx, 32, shmem>>>(engine->d_dsq, engine->d_offsets, engine->d_lengths,
                                                   seqidx ? engine->d_fwd_seqidx : NULL, nidx,
                                                   cuom->d_rfv, cuom->d_tfv, cuom->M, cuom->Qf, cuom->Kp,
                                                   cuom->xf_e_loop, cuom->xf_e_move,
                                                   cuom->xf_n_loop, cuom->xf_n_move,
                                                   cuom->xf_c_loop, cuom->xf_c_move,
                                                   cuom->xf_j_loop, cuom->xf_j_move,
                                                   cuom->nj, engine->d_fwd_scores, engine->d_fwd_statuses);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_forward_score_kernel launch")) != eslOK) goto CUDA_ERROR;
  }
  cudaEventRecord(k1);
  cudaEventSynchronize(k1);

  cudaEventRecord(d2h0);
  if ((status = cuda_status(cudaMemcpy(scores, engine->d_fwd_scores, sizeof(float) * nidx, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fwd scores)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(statuses, engine->d_fwd_statuses, sizeof(int) * nidx, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fwd statuses)")) != eslOK) goto CUDA_ERROR;
  cudaEventRecord(d2h1);
  cudaEventSynchronize(d2h1);

  engine->stats.fwd_h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.fwd_kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.fwd_d2h_seconds    += elapsed_seconds(d2h0, d2h1);
  engine->stats.fwd_nseqs          += nidx;
  for (int i = 0; i < nidx; i++) {
    int si = seqidx ? seqidx[i] : i;
    if (si >= 0 && si < nseq) engine->stats.fwd_nres += h_lengths[si];
  }
  engine->stats.fwd_nbatches       += 1;

CUDA_ERROR:
  cudaEventDestroy(h2d0);
  cudaEventDestroy(h2d1);
  cudaEventDestroy(k0);
  cudaEventDestroy(k1);
  cudaEventDestroy(d2h0);
  cudaEventDestroy(d2h1);
ERROR:
  free(h_offsets);
  free(h_lengths);
  return status;
}

extern "C" int
p7_cuda_ViterbiScoreDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                  ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                  float *scores, int *statuses,
                                  char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int nseq;
  int total = 0;
  int *h_offsets = NULL;
  int *h_lengths = NULL;
  int reuse_batch = FALSE;
  size_t shmem;
  cudaEvent_t h2d0, h2d1, k0, k1, d2h0, d2h1;

  if (!engine || !cuom || !chu || !scores || !statuses) return eslEINVAL;
  nseq = chu->N;
  if (nidx <= 0) return eslOK;
  if (nseq <= 0) return eslOK;

  h_offsets = (int *) malloc(sizeof(int) * nseq);
  h_lengths = (int *) malloc(sizeof(int) * nseq);
  if (!h_offsets || !h_lengths) { status = eslEMEM; goto ERROR; }

  for (int i = 0; i < nseq; i++) {
    h_offsets[i] = total;
    if (chu->L[i] > INT32_MAX) {
      if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "dsqdata sequence length exceeds CUDA Viterbi v1 limit");
      status = eslERANGE;
      goto ERROR;
    }
    h_lengths[i] = (int) chu->L[i];
    total += h_lengths[i] + 2;
  }
  reuse_batch = (engine->batch_owner == chu && engine->batch_nseq == nseq && engine->batch_total == total);

  if (engine->dsq_alloc < total) {
    if (engine->d_dsq) cudaFree(engine->d_dsq);
    if (engine->h_dsq) cudaFreeHost(engine->h_dsq);
    engine->d_dsq = NULL;
    engine->h_dsq = NULL;
    engine->dsq_alloc = 0;
    engine->h_dsq_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_dsq, total), errbuf, errbuf_size, "cudaMalloc(batch dsq)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMallocHost((void **) &engine->h_dsq, total), errbuf, errbuf_size, "cudaMallocHost(batch dsq)")) != eslOK) goto ERROR;
    engine->dsq_alloc = total;
    engine->h_dsq_alloc = total;
    reuse_batch = FALSE;
  }
  if (engine->meta_alloc < nseq) {
    if (engine->d_offsets) cudaFree(engine->d_offsets);
    if (engine->d_lengths) cudaFree(engine->d_lengths);
    if (engine->d_tjb_by_seq) cudaFree(engine->d_tjb_by_seq);
    engine->d_offsets = NULL;
    engine->d_lengths = NULL;
    engine->d_tjb_by_seq = NULL;
    engine->meta_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(offsets)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_lengths, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(lengths)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_tjb_by_seq, sizeof(uint8_t) * nseq), errbuf, errbuf_size, "cudaMalloc(tjb_by_seq)")) != eslOK) goto ERROR;
    engine->meta_alloc = nseq;
    reuse_batch = FALSE;
  }
  if (engine->vit_result_alloc < nidx) {
    if (engine->d_vit_scores) cudaFree(engine->d_vit_scores);
    if (engine->d_vit_statuses) cudaFree(engine->d_vit_statuses);
    if (engine->d_vit_seqidx) cudaFree(engine->d_vit_seqidx);
    engine->d_vit_scores = NULL;
    engine->d_vit_statuses = NULL;
    engine->d_vit_seqidx = NULL;
    engine->vit_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_vit_scores, sizeof(float) * nidx), errbuf, errbuf_size, "cudaMalloc(vit scores)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_vit_statuses, sizeof(int) * nidx), errbuf, errbuf_size, "cudaMalloc(vit statuses)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_vit_seqidx, sizeof(int) * nidx), errbuf, errbuf_size, "cudaMalloc(vit seqidx)")) != eslOK) goto ERROR;
    engine->vit_result_alloc = nidx;
  }
  shmem = sizeof(int16_t) * (size_t) cuom->Qw * 8 * 3 * 2;

  cudaEventCreate(&h2d0);
  cudaEventCreate(&h2d1);
  cudaEventCreate(&k0);
  cudaEventCreate(&k1);
  cudaEventCreate(&d2h0);
  cudaEventCreate(&d2h1);

  cudaEventRecord(h2d0);
  if (!reuse_batch) {
    for (int i = 0; i < nseq; i++) memcpy(engine->h_dsq + h_offsets[i], chu->dsq[i], h_lengths[i] + 2);
    if ((status = cuda_status(cudaMemcpy(engine->d_dsq, engine->h_dsq, total, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(batch dsq)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_offsets, h_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(offsets)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_lengths, h_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(lengths)")) != eslOK) goto CUDA_ERROR;
  }
  if (seqidx) {
    if ((status = cuda_status(cudaMemcpy(engine->d_vit_seqidx, seqidx, sizeof(int) * nidx, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(vit seqidx)")) != eslOK) goto CUDA_ERROR;
  }
  cudaEventRecord(h2d1);
  cudaEventSynchronize(h2d1);

  cudaEventRecord(k0);
  cuda_viterbi_score_kernel<<<nidx, 32, shmem>>>(engine->d_dsq, engine->d_offsets, engine->d_lengths,
                                                 seqidx ? engine->d_vit_seqidx : NULL, nidx,
                                                 cuom->d_rwv, cuom->d_twv, cuom->M, cuom->Qw, cuom->Kp,
                                                 cuom->xw_n_loop, cuom->xw_n_move,
                                                 cuom->xw_c_loop, cuom->xw_c_move,
                                                 cuom->xw_j_loop, cuom->xw_j_move,
                                                 cuom->xw_e_loop, cuom->xw_e_move,
                                                 cuom->scale_w, cuom->base_w, cuom->ddbound_w, cuom->nj,
                                                 engine->d_vit_scores, engine->d_vit_statuses);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_viterbi_score_kernel launch")) != eslOK) goto CUDA_ERROR;
  cudaEventRecord(k1);
  cudaEventSynchronize(k1);

  cudaEventRecord(d2h0);
  if ((status = cuda_status(cudaMemcpy(scores, engine->d_vit_scores, sizeof(float) * nidx, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(vit scores)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(statuses, engine->d_vit_statuses, sizeof(int) * nidx, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(vit statuses)")) != eslOK) goto CUDA_ERROR;
  cudaEventRecord(d2h1);
  cudaEventSynchronize(d2h1);

  engine->stats.vit_h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.vit_kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.vit_d2h_seconds    += elapsed_seconds(d2h0, d2h1);
  engine->stats.vit_nseqs          += nidx;
  for (int i = 0; i < nidx; i++) {
    int si = seqidx ? seqidx[i] : i;
    if (si >= 0 && si < nseq) engine->stats.vit_nres += h_lengths[si];
  }
  engine->stats.vit_nbatches       += 1;

CUDA_ERROR:
  cudaEventDestroy(h2d0);
  cudaEventDestroy(h2d1);
  cudaEventDestroy(k0);
  cudaEventDestroy(k1);
  cudaEventDestroy(d2h0);
  cudaEventDestroy(d2h1);
ERROR:
  free(h_offsets);
  free(h_lengths);
  return status;
}

extern "C" int
p7_cuda_MSVFilterBatch(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                       ESL_SQ_BLOCK *block, float *scores, int *statuses,
                       char *errbuf, int errbuf_size)
{
  ESL_DSQDATA_CHUNK chu;
  ESL_DSQ         **dsq = NULL;
  char            **name = NULL;
  char            **acc = NULL;
  char            **desc = NULL;
  int32_t          *taxid = NULL;
  int64_t          *L = NULL;
  int               nseq;
  int               status;

  if (!block) return eslEINVAL;
  nseq = block->count;
  if (nseq <= 0) return eslOK;

  dsq   = (ESL_DSQ **) malloc(sizeof(ESL_DSQ *) * nseq);
  name  = (char **)    malloc(sizeof(char *)    * nseq);
  acc   = (char **)    malloc(sizeof(char *)    * nseq);
  desc  = (char **)    malloc(sizeof(char *)    * nseq);
  taxid = (int32_t *)  malloc(sizeof(int32_t)    * nseq);
  L     = (int64_t *)  malloc(sizeof(int64_t)    * nseq);
  if (!dsq || !name || !acc || !desc || !taxid || !L) { status = eslEMEM; goto ERROR; }

  for (int i = 0; i < nseq; i++) {
    dsq[i]   = block->list[i].dsq;
    name[i]  = block->list[i].name;
    acc[i]   = block->list[i].acc;
    desc[i]  = block->list[i].desc;
    taxid[i] = block->list[i].tax_id;
    L[i]     = block->list[i].n;
  }

  memset(&chu, 0, sizeof(chu));
  chu.N     = nseq;
  chu.dsq   = dsq;
  chu.name  = name;
  chu.acc   = acc;
  chu.desc  = desc;
  chu.taxid = taxid;
  chu.L     = L;

  status = p7_cuda_MSVFilterDsqdataChunk(engine, cuom, &chu, scores, statuses, errbuf, errbuf_size);

ERROR:
  free(dsq);
  free(name);
  free(acc);
  free(desc);
  free(taxid);
  free(L);
  return status;
}

extern "C" void
p7_cuda_msvprofile_Destroy(P7_CUDA_MSVPROFILE *cuom)
{
  if (!cuom) return;
  if (cuom->d_rbv) cudaFree(cuom->d_rbv);
  if (cuom->d_rfv) cudaFree(cuom->d_rfv);
  if (cuom->d_tfv) cudaFree(cuom->d_tfv);
  if (cuom->d_rwv) cudaFree(cuom->d_rwv);
  if (cuom->d_twv) cudaFree(cuom->d_twv);
  free(cuom->h_tjb_by_len);
  free(cuom);
}

extern "C" int
p7_cuda_msvprofile_UpdateLength(P7_CUDA_MSVPROFILE *cuom, const P7_OPROFILE *om, int L, char *errbuf, int errbuf_size)
{
  if (!cuom || !om) return eslEINVAL;
  cuom->tjb_b = om->tjb_b;
  return eslOK;
}

extern "C" int
p7_cuda_MSVFilter(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                  const ESL_DSQ *dsq, int L, float *ret_sc,
                  char *errbuf, int errbuf_size)
{
  int raw = 0;
  int overflow = 0;
  int status;
  size_t dsq_bytes = (size_t) L + 2;
  size_t shmem = (size_t) (cuom->M + 1) * 2;
  cudaEvent_t h2d0, h2d1, k0, k1, d2h0, d2h1;

  if (!engine || !cuom || !dsq || L < 0 || !ret_sc) return eslEINVAL;
  if (shmem > 96 * 1024) {
    if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "CUDA MSV profile M=%d exceeds v1 shared-memory limit", cuom->M);
    return eslERANGE;
  }
  if (engine->dsq_alloc < (int) dsq_bytes) {
    if (engine->d_dsq) cudaFree(engine->d_dsq);
    engine->d_dsq = NULL;
    engine->dsq_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_dsq, dsq_bytes), errbuf, errbuf_size, "cudaMalloc(dsq)")) != eslOK) return status;
    engine->dsq_alloc = (int) dsq_bytes;
  }

  cudaEventCreate(&h2d0);
  cudaEventCreate(&h2d1);
  cudaEventCreate(&k0);
  cudaEventCreate(&k1);
  cudaEventCreate(&d2h0);
  cudaEventCreate(&d2h1);

  cudaEventRecord(h2d0);
  if ((status = cuda_status(cudaMemcpy(engine->d_dsq, dsq, dsq_bytes, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(dsq)")) != eslOK) goto ERROR;
  cudaEventRecord(h2d1);
  cudaEventSynchronize(h2d1);

  cudaEventRecord(k0);
  cuda_msv_kernel<<<1, P7_CUDA_MSV_BLOCK_THREADS, shmem>>>(engine->d_dsq, L, cuom->d_rbv, cuom->M, cuom->Q, cuom->Kp,
                                                           cuom->tbm_b, cuom->tec_b, cuom->tjb_b,
                                                           cuom->base_b, cuom->bias_b,
                                                           engine->d_raw, engine->d_overflow);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_msv_kernel launch")) != eslOK) goto ERROR;
  cudaEventRecord(k1);
  cudaEventSynchronize(k1);

  cudaEventRecord(d2h0);
  if ((status = cuda_status(cudaMemcpy(&raw, engine->d_raw, sizeof(int), cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(raw)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(&overflow, engine->d_overflow, sizeof(int), cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(overflow)")) != eslOK) goto ERROR;
  cudaEventRecord(d2h1);
  cudaEventSynchronize(d2h1);

  engine->stats.h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.d2h_seconds    += elapsed_seconds(d2h0, d2h1);
  engine->stats.nseqs++;
  engine->stats.nres += L;

  if (overflow) {
    *ret_sc = eslINFINITY;
    status = eslERANGE;
  } else {
    *ret_sc = ((float) (raw - cuom->tjb_b) - (float) cuom->base_b);
    *ret_sc /= cuom->scale_b;
    *ret_sc -= 3.0f;
    status = eslOK;
  }

ERROR:
  cudaEventDestroy(h2d0);
  cudaEventDestroy(h2d1);
  cudaEventDestroy(k0);
  cudaEventDestroy(k1);
  cudaEventDestroy(d2h0);
  cudaEventDestroy(d2h1);
  return status;
}
