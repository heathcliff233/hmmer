#include "p7_cuda_internal.h"

__device__ static inline int
fwd_tfv_idx(int t, int q, int lane, int Q)
{
  return (t == p7O_DD) ? (((7 * Q) + q) * 4 + lane) : (((q * 7) + t) * 4 + lane);
}

__device__ static inline int
cell_q(int c, int Q)
{
  return c % Q;
}

__device__ static inline int
cell_lane(int c, int Q)
{
  return c / Q;
}

__global__ static void
cuda_forward_parser_xmx_kernel(const uint8_t *dsq, int L,
                               const float *rfv, const float *tfv, int M, int Q, int Kp,
                               float xf_e_loop, float xf_e_move, float xf_n_loop_base, float xf_n_move_base,
                               float xf_c_loop_base, float xf_c_move_base, float xf_j_loop_base, float xf_j_move_base,
                               float nj, float *xmx, float *scores, int *statuses)
{
  extern __shared__ float fwd_parser_mem[];
  int N = Q * 4;
  float *prev = fwd_parser_mem;
  float *curr = prev + (size_t) N * 3;

  if (threadIdx.x != 0) return;

  float pmove = (2.0f + nj) / ((float) L + 2.0f + nj);
  float ploop = 1.0f - pmove;
  float xf_n_loop = xf_n_loop_base >= 0.0f ? xf_n_loop_base : ploop;
  float xf_n_move = xf_n_move_base >= 0.0f ? xf_n_move_base : pmove;
  float xf_c_loop = xf_c_loop_base >= 0.0f ? xf_c_loop_base : ploop;
  float xf_c_move = xf_c_move_base >= 0.0f ? xf_c_move_base : pmove;
  float xf_j_loop = xf_j_loop_base >= 0.0f ? xf_j_loop_base : ploop;
  float xf_j_move = xf_j_move_base >= 0.0f ? xf_j_move_base : pmove;
  float xN = 1.0f;
  float xJ = 0.0f;
  float xB = xf_n_move;
  float xC = 0.0f;
  float xE = 0.0f;
  float totscale = 0.0f;

  for (int c = 0; c < N * 3; c++) prev[c] = 0.0f;
  xmx[p7X_E]     = xE;
  xmx[p7X_N]     = xN;
  xmx[p7X_J]     = xJ;
  xmx[p7X_B]     = xB;
  xmx[p7X_C]     = xC;
  xmx[p7X_SCALE] = 1.0f;

  for (int i = 1; i <= L; i++) {
    uint8_t x = dsq[i];
    if (x >= Kp) {
      scores[0] = 0.0f;
      statuses[0] = eslEINVAL;
      return;
    }

    xE = 0.0f;
    for (int c = 0; c < N; c++) {
      int q = cell_q(c, Q);
      int lane = cell_lane(c, Q);
      int cell = c * 3;
      float mpv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 0];
      float dpv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 1];
      float ipv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 2];
      float m = xB * tfv[fwd_tfv_idx(p7O_BM, q, lane, Q)];
      m += mpv * tfv[fwd_tfv_idx(p7O_MM, q, lane, Q)];
      m += ipv * tfv[fwd_tfv_idx(p7O_IM, q, lane, Q)];
      m += dpv * tfv[fwd_tfv_idx(p7O_DM, q, lane, Q)];
      m *= rfv[((int) x * Q + q) * 4 + lane];
      curr[cell + 0] = m;
      curr[cell + 2] = prev[cell + 0] * tfv[fwd_tfv_idx(p7O_MI, q, lane, Q)]
                     + prev[cell + 2] * tfv[fwd_tfv_idx(p7O_II, q, lane, Q)];
    }

    curr[p7X_D] = 0.0f;
    for (int c = 1; c < N; c++) {
      int pq = cell_q(c - 1, Q);
      int plane = cell_lane(c - 1, Q);
      curr[c * 3 + 1] = curr[(c - 1) * 3 + 0] * tfv[fwd_tfv_idx(p7O_MD, pq, plane, Q)]
                      + curr[(c - 1) * 3 + 1] * tfv[fwd_tfv_idx(p7O_DD, pq, plane, Q)];
    }

    for (int c = 0; c < N; c++) xE += curr[c * 3 + 0] + curr[c * 3 + 1];

    xN = xN * xf_n_loop;
    xC = (xC * xf_c_loop) + (xE * xf_e_move);
    xJ = (xJ * xf_j_loop) + (xE * xf_e_loop);
    xB = (xJ * xf_j_move) + (xN * xf_n_move);

    float scale = 1.0f;
    if (xE > 1.0e4f) {
      scale = xE;
      float inv = 1.0f / xE;
      xN *= inv;
      xC *= inv;
      xJ *= inv;
      xB *= inv;
      for (int c = 0; c < N * 3; c++) curr[c] *= inv;
      totscale += logf(xE);
      xE = 1.0f;
    }

    xmx[i * p7X_NXCELLS + p7X_E]     = xE;
    xmx[i * p7X_NXCELLS + p7X_N]     = xN;
    xmx[i * p7X_NXCELLS + p7X_J]     = xJ;
    xmx[i * p7X_NXCELLS + p7X_B]     = xB;
    xmx[i * p7X_NXCELLS + p7X_C]     = xC;
    xmx[i * p7X_NXCELLS + p7X_SCALE] = scale;

    float *tmp = prev;
    prev = curr;
    curr = tmp;
  }

  if (isnan(xC) || (L > 0 && xC == 0.0f) || isinf(xC)) {
    scores[0] = 0.0f;
    statuses[0] = eslERANGE;
  } else {
    scores[0] = totscale + logf(xC * xf_c_move);
    statuses[0] = eslOK;
  }
}

__global__ static void
cuda_forward_parser_xmx_batch_kernel(const uint8_t *dsq, const int *offsets, const int *lengths,
                                     const int *seqidx, int nidx, const size_t *x_offsets,
                                     const float *rfv, const float *tfv, int M, int Q, int Kp,
                                     float xf_e_loop, float xf_e_move, float xf_n_loop_base, float xf_n_move_base,
                                     float xf_c_loop_base, float xf_c_move_base, float xf_j_loop_base, float xf_j_move_base,
                                     float nj, float *xmx, float *scores, int *statuses)
{
  extern __shared__ float fwd_parser_mem[];
  int b = blockIdx.x;
  int N = Q * 4;
  float *prev = fwd_parser_mem;
  float *curr = prev + (size_t) N * 3;
  int si, L;
  const uint8_t *sdsq;
  float *sxmx;
  float pmove, ploop;
  float xf_n_loop, xf_n_move, xf_c_loop, xf_c_move, xf_j_loop, xf_j_move;
  float xN = 1.0f;
  float xJ = 0.0f;
  float xB;
  float xC = 0.0f;
  float xE = 0.0f;
  float totscale = 0.0f;

  if (b >= nidx) return;
  if (threadIdx.x != 0) return;
  si = seqidx ? seqidx[b] : b;
  L = lengths[si];
  sdsq = dsq + offsets[si];
  sxmx = xmx + x_offsets[b];

  pmove = (2.0f + nj) / ((float) L + 2.0f + nj);
  ploop = 1.0f - pmove;
  xf_n_loop = xf_n_loop_base >= 0.0f ? xf_n_loop_base : ploop;
  xf_n_move = xf_n_move_base >= 0.0f ? xf_n_move_base : pmove;
  xf_c_loop = xf_c_loop_base >= 0.0f ? xf_c_loop_base : ploop;
  xf_c_move = xf_c_move_base >= 0.0f ? xf_c_move_base : pmove;
  xf_j_loop = xf_j_loop_base >= 0.0f ? xf_j_loop_base : ploop;
  xf_j_move = xf_j_move_base >= 0.0f ? xf_j_move_base : pmove;
  xB = xf_n_move;

  for (int c = 0; c < N * 3; c++) prev[c] = 0.0f;
  sxmx[p7X_E]     = xE;
  sxmx[p7X_N]     = xN;
  sxmx[p7X_J]     = xJ;
  sxmx[p7X_B]     = xB;
  sxmx[p7X_C]     = xC;
  sxmx[p7X_SCALE] = 1.0f;

  for (int i = 1; i <= L; i++) {
    uint8_t x = sdsq[i];
    if (x >= Kp) {
      scores[b * 2 + 0] = 0.0f;
      statuses[b * 2 + 0] = eslEINVAL;
      return;
    }

    xE = 0.0f;
    for (int c = 0; c < N; c++) {
      int q = cell_q(c, Q);
      int lane = cell_lane(c, Q);
      int cell = c * 3;
      float mpv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 0];
      float dpv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 1];
      float ipv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 2];
      float m = xB * tfv[fwd_tfv_idx(p7O_BM, q, lane, Q)];
      m += mpv * tfv[fwd_tfv_idx(p7O_MM, q, lane, Q)];
      m += ipv * tfv[fwd_tfv_idx(p7O_IM, q, lane, Q)];
      m += dpv * tfv[fwd_tfv_idx(p7O_DM, q, lane, Q)];
      m *= rfv[((int) x * Q + q) * 4 + lane];
      curr[cell + 0] = m;
      curr[cell + 2] = prev[cell + 0] * tfv[fwd_tfv_idx(p7O_MI, q, lane, Q)]
                     + prev[cell + 2] * tfv[fwd_tfv_idx(p7O_II, q, lane, Q)];
    }

    curr[p7X_D] = 0.0f;
    for (int c = 1; c < N; c++) {
      int pq = cell_q(c - 1, Q);
      int plane = cell_lane(c - 1, Q);
      curr[c * 3 + 1] = curr[(c - 1) * 3 + 0] * tfv[fwd_tfv_idx(p7O_MD, pq, plane, Q)]
                      + curr[(c - 1) * 3 + 1] * tfv[fwd_tfv_idx(p7O_DD, pq, plane, Q)];
    }

    for (int c = 0; c < N; c++) xE += curr[c * 3 + 0] + curr[c * 3 + 1];

    xN = xN * xf_n_loop;
    xC = (xC * xf_c_loop) + (xE * xf_e_move);
    xJ = (xJ * xf_j_loop) + (xE * xf_e_loop);
    xB = (xJ * xf_j_move) + (xN * xf_n_move);

    float scale = 1.0f;
    if (xE > 1.0e4f) {
      scale = xE;
      float inv = 1.0f / xE;
      xN *= inv;
      xC *= inv;
      xJ *= inv;
      xB *= inv;
      for (int c = 0; c < N * 3; c++) curr[c] *= inv;
      totscale += logf(xE);
      xE = 1.0f;
    }

    sxmx[i * p7X_NXCELLS + p7X_E]     = xE;
    sxmx[i * p7X_NXCELLS + p7X_N]     = xN;
    sxmx[i * p7X_NXCELLS + p7X_J]     = xJ;
    sxmx[i * p7X_NXCELLS + p7X_B]     = xB;
    sxmx[i * p7X_NXCELLS + p7X_C]     = xC;
    sxmx[i * p7X_NXCELLS + p7X_SCALE] = scale;

    float *tmp = prev;
    prev = curr;
    curr = tmp;
  }

  if (isnan(xC) || (L > 0 && xC == 0.0f) || isinf(xC)) {
    scores[b * 2 + 0] = 0.0f;
    statuses[b * 2 + 0] = eslERANGE;
  } else {
    scores[b * 2 + 0] = totscale + logf(xC * xf_c_move);
    statuses[b * 2 + 0] = eslOK;
  }
}

__global__ static void
cuda_forward_parser_xmx_batch_parallel_kernel(const uint8_t *dsq, const int *offsets, const int *lengths,
                                              const int *seqidx, int nidx, const size_t *x_offsets,
                                              const float *rfv, const float *tfv, int M, int Q, int Kp,
                                              float xf_e_loop, float xf_e_move, float xf_n_loop_base, float xf_n_move_base,
                                              float xf_c_loop_base, float xf_c_move_base, float xf_j_loop_base, float xf_j_move_base,
                                              float nj, float *xmx, float *scores, int *statuses)
{
  extern __shared__ float fwd_parser_parallel_mem[];
  int b = blockIdx.x;
  int tid = threadIdx.x;
  int T = blockDim.x;
  int N = Q * 4;
  float *prev = fwd_parser_parallel_mem;
  float *curr = prev + (size_t) N * 3;
  float *bcoef = curr + (size_t) N * 3;
  float *scanA = bcoef + N;
  float *scanB = scanA + T;
  __shared__ int si, L;
  __shared__ const uint8_t *sdsq;
  __shared__ float *sxmx;
  __shared__ float sxN, sxJ, sxB, sxC, sxE, stotscale;
  __shared__ float sscale, sinv;
  __shared__ float xf_n_loop, xf_n_move, xf_c_loop, xf_c_move, xf_j_loop, xf_j_move;

  if (b >= nidx) return;

  if (tid == 0) {
    si = seqidx ? seqidx[b] : b;
    L = lengths[si];
    sdsq = dsq + offsets[si];
    sxmx = xmx + x_offsets[b];

    {
      float pmove = (2.0f + nj) / ((float) L + 2.0f + nj);
      float ploop = 1.0f - pmove;
      xf_n_loop = xf_n_loop_base >= 0.0f ? xf_n_loop_base : ploop;
      xf_n_move = xf_n_move_base >= 0.0f ? xf_n_move_base : pmove;
      xf_c_loop = xf_c_loop_base >= 0.0f ? xf_c_loop_base : ploop;
      xf_c_move = xf_c_move_base >= 0.0f ? xf_c_move_base : pmove;
      xf_j_loop = xf_j_loop_base >= 0.0f ? xf_j_loop_base : ploop;
      xf_j_move = xf_j_move_base >= 0.0f ? xf_j_move_base : pmove;
    }

    sxN = 1.0f;
    sxJ = 0.0f;
    sxB = xf_n_move;
    sxC = 0.0f;
    sxE = 0.0f;
    stotscale = 0.0f;

    sxmx[p7X_E]     = 0.0f;
    sxmx[p7X_N]     = 1.0f;
    sxmx[p7X_J]     = 0.0f;
    sxmx[p7X_B]     = sxB;
    sxmx[p7X_C]     = 0.0f;
    sxmx[p7X_SCALE] = 1.0f;
  }
  __syncthreads();

  for (int cell = tid; cell < N * 3; cell += T) prev[cell] = 0.0f;
  __syncthreads();

  for (int i = 1; i <= L; i++) {
    uint8_t x = sdsq[i];
    if (x >= Kp) {
      if (tid == 0) {
        scores[b * 2 + 0] = 0.0f;
        statuses[b * 2 + 0] = eslEINVAL;
      }
      return;
    }

    for (int c = tid; c < N; c += T) {
      int q = c % Q;
      int lane = c / Q;
      int cell = c * 3;
      float mpv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 0];
      float dpv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 1];
      float ipv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 2];
      float m = sxB * tfv[fwd_tfv_idx(p7O_BM, q, lane, Q)];
      m += mpv * tfv[fwd_tfv_idx(p7O_MM, q, lane, Q)];
      m += ipv * tfv[fwd_tfv_idx(p7O_IM, q, lane, Q)];
      m += dpv * tfv[fwd_tfv_idx(p7O_DM, q, lane, Q)];
      m *= rfv[((int) x * Q + q) * 4 + lane];
      curr[cell + 0] = m;
      curr[cell + 2] = prev[cell + 0] * tfv[fwd_tfv_idx(p7O_MI, q, lane, Q)]
                     + prev[cell + 2] * tfv[fwd_tfv_idx(p7O_II, q, lane, Q)];
    }
    __syncthreads();

    for (int c = tid; c < N; c += T) {
      int cell = c * 3;
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
      float a_prev = 0.0f, b_prev = 1.0f, a_cur = 0.0f, b_cur = 1.0f;
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

    {
      float partial = 0.0f;
      for (int c = tid; c < N; c += T) partial += curr[c * 3 + 0] + curr[c * 3 + 1];
      scanA[tid] = partial;
    }
    __syncthreads();
    for (int off = T >> 1; off > 0; off >>= 1) {
      if (tid < off) scanA[tid] += scanA[tid + off];
      __syncthreads();
    }

    if (tid == 0) {
      sxE = scanA[0];
      sxN = sxN * xf_n_loop;
      sxC = (sxC * xf_c_loop) + (sxE * xf_e_move);
      sxJ = (sxJ * xf_j_loop) + (sxE * xf_e_loop);
      sxB = (sxJ * xf_j_move) + (sxN * xf_n_move);
      sscale = 1.0f;
      sinv = 1.0f;
      if (sxE > 1.0e4f) {
        sscale = sxE;
        sinv = 1.0f / sxE;
        sxN *= sinv;
        sxC *= sinv;
        sxJ *= sinv;
        sxB *= sinv;
        stotscale += logf(sxE);
        sxE = 1.0f;
      }

      sxmx[i * p7X_NXCELLS + p7X_E]     = sxE;
      sxmx[i * p7X_NXCELLS + p7X_N]     = sxN;
      sxmx[i * p7X_NXCELLS + p7X_J]     = sxJ;
      sxmx[i * p7X_NXCELLS + p7X_B]     = sxB;
      sxmx[i * p7X_NXCELLS + p7X_C]     = sxC;
      sxmx[i * p7X_NXCELLS + p7X_SCALE] = sscale;
    }
    __syncthreads();

    if (sscale > 1.0f) {
      for (int cell = tid; cell < N * 3; cell += T) curr[cell] *= sinv;
    }
    __syncthreads();

    {
      float *tmp = prev;
      prev = curr;
      curr = tmp;
    }
  }

  if (tid == 0) {
    if (isnan(sxC) || (L > 0 && sxC == 0.0f) || isinf(sxC)) {
      scores[b * 2 + 0] = 0.0f;
      statuses[b * 2 + 0] = eslERANGE;
    } else {
      scores[b * 2 + 0] = stotscale + logf(sxC * xf_c_move);
      statuses[b * 2 + 0] = eslOK;
    }
  }
}

__global__ static void
cuda_backward_parser_xmx_kernel(const uint8_t *dsq, int L,
                                const float *rfv, const float *tfv, int M, int Q, int Kp,
                                float xf_e_loop, float xf_e_move, float xf_n_loop_base, float xf_n_move_base,
                                float xf_c_loop_base, float xf_c_move_base, float xf_j_loop_base, float xf_j_move_base,
                                float nj, const float *xfwd, float *xbck, float *scores, int *statuses)
{
  extern __shared__ float bck_parser_mem[];
  int N = Q * 4;
  float *next = bck_parser_mem;
  float *curr = next + (size_t) N * 3;

  if (threadIdx.x != 0) return;

  float pmove = (2.0f + nj) / ((float) L + 2.0f + nj);
  float ploop = 1.0f - pmove;
  float xf_n_loop = xf_n_loop_base >= 0.0f ? xf_n_loop_base : ploop;
  float xf_n_move = xf_n_move_base >= 0.0f ? xf_n_move_base : pmove;
  float xf_c_loop = xf_c_loop_base >= 0.0f ? xf_c_loop_base : ploop;
  float xf_c_move = xf_c_move_base >= 0.0f ? xf_c_move_base : pmove;
  float xf_j_loop = xf_j_loop_base >= 0.0f ? xf_j_loop_base : ploop;
  float xf_j_move = xf_j_move_base >= 0.0f ? xf_j_move_base : pmove;
  float xJ = 0.0f;
  float xB = 0.0f;
  float xN = 0.0f;
  float xC = xf_c_move;
  float xE = xC * xf_e_move;
  float totscale = 0.0f;

  for (int c = 0; c < N; c++) {
    next[c * 3 + 0] = xE;
    next[c * 3 + 1] = xE;
    next[c * 3 + 2] = 0.0f;
  }
  for (int c = N - 2; c >= 0; c--) {
    int q = cell_q(c, Q);
    int lane = cell_lane(c, Q);
    next[c * 3 + 1] += next[(c + 1) * 3 + 1] * tfv[fwd_tfv_idx(p7O_DD, q, lane, Q)];
  }
  for (int c = N - 2; c >= 0; c--) {
    int q = cell_q(c, Q);
    int lane = cell_lane(c, Q);
    next[c * 3 + 0] += next[(c + 1) * 3 + 1] * tfv[fwd_tfv_idx(p7O_MD, q, lane, Q)];
  }

  float scale = xfwd[L * p7X_NXCELLS + p7X_SCALE];
  if (scale > 1.0f) {
    float inv = 1.0f / scale;
    xE *= inv;
    xN *= inv;
    xC *= inv;
    xJ *= inv;
    xB *= inv;
    for (int c = 0; c < N * 3; c++) next[c] *= inv;
    totscale += logf(scale);
  }
  xbck[L * p7X_NXCELLS + p7X_E]     = xE;
  xbck[L * p7X_NXCELLS + p7X_N]     = xN;
  xbck[L * p7X_NXCELLS + p7X_J]     = xJ;
  xbck[L * p7X_NXCELLS + p7X_B]     = xB;
  xbck[L * p7X_NXCELLS + p7X_C]     = xC;
  xbck[L * p7X_NXCELLS + p7X_SCALE] = scale;

  for (int i = L - 1; i >= 1; i--) {
    uint8_t x = dsq[i + 1];
    if (x >= Kp) {
      scores[1] = 0.0f;
      statuses[1] = eslEINVAL;
      return;
    }

    xB = 0.0f;
    for (int c = 0; c < N; c++) {
      int q = cell_q(c, Q);
      int lane = cell_lane(c, Q);
      float mpv = next[c * 3 + 0] * rfv[((int) x * Q + q) * 4 + lane];
      xB += mpv * tfv[fwd_tfv_idx(p7O_BM, q, lane, Q)];
    }
    for (int c = N - 1; c >= 0; c--) {
      int cell = c * 3;
      float mpv = 0.0f;
      if (c + 1 < N) {
        int nq = cell_q(c + 1, Q);
        int nlane = cell_lane(c + 1, Q);
        mpv = next[(c + 1) * 3 + 0] * rfv[((int) x * Q + nq) * 4 + nlane];
      }
      int q = cell_q(c, Q);
      int lane = cell_lane(c, Q);
      curr[cell + 2] = next[cell + 2] * tfv[fwd_tfv_idx(p7O_II, q, lane, Q)];
      curr[cell + 0] = next[cell + 2] * tfv[fwd_tfv_idx(p7O_MI, q, lane, Q)];
      if (c + 1 < N) {
        int nq = cell_q(c + 1, Q);
        int nlane = cell_lane(c + 1, Q);
        curr[cell + 2] += mpv * tfv[fwd_tfv_idx(p7O_IM, nq, nlane, Q)];
        curr[cell + 1]  = mpv * tfv[fwd_tfv_idx(p7O_DM, nq, nlane, Q)];
        curr[cell + 0] += mpv * tfv[fwd_tfv_idx(p7O_MM, nq, nlane, Q)];
      } else {
        curr[cell + 1] = 0.0f;
      }
    }

    xC = xC * xf_c_loop;
    xJ = (xB * xf_j_move) + (xJ * xf_j_loop);
    xN = (xB * xf_n_move) + (xN * xf_n_loop);
    xE = (xC * xf_e_move) + (xJ * xf_e_loop);

    for (int c = N - 1; c >= 0; c--) {
      curr[c * 3 + 0] += xE;
      curr[c * 3 + 1] += xE;
      if (c + 1 < N) {
        int q = cell_q(c, Q);
        int lane = cell_lane(c, Q);
        curr[c * 3 + 1] += curr[(c + 1) * 3 + 1] * tfv[fwd_tfv_idx(p7O_DD, q, lane, Q)];
      }
    }
    for (int c = N - 2; c >= 0; c--) {
      int q = cell_q(c, Q);
      int lane = cell_lane(c, Q);
      curr[c * 3 + 0] += curr[(c + 1) * 3 + 1] * tfv[fwd_tfv_idx(p7O_MD, q, lane, Q)];
    }

    scale = xfwd[i * p7X_NXCELLS + p7X_SCALE];
    if (scale > 1.0f) {
      float inv = 1.0f / scale;
      xE *= inv;
      xN *= inv;
      xJ *= inv;
      xB *= inv;
      xC *= inv;
      for (int c = 0; c < N * 3; c++) curr[c] *= inv;
      totscale += logf(scale);
    }

    xbck[i * p7X_NXCELLS + p7X_E]     = xE;
    xbck[i * p7X_NXCELLS + p7X_N]     = xN;
    xbck[i * p7X_NXCELLS + p7X_J]     = xJ;
    xbck[i * p7X_NXCELLS + p7X_B]     = xB;
    xbck[i * p7X_NXCELLS + p7X_C]     = xC;
    xbck[i * p7X_NXCELLS + p7X_SCALE] = scale;

    float *tmp = next;
    next = curr;
    curr = tmp;
  }

  if (L >= 1) {
    uint8_t x = dsq[1];
    if (x >= Kp) {
      scores[1] = 0.0f;
      statuses[1] = eslEINVAL;
      return;
    }
    xB = 0.0f;
    for (int c = 0; c < N; c++) {
      int q = cell_q(c, Q);
      int lane = cell_lane(c, Q);
      float mpv = next[c * 3 + 0] * rfv[((int) x * Q + q) * 4 + lane];
      xB += mpv * tfv[fwd_tfv_idx(p7O_BM, q, lane, Q)];
    }
    xN = (xB * xf_n_move) + (xN * xf_n_loop);
  }

  xbck[p7X_E]     = 0.0f;
  xbck[p7X_N]     = xN;
  xbck[p7X_J]     = 0.0f;
  xbck[p7X_B]     = xB;
  xbck[p7X_C]     = 0.0f;
  xbck[p7X_SCALE] = 1.0f;

  if (isnan(xN) || (L > 0 && xN == 0.0f) || isinf(xN)) {
    scores[1] = 0.0f;
    statuses[1] = eslERANGE;
  } else {
    scores[1] = totscale + logf(xN);
    statuses[1] = eslOK;
  }
}

__global__ static void
cuda_backward_parser_xmx_batch_kernel(const uint8_t *dsq, const int *offsets, const int *lengths,
                                      const int *seqidx, int nidx, const size_t *x_offsets,
                                      const float *rfv, const float *tfv, int M, int Q, int Kp,
                                      float xf_e_loop, float xf_e_move, float xf_n_loop_base, float xf_n_move_base,
                                      float xf_c_loop_base, float xf_c_move_base, float xf_j_loop_base, float xf_j_move_base,
                                      float nj, const float *xfwd, float *xbck, float *scores, int *statuses)
{
  extern __shared__ float bck_parser_mem[];
  int b = blockIdx.x;
  int N = Q * 4;
  float *next = bck_parser_mem;
  float *curr = next + (size_t) N * 3;
  int si, L;
  const uint8_t *sdsq;
  const float *sxfwd;
  float *sxbck;
  float pmove, ploop;
  float xf_n_loop, xf_n_move, xf_c_loop, xf_c_move, xf_j_loop, xf_j_move;
  float xJ = 0.0f;
  float xB = 0.0f;
  float xN = 0.0f;
  float xC;
  float xE;
  float totscale = 0.0f;
  float scale;

  if (b >= nidx) return;
  if (threadIdx.x != 0) return;
  si = seqidx ? seqidx[b] : b;
  L = lengths[si];
  sdsq = dsq + offsets[si];
  sxfwd = xfwd + x_offsets[b];
  sxbck = xbck + x_offsets[b];

  pmove = (2.0f + nj) / ((float) L + 2.0f + nj);
  ploop = 1.0f - pmove;
  xf_n_loop = xf_n_loop_base >= 0.0f ? xf_n_loop_base : ploop;
  xf_n_move = xf_n_move_base >= 0.0f ? xf_n_move_base : pmove;
  xf_c_loop = xf_c_loop_base >= 0.0f ? xf_c_loop_base : ploop;
  xf_c_move = xf_c_move_base >= 0.0f ? xf_c_move_base : pmove;
  xf_j_loop = xf_j_loop_base >= 0.0f ? xf_j_loop_base : ploop;
  xf_j_move = xf_j_move_base >= 0.0f ? xf_j_move_base : pmove;
  xC = xf_c_move;
  xE = xC * xf_e_move;

  for (int c = 0; c < N; c++) {
    next[c * 3 + 0] = xE;
    next[c * 3 + 1] = xE;
    next[c * 3 + 2] = 0.0f;
  }
  for (int c = N - 2; c >= 0; c--) {
    int q = cell_q(c, Q);
    int lane = cell_lane(c, Q);
    next[c * 3 + 1] += next[(c + 1) * 3 + 1] * tfv[fwd_tfv_idx(p7O_DD, q, lane, Q)];
  }
  for (int c = N - 2; c >= 0; c--) {
    int q = cell_q(c, Q);
    int lane = cell_lane(c, Q);
    next[c * 3 + 0] += next[(c + 1) * 3 + 1] * tfv[fwd_tfv_idx(p7O_MD, q, lane, Q)];
  }

  scale = sxfwd[L * p7X_NXCELLS + p7X_SCALE];
  if (scale > 1.0f) {
    float inv = 1.0f / scale;
    xE *= inv;
    xN *= inv;
    xC *= inv;
    xJ *= inv;
    xB *= inv;
    for (int c = 0; c < N * 3; c++) next[c] *= inv;
    totscale += logf(scale);
  }
  sxbck[L * p7X_NXCELLS + p7X_E]     = xE;
  sxbck[L * p7X_NXCELLS + p7X_N]     = xN;
  sxbck[L * p7X_NXCELLS + p7X_J]     = xJ;
  sxbck[L * p7X_NXCELLS + p7X_B]     = xB;
  sxbck[L * p7X_NXCELLS + p7X_C]     = xC;
  sxbck[L * p7X_NXCELLS + p7X_SCALE] = scale;

  for (int i = L - 1; i >= 1; i--) {
    uint8_t x = sdsq[i + 1];
    if (x >= Kp) {
      scores[b * 2 + 1] = 0.0f;
      statuses[b * 2 + 1] = eslEINVAL;
      return;
    }

    xB = 0.0f;
    for (int c = 0; c < N; c++) {
      int q = cell_q(c, Q);
      int lane = cell_lane(c, Q);
      float mpv = next[c * 3 + 0] * rfv[((int) x * Q + q) * 4 + lane];
      xB += mpv * tfv[fwd_tfv_idx(p7O_BM, q, lane, Q)];
    }
    for (int c = N - 1; c >= 0; c--) {
      int cell = c * 3;
      float mpv = 0.0f;
      if (c + 1 < N) {
        int nq = cell_q(c + 1, Q);
        int nlane = cell_lane(c + 1, Q);
        mpv = next[(c + 1) * 3 + 0] * rfv[((int) x * Q + nq) * 4 + nlane];
      }
      int q = cell_q(c, Q);
      int lane = cell_lane(c, Q);
      curr[cell + 2] = next[cell + 2] * tfv[fwd_tfv_idx(p7O_II, q, lane, Q)];
      curr[cell + 0] = next[cell + 2] * tfv[fwd_tfv_idx(p7O_MI, q, lane, Q)];
      if (c + 1 < N) {
        int nq = cell_q(c + 1, Q);
        int nlane = cell_lane(c + 1, Q);
        curr[cell + 2] += mpv * tfv[fwd_tfv_idx(p7O_IM, nq, nlane, Q)];
        curr[cell + 1]  = mpv * tfv[fwd_tfv_idx(p7O_DM, nq, nlane, Q)];
        curr[cell + 0] += mpv * tfv[fwd_tfv_idx(p7O_MM, nq, nlane, Q)];
      } else {
        curr[cell + 1] = 0.0f;
      }
    }

    xC = xC * xf_c_loop;
    xJ = (xB * xf_j_move) + (xJ * xf_j_loop);
    xN = (xB * xf_n_move) + (xN * xf_n_loop);
    xE = (xC * xf_e_move) + (xJ * xf_e_loop);

    for (int c = N - 1; c >= 0; c--) {
      curr[c * 3 + 0] += xE;
      curr[c * 3 + 1] += xE;
      if (c + 1 < N) {
        int q = cell_q(c, Q);
        int lane = cell_lane(c, Q);
        curr[c * 3 + 1] += curr[(c + 1) * 3 + 1] * tfv[fwd_tfv_idx(p7O_DD, q, lane, Q)];
      }
    }
    for (int c = N - 2; c >= 0; c--) {
      int q = cell_q(c, Q);
      int lane = cell_lane(c, Q);
      curr[c * 3 + 0] += curr[(c + 1) * 3 + 1] * tfv[fwd_tfv_idx(p7O_MD, q, lane, Q)];
    }

    scale = sxfwd[i * p7X_NXCELLS + p7X_SCALE];
    if (scale > 1.0f) {
      float inv = 1.0f / scale;
      xE *= inv;
      xN *= inv;
      xJ *= inv;
      xB *= inv;
      xC *= inv;
      for (int c = 0; c < N * 3; c++) curr[c] *= inv;
      totscale += logf(scale);
    }

    sxbck[i * p7X_NXCELLS + p7X_E]     = xE;
    sxbck[i * p7X_NXCELLS + p7X_N]     = xN;
    sxbck[i * p7X_NXCELLS + p7X_J]     = xJ;
    sxbck[i * p7X_NXCELLS + p7X_B]     = xB;
    sxbck[i * p7X_NXCELLS + p7X_C]     = xC;
    sxbck[i * p7X_NXCELLS + p7X_SCALE] = scale;

    float *tmp = next;
    next = curr;
    curr = tmp;
  }

  if (L >= 1) {
    uint8_t x = sdsq[1];
    if (x >= Kp) {
      scores[b * 2 + 1] = 0.0f;
      statuses[b * 2 + 1] = eslEINVAL;
      return;
    }
    xB = 0.0f;
    for (int c = 0; c < N; c++) {
      int q = cell_q(c, Q);
      int lane = cell_lane(c, Q);
      float mpv = next[c * 3 + 0] * rfv[((int) x * Q + q) * 4 + lane];
      xB += mpv * tfv[fwd_tfv_idx(p7O_BM, q, lane, Q)];
    }
    xN = (xB * xf_n_move) + (xN * xf_n_loop);
  }

  sxbck[p7X_E]     = 0.0f;
  sxbck[p7X_N]     = xN;
  sxbck[p7X_J]     = 0.0f;
  sxbck[p7X_B]     = xB;
  sxbck[p7X_C]     = 0.0f;
  sxbck[p7X_SCALE] = 1.0f;

  if (isnan(xN) || (L > 0 && xN == 0.0f) || isinf(xN)) {
    scores[b * 2 + 1] = 0.0f;
    statuses[b * 2 + 1] = eslERANGE;
  } else {
    scores[b * 2 + 1] = totscale + logf(xN);
    statuses[b * 2 + 1] = eslOK;
  }
}

__global__ static void
cuda_backward_parser_xmx_batch_parallel_kernel(const uint8_t *dsq, const int *offsets, const int *lengths,
                                               const int *seqidx, int nidx, const size_t *x_offsets,
                                               const float *rfv, const float *tfv, int M, int Q, int Kp,
                                               float xf_e_loop, float xf_e_move, float xf_n_loop_base, float xf_n_move_base,
                                               float xf_c_loop_base, float xf_c_move_base, float xf_j_loop_base, float xf_j_move_base,
                                               float nj, const float *xfwd, float *xbck, float *scores, int *statuses)
{
  extern __shared__ float bck_parser_parallel_mem[];
  int b = blockIdx.x;
  int tid = threadIdx.x;
  int T = blockDim.x;
  int N = Q * 4;
  float *next = bck_parser_parallel_mem;
  float *curr = next + (size_t) N * 3;
  float *bcoef = curr + (size_t) N * 3;
  float *scanA = bcoef + N;
  float *scanB = scanA + T;
  __shared__ float sxE, sxN, sxJ, sxB, sxC, stotscale;
  __shared__ float sscale, sinv;
  __shared__ int si, L;
  __shared__ const uint8_t *sdsq;
  __shared__ const float *sxfwd;
  __shared__ float *sxbck;
  __shared__ float xf_n_loop, xf_n_move, xf_c_loop, xf_c_move, xf_j_loop, xf_j_move;

  if (b >= nidx) return;
  if (tid == 0) {
    si = seqidx ? seqidx[b] : b;
    L = lengths[si];
    sdsq = dsq + offsets[si];
    sxfwd = xfwd + x_offsets[b];
    sxbck = xbck + x_offsets[b];
    float pmove = (2.0f + nj) / ((float) L + 2.0f + nj);
    float ploop = 1.0f - pmove;
    xf_n_loop = xf_n_loop_base >= 0.0f ? xf_n_loop_base : ploop;
    xf_n_move = xf_n_move_base >= 0.0f ? xf_n_move_base : pmove;
    xf_c_loop = xf_c_loop_base >= 0.0f ? xf_c_loop_base : ploop;
    xf_c_move = xf_c_move_base >= 0.0f ? xf_c_move_base : pmove;
    xf_j_loop = xf_j_loop_base >= 0.0f ? xf_j_loop_base : ploop;
    xf_j_move = xf_j_move_base >= 0.0f ? xf_j_move_base : pmove;
    sxJ = 0.0f;
    sxB = 0.0f;
    sxN = 0.0f;
    sxC = xf_c_move;
    sxE = sxC * xf_e_move;
    stotscale = 0.0f;
  }
  __syncthreads();

  for (int c = tid; c < N; c += T) {
    next[c * 3 + 0] = sxE;
    next[c * 3 + 1] = sxE;
    next[c * 3 + 2] = 0.0f;
  }
  __syncthreads();

  if (tid < T) {
    float a0, b0, a1, b1;
    int c0 = N - 1 - tid * 2;
    int c1 = c0 - 1;
    if (c0 >= 0 && c0 < N) {
      a0 = next[c0 * 3 + 1];
      b0 = (c0 + 1 < N) ? tfv[fwd_tfv_idx(p7O_DD, cell_q(c0, Q), cell_lane(c0, Q), Q)] : 0.0f;
      if (c1 >= 0) {
        a1 = next[c1 * 3 + 1];
        b1 = tfv[fwd_tfv_idx(p7O_DD, cell_q(c1, Q), cell_lane(c1, Q), Q)];
        scanA[tid] = a1 + b1 * a0;
        scanB[tid] = b1 * b0;
      } else {
        scanA[tid] = a0;
        scanB[tid] = b0;
      }
    } else if (tid < T) {
      scanA[tid] = 0.0f;
      scanB[tid] = 1.0f;
    }
  }
  __syncthreads();
  for (int off = 1; off < T; off <<= 1) {
    float a_prev = 0.0f, b_prev = 1.0f, a_cur = 0.0f, b_cur = 1.0f;
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
    int c0 = N - 1 - tid * 2;
    int c1 = c0 - 1;
    float incoming = (tid == 0) ? 0.0f : scanA[tid - 1];
    if (c0 >= 0 && c0 < N) {
      float b0 = (c0 + 1 < N) ? tfv[fwd_tfv_idx(p7O_DD, cell_q(c0, Q), cell_lane(c0, Q), Q)] : 0.0f;
      float d0 = next[c0 * 3 + 1] + b0 * incoming;
      next[c0 * 3 + 1] = d0;
      if (c1 >= 0) {
        float b1 = tfv[fwd_tfv_idx(p7O_DD, cell_q(c1, Q), cell_lane(c1, Q), Q)];
        next[c1 * 3 + 1] = next[c1 * 3 + 1] + b1 * d0;
      }
    }
  }
  __syncthreads();
  for (int c = tid; c < N - 1; c += T) {
    next[c * 3 + 0] += next[(c + 1) * 3 + 1] * tfv[fwd_tfv_idx(p7O_MD, cell_q(c, Q), cell_lane(c, Q), Q)];
  }
  __syncthreads();

  if (tid == 0) {
    sscale = sxfwd[L * p7X_NXCELLS + p7X_SCALE];
    if (sscale > 1.0f) {
      sinv = 1.0f / sscale;
      sxE *= sinv;
      sxN *= sinv;
      sxC *= sinv;
      sxJ *= sinv;
      sxB *= sinv;
      stotscale += logf(sscale);
    } else {
      sinv = 1.0f;
    }
    sxbck[L * p7X_NXCELLS + p7X_E]     = sxE;
    sxbck[L * p7X_NXCELLS + p7X_N]     = sxN;
    sxbck[L * p7X_NXCELLS + p7X_J]     = sxJ;
    sxbck[L * p7X_NXCELLS + p7X_B]     = sxB;
    sxbck[L * p7X_NXCELLS + p7X_C]     = sxC;
    sxbck[L * p7X_NXCELLS + p7X_SCALE] = sscale;
  }
  __syncthreads();
  if (sscale > 1.0f) {
    for (int c = tid; c < N * 3; c += T) next[c] *= sinv;
  }
  __syncthreads();

  for (int i = L - 1; i >= 1; i--) {
    uint8_t x = sdsq[i + 1];
    if (x >= Kp) {
      if (tid == 0) {
        scores[b * 2 + 1] = 0.0f;
        statuses[b * 2 + 1] = eslEINVAL;
      }
      return;
    }

    float partial = 0.0f;
    for (int c = tid; c < N; c += T) {
      int q = cell_q(c, Q);
      int lane = cell_lane(c, Q);
      float mpv = next[c * 3 + 0] * rfv[((int) x * Q + q) * 4 + lane];
      partial += mpv * tfv[fwd_tfv_idx(p7O_BM, q, lane, Q)];
    }
    scanA[tid] = partial;
    __syncthreads();
    for (int off = T >> 1; off > 0; off >>= 1) {
      if (tid < off) scanA[tid] += scanA[tid + off];
      __syncthreads();
    }
    if (tid == 0) {
      sxB = scanA[0];
      sxC = sxC * xf_c_loop;
      sxJ = (sxB * xf_j_move) + (sxJ * xf_j_loop);
      sxN = (sxB * xf_n_move) + (sxN * xf_n_loop);
      sxE = (sxC * xf_e_move) + (sxJ * xf_e_loop);
    }
    __syncthreads();

    for (int c = tid; c < N; c += T) {
      int cell = c * 3;
      float mpv = 0.0f;
      if (c + 1 < N) {
        int nq = cell_q(c + 1, Q);
        int nlane = cell_lane(c + 1, Q);
        mpv = next[(c + 1) * 3 + 0] * rfv[((int) x * Q + nq) * 4 + nlane];
      }
      int q = cell_q(c, Q);
      int lane = cell_lane(c, Q);
      curr[cell + 2] = next[cell + 2] * tfv[fwd_tfv_idx(p7O_II, q, lane, Q)];
      curr[cell + 0] = next[cell + 2] * tfv[fwd_tfv_idx(p7O_MI, q, lane, Q)];
      if (c + 1 < N) {
        int nq = cell_q(c + 1, Q);
        int nlane = cell_lane(c + 1, Q);
        curr[cell + 2] += mpv * tfv[fwd_tfv_idx(p7O_IM, nq, nlane, Q)];
        curr[cell + 1]  = mpv * tfv[fwd_tfv_idx(p7O_DM, nq, nlane, Q)];
        curr[cell + 0] += mpv * tfv[fwd_tfv_idx(p7O_MM, nq, nlane, Q)];
      } else {
        curr[cell + 1] = 0.0f;
      }
      curr[cell + 0] += sxE;
      curr[cell + 1] += sxE;
      bcoef[c] = (c + 1 < N) ? tfv[fwd_tfv_idx(p7O_DD, q, lane, Q)] : 0.0f;
    }
    __syncthreads();

    if (tid < T) {
      int c0 = N - 1 - tid * 2;
      int c1 = c0 - 1;
      if (c0 >= 0 && c0 < N) {
        float a0 = curr[c0 * 3 + 1];
        float b0 = bcoef[c0];
        if (c1 >= 0) {
          float a1 = curr[c1 * 3 + 1];
          float b1 = bcoef[c1];
          scanA[tid] = a1 + b1 * a0;
          scanB[tid] = b1 * b0;
        } else {
          scanA[tid] = a0;
          scanB[tid] = b0;
        }
      } else {
        scanA[tid] = 0.0f;
        scanB[tid] = 1.0f;
      }
    }
    __syncthreads();
    for (int off = 1; off < T; off <<= 1) {
      float a_prev = 0.0f, b_prev = 1.0f, a_cur = 0.0f, b_cur = 1.0f;
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
      int c0 = N - 1 - tid * 2;
      int c1 = c0 - 1;
      float incoming = (tid == 0) ? 0.0f : scanA[tid - 1];
      if (c0 >= 0 && c0 < N) {
        float d0 = curr[c0 * 3 + 1] + bcoef[c0] * incoming;
        curr[c0 * 3 + 1] = d0;
        if (c1 >= 0) curr[c1 * 3 + 1] = curr[c1 * 3 + 1] + bcoef[c1] * d0;
      }
    }
    __syncthreads();
    for (int c = tid; c < N - 1; c += T) {
      curr[c * 3 + 0] += curr[(c + 1) * 3 + 1] * tfv[fwd_tfv_idx(p7O_MD, cell_q(c, Q), cell_lane(c, Q), Q)];
    }
    __syncthreads();

    if (tid == 0) {
      sscale = sxfwd[i * p7X_NXCELLS + p7X_SCALE];
      if (sscale > 1.0f) {
        sinv = 1.0f / sscale;
        sxE *= sinv;
        sxN *= sinv;
        sxJ *= sinv;
        sxB *= sinv;
        sxC *= sinv;
        stotscale += logf(sscale);
      } else {
        sinv = 1.0f;
      }
      sxbck[i * p7X_NXCELLS + p7X_E]     = sxE;
      sxbck[i * p7X_NXCELLS + p7X_N]     = sxN;
      sxbck[i * p7X_NXCELLS + p7X_J]     = sxJ;
      sxbck[i * p7X_NXCELLS + p7X_B]     = sxB;
      sxbck[i * p7X_NXCELLS + p7X_C]     = sxC;
      sxbck[i * p7X_NXCELLS + p7X_SCALE] = sscale;
    }
    __syncthreads();
    if (sscale > 1.0f) {
      for (int c = tid; c < N * 3; c += T) curr[c] *= sinv;
    }
    __syncthreads();

    float *tmp = next;
    next = curr;
    curr = tmp;
  }

  if (tid == 0) {
    if (L >= 1) {
      uint8_t x = sdsq[1];
      if (x >= Kp) {
        scores[b * 2 + 1] = 0.0f;
        statuses[b * 2 + 1] = eslEINVAL;
        return;
      }
      sxB = 0.0f;
      for (int c = 0; c < N; c++) {
        int q = cell_q(c, Q);
        int lane = cell_lane(c, Q);
        float mpv = next[c * 3 + 0] * rfv[((int) x * Q + q) * 4 + lane];
        sxB += mpv * tfv[fwd_tfv_idx(p7O_BM, q, lane, Q)];
      }
      sxN = (sxB * xf_n_move) + (sxN * xf_n_loop);
    }
    sxbck[p7X_E]     = 0.0f;
    sxbck[p7X_N]     = sxN;
    sxbck[p7X_J]     = 0.0f;
    sxbck[p7X_B]     = sxB;
    sxbck[p7X_C]     = 0.0f;
    sxbck[p7X_SCALE] = 1.0f;
    if (isnan(sxN) || (L > 0 && sxN == 0.0f) || isinf(sxN)) {
      scores[b * 2 + 1] = 0.0f;
      statuses[b * 2 + 1] = eslERANGE;
    } else {
      scores[b * 2 + 1] = stotscale + logf(sxN);
      statuses[b * 2 + 1] = eslOK;
    }
  }
}
extern "C" int
p7_cuda_ForwardBackwardParser(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                              const ESL_DSQ *dsq, int L, P7_OMX *oxf, P7_OMX *oxb,
                              float *ret_fwdsc, float *ret_bcksc,
                              char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int h_statuses[2] = { eslOK, eslOK };
  float h_scores[2] = { 0.0f, 0.0f };
  size_t dsq_bytes;
  size_t xbytes;
  size_t shmem;
  cudaEvent_t h2d0, h2d1, fk0, fk1, bk0, bk1, d2h0, d2h1;

  if (!engine || !cuom || !dsq || L < 0 || !oxf || !oxb) return eslEINVAL;
  if (cuom->Qf * 4 < cuom->M) return eslEINVAL;

  dsq_bytes = (size_t) L + 2;
  xbytes = sizeof(float) * (size_t) (L + 1) * p7X_NXCELLS;
  shmem = sizeof(float) * (size_t) cuom->Qf * 4 * 3 * 2;
  if (shmem > 96 * 1024) {
    if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "CUDA Forward/Backward parser profile M=%d exceeds v1 shared-memory limit", cuom->M);
    return eslERANGE;
  }

  if ((status = p7_omx_GrowTo(oxf, cuom->M, 0, L)) != eslOK) return status;
  if ((status = p7_omx_GrowTo(oxb, cuom->M, 0, L)) != eslOK) return status;

  if (engine->dsq_alloc < (int) dsq_bytes) {
    if (engine->d_dsq) cudaFree(engine->d_dsq);
    engine->d_dsq = NULL;
    engine->dsq_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_dsq, dsq_bytes), errbuf, errbuf_size, "cudaMalloc(parser dsq)")) != eslOK) return status;
    engine->dsq_alloc = (int) dsq_bytes;
  }
  if (engine->parser_allocL < L + 1) {
    if (engine->d_parser_xf) cudaFree(engine->d_parser_xf);
    if (engine->d_parser_xb) cudaFree(engine->d_parser_xb);
    engine->d_parser_xf = NULL;
    engine->d_parser_xb = NULL;
    engine->parser_allocL = 0;
    engine->parser_cell_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_xf, xbytes), errbuf, errbuf_size, "cudaMalloc(parser forward xmx)")) != eslOK) return status;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_xb, xbytes), errbuf, errbuf_size, "cudaMalloc(parser backward xmx)")) != eslOK) return status;
    engine->parser_allocL = L + 1;
    engine->parser_cell_alloc = (size_t) (L + 1) * p7X_NXCELLS;
  }
  if (engine->d_parser_scores == NULL) {
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_scores, sizeof(float) * 2), errbuf, errbuf_size, "cudaMalloc(parser scores)")) != eslOK) return status;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_statuses, sizeof(int) * 2), errbuf, errbuf_size, "cudaMalloc(parser statuses)")) != eslOK) return status;
    engine->parser_result_alloc = 1;
  }

  cudaEventCreate(&h2d0);
  cudaEventCreate(&h2d1);
  cudaEventCreate(&fk0);
  cudaEventCreate(&fk1);
  cudaEventCreate(&bk0);
  cudaEventCreate(&bk1);
  cudaEventCreate(&d2h0);
  cudaEventCreate(&d2h1);

  cudaEventRecord(h2d0);
  if ((status = cuda_status(cudaMemcpy(engine->d_dsq, dsq, dsq_bytes, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(parser dsq)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemset(engine->d_parser_statuses, 0, sizeof(int) * 2), errbuf, errbuf_size, "cudaMemset(parser statuses)")) != eslOK) goto ERROR;
  cudaEventRecord(h2d1);
  cudaEventSynchronize(h2d1);

  cudaEventRecord(fk0);
  if (shmem > 48 * 1024) {
    if ((status = cuda_status(cudaFuncSetAttribute(cuda_forward_parser_xmx_kernel,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   (int) shmem),
                              errbuf, errbuf_size, "cudaFuncSetAttribute(forward parser shared memory)")) != eslOK) goto ERROR;
  }
  cuda_forward_parser_xmx_kernel<<<1, 1, shmem>>>(engine->d_dsq, L,
                                                  cuom->d_rfv, cuom->d_tfv, cuom->M, cuom->Qf, cuom->Kp,
                                                  cuom->xf_e_loop, cuom->xf_e_move,
                                                  cuom->xf_n_loop, cuom->xf_n_move,
                                                  cuom->xf_c_loop, cuom->xf_c_move,
                                                  cuom->xf_j_loop, cuom->xf_j_move,
                                                  cuom->nj, engine->d_parser_xf,
                                                  engine->d_parser_scores, engine->d_parser_statuses);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_forward_parser_xmx_kernel launch")) != eslOK) goto ERROR;
  cudaEventRecord(fk1);
  cudaEventSynchronize(fk1);

  cudaEventRecord(bk0);
  if (shmem > 48 * 1024) {
    if ((status = cuda_status(cudaFuncSetAttribute(cuda_backward_parser_xmx_kernel,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   (int) shmem),
                              errbuf, errbuf_size, "cudaFuncSetAttribute(backward parser shared memory)")) != eslOK) goto ERROR;
  }
  cuda_backward_parser_xmx_kernel<<<1, 1, shmem>>>(engine->d_dsq, L,
                                                   cuom->d_rfv, cuom->d_tfv, cuom->M, cuom->Qf, cuom->Kp,
                                                   cuom->xf_e_loop, cuom->xf_e_move,
                                                   cuom->xf_n_loop, cuom->xf_n_move,
                                                   cuom->xf_c_loop, cuom->xf_c_move,
                                                   cuom->xf_j_loop, cuom->xf_j_move,
                                                   cuom->nj, engine->d_parser_xf, engine->d_parser_xb,
                                                   engine->d_parser_scores, engine->d_parser_statuses);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_backward_parser_xmx_kernel launch")) != eslOK) goto ERROR;
  cudaEventRecord(bk1);
  cudaEventSynchronize(bk1);

  cudaEventRecord(d2h0);
  if ((status = cuda_status(cudaMemcpy(oxf->xmx, engine->d_parser_xf, xbytes, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(parser forward xmx)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(oxb->xmx, engine->d_parser_xb, xbytes, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(parser backward xmx)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(h_scores, engine->d_parser_scores, sizeof(float) * 2, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(parser scores)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(h_statuses, engine->d_parser_statuses, sizeof(int) * 2, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(parser statuses)")) != eslOK) goto ERROR;
  cudaEventRecord(d2h1);
  cudaEventSynchronize(d2h1);

  engine->stats.fwd_kernel_seconds += elapsed_seconds(fk0, fk1);
  engine->stats.fwd_nseqs          += 1;
  engine->stats.fwd_nres           += L;
  engine->stats.fwd_nbatches       += 1;
  engine->stats.fwd_parser_nseqs   += 1;
  engine->stats.fwd_parser_nres    += L;
  engine->stats.fwd_parser_nbatches += 1;
  engine->stats.bck_h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.bck_kernel_seconds += elapsed_seconds(bk0, bk1);
  engine->stats.bck_d2h_seconds    += elapsed_seconds(d2h0, d2h1);
  engine->stats.bck_nseqs          += 1;
  engine->stats.bck_nres           += L;
  engine->stats.bck_nbatches       += 1;

  oxf->M = cuom->M;
  oxf->L = L;
  oxf->has_own_scales = TRUE;
  oxf->totscale = h_scores[0];
  oxb->M = cuom->M;
  oxb->L = L;
  oxb->has_own_scales = FALSE;
  oxb->totscale = h_scores[1];

  if (ret_fwdsc) *ret_fwdsc = h_scores[0];
  if (ret_bcksc) *ret_bcksc = h_scores[1];
  if (h_statuses[0] != eslOK) { status = h_statuses[0]; goto ERROR; }
  if (h_statuses[1] != eslOK) { status = h_statuses[1]; goto ERROR; }

ERROR:
  cudaEventDestroy(h2d0);
  cudaEventDestroy(h2d1);
  cudaEventDestroy(fk0);
  cudaEventDestroy(fk1);
  cudaEventDestroy(bk0);
  cudaEventDestroy(bk1);
  cudaEventDestroy(d2h0);
  cudaEventDestroy(d2h1);
  return status;
}

extern "C" int
p7_cuda_ForwardBackwardParserDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                           ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                           const size_t *x_offsets, size_t total_xcells,
                                           float *xf, float *xb, float *scores, int *statuses,
                                           char *errbuf, int errbuf_size);

/* run_modes bit flags for fb_parser_subset_ex: */
#define FB_RUN_FORWARD  0x1
#define FB_RUN_BACKWARD 0x2

/* Internal core implementation; public wrappers delegate to this.
 * - If run_modes & FB_RUN_FORWARD: run Forward kernel, write xf D2H.
 * - If run_modes & FB_RUN_BACKWARD: run Backward kernel, write xb D2H.
 *   When Forward is NOT run in the same call, xf is INPUT (H2D'd to device)
 *   so the Backward kernel can consume pre-computed Forward xmx. */
static int
fb_parser_subset_ex(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                    ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                    const size_t *x_offsets, size_t total_xcells,
                    float *xf, float *xb, float *scores, int *statuses,
                    int run_modes,
                    char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int nseq;
  int total = 0;
  int *h_offsets = NULL;
  int *h_lengths = NULL;
  int reuse_batch = FALSE;
  int use_prefix_fwd = FALSE;
  int fwd_threads = 1;
  int use_parallel_bck = FALSE;
  int bck_threads = 1;
  size_t xbytes;
  size_t shmem;
  size_t fwd_shmem;
  size_t bck_shmem;
  cudaEvent_t h2d0, h2d1, fk0, fk1, bk0, bk1, d2h0, d2h1;

  if (!engine || !cuom || !chu || !seqidx || nidx < 0 || !x_offsets || !scores || !statuses) return eslEINVAL;
  if ((run_modes & (FB_RUN_FORWARD | FB_RUN_BACKWARD)) == 0) return eslEINVAL;
  if ((run_modes & FB_RUN_FORWARD) && !xf) return eslEINVAL;
  if ((run_modes & FB_RUN_BACKWARD) && !xb) return eslEINVAL;
  /* Backward-only needs xf as input (H2D'd by us). */
  if (!(run_modes & FB_RUN_FORWARD) && (run_modes & FB_RUN_BACKWARD) && !xf) return eslEINVAL;
  nseq = chu->N;
  if (nidx <= 0) return eslOK;
  if (nseq <= 0) return eslOK;
  if (cuom->Qf * 4 < cuom->M) return eslEINVAL;

  shmem = sizeof(float) * (size_t) cuom->Qf * 4 * 3 * 2;
  if (shmem > 96 * 1024) {
    if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "CUDA Forward/Backward parser profile M=%d exceeds v1 shared-memory limit", cuom->M);
    return eslERANGE;
  }
  if (cuom->Qf * 4 <= 2048) {
    fwd_threads = next_pow2_at_least((cuom->Qf * 4 + 1) / 2, 32);
    if (fwd_threads <= 1024) use_prefix_fwd = TRUE;
  }
  fwd_shmem = use_prefix_fwd ? (sizeof(float) * ((size_t) cuom->Qf * 4 * 3 * 2 + (size_t) cuom->Qf * 4 + (size_t) fwd_threads * 2))
                             : shmem;
  if (use_prefix_fwd && fwd_shmem > 48 * 1024) {
    int max_dynamic_shmem = 0;
    cudaError_t attr_status = cudaDeviceGetAttribute(&max_dynamic_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, engine->device_id);
    if (attr_status != cudaSuccess || fwd_shmem > (size_t) max_dynamic_shmem) {
      use_prefix_fwd = FALSE;
      fwd_shmem = shmem;
    }
  }
  if (cuom->Qf * 4 <= 2048) {
    bck_threads = next_pow2_at_least((cuom->Qf * 4 + 1) / 2, 32);
    if (bck_threads <= 1024) use_parallel_bck = TRUE;
  }
  bck_shmem = use_parallel_bck ? (sizeof(float) * ((size_t) cuom->Qf * 4 * 3 * 2 + (size_t) cuom->Qf * 4 + (size_t) bck_threads * 2))
                               : shmem;
  if (use_parallel_bck && bck_shmem > 48 * 1024) {
    int max_dynamic_shmem = 0;
    cudaError_t attr_status = cudaDeviceGetAttribute(&max_dynamic_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, engine->device_id);
    if (attr_status != cudaSuccess || bck_shmem > (size_t) max_dynamic_shmem) {
      use_parallel_bck = FALSE;
      bck_shmem = shmem;
    }
  }
  if (total_xcells == 0 || total_xcells > SIZE_MAX / sizeof(float)) return eslERANGE;
  xbytes = sizeof(float) * total_xcells;

  int use_resident = (engine->resident_active && engine->resident_batch_nseq == nseq);

  if (!use_resident) {
    h_offsets = (int *) malloc(sizeof(int) * nseq);
    h_lengths = (int *) malloc(sizeof(int) * nseq);
    if (!h_offsets || !h_lengths) { status = eslEMEM; goto ERROR; }

    for (int i = 0; i < nseq; i++) {
      h_offsets[i] = total;
      if (chu->L[i] > INT32_MAX) {
        if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "dsqdata sequence length exceeds CUDA parser v1 limit");
        status = eslERANGE;
        goto ERROR;
      }
      h_lengths[i] = (int) chu->L[i];
      total += h_lengths[i] + 1;
    }
    total += 1;
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
  }
  if (engine->parser_result_alloc < nidx || engine->d_parser_seqidx == NULL || engine->d_parser_x_offsets == NULL) {
    if (engine->d_parser_scores) cudaFree(engine->d_parser_scores);
    if (engine->d_parser_statuses) cudaFree(engine->d_parser_statuses);
    if (engine->d_parser_seqidx) cudaFree(engine->d_parser_seqidx);
    if (engine->d_parser_x_offsets) cudaFree(engine->d_parser_x_offsets);
    engine->d_parser_scores = NULL;
    engine->d_parser_statuses = NULL;
    engine->d_parser_seqidx = NULL;
    engine->d_parser_x_offsets = NULL;
    engine->parser_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_scores, sizeof(float) * 2 * nidx), errbuf, errbuf_size, "cudaMalloc(parser scores batch)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_statuses, sizeof(int) * 2 * nidx), errbuf, errbuf_size, "cudaMalloc(parser statuses batch)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_seqidx, sizeof(int) * nidx), errbuf, errbuf_size, "cudaMalloc(parser seqidx)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_x_offsets, sizeof(size_t) * nidx), errbuf, errbuf_size, "cudaMalloc(parser x offsets)")) != eslOK) goto ERROR;
    engine->parser_result_alloc = nidx;
  }
  if (engine->parser_cell_alloc < total_xcells) {
    if (engine->d_parser_xf) cudaFree(engine->d_parser_xf);
    if (engine->d_parser_xb) cudaFree(engine->d_parser_xb);
    engine->d_parser_xf = NULL;
    engine->d_parser_xb = NULL;
    engine->parser_allocL = 0;
    engine->parser_cell_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_xf, xbytes), errbuf, errbuf_size, "cudaMalloc(parser forward xmx batch)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_xb, xbytes), errbuf, errbuf_size, "cudaMalloc(parser backward xmx batch)")) != eslOK) goto ERROR;
    engine->parser_cell_alloc = total_xcells;
  }

  cudaEventCreate(&h2d0);
  cudaEventCreate(&h2d1);
  cudaEventCreate(&fk0);
  cudaEventCreate(&fk1);
  cudaEventCreate(&bk0);
  cudaEventCreate(&bk1);
  cudaEventCreate(&d2h0);
  cudaEventCreate(&d2h1);

  cudaEventRecord(h2d0);
  if (!use_resident && !reuse_batch) {
    if (chu->smem != NULL) {
      memcpy(engine->h_dsq, chu->smem, total);
    } else {
      for (int i = 0; i < nseq; i++)
        memcpy(engine->h_dsq + h_offsets[i], chu->dsq[i], h_lengths[i] + 1);
    }
    if ((status = cuda_status(cudaMemcpy(engine->d_dsq, engine->h_dsq, total, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(parser batch dsq)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_offsets, h_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(parser offsets)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_lengths, h_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(parser lengths)")) != eslOK) goto CUDA_ERROR;
  }
  if ((status = cuda_status(cudaMemcpy(engine->d_parser_seqidx, seqidx, sizeof(int) * nidx, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(parser seqidx)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_parser_x_offsets, x_offsets, sizeof(size_t) * nidx, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(parser x offsets)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemset(engine->d_parser_statuses, 0, sizeof(int) * 2 * nidx), errbuf, errbuf_size, "cudaMemset(parser statuses batch)")) != eslOK) goto CUDA_ERROR;
  /* Backward-only path: upload pre-computed xf to device. */
  if (!(run_modes & FB_RUN_FORWARD) && (run_modes & FB_RUN_BACKWARD)) {
    if ((status = cuda_status(cudaMemcpy(engine->d_parser_xf, xf, xbytes, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(parser forward xmx H2D)")) != eslOK) goto CUDA_ERROR;
  }
  cudaEventRecord(h2d1);
  cudaEventSynchronize(h2d1);

  cudaEventRecord(fk0);
  {
    uint8_t *d_dsq_ptr = use_resident ? engine->d_resident_dsq : engine->d_dsq;
    int     *d_off_ptr = use_resident ? (engine->d_resident_offsets + engine->resident_batch_seq0) : engine->d_offsets;
    int     *d_len_ptr = use_resident ? (engine->d_resident_lengths + engine->resident_batch_seq0) : engine->d_lengths;
  if (run_modes & FB_RUN_FORWARD) {
  if (use_prefix_fwd) {
    if (fwd_shmem > 48 * 1024) {
      if ((status = cuda_status(cudaFuncSetAttribute(cuda_forward_parser_xmx_batch_parallel_kernel,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     (int) fwd_shmem),
                                errbuf, errbuf_size, "cudaFuncSetAttribute(forward parser parallel shared memory)")) != eslOK) goto CUDA_ERROR;
    }
    cuda_forward_parser_xmx_batch_parallel_kernel<<<nidx, fwd_threads, fwd_shmem>>>(d_dsq_ptr, d_off_ptr, d_len_ptr,
                                                                                     engine->d_parser_seqidx, nidx, engine->d_parser_x_offsets,
                                                                                     cuom->d_rfv, cuom->d_tfv, cuom->M, cuom->Qf, cuom->Kp,
                                                                                     cuom->xf_e_loop, cuom->xf_e_move,
                                                                                     cuom->xf_n_loop, cuom->xf_n_move,
                                                                                     cuom->xf_c_loop, cuom->xf_c_move,
                                                                                     cuom->xf_j_loop, cuom->xf_j_move,
                                                                                     cuom->nj, engine->d_parser_xf,
                                                                                     engine->d_parser_scores, engine->d_parser_statuses);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_forward_parser_xmx_batch_parallel_kernel launch")) != eslOK) goto CUDA_ERROR;
  } else {
    if (shmem > 48 * 1024) {
      if ((status = cuda_status(cudaFuncSetAttribute(cuda_forward_parser_xmx_batch_kernel,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     (int) shmem),
                                errbuf, errbuf_size, "cudaFuncSetAttribute(forward parser shared memory)")) != eslOK) goto CUDA_ERROR;
    }
    cuda_forward_parser_xmx_batch_kernel<<<nidx, 1, shmem>>>(d_dsq_ptr, d_off_ptr, d_len_ptr,
                                                             engine->d_parser_seqidx, nidx, engine->d_parser_x_offsets,
                                                             cuom->d_rfv, cuom->d_tfv, cuom->M, cuom->Qf, cuom->Kp,
                                                             cuom->xf_e_loop, cuom->xf_e_move,
                                                             cuom->xf_n_loop, cuom->xf_n_move,
                                                             cuom->xf_c_loop, cuom->xf_c_move,
                                                             cuom->xf_j_loop, cuom->xf_j_move,
                                                             cuom->nj, engine->d_parser_xf,
                                                             engine->d_parser_scores, engine->d_parser_statuses);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_forward_parser_xmx_batch_kernel launch")) != eslOK) goto CUDA_ERROR;
  }
  }  /* end if (run_modes & FB_RUN_FORWARD) */
  cudaEventRecord(fk1);
  cudaEventSynchronize(fk1);

  cudaEventRecord(bk0);
  if (run_modes & FB_RUN_BACKWARD) {
  if (use_parallel_bck) {
    if (bck_shmem > 48 * 1024) {
      if ((status = cuda_status(cudaFuncSetAttribute(cuda_backward_parser_xmx_batch_parallel_kernel,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     (int) bck_shmem),
                                errbuf, errbuf_size, "cudaFuncSetAttribute(backward parser shared memory)")) != eslOK) goto CUDA_ERROR;
    }
    cuda_backward_parser_xmx_batch_parallel_kernel<<<nidx, bck_threads, bck_shmem>>>(d_dsq_ptr, d_off_ptr, d_len_ptr,
                                                                                     engine->d_parser_seqidx, nidx, engine->d_parser_x_offsets,
                                                                                     cuom->d_rfv, cuom->d_tfv, cuom->M, cuom->Qf, cuom->Kp,
                                                                                     cuom->xf_e_loop, cuom->xf_e_move,
                                                                                     cuom->xf_n_loop, cuom->xf_n_move,
                                                                                     cuom->xf_c_loop, cuom->xf_c_move,
                                                                                     cuom->xf_j_loop, cuom->xf_j_move,
                                                                                     cuom->nj, engine->d_parser_xf, engine->d_parser_xb,
                                                                                     engine->d_parser_scores, engine->d_parser_statuses);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_backward_parser_xmx_batch_parallel_kernel launch")) != eslOK) goto CUDA_ERROR;
  } else {
    if (shmem > 48 * 1024) {
      if ((status = cuda_status(cudaFuncSetAttribute(cuda_backward_parser_xmx_batch_kernel,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     (int) shmem),
                                errbuf, errbuf_size, "cudaFuncSetAttribute(backward parser shared memory)")) != eslOK) goto CUDA_ERROR;
    }
    cuda_backward_parser_xmx_batch_kernel<<<nidx, 1, shmem>>>(d_dsq_ptr, d_off_ptr, d_len_ptr,
                                                              engine->d_parser_seqidx, nidx, engine->d_parser_x_offsets,
                                                              cuom->d_rfv, cuom->d_tfv, cuom->M, cuom->Qf, cuom->Kp,
                                                              cuom->xf_e_loop, cuom->xf_e_move,
                                                              cuom->xf_n_loop, cuom->xf_n_move,
                                                              cuom->xf_c_loop, cuom->xf_c_move,
                                                              cuom->xf_j_loop, cuom->xf_j_move,
                                                              cuom->nj, engine->d_parser_xf, engine->d_parser_xb,
                                                              engine->d_parser_scores, engine->d_parser_statuses);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_backward_parser_xmx_batch_kernel launch")) != eslOK) goto CUDA_ERROR;
  }
  }  /* end if (run_modes & FB_RUN_BACKWARD) */
  }  /* end d_dsq_ptr scope */
  cudaEventRecord(bk1);
  cudaEventSynchronize(bk1);

  cudaEventRecord(d2h0);
  if (run_modes & FB_RUN_FORWARD) {
    if ((status = cuda_status(cudaMemcpy(xf, engine->d_parser_xf, xbytes, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(parser forward xmx batch)")) != eslOK) goto CUDA_ERROR;
  }
  if (run_modes & FB_RUN_BACKWARD) {
    if ((status = cuda_status(cudaMemcpy(xb, engine->d_parser_xb, xbytes, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(parser backward xmx batch)")) != eslOK) goto CUDA_ERROR;
  }
  if ((status = cuda_status(cudaMemcpy(scores, engine->d_parser_scores, sizeof(float) * 2 * nidx, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(parser scores batch)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(statuses, engine->d_parser_statuses, sizeof(int) * 2 * nidx, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(parser statuses batch)")) != eslOK) goto CUDA_ERROR;
  cudaEventRecord(d2h1);
  cudaEventSynchronize(d2h1);

  engine->stats.fwd_h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.fwd_kernel_seconds += elapsed_seconds(fk0, fk1);
  engine->stats.fwd_d2h_seconds    += elapsed_seconds(d2h0, d2h1) * 0.5;
  engine->stats.fwd_nseqs          += nidx;
  engine->stats.fwd_nbatches       += 1;
  engine->stats.fwd_parser_nseqs   += nidx;
  engine->stats.fwd_parser_nbatches += 1;
  engine->stats.bck_h2d_seconds    += 0.0;
  engine->stats.bck_kernel_seconds += elapsed_seconds(bk0, bk1);
  engine->stats.bck_d2h_seconds    += elapsed_seconds(d2h0, d2h1) * 0.5;
  engine->stats.bck_nseqs          += nidx;
  engine->stats.bck_nbatches       += 1;
  for (int i = 0; i < nidx; i++) {
    int si = seqidx[i];
    if (si >= 0 && si < nseq) {
      int slen = h_lengths ? h_lengths[si] : (int) chu->L[si];
      engine->stats.fwd_nres += slen;
      engine->stats.bck_nres += slen;
      engine->stats.fwd_parser_nres += slen;
    }
  }

CUDA_ERROR:
  cudaEventDestroy(h2d0);
  cudaEventDestroy(h2d1);
  cudaEventDestroy(fk0);
  cudaEventDestroy(fk1);
  cudaEventDestroy(bk0);
  cudaEventDestroy(bk1);
  cudaEventDestroy(d2h0);
  cudaEventDestroy(d2h1);
ERROR:
  free(h_offsets);
  free(h_lengths);
  return status;
}

extern "C" int
p7_cuda_ForwardBackwardParserDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                           ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                           const size_t *x_offsets, size_t total_xcells,
                                           float *xf, float *xb, float *scores, int *statuses,
                                           char *errbuf, int errbuf_size)
{
  return fb_parser_subset_ex(engine, cuom, chu, seqidx, nidx, x_offsets, total_xcells,
                             xf, xb, scores, statuses,
                             FB_RUN_FORWARD | FB_RUN_BACKWARD,
                             errbuf, errbuf_size);
}

extern "C" int
p7_cuda_ForwardParserDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                   ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                   const size_t *x_offsets, size_t total_xcells,
                                   float *xf, float *scores, int *statuses,
                                   char *errbuf, int errbuf_size)
{
  return fb_parser_subset_ex(engine, cuom, chu, seqidx, nidx, x_offsets, total_xcells,
                             xf, NULL, scores, statuses,
                             FB_RUN_FORWARD,
                             errbuf, errbuf_size);
}

extern "C" int
p7_cuda_BackwardParserDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                    ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                    const size_t *x_offsets, size_t total_xcells,
                                    const float *xf, float *xb, float *scores, int *statuses,
                                    char *errbuf, int errbuf_size)
{
  /* xf is input-only here; the ex path H2Ds it to device before launching Backward.
   * Cast away const because fb_parser_subset_ex reuses the xf slot for both input
   * (Backward-only) and output (Forward-only) semantics. */
  return fb_parser_subset_ex(engine, cuom, chu, seqidx, nidx, x_offsets, total_xcells,
                             (float *)xf, xb, scores, statuses,
                             FB_RUN_BACKWARD,
                             errbuf, errbuf_size);
}
