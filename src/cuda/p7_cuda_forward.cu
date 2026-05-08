#include "p7_cuda_internal.h"
#include <sys/time.h>

static inline double
seconds_now_fwd(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double) tv.tv_sec + (double) tv.tv_usec * 1e-6;
}

static inline int
cuda_wait_and_account_fwd(cudaEvent_t evt, char *errbuf, int errbuf_size, const char *where, double *wait_accum)
{
  int status;
  double t0 = seconds_now_fwd();
  status = cuda_status(cudaEventSynchronize(evt), errbuf, errbuf_size, where);
  if (wait_accum) *wait_accum += (seconds_now_fwd() - t0);
  return status;
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
    }
    __syncthreads();

    for (int c = tid; c < N; c += blockDim.x) {
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
cuda_forward_pass_kernel(const float *scores, const int *statuses, const float *filtersc,
                         int nidx, double mu, double lambda, double F3, int *passed)
{
  int bi = blockIdx.x * blockDim.x + threadIdx.x;
  if (bi >= nidx) return;
  if (statuses[bi] == eslERANGE) {
    passed[bi] = TRUE;
  } else if (statuses[bi] == eslOK) {
    double bits = ((double) scores[bi] - (double) filtersc[bi]) / eslCONST_LOG2;
    double P = (bits < mu) ? 1.0 : exp(-lambda * (bits - mu));
    passed[bi] = (P <= F3);
  } else {
    passed[bi] = FALSE;
  }
}

static int
p7_cuda_ForwardSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                      ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                      const float *filtersc, double ev_mu, double ev_lambda, double F3,
                      float *scores, int *statuses, int *passed,
                      char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int nseq;
  int total = 0;
  int *h_offsets = NULL;
  int *h_lengths = NULL;
  int reuse_batch = FALSE;
  int use_resident = FALSE;
  int use_prefix = FALSE;
  int prefix_threads = 0;
  size_t shmem;
  cudaEvent_t h2d0, h2d1, k0, k1, d2h0, d2h1;
  double submit_overhead = 0.0;
  double wait_barrier = 0.0;

  if (!engine || !cuom || !chu || !statuses) return eslEINVAL;
  if (!scores && !passed) return eslEINVAL;
  if (passed && !filtersc) return eslEINVAL;
  nseq = chu->N;
  if (nidx <= 0) return eslOK;
  if (nseq <= 0) return eslOK;

  use_resident = (engine->resident_active && engine->resident_batch_nseq == nseq);

  if (!use_resident) {
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
  if (engine->fwd_result_alloc < nidx) {
    if (engine->d_fwd_scores) cudaFree(engine->d_fwd_scores);
    if (engine->d_fwd_statuses) cudaFree(engine->d_fwd_statuses);
    if (engine->d_fwd_passed) cudaFree(engine->d_fwd_passed);
    if (engine->d_fwd_filtersc) cudaFree(engine->d_fwd_filtersc);
    if (engine->d_fwd_seqidx) cudaFree(engine->d_fwd_seqidx);
    engine->d_fwd_scores = NULL;
    engine->d_fwd_statuses = NULL;
    engine->d_fwd_passed = NULL;
    engine->d_fwd_filtersc = NULL;
    engine->d_fwd_seqidx = NULL;
    engine->fwd_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_fwd_scores, sizeof(float) * nidx), errbuf, errbuf_size, "cudaMalloc(fwd scores)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_fwd_statuses, sizeof(int) * nidx), errbuf, errbuf_size, "cudaMalloc(fwd statuses)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_fwd_passed, sizeof(int) * nidx), errbuf, errbuf_size, "cudaMalloc(fwd passed)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_fwd_filtersc, sizeof(float) * nidx), errbuf, errbuf_size, "cudaMalloc(fwd filtersc)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_fwd_seqidx, sizeof(int) * nidx), errbuf, errbuf_size, "cudaMalloc(fwd seqidx)")) != eslOK) goto ERROR;
    engine->fwd_result_alloc = nidx;
  }
  if (passed && engine->d_fwd_filtersc == NULL) {
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_fwd_filtersc, sizeof(float) * engine->fwd_result_alloc), errbuf, errbuf_size, "cudaMalloc(fwd filtersc)")) != eslOK) goto ERROR;
  }
  if (passed && engine->d_fwd_passed == NULL) {
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_fwd_passed, sizeof(int) * engine->fwd_result_alloc), errbuf, errbuf_size, "cudaMalloc(fwd passed)")) != eslOK) goto ERROR;
  }
  if (cuom->Qf * 4 <= 2048) {
    prefix_threads = next_pow2_at_least((cuom->Qf * 4 + 1) / 2, 32);
    if (prefix_threads <= 1024) use_prefix = TRUE;
  }
  shmem = use_prefix ? (sizeof(float) * ((size_t) cuom->Qf * 4 * 3 * 2 + (size_t) cuom->Qf * 4 + (size_t) prefix_threads * 2))
                     : (sizeof(float) * (size_t) cuom->Qf * 4 * 3 * 2);

  h2d0 = engine->evt_h2d0;
  h2d1 = engine->evt_h2d1;
  k0   = engine->evt_k0;
  k1   = engine->evt_k1;
  d2h0 = engine->evt_d2h0;
  d2h1 = engine->evt_d2h1;

  cudaEventRecord(h2d0);
  if (!use_resident && !reuse_batch) {
    double tsp0 = seconds_now_fwd();
    if (chu->smem != NULL) {
      memcpy(engine->h_dsq, chu->smem, total);
    } else {
      for (int i = 0; i < nseq; i++)
        memcpy(engine->h_dsq + h_offsets[i], chu->dsq[i], h_lengths[i] + 1);
    }
    submit_overhead += (seconds_now_fwd() - tsp0);
    if ((status = cuda_status(cudaMemcpy(engine->d_dsq, engine->h_dsq, total, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(batch dsq)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_offsets, h_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(offsets)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_lengths, h_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(lengths)")) != eslOK) goto CUDA_ERROR;
  }
  if (seqidx) {
    if ((status = cuda_status(cudaMemcpy(engine->d_fwd_seqidx, seqidx, sizeof(int) * nidx, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(fwd seqidx)")) != eslOK) goto CUDA_ERROR;
  }
  if (passed) {
    if ((status = cuda_status(cudaMemcpy(engine->d_fwd_filtersc, filtersc, sizeof(float) * nidx, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(fwd filtersc)")) != eslOK) goto CUDA_ERROR;
  }
  cudaEventRecord(h2d1);

  cudaEventRecord(k0);
  {
    uint8_t *d_dsq_ptr = use_resident ? engine->d_resident_dsq : engine->d_dsq;
    int     *d_off_ptr = use_resident ? (engine->d_resident_offsets + engine->resident_batch_seq0) : engine->d_offsets;
    int     *d_len_ptr = use_resident ? (engine->d_resident_lengths + engine->resident_batch_seq0) : engine->d_lengths;
    if (use_prefix) {
      if (shmem > 49152) {
        cudaFuncSetAttribute(cuda_forward_score_prefix_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem);
      }
      cuda_forward_score_prefix_kernel<<<nidx, prefix_threads, shmem>>>(d_dsq_ptr, d_off_ptr, d_len_ptr,
                                                      seqidx ? engine->d_fwd_seqidx : NULL, nidx,
                                                      cuom->d_rfv, cuom->d_tfv, cuom->M, cuom->Qf, cuom->Kp,
                                                      cuom->xf_e_loop, cuom->xf_e_move,
                                                      cuom->xf_n_loop, cuom->xf_n_move,
                                                      cuom->xf_c_loop, cuom->xf_c_move,
                                                      cuom->xf_j_loop, cuom->xf_j_move,
                                                      cuom->nj, engine->d_fwd_scores, engine->d_fwd_statuses);
      if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_forward_score_prefix_kernel launch")) != eslOK) goto CUDA_ERROR;
    } else {
      cuda_forward_score_kernel<<<nidx, 32, shmem>>>(d_dsq_ptr, d_off_ptr, d_len_ptr,
                                                     seqidx ? engine->d_fwd_seqidx : NULL, nidx,
                                                     cuom->d_rfv, cuom->d_tfv, cuom->M, cuom->Qf, cuom->Kp,
                                                     cuom->xf_e_loop, cuom->xf_e_move,
                                                     cuom->xf_n_loop, cuom->xf_n_move,
                                                     cuom->xf_c_loop, cuom->xf_c_move,
                                                     cuom->xf_j_loop, cuom->xf_j_move,
                                                     cuom->nj, engine->d_fwd_scores, engine->d_fwd_statuses);
      if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_forward_score_kernel launch")) != eslOK) goto CUDA_ERROR;
    }
  }
  if (passed) {
    int pass_threads = 256;
    int pass_blocks = (nidx + pass_threads - 1) / pass_threads;
    cuda_forward_pass_kernel<<<pass_blocks, pass_threads>>>(engine->d_fwd_scores, engine->d_fwd_statuses, engine->d_fwd_filtersc,
                                                            nidx, ev_mu, ev_lambda, F3, engine->d_fwd_passed);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_forward_pass_kernel launch")) != eslOK) goto CUDA_ERROR;
  }
  cudaEventRecord(k1);

  cudaEventRecord(d2h0);
  if (scores) {
    if ((status = cuda_status(cudaMemcpy(scores, engine->d_fwd_scores, sizeof(float) * nidx, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fwd scores)")) != eslOK) goto CUDA_ERROR;
  }
  if ((status = cuda_status(cudaMemcpy(statuses, engine->d_fwd_statuses, sizeof(int) * nidx, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fwd statuses)")) != eslOK) goto CUDA_ERROR;
  if (passed) {
    if ((status = cuda_status(cudaMemcpy(passed, engine->d_fwd_passed, sizeof(int) * nidx, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(fwd passed)")) != eslOK) goto CUDA_ERROR;
  }
  cudaEventRecord(d2h1);
  engine->stats.submit_overhead_seconds += submit_overhead;
  if ((status = cuda_wait_and_account_fwd(d2h1, errbuf, errbuf_size, "cudaEventSynchronize(fwd d2h1)", &wait_barrier)) != eslOK) goto CUDA_ERROR;
  engine->stats.wait_barrier_seconds += wait_barrier;
  engine->stats.dispatch_wait_seconds += wait_barrier;

  engine->stats.fwd_h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.fwd_kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.fwd_d2h_seconds    += elapsed_seconds(d2h0, d2h1);
  engine->stats.fwd_nseqs          += nidx;
  engine->stats.fwd_prefilter_nseqs += nidx;
  for (int i = 0; i < nidx; i++) {
    int si = seqidx ? seqidx[i] : i;
    if (si >= 0 && si < nseq) {
      int slen = h_lengths ? h_lengths[si] : (int) chu->L[si];
      engine->stats.fwd_nres += slen;
      engine->stats.fwd_prefilter_nres += slen;
    }
  }
  engine->stats.fwd_nbatches       += 1;
  engine->stats.fwd_prefilter_nbatches += 1;

CUDA_ERROR:
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
  return p7_cuda_ForwardSubset(engine, cuom, chu, seqidx, nidx, NULL, 0.0f, 0.0f, 0.0f,
                               scores, statuses, NULL, errbuf, errbuf_size);
}

extern "C" int
p7_cuda_ForwardFilterDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                   ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                   const float *filtersc, double ev_mu, double ev_lambda, double F3,
                                   float *scores, int *statuses, int *passed,
                                   char *errbuf, int errbuf_size)
{
  return p7_cuda_ForwardSubset(engine, cuom, chu, seqidx, nidx, filtersc, ev_mu, ev_lambda, F3,
                               scores, statuses, passed, errbuf, errbuf_size);
}
