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

__global__ static void
cuda_ssv_batch_kernel(const uint8_t *dsq, const int *offsets, const int *lengths, const uint8_t *tjb_by_seq, int nseq,
                      const uint8_t *rbv, int M, int Q, int Kp,
                      uint8_t tbm_b, uint8_t tec_b, uint8_t tjb_b, uint8_t base_b, uint8_t bias_b,
                      int *raw_sc, int *overflow, int *ssv_status)
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

  if ((int) seq_tjb_b + (int) tbm_b + (int) tec_b + (int) bias_b >= 127) {
    if (tid == 0) {
      raw_sc[seq] = 0;
      overflow[seq] = 0;
      ssv_status[seq] = P7_CUDA_SSV_SKIP;
    }
    return;
  }

  uint8_t xE_global_local = 0;
  uint8_t xB_ssv = u8_sub_sat(base_b, (uint8_t) (seq_tjb_b + tbm_b));

  for (int k = tid; k <= M; k += blockDim.x) prev[k] = 0;
  if (tid == 0) overflow[seq] = 0;
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
        ssv_status[seq] = P7_CUDA_SSV_NEED_MSV;
      } else {
        raw_sc[seq] = 0;
        overflow[seq] = 1;
        ssv_status[seq] = P7_CUDA_SSV_OVERFLOW;
      }
    } else if ((int) xE_global_local >= 255 - (int) bias_b) {
      raw_sc[seq] = 0;
      overflow[seq] = 1;
      ssv_status[seq] = P7_CUDA_SSV_OVERFLOW;
    } else {
      int xJ = (int) xE_global_local - (int) tec_b;
      if (xJ > (int) base_b) {
        ssv_status[seq] = P7_CUDA_SSV_NEED_MSV;
      } else {
        raw_sc[seq] = xJ;
        overflow[seq] = 0;
        ssv_status[seq] = P7_CUDA_SSV_OK;
      }
    }
  }
}

__global__ static void
cuda_msv_fallback_kernel(const uint8_t *dsq, const int *offsets, const int *lengths, const uint8_t *tjb_by_seq, int nseq,
                         const uint8_t *rbv, int M, int Q, int Kp,
                         uint8_t tbm_b, uint8_t tec_b, uint8_t tjb_b, uint8_t base_b, uint8_t bias_b,
                         const int *fallback_idx, int nfallback,
                         int *raw_sc, int *overflow)
{
  extern __shared__ uint8_t mem[];
  int fb = blockIdx.x;
  int tid = threadIdx.x;
  if (fb >= nfallback) return;

  int seq = fallback_idx[fb];
  uint8_t *prev = mem;
  uint8_t *curr = prev + M + 1;
  const uint8_t *sdsq = dsq + offsets[seq];
  int L = lengths[seq];
  uint8_t seq_tjb_b = tjb_by_seq[seq];

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
  int *h_ssv_status = NULL;
  int *h_fallback_idx = NULL;
  int nfallback = 0;
  size_t shmem = (size_t) (cuom->M + 1) * 2;
  cudaEvent_t h2d0, h2d1, k0, k1, d2h0, d2h1;
  double ht0;

  if (!engine || !cuom || !chu || !scores || !statuses) return eslEINVAL;
  nseq = chu->N;
  if (nseq <= 0) return eslOK;
  if (shmem > 96 * 1024) {
    if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "CUDA SSV profile M=%d exceeds shared-memory limit", cuom->M);
    return eslERANGE;
  }

  ht0 = host_seconds();
  h_offsets      = (int *)     malloc(sizeof(int) * nseq);
  h_lengths      = (int *)     malloc(sizeof(int) * nseq);
  h_tjb_by_seq   = (uint8_t *) malloc(sizeof(uint8_t) * nseq);
  h_raw          = (int *)     malloc(sizeof(int) * nseq);
  h_overflow     = (int *)     malloc(sizeof(int) * nseq);
  h_ssv_status   = (int *)     malloc(sizeof(int) * nseq);
  h_fallback_idx = (int *)     malloc(sizeof(int) * nseq);
  if (!h_offsets || !h_lengths || !h_tjb_by_seq || !h_raw || !h_overflow || !h_ssv_status || !h_fallback_idx)
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
  if (engine->ssv_alloc < nseq) {
    if (engine->d_ssv_status) cudaFree(engine->d_ssv_status);
    if (engine->d_fallback_idx) cudaFree(engine->d_fallback_idx);
    engine->d_ssv_status = NULL;
    engine->d_fallback_idx = NULL;
    engine->ssv_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_ssv_status, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(ssv_status)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_fallback_idx, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(fallback_idx)")) != eslOK) goto ERROR;
    engine->ssv_alloc = nseq;
  }

  ht0 = host_seconds();
  cudaEventCreate(&h2d0);
  cudaEventCreate(&h2d1);
  cudaEventCreate(&k0);
  cudaEventCreate(&k1);
  cudaEventCreate(&d2h0);
  cudaEventCreate(&d2h1);
  engine->stats.host_event_ops_seconds += host_seconds() - ht0;

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
  ht0 = host_seconds();
  cudaEventSynchronize(h2d1);
  engine->stats.host_sync_seconds += host_seconds() - ht0;

  /* Pass 1: SSV kernel */
  cudaEventRecord(k0);
  cuda_ssv_batch_kernel<<<nseq, P7_CUDA_MSV_BLOCK_THREADS, shmem>>>(engine->d_dsq, engine->d_offsets, engine->d_lengths, engine->d_tjb_by_seq, nseq,
                                                                    cuom->d_rbv, cuom->M, cuom->Q, cuom->Kp,
                                                                    cuom->tbm_b, cuom->tec_b, cuom->tjb_b,
                                                                    cuom->base_b, cuom->bias_b,
                                                                    engine->d_raw, engine->d_overflow, engine->d_ssv_status);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_ssv_batch_kernel launch")) != eslOK) goto CUDA_ERROR;
  cudaEventRecord(k1);
  ht0 = host_seconds();
  cudaEventSynchronize(k1);
  engine->stats.host_sync_seconds += host_seconds() - ht0;

  /* Copy back SSV status to decide fallback */
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(h_ssv_status, engine->d_ssv_status, sizeof(int) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(ssv_status)")) != eslOK) goto CUDA_ERROR;
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;

  /* Build fallback index */
  nfallback = 0;
  for (int i = 0; i < nseq; i++) {
    if (h_ssv_status[i] == P7_CUDA_SSV_NEED_MSV || h_ssv_status[i] == P7_CUDA_SSV_SKIP)
      h_fallback_idx[nfallback++] = i;
  }

  /* Pass 2: full MSV for sequences that need it */
  if (nfallback > 0) {
    cudaEvent_t fk0, fk1;
    cudaEventCreate(&fk0);
    cudaEventCreate(&fk1);

    ht0 = host_seconds();
    if ((status = cuda_status(cudaMemcpy(engine->d_fallback_idx, h_fallback_idx, sizeof(int) * nfallback, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(fallback_idx)")) != eslOK) goto CUDA_ERROR;
    engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;

    cudaEventRecord(fk0);
    cuda_msv_fallback_kernel<<<nfallback, P7_CUDA_MSV_BLOCK_THREADS, shmem>>>(engine->d_dsq, engine->d_offsets, engine->d_lengths, engine->d_tjb_by_seq, nseq,
                                                                              cuom->d_rbv, cuom->M, cuom->Q, cuom->Kp,
                                                                              cuom->tbm_b, cuom->tec_b, cuom->tjb_b,
                                                                              cuom->base_b, cuom->bias_b,
                                                                              engine->d_fallback_idx, nfallback,
                                                                              engine->d_raw, engine->d_overflow);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_msv_fallback_kernel launch")) != eslOK) { cudaEventDestroy(fk0); cudaEventDestroy(fk1); goto CUDA_ERROR; }
    cudaEventRecord(fk1);
    ht0 = host_seconds();
    cudaEventSynchronize(fk1);
    engine->stats.host_sync_seconds += host_seconds() - ht0;
    engine->stats.ssv_fallback_kernel_seconds += elapsed_seconds(fk0, fk1);
    engine->stats.ssv_fallback_nseqs += nfallback;

    cudaEventDestroy(fk0);
    cudaEventDestroy(fk1);
  }

  /* Copy back final results */
  cudaEventRecord(d2h0);
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(h_raw, engine->d_raw, sizeof(int) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(ssv raw)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(h_overflow, engine->d_overflow, sizeof(int) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(ssv overflow)")) != eslOK) goto CUDA_ERROR;
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(d2h1);
  ht0 = host_seconds();
  cudaEventSynchronize(d2h1);
  engine->stats.host_sync_seconds += host_seconds() - ht0;

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
  ht0 = host_seconds();
  cudaEventDestroy(h2d0);
  cudaEventDestroy(h2d1);
  cudaEventDestroy(k0);
  cudaEventDestroy(k1);
  cudaEventDestroy(d2h0);
  cudaEventDestroy(d2h1);
  engine->stats.host_event_ops_seconds += host_seconds() - ht0;
ERROR:
  ht0 = host_seconds();
  free(h_offsets);
  free(h_lengths);
  free(h_tjb_by_seq);
  free(h_raw);
  free(h_overflow);
  free(h_ssv_status);
  free(h_fallback_idx);
  engine->stats.host_malloc_free_seconds += host_seconds() - ht0;
  return status;
}
