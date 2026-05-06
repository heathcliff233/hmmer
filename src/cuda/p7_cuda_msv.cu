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
uint8_t
cuda_unbiased_byteify(const P7_CUDA_MSVPROFILE *cuom, float sc)
{
  sc = -1.0f * roundf(cuom->scale_b * sc);
  return (sc > 255.0f) ? 255 : (uint8_t) sc;
}

int
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
  engine->last_cuom   = cuom;
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
