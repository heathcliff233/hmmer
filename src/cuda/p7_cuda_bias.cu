#include "p7_cuda_internal.h"

__global__ static void
cuda_null_score_kernel(const int *lengths, int nseq, float *nullsc)
{
  int seq = blockIdx.x * blockDim.x + threadIdx.x;
  if (seq >= nseq) return;

  int L = lengths[seq];
  if (L <= 0) {
    nullsc[seq] = 0.0f;
    return;
  }

  {
    float p1 = (float) L / (float) (L + 1);
    nullsc[seq] = (float) L * logf(p1) + logf(1.0f - p1);
  }
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
    n0 = (p0 * t[0] + p1s * t[3]) * eo[(int) x * 2 + 0];
    n1 = (p0 * t[1] + p1s * t[4]) * eo[(int) x * 2 + 1];
    maxv = fmaxf(fmaxf(n0, n1), 0.0f);
    p0 = n0 / maxv;
    p1s = n1 / maxv;
    sc += logf(maxv);
  }

  sc += logf(p0 * t[2] + p1s * t[5]);
  filtersc[seq] = sc + (float) L * logf(len_p1) + logf(1.0f - len_p1);
}
extern "C" int
p7_cuda_NullScoreDsqdataChunk(P7_CUDA_ENGINE *engine, const P7_BG *bg,
                               ESL_DSQDATA_CHUNK *chu, float *nullsc,
                               char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int nseq;
  int total = 0;
  int *h_offsets = NULL;
  int *h_lengths = NULL;
  int reuse_batch = FALSE;
  cudaEvent_t h2d0, h2d1, k0, k1, d2h0, d2h1;

  (void) bg;
  if (!engine || !chu || !nullsc) return eslEINVAL;
  nseq = chu->N;
  if (nseq <= 0) return eslOK;

  h_offsets = (int *) malloc(sizeof(int) * nseq);
  h_lengths = (int *) malloc(sizeof(int) * nseq);
  if (!h_offsets || !h_lengths) { status = eslEMEM; goto ERROR; }

  for (int i = 0; i < nseq; i++) {
    h_offsets[i] = total;
    if (chu->L[i] > INT32_MAX) {
      if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "dsqdata sequence length exceeds CUDA null v1 limit");
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
  if (engine->null_result_alloc < nseq) {
    if (engine->d_null_scores) cudaFree(engine->d_null_scores);
    engine->d_null_scores = NULL;
    engine->null_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_null_scores, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(null scores)")) != eslOK) goto ERROR;
    engine->null_result_alloc = nseq;
  }

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
  cudaEventRecord(h2d1);
  cudaEventSynchronize(h2d1);

  cudaEventRecord(k0);
  cuda_null_score_kernel<<<(nseq + 127) / 128, 128>>>(engine->d_lengths, nseq, engine->d_null_scores);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_null_score_kernel launch")) != eslOK) goto CUDA_ERROR;
  cudaEventRecord(k1);
  cudaEventSynchronize(k1);

  cudaEventRecord(d2h0);
  if ((status = cuda_status(cudaMemcpy(nullsc, engine->d_null_scores, sizeof(float) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(null scores)")) != eslOK) goto CUDA_ERROR;
  cudaEventRecord(d2h1);
  cudaEventSynchronize(d2h1);

  engine->stats.null_h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.null_kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.null_d2h_seconds    += elapsed_seconds(d2h0, d2h1);

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
  if (engine->null_result_alloc < nseq) {
    if (engine->d_null_scores) cudaFree(engine->d_null_scores);
    engine->d_null_scores = NULL;
    engine->null_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_null_scores, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(null scores)")) != eslOK) goto ERROR;
    engine->null_result_alloc = nseq;
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
