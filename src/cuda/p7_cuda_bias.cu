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
  int use_resident = FALSE;
  cudaEvent_t h2d0, h2d1, k0, k1, d2h0, d2h1;
  double ht0;

  (void) bg;
  if (!engine || !chu || !nullsc) return eslEINVAL;
  nseq = chu->N;
  if (nseq <= 0) return eslOK;

  use_resident = (engine->resident_active && engine->resident_batch_nseq == nseq);

  ht0 = host_seconds();
  h_offsets = (int *) malloc(sizeof(int) * nseq);
  h_lengths = (int *) malloc(sizeof(int) * nseq);
  if (!h_offsets || !h_lengths) { status = eslEMEM; goto ERROR; }
  engine->stats.host_malloc_free_seconds += host_seconds() - ht0;

  ht0 = host_seconds();
  for (int i = 0; i < nseq; i++) {
    h_offsets[i] = total;
    if (chu->L[i] > INT32_MAX) {
      if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "dsqdata sequence length exceeds CUDA null v1 limit");
      status = eslERANGE;
      goto ERROR;
    }
    h_lengths[i] = (int) chu->L[i];
    total += h_lengths[i] + 1;
  }
  total += 1;
  engine->stats.host_metadata_loop_seconds += host_seconds() - ht0;
  reuse_batch = (engine->batch_owner == chu && engine->batch_nseq == nseq && engine->batch_total == total);

  if (!use_resident) {
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
  if (engine->null_result_alloc < nseq) {
    if (engine->d_null_scores) cudaFree(engine->d_null_scores);
    engine->d_null_scores = NULL;
    engine->null_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_null_scores, sizeof(float) * nseq), errbuf, errbuf_size, "cudaMalloc(null scores)")) != eslOK) goto ERROR;
    engine->null_result_alloc = nseq;
  }

  ht0 = host_seconds();
  cudaEventCreate(&h2d0);
  cudaEventCreate(&h2d1);
  cudaEventCreate(&k0);
  cudaEventCreate(&k1);
  cudaEventCreate(&d2h0);
  cudaEventCreate(&d2h1);
  engine->stats.host_event_ops_seconds += host_seconds() - ht0;

  cudaEventRecord(h2d0);
  if (!use_resident && !reuse_batch) {
    ht0 = host_seconds();
    if (chu->smem != NULL) {
      memcpy(engine->h_dsq, chu->smem, total);
    } else {
      for (int i = 0; i < nseq; i++)
        memcpy(engine->h_dsq + h_offsets[i], chu->dsq[i], h_lengths[i] + 1);
    }
    engine->stats.host_pack_loop_seconds += host_seconds() - ht0;
    ht0 = host_seconds();
    if ((status = cuda_status(cudaMemcpy(engine->d_dsq, engine->h_dsq, total, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(batch dsq)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_offsets, h_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(offsets)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_lengths, h_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(lengths)")) != eslOK) goto CUDA_ERROR;
    engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  }
  cudaEventRecord(h2d1);
  ht0 = host_seconds();
  cudaEventSynchronize(h2d1);
  engine->stats.host_sync_seconds += host_seconds() - ht0;

  {
    int *d_len_ptr = use_resident ? (engine->d_resident_lengths + engine->resident_batch_seq0) : engine->d_lengths;
    cudaEventRecord(k0);
    cuda_null_score_kernel<<<(nseq + 127) / 128, 128>>>(d_len_ptr, nseq, engine->d_null_scores);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_null_score_kernel launch")) != eslOK) goto CUDA_ERROR;
    cudaEventRecord(k1);
  }
  ht0 = host_seconds();
  cudaEventSynchronize(k1);
  engine->stats.host_sync_seconds += host_seconds() - ht0;

  cudaEventRecord(d2h0);
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(nullsc, engine->d_null_scores, sizeof(float) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(null scores)")) != eslOK) goto CUDA_ERROR;
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(d2h1);
  ht0 = host_seconds();
  cudaEventSynchronize(d2h1);
  engine->stats.host_sync_seconds += host_seconds() - ht0;

  engine->stats.null_h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.null_kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.null_d2h_seconds    += elapsed_seconds(d2h0, d2h1);

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
  engine->stats.host_malloc_free_seconds += host_seconds() - ht0;
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
  int use_resident = FALSE;
  cudaEvent_t h2d0, h2d1, k0, k1, d2h0, d2h1;
  double ht0;

  if (!engine || !bg || !bg->fhmm || !chu || !filtersc) return eslEINVAL;
  nseq = chu->N;
  if (nseq <= 0) return eslOK;

  use_resident = (engine->resident_active && engine->resident_batch_nseq == nseq);

  ht0 = host_seconds();
  h_offsets = (int *) malloc(sizeof(int) * nseq);
  h_lengths = (int *) malloc(sizeof(int) * nseq);
  if (!h_offsets || !h_lengths) { status = eslEMEM; goto ERROR; }
  engine->stats.host_malloc_free_seconds += host_seconds() - ht0;

  ht0 = host_seconds();
  for (int i = 0; i < nseq; i++) {
    h_offsets[i] = total;
    if (chu->L[i] > INT32_MAX) {
      if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "dsqdata sequence length exceeds CUDA bias v1 limit");
      status = eslERANGE;
      goto ERROR;
    }
    h_lengths[i] = (int) chu->L[i];
    total += h_lengths[i] + 1;
  }
  total += 1;
  engine->stats.host_metadata_loop_seconds += host_seconds() - ht0;
  reuse_batch = (engine->batch_owner == chu && engine->batch_nseq == nseq && engine->batch_total == total);

  if (!use_resident) {
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

  ht0 = host_seconds();
  cudaEventCreate(&h2d0);
  cudaEventCreate(&h2d1);
  cudaEventCreate(&k0);
  cudaEventCreate(&k1);
  cudaEventCreate(&d2h0);
  cudaEventCreate(&d2h1);
  engine->stats.host_event_ops_seconds += host_seconds() - ht0;

  cudaEventRecord(h2d0);
  if (!use_resident && !reuse_batch) {
    ht0 = host_seconds();
    if (chu->smem != NULL) {
      memcpy(engine->h_dsq, chu->smem, total);
    } else {
      for (int i = 0; i < nseq; i++)
        memcpy(engine->h_dsq + h_offsets[i], chu->dsq[i], h_lengths[i] + 1);
    }
    engine->stats.host_pack_loop_seconds += host_seconds() - ht0;
    ht0 = host_seconds();
    if ((status = cuda_status(cudaMemcpy(engine->d_dsq, engine->h_dsq, total, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(batch dsq)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_offsets, h_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(offsets)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_lengths, h_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(lengths)")) != eslOK) goto CUDA_ERROR;
    engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  }
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(engine->d_bias_pi, h_pi, sizeof(h_pi), cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(bias pi)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_bias_t, h_t, sizeof(h_t), cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(bias t)")) != eslOK) goto CUDA_ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_bias_eo, bg->fhmm->eo[0], eo_bytes, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(bias eo)")) != eslOK) goto CUDA_ERROR;
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(h2d1);
  ht0 = host_seconds();
  cudaEventSynchronize(h2d1);
  engine->stats.host_sync_seconds += host_seconds() - ht0;

  {
    uint8_t *d_dsq_ptr = use_resident ? engine->d_resident_dsq : engine->d_dsq;
    int     *d_off_ptr = use_resident ? (engine->d_resident_offsets + engine->resident_batch_seq0) : engine->d_offsets;
    int     *d_len_ptr = use_resident ? (engine->d_resident_lengths + engine->resident_batch_seq0) : engine->d_lengths;
    cudaEventRecord(k0);
    cuda_bias_filter_kernel<<<(nseq + 127) / 128, 128>>>(d_dsq_ptr, d_off_ptr, d_len_ptr, nseq,
                                                         engine->d_bias_pi, engine->d_bias_t, engine->d_bias_eo,
                                                         engine->d_bias_filtersc);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_bias_filter_kernel launch")) != eslOK) goto CUDA_ERROR;
    cudaEventRecord(k1);
  }
  ht0 = host_seconds();
  cudaEventSynchronize(k1);
  engine->stats.host_sync_seconds += host_seconds() - ht0;

  cudaEventRecord(d2h0);
  ht0 = host_seconds();
  if ((status = cuda_status(cudaMemcpy(filtersc, engine->d_bias_filtersc, sizeof(float) * nseq, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(bias filtersc)")) != eslOK) goto CUDA_ERROR;
  engine->stats.host_cudamemcpy_seconds += host_seconds() - ht0;
  cudaEventRecord(d2h1);
  ht0 = host_seconds();
  cudaEventSynchronize(d2h1);
  engine->stats.host_sync_seconds += host_seconds() - ht0;

  engine->stats.bias_h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.bias_kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.bias_d2h_seconds    += elapsed_seconds(d2h0, d2h1);
  engine->batch_owner = chu;
  engine->batch_nseq  = nseq;
  engine->batch_total = total;

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
  engine->stats.host_malloc_free_seconds += host_seconds() - ht0;
  return status;
}

__global__ static void
cuda_f1_gating_kernel(const int *raw, const int *overflow, const uint8_t *tjb_by_seq,
                      float scale_b, float base_b,
                      const float *null_scores, const float *bias_filtersc,
                      int nseq, int do_biasfilter,
                      float ev_mu, float ev_lambda, float F1, float log2_inv,
                      int *survivor_idx, int *counter)
{
  int seq = blockIdx.x * blockDim.x + threadIdx.x;
  if (seq >= nseq) return;

  float usc;
  if (overflow[seq]) {
    usc = 1e38f;
  } else {
    usc = ((float)(raw[seq] - (int)tjb_by_seq[seq]) - base_b) / scale_b - 3.0f;
  }

  float nullsc = null_scores[seq];
  float seq_score = (usc - nullsc) * log2_inv;
  float y = ev_lambda * (seq_score - ev_mu);
  float ey = -expf(-y);
  float P = (fabsf(ey) < 1e-7f) ? -ey : 1.0f - expf(ey);

  if (overflow[seq]) P = 0.0f;
  if (P > F1) return;

  if (do_biasfilter) {
    float filtersc = bias_filtersc[seq];
    float bias_score = (usc - filtersc) * log2_inv;
    float by = ev_lambda * (bias_score - ev_mu);
    float bey = -expf(-by);
    float bP = (fabsf(bey) < 1e-7f) ? -bey : 1.0f - expf(bey);
    if (bP > F1) return;
  }

  int idx = atomicAdd(counter, 1);
  survivor_idx[idx] = seq;
}

extern "C" int
p7_cuda_F1GatingDsqdataChunk(P7_CUDA_ENGINE *engine,
                              const float *msv_scores, const int *msv_statuses,
                              int nseq, int do_biasfilter,
                              double ev_mu, double ev_lambda, double F1,
                              int *survivor_idx, int *ret_nsurv,
                              char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int h_counter = 0;

  (void) msv_scores;
  (void) msv_statuses;

  if (!engine || !survivor_idx || !ret_nsurv) return eslEINVAL;
  if (nseq <= 0) { *ret_nsurv = 0; return eslOK; }

  if (engine->f1_result_alloc < nseq) {
    if (engine->d_f1_survivor_idx) cudaFree(engine->d_f1_survivor_idx);
    if (engine->d_f1_counter) cudaFree(engine->d_f1_counter);
    engine->d_f1_survivor_idx = NULL;
    engine->d_f1_counter = NULL;
    engine->f1_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_survivor_idx, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(f1 survivor idx)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_f1_counter, sizeof(int)), errbuf, errbuf_size, "cudaMalloc(f1 counter)")) != eslOK) goto ERROR;
    engine->f1_result_alloc = nseq;
  }

  if ((status = cuda_status(cudaMemcpy(engine->d_f1_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(f1 counter reset)")) != eslOK) goto ERROR;

  {
    const P7_CUDA_MSVPROFILE *cuom = engine->last_cuom;
    float scale_b = cuom->scale_b;
    float base_b = (float) cuom->base_b;
    float log2_inv = 1.0f / 0.693147180559945f;

    cuda_f1_gating_kernel<<<(nseq + 127) / 128, 128>>>(
        engine->d_raw, engine->d_overflow, engine->d_tjb_by_seq,
        scale_b, base_b,
        engine->d_null_scores, engine->d_bias_filtersc,
        nseq, do_biasfilter,
        (float) ev_mu, (float) ev_lambda, (float) F1, log2_inv,
        engine->d_f1_survivor_idx, engine->d_f1_counter);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_f1_gating_kernel launch")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaDeviceSynchronize(), errbuf, errbuf_size, "cuda_f1_gating_kernel sync")) != eslOK) goto ERROR;
  }

  if ((status = cuda_status(cudaMemcpy(&h_counter, engine->d_f1_counter, sizeof(int), cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(f1 counter read)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(survivor_idx, engine->d_f1_survivor_idx, sizeof(int) * h_counter, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(f1 survivor idx)")) != eslOK) goto ERROR;

  *ret_nsurv = h_counter;
  return eslOK;

ERROR:
  *ret_nsurv = 0;
  return status;
}
