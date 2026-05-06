#include "p7_cuda_internal.h"
#include <sys/time.h>

static inline double
seconds_now_vit(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double) tv.tv_sec + (double) tv.tv_usec * 1e-6;
}

static inline int
cuda_wait_and_account_vit(cudaEvent_t evt, char *errbuf, int errbuf_size, const char *where, double *wait_accum)
{
  int status;
  double t0 = seconds_now_vit();
  status = cuda_status(cudaEventSynchronize(evt), errbuf, errbuf_size, where);
  if (wait_accum) *wait_accum += (seconds_now_vit() - t0);
  return status;
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
  int groups_per_block = blockDim.x >> 5;
  int group = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int bi = blockIdx.x * groups_per_block + group;
  int N = Q * 8;
  size_t group_stride = (size_t) N * 3 * 2;
  int16_t *group_mem = vit_mem + ((size_t) group * group_stride);
  int16_t *prev = group_mem;
  int16_t *curr = prev + (size_t) N * 3;
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
cuda_viterbi_pass_kernel(const float *scores, const int *statuses, const float *filtersc,
                         int nidx, double mu, double lambda, double F2, int *passed)
{
  int bi = blockIdx.x * blockDim.x + threadIdx.x;
  if (bi >= nidx) return;
  if (statuses[bi] == eslERANGE) {
    passed[bi] = TRUE;
  } else if (statuses[bi] == eslOK) {
    double bits = ((double) scores[bi] - (double) filtersc[bi]) / eslCONST_LOG2;
    double y = lambda * (bits - mu);
    double ey = exp(-y);
    double P = (ey < eslSMALLX1) ? ey : 1.0 - exp(-ey);
    passed[bi] = (P <= F2);
  } else {
    passed[bi] = FALSE;
  }
}

static int
p7_cuda_ViterbiSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                      ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                      const float *filtersc, double ev_mu, double ev_lambda, double F2,
                      float *scores, int *statuses, int *passed,
                      char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int nseq;
  int total = 0;
  int *h_offsets = NULL;
  int *h_lengths = NULL;
  int reuse_batch = FALSE;
  int threads = 32;
  int groups_per_block = 1;
  size_t group_shmem;
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
    if (engine->d_vit_passed) cudaFree(engine->d_vit_passed);
    if (engine->d_vit_filtersc) cudaFree(engine->d_vit_filtersc);
    if (engine->d_vit_seqidx) cudaFree(engine->d_vit_seqidx);
    engine->d_vit_scores = NULL;
    engine->d_vit_statuses = NULL;
    engine->d_vit_passed = NULL;
    engine->d_vit_filtersc = NULL;
    engine->d_vit_seqidx = NULL;
    engine->vit_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_vit_scores, sizeof(float) * nidx), errbuf, errbuf_size, "cudaMalloc(vit scores)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_vit_statuses, sizeof(int) * nidx), errbuf, errbuf_size, "cudaMalloc(vit statuses)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_vit_passed, sizeof(int) * nidx), errbuf, errbuf_size, "cudaMalloc(vit passed)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_vit_filtersc, sizeof(float) * nidx), errbuf, errbuf_size, "cudaMalloc(vit filtersc)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_vit_seqidx, sizeof(int) * nidx), errbuf, errbuf_size, "cudaMalloc(vit seqidx)")) != eslOK) goto ERROR;
    engine->vit_result_alloc = nidx;
  }
  if (passed && engine->d_vit_filtersc == NULL) {
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_vit_filtersc, sizeof(float) * engine->vit_result_alloc), errbuf, errbuf_size, "cudaMalloc(vit filtersc)")) != eslOK) goto ERROR;
  }
  if (passed && engine->d_vit_passed == NULL) {
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_vit_passed, sizeof(int) * engine->vit_result_alloc), errbuf, errbuf_size, "cudaMalloc(vit passed)")) != eslOK) goto ERROR;
  }
  group_shmem = sizeof(int16_t) * (size_t) cuom->Qw * 8 * 3 * 2;
  if (group_shmem == 0 || group_shmem > (48u * 1024u)) {
    if (errbuf && errbuf_size > 0) snprintf(errbuf, errbuf_size, "CUDA Viterbi shared-memory requirement exceeds v1 limit");
    status = eslERANGE;
    goto ERROR;
  }
  while (groups_per_block < 4 && (size_t) (groups_per_block + 1) * group_shmem <= (48u * 1024u))
    groups_per_block++;
  if (groups_per_block > nidx) groups_per_block = nidx;
  if (groups_per_block < 1) groups_per_block = 1;
  threads = groups_per_block * 32;
  shmem = group_shmem * (size_t) groups_per_block;

  h2d0 = engine->evt_h2d0;
  h2d1 = engine->evt_h2d1;
  k0   = engine->evt_k0;
  k1   = engine->evt_k1;
  d2h0 = engine->evt_d2h0;
  d2h1 = engine->evt_d2h1;

  cudaEventRecord(h2d0);
  if (!reuse_batch) {
    double tsp0 = seconds_now_vit();
    for (int i = 0; i < nseq; i++) memcpy(engine->h_dsq + h_offsets[i], chu->dsq[i], h_lengths[i] + 2);
    submit_overhead += (seconds_now_vit() - tsp0);
    if ((status = cuda_status(cudaMemcpy(engine->d_dsq, engine->h_dsq, total, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(batch dsq)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_offsets, h_offsets, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(offsets)")) != eslOK) goto CUDA_ERROR;
    if ((status = cuda_status(cudaMemcpy(engine->d_lengths, h_lengths, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(lengths)")) != eslOK) goto CUDA_ERROR;
  }
  if (seqidx) {
    if ((status = cuda_status(cudaMemcpy(engine->d_vit_seqidx, seqidx, sizeof(int) * nidx, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(vit seqidx)")) != eslOK) goto CUDA_ERROR;
  }
  if (passed) {
    if ((status = cuda_status(cudaMemcpy(engine->d_vit_filtersc, filtersc, sizeof(float) * nidx, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(vit filtersc)")) != eslOK) goto CUDA_ERROR;
  }
  cudaEventRecord(h2d1);

  cudaEventRecord(k0);
  {
    int blocks = (nidx + groups_per_block - 1) / groups_per_block;
    cuda_viterbi_score_kernel<<<blocks, threads, shmem>>>(engine->d_dsq, engine->d_offsets, engine->d_lengths,
                                                           seqidx ? engine->d_vit_seqidx : NULL, nidx,
                                                           cuom->d_rwv, cuom->d_twv, cuom->M, cuom->Qw, cuom->Kp,
                                                           cuom->xw_n_loop, cuom->xw_n_move,
                                                           cuom->xw_c_loop, cuom->xw_c_move,
                                                           cuom->xw_j_loop, cuom->xw_j_move,
                                                           cuom->xw_e_loop, cuom->xw_e_move,
                                                           cuom->scale_w, cuom->base_w, cuom->ddbound_w, cuom->nj,
                                                           engine->d_vit_scores, engine->d_vit_statuses);
  }
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_viterbi_score_kernel launch")) != eslOK) goto CUDA_ERROR;
  if (passed) {
    int pass_threads = 256;
    int pass_blocks = (nidx + pass_threads - 1) / pass_threads;
    cuda_viterbi_pass_kernel<<<pass_blocks, pass_threads>>>(engine->d_vit_scores, engine->d_vit_statuses, engine->d_vit_filtersc,
                                                            nidx, ev_mu, ev_lambda, F2, engine->d_vit_passed);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_viterbi_pass_kernel launch")) != eslOK) goto CUDA_ERROR;
  }
  cudaEventRecord(k1);

  cudaEventRecord(d2h0);
  if (scores) {
    if ((status = cuda_status(cudaMemcpy(scores, engine->d_vit_scores, sizeof(float) * nidx, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(vit scores)")) != eslOK) goto CUDA_ERROR;
  }
  if ((status = cuda_status(cudaMemcpy(statuses, engine->d_vit_statuses, sizeof(int) * nidx, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(vit statuses)")) != eslOK) goto CUDA_ERROR;
  if (passed) {
    if ((status = cuda_status(cudaMemcpy(passed, engine->d_vit_passed, sizeof(int) * nidx, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(vit passed)")) != eslOK) goto CUDA_ERROR;
  }
  cudaEventRecord(d2h1);
  engine->stats.submit_overhead_seconds += submit_overhead;
  if ((status = cuda_wait_and_account_vit(d2h1, errbuf, errbuf_size, "cudaEventSynchronize(vit d2h1)", &wait_barrier)) != eslOK) goto CUDA_ERROR;
  engine->stats.wait_barrier_seconds += wait_barrier;
  engine->stats.dispatch_wait_seconds += wait_barrier;

  engine->stats.vit_h2d_seconds    += elapsed_seconds(h2d0, h2d1);
  engine->stats.vit_kernel_seconds += elapsed_seconds(k0, k1);
  engine->stats.vit_d2h_seconds    += elapsed_seconds(d2h0, d2h1);
  engine->stats.vit_nseqs          += nidx;
  for (int i = 0; i < nidx; i++) {
    int si = seqidx ? seqidx[i] : i;
    if (si >= 0 && si < nseq) engine->stats.vit_nres += (uint64_t) chu->L[si];
  }
  engine->stats.vit_nbatches       += 1;

CUDA_ERROR:
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
  return p7_cuda_ViterbiSubset(engine, cuom, chu, seqidx, nidx, NULL, 0.0f, 0.0f, 0.0f,
                               scores, statuses, NULL, errbuf, errbuf_size);
}

extern "C" int
p7_cuda_ViterbiFilterDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                   ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                   const float *filtersc, double ev_mu, double ev_lambda, double F2,
                                   float *scores, int *statuses, int *passed,
                                   char *errbuf, int errbuf_size)
{
  return p7_cuda_ViterbiSubset(engine, cuom, chu, seqidx, nidx, filtersc, ev_mu, ev_lambda, F2,
                               scores, statuses, passed, errbuf, errbuf_size);
}
