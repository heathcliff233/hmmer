#include "p7_cuda_internal.h"

/* GPU SSV longtarget kernel for nhmmer.
 *
 * Scans overlapping chunks of a long nucleotide sequence for high-scoring
 * SSV diagonals, exactly replicating the algorithm in p7_SSVFilter_longtarget().
 * Each block processes one chunk; output is a list of windows (diagonals).
 */

#define SSV_LT_MAX_STRIDE 64
#define SSV_LT_MAX_WINDOWS_PER_CHUNK 128

__device__ static inline uint8_t
lt_u8_sub_sat(uint8_t a, uint8_t b)
{
  return (a > b) ? (uint8_t)(a - b) : 0;
}

__device__ static inline uint8_t
lt_u8_add_sat(uint8_t a, uint8_t b)
{
  unsigned int v = (unsigned int)a + (unsigned int)b;
  return (v > 255) ? 255 : (uint8_t)v;
}

/* SSV longtarget kernel: one block (32 threads) per chunk.
 *
 * For each chunk, runs the SSV DP scan. When xE >= sc_thresh, performs
 * diagonal recovery (backward walk) and extension (forward walk), then
 * emits a window and resets DP state. This exactly replicates the CPU
 * p7_SSVFilter_longtarget() algorithm.
 *
 * Parameters:
 *   dsq          - packed digital sequences (all chunks contiguous)
 *   offsets      - byte offset of each chunk in dsq
 *   lengths      - length of each chunk
 *   nchunks      - number of chunks
 *   rbv          - profile emission scores [x*Q*16 + q*16 + z] (byte-encoded)
 *   ssv_scores   - scoredata for diagonal recovery [(k)*Kp + x]
 *   M, Q, Kp     - model dimensions
 *   tbm_b, tec_b, base_b, bias_b - profile parameters
 *   sc_thresh    - score threshold for window detection
 *   d_windows    - output window array
 *   d_win_count  - atomic counter for total windows emitted
 *   max_windows  - capacity of d_windows
 */
__global__ static void
cuda_ssv_longtarget_kernel(const uint8_t *dsq, const int *offsets, const int *lengths,
                           int nchunks,
                           const uint8_t *rbv, const uint8_t *ssv_scores,
                           int M, int Q, int Kp,
                           uint8_t tbm_b, uint8_t tec_b, uint8_t tjb_b,
                           uint8_t base_b, uint8_t bias_b,
                           uint8_t sc_thresh, float scale_b,
                           P7_CUDA_LT_WINDOW *d_windows, int *d_win_count, int max_windows)
{
  extern __shared__ uint8_t mem[];
  int chunk = blockIdx.x;
  int tid   = threadIdx.x;
  if (chunk >= nchunks) return;

  const uint8_t *sdsq = dsq + offsets[chunk];
  int L = lengths[chunk];

  int stride = (M + 31) / 32;
  int my_start = tid * stride;
  int my_count = stride;
  if (my_start + my_count > M) my_count = M - my_start;
  if (my_count < 0) my_count = 0;

  /* Shared memory layout: q_lookup[M] + z_lookup[M] */
  uint8_t *s_q = mem;
  uint8_t *s_z = mem + M;
  for (int k = tid; k < M; k += 32) {
    s_q[k] = (uint8_t)(k % Q);
    s_z[k] = (uint8_t)(k / Q);
  }
  __syncthreads();

  uint8_t xB_ssv = lt_u8_sub_sat(base_b, (uint8_t)(tjb_b + tbm_b));

  /* Register-based DP state */
  uint8_t prev_reg[SSV_LT_MAX_STRIDE];
  for (int j = 0; j < my_count; j++) prev_reg[j] = 0;

  uint8_t last_prev = 0;

  /* Shared variables for cross-warp communication */
  __shared__ int    s_hit_detected;
  __shared__ uint8_t s_best_score;
  __shared__ int    s_best_k;
  __shared__ int    s_skip_to;

  if (tid == 0) {
    s_hit_detected = 0;
    s_skip_to = 0;
  }
  __syncthreads();

  for (int i = 1; i <= L; i++) {
    if (i <= s_skip_to) continue;

    uint8_t x = sdsq[i];

    uint8_t from_left = __shfl_up_sync(0xffffffff, last_prev, 1);
    if (tid == 0) from_left = 0;

    uint8_t xE_local = 0;
    int     xE_k_local = -1;

    for (int j = 0; j < my_count; j++) {
      int km1 = my_start + j;
      uint8_t old_prev = prev_reg[j];
      uint8_t rsc = rbv[((int)x * Q + (int)s_q[km1]) * 16 + (int)s_z[km1]];
      uint8_t v = (from_left > xB_ssv) ? from_left : xB_ssv;
      v = lt_u8_add_sat(v, bias_b);
      v = lt_u8_sub_sat(v, rsc);
      prev_reg[j] = v;
      if (v > xE_local) {
        xE_local = v;
        xE_k_local = km1 + 1;
      }
      from_left = old_prev;
    }

    last_prev = (my_count > 0) ? prev_reg[my_count - 1] : 0;

    /* Warp reduction to find max xE and which model position hit it */
    for (int s = 16; s > 0; s >>= 1) {
      uint8_t other_sc = __shfl_down_sync(0xffffffff, xE_local, s);
      int     other_k  = __shfl_down_sync(0xffffffff, xE_k_local, s);
      if (other_sc > xE_local) {
        xE_local = other_sc;
        xE_k_local = other_k;
      }
    }

    /* Thread 0 checks threshold */
    if (tid == 0 && xE_local >= sc_thresh) {
      s_hit_detected = 1;
      s_best_score = xE_local;
      s_best_k = xE_k_local;
    }
    __syncthreads();

    if (s_hit_detected) {
      if (tid == 0) {
        int end = s_best_k;
        int rem_sc = (int)s_best_score;
        int sc = rem_sc;
        int start = end;
        int target_end = i;
        int target_start = i;

        /* Backward walk: recover diagonal start
         * Threshold: base_b - tjb_b - tbm_b (same as CPU) */
        int walk_threshold = (int)base_b - (int)tjb_b - (int)tbm_b;
        while (rem_sc > walk_threshold && target_start > 0) {
          rem_sc -= (int)bias_b - (int)ssv_scores[start * Kp + (int)sdsq[target_start]];
          --start;
          --target_start;
        }
        start++;
        target_start++;

        /* Forward walk: extend diagonal (up to 5 positions past max) */
        int k = end + 1;
        int n = target_end + 1;
        int max_end = target_end;
        int max_sc = sc;
        int pos_since_max = 0;
        while (k < M && n <= L) {
          sc += (int)bias_b - (int)ssv_scores[k * Kp + (int)sdsq[n]];
          if (sc >= max_sc) {
            max_sc = sc;
            max_end = n;
            pos_since_max = 0;
          } else {
            pos_since_max++;
            if (pos_since_max == 5) break;
          }
          k++;
          n++;
        }

        end += (max_end - target_end);
        target_end = max_end;

        /* Score conversion: same as CPU
         * ret_sc = ((max_sc - tjb_b) - base_b) / scale_b - 3.0 */
        float ret_sc = ((float)(max_sc - (int)tjb_b) - (float)base_b);
        ret_sc /= scale_b;
        ret_sc -= 3.0f;

        /* Emit window */
        int widx = atomicAdd(d_win_count, 1);
        if (widx < max_windows) {
          d_windows[widx].chunk_id     = chunk;
          d_windows[widx].target_start = target_start;
          d_windows[widx].target_end   = target_end;
          d_windows[widx].model_start  = (int16_t)start;
          d_windows[widx].model_end    = (int16_t)end;
          d_windows[widx].score        = ret_sc;
        }

        /* Skip forward past the hit (same as CPU: i = target_end) */
        s_skip_to = target_end;
        s_hit_detected = 0;
      }
      __syncthreads();

      /* Reset DP state after hit */
      for (int j = 0; j < my_count; j++) prev_reg[j] = 0;
      last_prev = 0;
    }
  }
}

/* Host wrapper: chunk a long sequence, upload, launch kernel, download windows.
 *
 * The caller provides the full digital sequence (dsq[1..L]) and the model
 * parameters. This function splits the sequence into overlapping chunks,
 * packs them for GPU, launches the SSV longtarget kernel, and returns
 * the detected windows with coordinates translated to the original sequence.
 *
 * Uses engine-persistent device buffers (grow-on-demand, never freed until
 * engine destroy) to avoid per-call cudaMalloc/cudaFree overhead.
 */
extern "C" int
p7_cuda_SSVLongtarget(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                      const ESL_DSQ *dsq, int L,
                      const uint8_t *ssv_scores_host, int Kp,
                      uint8_t sc_thresh, float scale_b,
                      int chunk_size, int overlap,
                      P7_CUDA_LT_WINDOW **ret_windows, int *ret_nwindows,
                      char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int M = cuom->M;
  int Q = cuom->Q;

  /* Compute chunking */
  int step = chunk_size - overlap;
  if (step < 1) step = 1;
  int nchunks = (L <= chunk_size) ? 1 : 1 + (L - chunk_size + step - 1) / step;

  /* Host arrays (stack-allocated metadata) */
  int *h_offsets  = NULL;
  int *h_lengths  = NULL;
  P7_CUDA_LT_WINDOW *h_windows = NULL;
  int h_win_count = 0;
  int total_packed = 0;

  int max_windows = nchunks * SSV_LT_MAX_WINDOWS_PER_CHUNK;
  int ssv_scores_size = (M + 1) * Kp;

  /* Allocate host metadata arrays */
  h_offsets = (int *)malloc(sizeof(int) * nchunks);
  h_lengths = (int *)malloc(sizeof(int) * nchunks);
  if (!h_offsets || !h_lengths) { status = eslEMEM; goto ERROR; }

  /* Compute chunk offsets and lengths */
  for (int c = 0; c < nchunks; c++) {
    int seq_start = c * step;
    int seq_end   = seq_start + chunk_size - 1;
    if (seq_end >= L) seq_end = L - 1;
    int clen = seq_end - seq_start + 1;
    h_offsets[c] = total_packed;
    h_lengths[c] = clen;
    total_packed += clen + 1;
  }
  total_packed += 1;

  /* Grow engine-persistent device buffers as needed */
  if (engine->lt_dsq_alloc < total_packed) {
    if (engine->d_lt_dsq) cudaFree(engine->d_lt_dsq);
    engine->d_lt_dsq = NULL;
    engine->lt_dsq_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_lt_dsq, total_packed), errbuf, errbuf_size, "cudaMalloc(lt dsq)")) != eslOK) goto ERROR;
    engine->lt_dsq_alloc = total_packed;
  }
  if (engine->h_lt_dsq_alloc < total_packed) {
    if (engine->h_lt_dsq) cudaFreeHost(engine->h_lt_dsq);
    engine->h_lt_dsq = NULL;
    engine->h_lt_dsq_alloc = 0;
    if ((status = cuda_status(cudaMallocHost((void **)&engine->h_lt_dsq, total_packed), errbuf, errbuf_size, "cudaMallocHost(lt dsq)")) != eslOK) goto ERROR;
    engine->h_lt_dsq_alloc = total_packed;
  }
  if (engine->lt_meta_alloc < nchunks) {
    if (engine->d_lt_offsets) cudaFree(engine->d_lt_offsets);
    if (engine->d_lt_lengths) cudaFree(engine->d_lt_lengths);
    engine->d_lt_offsets = NULL;
    engine->d_lt_lengths = NULL;
    engine->lt_meta_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_lt_offsets, sizeof(int) * nchunks), errbuf, errbuf_size, "cudaMalloc(lt offsets)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_lt_lengths, sizeof(int) * nchunks), errbuf, errbuf_size, "cudaMalloc(lt lengths)")) != eslOK) goto ERROR;
    engine->lt_meta_alloc = nchunks;
  }
  if (engine->lt_ssv_alloc < ssv_scores_size) {
    if (engine->d_lt_ssv_scores) cudaFree(engine->d_lt_ssv_scores);
    engine->d_lt_ssv_scores = NULL;
    engine->lt_ssv_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_lt_ssv_scores, ssv_scores_size), errbuf, errbuf_size, "cudaMalloc(lt ssv_scores)")) != eslOK) goto ERROR;
    engine->lt_ssv_alloc = ssv_scores_size;
  }
  if (engine->lt_win_alloc < max_windows) {
    if (engine->d_lt_windows) cudaFree(engine->d_lt_windows);
    if (engine->d_lt_win_count) cudaFree(engine->d_lt_win_count);
    engine->d_lt_windows = NULL;
    engine->d_lt_win_count = NULL;
    engine->lt_win_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_lt_windows, sizeof(P7_CUDA_LT_WINDOW) * max_windows), errbuf, errbuf_size, "cudaMalloc(lt windows)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_lt_win_count, sizeof(int)), errbuf, errbuf_size, "cudaMalloc(lt win_count)")) != eslOK) goto ERROR;
    engine->lt_win_alloc = max_windows;
  }

  /* Pack chunks into pinned host buffer */
  for (int c = 0; c < nchunks; c++) {
    int seq_start = c * step;
    int clen = h_lengths[c];
    uint8_t *dest = engine->h_lt_dsq + h_offsets[c];
    dest[0] = eslDSQ_SENTINEL;
    memcpy(dest + 1, dsq + 1 + seq_start, clen);
  }

  /* Upload data */
  if ((status = cuda_status(cudaMemcpy(engine->d_lt_dsq, engine->h_lt_dsq, total_packed, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(lt dsq)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_lt_offsets, h_offsets, sizeof(int) * nchunks, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(lt offsets)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_lt_lengths, h_lengths, sizeof(int) * nchunks, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(lt lengths)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_lt_ssv_scores, ssv_scores_host, ssv_scores_size, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(lt ssv_scores)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemset(engine->d_lt_win_count, 0, sizeof(int)), errbuf, errbuf_size, "cudaMemset(lt win_count)")) != eslOK) goto ERROR;

  /* Launch kernel */
  {
    size_t shmem = (size_t)M * 2;
    cuda_ssv_longtarget_kernel<<<nchunks, 32, shmem>>>(
      engine->d_lt_dsq, engine->d_lt_offsets, engine->d_lt_lengths, nchunks,
      cuom->d_rbv, engine->d_lt_ssv_scores,
      M, Q, Kp,
      cuom->tbm_b, cuom->tec_b, cuom->tjb_b,
      cuom->base_b, cuom->bias_b,
      sc_thresh, scale_b,
      engine->d_lt_windows, engine->d_lt_win_count, engine->lt_win_alloc);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_ssv_longtarget_kernel launch")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaDeviceSynchronize(), errbuf, errbuf_size, "cuda_ssv_longtarget sync")) != eslOK) goto ERROR;
  }

  /* Download results */
  if ((status = cuda_status(cudaMemcpy(&h_win_count, engine->d_lt_win_count, sizeof(int), cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(lt win_count)")) != eslOK) goto ERROR;
  if (h_win_count > engine->lt_win_alloc) h_win_count = engine->lt_win_alloc;

  if (h_win_count > 0) {
    h_windows = (P7_CUDA_LT_WINDOW *)malloc(sizeof(P7_CUDA_LT_WINDOW) * h_win_count);
    if (!h_windows) { status = eslEMEM; goto ERROR; }
    if ((status = cuda_status(cudaMemcpy(h_windows, engine->d_lt_windows, sizeof(P7_CUDA_LT_WINDOW) * h_win_count, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(lt windows)")) != eslOK) goto ERROR;

    /* Translate chunk-local coordinates to global sequence coordinates */
    for (int w = 0; w < h_win_count; w++) {
      int c = h_windows[w].chunk_id;
      int global_offset = c * step;
      h_windows[w].target_start += global_offset;
      h_windows[w].target_end   += global_offset;
    }
  }

  *ret_windows  = h_windows;
  *ret_nwindows = h_win_count;
  h_windows = NULL;

ERROR:
  free(h_offsets);
  free(h_lengths);
  free(h_windows);
  return status;
}

/* Resident variant: nucdb data already on GPU. Takes pre-computed chunk offsets
 * and lengths (from the nucdb chunk index) — no packing or H2D copy needed.
 * The offsets point into d_nucdb_data (the uploaded nucdb data region).
 * Each offset points to a sentinel byte followed by residues, matching what
 * the kernel expects. The step parameter is needed to translate chunk-local
 * hit coordinates back to global sequence coordinates. */
extern "C" int
p7_cuda_SSVLongtargetResident(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                              const uint8_t *d_nucdb_data, int nchunks,
                              const int *h_offsets, const int *h_lengths,
                              const uint8_t *ssv_scores_host, int Kp,
                              uint8_t sc_thresh, float scale_b,
                              int step,
                              P7_CUDA_LT_WINDOW **ret_windows, int *ret_nwindows,
                              char *errbuf, int errbuf_size)
{
  int status = eslOK;
  int M = cuom->M;
  int Q = cuom->Q;

  P7_CUDA_LT_WINDOW *h_windows = NULL;
  int h_win_count = 0;

  int max_windows = nchunks * SSV_LT_MAX_WINDOWS_PER_CHUNK;
  int ssv_scores_size = (M + 1) * Kp;

  /* Grow device metadata/result buffers */
  if (engine->lt_meta_alloc < nchunks) {
    if (engine->d_lt_offsets) cudaFree(engine->d_lt_offsets);
    if (engine->d_lt_lengths) cudaFree(engine->d_lt_lengths);
    engine->d_lt_offsets = NULL;
    engine->d_lt_lengths = NULL;
    engine->lt_meta_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_lt_offsets, sizeof(int) * nchunks), errbuf, errbuf_size, "cudaMalloc(lt offsets)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_lt_lengths, sizeof(int) * nchunks), errbuf, errbuf_size, "cudaMalloc(lt lengths)")) != eslOK) goto ERROR;
    engine->lt_meta_alloc = nchunks;
  }
  if (engine->lt_ssv_alloc < ssv_scores_size) {
    if (engine->d_lt_ssv_scores) cudaFree(engine->d_lt_ssv_scores);
    engine->d_lt_ssv_scores = NULL;
    engine->lt_ssv_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_lt_ssv_scores, ssv_scores_size), errbuf, errbuf_size, "cudaMalloc(lt ssv_scores)")) != eslOK) goto ERROR;
    engine->lt_ssv_alloc = ssv_scores_size;
  }
  if (engine->lt_win_alloc < max_windows) {
    if (engine->d_lt_windows) cudaFree(engine->d_lt_windows);
    if (engine->d_lt_win_count) cudaFree(engine->d_lt_win_count);
    engine->d_lt_windows = NULL;
    engine->d_lt_win_count = NULL;
    engine->lt_win_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_lt_windows, sizeof(P7_CUDA_LT_WINDOW) * max_windows), errbuf, errbuf_size, "cudaMalloc(lt windows)")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc((void **)&engine->d_lt_win_count, sizeof(int)), errbuf, errbuf_size, "cudaMalloc(lt win_count)")) != eslOK) goto ERROR;
    engine->lt_win_alloc = max_windows;
  }

  /* Upload metadata (small arrays from host) */
  if ((status = cuda_status(cudaMemcpy(engine->d_lt_offsets, h_offsets, sizeof(int) * nchunks, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(lt offsets)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_lt_lengths, h_lengths, sizeof(int) * nchunks, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(lt lengths)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_lt_ssv_scores, ssv_scores_host, ssv_scores_size, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(lt ssv_scores)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemset(engine->d_lt_win_count, 0, sizeof(int)), errbuf, errbuf_size, "cudaMemset(lt win_count)")) != eslOK) goto ERROR;

  /* Launch kernel — data comes directly from resident nucdb on device */
  {
    size_t shmem = (size_t)M * 2;
    cuda_ssv_longtarget_kernel<<<nchunks, 32, shmem>>>(
      d_nucdb_data, engine->d_lt_offsets, engine->d_lt_lengths, nchunks,
      cuom->d_rbv, engine->d_lt_ssv_scores,
      M, Q, Kp,
      cuom->tbm_b, cuom->tec_b, cuom->tjb_b,
      cuom->base_b, cuom->bias_b,
      sc_thresh, scale_b,
      engine->d_lt_windows, engine->d_lt_win_count, engine->lt_win_alloc);
    if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "cuda_ssv_longtarget_kernel launch")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaDeviceSynchronize(), errbuf, errbuf_size, "cuda_ssv_longtarget sync")) != eslOK) goto ERROR;
  }

  /* Download results */
  if ((status = cuda_status(cudaMemcpy(&h_win_count, engine->d_lt_win_count, sizeof(int), cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(lt win_count)")) != eslOK) goto ERROR;
  if (h_win_count > engine->lt_win_alloc) h_win_count = engine->lt_win_alloc;

  if (h_win_count > 0) {
    h_windows = (P7_CUDA_LT_WINDOW *)malloc(sizeof(P7_CUDA_LT_WINDOW) * h_win_count);
    if (!h_windows) { status = eslEMEM; goto ERROR; }
    if ((status = cuda_status(cudaMemcpy(h_windows, engine->d_lt_windows, sizeof(P7_CUDA_LT_WINDOW) * h_win_count, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "cudaMemcpy(lt windows)")) != eslOK) goto ERROR;

    /* Translate chunk-local coordinates to global sequence coordinates */
    for (int w = 0; w < h_win_count; w++) {
      int c = h_windows[w].chunk_id;
      int global_offset = c * step;
      h_windows[w].target_start += global_offset;
      h_windows[w].target_end   += global_offset;
    }
  }

  *ret_windows  = h_windows;
  *ret_nwindows = h_win_count;
  h_windows = NULL;

ERROR:
  free(h_windows);
  return status;
}
