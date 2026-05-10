#include "p7_cuda_internal.h"

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

  if ((status = cuda_status(cudaEventCreate(&engine->evt_h2d0), errbuf, errbuf_size, "cudaEventCreate(evt_h2d0)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaEventCreate(&engine->evt_h2d1), errbuf, errbuf_size, "cudaEventCreate(evt_h2d1)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaEventCreate(&engine->evt_k0),   errbuf, errbuf_size, "cudaEventCreate(evt_k0)"))   != eslOK) goto ERROR;
  if ((status = cuda_status(cudaEventCreate(&engine->evt_k1),   errbuf, errbuf_size, "cudaEventCreate(evt_k1)"))   != eslOK) goto ERROR;
  if ((status = cuda_status(cudaEventCreate(&engine->evt_d2h0), errbuf, errbuf_size, "cudaEventCreate(evt_d2h0)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaEventCreate(&engine->evt_d2h1), errbuf, errbuf_size, "cudaEventCreate(evt_d2h1)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaStreamCreateWithFlags(&engine->vlt_stream_copy,   cudaStreamNonBlocking), errbuf, errbuf_size, "cudaStreamCreate(vlt copy)"))   != eslOK) goto ERROR;
  if ((status = cuda_status(cudaStreamCreateWithFlags(&engine->vlt_stream_thresh, cudaStreamNonBlocking), errbuf, errbuf_size, "cudaStreamCreate(vlt thresh)")) != eslOK) goto ERROR;
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
  if (engine->evt_h2d0) cudaEventDestroy(engine->evt_h2d0);
  if (engine->evt_h2d1) cudaEventDestroy(engine->evt_h2d1);
  if (engine->evt_k0)   cudaEventDestroy(engine->evt_k0);
  if (engine->evt_k1)   cudaEventDestroy(engine->evt_k1);
  if (engine->evt_d2h0) cudaEventDestroy(engine->evt_d2h0);
  if (engine->evt_d2h1) cudaEventDestroy(engine->evt_d2h1);
  if (engine->vlt_stream_copy)   cudaStreamDestroy(engine->vlt_stream_copy);
  if (engine->vlt_stream_thresh) cudaStreamDestroy(engine->vlt_stream_thresh);
  if (engine->d_dsq)      cudaFree(engine->d_dsq);
  if (engine->h_dsq)      cudaFreeHost(engine->h_dsq);
  if (engine->d_offsets)  cudaFree(engine->d_offsets);
  if (engine->d_lengths)  cudaFree(engine->d_lengths);
  if (engine->d_tjb_by_seq) cudaFree(engine->d_tjb_by_seq);
  if (engine->d_raw)      cudaFree(engine->d_raw);
  if (engine->d_overflow) cudaFree(engine->d_overflow);
  if (engine->d_null_scores)    cudaFree(engine->d_null_scores);
  if (engine->d_bias_filtersc) cudaFree(engine->d_bias_filtersc);
  if (engine->d_bias_pi)       cudaFree(engine->d_bias_pi);
  if (engine->d_bias_t)        cudaFree(engine->d_bias_t);
  if (engine->d_bias_eo)       cudaFree(engine->d_bias_eo);
  if (engine->d_fwd_scores)    cudaFree(engine->d_fwd_scores);
  if (engine->d_fwd_statuses)  cudaFree(engine->d_fwd_statuses);
  if (engine->d_fwd_passed)    cudaFree(engine->d_fwd_passed);
  if (engine->d_fwd_filtersc)  cudaFree(engine->d_fwd_filtersc);
  if (engine->d_fwd_seqidx)    cudaFree(engine->d_fwd_seqidx);
  if (engine->d_vit_scores)    cudaFree(engine->d_vit_scores);
  if (engine->d_vit_statuses)  cudaFree(engine->d_vit_statuses);
  if (engine->d_vit_passed)    cudaFree(engine->d_vit_passed);
  if (engine->d_vit_filtersc)  cudaFree(engine->d_vit_filtersc);
  if (engine->d_vit_seqidx)    cudaFree(engine->d_vit_seqidx);
  if (engine->d_fwd_prev)      cudaFree(engine->d_fwd_prev);
  if (engine->d_fwd_curr)      cudaFree(engine->d_fwd_curr);
  if (engine->d_parser_xf)     cudaFree(engine->d_parser_xf);
  if (engine->d_parser_xb)     cudaFree(engine->d_parser_xb);
  if (engine->d_parser_scores) cudaFree(engine->d_parser_scores);
  if (engine->d_parser_statuses) cudaFree(engine->d_parser_statuses);
  if (engine->d_parser_seqidx) cudaFree(engine->d_parser_seqidx);
  if (engine->d_parser_x_offsets) cudaFree(engine->d_parser_x_offsets);
  if (engine->d_parser_surv_idx) cudaFree(engine->d_parser_surv_idx);
  if (engine->d_parser_surv_x_offsets) cudaFree(engine->d_parser_surv_x_offsets);
  if (engine->d_parser_xf_compact) cudaFree(engine->d_parser_xf_compact);
  if (engine->d_f1_survivor_idx) cudaFree(engine->d_f1_survivor_idx);
  if (engine->d_f1_counter) cudaFree(engine->d_f1_counter);
  if (engine->d_f1_survivor_usc) cudaFree(engine->d_f1_survivor_usc);
  if (engine->d_f1_survivor_status) cudaFree(engine->d_f1_survivor_status);
  if (engine->d_bias_surv_filtersc) cudaFree(engine->d_bias_surv_filtersc);
  if (engine->d_resident_dsq)     cudaFree(engine->d_resident_dsq);
  if (engine->d_resident_offsets) cudaFree(engine->d_resident_offsets);
  if (engine->d_resident_lengths) cudaFree(engine->d_resident_lengths);
  if (engine->d_lt_dsq)        cudaFree(engine->d_lt_dsq);
  if (engine->h_lt_dsq)        cudaFreeHost(engine->h_lt_dsq);
  if (engine->d_lt_offsets)    cudaFree(engine->d_lt_offsets);
  if (engine->d_lt_lengths)    cudaFree(engine->d_lt_lengths);
  if (engine->d_lt_ssv_scores) cudaFree(engine->d_lt_ssv_scores);
  if (engine->d_lt_windows)    cudaFree(engine->d_lt_windows);
  if (engine->d_lt_win_count)  cudaFree(engine->d_lt_win_count);
  if (engine->d_vlt_dsq)       cudaFree(engine->d_vlt_dsq);
  if (engine->h_vlt_dsq)       cudaFreeHost(engine->h_vlt_dsq);
  if (engine->d_vlt_offsets)   cudaFree(engine->d_vlt_offsets);
  if (engine->d_vlt_lengths)   cudaFree(engine->d_vlt_lengths);
  if (engine->d_vlt_thresholds) cudaFree(engine->d_vlt_thresholds);
  if (engine->d_vlt_bias)      cudaFree(engine->d_vlt_bias);
  if (engine->d_vlt_windows)   cudaFree(engine->d_vlt_windows);
  if (engine->d_vlt_win_count) cudaFree(engine->d_vlt_win_count);
  if (engine->d_resident_nucdb) cudaFree(engine->d_resident_nucdb);
  /* Domain rescore grow-only buffers */
  for (int i = 0; i < 4; i++) {
    if (engine->d_dom_dpf[i]) cudaFree(engine->d_dom_dpf[i]);
    if (engine->d_dom_xmx[i]) cudaFree(engine->d_dom_xmx[i]);
  }
  if (engine->d_dom_rfv_all)      cudaFree(engine->d_dom_rfv_all);
  if (engine->d_dom_dsq_all)      cudaFree(engine->d_dom_dsq_all);
  if (engine->d_dom_dsq_ptrs)     cudaFree(engine->d_dom_dsq_ptrs);
  if (engine->d_dom_rfv_ptrs)     cudaFree(engine->d_dom_rfv_ptrs);
  if (engine->d_dom_lengths)      cudaFree(engine->d_dom_lengths);
  if (engine->d_dom_dp_offsets)   cudaFree(engine->d_dom_dp_offsets);
  if (engine->d_dom_xmx_offsets)  cudaFree(engine->d_dom_xmx_offsets);
  if (engine->d_dom_envsc)        cudaFree(engine->d_dom_envsc);
  if (engine->d_dom_bcksc)        cudaFree(engine->d_dom_bcksc);
  if (engine->d_dom_oasc)         cudaFree(engine->d_dom_oasc);
  if (engine->d_dom_domcorr)      cudaFree(engine->d_dom_domcorr);
  if (engine->d_dom_statuses)     cudaFree(engine->d_dom_statuses);
  if (engine->d_dom_trace_st)     cudaFree(engine->d_dom_trace_st);
  if (engine->d_dom_trace_k)      cudaFree(engine->d_dom_trace_k);
  if (engine->d_dom_trace_i)      cudaFree(engine->d_dom_trace_i);
  if (engine->d_dom_trace_pp)     cudaFree(engine->d_dom_trace_pp);
  if (engine->d_dom_trace_N)      cudaFree(engine->d_dom_trace_N);
  if (engine->d_dom_orig_rfv)     cudaFree(engine->d_dom_orig_rfv);
  free(engine);
}

extern "C" void
p7_cuda_engine_Reset(P7_CUDA_ENGINE *engine)
{
  if (!engine) return;
  engine->batch_owner = NULL;
  engine->batch_nseq  = 0;
  engine->batch_total = 0;
  engine->bias_params_uploaded = 0;
  memset(&engine->stats, 0, sizeof(engine->stats));
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

  /* Build linear emission layout: rbv_lin[x * M + k] = rbv[(x*Q + k%Q) * 16 + k/Q] */
  {
    size_t lin_bytes = (size_t) cuom->Kp * (size_t) cuom->M;
    uint8_t *h_rbv_lin = (uint8_t *) malloc(lin_bytes);
    if (!h_rbv_lin) { status = eslEMEM; goto ERROR; }
    const uint8_t *src = (const uint8_t *) om->rbv[0];
    for (int x = 0; x < cuom->Kp; x++) {
      for (int k = 0; k < cuom->M; k++) {
        int q = k % cuom->Q;
        int z = k / cuom->Q;
        h_rbv_lin[x * cuom->M + k] = src[(x * cuom->Q + q) * 16 + z];
      }
    }
    if ((status = cuda_status(cudaMalloc((void **) &cuom->d_rbv_lin, lin_bytes), errbuf, errbuf_size, "cudaMalloc(rbv_lin)")) != eslOK) { free(h_rbv_lin); goto ERROR; }
    if ((status = cuda_status(cudaMemcpy(cuom->d_rbv_lin, h_rbv_lin, lin_bytes, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(rbv_lin)")) != eslOK) { free(h_rbv_lin); goto ERROR; }
    free(h_rbv_lin);
  }

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
extern "C" void
p7_cuda_msvprofile_Destroy(P7_CUDA_MSVPROFILE *cuom)
{
  if (!cuom) return;
  if (cuom->d_rbv) cudaFree(cuom->d_rbv);
  if (cuom->d_rbv_lin) cudaFree(cuom->d_rbv_lin);
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
p7_cuda_engine_UploadDatabase(P7_CUDA_ENGINE *engine, const uint8_t *seq_data, int64_t dsq_size,
                               const int64_t *offsets, const int32_t *lengths, int64_t nseq,
                               char *errbuf, int errbuf_size)
{
  size_t free_mem, total_mem;
  size_t needed;
  int   *h_offsets_int = NULL;
  int   *h_lengths_int = NULL;
  int    status;

  if (!engine || !seq_data || !offsets || !lengths || nseq <= 0)
    return eslEINVAL;

  cudaSetDevice(engine->device_id);
  cudaMemGetInfo(&free_mem, &total_mem);

  needed = (size_t) dsq_size + (size_t) nseq * (sizeof(int) + sizeof(int)) + 64;
  if (needed > free_mem * 8 / 10) {
    if (errbuf && errbuf_size > 0)
      snprintf(errbuf, errbuf_size, "database (%zu MB) exceeds 80%% of free GPU memory (%zu MB)",
               needed / (1024*1024), free_mem / (1024*1024));
    return eslENORESULT;
  }

  p7_cuda_engine_ReleaseDatabase(engine);

  /* Pin the host memory for faster DMA transfer */
  cudaHostRegister((void *) seq_data, dsq_size, cudaHostRegisterReadOnly);

  if ((status = cuda_status(cudaMalloc((void **) &engine->d_resident_dsq, dsq_size), errbuf, errbuf_size, "cudaMalloc(resident dsq)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMalloc((void **) &engine->d_resident_offsets, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(resident offsets)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMalloc((void **) &engine->d_resident_lengths, sizeof(int) * nseq), errbuf, errbuf_size, "cudaMalloc(resident lengths)")) != eslOK) goto ERROR;

  if ((status = cuda_status(cudaMemcpy(engine->d_resident_dsq, seq_data, dsq_size, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(resident dsq)")) != eslOK) goto ERROR;

  h_offsets_int = (int *) malloc(sizeof(int) * nseq);
  h_lengths_int = (int *) malloc(sizeof(int) * nseq);
  if (!h_offsets_int || !h_lengths_int) { status = eslEMEM; goto ERROR; }

  for (int64_t i = 0; i < nseq; i++) {
    h_offsets_int[i] = (int) offsets[i];
    h_lengths_int[i] = (int) lengths[i];
  }

  if ((status = cuda_status(cudaMemcpy(engine->d_resident_offsets, h_offsets_int, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(resident offsets)")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_resident_lengths, h_lengths_int, sizeof(int) * nseq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(resident lengths)")) != eslOK) goto ERROR;

  engine->resident_nseq     = nseq;
  engine->resident_dsq_size = dsq_size;
  engine->resident_active   = 1;

  cudaHostUnregister((void *) seq_data);

  free(h_offsets_int);
  free(h_lengths_int);
  return eslOK;

ERROR:
  free(h_offsets_int);
  free(h_lengths_int);
  p7_cuda_engine_ReleaseDatabase(engine);
  return status;
}

extern "C" void
p7_cuda_engine_ReleaseDatabase(P7_CUDA_ENGINE *engine)
{
  if (!engine) return;
  if (engine->d_resident_dsq)     { cudaFree(engine->d_resident_dsq);     engine->d_resident_dsq = NULL; }
  if (engine->d_resident_offsets) { cudaFree(engine->d_resident_offsets); engine->d_resident_offsets = NULL; }
  if (engine->d_resident_lengths) { cudaFree(engine->d_resident_lengths); engine->d_resident_lengths = NULL; }
  engine->resident_nseq     = 0;
  engine->resident_dsq_size = 0;
  engine->resident_active   = 0;
}

extern "C" int
p7_cuda_engine_IsResident(const P7_CUDA_ENGINE *engine)
{
  return (engine && engine->resident_active) ? 1 : 0;
}

extern "C" int
p7_cuda_engine_PreallocParser(P7_CUDA_ENGINE *engine, int max_nseq, int max_L,
                               char *errbuf, int errbuf_size)
{
  size_t total_xcells;
  int    status;

  if (!engine || max_nseq <= 0 || max_L <= 0) return eslEINVAL;

  cudaSetDevice(engine->device_id);

  if (engine->parser_result_alloc < max_nseq) {
    if (engine->d_parser_scores)    cudaFree(engine->d_parser_scores);
    if (engine->d_parser_statuses)  cudaFree(engine->d_parser_statuses);
    if (engine->d_parser_seqidx)    cudaFree(engine->d_parser_seqidx);
    if (engine->d_parser_x_offsets) cudaFree(engine->d_parser_x_offsets);
    engine->d_parser_scores    = NULL;
    engine->d_parser_statuses  = NULL;
    engine->d_parser_seqidx    = NULL;
    engine->d_parser_x_offsets = NULL;
    engine->parser_result_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_scores, sizeof(float) * 2 * max_nseq), errbuf, errbuf_size, "cudaMalloc(prealloc parser scores)")) != eslOK) return status;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_statuses, sizeof(int) * 2 * max_nseq), errbuf, errbuf_size, "cudaMalloc(prealloc parser statuses)")) != eslOK) return status;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_seqidx, sizeof(int) * max_nseq), errbuf, errbuf_size, "cudaMalloc(prealloc parser seqidx)")) != eslOK) return status;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_x_offsets, sizeof(size_t) * max_nseq), errbuf, errbuf_size, "cudaMalloc(prealloc parser x_offsets)")) != eslOK) return status;
    engine->parser_result_alloc = max_nseq;
  }

  total_xcells = (size_t) max_nseq * (size_t) (max_L + 1) * p7X_NXCELLS;
  if (engine->parser_cell_alloc < total_xcells) {
    if (engine->d_parser_xf) cudaFree(engine->d_parser_xf);
    if (engine->d_parser_xb) cudaFree(engine->d_parser_xb);
    engine->d_parser_xf = NULL;
    engine->d_parser_xb = NULL;
    engine->parser_allocL   = 0;
    engine->parser_cell_alloc = 0;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_xf, sizeof(float) * total_xcells), errbuf, errbuf_size, "cudaMalloc(prealloc parser xf)")) != eslOK) return status;
    if ((status = cuda_status(cudaMalloc((void **) &engine->d_parser_xb, sizeof(float) * total_xcells), errbuf, errbuf_size, "cudaMalloc(prealloc parser xb)")) != eslOK) return status;
    engine->parser_cell_alloc = total_xcells;
  }

  return eslOK;
}

/* ------------------------------------------------------------------
 * Warps-per-block auto-tuner.
 *
 * Multi-warp-per-block kernels: each warp handles one sequence; blocks
 * pack W warps. Block dim = 32*W. Per-block shmem = base + W*per_warp.
 *
 *   kernel_id = 0 : fused SSV+null+bias+gate
 *                   base = 2*M (s_q/s_z lookup, shared across warps)
 *                   per_warp = 0 (per-warp scratch is in registers / a few
 *                                 padded shared bytes accounted in the
 *                                 launch-site shmem calc; treat as 64B here)
 *   kernel_id = 1 : Viterbi opt
 *                   base = 2*M (s_q/s_z lookup)
 *                   per_warp = 0 (state lives in registers)
 *
 * Picks the W in {1,2,3,4,6,8} that maximizes W * blocks_per_sm where
 * blocks_per_sm = min(maxBlocksPerSM, maxThreadsPerSM/(32*W),
 *                     sharedMemPerSM/per_block_shmem).
 *
 * If user_w > 0, returns user_w clamped to the supported set.
 * ------------------------------------------------------------------ */
extern "C" int
p7_cuda_DefaultWarpsPerBlock(int device_id, int kernel_id,
                             const P7_CUDA_MSVPROFILE *cuom, int user_w)
{
  static const int W_CHOICES[] = {1, 2, 3, 4, 6, 8};
  static const int N_CHOICES   = (int) (sizeof(W_CHOICES) / sizeof(W_CHOICES[0]));

  /* User override: clamp to supported set (round up to nearest). */
  if (user_w > 0) {
    int chosen = W_CHOICES[N_CHOICES - 1];
    for (int i = 0; i < N_CHOICES; i++) {
      if (W_CHOICES[i] >= user_w) { chosen = W_CHOICES[i]; break; }
    }
    return chosen;
  }

  if (!cuom) return 4;
  int M = cuom->M;

  /* Cache device props per device id. */
  static int           cached_dev = -1;
  static cudaDeviceProp cached_prop;
  if (cached_dev != device_id) {
    if (cudaGetDeviceProperties(&cached_prop, device_id) != cudaSuccess) {
      /* Fall back to a sensible default if we can't query. */
      return 4;
    }
    cached_dev = device_id;
  }

  size_t shmem_per_sm  = cached_prop.sharedMemPerMultiprocessor;
  int    threads_per_sm = cached_prop.maxThreadsPerMultiProcessor;
  int    max_blocks_sm  = cached_prop.maxBlocksPerMultiProcessor;
  if (max_blocks_sm <= 0) max_blocks_sm = 32;  /* sm_89 reports 24; fallback */

  /* Per-block shmem accounting differs by kernel:
   *   SSV fused     : 2*M (s_q/s_z) + W*2*(M+1) (per-warp prev/curr fallback).
   *   Viterbi opt   : 2*M (s_q/s_z), no per-warp shmem (state in registers).
   * For more aggressive packing on Viterbi we ignore the prev/curr term. */
  size_t base_bytes = (size_t) (2 * M);
  size_t per_warp_bytes;
  switch (kernel_id) {
    case 0: per_warp_bytes = (size_t) 2 * (M + 1); break;
    case 1: per_warp_bytes = 0; break;
    default: per_warp_bytes = (size_t) 2 * (M + 1); break;
  }

  int best_w = 1;
  int best_warps_resident = 0;
  /* Empirically W=4 is the sweet spot on sm_89: bigger W reduces per-block
   * launch overhead, but past 4 the per-block work gets too coarse and
   * register pressure costs occupancy. Bias the tie-break toward W=4. */
  const int W_PREFERRED = 4;
  int best_distance = 1 << 20;

  for (int i = 0; i < N_CHOICES; i++) {
    int W = W_CHOICES[i];
    size_t block_shmem = base_bytes + (size_t) W * per_warp_bytes;
    if (block_shmem > shmem_per_sm) continue;
    int by_blocks  = max_blocks_sm;
    int by_threads = threads_per_sm / (32 * W);
    int by_shmem   = (int) (shmem_per_sm / block_shmem);
    int blocks_per_sm = by_blocks;
    if (by_threads < blocks_per_sm) blocks_per_sm = by_threads;
    if (by_shmem   < blocks_per_sm) blocks_per_sm = by_shmem;
    if (blocks_per_sm <= 0) continue;
    int warps_resident = blocks_per_sm * W;
    int distance = (W >= W_PREFERRED) ? (W - W_PREFERRED) : (W_PREFERRED - W);
    if (warps_resident > best_warps_resident ||
        (warps_resident == best_warps_resident && distance < best_distance)) {
      best_warps_resident = warps_resident;
      best_distance = distance;
      best_w = W;
    }
  }
  return best_w;
}

extern "C" int
p7_cuda_engine_UploadNucdb(P7_CUDA_ENGINE *engine, const uint8_t *data, int64_t size,
                           char *errbuf, int errbuf_size)
{
  int status = eslOK;
  if (!engine || !data || size <= 0) return eslEINVAL;

  p7_cuda_engine_ReleaseNucdb(engine);

  if ((status = cuda_status(cudaMalloc((void **)&engine->d_resident_nucdb, size), errbuf, errbuf_size, "cudaMalloc(nucdb)")) != eslOK) return status;
  if ((status = cuda_status(cudaMemcpy(engine->d_resident_nucdb, data, size, cudaMemcpyHostToDevice), errbuf, errbuf_size, "cudaMemcpy(nucdb)")) != eslOK) {
    cudaFree(engine->d_resident_nucdb);
    engine->d_resident_nucdb = NULL;
    return status;
  }
  engine->resident_nucdb_size = size;
  return eslOK;
}

extern "C" void
p7_cuda_engine_ReleaseNucdb(P7_CUDA_ENGINE *engine)
{
  if (!engine) return;
  if (engine->d_resident_nucdb) { cudaFree(engine->d_resident_nucdb); engine->d_resident_nucdb = NULL; }
  engine->resident_nucdb_size = 0;
}

extern "C" const uint8_t *
p7_cuda_engine_NucdbDevPtr(const P7_CUDA_ENGINE *engine)
{
  return engine ? engine->d_resident_nucdb : NULL;
}
