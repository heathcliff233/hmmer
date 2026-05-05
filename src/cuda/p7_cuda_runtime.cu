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
