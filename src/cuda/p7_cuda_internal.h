#ifndef P7_CUDA_INTERNAL_INCLUDED
#define P7_CUDA_INTERNAL_INCLUDED

#include <p7_config.h>

#include <cuda_runtime.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "p7_cuda.h"
}

#define P7_CUDA_MSV_BLOCK_THREADS 32

struct p7_cuda_msv_engine_s {
  int                 device_id;
  uint8_t            *d_dsq;
  int                 dsq_alloc;
  uint8_t            *h_dsq;
  int                 h_dsq_alloc;
  const void         *batch_owner;
  int                 batch_nseq;
  int                 batch_total;
  int                *d_offsets;
  int                *d_lengths;
  uint8_t            *d_tjb_by_seq;
  int                 meta_alloc;
  int                *d_raw;
  int                *d_overflow;
  int                 result_alloc;
  float              *d_null_scores;
  int                 null_result_alloc;
  float              *d_bias_filtersc;
  int                 bias_result_alloc;
  float              *d_bias_pi;
  float              *d_bias_t;
  float              *d_bias_eo;
  float              *d_fwd_scores;
  int                *d_fwd_statuses;
  int                *d_fwd_seqidx;
  int                 fwd_result_alloc;
  float              *d_vit_scores;
  int                *d_vit_statuses;
  int                *d_vit_seqidx;
  int                 vit_result_alloc;
  float              *d_fwd_prev;
  float              *d_fwd_curr;
  size_t              fwd_dp_alloc;
  float              *d_parser_xf;
  float              *d_parser_xb;
  float              *d_parser_scores;
  int                *d_parser_statuses;
  int                *d_parser_seqidx;
  size_t             *d_parser_x_offsets;
  int                 parser_allocL;
  int                 parser_result_alloc;
  size_t              parser_cell_alloc;
  P7_CUDA_MSV_STATS   stats;
};

struct p7_cuda_msv_profile_s {
  int      M;
  int      Q;
  int      Kp;
  uint8_t  tbm_b;
  uint8_t  tec_b;
  uint8_t  tjb_b;
  uint8_t *h_tjb_by_len;
  int      tjb_len_alloc;
  float    scale_b;
  uint8_t  base_b;
  uint8_t  bias_b;
  uint8_t *d_rbv;
  int      Qf;
  float   *d_rfv;
  float   *d_tfv;
  float    xf_e_loop;
  float    xf_e_move;
  float    xf_n_loop;
  float    xf_n_move;
  float    xf_c_loop;
  float    xf_c_move;
  float    xf_j_loop;
  float    xf_j_move;
  float    nj;
  int      Qw;
  int16_t *d_rwv;
  int16_t *d_twv;
  int16_t  xw_n_loop;
  int16_t  xw_n_move;
  int16_t  xw_c_loop;
  int16_t  xw_c_move;
  int16_t  xw_j_loop;
  int16_t  xw_j_move;
  int16_t  xw_e_loop;
  int16_t  xw_e_move;
  float    scale_w;
  int16_t  base_w;
  int16_t  ddbound_w;
};

static inline int
cuda_status(cudaError_t cerr, char *errbuf, int errbuf_size, const char *where)
{
  if (cerr == cudaSuccess) return eslOK;
  if (errbuf && errbuf_size > 0)
    snprintf(errbuf, errbuf_size, "%s: %s", where, cudaGetErrorString(cerr));
  return eslFAIL;
}

static inline double
elapsed_seconds(cudaEvent_t start, cudaEvent_t stop)
{
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  return (double) ms / 1000.0;
}

static inline int
next_pow2_at_least(int n, int min_n)
{
  int p = min_n;
  while (p < n) p <<= 1;
  return p;
}

uint8_t cuda_unbiased_byteify(const P7_CUDA_MSVPROFILE *cuom, float sc);
int cuda_msvprofile_GrowLengthLookup(P7_CUDA_MSVPROFILE *cuom, int L);

#endif /*P7_CUDA_INTERNAL_INCLUDED*/
