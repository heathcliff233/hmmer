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
#include <time.h>

extern "C" {
#include "p7_cuda.h"
}

#define P7_CUDA_MSV_BLOCK_THREADS 32
#define VIT_OPT_MAX_STRIDE_SMALL  24   /* register-based Viterbi for M <= 768 */
#define VIT_OPT_MAX_STRIDE_LARGE  64   /* register-based Viterbi for M <= 2048 */

struct p7_cuda_msv_engine_s {
  int                 device_id;
  cudaEvent_t         evt_h2d0;
  cudaEvent_t         evt_h2d1;
  cudaEvent_t         evt_k0;
  cudaEvent_t         evt_k1;
  cudaEvent_t         evt_k2;
  cudaEvent_t         evt_k3;
  cudaEvent_t         evt_d2h0;
  cudaEvent_t         evt_d2h1;
  uint8_t            *d_dsq;
  int                 dsq_alloc;
  uint8_t            *h_dsq;
  int                 h_dsq_alloc;
  const void         *batch_owner;
  int                 batch_nseq;
  int                 batch_total;
  const uint8_t      *d_batch_dsq;
  int                *d_offsets;
  int                *d_lengths;
  uint8_t            *d_tjb_by_seq;
  int                 meta_alloc;
  int                *d_gather_src1_offsets;
  int                *d_gather_src1_lengths;
  int                *d_gather_src2_offsets;
  int                *d_gather_lengths;
  int                *d_gather_dst_offsets;
  int                 gather_alloc;
  int                *h_meta_offsets;
  int                *h_meta_lengths;
  uint8_t            *h_meta_tjb_by_seq;
  int                 h_meta_alloc;
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
  int                *d_fwd_passed;
  float              *d_fwd_filtersc;
  int                *d_fwd_seqidx;
  int                 fwd_result_alloc;
  float              *d_vit_scores;
  int                *d_vit_statuses;
  int                *d_vit_passed;
  float              *d_vit_filtersc;
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
  int                *d_parser_surv_idx;
  size_t             *d_parser_surv_x_offsets;
  size_t             *d_parser_surv_total;
  int                 parser_allocL;
  int                 parser_result_alloc;
  int                 parser_surv_alloc;
  size_t              parser_cell_alloc;
  float              *d_parser_xf_compact;
  size_t              parser_compact_cell_alloc;
  int                *d_f1_survivor_idx;
  int                *d_f1_counter;
  float              *d_f1_survivor_usc;
  float              *d_f1_survivor_filtersc;
  int                *d_f1_survivor_status;
  int                 f1_result_alloc;
  int                *d_f1_pass_mask;
  int                 f1_order_alloc;
  int                *h_f1_pass_mask;
  float              *h_f1_filtersc;
  int                 h_f1_order_alloc;
  float              *d_bias_surv_filtersc;
  int                 bias_surv_alloc;
  int                 bias_params_uploaded;
  /* SSV longtarget persistent buffers */
  uint8_t            *d_lt_dsq;
  int                 lt_dsq_alloc;
  int                *d_lt_offsets;
  int                *d_lt_lengths;
  int                 lt_meta_alloc;
  uint8_t            *d_lt_ssv_scores;
  int                 lt_ssv_alloc;
  const uint8_t      *lt_ssv_cached_ptr;
  int                 lt_ssv_cached_size;
  P7_CUDA_LT_WINDOW *d_lt_windows;
  int                *d_lt_win_count;
  P7_CUDA_LT_WINDOW *d_lt_windows_compact;
  P7_CUDA_LT_WINDOW *h_lt_windows_compact;
  P7_HMM_WINDOW     *d_lt_hmm_windows;
  P7_HMM_WINDOW     *h_lt_hmm_windows;
  int                *d_lt_win_offsets;
  int                 lt_compact_alloc;
  int                 lt_hmm_alloc;
  int                 lt_count_alloc;
  int                 lt_win_alloc;
  uint8_t            *h_lt_dsq;
  int                 h_lt_dsq_alloc;
  int                *h_lt_win_counts;
  int                 h_lt_prefix_alloc;

  /* Viterbi longtarget persistent buffers */
  uint8_t            *d_vlt_dsq;
  int                 vlt_dsq_alloc;
  int                *d_vlt_offsets;
  int                *d_vlt_lengths;
  int                 vlt_meta_alloc;
  int16_t            *d_vlt_thresholds;
  int                 vlt_thresh_alloc;
  float              *d_vlt_bias;
  int                 vlt_bias_alloc;
  P7_CUDA_VIT_LT_WINDOW *d_vlt_windows;
  P7_CUDA_VIT_LT_WINDOW *d_vlt_windows_compact;
  P7_CUDA_VIT_LT_WINDOW *h_vlt_windows;
  P7_HMM_WINDOW         *d_vlt_input_windows;
  P7_HMM_WINDOW         *d_vlt_hmm_tmp;
  P7_HMM_WINDOW         *d_vlt_hmm_windows;
  P7_HMM_WINDOW         *h_vlt_hmm_windows;
  int                *d_vlt_win_count;
  int                *d_vlt_win_offsets;
  int                *d_vlt_hmm_count;
  int                 vlt_win_alloc;
  int                 vlt_count_alloc;
  int                 vlt_compact_alloc;
  int                 vlt_input_win_alloc;
  int                 vlt_hmm_tmp_alloc;
  int                 vlt_hmm_alloc;
  uint8_t            *h_vlt_dsq;
  int                 h_vlt_dsq_alloc;
  int                *h_vlt_win_counts;
  int                 h_vlt_prefix_alloc;
  cudaStream_t        vlt_stream_copy;
  cudaStream_t        vlt_stream_thresh;

  const struct p7_cuda_msv_profile_s *last_cuom;
  P7_CUDA_MSV_STATS   stats;

  uint8_t            *d_resident_dsq;
  int                *d_resident_offsets;
  int                *d_resident_lengths;
  int64_t             resident_nseq;
  int64_t             resident_dsq_size;
  int                 resident_active;
  int64_t             resident_batch_seq0;
  int                 resident_batch_nseq;

  /* Nucdb resident data (uploaded once, reused across sequences/queries) */
  uint8_t            *d_resident_nucdb;
  int64_t             resident_nucdb_size;

  /* Domain rescore grow-only buffers */
  float              *d_dom_dpf[4];       /* [0]=fwd [1]=bck/pp [2]=pp [3]=oa */
  size_t              dom_dp_alloc;        /* total_dp_cells currently allocated */
  float              *d_dom_xmx[4];       /* [0]=fwd [1]=bck [2]=pp [3]=oa */
  size_t              dom_xmx_alloc;       /* total_xmx floats allocated */
  float              *d_dom_rfv_all;
  size_t              dom_rfv_alloc;       /* bytes */
  uint8_t            *d_dom_dsq_all;
  size_t              dom_dsq_alloc;
  const uint8_t     **d_dom_dsq_ptrs;
  const float       **d_dom_rfv_ptrs;
  int                *d_dom_lengths;
  size_t             *d_dom_dp_offsets;
  size_t             *d_dom_xmx_offsets;
  float              *d_dom_envsc;
  float              *d_dom_bcksc;
  float              *d_dom_oasc;
  float              *d_dom_domcorr;
  int                *d_dom_statuses;
  int                 dom_meta_alloc;      /* ndomains currently allocated */
  int8_t             *d_dom_trace_st;
  int                *d_dom_trace_k;
  int                *d_dom_trace_i;
  float              *d_dom_trace_pp;
  int                *d_dom_trace_N;
  size_t              dom_trace_alloc;     /* ndomains * max_trace_len */
  float              *d_dom_orig_rfv;
  size_t              dom_orig_rfv_alloc;  /* bytes */
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
  uint8_t *d_rbv_lin;   /* linear layout: d_rbv_lin[x * M + k] for k in 0..M-1 */
  float   *d_prefix_lengths;
  float   *d_suffix_lengths;
  int      window_lengths_alloc;
  int      window_lengths_uploaded;
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

  /* nhmmer-specific 4-row emission tables (codes 0..3 only) */
  int      Kp_nuc;         /* = 4 when nuc tables active, 0 otherwise */
  uint8_t *d_rbv_lin_nuc;  /* [x * M + k] for x in {0,1,2,3} */
  int16_t *d_rwv_nuc;      /* [((int)x * Qw + q) * 8 + lane] for x in {0,1,2,3} */
  float   *d_rfv_nuc;      /* [((int)x * Qf + q) * 4 + lane] for x in {0,1,2,3} */
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

static inline double
host_seconds(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

uint8_t cuda_unbiased_byteify(const P7_CUDA_MSVPROFILE *cuom, float sc);
int cuda_msvprofile_GrowLengthLookup(P7_CUDA_MSVPROFILE *cuom, int L);
int cuda_msvprofile_UploadWindowLengths(P7_CUDA_MSVPROFILE *cuom, const P7_SCOREDATA *data,
                                        char *errbuf, int errbuf_size);

#endif /*P7_CUDA_INTERNAL_INCLUDED*/
