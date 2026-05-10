/* CUDA acceleration interface.
 *
 * This is intentionally separate from the selected SIMD implementation.
 */
#ifndef P7_CUDA_INCLUDED
#define P7_CUDA_INCLUDED

#include <stdint.h>

#include "easel.h"
#include "esl_dsqdata.h"
#include "hmmer.h"

typedef struct p7_cuda_msv_profile_s P7_CUDA_MSVPROFILE;
typedef struct p7_cuda_msv_engine_s  P7_CUDA_ENGINE;

typedef struct p7_cuda_msv_stats_s {
  double   h2d_seconds;
  double   kernel_seconds;
  double   d2h_seconds;
  double   null_h2d_seconds;
  double   null_kernel_seconds;
  double   null_d2h_seconds;
  double   bias_h2d_seconds;
  double   bias_kernel_seconds;
  double   bias_d2h_seconds;
  double   fwd_h2d_seconds;
  double   fwd_kernel_seconds;
  double   fwd_d2h_seconds;
  double   vit_h2d_seconds;
  double   vit_kernel_seconds;
  double   vit_d2h_seconds;
  double   bck_h2d_seconds;
  double   bck_kernel_seconds;
  double   bck_d2h_seconds;
  double   dispatch_wait_seconds;
  double   submit_overhead_seconds;
  double   wait_barrier_seconds;
  double   event_overhead_seconds;
  double   overlap_hidden_h2d_seconds;
  double   overlap_hidden_d2h_seconds;
  double   host_malloc_free_seconds;
  double   host_metadata_loop_seconds;
  double   host_event_ops_seconds;
  double   host_pack_loop_seconds;
  double   host_score_convert_seconds;
  double   host_sync_seconds;
  double   host_cudamemcpy_seconds;
  uint64_t nseqs;
  uint64_t nres;
  uint64_t nbatches;
  uint64_t fwd_nseqs;
  uint64_t fwd_nres;
  uint64_t fwd_nbatches;
  uint64_t fwd_prefilter_nseqs;
  uint64_t fwd_prefilter_nres;
  uint64_t fwd_prefilter_nbatches;
  uint64_t fwd_parser_nseqs;
  uint64_t fwd_parser_nres;
  uint64_t fwd_parser_nbatches;
  uint64_t vit_nseqs;
  uint64_t vit_nres;
  uint64_t vit_nbatches;
  uint64_t bck_nseqs;
  uint64_t bck_nres;
  uint64_t bck_nbatches;
  double   ssv_kernel_seconds;
} P7_CUDA_MSV_STATS;

extern int  p7_cuda_Available(char *errbuf, int errbuf_size);
extern int  p7_cuda_engine_Create(int device_id, P7_CUDA_ENGINE **ret_engine, char *errbuf, int errbuf_size);
extern void p7_cuda_engine_Destroy(P7_CUDA_ENGINE *engine);
extern void p7_cuda_engine_Reset(P7_CUDA_ENGINE *engine);
extern void p7_cuda_engine_GetStats(const P7_CUDA_ENGINE *engine, P7_CUDA_MSV_STATS *stats);

extern int  p7_cuda_msvprofile_Create(const P7_OPROFILE *om, P7_CUDA_MSVPROFILE **ret_cuom, char *errbuf, int errbuf_size);
extern void p7_cuda_msvprofile_Destroy(P7_CUDA_MSVPROFILE *cuom);
extern int  p7_cuda_msvprofile_UpdateLength(P7_CUDA_MSVPROFILE *cuom, const P7_OPROFILE *om, int L, char *errbuf, int errbuf_size);

extern int  p7_cuda_MSVFilter(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                              const ESL_DSQ *dsq, int L, float *ret_sc,
                              char *errbuf, int errbuf_size);
extern int  p7_cuda_MSVFilterBatch(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                   ESL_SQ_BLOCK *block, float *scores, int *statuses,
                                   char *errbuf, int errbuf_size);
extern int  p7_cuda_MSVFilterDsqdataChunk(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                          ESL_DSQDATA_CHUNK *chu, float *scores, int *statuses,
                                          char *errbuf, int errbuf_size);
extern int  p7_cuda_NullScoreDsqdataChunk(P7_CUDA_ENGINE *engine, const P7_BG *bg,
                                           ESL_DSQDATA_CHUNK *chu, float *nullsc,
                                           char *errbuf, int errbuf_size);
extern int  p7_cuda_BiasFilterDsqdataChunk(P7_CUDA_ENGINE *engine, const P7_BG *bg,
                                           ESL_DSQDATA_CHUNK *chu, float *filtersc,
                                           char *errbuf, int errbuf_size);
extern int  p7_cuda_ForwardScoreDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                              ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                              float *scores, int *statuses,
                                              char *errbuf, int errbuf_size);
extern int  p7_cuda_ForwardFilterDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                               ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                               const float *filtersc, double ev_mu, double ev_lambda, double F3,
                                               float *scores, int *statuses, int *passed,
                                               char *errbuf, int errbuf_size);
extern int  p7_cuda_ViterbiScoreDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                              ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                              float *scores, int *statuses,
                                              int warps_per_block,
                                              char *errbuf, int errbuf_size);
extern int  p7_cuda_ViterbiFilterDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                               ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                               const float *filtersc, double ev_mu, double ev_lambda, double F2,
                                               float *scores, int *statuses, int *passed,
                                               int warps_per_block,
                                               char *errbuf, int errbuf_size);
extern int  p7_cuda_ForwardBackwardParser(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                           const ESL_DSQ *dsq, int L, P7_OMX *oxf, P7_OMX *oxb,
                                           float *ret_fwdsc, float *ret_bcksc,
                                           char *errbuf, int errbuf_size);
extern int  p7_cuda_ForwardBackwardParserDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                                        ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                                        const size_t *x_offsets, size_t total_xcells,
                                                        float *xf, float *xb, float *scores, int *statuses,
                                                        char *errbuf, int errbuf_size);
/* Forward-only variant: runs Forward parser kernel, writes xf xmx and Forward
 * scores. xb is unused; scores[2k+1] is left untouched. */
extern int  p7_cuda_ForwardParserDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                                ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                                const size_t *x_offsets, size_t total_xcells,
                                                float *xf, float *scores, int *statuses,
                                                char *errbuf, int errbuf_size);
/* Backward-only variant: takes xf as INPUT (H2D'd by this call), runs Backward
 * parser kernel, writes xb xmx and Backward scores. scores[2k+0] is left
 * untouched; scores[2k+1] holds Backward score. */
extern int  p7_cuda_BackwardParserDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                                 ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                                 const size_t *x_offsets, size_t total_xcells,
                                                 const float *xf, float *xb, float *scores, int *statuses,
                                                 char *errbuf, int errbuf_size);
extern int  p7_cuda_F1GatingDsqdataChunk(P7_CUDA_ENGINE *engine,
                                          const float *msv_scores, const int *msv_statuses,
                                          int nseq, int do_biasfilter,
                                          double ev_mu, double ev_lambda, double F1,
                                          int *survivor_idx, int *ret_nsurv,
                                          char *errbuf, int errbuf_size);
extern int  p7_cuda_BiasFilterSurvivors(P7_CUDA_ENGINE *engine, const P7_BG *bg,
                                         int nsurv, float *h_filtersc,
                                         char *errbuf, int errbuf_size);
extern int  p7_cuda_SSVFilterDsqdataChunk(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                          ESL_DSQDATA_CHUNK *chu, float *scores, int *statuses,
                                          char *errbuf, int errbuf_size);
extern int  p7_cuda_SSVFilterResident(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                       int64_t seq0, int nseq, float *scores, int *statuses,
                                       char *errbuf, int errbuf_size);

extern int  p7_cuda_SSVNullBiasGateResident(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                             const P7_BG *bg, int64_t seq0, int nseq, int do_biasfilter,
                                             double ev_mu, double ev_lambda, double F1,
                                             int *survivor_idx, int *ret_nsurv,
                                             float *nullsc, float *filtersc,
                                             float *survivor_scores, int *survivor_statuses,
                                             int warps_per_block,
                                             char *errbuf, int errbuf_size);
extern int  p7_cuda_SSVNullBiasGateDsqdataChunk(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                                 const P7_BG *bg, ESL_DSQDATA_CHUNK *chu, int do_biasfilter,
                                                 double ev_mu, double ev_lambda, double F1,
                                                 int *survivor_idx, int *ret_nsurv,
                                                 float *nullsc, float *filtersc,
                                                 float *survivor_scores, int *survivor_statuses,
                                                 int warps_per_block,
                                                 char *errbuf, int errbuf_size);

extern int  p7_cuda_engine_UploadDatabase(P7_CUDA_ENGINE *engine, const uint8_t *seq_data, int64_t dsq_size,
                                           const int64_t *offsets, const int32_t *lengths, int64_t nseq,
                                           char *errbuf, int errbuf_size);
extern void p7_cuda_engine_ReleaseDatabase(P7_CUDA_ENGINE *engine);
extern int  p7_cuda_engine_IsResident(const P7_CUDA_ENGINE *engine);
extern int  p7_cuda_engine_PreallocParser(P7_CUDA_ENGINE *engine, int max_nseq, int max_L,
                                           char *errbuf, int errbuf_size);

extern int  p7_cuda_MSVFilterResident(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                       int64_t seq0, int nseq, float *scores, int *statuses,
                                       char *errbuf, int errbuf_size);

/* Auto-tune helpers: pick warps-per-block for the multi-warp DP kernels.
 * kernel_id: 0 = fused SSV/null/bias/gate, 1 = Viterbi opt.
 * Returns a value in {1,2,3,4,6,8} chosen from per-block shmem cost and
 * device shmem-per-SM / threads-per-SM limits to maximize resident warps.
 * If user_w > 0, returns user_w clamped to the supported set. */
extern int  p7_cuda_DefaultWarpsPerBlock(int device_id, int kernel_id,
                                          const P7_CUDA_MSVPROFILE *cuom, int user_w);

/* SSV longtarget for nhmmer: scans a long nucleotide sequence for high-scoring windows */
typedef struct {
  int32_t  chunk_id;
  int32_t  target_start;
  int32_t  target_end;
  int16_t  model_start;
  int16_t  model_end;
  float    score;
} P7_CUDA_LT_WINDOW;

/* Viterbi longtarget for nhmmer: scanning Viterbi emits sub-windows within merged windows */
typedef struct {
  int32_t  window_id;
  int32_t  position;
  int16_t  model_k;
  int16_t  pad;
} P7_CUDA_VIT_LT_WINDOW;

typedef struct {
  int      grid_blocks;
  int      block_threads;
  int      dynamic_smem_bytes;
  int      active_blocks_per_sm;
  int      active_warps_per_sm;
  int      max_warps_per_sm;
  int      sm_count;
  double   theoretical_occupancy;
  double   grid_sm_coverage;
  double   device_active_seconds;
  double   pack_seconds;
  double   h2d_seconds;
  double   threshold_kernel_seconds;
  double   kernel_seconds;
  double   d2h_seconds;
  double   alloc_seconds;
  double   stream_seconds;
  int64_t  packed_bytes;
  int      nwindows_in;
  int      nwindows_out;
  int      warps_per_block;
} P7_CUDA_VIT_LT_STATS;

typedef struct {
  int      grid_blocks;
  int      block_threads;
  int      dynamic_smem_bytes;
  int      active_blocks_per_sm;
  int      active_warps_per_sm;
  int      max_warps_per_sm;
  int      sm_count;
  int      nchunks;
  int      chunk_size;
  double   theoretical_occupancy;
  double   grid_sm_coverage;
  double   kernel_seconds;
} P7_CUDA_SSV_LT_STATS;

extern int  p7_cuda_SSVLongtarget(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                  const ESL_DSQ *dsq, int L,
                                  const uint8_t *ssv_scores_host, int Kp,
                                  uint8_t sc_thresh, float scale_b,
                                  int chunk_size, int overlap,
                                  P7_CUDA_LT_WINDOW **ret_windows, int *ret_nwindows,
                                  P7_CUDA_SSV_LT_STATS *stats,
                                  char *errbuf, int errbuf_size);

extern int  p7_cuda_SSVLongtargetResident(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                          const uint8_t *d_nucdb_data, int nchunks,
                                          const int *h_offsets, const int *h_lengths,
                                          const uint8_t *ssv_scores_host, int Kp,
                                          uint8_t sc_thresh, float scale_b,
                                          int step,
                                          P7_CUDA_LT_WINDOW **ret_windows, int *ret_nwindows,
                                          P7_CUDA_SSV_LT_STATS *stats,
                                          char *errbuf, int errbuf_size);

extern int  p7_cuda_engine_UploadNucdb(P7_CUDA_ENGINE *engine, const uint8_t *data, int64_t size,
                                       char *errbuf, int errbuf_size);
extern void p7_cuda_engine_ReleaseNucdb(P7_CUDA_ENGINE *engine);
extern const uint8_t *p7_cuda_engine_NucdbDevPtr(const P7_CUDA_ENGINE *engine);

extern int  p7_cuda_DomainRescoreBatch(P7_CUDA_ENGINE *engine,
                                       const P7_CUDA_MSVPROFILE *cuom,
                                       int ndomains,
                                       const uint8_t **h_dsq_ptrs,
                                       const int      *h_lengths,
                                       const float   **h_rfv_ptrs,
                                       const float    *h_orig_rfv,
                                       int Q, int Kp, float nj,
                                       float   *h_envsc,
                                       float   *h_domcorrection,
                                       float   *h_oasc,
                                       int8_t  *h_trace_st,
                                       int     *h_trace_k,
                                       int     *h_trace_i,
                                       float   *h_trace_pp,
                                       int     *h_trace_N,
                                       int      max_trace_len,
                                       int     *h_statuses,
                                       char    *errbuf, int errbuf_size);

extern int  p7_cuda_ViterbiLongtarget(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                      const ESL_DSQ *dsq, int L,
                                      const P7_HMM_WINDOW *windows, int nwindows,
                                      const float *bias_scores, int do_biasfilter,
                                      int B2, float F2, float vmu, float vlambda,
                                      float scale_w, float xw_e_move, float nj,
                                      float base_w, int max_length,
                                      P7_CUDA_VIT_LT_WINDOW **ret_windows, int *ret_nwindows,
                                      P7_CUDA_VIT_LT_STATS *stats,
                                      char *errbuf, int errbuf_size);
extern int  p7_cuda_ViterbiLongtarget_GetThresholds(P7_CUDA_ENGINE *engine,
                                                     int16_t *h_thresholds, int nwindows);

#endif /*P7_CUDA_INCLUDED*/
