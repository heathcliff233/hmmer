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
  double   ssv_fallback_kernel_seconds;
  uint64_t ssv_fallback_nseqs;
} P7_CUDA_MSV_STATS;

extern int  p7_cuda_Available(char *errbuf, int errbuf_size);
extern int  p7_cuda_engine_Create(int device_id, P7_CUDA_ENGINE **ret_engine, char *errbuf, int errbuf_size);
extern void p7_cuda_engine_Destroy(P7_CUDA_ENGINE *engine);
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
                                              char *errbuf, int errbuf_size);
extern int  p7_cuda_ViterbiFilterDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                               ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                               const float *filtersc, double ev_mu, double ev_lambda, double F2,
                                               float *scores, int *statuses, int *passed,
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
extern int  p7_cuda_F1GatingDsqdataChunk(P7_CUDA_ENGINE *engine,
                                          const float *msv_scores, const int *msv_statuses,
                                          int nseq, int do_biasfilter,
                                          double ev_mu, double ev_lambda, double F1,
                                          int *survivor_idx, int *ret_nsurv,
                                          char *errbuf, int errbuf_size);
extern int  p7_cuda_SSVFilterDsqdataChunk(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                          ESL_DSQDATA_CHUNK *chu, float *scores, int *statuses,
                                          char *errbuf, int errbuf_size);

#endif /*P7_CUDA_INCLUDED*/
