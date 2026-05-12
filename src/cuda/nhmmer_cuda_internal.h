#ifndef NHMMER_CUDA_INTERNAL_INCLUDED
#define NHMMER_CUDA_INTERNAL_INCLUDED

/* Internal contract shared between the nhmmer GPU pipeline translation
 * units. Public callers (nhmmer.c) use nhmmer_internal.h instead — this
 * header carries only the cross-TU plumbing that the modular split
 * introduced. Wrapped in HMMER_CUDA so CPU-only builds never see it.
 */

#ifdef HMMER_CUDA

#include <p7_config.h>

#include <time.h>

#include "easel.h"
#include "esl_sq.h"

#ifdef HMMER_THREADS
#include <pthread.h>
#endif

#include "hmmer.h"
#include "nhmmer_internal.h"
#include "cuda/p7_cuda.h"

#define NHMMER_GPU_CHUNK_SIZE  65536
#define NHMMER_GPU_BLOCK_SIZE  (512 * 1024 * 1024)
#define NHMMER_GPU_BATCH_MIN   32

extern char nhmmer_empty_str[];

/* Nucdb sequence reconstruction helpers (nhmmer_gpu_seqhelpers.c). */
int nhmmer_gpu_nucdb_reconstruct_sq(const P7_NUCDB *ndb, const ESL_ALPHABET *abc,
                                    int64_t si, int complementarity, ESL_SQ **ret_sq);
int nhmmer_gpu_nucdb_get_cached_sq(const P7_NUCDB *ndb, const ESL_ALPHABET *abc,
                                   int64_t si, int complementarity,
                                   ESL_SQ **ret_sq, int *ret_built);
int nhmmer_gpu_nucdb_create_seq_shell(const P7_NUCDB *ndb, const ESL_ALPHABET *abc,
                                      int64_t si, ESL_SQ **ret_sq);
int nhmmer_gpu_nucdb_fill_slice(const P7_NUCDB *ndb, const ESL_ALPHABET *abc,
                                int64_t si, int complementarity,
                                uint64_t start, int length, const char *seqname,
                                ESL_SQ **ret_sq, ESL_DSQ **ret_dsq, int64_t *ret_alloc);

/* Window helpers and synthetic-chunk lifecycle (nhmmer_gpu_windows.c). */
int  nhmmer_gpu_windowlist_alloc(P7_HMM_WINDOWLIST *wl, int n);
void nhmmer_gpu_fill_ssv_windows(P7_HMM_WINDOWLIST *wl, const P7_CUDA_LT_WINDOW *gpu_windows,
                                 int n, uint32_t target_len);
int  nhmmer_gpu_copy_windows(P7_HMM_WINDOWLIST *wl, const P7_HMM_WINDOW *windows, int n);
void nhmmer_gpu_fill_vit_seed_windows(P7_HMM_WINDOWLIST *wl, const P7_CUDA_VIT_LT_WINDOW *gpu_windows,
                                      int n, const P7_HMM_WINDOWLIST *input_wl);
int  nhmmer_gpu_vit_window_compare(const void *a, const void *b);
int  nhmmer_gpu_window_batch_init(NHMMER_GPU_WINDOW_BATCH *wb, int alloc);
void nhmmer_gpu_window_batch_free(NHMMER_GPU_WINDOW_BATCH *wb);
int  nhmmer_gpu_window_batch_pack(NHMMER_GPU_WINDOW_BATCH *wb, const ESL_SQ *sq,
                                  P7_HMM_WINDOWLIST *wl);
int  nhmmer_gpu_window_batch_describe(NHMMER_GPU_WINDOW_BATCH *wb, P7_HMM_WINDOWLIST *wl);

/* CPU worker pool, OMX binding, and helper tables (nhmmer_gpu_workers.c). */
typedef struct {
  P7_OPROFILE      *om;
  P7_BG            *bg;
  P7_PIPELINE      *pli;
  P7_TOPHITS       *th;
  P7_SCOREDATA     *scoredata;
  P7_PIPELINE_LONGTARGET_OBJS *pli_tmp;
  P7_HMM_WINDOWLIST vit_windowlist;

  const ESL_SQ     *sq;
  const P7_NUCDB   *ndb;
  ESL_SQ           *slice_sq;
  ESL_DSQ          *slice_dsq;
  int64_t           slice_alloc;
  P7_HMM_WINDOW   *windows;
  int               nwindows;
  int64_t           seq_id;
  int               complementarity;
  int               status;
  /* GPU Forward prefilter results (non-owning pointer into shared xf buffer) */
  float            *prefilter_xf;
  float            *prefilter_fwdsc;
  size_t            prefilter_xf_offset;
  size_t           *prefilter_x_offsets;
  /* GPU FB parser results (non-owning pointers into shared buffers) */
  float            *gpu_xf;
  float            *gpu_xb;
  float            *gpu_scores;
  int              *gpu_statuses;
  size_t           *gpu_x_offsets;
  int              *gpu_L_eff;
  int               use_gpu_fb;
  P7_CUDA_ENGINE   *cuda_engine;
  P7_CUDA_MSVPROFILE *cuda_msv;
  int               worker_id;
  int               do_domain_trace;
  /* GPU domcorrection deferral list (P1). Populated during phase 1 (the
   * existing pthread fan-out) when do_gpu_domcorr is set on the worker's
   * pli; consumed by a single GPU batch call in the strand orchestrator
   * after pthread_join. */
  P7_GPU_DOMCORR_PENDING domcorr_pending;
#ifdef HMMER_THREADS
  pthread_mutex_t  *work_mutex;
#endif
  int              *next_window;
  int               global_nwindows;
} NHMMER_GPU_WORKER;

typedef struct {
  float *xmx;
  int    allocXR;
  int    M;
  int    L;
  int    has_own_scales;
  float  totscale;
} NHMMER_OMX_BINDING;

extern const char gpu_to_p7_state[];

int  nhmmer_gpu_worker_next_window(NHMMER_GPU_WORKER *w);
void nhmmer_gpu_trace_domain_window(const NHMMER_GPU_WORKER *w, const char *path,
                                    int local_idx, const P7_HMM_WINDOW *window,
                                    double domain_before, int status);
int  nhmmer_gpu_worker_init(NHMMER_GPU_WORKER *w, const NHMMER_GPU_INFO *info);
void nhmmer_gpu_worker_destroy(NHMMER_GPU_WORKER *w);
void nhmmer_gpu_worker_process(NHMMER_GPU_WORKER *w);
void nhmmer_gpu_worker_process_post_vit(NHMMER_GPU_WORKER *w);
void nhmmer_gpu_worker_process_post_fwd(NHMMER_GPU_WORKER *w);
void nhmmer_gpu_worker_process_post_fb(NHMMER_GPU_WORKER *w);
/* P1: after pthread_join, batches all workers' deferred 2nd-pass Forwards
 * on the GPU and patches hit->dcl[0].dombias / .domcorrection per entry.
 * No-op if no worker has pending entries. */
int  nhmmer_gpu_flush_domcorr(NHMMER_GPU_INFO *info, NHMMER_GPU_WORKER *workers, int nworkers,
                              char *errbuf, int errbuf_size);
void nhmmer_gpu_BindOmxXmx(P7_OMX *ox, float *xmx, int M, int L,
                           int has_own_scales, float totscale,
                           NHMMER_OMX_BINDING *saved);
void nhmmer_gpu_RestoreOmxXmx(P7_OMX *ox, const NHMMER_OMX_BINDING *saved);
int  nhmmer_gpu_computeAliScores(P7_DOMAIN *dom, ESL_DSQ *seq,
                                 const P7_SCOREDATA *data, int K);
int  nhmmer_gpu_trace_from_gpu(P7_TRACE *tr, int8_t *st, int *tk, int *ti, float *tp,
                               int tN, int M, int L);
#ifdef HMMER_THREADS
void *nhmmer_gpu_thread_func(void *arg);
void *nhmmer_gpu_thread_func_post_vit(void *arg);
void *nhmmer_gpu_thread_func_post_fwd(void *arg);
void *nhmmer_gpu_thread_func_post_fb(void *arg);
#endif

int nhmmer_gpu_scalar_viterbi_debug(P7_OPROFILE *om, const ESL_DSQ *dsq, int L,
                                    float filtersc, float F2, int16_t override_thresh,
                                    int *ret_nseeds, int16_t *xE_trace, int trace_len);

/* Timing/launch-stat helpers, scratch arena lifecycle, and GPU batch SSV/bias/F1
 * filter dispatch (cuda/p7_cuda_nhmmer_filters.c). */
double nhmmer_gpu_elapsed(const struct timespec *ts0, const struct timespec *ts1);
void   nhmmer_gpu_record_ssv_launch(NHMMER_GPU_INFO *info, const P7_CUDA_SSV_LT_STATS *stats);
void   nhmmer_gpu_record_vit_launch(NHMMER_GPU_INFO *info, const P7_CUDA_VIT_LT_STATS *stats);
int    nhmmer_gpu_ensure_scratch(NHMMER_GPU_INFO *info, int N);
int    nhmmer_gpu_ensure_nucdb_chunk_scratch(NHMMER_GPU_INFO *info, int n);
int    nhmmer_gpu_ensure_nucdb_window_scratch(NHMMER_GPU_INFO *info, int n);
int    nhmmer_gpu_ensure_parser_scratch(NHMMER_GPU_INFO *info, int n);
int    nhmmer_gpu_batch_filter(NHMMER_GPU_INFO *info, NHMMER_GPU_WINDOW_BATCH *wb,
                               P7_HMM_WINDOWLIST *wl, int *ret_nsurv,
                               char *errbuf, int errbuf_size);
int    nhmmer_gpu_batch_filter_resident(NHMMER_GPU_INFO *info, const uint8_t *d_dsq_base,
                                        const int *h_offsets, const int *h_lengths,
                                        P7_HMM_WINDOWLIST *wl, int *ret_nsurv,
                                        char *errbuf, int errbuf_size);
int    nhmmer_gpu_batch_filter_resident_gather(NHMMER_GPU_INFO *info, const uint8_t *d_dsq_base,
                                               const int *h_offsets, const int *h_src1_lengths,
                                               const int *h_src2_offsets, const int *h_lengths,
                                               P7_HMM_WINDOWLIST *wl, int *ret_nsurv,
                                               char *errbuf, int errbuf_size);
int    nhmmer_gpu_try_map_nucdb_windows(NHMMER_GPU_INFO *info, const P7_NUCDB *ndb,
                                        int chunk_start, int chunk_count,
                                        int complementarity,
                                        const P7_HMM_WINDOWLIST *wl,
                                        int **ret_offsets, int **ret_lengths,
                                        int **ret_src1_lengths, int **ret_src2_offsets,
                                        int *ret_needs_gather);
int    nhmmer_gpu_prepare_parser_resident_batch(NHMMER_GPU_INFO *info, const P7_NUCDB *ndb,
                                                int chunk_start, int chunk_count,
                                                int complementarity,
                                                const uint8_t *d_nucdb,
                                                P7_HMM_WINDOW *windows, int nwindows,
                                                const void *batch_owner,
                                                char *errbuf, int errbuf_size);

/* GPU Forward prefilter + Forward/Backward parser dispatch (cuda/p7_cuda_nhmmer_fwd.c). */
int nhmmer_gpu_forward_prefilter(NHMMER_GPU_INFO *info, const ESL_SQ *sq, int64_t seq_id,
                                 int complementarity, P7_HMM_WINDOW *windows, int nwindows,
                                 P7_HMM_WINDOW **ret_survivors, int *ret_nsurv,
                                 float **ret_surv_xf, float **ret_surv_fwdsc,
                                 char *errbuf, int errbuf_size);
int nhmmer_gpu_forward_backward_compact(NHMMER_GPU_INFO *info, const ESL_SQ *sq,
                                        int64_t seq_id, int complementarity,
                                        const P7_NUCDB *ndb, int chunk_start, int chunk_count,
                                        const uint8_t *d_nucdb,
                                        P7_HMM_WINDOW *windows, int nwindows,
                                        P7_HMM_WINDOW **ret_survivors, int *ret_nsurv,
                                        float **ret_xf, float **ret_xb,
                                        float **ret_scores, int **ret_statuses,
                                        size_t **ret_x_offsets, int **ret_L_eff,
                                        char *errbuf, int errbuf_size);
int nhmmer_gpu_run_fb_parser_batch(NHMMER_GPU_INFO *info, const ESL_SQ *sq,
                                   P7_HMM_WINDOW *windows, int nwindows,
                                   float *precomputed_xf, const float *precomputed_fwdsc,
                                   float **ret_xf, float **ret_xb,
                                   float **ret_scores, int **ret_statuses,
                                   size_t **ret_x_offsets, int **ret_L_eff,
                                   char *errbuf, int errbuf_size);

/* GPU scanning Viterbi longtarget orchestration (cuda/p7_cuda_nhmmer_viterbi.c). */
int nhmmer_gpu_viterbi_longtarget(NHMMER_GPU_INFO *info, const ESL_SQ *sq,
                                  P7_HMM_WINDOWLIST *input_wl,
                                  const float *precomputed_bias_scores,
                                  int use_f1_resident,
                                  P7_HMM_WINDOWLIST *ret_vit_wl,
                                  char *errbuf, int errbuf_size);

/* Per-strand orchestration drivers (cuda/p7_cuda_nhmmer_strand.c). */
int nhmmer_gpu_process_strand(NHMMER_GPU_INFO *info, const ESL_SQ *sq, int complementarity,
                              int64_t seq_id, uint8_t sc_thresh, int chunk_size, int overlap,
                              char *errbuf, int errbuf_size);
int nhmmer_gpu_process_nucdb_strand(NHMMER_GPU_INFO *info,
                                    const P7_NUCDB *ndb,
                                    int chunk_start, int chunk_count,
                                    const ESL_SQ *sq, int complementarity,
                                    int64_t seq_id,
                                    char *errbuf, int errbuf_size);

#endif /* HMMER_CUDA */

#endif /* NHMMER_CUDA_INTERNAL_INCLUDED */
