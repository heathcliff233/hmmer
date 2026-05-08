#ifndef NHMMER_INTERNAL_INCLUDED
#define NHMMER_INTERNAL_INCLUDED

#include "easel.h"
#include "esl_dsqdata.h"
#include "esl_getopts.h"
#include "esl_sq.h"

#include "hmmer.h"
#include "cuda_msv.h"
#include "p7_nucdb.h"

typedef int (*NHMMER_GPU_IDLEN_CB)(void *data, int id, int64_t L);

/* Synthetic batch: packs nhmmer windows as "virtual sequences" for GPU batch APIs.
 * dsq pointers point into the parent ESL_SQ (zero-copy). */
typedef struct {
  ESL_DSQDATA_CHUNK  chu;         /* synthetic chunk passed to GPU batch APIs */
  ESL_DSQ          **dsq_ptrs;    /* dsq[i] = parent_sq->dsq + window->n - 1 */
  int64_t           *lengths;     /* window lengths */
  char             **names;       /* all point to empty_str */
  char             **accs;
  char             **descs;
  int32_t           *taxids;      /* all -1 */
  int               *win_idx;     /* index into original merged window list */
  int                nwindows;
  int                alloc;
} NHMMER_GPU_WINDOW_BATCH;

typedef struct {
  P7_BG            *bg;
  P7_PIPELINE      *pli;
  P7_TOPHITS       *th;
  P7_OPROFILE      *om;
  P7_SCOREDATA     *scoredata;
  P7_CUDA_ENGINE   *cuda_engine;
  P7_CUDA_MSVPROFILE *cuda_msv;
  const ESL_GETOPTS *go;
  int               gpu_chunk_size;
  int               ncpus;
  int               do_gpu_batch;       /* --gpu-batch: batch SSV/bias on GPU */
  int               do_gpu_vit;         /* --gpu-vit-prefilter */
  int               do_gpu_vit_lt;      /* --gpu-vit-longtarget: scanning Viterbi on GPU */
  int               do_gpu_fwd;         /* --gpu-fwd-prefilter */
  /* Persistent scratch arrays (grow-only, freed at end) */
  float            *h_ssv_scores;
  int              *h_ssv_status;
  float            *h_null_scores;
  float            *h_bias_scores;
  int               h_filter_alloc;
  /* Instrumentation counters (accumulated across strands/blocks) */
  int64_t           n_vit_lt_windows_in;   /* windows entering scanning Viterbi */
  int64_t           n_vit_lt_windows_out;  /* sub-windows from scanning Viterbi */
  int64_t           n_post_vit_windows;    /* windows sent to CPU ForwardParser */
  /* Timing breakdown (seconds, accumulated across strands) */
  double            t_ssv;           /* GPU SSV longtarget kernel */
  double            t_merge;         /* window extend + merge */
  double            t_batch_filter;  /* GPU batch SSV/bias/F1 + Viterbi pre-filter */
  double            t_vit_lt;        /* GPU scanning Viterbi longtarget */
  double            t_fwd_prefilter; /* GPU Forward pre-filter */
  double            t_gpu_fb_parser; /* GPU ForwardBackward parser batch */
  double            t_cpu_workers;   /* CPU domaindef + hit reporting */
#ifdef HMMER_THREADS
  pthread_mutex_t   gpu_domain_mutex; /* serialize GPU domain rescore across threads */
#endif
} NHMMER_GPU_INFO;

int nhmmer_gpu_serial_loop(NHMMER_GPU_INFO *info, ESL_SQFILE *dbfp,
                           int strands, NHMMER_GPU_IDLEN_CB idlen_cb, void *idlen_data,
                           int *ret_nseqs, int64_t *ret_nres);

int nhmmer_gpu_nucdb_loop(NHMMER_GPU_INFO *info, P7_NUCDB *ndb,
                          int strands, NHMMER_GPU_IDLEN_CB idlen_cb, void *idlen_data,
                          int *ret_nseqs, int64_t *ret_nres);

#endif /*NHMMER_INTERNAL_INCLUDED*/
