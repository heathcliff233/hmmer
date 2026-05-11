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
  int               do_gpu_vit_lt;      /* --gpu-vit-longtarget: scanning Viterbi on GPU */
  int               do_gpu_fwd;         /* default GPU Fwd/Bwd parser handoff */
  int               do_cpu_postmsv;     /* --gpu-cpu-postmsv: bypass GPU Vit+Fwd, use CPU postSSV */
  int               do_compare;         /* --gpu-compare: print GPU vs CPU score mismatches */
  int               do_domain_trace;    /* env HMMER_NHMMER_GPU_DOMAIN_TRACE: per-window CPU domain timing */
  int               nucdb_resident;     /* .nucdb data was uploaded before the per-query search loop */
  /* Persistent scratch arrays (grow-only, freed at end) */
  float            *h_ssv_scores;
  int              *h_ssv_status;
  float            *h_null_scores;
  float            *h_bias_scores;
  int              *h_f1_survivor_idx;
  float            *h_f1_survivor_bias;
  int              *h_parser_seqidx;
  size_t           *h_parser_x_offsets;
  int              *h_parser_surv_src_idx;
  int              *h_nucdb_chunk_offsets;
  int              *h_nucdb_chunk_lengths;
  int              *h_nucdb_window_offsets;
  int              *h_nucdb_window_lengths;
  int              *h_nucdb_window_src1_lengths;
  int              *h_nucdb_window_src2_offsets;
  int               h_filter_alloc;
  int               h_parser_alloc;
  int               h_nucdb_chunk_alloc;
  int               h_nucdb_window_alloc;
  /* Instrumentation counters (accumulated across strands/blocks) */
  int64_t           n_vit_lt_windows_in;   /* windows entering scanning Viterbi */
  int64_t           n_vit_lt_windows_out;  /* sub-windows from scanning Viterbi */
  int64_t           n_post_vit_windows;    /* windows entering GPU Forward prefilter / post-Vit continuation */
  int64_t           n_fwd_survivor_windows;/* windows surviving GPU Forward prefilter */
  /* Timing breakdown (seconds, accumulated across strands) */
  double            t_ssv;           /* GPU SSV longtarget kernel */
  int64_t           ssv_launches;     /* GPU SSV launch count */
  int               ssv_grid_blocks;
  int               ssv_block_threads;
  int               ssv_dynamic_smem;
  int               ssv_active_blocks_per_sm;
  int               ssv_active_warps_per_sm;
  int               ssv_max_warps_per_sm;
  int               ssv_sm_count;
  int               ssv_nchunks;
  int               ssv_chunk_size;
  double            ssv_theoretical_occupancy;
  double            ssv_grid_sm_coverage;
  double            ssv_kernel_seconds;
  double            t_merge;         /* window extend + merge */
  double            t_batch_filter;  /* GPU batch SSV/bias/F1 filter */
  double            t_batch_h2d;     /* CUDA H2D inside GPU batch F1 filter */
  double            t_batch_kernel;  /* CUDA kernels inside GPU batch F1 filter */
  double            t_batch_f1_gate; /* CUDA fused SSV/null/bias/F1 gate kernel */
  double            t_batch_compact; /* CUDA ordered survivor compaction kernel */
  double            t_batch_d2h;     /* CUDA D2H inside GPU batch F1 filter */
  int64_t           batch_packed_bytes; /* total batch F1 sequence bytes uploaded */
  double            t_vit_lt;        /* GPU scanning Viterbi longtarget */
  double            t_vit_bias;      /* Bias-score recomputation before GPU scanning Viterbi */
  double            t_vit_cuda;      /* p7_cuda_ViterbiLongtarget call wall time */
  double            t_vit_sort;      /* CPU sort of GPU Viterbi seeds */
  double            t_vit_extend;    /* CPU Viterbi seed extend/merge/split/coordinate fixup */
  double            t_vit_pack;      /* Host packing inside p7_cuda_ViterbiLongtarget */
  double            t_vit_h2d;       /* CUDA H2D inside p7_cuda_ViterbiLongtarget */
  double            t_vit_f1_resident; /* Viterbi calls reading F1-packed device windows */
  double            t_vit_thresh;    /* CUDA threshold kernel inside p7_cuda_ViterbiLongtarget */
  double            t_vit_kernel;    /* CUDA scanning Viterbi kernel */
  double            t_vit_d2h;       /* CUDA D2H inside p7_cuda_ViterbiLongtarget */
  double            t_vit_alloc;     /* CUDA buffer growth/allocation inside p7_cuda_ViterbiLongtarget */
  double            t_vit_stream;    /* CUDA stream create/sync/destroy overhead in Viterbi helper */
  int64_t           vit_packed_bytes;/* Total packed Viterbi window bytes uploaded */
  int64_t           vit_launches;     /* GPU scanning Viterbi launch count */
  int               vit_grid_blocks;
  int               vit_block_threads;
  int               vit_dynamic_smem;
  int               vit_active_blocks_per_sm;
  int               vit_active_warps_per_sm;
  int               vit_max_warps_per_sm;
  int               vit_sm_count;
  double            vit_theoretical_occupancy;
  double            vit_grid_sm_coverage;
  double            vit_device_active_seconds;
  double            t_fwd_prefilter; /* GPU Forward pre-filter */
  double            t_gpu_fb_parser; /* GPU ForwardBackward parser batch */
  double            t_gpu_fb_pack;   /* host metadata/sequence packing before GPU FB parser */
  double            t_gpu_fb_h2d;    /* CUDA H2D inside GPU FB parser */
  double            t_gpu_fb_kernel; /* CUDA kernels inside GPU FB parser */
  double            t_gpu_fb_d2h;    /* CUDA D2H inside GPU FB parser */
  int64_t           gpu_fb_packed_bytes; /* total parser sequence bytes uploaded */
  double            t_cpu_workers;   /* CPU domaindef + hit reporting (wallclock around threaded section) */
  double            t_nucdb_open;     /* one-time .nucdb open/mmap, charged outside per-query search */
  double            t_nucdb_upload;   /* one-time .nucdb H2D upload, charged outside per-query search */
  double            t_nucdb_reconstruct; /* CPU reconstruction of full ESL_SQ strands from .nucdb chunks */
  double            t_post_search;    /* E-value/sort/dedup/threshold/output after GPU search loop */
  double            t_cuda_reset;     /* CUDA engine reset after a query */
  double            t_cuda_destroy;   /* CUDA engine/database teardown after all queries */
  double            t_gpu_loop_wall;  /* wall time around serial/nucdb GPU search loop */
  double            t_query_elapsed;  /* HMMER stopwatch-equivalent wall time for current query */
  double            t_query_presearch; /* per-query model/profile/output setup before GPU search loop */
  double            t_cuda_profile_create; /* per-query CUDA profile upload/setup before GPU search loop */
  double            t_cuda_profile_destroy; /* per-query CUDA profile teardown after GPU search loop */
  double            t_gpu_scratch_free;     /* per-query host scratch free after GPU search loop */
  double            t_query_postloop;  /* per-query cleanup/status work between GPU loop and post-search */
  double            t_program_prequery; /* setup before first query stopwatch starts */
  double            t_program_total;  /* process wall from main timing start to teardown */
  /* Sub-bucket breakdown of t_cpu_workers (wall-clock = max across worker threads) */
  double            t_worker_null;    /* base null model scoring inside CPU workers */
  double            t_worker_bias;    /* biased-composition scoring inside CPU workers */
  double            t_worker_bck;     /* CPU Backward parser inside CPU workers */
  double            t_worker_domain;  /* domain definition workflow inside CPU workers */
  double            t_worker_output;  /* hit object construction/threshold prep */
} NHMMER_GPU_INFO;

int nhmmer_gpu_serial_loop(NHMMER_GPU_INFO *info, ESL_SQFILE *dbfp,
                           int strands, NHMMER_GPU_IDLEN_CB idlen_cb, void *idlen_data,
                           int *ret_nseqs, int64_t *ret_nres);

int nhmmer_gpu_nucdb_loop(NHMMER_GPU_INFO *info, P7_NUCDB *ndb,
                          int strands, NHMMER_GPU_IDLEN_CB idlen_cb, void *idlen_data,
                          int *ret_nseqs, int64_t *ret_nres);

int nhmmer_gpu_nucdb_upload(NHMMER_GPU_INFO *info, P7_NUCDB *ndb,
                            char *errbuf, int errbuf_size);

#endif /*NHMMER_INTERNAL_INCLUDED*/
