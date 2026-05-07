#ifndef HMMSEARCH_INTERNAL_INCLUDED
#define HMMSEARCH_INTERNAL_INCLUDED

#include "easel.h"
#include "esl_dsqdata.h"
#include "esl_sq.h"

#ifdef HMMER_THREADS
#include "esl_workqueue.h"
#endif

#include "hmmer.h"
#include "cuda_msv.h"
#include "p7_gpudb.h"

typedef struct {
#ifdef HMMER_THREADS
  ESL_WORK_QUEUE   *queue;
#endif
  P7_BG            *bg;	         /* null model                              */
  P7_PIPELINE      *pli;         /* work pipeline                           */
  P7_TOPHITS       *th;          /* top hit results                         */
  P7_OPROFILE      *om;          /* optimized query profile                 */
  P7_CUDA_ENGINE   *cuda_engine; /* optional GPU engine                      */
  P7_CUDA_MSVPROFILE *cuda_msv;  /* optional GPU profile                     */
  int               gpu_batch_seqs; /* target sequences per GPU search batch  */
  int               gpu_batch_res;  /* target residues per GPU search batch   */
  int               gpu_load_seqs;  /* target sequences per dsqdata load      */
  int               gpu_load_res;   /* target residues per dsqdata load       */
  float             gpu_msv_slack;  /* optional extra nats for experiments     */
  int               gpu_fwd_prefilter; /* experimental CUDA Forward score gate  */
  int               gpu_fb_parser;     /* experimental CUDA Forward/Backward parser state */
  int               gpu_vit_prefilter; /* experimental CUDA Viterbi score gate  */
  int               gpu_vit_largem;    /* allow Viterbi prefilter on large-M models */
  int               gpu_fwd_largem;    /* allow Forward prefilter on large-M models */
  int               gpu_fwd_min_seqs;  /* min candidates for CUDA Forward batch */
  int               gpu_vit_min_seqs;  /* min candidates for CUDA Viterbi batch */
  int               gpu_fwd_min_res;   /* min residues for CUDA Forward batch */
  int               gpu_vit_min_res;   /* min residues for CUDA Viterbi batch */
  int               gpu_fwd_collect_seqs; /* preferred Forward candidate slab size */
  int               gpu_vit_collect_seqs; /* preferred Viterbi candidate slab size */
  int               gpu_fwd_collect_res;  /* preferred Forward candidate slab residues */
  int               gpu_vit_collect_res;  /* preferred Viterbi candidate slab residues */
  int               gpu_fwd_compare;   /* debug CUDA-vs-CPU Forward score check */
  int               gpu_vit_compare;   /* debug CUDA-vs-CPU Viterbi score check */
  int               gpu_fb_compare;    /* debug CUDA-vs-CPU parser state check  */
  int               gpu_previt_compare;/* debug CUDA-vs-CPU null/bias/F1 checks */
  int               gpu_ssv_compare;   /* debug CUDA SSV-vs-monolithic MSV check */
  P7_GPUDB         *gpudb;             /* optional GPU-native mmap'd database    */
} WORKER_INFO;

double hmmsearch_WallTime(void);
int    hmmsearch_gpu_serial_loop(WORKER_INFO *info, ESL_DSQDATA *dd, int n_targetseqs);

#endif /*HMMSEARCH_INTERNAL_INCLUDED*/
