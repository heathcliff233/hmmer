#ifndef NHMMER_INTERNAL_INCLUDED
#define NHMMER_INTERNAL_INCLUDED

#include "easel.h"
#include "esl_getopts.h"
#include "esl_sq.h"

#include "hmmer.h"
#include "cuda_msv.h"

typedef int (*NHMMER_GPU_IDLEN_CB)(void *data, int id, int64_t L);

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
} NHMMER_GPU_INFO;

int nhmmer_gpu_serial_loop(NHMMER_GPU_INFO *info, ESL_SQFILE *dbfp,
                           int strands, NHMMER_GPU_IDLEN_CB idlen_cb, void *idlen_data,
                           int *ret_nseqs, int64_t *ret_nres);

#endif /*NHMMER_INTERNAL_INCLUDED*/
