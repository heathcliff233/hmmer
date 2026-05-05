#include <p7_config.h>

#include <stdio.h>

#include "cuda_msv.h"

int
p7_cuda_Available(char *errbuf, int errbuf_size)
{
  if (errbuf && errbuf_size > 0)
    snprintf(errbuf, errbuf_size, "HMMER was built without CUDA support");
  return eslENOTFOUND;
}

int
p7_cuda_engine_Create(int device_id, P7_CUDA_ENGINE **ret_engine, char *errbuf, int errbuf_size)
{
  if (ret_engine) *ret_engine = NULL;
  if (errbuf && errbuf_size > 0)
    snprintf(errbuf, errbuf_size, "HMMER was built without CUDA support");
  return eslENOTFOUND;
}

void
p7_cuda_engine_Destroy(P7_CUDA_ENGINE *engine)
{
  return;
}

void
p7_cuda_engine_GetStats(const P7_CUDA_ENGINE *engine, P7_CUDA_MSV_STATS *stats)
{
  if (stats) {
    stats->h2d_seconds     = 0.0;
    stats->kernel_seconds  = 0.0;
    stats->d2h_seconds     = 0.0;
    stats->bias_h2d_seconds    = 0.0;
    stats->bias_kernel_seconds = 0.0;
    stats->bias_d2h_seconds    = 0.0;
    stats->fwd_h2d_seconds     = 0.0;
    stats->fwd_kernel_seconds  = 0.0;
    stats->fwd_d2h_seconds     = 0.0;
    stats->vit_h2d_seconds     = 0.0;
    stats->vit_kernel_seconds  = 0.0;
    stats->vit_d2h_seconds     = 0.0;
    stats->bck_h2d_seconds     = 0.0;
    stats->bck_kernel_seconds  = 0.0;
    stats->bck_d2h_seconds     = 0.0;
    stats->nseqs           = 0;
    stats->nres            = 0;
    stats->nbatches        = 0;
    stats->fwd_nseqs       = 0;
    stats->fwd_nres        = 0;
    stats->fwd_nbatches    = 0;
    stats->vit_nseqs       = 0;
    stats->vit_nres        = 0;
    stats->vit_nbatches    = 0;
    stats->bck_nseqs       = 0;
    stats->bck_nres        = 0;
    stats->bck_nbatches    = 0;
  }
}

int
p7_cuda_msvprofile_Create(const P7_OPROFILE *om, P7_CUDA_MSVPROFILE **ret_cuom, char *errbuf, int errbuf_size)
{
  if (ret_cuom) *ret_cuom = NULL;
  if (errbuf && errbuf_size > 0)
    snprintf(errbuf, errbuf_size, "HMMER was built without CUDA support");
  return eslENOTFOUND;
}

void
p7_cuda_msvprofile_Destroy(P7_CUDA_MSVPROFILE *cuom)
{
  return;
}

int
p7_cuda_msvprofile_UpdateLength(P7_CUDA_MSVPROFILE *cuom, const P7_OPROFILE *om, int L, char *errbuf, int errbuf_size)
{
  if (errbuf && errbuf_size > 0)
    snprintf(errbuf, errbuf_size, "HMMER was built without CUDA support");
  return eslENOTFOUND;
}

int
p7_cuda_MSVFilter(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                  const ESL_DSQ *dsq, int L, float *ret_sc,
                  char *errbuf, int errbuf_size)
{
  if (errbuf && errbuf_size > 0)
    snprintf(errbuf, errbuf_size, "HMMER was built without CUDA support");
  return eslENOTFOUND;
}

int
p7_cuda_MSVFilterBatch(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                       ESL_SQ_BLOCK *block, float *scores, int *statuses,
                       char *errbuf, int errbuf_size)
{
  if (errbuf && errbuf_size > 0)
    snprintf(errbuf, errbuf_size, "HMMER was built without CUDA support");
  return eslENOTFOUND;
}

int
p7_cuda_MSVFilterDsqdataChunk(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                              ESL_DSQDATA_CHUNK *chu, float *scores, int *statuses,
                              char *errbuf, int errbuf_size)
{
  if (errbuf && errbuf_size > 0)
    snprintf(errbuf, errbuf_size, "HMMER was built without CUDA support");
  return eslENOTFOUND;
}

int
p7_cuda_BiasFilterDsqdataChunk(P7_CUDA_ENGINE *engine, const P7_BG *bg,
                               ESL_DSQDATA_CHUNK *chu, float *filtersc,
                               char *errbuf, int errbuf_size)
{
  if (errbuf && errbuf_size > 0)
    snprintf(errbuf, errbuf_size, "HMMER was built without CUDA support");
  return eslENOTFOUND;
}

int
p7_cuda_ForwardScoreDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                  ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                  float *scores, int *statuses,
                                  char *errbuf, int errbuf_size)
{
  if (errbuf && errbuf_size > 0)
    snprintf(errbuf, errbuf_size, "HMMER was built without CUDA support");
  return eslENOTFOUND;
}

int
p7_cuda_ViterbiScoreDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                  ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                  float *scores, int *statuses,
                                  char *errbuf, int errbuf_size)
{
  if (errbuf && errbuf_size > 0)
    snprintf(errbuf, errbuf_size, "HMMER was built without CUDA support");
  return eslENOTFOUND;
}

int
p7_cuda_ForwardBackwardParser(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                              const ESL_DSQ *dsq, int L, P7_OMX *oxf, P7_OMX *oxb,
                              float *ret_fwdsc, float *ret_bcksc,
                              char *errbuf, int errbuf_size)
{
  if (errbuf && errbuf_size > 0)
    snprintf(errbuf, errbuf_size, "HMMER was built without CUDA support");
  return eslENOTFOUND;
}

int
p7_cuda_ForwardBackwardParserDsqdataSubset(P7_CUDA_ENGINE *engine, const P7_CUDA_MSVPROFILE *cuom,
                                           ESL_DSQDATA_CHUNK *chu, const int *seqidx, int nidx,
                                           const size_t *x_offsets, size_t total_xcells,
                                           float *xf, float *xb, float *scores, int *statuses,
                                           char *errbuf, int errbuf_size)
{
  if (errbuf && errbuf_size > 0)
    snprintf(errbuf, errbuf_size, "HMMER was built without CUDA support");
  return eslENOTFOUND;
}
