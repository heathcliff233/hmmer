/* nhmmer GPU pipeline: CPU-only stubs.
 *
 * The full GPU pipeline lives in cuda/p7_cuda_nhmmer_search.c,
 * cuda/p7_cuda_nhmmer_filters.c, nhmmer_gpu_workers.c,
 * nhmmer_gpu_windows.c, and nhmmer_gpu_seqhelpers.c. This translation
 * unit only provides fallback definitions for the public entry points
 * when HMMER is configured without CUDA support, so the link still
 * resolves the symbols that nhmmer.c references unconditionally.
 */
#include <p7_config.h>

#include "easel.h"

#include "hmmer.h"
#include "nhmmer_internal.h"

#ifndef HMMER_CUDA

int
nhmmer_gpu_nucdb_loop(NHMMER_GPU_INFO *info, P7_NUCDB *ndb,
                      int strands, NHMMER_GPU_IDLEN_CB idlen_cb, void *idlen_data,
                      int *ret_nseqs, int64_t *ret_nres)
{
  ESL_UNUSED(info); ESL_UNUSED(ndb); ESL_UNUSED(strands);
  ESL_UNUSED(idlen_cb); ESL_UNUSED(idlen_data);
  ESL_UNUSED(ret_nseqs); ESL_UNUSED(ret_nres);
  return eslENORESULT;
}

int
nhmmer_gpu_nucdb_upload(NHMMER_GPU_INFO *info, P7_NUCDB *ndb,
                        char *errbuf, int errbuf_size)
{
  ESL_UNUSED(info); ESL_UNUSED(ndb);
  ESL_UNUSED(errbuf); ESL_UNUSED(errbuf_size);
  return eslENORESULT;
}

#endif /* !HMMER_CUDA */
