/* nhmmer GPU pipeline: window-list helpers and synthetic-chunk lifecycle.
 *
 * Pure host-side glue used by the GPU pipeline to translate between
 * P7_HMM_WINDOWLIST, the GPU's window structs, and the synthetic
 * ESL_DSQDATA_CHUNK we hand to GPU batch APIs. No CUDA calls.
 */
#include <p7_config.h>

#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "easel.h"
#include "esl_dsqdata.h"
#include "esl_sq.h"

#include "hmmer.h"
#include "nhmmer_internal.h"

#ifdef HMMER_CUDA
#include "cuda/p7_cuda.h"
#include "cuda/nhmmer_cuda_internal.h"

int
nhmmer_gpu_windowlist_alloc(P7_HMM_WINDOWLIST *wl, int n)
{
  int status;
  wl->count = 0;
  wl->size  = ESL_MAX(n, 1);
  ESL_ALLOC(wl->windows, sizeof(P7_HMM_WINDOW) * wl->size);
  return eslOK;

ERROR:
  wl->windows = NULL;
  wl->size = wl->count = 0;
  return eslEMEM;
}

void
nhmmer_gpu_fill_ssv_windows(P7_HMM_WINDOWLIST *wl, const P7_CUDA_LT_WINDOW *gpu_windows,
                            int n, uint32_t target_len)
{
  for (int i = 0; i < n; i++) {
    P7_HMM_WINDOW *w = wl->windows + i;
    w->id              = 0;
    w->n               = gpu_windows[i].target_start;
    w->fm_n            = 0;
    w->k               = (uint16_t) gpu_windows[i].model_end;
    w->length          = gpu_windows[i].model_end - gpu_windows[i].model_start + 1;
    w->score           = gpu_windows[i].score;
    w->complementarity = p7_NOCOMPLEMENT;
    w->target_len      = target_len;
  }
  wl->count = n;
}

int
nhmmer_gpu_copy_windows(P7_HMM_WINDOWLIST *wl, const P7_HMM_WINDOW *windows, int n)
{
  int status;
  status = nhmmer_gpu_windowlist_alloc(wl, n);
  if (status != eslOK) return status;
  if (n > 0) memcpy(wl->windows, windows, sizeof(P7_HMM_WINDOW) * n);
  wl->count = n;
  return eslOK;
}

void
nhmmer_gpu_fill_vit_seed_windows(P7_HMM_WINDOWLIST *wl, const P7_CUDA_VIT_LT_WINDOW *gpu_windows,
                                 int n, const P7_HMM_WINDOWLIST *input_wl)
{
  for (int i = 0; i < n; i++) {
    int win_id = gpu_windows[i].window_id;
    P7_HMM_WINDOW *w = wl->windows + i;
    w->id              = (uint32_t) win_id;
    w->n               = (uint32_t) gpu_windows[i].position;
    w->fm_n            = 0;
    w->k               = (uint16_t) gpu_windows[i].model_k;
    w->length          = 1;
    w->score           = 0.0f;
    w->complementarity = p7_NOCOMPLEMENT;
    w->target_len      = (uint32_t) input_wl->windows[win_id].length;
  }
  wl->count = n;
}

int
nhmmer_gpu_vit_window_compare(const void *a, const void *b)
{
  const P7_CUDA_VIT_LT_WINDOW *wa = (const P7_CUDA_VIT_LT_WINDOW *)a;
  const P7_CUDA_VIT_LT_WINDOW *wb = (const P7_CUDA_VIT_LT_WINDOW *)b;

  if (wa->window_id != wb->window_id) return (wa->window_id < wb->window_id) ? -1 : 1;
  if (wa->position  != wb->position)  return (wa->position  < wb->position)  ? -1 : 1;
  if (wa->model_k   != wb->model_k)   return (wa->model_k   < wb->model_k)   ? -1 : 1;
  return 0;
}

int
nhmmer_gpu_window_batch_init(NHMMER_GPU_WINDOW_BATCH *wb, int alloc)
{
  int status;
  wb->nwindows = 0;
  wb->alloc    = alloc;

  ESL_ALLOC(wb->dsq_ptrs, sizeof(ESL_DSQ *)  * alloc);
  ESL_ALLOC(wb->lengths,  sizeof(int64_t)    * alloc);
  ESL_ALLOC(wb->names,    sizeof(char *)     * alloc);
  ESL_ALLOC(wb->accs,     sizeof(char *)     * alloc);
  ESL_ALLOC(wb->descs,    sizeof(char *)     * alloc);
  ESL_ALLOC(wb->taxids,   sizeof(int32_t)    * alloc);
  ESL_ALLOC(wb->win_idx,  sizeof(int)        * alloc);

  memset(&wb->chu, 0, sizeof(ESL_DSQDATA_CHUNK));
  return eslOK;

ERROR:
  return eslEMEM;
}

void
nhmmer_gpu_window_batch_free(NHMMER_GPU_WINDOW_BATCH *wb)
{
  if (wb->dsq_ptrs) free(wb->dsq_ptrs);
  if (wb->lengths)  free(wb->lengths);
  if (wb->names)    free(wb->names);
  if (wb->accs)     free(wb->accs);
  if (wb->descs)    free(wb->descs);
  if (wb->taxids)   free(wb->taxids);
  if (wb->win_idx)  free(wb->win_idx);
}

/* Pack merged windows into a synthetic ESL_DSQDATA_CHUNK for GPU batch APIs.
 * Zero-copy: dsq pointers point into the parent sequence. */
int
nhmmer_gpu_window_batch_pack(NHMMER_GPU_WINDOW_BATCH *wb, const ESL_SQ *sq,
                             P7_HMM_WINDOWLIST *wl)
{
  int i;
  int status;

  if (wl->count > wb->alloc) {
    wb->alloc = wl->count;
    ESL_REALLOC(wb->dsq_ptrs, sizeof(ESL_DSQ *)  * wb->alloc);
    ESL_REALLOC(wb->lengths,  sizeof(int64_t)    * wb->alloc);
    ESL_REALLOC(wb->names,    sizeof(char *)     * wb->alloc);
    ESL_REALLOC(wb->accs,     sizeof(char *)     * wb->alloc);
    ESL_REALLOC(wb->descs,    sizeof(char *)     * wb->alloc);
    ESL_REALLOC(wb->taxids,   sizeof(int32_t)    * wb->alloc);
    ESL_REALLOC(wb->win_idx,  sizeof(int)        * wb->alloc);
  }

  wb->nwindows = wl->count;
  for (i = 0; i < wl->count; i++) {
    wb->dsq_ptrs[i] = sq->dsq + wl->windows[i].n - 1;
    wb->lengths[i]  = wl->windows[i].length;
    wb->names[i]    = nhmmer_empty_str;
    wb->accs[i]     = nhmmer_empty_str;
    wb->descs[i]    = nhmmer_empty_str;
    wb->taxids[i]   = -1;
    wb->win_idx[i]  = i;
  }

  wb->chu.i0       = 0;
  wb->chu.N        = wl->count;
  wb->chu.dsq      = wb->dsq_ptrs;
  wb->chu.name     = wb->names;
  wb->chu.acc      = wb->accs;
  wb->chu.desc     = wb->descs;
  wb->chu.taxid    = wb->taxids;
  wb->chu.L        = wb->lengths;
  wb->chu.smem     = NULL;
  wb->chu.psq      = NULL;
  wb->chu.pn       = 0;
  wb->chu.metadata = NULL;
  wb->chu.mdalloc  = 0;
  wb->chu.nxt      = NULL;

  return eslOK;

ERROR:
  return eslEMEM;
}

/* Describe merged windows as a synthetic chunk without binding host sequence
 * pointers. Resident .nucdb parser batches gather sequence bytes on the GPU;
 * the parser API still needs the chunk's lengths/metadata for allocation,
 * reuse checks, and statistics. */
int
nhmmer_gpu_window_batch_describe(NHMMER_GPU_WINDOW_BATCH *wb, P7_HMM_WINDOWLIST *wl)
{
  int i;
  int status;

  if (wl->count > wb->alloc) {
    wb->alloc = wl->count;
    ESL_REALLOC(wb->dsq_ptrs, sizeof(ESL_DSQ *)  * wb->alloc);
    ESL_REALLOC(wb->lengths,  sizeof(int64_t)    * wb->alloc);
    ESL_REALLOC(wb->names,    sizeof(char *)     * wb->alloc);
    ESL_REALLOC(wb->accs,     sizeof(char *)     * wb->alloc);
    ESL_REALLOC(wb->descs,    sizeof(char *)     * wb->alloc);
    ESL_REALLOC(wb->taxids,   sizeof(int32_t)    * wb->alloc);
    ESL_REALLOC(wb->win_idx,  sizeof(int)        * wb->alloc);
  }

  wb->nwindows = wl->count;
  for (i = 0; i < wl->count; i++) {
    wb->dsq_ptrs[i] = NULL;
    wb->lengths[i]  = wl->windows[i].length;
    wb->names[i]    = nhmmer_empty_str;
    wb->accs[i]     = nhmmer_empty_str;
    wb->descs[i]    = nhmmer_empty_str;
    wb->taxids[i]   = -1;
    wb->win_idx[i]  = i;
  }

  wb->chu.i0       = 0;
  wb->chu.N        = wl->count;
  wb->chu.dsq      = wb->dsq_ptrs;
  wb->chu.name     = wb->names;
  wb->chu.acc      = wb->accs;
  wb->chu.desc     = wb->descs;
  wb->chu.taxid    = wb->taxids;
  wb->chu.L        = wb->lengths;
  wb->chu.smem     = NULL;
  wb->chu.psq      = NULL;
  wb->chu.pn       = 0;
  wb->chu.metadata = NULL;
  wb->chu.mdalloc  = 0;
  wb->chu.nxt      = NULL;

  return eslOK;

ERROR:
  return eslEMEM;
}

#endif /* HMMER_CUDA */
