#include <p7_config.h>

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "easel.h"
#include "esl_dsqdata.h"
#include "esl_exponential.h"
#include "esl_gumbel.h"
#include "esl_sq.h"

#include "hmmer.h"
#include "hmmsearch_internal.h"
#include "p7_gpudb.h"

typedef struct {
  int                nchunks;
  int                chunk_alloc;
  ESL_DSQDATA_CHUNK **chunks;
  ESL_DSQDATA_CHUNK  view;
} GPU_SEARCH_BATCH;

typedef struct {
  float nullsc;
  float filtersc;
  float usc;
  int   passed_msv;
  int   passed_bias;
} GPU_PREVIT_RESULT;

typedef struct {
  float *xmx;
  int    allocXR;
  int    M;
  int    L;
  int    has_own_scales;
  float  totscale;
} GPU_OMX_BINDING;

static int gpu_search_batch_Init(GPU_SEARCH_BATCH *batch);
static void gpu_search_batch_Reset(GPU_SEARCH_BATCH *batch);
static void gpu_search_batch_Destroy(GPU_SEARCH_BATCH *batch);
static int gpu_search_batch_AddChunk(GPU_SEARCH_BATCH *batch, ESL_DSQDATA_CHUNK *chu);
static int gpu_OverlayGpudb(ESL_DSQDATA_CHUNK *chu, const P7_GPUDB *gdb);
static int gpu_AssembleBatchFromGpudb(GPU_SEARCH_BATCH *batch, const P7_GPUDB *gdb,
                                      int64_t seq0, int nseq);
static int gpu_PreViterbiBoundary(WORKER_INFO *info, const ESL_SQ *dbsq, float gpu_usc, int gpu_msv_status,
                                  float gpu_nullsc, float gpu_filtersc, GPU_PREVIT_RESULT *ret);
static int gpu_BindSeqView(WORKER_INFO *info, ESL_SQ *dbsq, ESL_DSQ *dbsq_dsqmem, int64_t dbsq_salloc,
                           ESL_DSQDATA_CHUNK *chu, int i);
static int gpu_MaterializeSeq(WORKER_INFO *info, ESL_SQ *dbsq, ESL_DSQ *dbsq_dsqmem, int64_t dbsq_salloc,
                              ESL_DSQDATA_CHUNK *chu, int i);
static void gpu_RestoreSeqStorage(ESL_SQ *dbsq, ESL_DSQ *dbsq_dsqmem, int64_t dbsq_salloc);
static void gpu_BindOmxXmx(P7_OMX *ox, float *xmx, int M, int L, int has_own_scales, float totscale, GPU_OMX_BINDING *saved);
static void gpu_RestoreOmxXmx(P7_OMX *ox, const GPU_OMX_BINDING *saved);
static int gpu_PostFwd(WORKER_INFO *info, const ESL_SQ *dbsq, float nullsc, float filtersc, float fwdsc, char *errbuf, int errbuf_size);
static int gpu_fb_batch_Add(WORKER_INFO *info, int **ret_idx, float **ret_nullsc, float **ret_filtersc, float **ret_fwdsc,
                            int *ret_n, int *ret_alloc, int idx, float nullsc, float filtersc, float fwdsc);
static int gpu_fb_batch_Append(int **ret_idx, float **ret_nullsc, float **ret_filtersc, float **ret_fwdsc,
                               int *ret_n, int *ret_alloc, int idx, float nullsc, float filtersc, float fwdsc);
static int gpu_AppendFwdCandidate(WORKER_INFO *info, int *gpu_fwd_idx, float *gpu_fwd_filtersc_subset,
                                  int *gpu_fwd_n, int i, float filtersc);
static int gpu_ProcessFbBatch(WORKER_INFO *info, ESL_DSQDATA_CHUNK *chu, ESL_SQ *dbsq, ESL_DSQ *dbsq_dsqmem,
                              int64_t dbsq_salloc, int *idx, float *nullsc, float *filtersc, float *fwdsc, int n,
                              char *errbuf, int errbuf_size);
static void gpu_CompareParserState(WORKER_INFO *info, const ESL_SQ *dbsq, const P7_OMX *cpu_oxf, const P7_OMX *cpu_oxb,
                                   float cpu_fwdsc, float gpu_fwdsc, float gpu_bcksc);
static int gpu_CandidateResidues(const ESL_DSQDATA_CHUNK *chu, const int *idx, int n);
static void gpu_ResolveSurvivorThresholds(const WORKER_INFO *info, int M,
                                          int *vit_min_cands, int *fwd_min_cands,
                                          int *vit_collect_cands, int *fwd_collect_cands,
                                          int *vit_min_res, int *fwd_min_res,
                                          int *vit_collect_res, int *fwd_collect_res);

static int
gpu_AppendFwdCandidate(WORKER_INFO *info, int *gpu_fwd_idx, float *gpu_fwd_filtersc_subset,
                       int *gpu_fwd_n, int i, float filtersc)
{
  if (!gpu_fwd_idx || !gpu_fwd_filtersc_subset || !gpu_fwd_n) return eslEINVAL;
  gpu_fwd_idx[*gpu_fwd_n] = i;
  gpu_fwd_filtersc_subset[*gpu_fwd_n] = filtersc;
  (*gpu_fwd_n)++;
  return eslOK;
}

static int
gpu_CandidateResidues(const ESL_DSQDATA_CHUNK *chu, const int *idx, int n)
{
  int64_t total = 0;
  int j;

  if (!chu || !idx || n <= 0) return 0;
  for (j = 0; j < n; j++) {
    int si = idx[j];
    if (si < 0 || si >= chu->N) continue;
    total += (int64_t) chu->L[si];
    if (total >= INT32_MAX) return INT32_MAX;
  }
  return (int) total;
}

static int
gpu_ChooseTileCandidates(const WORKER_INFO *info, int is_fwd, int total_n, int total_res)
{
  int M = info->om ? info->om->M : 0;
  int tile_n;
  int tile_res;

  if (is_fwd) {
    tile_n = (M > 2048) ? 96 : (M > 1200) ? 192 : (M > 700 ? 320 : 640);
    tile_res = (M > 2048) ? 100000 : (M > 1200) ? 200000 : (M > 700 ? 350000 : 700000);
  } else {
    tile_n = (M > 2048) ? 512 : (M > 1200) ? 1024 : (M > 700 ? 2048 : 4096);
    tile_res = (M > 2048) ? 600000 : (M > 1200) ? 1120000 : (M > 700 ? 2000000 : 4000000);
  }
  if (total_n < tile_n) tile_n = total_n;
  if (tile_n < 1) tile_n = 1;
  if (total_res > 0) {
    int64_t by_res = ESL_MAX(1, ((int64_t) tile_n * tile_res) / total_res);
    if (by_res < tile_n) tile_n = (int) by_res;
    if (tile_n < 1) tile_n = 1;
  }
  return tile_n;
}

static void
gpu_ResolveSurvivorThresholds(const WORKER_INFO *info, int M,
                              int *vit_min_cands, int *fwd_min_cands,
                              int *vit_collect_cands, int *fwd_collect_cands,
                              int *vit_min_res, int *fwd_min_res,
                              int *vit_collect_res, int *fwd_collect_res)
{
  int batch_seqs, batch_res, mscale;
  int auto_vit_min_cands, auto_fwd_min_cands;
  int auto_vit_collect_cands, auto_fwd_collect_cands;
  int auto_vit_min_res, auto_fwd_min_res;
  int auto_vit_collect_res, auto_fwd_collect_res;

  batch_seqs = ESL_MAX(1, info->gpu_batch_seqs);
  batch_res  = ESL_MAX(1, info->gpu_batch_res);
  mscale     = ESL_MAX(1, M);

  auto_vit_min_cands     = ESL_MAX(32, ESL_MIN(256, batch_seqs / 64));
  auto_fwd_min_cands     = ESL_MAX(24, ESL_MIN(192, batch_seqs / 96));
  auto_vit_collect_cands = ESL_MAX(auto_vit_min_cands, ESL_MIN(1024, batch_seqs / 8));
  auto_fwd_collect_cands = ESL_MAX(auto_fwd_min_cands, ESL_MIN(768, batch_seqs / 10));

  auto_vit_min_res       = ESL_MAX(25000, ESL_MIN(batch_res / 8, mscale * 120));
  auto_fwd_min_res       = ESL_MAX(50000, ESL_MIN(batch_res / 6, mscale * 220));
  auto_vit_collect_res   = ESL_MAX(auto_vit_min_res, ESL_MIN(batch_res / 3, mscale * 480));
  auto_fwd_collect_res   = ESL_MAX(auto_fwd_min_res, ESL_MIN(batch_res / 2, mscale * 700));

  *vit_min_cands = (info->gpu_vit_min_seqs > 0) ? info->gpu_vit_min_seqs : auto_vit_min_cands;
  *fwd_min_cands = (info->gpu_fwd_min_seqs > 0) ? info->gpu_fwd_min_seqs : auto_fwd_min_cands;
  *vit_collect_cands = (info->gpu_vit_collect_seqs > 0) ? info->gpu_vit_collect_seqs : auto_vit_collect_cands;
  *fwd_collect_cands = (info->gpu_fwd_collect_seqs > 0) ? info->gpu_fwd_collect_seqs : auto_fwd_collect_cands;
  *vit_min_res = (info->gpu_vit_min_res > 0) ? info->gpu_vit_min_res : auto_vit_min_res;
  *fwd_min_res = (info->gpu_fwd_min_res > 0) ? info->gpu_fwd_min_res : auto_fwd_min_res;
  *vit_collect_res = (info->gpu_vit_collect_res > 0) ? info->gpu_vit_collect_res : auto_vit_collect_res;
  *fwd_collect_res = (info->gpu_fwd_collect_res > 0) ? info->gpu_fwd_collect_res : auto_fwd_collect_res;

  *vit_min_cands     = ESL_MAX(1, *vit_min_cands);
  *fwd_min_cands     = ESL_MAX(1, *fwd_min_cands);
  *vit_collect_cands = ESL_MAX(*vit_min_cands, *vit_collect_cands);
  *fwd_collect_cands = ESL_MAX(*fwd_min_cands, *fwd_collect_cands);
  *vit_min_res       = ESL_MAX(1, *vit_min_res);
  *fwd_min_res       = ESL_MAX(1, *fwd_min_res);
  *vit_collect_res   = ESL_MAX(*vit_min_res, *vit_collect_res);
  *fwd_collect_res   = ESL_MAX(*fwd_min_res, *fwd_collect_res);
}

static int
gpu_BindSeqView(WORKER_INFO *info, ESL_SQ *dbsq, ESL_DSQ *dbsq_dsqmem, int64_t dbsq_salloc,
                ESL_DSQDATA_CHUNK *chu, int i)
{
  if (!info || !dbsq || !chu || i < 0 || i >= chu->N) return eslEINVAL;

  dbsq->dsq    = dbsq_dsqmem;
  dbsq->salloc = dbsq_salloc;
  dbsq->name[0] = '\0';
  dbsq->acc[0]  = '\0';
  dbsq->desc[0] = '\0';
  dbsq->tax_id = chu->taxid[i];
  dbsq->dsq    = chu->dsq[i];
  dbsq->salloc = chu->L[i] + 2;
  dbsq->n      = chu->L[i];
  dbsq->start  = 1;
  dbsq->end    = chu->L[i];
  dbsq->C      = 0;
  dbsq->W      = chu->L[i];
  dbsq->L      = chu->L[i];
  return eslOK;
}

static int
gpu_MaterializeSeq(WORKER_INFO *info, ESL_SQ *dbsq, ESL_DSQ *dbsq_dsqmem, int64_t dbsq_salloc,
                   ESL_DSQDATA_CHUNK *chu, int i)
{
  double t0;
  int status;

  if (!info || !dbsq || !chu || i < 0 || i >= chu->N) return eslEINVAL;

  t0 = hmmsearch_WallTime();
  dbsq->dsq    = dbsq_dsqmem;
  dbsq->salloc = dbsq_salloc;
  esl_sq_Reuse(dbsq);

  if (chu->name[i] != NULL) {
    status = esl_sq_SetName(dbsq, chu->name[i]);
    if (status == eslOK) status = esl_sq_SetAccession(dbsq, chu->acc[i]);
    if (status == eslOK) status = esl_sq_SetDesc(dbsq, chu->desc[i]);
    if (status == eslOK) status = esl_sq_SetCoordComplete(dbsq, chu->L[i]);
    if (status != eslOK) return status;
    dbsq->tax_id = chu->taxid[i];
  } else if (info->gpudb && info->gpudb->has_metadata) {
    const char *name, *acc, *desc;
    int taxid;
    int64_t abs_idx = chu->i0 + i;
    status = p7_gpudb_GetMetadata(info->gpudb, abs_idx, &name, &acc, &desc, &taxid);
    if (status != eslOK) return status;
    status = esl_sq_SetName(dbsq, name);
    if (status == eslOK) status = esl_sq_SetAccession(dbsq, acc);
    if (status == eslOK) status = esl_sq_SetDesc(dbsq, desc);
    if (status == eslOK) status = esl_sq_SetCoordComplete(dbsq, chu->L[i]);
    if (status != eslOK) return status;
    dbsq->tax_id = taxid;
  } else {
    status = esl_sq_SetName(dbsq, "?");
    if (status == eslOK) status = esl_sq_SetCoordComplete(dbsq, chu->L[i]);
    if (status != eslOK) return status;
    dbsq->tax_id = -1;
  }

  dbsq->dsq    = chu->dsq[i];
  dbsq->salloc = chu->L[i] + 2;
  {
    double dt = hmmsearch_WallTime() - t0;
    info->pli->time_gpu_meta += dt;
    info->pli->exact_host_survivor_orchestration += dt;
  }
  return eslOK;
}

static void
gpu_RestoreSeqStorage(ESL_SQ *dbsq, ESL_DSQ *dbsq_dsqmem, int64_t dbsq_salloc)
{
  if (!dbsq) return;
  dbsq->dsq    = dbsq_dsqmem;
  dbsq->salloc = dbsq_salloc;
}

static void
gpu_BindOmxXmx(P7_OMX *ox, float *xmx, int M, int L, int has_own_scales, float totscale, GPU_OMX_BINDING *saved)
{
  if (!ox || !saved) return;
  saved->xmx            = ox->xmx;
  saved->allocXR        = ox->allocXR;
  saved->M              = ox->M;
  saved->L              = ox->L;
  saved->has_own_scales = ox->has_own_scales;
  saved->totscale       = ox->totscale;
  ox->xmx            = xmx;
  ox->allocXR        = L + 1;
  ox->M              = M;
  ox->L              = L;
  ox->has_own_scales = has_own_scales;
  ox->totscale       = totscale;
}

static void
gpu_RestoreOmxXmx(P7_OMX *ox, const GPU_OMX_BINDING *saved)
{
  if (!ox || !saved) return;
  ox->xmx            = saved->xmx;
  ox->allocXR        = saved->allocXR;
  ox->M              = saved->M;
  ox->L              = saved->L;
  ox->has_own_scales = saved->has_own_scales;
  ox->totscale       = saved->totscale;
}

static int
gpu_search_batch_Init(GPU_SEARCH_BATCH *batch)
{
  if (batch == NULL) return eslEINVAL;
  memset(batch, 0, sizeof(*batch));
  return eslOK;
}

static void
gpu_search_batch_Reset(GPU_SEARCH_BATCH *batch)
{
  if (batch == NULL) return;
  batch->nchunks = 0;
  batch->view.i0 = 0;
  batch->view.N  = 0;
  batch->view.smem = NULL;
}

static void
gpu_search_batch_Destroy(GPU_SEARCH_BATCH *batch)
{
  if (batch == NULL) return;
  free(batch->chunks);
  free(batch->view.dsq);
  free(batch->view.name);
  free(batch->view.acc);
  free(batch->view.desc);
  free(batch->view.taxid);
  free(batch->view.L);
  memset(batch, 0, sizeof(*batch));
}

static int
gpu_search_batch_Grow(GPU_SEARCH_BATCH *batch, int nchunks, int nseq)
{
  void *p;
  int status;

  if (nchunks > batch->chunk_alloc) {
    int new_alloc = batch->chunk_alloc ? batch->chunk_alloc : 4;
    while (new_alloc < nchunks) new_alloc *= 2;
    ESL_RALLOC(batch->chunks, p, sizeof(*batch->chunks) * new_alloc);
    batch->chunk_alloc = new_alloc;
  }
  if (nseq > batch->view.mdalloc) {
    int new_alloc = batch->view.mdalloc ? batch->view.mdalloc : 4096;
    while (new_alloc < nseq) new_alloc *= 2;
    ESL_RALLOC(batch->view.dsq,   p, sizeof(*batch->view.dsq)   * new_alloc);
    ESL_RALLOC(batch->view.name,  p, sizeof(*batch->view.name)  * new_alloc);
    ESL_RALLOC(batch->view.acc,   p, sizeof(*batch->view.acc)   * new_alloc);
    ESL_RALLOC(batch->view.desc,  p, sizeof(*batch->view.desc)  * new_alloc);
    ESL_RALLOC(batch->view.taxid, p, sizeof(*batch->view.taxid) * new_alloc);
    ESL_RALLOC(batch->view.L,     p, sizeof(*batch->view.L)     * new_alloc);
    batch->view.mdalloc = new_alloc;
  }
  return eslOK;

ERROR:
  return eslEMEM;
}

static int
gpu_search_batch_AddChunk(GPU_SEARCH_BATCH *batch, ESL_DSQDATA_CHUNK *chu)
{
  int offset;
  int status;

  if (batch == NULL || chu == NULL) return eslEINVAL;
  if ((status = gpu_search_batch_Grow(batch, batch->nchunks + 1, batch->view.N + chu->N)) != eslOK) return status;

  if (batch->nchunks == 0) batch->view.i0 = chu->i0;
  batch->chunks[batch->nchunks++] = chu;
  offset = batch->view.N;

  memcpy(batch->view.dsq   + offset, chu->dsq,   sizeof(*batch->view.dsq)   * chu->N);
  memcpy(batch->view.name  + offset, chu->name,  sizeof(*batch->view.name)  * chu->N);
  memcpy(batch->view.acc   + offset, chu->acc,   sizeof(*batch->view.acc)   * chu->N);
  memcpy(batch->view.desc  + offset, chu->desc,  sizeof(*batch->view.desc)  * chu->N);
  memcpy(batch->view.taxid + offset, chu->taxid, sizeof(*batch->view.taxid) * chu->N);
  memcpy(batch->view.L     + offset, chu->L,     sizeof(*batch->view.L)     * chu->N);
  batch->view.N += chu->N;
  return eslOK;
}

/* gpu_OverlayGpudb()
 * Replace a dsqdata chunk's dsq[] pointers with gpudb mmap pointers.
 * The chunk retains its metadata (name/acc/desc/taxid) from dsqdata.
 * We do NOT modify chu->smem (dsqdata owns that allocation for recycle).
 * The dsq[i] pointers are redirected to gpudb mmap for survivor materialization.
 */
static int
gpu_OverlayGpudb(ESL_DSQDATA_CHUNK *chu, const P7_GPUDB *gdb)
{
  int64_t seq0;
  int i;

  if (!chu || !gdb) return eslEINVAL;
  seq0 = chu->i0;
  if (seq0 < 0 || seq0 + chu->N > (int64_t) gdb->hdr.nseq) return eslEINVAL;

  for (i = 0; i < chu->N; i++)
    chu->dsq[i] = gdb->seq_data + gdb->offsets[seq0 + i];
  return eslOK;
}

/* gpu_AssembleBatchFromGpudb()
 * Populate batch.view directly from gpudb, avoiding dsqdata entirely.
 * Sets up N, i0, dsq[], L[] from gpudb's mmap'd index.
 * name/acc/desc/taxid are left NULL — MaterializeSeq reads from gpudb metadata.
 */
static int
gpu_AssembleBatchFromGpudb(GPU_SEARCH_BATCH *batch, const P7_GPUDB *gdb,
                           int64_t seq0, int nseq)
{
  int status;

  if (!batch || !gdb) return eslEINVAL;
  if (seq0 < 0 || seq0 + nseq > (int64_t) gdb->hdr.nseq) return eslEINVAL;

  if ((status = gpu_search_batch_Grow(batch, 1, nseq)) != eslOK) return status;

  batch->view.i0 = seq0;
  batch->view.N  = nseq;
  batch->nchunks = 0;

  for (int i = 0; i < nseq; i++) {
    batch->view.dsq[i]   = gdb->seq_data + gdb->offsets[seq0 + i];
    batch->view.L[i]     = (int64_t) gdb->lengths[seq0 + i];
    batch->view.name[i]  = NULL;
    batch->view.acc[i]   = NULL;
    batch->view.desc[i]  = NULL;
    batch->view.taxid[i] = -1;
  }
  batch->view.smem = gdb->seq_data + gdb->offsets[seq0];
  return eslOK;
}

static int
gpu_fb_batch_Append(int **ret_idx, float **ret_nullsc, float **ret_filtersc, float **ret_fwdsc,
                    int *ret_n, int *ret_alloc, int idx, float nullsc, float filtersc, float fwdsc)
{
  int n;
  int alloc;
  int status;

  if (!ret_idx || !ret_nullsc || !ret_filtersc || !ret_fwdsc || !ret_n || !ret_alloc) return eslEINVAL;

  n = *ret_n;
  alloc = *ret_alloc;
  if (n >= alloc) {
    int new_alloc = alloc > 0 ? alloc * 2 : 16;
    void *p;
    ESL_RALLOC(*ret_idx,     p, sizeof(int)   * new_alloc);
    ESL_RALLOC(*ret_nullsc,  p, sizeof(float) * new_alloc);
    ESL_RALLOC(*ret_filtersc,p, sizeof(float) * new_alloc);
    ESL_RALLOC(*ret_fwdsc,   p, sizeof(float) * new_alloc);
    *ret_alloc = new_alloc;
  }
  (*ret_idx)[n] = idx;
  (*ret_nullsc)[n] = nullsc;
  (*ret_filtersc)[n] = filtersc;
  (*ret_fwdsc)[n] = fwdsc;
  *ret_n = n + 1;
  return eslOK;

ERROR:
  return eslEMEM;
}

static int
gpu_fb_batch_Add(WORKER_INFO *info, int **ret_idx, float **ret_nullsc, float **ret_filtersc, float **ret_fwdsc,
                 int *ret_n, int *ret_alloc, int idx, float nullsc, float filtersc, float fwdsc)
{
  float seq_score;
  double P;

  if (!ret_idx || !ret_nullsc || !ret_filtersc || !ret_fwdsc || !ret_n || !ret_alloc) return eslEINVAL;
  if (!info) return eslEINVAL;
  seq_score = (fwdsc - filtersc) / eslCONST_LOG2;
  P = esl_exp_surv(seq_score, info->om->evparam[p7_FTAU], info->om->evparam[p7_FLAMBDA]);
  if (P > info->pli->F3) return eslOK;

  return gpu_fb_batch_Append(ret_idx, ret_nullsc, ret_filtersc, ret_fwdsc,
                             ret_n, ret_alloc, idx, nullsc, filtersc, fwdsc);
}

static void
gpu_CompareParserState(WORKER_INFO *info, const ESL_SQ *dbsq, const P7_OMX *cpu_oxf, const P7_OMX *cpu_oxb,
                       float cpu_fwdsc, float gpu_fwdsc, float gpu_bcksc)
{
  P7_DOMAINDEF *cpu_ddef = NULL;
  P7_DOMAINDEF *gpu_ddef = NULL;
  float cpu_bcksc = 0.0f;
  float max_fwd = 0.0f;
  float max_bck = 0.0f;
  float max_mocc = 0.0f;
  float max_btot = 0.0f;
  float max_etot = 0.0f;
  int   max_fwd_i = 0;
  int   max_fwd_s = 0;
  int   max_bck_i = 0;
  int   max_bck_s = 0;
  int   max_mocc_i = 0;
  int   max_btot_i = 0;
  int   max_etot_i = 0;
  int   status;

  if (!info || !dbsq || !cpu_oxf || !cpu_oxb) return;
  if (cpu_oxb->xmx[p7X_N] > 0.0f) cpu_bcksc = cpu_oxb->totscale + logf(cpu_oxb->xmx[p7X_N]);

  for (int i = 0; i <= dbsq->n; i++) {
    for (int s = 0; s < p7X_NXCELLS; s++) {
      float df = fabsf(cpu_oxf->xmx[i*p7X_NXCELLS+s] - info->pli->oxf->xmx[i*p7X_NXCELLS+s]);
      float db = fabsf(cpu_oxb->xmx[i*p7X_NXCELLS+s] - info->pli->oxb->xmx[i*p7X_NXCELLS+s]);
      if (df > max_fwd) { max_fwd = df; max_fwd_i = i; max_fwd_s = s; }
      if (db > max_bck) { max_bck = db; max_bck_i = i; max_bck_s = s; }
    }
  }

  cpu_ddef = p7_domaindef_Create(NULL);
  gpu_ddef = p7_domaindef_Create(NULL);
  if (!cpu_ddef || !gpu_ddef) goto REPORT;
  if ((status = p7_domaindef_GrowTo(cpu_ddef, dbsq->n)) != eslOK) goto REPORT;
  if ((status = p7_domaindef_GrowTo(gpu_ddef, dbsq->n)) != eslOK) goto REPORT;
  if ((status = p7_DomainDecoding(info->om, cpu_oxf, cpu_oxb, cpu_ddef)) != eslOK) goto REPORT;
  if ((status = p7_DomainDecoding(info->om, info->pli->oxf, info->pli->oxb, gpu_ddef)) != eslOK) goto REPORT;
  for (int i = 0; i <= dbsq->n; i++) {
    float dm = fabsf(cpu_ddef->mocc[i] - gpu_ddef->mocc[i]);
    float db = fabsf(cpu_ddef->btot[i] - gpu_ddef->btot[i]);
    float de = fabsf(cpu_ddef->etot[i] - gpu_ddef->etot[i]);
    if (dm > max_mocc) { max_mocc = dm; max_mocc_i = i; }
    if (db > max_btot) { max_btot = db; max_btot_i = i; }
    if (de > max_etot) { max_etot = de; max_etot_i = i; }
  }

REPORT:
  if (fabsf(cpu_fwdsc - gpu_fwdsc) > 0.01f || fabsf(cpu_bcksc - gpu_bcksc) > 0.01f ||
      max_fwd > 0.01f || max_bck > 0.01f || max_mocc > 0.01f || max_btot > 0.01f || max_etot > 0.01f) {
    fprintf(stderr, "CUDAFB\t%s\tL=%ld\tcpu_fwd=%.6f\tgpu_fwd=%.6f\tcpu_bck=%.6f\tgpu_bck=%.6f\tmax_fwd=%.6f@%d/%d\tmax_bck=%.6f@%d/%d\tmax_mocc=%.6f@%d\tmax_btot=%.6f@%d\tmax_etot=%.6f@%d\n",
            dbsq->name ? dbsq->name : "-", (long) dbsq->n,
            cpu_fwdsc, gpu_fwdsc, cpu_bcksc, gpu_bcksc,
            max_fwd, max_fwd_i, max_fwd_s, max_bck, max_bck_i, max_bck_s,
            max_mocc, max_mocc_i, max_btot, max_btot_i, max_etot, max_etot_i);
  }
  p7_domaindef_Destroy(cpu_ddef);
  p7_domaindef_Destroy(gpu_ddef);
}

static int
gpu_PreViterbiBoundary(WORKER_INFO *info, const ESL_SQ *dbsq, float gpu_usc, int gpu_msv_status,
                       float gpu_nullsc, float gpu_filtersc, GPU_PREVIT_RESULT *ret)
{
  float usc = gpu_usc;
  float nullsc = gpu_nullsc;
  float filtersc = gpu_filtersc;
  float seq_score;
  double P;
  int passed_msv = FALSE;
  int passed_bias = FALSE;

  if (!info || !dbsq || !ret) return eslEINVAL;
  memset(ret, 0, sizeof(*ret));

  seq_score = (usc - nullsc) / eslCONST_LOG2;
  P = (gpu_msv_status == eslERANGE) ? 0.0 : esl_gumbel_surv(seq_score, info->om->evparam[p7_MMU], info->om->evparam[p7_MLAMBDA]);
  if (P <= info->pli->F1) {
    passed_msv = TRUE;
    if (info->pli->do_biasfilter) {
      seq_score = (usc + info->gpu_msv_slack - filtersc) / eslCONST_LOG2;
      P = esl_gumbel_surv(seq_score, info->om->evparam[p7_MMU], info->om->evparam[p7_MLAMBDA]);
      if (P <= info->pli->F1) passed_bias = TRUE;
    } else {
      filtersc = nullsc;
      passed_bias = TRUE;
    }
  }

  if (info->gpu_previt_compare) {
    float cpu_nullsc = 0.0f;
    float cpu_filtersc = 0.0f;
    float cpu_usc = -eslINFINITY;
    float cpu_usc_bias = -eslINFINITY;
    int cpu_msv_status = eslOK;
    int cpu_msv_status_bias = eslOK;
    int cpu_pass_msv = FALSE;
    int cpu_pass_bias = FALSE;
    float cpu_diff = 0.0f;
    float null_diff = 0.0f;
    float bias_diff = 0.0f;
    p7_bg_NullOne(info->bg, dbsq->dsq, dbsq->n, &cpu_nullsc);
    null_diff = gpu_nullsc - cpu_nullsc;
    if (info->pli->do_biasfilter) p7_bg_FilterScore(info->bg, dbsq->dsq, dbsq->n, &cpu_filtersc);
    else cpu_filtersc = cpu_nullsc;
    bias_diff = filtersc - cpu_filtersc;
    if (p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n) != eslOK) return eslEMEM;
    cpu_msv_status = p7_MSVFilter(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &cpu_usc);
    cpu_usc_bias = cpu_usc;
    cpu_msv_status_bias = cpu_msv_status;
    if (cpu_msv_status == eslERANGE) cpu_pass_msv = TRUE;
    else if (cpu_msv_status == eslOK) {
      seq_score = (cpu_usc - cpu_nullsc) / eslCONST_LOG2;
      cpu_pass_msv = (esl_gumbel_surv(seq_score, info->om->evparam[p7_MMU], info->om->evparam[p7_MLAMBDA]) <= info->pli->F1);
    }
    if (cpu_pass_msv) {
      if (info->pli->do_biasfilter) {
        seq_score = (cpu_usc - cpu_filtersc) / eslCONST_LOG2;
        P = esl_gumbel_surv(seq_score, info->om->evparam[p7_MMU], info->om->evparam[p7_MLAMBDA]);
        if (P > info->pli->F1 && cpu_msv_status == eslOK) {
          cpu_msv_status_bias = p7_MSVFilter(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &cpu_usc_bias);
          if (cpu_msv_status_bias == eslERANGE) cpu_pass_bias = TRUE;
          else if (cpu_msv_status_bias == eslOK) {
            seq_score = (cpu_usc_bias - cpu_filtersc) / eslCONST_LOG2;
            cpu_pass_bias = (esl_gumbel_surv(seq_score, info->om->evparam[p7_MMU], info->om->evparam[p7_MLAMBDA]) <= info->pli->F1);
          }
        } else {
          cpu_pass_bias = (P <= info->pli->F1);
        }
      } else cpu_pass_bias = TRUE;
    }
    cpu_diff = usc - cpu_usc_bias;
    if (cpu_pass_msv != passed_msv || cpu_pass_bias != passed_bias || fabsf(null_diff) > 1e-5f || (info->pli->do_biasfilter && fabsf(bias_diff) > 1e-3f)) {
      fprintf(stderr, "CUDAPREVIT\t%s\tL=%ld\tgpu_msv_status=%d\tcpu_msv_status=%d\tgpu_null=%.6f\tcpu_null=%.6f\tdnull=%.6g\tgpu_bias=%.6f\tcpu_bias=%.6f\tdbias=%.6g\tgpu_usc=%.6f\tcpu_usc=%.6f\tdusc=%.6g\tgpu_pass_msv=%d\tcpu_pass_msv=%d\tgpu_pass_bias=%d\tcpu_pass_bias=%d\n",
              dbsq->name ? dbsq->name : "-", (long) dbsq->n,
              gpu_msv_status, cpu_msv_status_bias,
              gpu_nullsc, cpu_nullsc, null_diff,
              filtersc, cpu_filtersc, bias_diff,
              usc, cpu_usc_bias, cpu_diff,
              passed_msv, cpu_pass_msv, passed_bias, cpu_pass_bias);
    }
  }

  ret->nullsc = nullsc;
  ret->filtersc = filtersc;
  ret->usc = usc;
  ret->passed_msv = passed_msv;
  ret->passed_bias = passed_bias;
  return eslOK;
}

static int
gpu_PostFwd(WORKER_INFO *info, const ESL_SQ *dbsq, float nullsc, float filtersc, float fwdsc, char *errbuf, int errbuf_size)
{
  P7_OMX *cpu_oxf = NULL;
  P7_OMX *cpu_oxb = NULL;
  float gpu_fwdsc = 0.0f;
  float gpu_bcksc = 0.0f;
  float seq_score;
  double P;
  double t0;
  double tpost0;
  double tpost1;
  int status;

  if (!info || !dbsq) return eslEINVAL;
  if (!info->gpu_fb_parser) {
    tpost0 = hmmsearch_WallTime();
    status = p7_Pipeline_PostFwd(info->pli, info->om, info->bg, dbsq, NULL, info->th, nullsc, filtersc, fwdsc);
    tpost1 = hmmsearch_WallTime();
    info->pli->exact_cpu_postfwd_domain_null2_output += tpost1 - tpost0;
    return status;
  }

  seq_score = (fwdsc - filtersc) / eslCONST_LOG2;
  P = esl_exp_surv(seq_score, info->om->evparam[p7_FTAU], info->om->evparam[p7_FLAMBDA]);
  if (P > info->pli->F3) return eslOK;

  if (info->gpu_fb_compare) {
    cpu_oxf = p7_omx_Create(info->om->M, 0, dbsq->n);
    cpu_oxb = p7_omx_Create(info->om->M, 0, dbsq->n);
    if (!cpu_oxf || !cpu_oxb) { status = eslEMEM; goto ERROR; }
    p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, cpu_oxf, &fwdsc);
    p7_BackwardParser(dbsq->dsq, dbsq->n, info->om, cpu_oxf, cpu_oxb, NULL);
  }

  t0 = hmmsearch_WallTime();
  status = p7_cuda_ForwardBackwardParser(info->cuda_engine, info->cuda_msv, dbsq->dsq, dbsq->n,
                                         info->pli->oxf, info->pli->oxb, &gpu_fwdsc, &gpu_bcksc,
                                         errbuf, errbuf_size);
  info->pli->time_bck += hmmsearch_WallTime() - t0;
  if (status != eslOK) goto ERROR;

  if (info->gpu_fb_compare) gpu_CompareParserState(info, dbsq, cpu_oxf, cpu_oxb, fwdsc, gpu_fwdsc, gpu_bcksc);
  tpost0 = hmmsearch_WallTime();
  status = p7_Pipeline_PostFwdWithParserMatrices(info->pli, info->om, info->bg, dbsq, NULL, info->th, nullsc, filtersc, gpu_fwdsc);
  tpost1 = hmmsearch_WallTime();
  info->pli->exact_cpu_postfwd_domain_null2_output += tpost1 - tpost0;

ERROR:
  p7_omx_Destroy(cpu_oxf);
  p7_omx_Destroy(cpu_oxb);
  return status;
}

static int
gpu_ProcessFbBatch(WORKER_INFO *info, ESL_DSQDATA_CHUNK *chu, ESL_SQ *dbsq, ESL_DSQ *dbsq_dsqmem,
                   int64_t dbsq_salloc, int *idx, float *nullsc, float *filtersc, float *fwdsc, int n,
                   char *errbuf, int errbuf_size)
{
  size_t *x_offsets = NULL;
  float *xf = NULL;
  float *xb = NULL;
  float *scores = NULL;
  int *statuses = NULL;
  size_t total_xcells = 0;
  double t0;
  int status = eslOK;

  if (!info || !chu || !dbsq || !idx || !nullsc || !filtersc || !fwdsc) return eslEINVAL;
  if (n <= 0) return eslOK;

  ESL_ALLOC(x_offsets, sizeof(size_t) * n);
  ESL_ALLOC(scores, sizeof(float) * 2 * n);
  ESL_ALLOC(statuses, sizeof(int) * 2 * n);
  for (int j = 0; j < n; j++) {
    int i = idx[j];
    if (i < 0 || i >= chu->N) { status = eslEINVAL; goto ERROR; }
    x_offsets[j] = total_xcells;
    total_xcells += (size_t) (chu->L[i] + 1) * p7X_NXCELLS;
  }
  ESL_ALLOC(xf, sizeof(float) * total_xcells);
  ESL_ALLOC(xb, sizeof(float) * total_xcells);

  t0 = hmmsearch_WallTime();
  status = p7_cuda_ForwardBackwardParserDsqdataSubset(info->cuda_engine, info->cuda_msv, chu, idx, n,
                                                      x_offsets, total_xcells, xf, xb, scores, statuses,
                                                      errbuf, errbuf_size);
  info->pli->time_bck += hmmsearch_WallTime() - t0;
  if (status != eslOK) goto ERROR;

  for (int j = 0; j < n; j++) {
    int i = idx[j];
    double tpost0;
    double tpost1;
    GPU_OMX_BINDING saved_oxf;
    GPU_OMX_BINDING saved_oxb;
    P7_OMX *cpu_oxf = NULL;
    P7_OMX *cpu_oxb = NULL;

    status = gpu_MaterializeSeq(info, dbsq, dbsq_dsqmem, dbsq_salloc, chu, i);
    if (status != eslOK) goto ERROR;
    t0 = hmmsearch_WallTime();
    p7_bg_SetLength(info->bg, dbsq->n);
    p7_oprofile_ReconfigLength(info->om, dbsq->n);
    info->pli->exact_host_survivor_orchestration += hmmsearch_WallTime() - t0;

    if (statuses[j * 2 + 0] != eslOK) { status = statuses[j * 2 + 0]; goto ERROR; }
    if (statuses[j * 2 + 1] != eslOK) { status = statuses[j * 2 + 1]; goto ERROR; }
    fwdsc[j] = scores[j * 2 + 0];

    gpu_BindOmxXmx(info->pli->oxf, xf + x_offsets[j], info->om->M, dbsq->n, TRUE, scores[j * 2 + 0], &saved_oxf);
    gpu_BindOmxXmx(info->pli->oxb, xb + x_offsets[j], info->om->M, dbsq->n, FALSE, scores[j * 2 + 1], &saved_oxb);

    if (info->gpu_fb_compare) {
      cpu_oxf = p7_omx_Create(info->om->M, 0, dbsq->n);
      cpu_oxb = p7_omx_Create(info->om->M, 0, dbsq->n);
      if (!cpu_oxf || !cpu_oxb) {
        status = eslEMEM;
        p7_omx_Destroy(cpu_oxf);
        p7_omx_Destroy(cpu_oxb);
        gpu_RestoreOmxXmx(info->pli->oxf, &saved_oxf);
        gpu_RestoreOmxXmx(info->pli->oxb, &saved_oxb);
        goto ERROR;
      }
      p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, cpu_oxf, &fwdsc[j]);
      p7_BackwardParser(dbsq->dsq, dbsq->n, info->om, cpu_oxf, cpu_oxb, NULL);
      gpu_CompareParserState(info, dbsq, cpu_oxf, cpu_oxb, fwdsc[j], scores[j * 2 + 0], scores[j * 2 + 1]);
      p7_omx_Destroy(cpu_oxf);
      p7_omx_Destroy(cpu_oxb);
    }

    tpost0 = hmmsearch_WallTime();
    status = p7_Pipeline_PostFwdWithParserMatrices(info->pli, info->om, info->bg, dbsq, NULL, info->th, nullsc[j], filtersc[j], fwdsc[j]);
    tpost1 = hmmsearch_WallTime();
    info->pli->exact_cpu_postfwd_domain_null2_output += tpost1 - tpost0;
    gpu_RestoreOmxXmx(info->pli->oxf, &saved_oxf);
    gpu_RestoreOmxXmx(info->pli->oxb, &saved_oxb);
    if (status != eslOK) goto ERROR;
    p7_pipeline_Reuse(info->pli);
    gpu_RestoreSeqStorage(dbsq, dbsq_dsqmem, dbsq_salloc);
  }

ERROR:
  free(x_offsets);
  free(xf);
  free(xb);
  free(scores);
  free(statuses);
  return status;
}

int
hmmsearch_gpu_serial_loop(WORKER_INFO *info, ESL_DSQDATA *dd, int n_targetseqs)
{
  int      sstatus = eslOK;
  ESL_SQ  *dbsq = NULL;
  int      seq_cnt = 0;
  int      status = eslOK;
  int      i;
  float    nullsc;
  float    usc;
  float    seq_score;
  double   P;
  char     errbuf[eslERRBUFSIZE];
  ESL_DSQDATA_CHUNK *chu = NULL;
  ESL_DSQDATA_CHUNK *search_chu = NULL;
  GPU_SEARCH_BATCH batch;
  float *gpu_scores = NULL;
  float *gpu_filtersc = NULL;
  float *gpu_fwd_scores = NULL;
  float *gpu_vit_scores = NULL;
  float *gpu_fwd_filtersc_subset = NULL;
  float *gpu_vit_filtersc_subset = NULL;
  float *gpu_nullsc = NULL;
  int *gpu_statuses = NULL;
  int *gpu_fwd_statuses = NULL;
  int *gpu_vit_statuses = NULL;
  int *gpu_fwd_passed = NULL;
  int *gpu_vit_passed = NULL;
  int *gpu_fwd_idx = NULL;
  int *gpu_vit_idx = NULL;
  int *gpu_fb_idx = NULL;
  float *gpu_fb_nullsc = NULL;
  float *gpu_fb_filtersc = NULL;
  float *gpu_fb_fwdsc = NULL;
  int gpu_fwd_n = 0;
  int gpu_vit_n = 0;
  int gpu_fb_n = 0;
  int gpu_fb_alloc = 0;
  int gpu_processed_n = 0;
  int gpu_pending_max_chunks = 2;
  int gpu_end_of_input = FALSE;
  ESL_DSQ *dbsq_dsqmem = NULL;
  int64_t  dbsq_salloc = 0;
  int      gpu_capacity = ESL_MAX(info->gpu_batch_seqs > 0 ? info->gpu_batch_seqs : eslDSQDATA_CHUNK_MAXSEQ,
                                  ESL_MAX(1, info->gpu_load_seqs) * gpu_pending_max_chunks);
  int64_t  batch_res = 0;
  int64_t  effective_load_res = 0;
  int64_t  batch_res_cap = 0;
  double   t0;
  int      gpu_vit_active = info->gpu_vit_prefilter && (info->gpu_vit_largem || info->om->M <= 2048);
  int      gpu_fwd_active = info->gpu_fwd_prefilter && (info->gpu_fwd_largem || info->om->M <= 1024);
  int      gpu_strict_mode = !(info->gpu_vit_compare || info->gpu_fwd_compare || info->gpu_fb_compare || info->gpu_previt_compare || info->gpu_ssv_compare);
  int      vit_cmp_status_mismatch = 0;
  int      vit_cmp_pass_mismatch = 0;
  int      vit_cmp_score_drift = 0;
  int      gpu_vit_res = 0;
  int      gpu_fwd_res = 0;
  int      gpu_vit_min_cands;
  int      gpu_fwd_min_cands;
  int      gpu_vit_collect_cands;
  int      gpu_fwd_collect_cands;
  int      gpu_vit_min_res;
  int      gpu_fwd_min_res;
  int      gpu_vit_collect_res;
  int      gpu_fwd_collect_res;

  gpu_search_batch_Init(&batch);
  if (gpu_strict_mode && info->gpu_vit_prefilter) gpu_vit_active = TRUE;
  if (gpu_strict_mode && info->gpu_fwd_prefilter) gpu_fwd_active = TRUE;
  if (dd == NULL && (!info->gpudb || !info->gpudb->has_metadata)) return eslEINVAL;
    dbsq = esl_sq_CreateDigital(info->om->abc);
    if (dbsq == NULL) { status = eslEMEM; goto ERROR; }
    dbsq_dsqmem = dbsq->dsq;
    dbsq_salloc = dbsq->salloc;
    ESL_ALLOC(gpu_scores, sizeof(float) * gpu_capacity);
    ESL_ALLOC(gpu_filtersc, sizeof(float) * gpu_capacity);
    ESL_ALLOC(gpu_fwd_scores, sizeof(float) * gpu_capacity);
    ESL_ALLOC(gpu_vit_scores, sizeof(float) * gpu_capacity);
    ESL_ALLOC(gpu_fwd_filtersc_subset, sizeof(float) * gpu_capacity);
    ESL_ALLOC(gpu_vit_filtersc_subset, sizeof(float) * gpu_capacity);
    ESL_ALLOC(gpu_nullsc, sizeof(float) * gpu_capacity);
    ESL_ALLOC(gpu_statuses, sizeof(int) * gpu_capacity);
    ESL_ALLOC(gpu_fwd_statuses, sizeof(int) * gpu_capacity);
    ESL_ALLOC(gpu_vit_statuses, sizeof(int) * gpu_capacity);
    ESL_ALLOC(gpu_fwd_passed, sizeof(int) * gpu_capacity);
    ESL_ALLOC(gpu_vit_passed, sizeof(int) * gpu_capacity);
    ESL_ALLOC(gpu_fwd_idx, sizeof(int) * gpu_capacity);
    ESL_ALLOC(gpu_vit_idx, sizeof(int) * gpu_capacity);
    int *gpu_f1_survivor_idx = NULL;
    int  gpu_f1_nsurv = 0;
    ESL_ALLOC(gpu_f1_survivor_idx, sizeof(int) * gpu_capacity);
    gpu_ResolveSurvivorThresholds(info, info->om->M,
                                  &gpu_vit_min_cands, &gpu_fwd_min_cands,
                                  &gpu_vit_collect_cands, &gpu_fwd_collect_cands,
                                  &gpu_vit_min_res, &gpu_fwd_min_res,
                                  &gpu_vit_collect_res, &gpu_fwd_collect_res);
    effective_load_res = (int64_t) 6 * ESL_MAX(eslDSQDATA_CHUNK_MAXPACKET, (info->gpu_load_res + 5) / 6);
    batch_res_cap = ESL_MAX((int64_t) info->gpu_batch_res, effective_load_res * (int64_t) gpu_pending_max_chunks);
    while (n_targetseqs == -1 || seq_cnt < n_targetseqs) {
      gpu_end_of_input = (n_targetseqs != -1 && seq_cnt >= n_targetseqs);
      gpu_search_batch_Reset(&batch);
      batch_res = 0;

      if (dd != NULL) {
        /* dsqdata path: read chunks, overlay gpudb pointers if available */
        while (!gpu_end_of_input &&
               batch.nchunks < gpu_pending_max_chunks &&
               batch_res + effective_load_res <= batch_res_cap) {
          t0 = hmmsearch_WallTime();
          sstatus = esl_dsqdata_Read(dd, &chu);
          {
            double dt = hmmsearch_WallTime() - t0;
            info->pli->time_gpu_read += dt;
            info->pli->exact_io_read_unpack += dt;
          }
          if (sstatus != eslOK) { gpu_end_of_input = TRUE; break; }
          if (info->gpudb) gpu_OverlayGpudb(chu, info->gpudb);
          if (chu->N > gpu_capacity)
            p7_Fail("--gpu-batch-seqs %d is smaller than dsqdata load chunk size %d; use --gpu-batch-seqs >= %d or reduce --gpu-load-seqs\n",
                    gpu_capacity, chu->N, chu->N);
          status = gpu_search_batch_AddChunk(&batch, chu);
          if (status != eslOK) goto ERROR;
          batch_res += effective_load_res;
          chu = NULL;
          if (batch.view.N >= gpu_capacity) break;
        }
      } else {
        /* gpudb-only resident path: assemble batch directly from gpudb mmap */
        int64_t remaining = (n_targetseqs == -1) ? (int64_t) info->gpudb->hdr.nseq - seq_cnt
                                                  : ESL_MIN((int64_t)(n_targetseqs - seq_cnt),
                                                            (int64_t) info->gpudb->hdr.nseq - seq_cnt);
        int batch_n = (int) ESL_MIN(remaining, (int64_t) gpu_capacity);
        if (batch_n <= 0) { gpu_end_of_input = TRUE; break; }
        status = gpu_AssembleBatchFromGpudb(&batch, info->gpudb, (int64_t) seq_cnt, batch_n);
        if (status != eslOK) goto ERROR;
        if ((int64_t) seq_cnt + batch_n >= (int64_t) info->gpudb->hdr.nseq)
          gpu_end_of_input = TRUE;
      }

      if (batch.view.N == 0) break;
      if (info->gpudb) {
        search_chu = &batch.view;
        batch.view.smem = info->gpudb->seq_data + info->gpudb->offsets[batch.view.i0];
      } else {
        search_chu = (batch.nchunks == 1) ? batch.chunks[0] : &batch.view;
      }

      t0 = hmmsearch_WallTime();
      if (p7_cuda_engine_IsResident(info->cuda_engine))
        status = p7_cuda_SSVFilterResident(info->cuda_engine, info->cuda_msv, (int64_t) batch.view.i0, batch.view.N, gpu_scores, gpu_statuses, errbuf, sizeof(errbuf));
      else
        status = p7_cuda_SSVFilterDsqdataChunk(info->cuda_engine, info->cuda_msv, search_chu, gpu_scores, gpu_statuses, errbuf, sizeof(errbuf));
      info->pli->time_msv += hmmsearch_WallTime() - t0;
      if (status != eslOK) p7_Fail("--gpu requested, but CUDA batch MSV/SSV failed: %s\n", errbuf);

      if (info->gpu_ssv_compare) {
        float *msv_scores = (float *) malloc(sizeof(float) * search_chu->N);
        int   *msv_statuses = (int *) malloc(sizeof(int) * search_chu->N);
        if (p7_cuda_engine_IsResident(info->cuda_engine))
          status = p7_cuda_MSVFilterResident(info->cuda_engine, info->cuda_msv, (int64_t) batch.view.i0, batch.view.N, msv_scores, msv_statuses, errbuf, sizeof(errbuf));
        else
          status = p7_cuda_MSVFilterDsqdataChunk(info->cuda_engine, info->cuda_msv, search_chu, msv_scores, msv_statuses, errbuf, sizeof(errbuf));
        if (status != eslOK) p7_Fail("--gpu-ssv-compare: monolithic MSV failed: %s\n", errbuf);
        for (int ci = 0; ci < search_chu->N; ci++) {
          if (gpu_statuses[ci] != msv_statuses[ci] || (gpu_statuses[ci] == eslOK && gpu_scores[ci] != msv_scores[ci]))
            fprintf(stderr, "SSV_COMPARE_MISMATCH seq=%d ssv_score=%.6f msv_score=%.6f ssv_status=%d msv_status=%d\n",
                    ci, gpu_scores[ci], msv_scores[ci], gpu_statuses[ci], msv_statuses[ci]);
        }
        free(msv_scores);
        free(msv_statuses);
      }
      t0 = hmmsearch_WallTime();
      status = p7_cuda_NullScoreDsqdataChunk(info->cuda_engine, info->bg, search_chu, gpu_nullsc, errbuf, sizeof(errbuf));
      info->pli->time_null += hmmsearch_WallTime() - t0;
      if (status != eslOK) p7_Fail("--gpu requested, but CUDA batch null score failed: %s\n", errbuf);
      if (info->pli->do_biasfilter) {
        t0 = hmmsearch_WallTime();
        status = p7_cuda_BiasFilterDsqdataChunk(info->cuda_engine, info->bg, search_chu, gpu_filtersc, errbuf, sizeof(errbuf));
        info->pli->time_bias += hmmsearch_WallTime() - t0;
        if (status != eslOK) p7_Fail("--gpu requested, but CUDA batch bias filter failed: %s\n", errbuf);
      }

      gpu_fwd_n = 0;
      gpu_vit_n = 0;
      gpu_fwd_res = 0;
      gpu_vit_res = 0;
      gpu_fb_n = 0;
      gpu_processed_n = (n_targetseqs == -1) ? search_chu->N : ESL_MIN(search_chu->N, n_targetseqs - seq_cnt);

      t0 = hmmsearch_WallTime();
      status = p7_cuda_F1GatingDsqdataChunk(info->cuda_engine,
                                            gpu_scores, gpu_statuses,
                                            gpu_processed_n, info->pli->do_biasfilter,
                                            info->om->evparam[p7_MMU], info->om->evparam[p7_MLAMBDA], info->pli->F1,
                                            gpu_f1_survivor_idx, &gpu_f1_nsurv,
                                            errbuf, sizeof(errbuf));
      info->pli->exact_host_survivor_orchestration += hmmsearch_WallTime() - t0;
      if (status != eslOK) p7_Fail("--gpu requested, but CUDA F1 gating failed: %s\n", errbuf);

      {
        int64_t batch_nres = 0;
        for (i = 0; i < gpu_processed_n; i++) batch_nres += search_chu->L[i];
        info->pli->nseqs += gpu_processed_n;
        info->pli->nres  += batch_nres;
        if (info->pli->Z_setby == p7_ZSETBY_NTARGETS && info->pli->mode == p7_SEARCH_SEQS)
          info->pli->Z = info->pli->nseqs;
      }

      { double surv_loop_t0 = hmmsearch_WallTime();
      if (info->pli->do_biasfilter && gpu_f1_nsurv > 0) {
        float *surv_filtersc = NULL;
        ESL_ALLOC(surv_filtersc, sizeof(float) * gpu_f1_nsurv);
        status = p7_cuda_BiasFilterSurvivors(info->cuda_engine, info->bg,
                                             gpu_f1_nsurv, surv_filtersc,
                                             errbuf, sizeof(errbuf));
        if (status != eslOK) p7_Fail("CUDA survivor bias filter failed: %s\n", errbuf);
        for (int si = 0; si < gpu_f1_nsurv; si++)
          gpu_filtersc[gpu_f1_survivor_idx[si]] = surv_filtersc[si];
        free(surv_filtersc);
      }

      /* Sort survivors by sequence length to maximize ReconfigLength cache hits */
      {
        const int64_t *surv_L = search_chu->L;
        for (int a = 1; a < gpu_f1_nsurv; a++) {
          int key = gpu_f1_survivor_idx[a];
          int64_t keyL = surv_L[key];
          int b = a - 1;
          while (b >= 0 && surv_L[gpu_f1_survivor_idx[b]] > keyL) {
            gpu_f1_survivor_idx[b+1] = gpu_f1_survivor_idx[b];
            b--;
          }
          gpu_f1_survivor_idx[b+1] = key;
        }
      }

      int last_reconfig_L = -1;
      for (int si = 0; si < gpu_f1_nsurv; si++) {
        i = gpu_f1_survivor_idx[si];
        if (search_chu->L[i] == 0) continue;
        if (search_chu->L[i] > 100000) ESL_EXCEPTION(eslETYPE, "Target sequence length > 100K, over comparison pipeline limit.\n(Did you mean to use nhmmer/nhmmscan?)");

        t0 = hmmsearch_WallTime();
        status = gpu_BindSeqView(info, dbsq, dbsq_dsqmem, dbsq_salloc, search_chu, i);
        info->pli->exact_host_survivor_orchestration += hmmsearch_WallTime() - t0;
        if (status != eslOK) goto ERROR;
        t0 = hmmsearch_WallTime();
        if ((int)dbsq->n != last_reconfig_L) {
          p7_bg_SetLength(info->bg, dbsq->n);
          p7_oprofile_ReconfigLength(info->om, dbsq->n);
          last_reconfig_L = (int)dbsq->n;
        }
        info->pli->exact_host_survivor_orchestration += hmmsearch_WallTime() - t0;
        {
          GPU_PREVIT_RESULT previt;
          status = gpu_PreViterbiBoundary(info, dbsq, gpu_scores[i], gpu_statuses[i], gpu_nullsc[i],
                                          (info->pli->do_biasfilter ? gpu_filtersc[i] : gpu_nullsc[i]), &previt);
          if (status != eslOK) goto ERROR;
          nullsc = previt.nullsc;
          usc = previt.usc;
          if (previt.passed_msv) {
            int passed = FALSE;
            info->pli->n_past_msv++;
            t0 = hmmsearch_WallTime();
            if (previt.passed_bias) {
              info->pli->n_past_bias++;
              if (gpu_vit_active) {
                seq_score = (usc + info->gpu_msv_slack - previt.filtersc) / eslCONST_LOG2;
                P = esl_gumbel_surv(seq_score, info->om->evparam[p7_MMU], info->om->evparam[p7_MLAMBDA]);
                if (P <= info->pli->F2) {
                  info->pli->n_past_vit++;
                  if (gpu_fwd_active) {
                    status = gpu_AppendFwdCandidate(info, gpu_fwd_idx, gpu_fwd_filtersc_subset, &gpu_fwd_n, i, previt.filtersc);
                    if (status != eslOK) goto ERROR;
                    gpu_fwd_res += (int) dbsq->n;
                  } else {
                    float fwdsc;
                    status = gpu_MaterializeSeq(info, dbsq, dbsq_dsqmem, dbsq_salloc, search_chu, i);
                    if (status != eslOK) goto ERROR;
                    p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
                    t0 = hmmsearch_WallTime();
                    p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &fwdsc);
                    {
                      double dt = hmmsearch_WallTime() - t0;
                      info->pli->time_fwd += dt;
                      info->pli->exact_cpu_survivor_total += dt;
                    }
                    if (info->gpu_fb_parser) {
                      status = gpu_fb_batch_Add(info, &gpu_fb_idx, &gpu_fb_nullsc, &gpu_fb_filtersc, &gpu_fb_fwdsc,
                                                &gpu_fb_n, &gpu_fb_alloc, i, nullsc, previt.filtersc, fwdsc);
                      if (status != eslOK) goto ERROR;
                    } else {
                      status = gpu_PostFwd(info, dbsq, nullsc, previt.filtersc, fwdsc, errbuf, sizeof(errbuf));
                      if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
                    }
                  }
                } else {
                  gpu_vit_idx[gpu_vit_n++] = i;
                  gpu_vit_filtersc_subset[gpu_vit_n-1] = previt.filtersc;
                  gpu_vit_res += (int) dbsq->n;
                }
              } else if (gpu_fwd_active) {
                if (info->gpu_vit_compare) {
                  gpu_vit_idx[gpu_vit_n++] = i;
                  gpu_vit_filtersc_subset[gpu_vit_n-1] = previt.filtersc;
                  gpu_vit_res += (int) dbsq->n;
                }
                p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
                status = p7_Pipeline_PostMSVWithFilterPreFwd(info->pli, info->om, info->bg, dbsq, usc + info->gpu_msv_slack, previt.filtersc, &passed);
                if (status != eslOK) goto ERROR;
                if (passed) {
                  status = gpu_AppendFwdCandidate(info, gpu_fwd_idx, gpu_fwd_filtersc_subset, &gpu_fwd_n, i, previt.filtersc);
                  if (status != eslOK) goto ERROR;
                  gpu_fwd_res += (int) dbsq->n;
                }
              } else {
                if (info->gpu_vit_compare) {
                  gpu_vit_idx[gpu_vit_n++] = i;
                  gpu_vit_filtersc_subset[gpu_vit_n-1] = previt.filtersc;
                  gpu_vit_res += (int) dbsq->n;
                }
                if (info->gpu_fb_parser) {
                  float fwdsc;
                  p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
                  status = p7_Pipeline_PostMSVWithFilterPreFwd(info->pli, info->om, info->bg, dbsq, usc + info->gpu_msv_slack, previt.filtersc, &passed);
                  if (status != eslOK) goto ERROR;
                  if (passed) {
                    status = gpu_MaterializeSeq(info, dbsq, dbsq_dsqmem, dbsq_salloc, search_chu, i);
                    if (status != eslOK) goto ERROR;
                    t0 = hmmsearch_WallTime();
                    p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &fwdsc);
                    {
                      double dt = hmmsearch_WallTime() - t0;
                      info->pli->time_fwd += dt;
                      info->pli->exact_cpu_survivor_total += dt;
                    }
                    if (info->gpu_fb_parser) {
                      status = gpu_fb_batch_Add(info, &gpu_fb_idx, &gpu_fb_nullsc, &gpu_fb_filtersc, &gpu_fb_fwdsc,
                                                &gpu_fb_n, &gpu_fb_alloc, i, nullsc, previt.filtersc, fwdsc);
                      if (status != eslOK) goto ERROR;
                    } else {
                      status = gpu_PostFwd(info, dbsq, nullsc, previt.filtersc, fwdsc, errbuf, sizeof(errbuf));
                      if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
                    }
                  }
                } else {
                  status = gpu_MaterializeSeq(info, dbsq, dbsq_dsqmem, dbsq_salloc, search_chu, i);
                  if (status != eslOK) goto ERROR;
                  p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
                  p7_Pipeline_PostMSVWithFilter(info->pli, info->om, info->bg, dbsq, NULL, info->th, nullsc, usc + info->gpu_msv_slack, previt.filtersc);
                }
              }
            }
            info->pli->time_gpu_survivor += hmmsearch_WallTime() - t0;
          }
        }
        p7_pipeline_Reuse(info->pli);
        gpu_RestoreSeqStorage(dbsq, dbsq_dsqmem, dbsq_salloc);
      }
      { double surv_loop_wall = hmmsearch_WallTime() - surv_loop_t0;
        double surv_loop_known = 0.0;
        info->pli->exact_survivor_loop_other += surv_loop_wall;
      } }

      { double vit_fwd_t0 = hmmsearch_WallTime();
      if ((info->gpu_vit_compare || gpu_vit_active) && gpu_vit_n > 0) {
        int run_cuda_vit;
        int enough_min;
        int enough_collect;
        int vit_launch_n;
        int vit_launch_res;
        int vit_res_left = gpu_vit_res;
        int vit_off = 0;
        gpu_vit_res = gpu_CandidateResidues(search_chu, gpu_vit_idx, gpu_vit_n);
        info->pli->gpu_vit_candidates_total += (uint64_t) gpu_vit_n;
        info->pli->gpu_vit_residues_total += (uint64_t) gpu_vit_res;
        enough_min = (gpu_vit_n >= gpu_vit_min_cands || gpu_vit_res >= gpu_vit_min_res);
        enough_collect = (gpu_vit_n >= gpu_vit_collect_cands || gpu_vit_res >= gpu_vit_collect_res);
        run_cuda_vit = info->gpu_vit_compare || (enough_min && enough_collect);
        if (gpu_strict_mode && gpu_vit_active) run_cuda_vit = TRUE;
        if (run_cuda_vit) {
          while (vit_off < gpu_vit_n) {
            int t;
            vit_launch_n = gpu_ChooseTileCandidates(info, FALSE, gpu_vit_n - vit_off, vit_res_left);
            if (vit_launch_n < 1) vit_launch_n = gpu_vit_n - vit_off;
            vit_launch_res = 0;
            for (t = 0; t < vit_launch_n; t++) vit_launch_res += (int) search_chu->L[gpu_vit_idx[vit_off + t]];
            info->pli->gpu_vit_launches++;
            t0 = hmmsearch_WallTime();
            status = p7_cuda_ViterbiFilterDsqdataSubset(info->cuda_engine, info->cuda_msv, search_chu,
                                                        gpu_vit_idx + vit_off, vit_launch_n, gpu_vit_filtersc_subset + vit_off,
                                                        info->om->evparam[p7_VMU], info->om->evparam[p7_VLAMBDA], info->pli->F2,
                                                        info->gpu_vit_compare ? (gpu_vit_scores + vit_off) : NULL,
                                                        gpu_vit_statuses + vit_off, gpu_vit_passed + vit_off,
                                                        errbuf, sizeof(errbuf));
            info->pli->time_vit += hmmsearch_WallTime() - t0;
            if (status != eslOK) p7_Fail("--gpu requested, but CUDA Viterbi failed: %s\n", errbuf);
            vit_off += vit_launch_n;
            vit_res_left -= vit_launch_res;
          }
        }

        { int last_vit_reconfig_L = -1;
        for (int j = 0; j < gpu_vit_n; j++) {
          float cpu_vitsc = -eslINFINITY;
          float filtersc;
          int cpu_status = eslOK;
          int cpu_pass = FALSE;
          int gpu_pass;
          i = gpu_vit_idx[j];
          t0 = hmmsearch_WallTime();
          status = gpu_BindSeqView(info, dbsq, dbsq_dsqmem, dbsq_salloc, search_chu, i);
          info->pli->exact_host_survivor_orchestration += hmmsearch_WallTime() - t0;
          if (status != eslOK) goto ERROR;
          t0 = hmmsearch_WallTime();
          if ((int)dbsq->n != last_vit_reconfig_L) {
            p7_bg_SetLength(info->bg, dbsq->n);
            p7_oprofile_ReconfigLength(info->om, dbsq->n);
            last_vit_reconfig_L = (int)dbsq->n;
          }
          info->pli->exact_host_survivor_orchestration += hmmsearch_WallTime() - t0;
          filtersc = gpu_vit_filtersc_subset[j];
          if (run_cuda_vit) {
            gpu_pass = gpu_vit_passed[j];
            if (info->gpu_vit_compare) {
              gpu_pass = (gpu_vit_statuses[j] == eslERANGE);
              if (gpu_vit_statuses[j] == eslOK) {
                seq_score = (gpu_vit_scores[j] - filtersc) / eslCONST_LOG2;
                gpu_pass = (esl_gumbel_surv(seq_score, info->om->evparam[p7_VMU], info->om->evparam[p7_VLAMBDA]) <= info->pli->F2);
              }
            }
          } else {
            if (gpu_strict_mode && gpu_vit_active) {
              p7_Fail("--gpu requested strict CUDA Viterbi, but launch gating skipped CUDA path for %d candidates/%d residues; adjust --gpu-vit-min-* or --gpu-vit-collect-* thresholds\n",
                      gpu_vit_n, gpu_vit_res);
            }
            p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
            t0 = hmmsearch_WallTime();
            cpu_status = p7_ViterbiFilter(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &cpu_vitsc);
            {
              double dt = hmmsearch_WallTime() - t0;
              info->pli->time_vit += dt;
              info->pli->exact_cpu_survivor_total += dt;
            }
            gpu_pass = (cpu_status == eslERANGE);
            if (cpu_status == eslOK) {
              seq_score = (cpu_vitsc - filtersc) / eslCONST_LOG2;
              gpu_pass = (esl_gumbel_surv(seq_score, info->om->evparam[p7_VMU], info->om->evparam[p7_VLAMBDA]) <= info->pli->F2);
            }
          }
          if (info->gpu_vit_compare) {
            int status_mismatch = 0;
            int pass_mismatch = 0;
            int score_drift = 0;
            status = gpu_MaterializeSeq(info, dbsq, dbsq_dsqmem, dbsq_salloc, search_chu, i);
            if (status != eslOK) goto ERROR;
            p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
            cpu_status = p7_ViterbiFilter(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &cpu_vitsc);
            cpu_pass = (cpu_status == eslERANGE);
            if (cpu_status == eslOK) {
              seq_score = (cpu_vitsc - filtersc) / eslCONST_LOG2;
              cpu_pass = (esl_gumbel_surv(seq_score, info->om->evparam[p7_VMU], info->om->evparam[p7_VLAMBDA]) <= info->pli->F2);
            }
            status_mismatch = (cpu_status != gpu_vit_statuses[j]);
            pass_mismatch = (cpu_pass != gpu_pass);
            score_drift = (cpu_status == eslOK && fabsf(cpu_vitsc - gpu_vit_scores[j]) > 1e-3f);
            if (status_mismatch) vit_cmp_status_mismatch++;
            if (pass_mismatch) vit_cmp_pass_mismatch++;
            if (score_drift) vit_cmp_score_drift++;
            if (status_mismatch || pass_mismatch || score_drift) {
              fprintf(stderr, "CUDAVIT\t%s\tL=%ld\tcpu_status=%d\tgpu_status=%d\tcpu=%.6f\tgpu=%.6f\tdiff=%.6f\tcpu_pass=%d\tgpu_pass=%d\n",
                      dbsq->name ? dbsq->name : "-", (long) dbsq->n, cpu_status, gpu_vit_statuses[j],
                      cpu_vitsc, gpu_vit_scores[j], gpu_vit_scores[j] - cpu_vitsc, cpu_pass, gpu_pass);
            }
          }
          if (gpu_vit_active && gpu_pass) {
            info->pli->n_past_vit++;
            if (gpu_fwd_active) {
              status = gpu_AppendFwdCandidate(info, gpu_fwd_idx, gpu_fwd_filtersc_subset, &gpu_fwd_n, i, filtersc);
              if (status != eslOK) goto ERROR;
              gpu_fwd_res += (int) dbsq->n;
            } else {
              float fwdsc;
              status = gpu_MaterializeSeq(info, dbsq, dbsq_dsqmem, dbsq_salloc, search_chu, i);
              if (status != eslOK) goto ERROR;
              p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
              t0 = hmmsearch_WallTime();
              p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &fwdsc);
              {
                double dt = hmmsearch_WallTime() - t0;
                info->pli->time_fwd += dt;
                info->pli->exact_cpu_survivor_total += dt;
              }
              if (info->gpu_fb_parser) {
                status = gpu_fb_batch_Add(info, &gpu_fb_idx, &gpu_fb_nullsc, &gpu_fb_filtersc, &gpu_fb_fwdsc,
                                          &gpu_fb_n, &gpu_fb_alloc, i, gpu_nullsc[i], filtersc, fwdsc);
                if (status != eslOK) goto ERROR;
              } else {
                status = gpu_PostFwd(info, dbsq, gpu_nullsc[i], filtersc, fwdsc, errbuf, sizeof(errbuf));
                if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
              }
            }
          }
          p7_pipeline_Reuse(info->pli);
          gpu_RestoreSeqStorage(dbsq, dbsq_dsqmem, dbsq_salloc);
        }
        } /* end last_vit_reconfig_L scope */
      }

      if (gpu_fwd_active && gpu_fwd_n > 0) {
        int run_cuda_fwd;
        int enough_min;
        int enough_collect;
        int fwd_launch_n;
        int fwd_launch_res;
        int fwd_res_left = gpu_fwd_res;
        int fwd_off = 0;
        float *fwd_score_ret = (info->gpu_fwd_compare || !info->gpu_fb_parser) ? gpu_fwd_scores : NULL;
        gpu_fwd_res = gpu_CandidateResidues(search_chu, gpu_fwd_idx, gpu_fwd_n);
        info->pli->gpu_fwd_candidates_total += (uint64_t) gpu_fwd_n;
        info->pli->gpu_fwd_residues_total += (uint64_t) gpu_fwd_res;
        enough_min = (gpu_fwd_n >= gpu_fwd_min_cands || gpu_fwd_res >= gpu_fwd_min_res);
        enough_collect = (gpu_fwd_n >= gpu_fwd_collect_cands || gpu_fwd_res >= gpu_fwd_collect_res);
        run_cuda_fwd = info->gpu_fwd_compare || (enough_min && enough_collect);
        if (gpu_strict_mode && gpu_fwd_active) run_cuda_fwd = TRUE;
        if (run_cuda_fwd) {
          while (fwd_off < gpu_fwd_n) {
            int t;
            fwd_launch_n = gpu_ChooseTileCandidates(info, TRUE, gpu_fwd_n - fwd_off, fwd_res_left);
            if (fwd_launch_n < 1) fwd_launch_n = gpu_fwd_n - fwd_off;
            fwd_launch_res = 0;
            for (t = 0; t < fwd_launch_n; t++) fwd_launch_res += (int) search_chu->L[gpu_fwd_idx[fwd_off + t]];
            info->pli->gpu_fwd_launches++;
            t0 = hmmsearch_WallTime();
            status = p7_cuda_ForwardFilterDsqdataSubset(info->cuda_engine, info->cuda_msv, search_chu,
                                                        gpu_fwd_idx + fwd_off, fwd_launch_n, gpu_fwd_filtersc_subset + fwd_off,
                                                        info->om->evparam[p7_FTAU], info->om->evparam[p7_FLAMBDA], info->pli->F3,
                                                        fwd_score_ret ? (fwd_score_ret + fwd_off) : NULL,
                                                        gpu_fwd_statuses + fwd_off, gpu_fwd_passed + fwd_off,
                                                        errbuf, sizeof(errbuf));
            info->pli->time_fwd += hmmsearch_WallTime() - t0;
            if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward prefilter failed: %s\n", errbuf);
            fwd_off += fwd_launch_n;
            fwd_res_left -= fwd_launch_res;
          }
        }

        { int last_fwd_reconfig_L = -1;
        for (int j = 0; j < gpu_fwd_n; j++) {
          i = gpu_fwd_idx[j];
          t0 = hmmsearch_WallTime();
          status = gpu_BindSeqView(info, dbsq, dbsq_dsqmem, dbsq_salloc, search_chu, i);
          info->pli->exact_host_survivor_orchestration += hmmsearch_WallTime() - t0;
          if (status != eslOK) goto ERROR;
          t0 = hmmsearch_WallTime();
          if ((int)dbsq->n != last_fwd_reconfig_L) {
            p7_bg_SetLength(info->bg, dbsq->n);
            p7_oprofile_ReconfigLength(info->om, dbsq->n);
            last_fwd_reconfig_L = (int)dbsq->n;
          }
          info->pli->exact_host_survivor_orchestration += hmmsearch_WallTime() - t0;
          nullsc = gpu_nullsc[i];

          if (!run_cuda_fwd) {
            if (gpu_strict_mode && gpu_fwd_active) {
              p7_Fail("--gpu requested strict CUDA Forward, but launch gating skipped CUDA path for %d candidates/%d residues; adjust --gpu-fwd-min-* or --gpu-fwd-collect-* thresholds\n",
                      gpu_fwd_n, gpu_fwd_res);
            }
            float fwdsc;
            float filtersc = gpu_fwd_filtersc_subset[j];
            status = gpu_MaterializeSeq(info, dbsq, dbsq_dsqmem, dbsq_salloc, search_chu, i);
            if (status != eslOK) goto ERROR;
            p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
            t0 = hmmsearch_WallTime();
            p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &fwdsc);
            {
              double dt = hmmsearch_WallTime() - t0;
              info->pli->time_fwd += dt;
              info->pli->exact_cpu_survivor_total += dt;
            }
            if (info->gpu_fb_parser) {
                status = gpu_fb_batch_Add(info, &gpu_fb_idx, &gpu_fb_nullsc, &gpu_fb_filtersc, &gpu_fb_fwdsc,
                                          &gpu_fb_n, &gpu_fb_alloc, i, nullsc, filtersc, fwdsc);
                if (status != eslOK) goto ERROR;
              } else {
                status = gpu_PostFwd(info, dbsq, nullsc, filtersc, fwdsc, errbuf, sizeof(errbuf));
                if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
              }
            p7_pipeline_Reuse(info->pli);
            gpu_RestoreSeqStorage(dbsq, dbsq_dsqmem, dbsq_salloc);
            continue;
          }

          if (info->gpu_fwd_compare) {
            float cpu_fwdsc;
            float filtersc = gpu_fwd_filtersc_subset[j];
            double gpuP = 1.0;
            double cpuP = 1.0;
            int gpu_pass = gpu_fwd_passed[j];
            int cpu_pass = FALSE;
            status = gpu_MaterializeSeq(info, dbsq, dbsq_dsqmem, dbsq_salloc, search_chu, i);
            if (status != eslOK) goto ERROR;
            p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
            t0 = hmmsearch_WallTime();
            status = p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &cpu_fwdsc);
            {
              double dt = hmmsearch_WallTime() - t0;
              info->pli->time_fwd += dt;
              info->pli->exact_cpu_survivor_total += dt;
            }
            if (status == eslOK) {
              seq_score = (cpu_fwdsc - filtersc) / eslCONST_LOG2;
              cpuP = esl_exp_surv(seq_score, info->om->evparam[p7_FTAU], info->om->evparam[p7_FLAMBDA]);
              cpu_pass = (cpuP <= info->pli->F3);
            }
            if (gpu_fwd_statuses[j] == eslOK) {
              seq_score = (gpu_fwd_scores[j] - filtersc) / eslCONST_LOG2;
              gpuP = esl_exp_surv(seq_score, info->om->evparam[p7_FTAU], info->om->evparam[p7_FLAMBDA]);
              gpu_pass = (gpuP <= info->pli->F3);
            } else if (gpu_fwd_statuses[j] == eslERANGE) {
              gpu_pass = TRUE;
            }
            if (cpu_pass != gpu_pass || (status == eslOK && gpu_fwd_statuses[j] == eslOK && fabsf(cpu_fwdsc - gpu_fwd_scores[j]) > 1e-3f)) {
              fprintf(stderr, "CUDAFWD\t%s\tL=%ld\tcpu_status=%d\tgpu_status=%d\tcpu=%.6f\tgpu=%.6f\tdiff=%.6f\tcpuP=%.6g\tgpuP=%.6g\tcpu_pass=%d\tgpu_pass=%d\n",
                      dbsq->name ? dbsq->name : "-", (long) dbsq->n, status, gpu_fwd_statuses[j],
                      cpu_fwdsc, gpu_fwd_scores[j], gpu_fwd_scores[j] - cpu_fwdsc, cpuP, gpuP, cpu_pass, gpu_pass);
            }
            if (status != eslOK) {
              p7_pipeline_Reuse(info->pli);
              gpu_RestoreSeqStorage(dbsq, dbsq_dsqmem, dbsq_salloc);
              continue;
            }
            if (cpu_pass) {
              status = gpu_PostFwd(info, dbsq, nullsc, filtersc, cpu_fwdsc, errbuf, sizeof(errbuf));
              if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
            }
            p7_pipeline_Reuse(info->pli);
            gpu_RestoreSeqStorage(dbsq, dbsq_dsqmem, dbsq_salloc);
            continue;
          }

          if (gpu_fwd_statuses[j] == eslOK) {
            float filtersc = gpu_fwd_filtersc_subset[j];
            if (gpu_fwd_passed[j]) {
              if (info->gpu_fb_parser) {
                status = gpu_fb_batch_Append(&gpu_fb_idx, &gpu_fb_nullsc, &gpu_fb_filtersc, &gpu_fb_fwdsc,
                                             &gpu_fb_n, &gpu_fb_alloc, i, nullsc, filtersc, 0.0f);
                if (status != eslOK) goto ERROR;
              } else {
                status = gpu_MaterializeSeq(info, dbsq, dbsq_dsqmem, dbsq_salloc, search_chu, i);
                if (status != eslOK) goto ERROR;
                status = gpu_PostFwd(info, dbsq, nullsc, filtersc, gpu_fwd_scores[j], errbuf, sizeof(errbuf));
                if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
              }
            }
          } else if (gpu_fwd_statuses[j] == eslERANGE) {
            float filtersc = gpu_fwd_filtersc_subset[j];
            float fwdsc;
            /* Keep F3 gating GPU-led in normal mode; only recover a finite score for downstream parser/post processing. */
            status = gpu_MaterializeSeq(info, dbsq, dbsq_dsqmem, dbsq_salloc, search_chu, i);
            if (status != eslOK) goto ERROR;
            p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
            t0 = hmmsearch_WallTime();
            p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &fwdsc);
            {
              double dt = hmmsearch_WallTime() - t0;
              info->pli->time_fwd += dt;
              info->pli->exact_cpu_survivor_total += dt;
            }
            if (info->gpu_fb_parser) {
              status = gpu_fb_batch_Append(&gpu_fb_idx, &gpu_fb_nullsc, &gpu_fb_filtersc, &gpu_fb_fwdsc,
                                           &gpu_fb_n, &gpu_fb_alloc, i, nullsc, filtersc, fwdsc);
              if (status != eslOK) goto ERROR;
            } else {
              status = gpu_PostFwd(info, dbsq, nullsc, filtersc, fwdsc, errbuf, sizeof(errbuf));
              if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
            }
          }
          p7_pipeline_Reuse(info->pli);
          gpu_RestoreSeqStorage(dbsq, dbsq_dsqmem, dbsq_salloc);
        }
        } /* end last_fwd_reconfig_L scope */
      }
      if (gpu_fb_n > 0) {
        status = gpu_ProcessFbBatch(info, search_chu, dbsq, dbsq_dsqmem, dbsq_salloc,
                                    gpu_fb_idx, gpu_fb_nullsc, gpu_fb_filtersc, gpu_fb_fwdsc, gpu_fb_n,
                                    errbuf, sizeof(errbuf));
        if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser batch failed: %s\n", errbuf);
        gpu_fb_n = 0;
      }
      info->pli->exact_vit_fwd_dispatch += hmmsearch_WallTime() - vit_fwd_t0;
      }
      seq_cnt += gpu_processed_n;
      dbsq->dsq = NULL;
      if (dd != NULL) {
        for (i = 0; i < batch.nchunks; i++)
          esl_dsqdata_Recycle(dd, batch.chunks[i]);
      }
      gpu_search_batch_Reset(&batch);
      search_chu = NULL;
    }
    if (n_targetseqs!=-1 && seq_cnt==n_targetseqs) sstatus = eslEOF;
    if (dd == NULL) sstatus = eslEOF;
    if (dd != NULL) {
      if (chu) esl_dsqdata_Recycle(dd, chu);
      for (i = 0; i < batch.nchunks; i++)
        esl_dsqdata_Recycle(dd, batch.chunks[i]);
    }
    gpu_search_batch_Destroy(&batch);
    dbsq->dsq    = dbsq_dsqmem;
    dbsq->salloc = dbsq_salloc;
    esl_sq_Destroy(dbsq);
    free(gpu_scores);
    free(gpu_filtersc);
    free(gpu_fwd_scores);
    free(gpu_vit_scores);
    free(gpu_fwd_filtersc_subset);
    free(gpu_vit_filtersc_subset);
    free(gpu_nullsc);
    free(gpu_statuses);
    free(gpu_fwd_statuses);
    free(gpu_vit_statuses);
    free(gpu_fwd_passed);
    free(gpu_vit_passed);
    free(gpu_fwd_idx);
    free(gpu_vit_idx);
    free(gpu_f1_survivor_idx);
    free(gpu_fb_idx);
    free(gpu_fb_nullsc);
    free(gpu_fb_filtersc);
    free(gpu_fb_fwdsc);
    if (info->gpu_vit_compare) {
      fprintf(stderr, "CUDAVIT_SUMMARY\tstatus_mismatch=%d\tpass_mismatch=%d\tscore_drift=%d\n",
              vit_cmp_status_mismatch, vit_cmp_pass_mismatch, vit_cmp_score_drift);
    }
  return sstatus;

ERROR:
  if (dd != NULL) {
    if (chu) esl_dsqdata_Recycle(dd, chu);
    for (i = 0; i < batch.nchunks; i++)
      esl_dsqdata_Recycle(dd, batch.chunks[i]);
  }
  gpu_search_batch_Destroy(&batch);
  if (dbsq && dbsq_dsqmem) {
    dbsq->dsq    = dbsq_dsqmem;
    dbsq->salloc = dbsq_salloc;
  }
  if (dbsq) esl_sq_Destroy(dbsq);
  free(gpu_scores);
  free(gpu_filtersc);
  free(gpu_fwd_scores);
  free(gpu_vit_scores);
  free(gpu_fwd_filtersc_subset);
  free(gpu_vit_filtersc_subset);
  free(gpu_nullsc);
  free(gpu_statuses);
  free(gpu_fwd_statuses);
  free(gpu_vit_statuses);
  free(gpu_fwd_passed);
  free(gpu_vit_passed);
  free(gpu_fwd_idx);
  free(gpu_vit_idx);
  free(gpu_f1_survivor_idx);
  free(gpu_fb_idx);
  free(gpu_fb_nullsc);
  free(gpu_fb_filtersc);
  free(gpu_fb_fwdsc);
  return status;
}
