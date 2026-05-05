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

static int gpu_search_batch_Init(GPU_SEARCH_BATCH *batch);
static void gpu_search_batch_Reset(GPU_SEARCH_BATCH *batch);
static void gpu_search_batch_Destroy(GPU_SEARCH_BATCH *batch);
static int gpu_search_batch_AddChunk(GPU_SEARCH_BATCH *batch, ESL_DSQDATA_CHUNK *chu);
static int gpu_PreViterbiBoundary(WORKER_INFO *info, const ESL_SQ *dbsq, float gpu_usc, int gpu_msv_status,
                                  float gpu_nullsc, float gpu_filtersc, GPU_PREVIT_RESULT *ret);
static int gpu_PostFwd(WORKER_INFO *info, const ESL_SQ *dbsq, float nullsc, float filtersc, float fwdsc, char *errbuf, int errbuf_size);
static int gpu_fb_batch_Add(WORKER_INFO *info, int **ret_idx, float **ret_nullsc, float **ret_filtersc, float **ret_fwdsc,
                            int *ret_n, int *ret_alloc, int idx, float nullsc, float filtersc, float fwdsc);
static int gpu_ProcessFbBatch(WORKER_INFO *info, ESL_DSQDATA_CHUNK *chu, ESL_SQ *dbsq, ESL_DSQ *dbsq_dsqmem,
                              int64_t dbsq_salloc, int *idx, float *nullsc, float *filtersc, float *fwdsc, int n,
                              char *errbuf, int errbuf_size);
static void gpu_CompareParserState(WORKER_INFO *info, const ESL_SQ *dbsq, const P7_OMX *cpu_oxf, const P7_OMX *cpu_oxb,
                                   float cpu_fwdsc, float gpu_fwdsc, float gpu_bcksc);

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

static int
gpu_fb_batch_Add(WORKER_INFO *info, int **ret_idx, float **ret_nullsc, float **ret_filtersc, float **ret_fwdsc,
                 int *ret_n, int *ret_alloc, int idx, float nullsc, float filtersc, float fwdsc)
{
  int n;
  int alloc;
  float seq_score;
  double P;

  if (!ret_idx || !ret_nullsc || !ret_filtersc || !ret_fwdsc || !ret_n || !ret_alloc) return eslEINVAL;
  if (!info) return eslEINVAL;
  seq_score = (fwdsc - filtersc) / eslCONST_LOG2;
  P = esl_exp_surv(seq_score, info->om->evparam[p7_FTAU], info->om->evparam[p7_FLAMBDA]);
  if (P > info->pli->F3) return eslOK;

  n = *ret_n;
  alloc = *ret_alloc;
  if (n >= alloc) {
    int new_alloc = alloc > 0 ? alloc * 2 : 16;
    int *new_idx = malloc(sizeof(int) * new_alloc);
    float *new_nullsc = malloc(sizeof(float) * new_alloc);
    float *new_filtersc = malloc(sizeof(float) * new_alloc);
    float *new_fwdsc = malloc(sizeof(float) * new_alloc);
    if (!new_idx || !new_nullsc || !new_filtersc || !new_fwdsc) {
      free(new_idx);
      free(new_nullsc);
      free(new_filtersc);
      free(new_fwdsc);
      return eslEMEM;
    }
    if (n > 0) {
      memcpy(new_idx, *ret_idx, sizeof(int) * n);
      memcpy(new_nullsc, *ret_nullsc, sizeof(float) * n);
      memcpy(new_filtersc, *ret_filtersc, sizeof(float) * n);
      memcpy(new_fwdsc, *ret_fwdsc, sizeof(float) * n);
    }
    free(*ret_idx);
    free(*ret_nullsc);
    free(*ret_filtersc);
    free(*ret_fwdsc);
    *ret_idx = new_idx;
    *ret_nullsc = new_nullsc;
    *ret_filtersc = new_filtersc;
    *ret_fwdsc = new_fwdsc;
    *ret_alloc = new_alloc;
  }
  (*ret_idx)[n] = idx;
  (*ret_nullsc)[n] = nullsc;
  (*ret_filtersc)[n] = filtersc;
  (*ret_fwdsc)[n] = fwdsc;
  *ret_n = n + 1;
  return eslOK;
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
      double tbias0 = hmmsearch_WallTime();
      p7_bg_FilterScore(info->bg, dbsq->dsq, dbsq->n, &filtersc);
      info->pli->time_bias += hmmsearch_WallTime() - tbias0;
      seq_score = (usc + info->gpu_msv_slack - filtersc) / eslCONST_LOG2;
      P = esl_gumbel_surv(seq_score, info->om->evparam[p7_MMU], info->om->evparam[p7_MLAMBDA]);
      if (P > info->pli->F1 && gpu_msv_status == eslOK) {
        float cpu_usc = -eslINFINITY;
        int cpu_msv_status;
        double tmsv0 = hmmsearch_WallTime();
        cpu_msv_status = p7_MSVFilter(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &cpu_usc);
        info->pli->time_msv += hmmsearch_WallTime() - tmsv0;
        if (cpu_msv_status == eslERANGE) {
          usc = cpu_usc;
          P = 0.0;
        } else if (cpu_msv_status == eslOK) {
          seq_score = (cpu_usc - filtersc) / eslCONST_LOG2;
          P = esl_gumbel_surv(seq_score, info->om->evparam[p7_MMU], info->om->evparam[p7_MLAMBDA]);
          if (P <= info->pli->F1) usc = cpu_usc;
        }
      }
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
  int status;

  if (!info || !dbsq) return eslEINVAL;
  if (!info->gpu_fb_parser) {
    return p7_Pipeline_PostFwd(info->pli, info->om, info->bg, dbsq, NULL, info->th, nullsc, filtersc, fwdsc);
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
  status = p7_Pipeline_PostFwdWithParserMatrices(info->pli, info->om, info->bg, dbsq, NULL, info->th, nullsc, filtersc, gpu_fwdsc);

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
    P7_OMX *cpu_oxf = NULL;
    P7_OMX *cpu_oxb = NULL;

    dbsq->dsq    = dbsq_dsqmem;
    dbsq->salloc = dbsq_salloc;
    esl_sq_Reuse(dbsq);
    status = esl_sq_SetName(dbsq, chu->name[i]);
    if (status == eslOK) status = esl_sq_SetAccession(dbsq, chu->acc[i]);
    if (status == eslOK) status = esl_sq_SetDesc(dbsq, chu->desc[i]);
    if (status == eslOK) status = esl_sq_SetCoordComplete(dbsq, chu->L[i]);
    if (status != eslOK) goto ERROR;
    dbsq->tax_id = chu->taxid[i];
    dbsq->dsq    = chu->dsq[i];

    p7_bg_SetLength(info->bg, dbsq->n);
    p7_oprofile_ReconfigLength(info->om, dbsq->n);
    if ((status = p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n)) != eslOK) goto ERROR;
    if ((status = p7_omx_GrowTo(info->pli->oxb, info->om->M, 0, dbsq->n)) != eslOK) goto ERROR;
    memcpy(info->pli->oxf->xmx, xf + x_offsets[j], sizeof(float) * (size_t) (dbsq->n + 1) * p7X_NXCELLS);
    memcpy(info->pli->oxb->xmx, xb + x_offsets[j], sizeof(float) * (size_t) (dbsq->n + 1) * p7X_NXCELLS);
    info->pli->oxf->M = info->om->M;
    info->pli->oxf->L = dbsq->n;
    info->pli->oxf->has_own_scales = TRUE;
    info->pli->oxf->totscale = scores[j * 2 + 0];
    info->pli->oxb->M = info->om->M;
    info->pli->oxb->L = dbsq->n;
    info->pli->oxb->has_own_scales = FALSE;
    info->pli->oxb->totscale = scores[j * 2 + 1];

    if (statuses[j * 2 + 0] != eslOK) { status = statuses[j * 2 + 0]; goto ERROR; }
    if (statuses[j * 2 + 1] != eslOK) { status = statuses[j * 2 + 1]; goto ERROR; }

    if (info->gpu_fb_compare) {
      cpu_oxf = p7_omx_Create(info->om->M, 0, dbsq->n);
      cpu_oxb = p7_omx_Create(info->om->M, 0, dbsq->n);
      if (!cpu_oxf || !cpu_oxb) { status = eslEMEM; p7_omx_Destroy(cpu_oxf); p7_omx_Destroy(cpu_oxb); goto ERROR; }
      p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, cpu_oxf, &fwdsc[j]);
      p7_BackwardParser(dbsq->dsq, dbsq->n, info->om, cpu_oxf, cpu_oxb, NULL);
      gpu_CompareParserState(info, dbsq, cpu_oxf, cpu_oxb, fwdsc[j], scores[j * 2 + 0], scores[j * 2 + 1]);
      p7_omx_Destroy(cpu_oxf);
      p7_omx_Destroy(cpu_oxb);
    }

    status = p7_Pipeline_PostFwdWithParserMatrices(info->pli, info->om, info->bg, dbsq, NULL, info->th, nullsc[j], filtersc[j], scores[j * 2 + 0]);
    if (status != eslOK) goto ERROR;
    p7_pipeline_Reuse(info->pli);
    dbsq->dsq = NULL;
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
  float *gpu_nullsc = NULL;
  int *gpu_statuses = NULL;
  int *gpu_fwd_statuses = NULL;
  int *gpu_vit_statuses = NULL;
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
  ESL_DSQ *dbsq_dsqmem = NULL;
  int64_t  dbsq_salloc = 0;
  int      gpu_capacity = info->gpu_batch_seqs > 0 ? info->gpu_batch_seqs : eslDSQDATA_CHUNK_MAXSEQ;
  int64_t  batch_res = 0;
  int64_t  effective_load_res = 0;
  double   t0;
  int      gpu_vit_active = info->gpu_vit_prefilter && info->om->M <= 512;
  int      gpu_fwd_active = info->gpu_fwd_prefilter && info->om->M <= 256;
  int      vit_cmp_status_mismatch = 0;
  int      vit_cmp_pass_mismatch = 0;
  int      vit_cmp_score_drift = 0;

  gpu_search_batch_Init(&batch);
  if (dd == NULL) return eslEINVAL;
    dbsq = esl_sq_CreateDigital(info->om->abc);
    if (dbsq == NULL) { status = eslEMEM; goto ERROR; }
    dbsq_dsqmem = dbsq->dsq;
    dbsq_salloc = dbsq->salloc;
    ESL_ALLOC(gpu_scores, sizeof(float) * gpu_capacity);
    ESL_ALLOC(gpu_filtersc, sizeof(float) * gpu_capacity);
    ESL_ALLOC(gpu_fwd_scores, sizeof(float) * gpu_capacity);
    ESL_ALLOC(gpu_vit_scores, sizeof(float) * gpu_capacity);
    ESL_ALLOC(gpu_nullsc, sizeof(float) * gpu_capacity);
    ESL_ALLOC(gpu_statuses, sizeof(int) * gpu_capacity);
    ESL_ALLOC(gpu_fwd_statuses, sizeof(int) * gpu_capacity);
    ESL_ALLOC(gpu_vit_statuses, sizeof(int) * gpu_capacity);
    ESL_ALLOC(gpu_fwd_idx, sizeof(int) * gpu_capacity);
    ESL_ALLOC(gpu_vit_idx, sizeof(int) * gpu_capacity);
    effective_load_res = (int64_t) 6 * ESL_MAX(eslDSQDATA_CHUNK_MAXPACKET, (info->gpu_load_res + 5) / 6);
    while (n_targetseqs == -1 || seq_cnt < n_targetseqs) {
      gpu_search_batch_Reset(&batch);
      batch_res = 0;

      do {
        t0 = hmmsearch_WallTime();
        sstatus = esl_dsqdata_Read(dd, &chu);
        info->pli->time_gpu_read += hmmsearch_WallTime() - t0;
        if (sstatus != eslOK) break;
        if (chu->N > gpu_capacity)
          p7_Fail("--gpu-batch-seqs %d is smaller than dsqdata load chunk size %d; use --gpu-batch-seqs >= %d or reduce --gpu-load-seqs\n",
                  gpu_capacity, chu->N, chu->N);
        status = gpu_search_batch_AddChunk(&batch, chu);
        if (status != eslOK) goto ERROR;
        batch_res += effective_load_res;
        chu = NULL;
      } while ((n_targetseqs == -1 || seq_cnt + batch.view.N < n_targetseqs) &&
               batch.view.N + info->gpu_load_seqs <= info->gpu_batch_seqs &&
               batch_res + effective_load_res <= info->gpu_batch_res);

      if (batch.view.N == 0) break;
      search_chu = (batch.nchunks == 1) ? batch.chunks[0] : &batch.view;

      t0 = hmmsearch_WallTime();
      status = p7_cuda_MSVFilterDsqdataChunk(info->cuda_engine, info->cuda_msv, search_chu, gpu_scores, gpu_statuses, errbuf, sizeof(errbuf));
      info->pli->time_msv += hmmsearch_WallTime() - t0;
      if (status != eslOK) p7_Fail("--gpu requested, but CUDA batch MSV failed: %s\n", errbuf);
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
      gpu_fb_n = 0;
      gpu_processed_n = 0;
      for (i = 0; i < search_chu->N && (n_targetseqs == -1 || seq_cnt + i < n_targetseqs); i++, gpu_processed_n++) {
        t0 = hmmsearch_WallTime();
        dbsq->dsq    = dbsq_dsqmem;
        dbsq->salloc = dbsq_salloc;
        esl_sq_Reuse(dbsq);
        status = esl_sq_SetName(dbsq, search_chu->name[i]);
        if (status == eslOK) status = esl_sq_SetAccession(dbsq, search_chu->acc[i]);
        if (status == eslOK) status = esl_sq_SetDesc(dbsq, search_chu->desc[i]);
        if (status == eslOK) status = esl_sq_SetCoordComplete(dbsq, search_chu->L[i]);
        if (status != eslOK) goto ERROR;
        dbsq->tax_id = search_chu->taxid[i];
        dbsq->dsq    = search_chu->dsq[i];
        info->pli->time_gpu_meta += hmmsearch_WallTime() - t0;

        p7_pli_NewSeq(info->pli, dbsq);
        if (dbsq->n == 0) { dbsq->dsq = NULL; p7_pipeline_Reuse(info->pli); continue; }
        if (dbsq->n > 100000) ESL_EXCEPTION(eslETYPE, "Target sequence length > 100K, over comparison pipeline limit.\n(Did you mean to use nhmmer/nhmmscan?)");
        p7_bg_SetLength(info->bg, dbsq->n);
        p7_oprofile_ReconfigLength(info->om, dbsq->n);
        {
          GPU_PREVIT_RESULT previt;
          status = gpu_PreViterbiBoundary(info, dbsq, gpu_scores[i], gpu_statuses[i], gpu_nullsc[i],
                                          (info->pli->do_biasfilter ? gpu_filtersc[i] : gpu_nullsc[i]), &previt);
          if (status != eslOK) goto ERROR;
          nullsc = previt.nullsc;
          usc = previt.usc;
          if (info->pli->do_biasfilter) gpu_filtersc[i] = previt.filtersc;
          if (previt.passed_msv) {
            int passed = FALSE;
            p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
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
                    gpu_fwd_idx[gpu_fwd_n++] = i;
                  } else {
                    float fwdsc;
                    p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
                    t0 = hmmsearch_WallTime();
                    p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &fwdsc);
                    info->pli->time_fwd += hmmsearch_WallTime() - t0;
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
                }
              } else if (gpu_fwd_active) {
                if (info->gpu_vit_compare) gpu_vit_idx[gpu_vit_n++] = i;
                status = p7_Pipeline_PostMSVWithFilterPreFwd(info->pli, info->om, info->bg, dbsq, usc + info->gpu_msv_slack, previt.filtersc, &passed);
                if (status != eslOK) goto ERROR;
                if (passed) gpu_fwd_idx[gpu_fwd_n++] = i;
              } else {
                if (info->gpu_vit_compare) gpu_vit_idx[gpu_vit_n++] = i;
                if (info->gpu_fb_parser) {
                  float fwdsc;
                  status = p7_Pipeline_PostMSVWithFilterPreFwd(info->pli, info->om, info->bg, dbsq, usc + info->gpu_msv_slack, previt.filtersc, &passed);
                  if (status != eslOK) goto ERROR;
                  if (passed) {
                    p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
                    t0 = hmmsearch_WallTime();
                    p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &fwdsc);
                    info->pli->time_fwd += hmmsearch_WallTime() - t0;
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
                  p7_Pipeline_PostMSVWithFilter(info->pli, info->om, info->bg, dbsq, NULL, info->th, nullsc, usc + info->gpu_msv_slack, previt.filtersc);
                }
              }
            }
            info->pli->time_gpu_survivor += hmmsearch_WallTime() - t0;
          }
        }
        p7_pipeline_Reuse(info->pli);
        dbsq->dsq = NULL;
      }

      if ((info->gpu_vit_compare || gpu_vit_active) && gpu_vit_n > 0) {
        int run_cuda_vit = info->gpu_vit_compare || gpu_vit_n >= info->gpu_vit_min_seqs;
        if (run_cuda_vit) {
          status = p7_cuda_ViterbiScoreDsqdataSubset(info->cuda_engine, info->cuda_msv, search_chu,
                                                     gpu_vit_idx, gpu_vit_n, gpu_vit_scores, gpu_vit_statuses,
                                                     errbuf, sizeof(errbuf));
          if (status != eslOK) p7_Fail("--gpu requested, but CUDA Viterbi failed: %s\n", errbuf);
        }

        for (int j = 0; j < gpu_vit_n; j++) {
          float cpu_vitsc = -eslINFINITY;
          float filtersc;
          int cpu_status = eslOK;
          int cpu_pass = FALSE;
          int gpu_pass;
          i = gpu_vit_idx[j];
          t0 = hmmsearch_WallTime();
          dbsq->dsq    = dbsq_dsqmem;
          dbsq->salloc = dbsq_salloc;
          esl_sq_Reuse(dbsq);
          status = esl_sq_SetName(dbsq, search_chu->name[i]);
          if (status == eslOK) status = esl_sq_SetAccession(dbsq, search_chu->acc[i]);
          if (status == eslOK) status = esl_sq_SetDesc(dbsq, search_chu->desc[i]);
          if (status == eslOK) status = esl_sq_SetCoordComplete(dbsq, search_chu->L[i]);
          if (status != eslOK) goto ERROR;
          dbsq->tax_id = search_chu->taxid[i];
          dbsq->dsq    = search_chu->dsq[i];
          info->pli->time_gpu_meta += hmmsearch_WallTime() - t0;
          p7_bg_SetLength(info->bg, dbsq->n);
          p7_oprofile_ReconfigLength(info->om, dbsq->n);
          filtersc = info->pli->do_biasfilter ? gpu_filtersc[i] : gpu_nullsc[i];
          if (run_cuda_vit) {
            gpu_pass = (gpu_vit_statuses[j] == eslERANGE);
            if (gpu_vit_statuses[j] == eslOK) {
              seq_score = (gpu_vit_scores[j] - filtersc) / eslCONST_LOG2;
              gpu_pass = (esl_gumbel_surv(seq_score, info->om->evparam[p7_VMU], info->om->evparam[p7_VLAMBDA]) <= info->pli->F2);
            }
          } else {
            p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
            t0 = hmmsearch_WallTime();
            cpu_status = p7_ViterbiFilter(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &cpu_vitsc);
            info->pli->time_vit += hmmsearch_WallTime() - t0;
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
            p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
            cpu_status = p7_ViterbiFilter(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &cpu_vitsc);
            cpu_pass = (cpu_status == eslERANGE);
            if (cpu_status == eslOK) {
              seq_score = (cpu_vitsc - filtersc) / eslCONST_LOG2;
              cpu_pass = (esl_gumbel_surv(seq_score, info->om->evparam[p7_VMU], info->om->evparam[p7_VLAMBDA]) <= info->pli->F2);
            }
            status_mismatch = (cpu_status != gpu_vit_statuses[j]);
            pass_mismatch = (cpu_pass != gpu_pass);
            score_drift = (cpu_status == eslOK && fabsf(cpu_vitsc - gpu_vit_scores[j]) > 0.01f);
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
              gpu_fwd_idx[gpu_fwd_n++] = i;
            } else {
              float fwdsc;
              p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
              t0 = hmmsearch_WallTime();
              p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &fwdsc);
              info->pli->time_fwd += hmmsearch_WallTime() - t0;
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
          dbsq->dsq = NULL;
        }
      }

      if (gpu_fwd_active && gpu_fwd_n > 0) {
        int run_cuda_fwd = info->gpu_fwd_compare || gpu_fwd_n >= info->gpu_fwd_min_seqs;
        if (run_cuda_fwd) {
          t0 = hmmsearch_WallTime();
          status = p7_cuda_ForwardScoreDsqdataSubset(info->cuda_engine, info->cuda_msv, search_chu,
                                                     gpu_fwd_idx, gpu_fwd_n, gpu_fwd_scores, gpu_fwd_statuses,
                                                     errbuf, sizeof(errbuf));
          info->pli->time_fwd += hmmsearch_WallTime() - t0;
          if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward prefilter failed: %s\n", errbuf);
        }

        for (int j = 0; j < gpu_fwd_n; j++) {
          i = gpu_fwd_idx[j];
          t0 = hmmsearch_WallTime();
          dbsq->dsq    = dbsq_dsqmem;
          dbsq->salloc = dbsq_salloc;
          esl_sq_Reuse(dbsq);
          status = esl_sq_SetName(dbsq, search_chu->name[i]);
          if (status == eslOK) status = esl_sq_SetAccession(dbsq, search_chu->acc[i]);
          if (status == eslOK) status = esl_sq_SetDesc(dbsq, search_chu->desc[i]);
          if (status == eslOK) status = esl_sq_SetCoordComplete(dbsq, search_chu->L[i]);
          if (status != eslOK) goto ERROR;
          dbsq->tax_id = search_chu->taxid[i];
          dbsq->dsq    = search_chu->dsq[i];
          info->pli->time_gpu_meta += hmmsearch_WallTime() - t0;

          p7_bg_SetLength(info->bg, dbsq->n);
          p7_oprofile_ReconfigLength(info->om, dbsq->n);
          nullsc = gpu_nullsc[i];

          if (!run_cuda_fwd) {
            float fwdsc;
            float filtersc = info->pli->do_biasfilter ? gpu_filtersc[i] : nullsc;
            p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
            t0 = hmmsearch_WallTime();
            p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &fwdsc);
            info->pli->time_fwd += hmmsearch_WallTime() - t0;
            if (info->gpu_fb_parser) {
                status = gpu_fb_batch_Add(info, &gpu_fb_idx, &gpu_fb_nullsc, &gpu_fb_filtersc, &gpu_fb_fwdsc,
                                          &gpu_fb_n, &gpu_fb_alloc, i, nullsc, filtersc, fwdsc);
                if (status != eslOK) goto ERROR;
              } else {
                status = gpu_PostFwd(info, dbsq, nullsc, filtersc, fwdsc, errbuf, sizeof(errbuf));
                if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
              }
            p7_pipeline_Reuse(info->pli);
            dbsq->dsq = NULL;
            continue;
          }

          if (info->gpu_fwd_compare) {
            float cpu_fwdsc;
            float filtersc = info->pli->do_biasfilter ? gpu_filtersc[i] : nullsc;
            double gpuP = 1.0;
            double cpuP = 1.0;
            int gpu_pass = FALSE;
            int cpu_pass = FALSE;
            p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
            t0 = hmmsearch_WallTime();
            status = p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &cpu_fwdsc);
            info->pli->time_fwd += hmmsearch_WallTime() - t0;
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
            if (cpu_pass != gpu_pass || (status == eslOK && gpu_fwd_statuses[j] == eslOK && fabsf(cpu_fwdsc - gpu_fwd_scores[j]) > 0.01f)) {
              fprintf(stderr, "CUDAFWD\t%s\tL=%ld\tcpu_status=%d\tgpu_status=%d\tcpu=%.6f\tgpu=%.6f\tdiff=%.6f\tcpuP=%.6g\tgpuP=%.6g\tcpu_pass=%d\tgpu_pass=%d\n",
                      dbsq->name ? dbsq->name : "-", (long) dbsq->n, status, gpu_fwd_statuses[j],
                      cpu_fwdsc, gpu_fwd_scores[j], gpu_fwd_scores[j] - cpu_fwdsc, cpuP, gpuP, cpu_pass, gpu_pass);
            }
            if (status != eslOK) {
              p7_pipeline_Reuse(info->pli);
              dbsq->dsq = NULL;
              continue;
            }
            if (cpu_pass) {
              status = gpu_PostFwd(info, dbsq, nullsc, filtersc, cpu_fwdsc, errbuf, sizeof(errbuf));
              if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
            }
            p7_pipeline_Reuse(info->pli);
            dbsq->dsq = NULL;
            continue;
          }

          if (gpu_fwd_statuses[j] == eslOK) {
            float filtersc = info->pli->do_biasfilter ? gpu_filtersc[i] : nullsc;
            seq_score = (gpu_fwd_scores[j] - filtersc) / eslCONST_LOG2;
            P = esl_exp_surv(seq_score, info->om->evparam[p7_FTAU], info->om->evparam[p7_FLAMBDA]);
            if (P <= info->pli->F3) {
              if (info->gpu_fb_parser) {
                status = gpu_fb_batch_Add(info, &gpu_fb_idx, &gpu_fb_nullsc, &gpu_fb_filtersc, &gpu_fb_fwdsc,
                                          &gpu_fb_n, &gpu_fb_alloc, i, nullsc, filtersc, gpu_fwd_scores[j]);
                if (status != eslOK) goto ERROR;
              } else {
                status = gpu_PostFwd(info, dbsq, nullsc, filtersc, gpu_fwd_scores[j], errbuf, sizeof(errbuf));
                if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
              }
            }
          } else if (gpu_fwd_statuses[j] == eslERANGE) {
            float filtersc = info->pli->do_biasfilter ? gpu_filtersc[i] : nullsc;
            float fwdsc;
            /* Keep F3 gating GPU-led in normal mode; only recover a finite score for downstream parser/post processing. */
            p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
            t0 = hmmsearch_WallTime();
            p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &fwdsc);
            info->pli->time_fwd += hmmsearch_WallTime() - t0;
            if (info->gpu_fb_parser) {
              status = gpu_fb_batch_Add(info, &gpu_fb_idx, &gpu_fb_nullsc, &gpu_fb_filtersc, &gpu_fb_fwdsc,
                                        &gpu_fb_n, &gpu_fb_alloc, i, nullsc, filtersc, fwdsc);
              if (status != eslOK) goto ERROR;
            } else {
              status = gpu_PostFwd(info, dbsq, nullsc, filtersc, fwdsc, errbuf, sizeof(errbuf));
              if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
            }
          }
          p7_pipeline_Reuse(info->pli);
          dbsq->dsq = NULL;
        }
      }
      if (gpu_fb_n > 0) {
        status = gpu_ProcessFbBatch(info, search_chu, dbsq, dbsq_dsqmem, dbsq_salloc,
                                    gpu_fb_idx, gpu_fb_nullsc, gpu_fb_filtersc, gpu_fb_fwdsc, gpu_fb_n,
                                    errbuf, sizeof(errbuf));
        if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser batch failed: %s\n", errbuf);
        gpu_fb_n = 0;
      }
      seq_cnt += gpu_processed_n;
      dbsq->dsq = NULL;
      for (i = 0; i < batch.nchunks; i++)
        esl_dsqdata_Recycle(dd, batch.chunks[i]);
      gpu_search_batch_Reset(&batch);
      search_chu = NULL;
    }
    if (n_targetseqs!=-1 && seq_cnt==n_targetseqs) sstatus = eslEOF;
    if (chu) esl_dsqdata_Recycle(dd, chu);
    for (i = 0; i < batch.nchunks; i++)
      esl_dsqdata_Recycle(dd, batch.chunks[i]);
    gpu_search_batch_Destroy(&batch);
    dbsq->dsq    = dbsq_dsqmem;
    dbsq->salloc = dbsq_salloc;
    esl_sq_Destroy(dbsq);
    free(gpu_scores);
    free(gpu_filtersc);
    free(gpu_fwd_scores);
    free(gpu_vit_scores);
    free(gpu_nullsc);
    free(gpu_statuses);
    free(gpu_fwd_statuses);
    free(gpu_vit_statuses);
    free(gpu_fwd_idx);
    free(gpu_vit_idx);
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
  if (chu) esl_dsqdata_Recycle(dd, chu);
  for (i = 0; i < batch.nchunks; i++)
    esl_dsqdata_Recycle(dd, batch.chunks[i]);
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
  free(gpu_nullsc);
  free(gpu_statuses);
  free(gpu_fwd_statuses);
  free(gpu_vit_statuses);
  free(gpu_fwd_idx);
  free(gpu_vit_idx);
  free(gpu_fb_idx);
  free(gpu_fb_nullsc);
  free(gpu_fb_filtersc);
  free(gpu_fb_fwdsc);
  return status;
}
