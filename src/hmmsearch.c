/* hmmsearch: search profile HMM(s) against a sequence database.
 *
 * To do:
 *  - in MPI mode, add a check to make sure ncpus >= 2. If 1, then we
 *    only have a master, no workers. See Infernal commit r3972 on the
 *    same point; and same note in hmmscan.c's to do list.
 */
#include <p7_config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_dsqdata.h"
#include "esl_exponential.h"
#include "esl_getopts.h"
#include "esl_gumbel.h"
#include "esl_msa.h"
#include "esl_msafile.h"
#include "esl_sq.h"
#include "esl_sqio.h"
#include "esl_stopwatch.h"

#ifdef HMMER_MPI
#include "mpi.h"
#include "esl_mpi.h"
#endif 

#ifdef HMMER_THREADS
#include <unistd.h>
#include "esl_threads.h"
#include "esl_workqueue.h"
#endif 

#include "hmmer.h"
#include "cuda_msv.h"

/* Provided by the build-time Easel dsqdata chunk sizing patch. */
extern int esl_dsqdata_OpenSized(ESL_ALPHABET **byp_abc, char *basename, int nconsumers,
                                 int chunk_maxseq, int chunk_maxpacket, ESL_DSQDATA **ret_dd);

typedef struct {
  int                nchunks;
  int                chunk_alloc;
  ESL_DSQDATA_CHUNK **chunks;
  ESL_DSQDATA_CHUNK  view;
} GPU_SEARCH_BATCH;

typedef struct {
#ifdef HMMER_THREADS
  ESL_WORK_QUEUE   *queue;
#endif 
  P7_BG            *bg;	         /* null model                              */
  P7_PIPELINE      *pli;         /* work pipeline                           */
  P7_TOPHITS       *th;          /* top hit results                         */
  P7_OPROFILE      *om;          /* optimized query profile                 */
  P7_CUDA_ENGINE   *cuda_engine; /* optional GPU MSV engine                 */
  P7_CUDA_MSVPROFILE *cuda_msv;  /* optional GPU MSV profile                */
  int               gpu_batch_seqs; /* target sequences per GPU MSV batch     */
  int               gpu_batch_res;  /* target residues per GPU MSV batch      */
  int               gpu_load_seqs;  /* target sequences per dsqdata load      */
  int               gpu_load_res;   /* target residues per dsqdata load       */
  float             gpu_msv_slack;  /* optional extra nats for experiments     */
  int               gpu_fwd_prefilter; /* experimental CUDA Forward score gate  */
  int               gpu_fb_parser;     /* experimental CUDA Forward/Backward parser state */
  int               gpu_vit_prefilter; /* experimental CUDA Viterbi score gate  */
  int               gpu_fwd_min_seqs;  /* min candidates for CUDA Forward batch */
  int               gpu_vit_min_seqs;  /* min candidates for CUDA Viterbi batch */
  int               gpu_fwd_compare;   /* debug CUDA-vs-CPU Forward score check */
  int               gpu_vit_compare;   /* debug CUDA-vs-CPU Viterbi score check */
  int               gpu_fb_compare;    /* debug CUDA-vs-CPU parser state check  */
} WORKER_INFO;

static double
hmmsearch_WallTime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double) tv.tv_sec + (double) tv.tv_usec * 1e-6;
}

static int gpu_search_batch_Init(GPU_SEARCH_BATCH *batch);
static void gpu_search_batch_Reset(GPU_SEARCH_BATCH *batch);
static void gpu_search_batch_Destroy(GPU_SEARCH_BATCH *batch);
static int gpu_search_batch_AddChunk(GPU_SEARCH_BATCH *batch, ESL_DSQDATA_CHUNK *chu);
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
  float cpu_bcksc = 0.0f;
  float max_fwd = 0.0f;
  float max_bck = 0.0f;
  int   max_fwd_i = 0;
  int   max_fwd_s = 0;
  int   max_bck_i = 0;
  int   max_bck_s = 0;

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

  if (fabsf(cpu_fwdsc - gpu_fwdsc) > 0.01f || fabsf(cpu_bcksc - gpu_bcksc) > 0.01f || max_fwd > 0.01f || max_bck > 0.01f) {
    fprintf(stderr, "CUDAFB\t%s\tL=%ld\tcpu_fwd=%.6f\tgpu_fwd=%.6f\tcpu_bck=%.6f\tgpu_bck=%.6f\tmax_fwd=%.6f@%d/%d\tmax_bck=%.6f@%d/%d\n",
            dbsq->name ? dbsq->name : "-", (long) dbsq->n,
            cpu_fwdsc, gpu_fwdsc, cpu_bcksc, gpu_bcksc,
            max_fwd, max_fwd_i, max_fwd_s, max_bck, max_bck_i, max_bck_s);
  }
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

#define REPOPTS     "-E,-T,--cut_ga,--cut_nc,--cut_tc"
#define DOMREPOPTS  "--domE,--domT,--cut_ga,--cut_nc,--cut_tc"
#define INCOPTS     "--incE,--incT,--cut_ga,--cut_nc,--cut_tc"
#define INCDOMOPTS  "--incdomE,--incdomT,--cut_ga,--cut_nc,--cut_tc"
#define THRESHOPTS  "-E,-T,--domE,--domT,--incE,--incT,--incdomE,--incdomT,--cut_ga,--cut_nc,--cut_tc"

#if defined (HMMER_THREADS) && defined (HMMER_MPI)
#define CPUOPTS     "--mpi"
#define MPIOPTS     "--cpu"
#else
#define CPUOPTS     NULL
#define MPIOPTS     NULL
#endif

static ESL_OPTIONS options[] = {
  /* name           type      default  env  range     toggles   reqs   incomp              help                                                      docgroup*/
  { "-h",           eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL,  NULL,            "show brief help on version and usage",                         1 },
  /* Control of output */
  { "-o",           eslARG_OUTFILE, NULL, NULL, NULL,    NULL,  NULL,  NULL,            "direct output to file <f>, not stdout",                        2 },
  { "-A",           eslARG_OUTFILE, NULL, NULL, NULL,    NULL,  NULL,  NULL,            "save multiple alignment of all hits to file <f>",              2 },
  { "--tblout",     eslARG_OUTFILE, NULL, NULL, NULL,    NULL,  NULL,  NULL,            "save parseable table of per-sequence hits to file <f>",        2 },
  { "--domtblout",  eslARG_OUTFILE, NULL, NULL, NULL,    NULL,  NULL,  NULL,            "save parseable table of per-domain hits to file <f>",          2 },
  { "--pfamtblout", eslARG_OUTFILE, NULL, NULL, NULL,    NULL,  NULL,  NULL,            "save table of hits and domains to file, in Pfam format <f>",   2 },
  { "--acc",        eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL,  NULL,            "prefer accessions over names in output",                       2 },
  { "--noali",      eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL,  NULL,            "don't output alignments, so output is smaller",                2 },
  { "--notextw",    eslARG_NONE,    NULL, NULL, NULL,    NULL,  NULL, "--textw",        "unlimit ASCII text output line width",                         2 },
  { "--textw",      eslARG_INT,    "120", NULL, "n>=120",NULL,  NULL, "--notextw",      "set max width of ASCII text output lines",                     2 },
  /* Control of reporting thresholds */
  { "-E",           eslARG_REAL,  "10.0", NULL, "x>0",   NULL,  NULL,  REPOPTS,         "report sequences <= this E-value threshold in output",         4 },
  { "-T",           eslARG_REAL,   FALSE, NULL, NULL,    NULL,  NULL,  REPOPTS,         "report sequences >= this score threshold in output",           4 },
  { "--domE",       eslARG_REAL,  "10.0", NULL, "x>0",   NULL,  NULL,  DOMREPOPTS,      "report domains <= this E-value threshold in output",           4 },
  { "--domT",       eslARG_REAL,   FALSE, NULL, NULL,    NULL,  NULL,  DOMREPOPTS,      "report domains >= this score cutoff in output",                4 },
  /* Control of inclusion (significance) thresholds */
  { "--incE",       eslARG_REAL,  "0.01", NULL, "x>0",   NULL,  NULL,  INCOPTS,         "consider sequences <= this E-value threshold as significant",  5 },
  { "--incT",       eslARG_REAL,   FALSE, NULL, NULL,    NULL,  NULL,  INCOPTS,         "consider sequences >= this score threshold as significant",    5 },
  { "--incdomE",    eslARG_REAL,  "0.01", NULL, "x>0",   NULL,  NULL,  INCDOMOPTS,      "consider domains <= this E-value threshold as significant",    5 },
  { "--incdomT",    eslARG_REAL,   FALSE, NULL, NULL,    NULL,  NULL,  INCDOMOPTS,      "consider domains >= this score threshold as significant",      5 },
  /* Model-specific thresholding for both reporting and inclusion */
  { "--cut_ga",     eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL,  THRESHOPTS,      "use profile's GA gathering cutoffs to set all thresholding",   6 },
  { "--cut_nc",     eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL,  THRESHOPTS,      "use profile's NC noise cutoffs to set all thresholding",       6 },
  { "--cut_tc",     eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL,  THRESHOPTS,      "use profile's TC trusted cutoffs to set all thresholding",     6 },
  /* Control of acceleration pipeline */
  { "--max",        eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL, "--F1,--F2,--F3", "Turn all heuristic filters off (less speed, more power)",      7 },
  { "--F1",         eslARG_REAL,  "0.02", NULL, NULL,    NULL,  NULL, "--max",          "Stage 1 (MSV) threshold: promote hits w/ P <= F1",             7 },
  { "--F2",         eslARG_REAL,  "1e-3", NULL, NULL,    NULL,  NULL, "--max",          "Stage 2 (Vit) threshold: promote hits w/ P <= F2",             7 },
  { "--F3",         eslARG_REAL,  "1e-5", NULL, NULL,    NULL,  NULL, "--max",          "Stage 3 (Fwd) threshold: promote hits w/ P <= F3",             7 },
  { "--nobias",     eslARG_NONE,   NULL,  NULL, NULL,    NULL,  NULL, "--max",          "turn off composition bias filter",                             7 },
  { "--gpu",        eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL, MPIOPTS,          "use CUDA GPU for the MSV filter (no CPU fallback)",            7 },
  { "--gpu-device", eslARG_INT,      "0", NULL, "n>=0",  NULL, "--gpu", NULL,           "CUDA device id for --gpu",                                     99 },
  { "--gpu-batch-seqs", eslARG_INT, "32768", NULL, "n>0", NULL, "--gpu", NULL,          "maximum target sequences per CUDA search batch",               99 },
  { "--gpu-batch-res", eslARG_INT, "8000000", NULL, "n>0", NULL, "--gpu", NULL,         "approximate target residues per CUDA search batch",            99 },
  { "--gpu-load-seqs",  eslARG_INT, "32768", NULL, "n>0", NULL, "--gpu", NULL,          "maximum target sequences per dsqdata load chunk",              99 },
  { "--gpu-load-res",   eslARG_INT, "8000000", NULL, "n>0", NULL, "--gpu", NULL,        "approximate target residues per dsqdata load chunk",           99 },
  { "--gpu-msv-slack", eslARG_REAL,  "0.0", NULL, "x>=0", NULL, "--gpu", NULL,           "experimental extra nats added to GPU MSV scores",              99 },
  { "--gpu-vit-prefilter", eslARG_NONE, FALSE, NULL, NULL, NULL, "--gpu", NULL,          "experimental CUDA Viterbi score prefilter before CPU Forward", 99 },
  { "--gpu-fwd-prefilter", eslARG_NONE, FALSE, NULL, NULL, NULL, "--gpu", NULL,          "experimental CUDA Forward score prefilter before CPU Forward", 99 },
  { "--gpu-fb-parser", eslARG_NONE, FALSE, NULL, NULL, NULL, "--gpu", NULL,              "experimental CUDA Forward/Backward parser state handoff",      99 },
  { "--gpu-vit-min-seqs", eslARG_INT, "1", NULL, "n>0", NULL, "--gpu", NULL,              "minimum candidates needed to launch CUDA Viterbi prefilter",   99 },
  { "--gpu-fwd-min-seqs", eslARG_INT, "1", NULL, "n>0", NULL, "--gpu", NULL,              "minimum candidates needed to launch CUDA Forward prefilter",   99 },
  { "--gpu-fwd-compare", eslARG_NONE, FALSE, NULL, NULL, NULL, "--gpu", NULL,            "debug compare CUDA Forward scores to CPU Forward scores",      99 },
  { "--gpu-vit-compare", eslARG_NONE, FALSE, NULL, NULL, NULL, "--gpu", NULL,            "debug compare CUDA Viterbi scores to CPU Viterbi scores",      99 },
  { "--gpu-fb-compare", eslARG_NONE, FALSE, NULL, NULL, NULL, "--gpu", NULL,             "debug compare CUDA parser state to CPU Forward/Backward",      99 },

/* Other options */
  { "--nonull2",    eslARG_NONE,   NULL,  NULL, NULL,    NULL,  NULL,  NULL,            "turn off biased composition score corrections",               12 },
  { "-Z",           eslARG_REAL,   FALSE, NULL, "x>0",   NULL,  NULL,  NULL,            "set # of comparisons done, for E-value calculation",          12 },
  { "--domZ",       eslARG_REAL,   FALSE, NULL, "x>0",   NULL,  NULL,  NULL,            "set # of significant seqs, for domain E-value calculation",   12 },
  { "--seed",       eslARG_INT,    "42",  NULL, "n>=0",  NULL,  NULL,  NULL,            "set RNG seed to <n> (if 0: one-time arbitrary seed)",         12 },
  { "--tformat",    eslARG_STRING,  NULL, NULL, NULL,    NULL,  NULL,  NULL,            "assert target <seqfile> is in format <s>: no autodetection",  12 },

#ifdef HMMER_THREADS 
  { "--cpu",        eslARG_INT, p7_NCPU,"HMMER_NCPU","n>=0",NULL,  NULL,  CPUOPTS,      "number of parallel CPU workers to use for multithreads",      12 },
#endif
#ifdef HMMER_MPI
  { "--stall",      eslARG_NONE,   FALSE, NULL, NULL,    NULL,"--mpi", NULL,            "arrest after start: for debugging MPI under gdb",             12 },  
  { "--mpi",        eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL,  MPIOPTS,         "run as an MPI parallel program",                              12 },
#endif

  /* Restrict search to subset of database - hidden because these flags are
   *   (a) currently for internal use
   *   (b) probably going to change
   * Doesn't work with MPI
   */
  { "--restrictdb_stkey", eslARG_STRING, "0",  NULL, NULL,    NULL,  NULL,  NULL,       "Search starts at the sequence with name <s> (not with MPI)",     99 },
  { "--restrictdb_n",eslARG_INT,        "-1",  NULL, NULL,    NULL,  NULL,  NULL,       "Search <j> target sequences (starting at --restrictdb_stkey)",   99 },
  { "--ssifile",    eslARG_STRING,       NULL, NULL, NULL,    NULL,  NULL,  NULL,       "restrictdb_x values require ssi file. Override default to <s>",  99 },

  {  0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

static char usage[]  = "[options] <hmmfile> <seqdb>";
static char banner[] = "search profile(s) against a sequence database";

/* struct cfg_s : "Global" application configuration shared by all threads/processes
 * 
 * This structure is passed to routines within main.c, as a means of semi-encapsulation
 * of shared data amongst different parallel processes (threads or MPI processes).
 */
struct cfg_s {
  char            *dbfile;            /* target sequence database file                   */
  char            *hmmfile;           /* query HMM file                                  */

  int              do_mpi;            /* TRUE if we're doing MPI parallelization         */
  int              nproc;             /* how many MPI processes, total                   */
  int              my_rank;           /* who am I, in 0..nproc-1                         */

  char             *firstseq_key;     /* name of the first sequence in the restricted db range */
  int              n_targetseq;       /* number of sequences in the restricted range */
};

static int  serial_master(ESL_GETOPTS *go, struct cfg_s *cfg);
static int  serial_loop  (WORKER_INFO *info, ESL_SQFILE *dbfp, ESL_DSQDATA *dd, int n_targetseqs);

#ifdef HMMER_THREADS
#define BLOCK_SIZE 1000

static int  thread_loop(ESL_THREADS *obj, ESL_WORK_QUEUE *queue, ESL_SQFILE *dbfp, int n_targetseqs);
static void pipeline_thread(void *arg);
#endif 

#ifdef HMMER_MPI
static int  mpi_master   (ESL_GETOPTS *go, struct cfg_s *cfg);
static int  mpi_worker   (ESL_GETOPTS *go, struct cfg_s *cfg);
#endif 


static int
process_commandline(int argc, char **argv, ESL_GETOPTS **ret_go, char **ret_hmmfile, char **ret_seqfile)
{
  ESL_GETOPTS *go = esl_getopts_Create(options);
  int          status;

  if (esl_opt_ProcessEnvironment(go)         != eslOK)  { if (printf("Failed to process environment: %s\n", go->errbuf) < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }
  if (esl_opt_ProcessCmdline(go, argc, argv) != eslOK)  { if (printf("Failed to parse command line: %s\n",  go->errbuf) < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }
  if (esl_opt_VerifyConfig(go)               != eslOK)  { if (printf("Failed to parse command line: %s\n",  go->errbuf) < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }

  /* help format: */
  if (esl_opt_GetBoolean(go, "-h") == TRUE) 
    {
      p7_banner(stdout, argv[0], banner);
      esl_usage(stdout, argv[0], usage);
      if (puts("\nBasic options:")                                           < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
      esl_opt_DisplayHelp(stdout, go, 1, 2, 80); /* 1= group; 2 = indentation; 80=textwidth*/

      if (puts("\nOptions directing output:")                                < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
      esl_opt_DisplayHelp(stdout, go, 2, 2, 80); 

      if (puts("\nOptions controlling reporting thresholds:")                < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
      esl_opt_DisplayHelp(stdout, go, 4, 2, 80); 

      if (puts("\nOptions controlling inclusion (significance) thresholds:") < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
      esl_opt_DisplayHelp(stdout, go, 5, 2, 80); 

      if (puts("\nOptions controlling model-specific thresholding:")         < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
      esl_opt_DisplayHelp(stdout, go, 6, 2, 80); 

      if (puts("\nOptions controlling acceleration heuristics:")             < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
      esl_opt_DisplayHelp(stdout, go, 7, 2, 80); 

      if (puts("\nOther expert options:")                                    < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
      esl_opt_DisplayHelp(stdout, go, 12, 2, 80); 
      exit(0);
    }

  if (esl_opt_ArgNumber(go)                  != 2)     { if (puts("Incorrect number of command line arguments.")      < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }
  if ((*ret_hmmfile = esl_opt_GetArg(go, 1)) == NULL)  { if (puts("Failed to get <hmmfile> argument on command line") < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }
  if ((*ret_seqfile = esl_opt_GetArg(go, 2)) == NULL)  { if (puts("Failed to get <seqdb> argument on command line")   < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }

  /* Validate any attempted use of stdin streams */
  if (strcmp(*ret_hmmfile, "-") == 0 && strcmp(*ret_seqfile, "-") == 0) 
    { if (puts("Either <hmmfile> or <seqdb> may be '-' (to read from stdin), but not both.") < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }

  *ret_go = go;
  return eslOK;
  
 FAILURE:  /* all errors handled here are user errors, so be polite.  */
  esl_usage(stdout, argv[0], usage);
  if (puts("\nwhere most common options are:")                                 < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
  esl_opt_DisplayHelp(stdout, go, 1, 2, 80); /* 1= group; 2 = indentation; 80=textwidth*/
  if (printf("\nTo see more help on available options, do %s -h\n\n", argv[0]) < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
  esl_getopts_Destroy(go);
  exit(1);  

 ERROR:
  if (go) esl_getopts_Destroy(go);
  exit(status);
}

static int
output_header(FILE *ofp, const ESL_GETOPTS *go, char *hmmfile, char *seqfile)
{
  p7_banner(ofp, go->argv[0], banner);
  
  if (fprintf(ofp, "# query HMM file:                  %s\n", hmmfile)                                                                                 < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (fprintf(ofp, "# target sequence database:        %s\n", seqfile)                                                                                 < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "-o")           && fprintf(ofp, "# output directed to file:         %s\n",             esl_opt_GetString(go, "-o"))           < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "-A")           && fprintf(ofp, "# MSA of all hits saved to file:   %s\n",             esl_opt_GetString(go, "-A"))           < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--tblout")     && fprintf(ofp, "# per-seq hits tabular output:     %s\n",             esl_opt_GetString(go, "--tblout"))     < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--domtblout")  && fprintf(ofp, "# per-dom hits tabular output:     %s\n",             esl_opt_GetString(go, "--domtblout"))  < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--pfamtblout") && fprintf(ofp, "# pfam-style tabular hit output:   %s\n",             esl_opt_GetString(go, "--pfamtblout")) < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--acc")        && fprintf(ofp, "# prefer accessions over names:    yes\n")                                                   < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--noali")      && fprintf(ofp, "# show alignments in output:       no\n")                                                    < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--notextw")    && fprintf(ofp, "# max ASCII text line length:      unlimited\n")                                             < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--textw")      && fprintf(ofp, "# max ASCII text line length:      %d\n",             esl_opt_GetInteger(go, "--textw"))     < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "-E")           && fprintf(ofp, "# sequence reporting threshold:    E-value <= %g\n",  esl_opt_GetReal(go, "-E"))             < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "-T")           && fprintf(ofp, "# sequence reporting threshold:    score >= %g\n",    esl_opt_GetReal(go, "-T"))             < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--domE")       && fprintf(ofp, "# domain reporting threshold:      E-value <= %g\n",  esl_opt_GetReal(go, "--domE"))         < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--domT")       && fprintf(ofp, "# domain reporting threshold:      score >= %g\n",    esl_opt_GetReal(go, "--domT"))         < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--incE")       && fprintf(ofp, "# sequence inclusion threshold:    E-value <= %g\n",  esl_opt_GetReal(go, "--incE"))         < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--incT")       && fprintf(ofp, "# sequence inclusion threshold:    score >= %g\n",    esl_opt_GetReal(go, "--incT"))         < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--incdomE")    && fprintf(ofp, "# domain inclusion threshold:      E-value <= %g\n",  esl_opt_GetReal(go, "--incdomE"))      < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--incdomT")    && fprintf(ofp, "# domain inclusion threshold:      score >= %g\n",    esl_opt_GetReal(go, "--incdomT"))      < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--cut_ga")     && fprintf(ofp, "# model-specific thresholding:     GA cutoffs\n")                                            < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed"); 
  if (esl_opt_IsUsed(go, "--cut_nc")     && fprintf(ofp, "# model-specific thresholding:     NC cutoffs\n")                                            < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed"); 
  if (esl_opt_IsUsed(go, "--cut_tc")     && fprintf(ofp, "# model-specific thresholding:     TC cutoffs\n")                                            < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--max")        && fprintf(ofp, "# Max sensitivity mode:            on [all heuristic filters off]\n")                        < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--F1")         && fprintf(ofp, "# MSV filter P threshold:       <= %g\n",             esl_opt_GetReal(go, "--F1"))           < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--F2")         && fprintf(ofp, "# Vit filter P threshold:       <= %g\n",             esl_opt_GetReal(go, "--F2"))           < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--F3")         && fprintf(ofp, "# Fwd filter P threshold:       <= %g\n",             esl_opt_GetReal(go, "--F3"))           < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--nobias")     && fprintf(ofp, "# biased composition HMM filter:   off\n")                                                   < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--gpu")        && fprintf(ofp, "# CUDA MSV filter:                on [device %d]\n", esl_opt_GetInteger(go, "--gpu-device")) < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--restrictdb_stkey") && fprintf(ofp, "# Restrict db to start at seq key: %s\n",            esl_opt_GetString(go, "--restrictdb_stkey"))  < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--restrictdb_n")     && fprintf(ofp, "# Restrict db to # target seqs:    %d\n",            esl_opt_GetInteger(go, "--restrictdb_n")) < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--ssifile")          && fprintf(ofp, "# Override ssi file to:            %s\n",            esl_opt_GetString(go, "--ssifile"))       < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");

  if (esl_opt_IsUsed(go, "--nonull2")    && fprintf(ofp, "# null2 bias corrections:          off\n")                                                   < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "-Z")           && fprintf(ofp, "# sequence search space set to:    %.0f\n",           esl_opt_GetReal(go, "-Z"))             < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--domZ")       && fprintf(ofp, "# domain search space set to:      %.0f\n",           esl_opt_GetReal(go, "--domZ"))         < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--seed"))  {
    if (esl_opt_GetInteger(go, "--seed") == 0 && fprintf(ofp, "# random number seed:              one-time arbitrary\n")                               < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
    else if (                               fprintf(ofp, "# random number seed set to:       %d\n",             esl_opt_GetInteger(go, "--seed"))      < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  }
  if (esl_opt_IsUsed(go, "--tformat")    && fprintf(ofp, "# targ <seqfile> format asserted:  %s\n",             esl_opt_GetString(go, "--tformat"))    < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
#ifdef HMMER_THREADS
  if (esl_opt_IsUsed(go, "--cpu")        && fprintf(ofp, "# number of worker threads:        %d\n",             esl_opt_GetInteger(go, "--cpu"))       < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");  
#endif
#ifdef HMMER_MPI
  if (esl_opt_IsUsed(go, "--mpi")        && fprintf(ofp, "# MPI:                             on\n")                                                    < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
#endif
  if (fprintf(ofp, "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")                                                    < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  return eslOK;
}

int
main(int argc, char **argv)
{
  ESL_GETOPTS     *go       = NULL;	
  struct cfg_s     cfg;        
  int              status   = eslOK;

  impl_Init();                  /* processor specific initialization */
  p7_FLogsumInit();		/* we're going to use table-driven Logsum() approximations at times */

  /* Initialize what we can in the config structure (without knowing the alphabet yet) 
   */
  cfg.hmmfile    = NULL;
  cfg.dbfile     = NULL;
  cfg.do_mpi     = FALSE;	           /* this gets reset below, if we init MPI */
  cfg.nproc      = 0;		           /* this gets reset below, if we init MPI */
  cfg.my_rank    = 0;		           /* this gets reset below, if we init MPI */
  cfg.firstseq_key = NULL;
  cfg.n_targetseq  = -1;

  process_commandline(argc, argv, &go, &cfg.hmmfile, &cfg.dbfile);    

/* is the range restricted? */
  if (esl_opt_IsUsed(go, "--restrictdb_stkey") )
    if ((cfg.firstseq_key = esl_opt_GetString(go, "--restrictdb_stkey")) == NULL)  p7_Fail("Failure capturing --restrictdb_stkey\n");

  if (esl_opt_IsUsed(go, "--restrictdb_n") )
    cfg.n_targetseq = esl_opt_GetInteger(go, "--restrictdb_n");

  if ( cfg.n_targetseq != -1 && cfg.n_targetseq < 1 )
    p7_Fail("--restrictdb_n must be >= 1\n");

  if (esl_opt_GetBoolean(go, "--gpu")) {
#ifdef HMMER_THREADS
    if (esl_opt_GetInteger(go, "--cpu") > 0)
      p7_Fail("--gpu currently requires serial hmmsearch; use --cpu 0\n");
#endif
    if (esl_opt_IsOn(go, "--tformat"))
      p7_Fail("--gpu requires an hmmseqdb/dsqdata target database; --tformat is only for ordinary sequence files\n");
    if (esl_opt_IsUsed(go, "--restrictdb_stkey") || esl_opt_IsUsed(go, "--restrictdb_n"))
      p7_Fail("--gpu does not currently support restricted target database ranges\n");
    if (esl_opt_GetInteger(go, "--gpu-batch-res") > 100000000)
      p7_Fail("--gpu-batch-res is currently limited to 100000000 residues for CUDA MSV v1\n");
    if (esl_opt_GetInteger(go, "--gpu-load-res") > 100000000)
      p7_Fail("--gpu-load-res is currently limited to 100000000 residues for dsqdata GPU mode\n");
    if (esl_opt_GetInteger(go, "--gpu-load-seqs") > esl_opt_GetInteger(go, "--gpu-batch-seqs"))
      p7_Fail("--gpu-load-seqs must be <= --gpu-batch-seqs so each loaded chunk fits in a CUDA search batch\n");
    if (esl_opt_GetInteger(go, "--gpu-load-res") > esl_opt_GetInteger(go, "--gpu-batch-res"))
      p7_Fail("--gpu-load-res must be <= --gpu-batch-res so each loaded chunk fits in a CUDA search batch\n");
  }


  /* Figure out who we are, and send control there: 
   * we might be an MPI master, an MPI worker, or a serial program.
   */
#ifdef HMMER_MPI
  /* pause the execution of the programs execution until the user has a
   * chance to attach with a debugger and send a signal to resume execution
   * i.e. (gdb) signal SIGCONT
   */
  if (esl_opt_GetBoolean(go, "--stall")) pause();

  if (esl_opt_GetBoolean(go, "--mpi")) 
    {
      cfg.do_mpi     = TRUE;
      MPI_Init(&argc, &argv);
      MPI_Comm_rank(MPI_COMM_WORLD, &(cfg.my_rank));
      MPI_Comm_size(MPI_COMM_WORLD, &(cfg.nproc));

      if (cfg.my_rank > 0)  status = mpi_worker(go, &cfg);
      else 		    status = mpi_master(go, &cfg);

      MPI_Finalize();
    }
  else
#endif /*HMMER_MPI*/
    {
      status = serial_master(go, &cfg);
    }

  esl_getopts_Destroy(go);

  return status;
}


/* serial_master()
 * The serial version of hmmsearch.
 * For each query HMM in <hmmfile> search the database for hits.
 * 
 * A master can only return if it's successful. All errors are handled
 * immediately and fatally with p7_Fail().  We also use the
 * ESL_EXCEPTION and ERROR: mechanisms, but only because we know we're
 * using a fatal exception handler.
 */
static int
serial_master(ESL_GETOPTS *go, struct cfg_s *cfg)
{
  FILE            *ofp      = stdout;            /* results output file (-o)                        */
  FILE            *afp      = NULL;              /* alignment output file (-A)                      */
  FILE            *tblfp    = NULL;              /* output stream for tabular per-seq (--tblout)    */
  FILE            *domtblfp = NULL;              /* output stream for tabular per-dom (--domtblout) */
  FILE            *pfamtblfp= NULL;              /* output stream for pfam tabular output (--pfamtblout)    */
  P7_HMMFILE      *hfp      = NULL;              /* open input HMM file                             */
  ESL_SQFILE      *dbfp     = NULL;              /* open input sequence file                        */
  ESL_DSQDATA     *dd       = NULL;              /* open dsqdata sequence database for --gpu         */
  P7_HMM          *hmm      = NULL;              /* one HMM query                                   */
  ESL_ALPHABET    *abc      = NULL;              /* digital alphabet                                */
  int              dbfmt    = eslSQFILE_UNKNOWN; /* format code for sequence database file          */
  ESL_STOPWATCH   *w;
  int              textw    = 0;
  int              nquery   = 0;
  int              status   = eslOK;
  int              hstatus  = eslOK;
  int              sstatus  = eslOK;
  int              i;
  int              do_gpu   = esl_opt_GetBoolean(go, "--gpu");

  int              ncpus    = 0;

  int              infocnt  = 0;
  WORKER_INFO     *info     = NULL;
#ifdef HMMER_THREADS
  ESL_SQ_BLOCK    *block    = NULL;
  ESL_THREADS     *threadObj= NULL;
  ESL_WORK_QUEUE  *queue    = NULL;
#endif
  char             errbuf[eslERRBUFSIZE];

  w = esl_stopwatch_Create();

  if (esl_opt_GetBoolean(go, "--notextw")) textw = 0;
  else                                     textw = esl_opt_GetInteger(go, "--textw");

  if (esl_opt_IsOn(go, "--tformat")) {
    dbfmt = esl_sqio_EncodeFormat(esl_opt_GetString(go, "--tformat"));
    if (dbfmt == eslSQFILE_UNKNOWN) p7_Fail("%s is not a recognized sequence database file format\n", esl_opt_GetString(go, "--tformat"));
  }

  /* Open the target sequence database. GPU mode opens dsqdata after the HMM
   * alphabet is known, so the reader can validate that it is protein dsqdata.
   */
  if (!do_gpu) {
    status = esl_sqfile_Open(cfg->dbfile, dbfmt, p7_SEQDBENV, &dbfp);
    if      (status == eslENOTFOUND) p7_Fail("Failed to open sequence file %s for reading\n",          cfg->dbfile);
    else if (status == eslEFORMAT)   p7_Fail("Sequence file %s is empty or misformatted\n",            cfg->dbfile);
    else if (status == eslEINVAL)    p7_Fail("Can't autodetect format of a stdin or .gz seqfile");
    else if (status != eslOK)        p7_Fail("Unexpected error %d opening sequence file %s\n", status, cfg->dbfile);
  }


  if (!do_gpu && (esl_opt_IsUsed(go, "--restrictdb_stkey") || esl_opt_IsUsed(go, "--restrictdb_n"))) {
    if (esl_opt_IsUsed(go, "--ssifile"))
      esl_sqfile_OpenSSI(dbfp, esl_opt_GetString(go, "--ssifile"));
    else
      esl_sqfile_OpenSSI(dbfp, NULL);
  }



  /* Open the query profile HMM file */
  status = p7_hmmfile_Open(cfg->hmmfile, NULL, &hfp, errbuf);
  if      (status == eslENOTFOUND) p7_Fail("File existence/permissions problem in trying to open HMM file %s.\n%s\n", cfg->hmmfile, errbuf);
  else if (status == eslEFORMAT)   p7_Fail("File format problem in trying to open HMM file %s.\n%s\n",                cfg->hmmfile, errbuf);
  else if (status != eslOK)        p7_Fail("Unexpected error %d in opening HMM file %s.\n%s\n",               status, cfg->hmmfile, errbuf);  

  /* Open the results output files */
  if (esl_opt_IsOn(go, "-o"))          { if ((ofp      = fopen(esl_opt_GetString(go, "-o"), "w")) == NULL) p7_Fail("Failed to open output file %s for writing\n",    esl_opt_GetString(go, "-o")); }
  if (esl_opt_IsOn(go, "-A"))          { if ((afp      = fopen(esl_opt_GetString(go, "-A"), "w")) == NULL) p7_Fail("Failed to open alignment file %s for writing\n", esl_opt_GetString(go, "-A")); }
  if (esl_opt_IsOn(go, "--tblout"))    { if ((tblfp    = fopen(esl_opt_GetString(go, "--tblout"),    "w")) == NULL)  esl_fatal("Failed to open tabular per-seq output file %s for writing\n", esl_opt_GetString(go, "--tblout")); }
  if (esl_opt_IsOn(go, "--domtblout")) { if ((domtblfp = fopen(esl_opt_GetString(go, "--domtblout"), "w")) == NULL)  esl_fatal("Failed to open tabular per-dom output file %s for writing\n", esl_opt_GetString(go, "--domtblout")); }
  if (esl_opt_IsOn(go, "--pfamtblout")){ if ((pfamtblfp = fopen(esl_opt_GetString(go, "--pfamtblout"), "w")) == NULL)  esl_fatal("Failed to open pfam-style tabular output file %s for writing\n", esl_opt_GetString(go, "--pfamtblout")); }

#ifdef HMMER_THREADS
  /* initialize thread data */
  ncpus = do_gpu ? 0 : ESL_MIN( esl_opt_GetInteger(go, "--cpu"), esl_threads_GetCPUCount());
  if (ncpus > 0)
    {
      threadObj = esl_threads_Create(&pipeline_thread);
      queue = esl_workqueue_Create(ncpus * 2);
    }
#endif

  infocnt = (ncpus == 0) ? 1 : ncpus;
  ESL_ALLOC(info, (ptrdiff_t) sizeof(*info) * infocnt);

  /* <abc> is not known 'til first HMM is read. */
  hstatus = p7_hmmfile_Read(hfp, &abc, &hmm);
  if (hstatus == eslOK)
    {
      /* One-time initializations after alphabet <abc> becomes known */
      output_header(ofp, go, cfg->hmmfile, cfg->dbfile);
      if (abc->type != eslAMINO && do_gpu)
        p7_Fail("--gpu is only supported for protein hmmsearch queries\n");
      if (!do_gpu)
        esl_sqfile_SetDigital(dbfp, abc); //ReadBlock requires knowledge of the alphabet to decide how best to read blocks

      for (i = 0; i < infocnt; ++i)
	{
	  info[i].bg    = p7_bg_Create(abc);
    info[i].cuda_engine = NULL;
    info[i].cuda_msv    = NULL;
    info[i].gpu_batch_seqs = esl_opt_GetInteger(go, "--gpu-batch-seqs");
    info[i].gpu_batch_res  = esl_opt_GetInteger(go, "--gpu-batch-res");
    info[i].gpu_load_seqs  = esl_opt_GetInteger(go, "--gpu-load-seqs");
    info[i].gpu_load_res   = esl_opt_GetInteger(go, "--gpu-load-res");
    info[i].gpu_msv_slack  = esl_opt_GetReal(go, "--gpu-msv-slack");
    info[i].gpu_vit_prefilter = esl_opt_GetBoolean(go, "--gpu-vit-prefilter");
    info[i].gpu_fwd_prefilter = esl_opt_GetBoolean(go, "--gpu-fwd-prefilter");
    info[i].gpu_fb_parser     = esl_opt_GetBoolean(go, "--gpu-fb-parser");
    info[i].gpu_vit_min_seqs  = esl_opt_GetInteger(go, "--gpu-vit-min-seqs");
    info[i].gpu_fwd_min_seqs  = esl_opt_GetInteger(go, "--gpu-fwd-min-seqs");
    info[i].gpu_fwd_compare   = esl_opt_GetBoolean(go, "--gpu-fwd-compare");
    info[i].gpu_vit_compare   = esl_opt_GetBoolean(go, "--gpu-vit-compare");
    info[i].gpu_fb_compare    = esl_opt_GetBoolean(go, "--gpu-fb-compare");
#ifdef HMMER_THREADS
	  info[i].queue = queue;
#endif
	}

#ifdef HMMER_THREADS
      for (i = 0; i < ncpus * 2; ++i)
	{
	  block = esl_sq_CreateDigitalBlock(BLOCK_SIZE, abc);
	  if (block == NULL) 	      esl_fatal("Failed to allocate sequence block");

 	  status = esl_workqueue_Init(queue, block);
	  if (status != eslOK)	      esl_fatal("Failed to add block to work queue");
	}
#endif
    }

  /* Outer loop: over each query HMM in <hmmfile>. */
  while (hstatus == eslOK) 
    {
      P7_PROFILE      *gm      = NULL;
      P7_OPROFILE     *om      = NULL;       /* optimized query profile                  */

      nquery++;
      esl_stopwatch_Start(w);

      /* seqfile may need to be rewound (multiquery mode) */
      if (nquery > 1)
      {
        if (!do_gpu && ! esl_sqfile_IsRewindable(dbfp))
          esl_fatal("Target sequence file %s isn't rewindable; can't search it with multiple queries", cfg->dbfile);

        if (!do_gpu && ! esl_opt_IsUsed(go, "--restrictdb_stkey") )
          esl_sqfile_Position(dbfp, 0); //only re-set current position to 0 if we're not planning to set it in a moment
      }

      if (!do_gpu && cfg->firstseq_key != NULL ) { //it's tempting to want to do this once and capture the offset position for future passes, but ncbi files make this non-trivial, so this keeps it general
        sstatus = esl_sqfile_PositionByKey(dbfp, cfg->firstseq_key);
        if (sstatus != eslOK)
          p7_Fail("Failure setting restrictdb_stkey to %d\n", cfg->firstseq_key);
      }

      if (fprintf(ofp, "Query:       %s  [M=%d]\n", hmm->name, hmm->M)  < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
      if (hmm->acc)  { if (fprintf(ofp, "Accession:   %s\n", hmm->acc)  < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed"); }
      if (hmm->desc) { if (fprintf(ofp, "Description: %s\n", hmm->desc) < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed"); }

      /* Convert to an optimized model */
      gm = p7_profile_Create (hmm->M, abc);
      om = p7_oprofile_Create(hmm->M, abc);
      p7_ProfileConfig(hmm, info->bg, gm, 100, p7_LOCAL); /* 100 is a dummy length for now; and MSVFilter requires local mode */
      p7_oprofile_Convert(gm, om);                  /* <om> is now p7_LOCAL, multihit */

      if (do_gpu) {
        int chunk_maxseq    = esl_opt_GetInteger(go, "--gpu-load-seqs");
        int chunk_maxpacket = ESL_MAX(eslDSQDATA_CHUNK_MAXPACKET, (esl_opt_GetInteger(go, "--gpu-load-res") + 5) / 6);
        status = esl_dsqdata_OpenSized(&abc, cfg->dbfile, 1, chunk_maxseq, chunk_maxpacket, &dd);
        if      (status == eslENOTFOUND) p7_Fail("--gpu requires an hmmseqdb/dsqdata target database; failed to open %s: %s\n", cfg->dbfile, dd ? dd->errbuf : "");
        else if (status == eslEFORMAT)   p7_Fail("--gpu target database %s is not compatible protein dsqdata: %s\n", cfg->dbfile, dd ? dd->errbuf : "");
        else if (status != eslOK)        p7_Fail("Unexpected error %d opening dsqdata target database %s\n", status, cfg->dbfile);
        if (abc->type != eslAMINO)       p7_Fail("--gpu requires a protein hmmseqdb/dsqdata target database\n");
      }

      for (i = 0; i < infocnt; ++i)
      {
        /* Create processing pipeline and hit list */
        info[i].th  = p7_tophits_Create();
        info[i].om  = p7_oprofile_Clone(om);
        info[i].pli = p7_pipeline_Create(go, om->M, 100, FALSE, p7_SEARCH_SEQS); /* L_hint = 100 is just a dummy for now */
        info[i].pli->cuda_engine = info[i].cuda_engine;
        info[i].pli->cuda_msv    = info[i].cuda_msv;
        status = p7_pli_NewModel(info[i].pli, info[i].om, info[i].bg);
        if (status == eslEINVAL) p7_Fail(info->pli->errbuf);

        if (do_gpu)
        {
          status = p7_cuda_engine_Create(esl_opt_GetInteger(go, "--gpu-device"), &info[i].cuda_engine, errbuf, sizeof(errbuf));
          if (status != eslOK) p7_Fail("--gpu requested, but CUDA initialization failed: %s\n", errbuf);
          status = p7_cuda_msvprofile_Create(info[i].om, &info[i].cuda_msv, errbuf, sizeof(errbuf));
          if (status != eslOK) p7_Fail("--gpu requested, but CUDA MSV profile creation failed: %s\n", errbuf);
          info[i].pli->cuda_engine = info[i].cuda_engine;
          info[i].pli->cuda_msv    = info[i].cuda_msv;
        }

#ifdef HMMER_THREADS
        if (ncpus > 0) esl_threads_AddThread(threadObj, &info[i]);
#endif
      }

#ifdef HMMER_THREADS
      if (ncpus > 0)  sstatus = thread_loop(threadObj, queue, dbfp, cfg->n_targetseq);
      else            sstatus = serial_loop(info, dbfp, dd, cfg->n_targetseq);
#else
      sstatus = serial_loop(info, dbfp, dd, cfg->n_targetseq);
#endif
      switch(sstatus)
      {
      case eslEFORMAT:
        esl_fatal("Parse failed (sequence file %s):\n%s\n",
            do_gpu ? cfg->dbfile : dbfp->filename, do_gpu ? (dd ? dd->errbuf : "") : esl_sqfile_GetErrorBuf(dbfp));
        break;
      case eslEOF:
        /* do nothing */
        break;
      default:
        esl_fatal("Unexpected error %d reading sequence file %s", sstatus, do_gpu ? cfg->dbfile : dbfp->filename);
      }

      /* merge the results of the search results */
      for (i = 1; i < infocnt; ++i)
      {
        p7_tophits_Merge(info[0].th, info[i].th);
        p7_pipeline_Merge(info[0].pli, info[i].pli);

        p7_pipeline_Destroy(info[i].pli);
        p7_tophits_Destroy(info[i].th);
        p7_cuda_msvprofile_Destroy(info[i].cuda_msv);
        p7_cuda_engine_Destroy(info[i].cuda_engine);
        info[i].cuda_msv    = NULL;
        info[i].cuda_engine = NULL;
        p7_oprofile_Destroy(info[i].om);
      }

      /* Print the results.  */
      p7_tophits_SortBySortkey(info->th);
      p7_tophits_Threshold(info->th, info->pli);
      p7_tophits_Targets(ofp, info->th, info->pli, textw); if (fprintf(ofp, "\n\n") < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
      p7_tophits_Domains(ofp, info->th, info->pli, textw); if (fprintf(ofp, "\n\n") < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");

      if (tblfp)     p7_tophits_TabularTargets(tblfp,    hmm->name, hmm->acc, info->th, info->pli, (nquery == 1));
      if (domtblfp)  p7_tophits_TabularDomains(domtblfp, hmm->name, hmm->acc, info->th, info->pli, (nquery == 1));
      if (pfamtblfp) p7_tophits_TabularXfam(pfamtblfp, hmm->name, hmm->acc, info->th, info->pli);
  
      esl_stopwatch_Stop(w);
      p7_pli_Statistics(ofp, info->pli, w);
      if (fprintf(ofp, "//\n") < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");

      /* Output the results in an MSA (-A option) */
      if (afp) {
	ESL_MSA *msa = NULL;

	if (p7_tophits_Alignment(info->th, abc, NULL, NULL, 0, p7_ALL_CONSENSUS_COLS, &msa) == eslOK)
	  {
	    esl_msa_SetName     (msa, hmm->name, -1);
	    esl_msa_SetAccession(msa, hmm->acc,  -1);
	    esl_msa_SetDesc     (msa, hmm->desc, -1);
	    esl_msa_FormatAuthor(msa, "hmmsearch (HMMER %s)", HMMER_VERSION);

	    if (textw > 0) esl_msafile_Write(afp, msa, eslMSAFILE_STOCKHOLM);
	    else           esl_msafile_Write(afp, msa, eslMSAFILE_PFAM);
	  
	    if (fprintf(ofp, "# Alignment of %d hits satisfying inclusion thresholds saved to: %s\n", msa->nseq, esl_opt_GetString(go, "-A")) < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
	  } 
	else { if (fprintf(ofp, "# No hits satisfy inclusion thresholds; no alignment saved\n") < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed"); }
	  
	esl_msa_Destroy(msa);
      }

      p7_pipeline_Destroy(info->pli);
      p7_tophits_Destroy(info->th);
      p7_cuda_msvprofile_Destroy(info->cuda_msv);
      p7_cuda_engine_Destroy(info->cuda_engine);
      info->cuda_msv    = NULL;
      info->cuda_engine = NULL;
      p7_oprofile_Destroy(info->om);
      p7_oprofile_Destroy(om);
      p7_profile_Destroy(gm);
      p7_hmm_Destroy(hmm);
      if (dd) {
        esl_dsqdata_Close(dd);
        dd = NULL;
      }

      hstatus = p7_hmmfile_Read(hfp, &abc, &hmm);
    } /* end outer loop over query HMMs */

  switch(hstatus) {
  case eslEOD:       p7_Fail("read failed, HMM file %s may be truncated?", cfg->hmmfile);      break;
  case eslEFORMAT:   p7_Fail("bad file format in HMM file %s",             cfg->hmmfile);      break;
  case eslEINCOMPAT: p7_Fail("HMM file %s contains different alphabets",   cfg->hmmfile);      break;
  case eslEOF:       /* do nothing. EOF is what we want. */                                    break;
  default:           p7_Fail("Unexpected error (%d) in reading HMMs from %s", hstatus, cfg->hmmfile);
  }


  /* Terminate outputs... any last words?
   */
  if (tblfp)    p7_tophits_TabularTail(tblfp,    "hmmsearch", p7_SEARCH_SEQS, cfg->hmmfile, cfg->dbfile, go);
  if (domtblfp) p7_tophits_TabularTail(domtblfp, "hmmsearch", p7_SEARCH_SEQS, cfg->hmmfile, cfg->dbfile, go);
  if (pfamtblfp) p7_tophits_TabularTail(pfamtblfp,"hmmsearch", p7_SEARCH_SEQS, cfg->hmmfile, cfg->dbfile, go);
  if (ofp)      { if (fprintf(ofp, "[ok]\n") < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed"); }

  /* Cleanup - prepare for exit
   */
  for (i = 0; i < infocnt; ++i)
    p7_bg_Destroy(info[i].bg);

#ifdef HMMER_THREADS
  if (ncpus > 0)
    {
      esl_workqueue_Reset(queue);
      while (esl_workqueue_Remove(queue, (void **) &block) == eslOK)
	esl_sq_DestroyBlock(block);
      esl_workqueue_Destroy(queue);
      esl_threads_Destroy(threadObj);
    }
#endif

  free(info);
  p7_hmmfile_Close(hfp);
  if (dbfp) esl_sqfile_Close(dbfp);
  esl_alphabet_Destroy(abc);
  esl_stopwatch_Destroy(w);

  if (ofp != stdout) fclose(ofp);
  if (afp)           fclose(afp);
  if (tblfp)         fclose(tblfp);
  if (domtblfp)      fclose(domtblfp);
  if (pfamtblfp)     fclose(pfamtblfp);

  return eslOK;

 ERROR:
  return eslFAIL;
}

#ifdef HMMER_MPI

/* Define common tags used by the MPI master/slave processes */
#define HMMER_ERROR_TAG          1
#define HMMER_HMM_TAG            2
#define HMMER_SEQUENCE_TAG       3
#define HMMER_BLOCK_TAG          4
#define HMMER_PIPELINE_TAG       5
#define HMMER_TOPHITS_TAG        6
#define HMMER_HIT_TAG            7
#define HMMER_TERMINATING_TAG    8
#define HMMER_READY_TAG          9

/* mpi_failure()
 * Generate an error message.  If the clients rank is not 0, a
 * message is created with the error message and sent to the
 * master process for handling.
 */
static void
mpi_failure(char *format, ...)
{
  va_list  argp;
  int      status = eslFAIL;
  int      len;
  int      rank;
  char     str[512];

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* format the error mesg */
  va_start(argp, format);
  len = vsnprintf(str, sizeof(str), format, argp);
  va_end(argp);

  /* make sure the error string is terminated */
  str[sizeof(str)-1] = '\0';

  /* if the caller is the master, print the results and abort */
  if (rank == 0)
    {
      if (fprintf(stderr, "\nError: ") < 0) exit(eslEWRITE);
      if (fprintf(stderr, "%s", str)   < 0) exit(eslEWRITE);
      if (fprintf(stderr, "\n")        < 0) exit(eslEWRITE);
      fflush(stderr);

      MPI_Abort(MPI_COMM_WORLD, status);
      exit(1);
    }
  else
    {
      MPI_Send(str, len, MPI_CHAR, 0, HMMER_ERROR_TAG, MPI_COMM_WORLD);
      pause();
    }
}

#define MAX_BLOCK_SIZE (512*1024)

typedef struct {
  uint64_t  offset;
  uint64_t  length;
  uint64_t  count;
} SEQ_BLOCK;

typedef struct {
  int        complete;
  int        size;
  int        current;
  int        last;
  SEQ_BLOCK *blocks;
} BLOCK_LIST;

/* this routine parses the database keeping track of the blocks
 * offset within the file, number of sequences and the length
 * of the block.  These blocks are passed as work units to the
 * MPI workers.  If multiple hmm's are in the query file, the
 * blocks are reused without parsing the database a second time.
 */
int next_block(ESL_SQFILE *sqfp, ESL_SQ *sq, BLOCK_LIST *list, SEQ_BLOCK *block, int n_targetseqs)
{
  int      status   = eslOK;

  /* if the list has been calculated, use it instead of parsing the database */
  if (list->complete)
    {
      if (list->current == list->last)
      {
        block->offset = 0;
        block->length = 0;
        block->count  = 0;

        status = eslEOF;
      }
      else
      {
        int inx = list->current++;

        block->offset = list->blocks[inx].offset;
        block->length = list->blocks[inx].length;
        block->count  = list->blocks[inx].count;

        status = eslOK;
      }

      return status;
    }

  block->offset = 0;
  block->length = 0;
  block->count = 0;

  esl_sq_Reuse(sq);
  if (n_targetseqs == 0) status = eslEOF; //this is to handle the end-case of a restrictdb scenario, where no more targets are required, and we want to mark the list as complete
  while (block->length < MAX_BLOCK_SIZE && (n_targetseqs <0 || block->count < n_targetseqs) && (status = esl_sqio_ReadInfo(sqfp, sq)) == eslOK)
    {
      if (block->count == 0) block->offset = sq->roff;
      block->length = sq->eoff - block->offset + 1;
      block->count++;
      esl_sq_Reuse(sq);
    }

  if (block->count > 0)
    if (status == eslEOF || block->count == n_targetseqs)
      status = eslOK;
  if (status == eslEOF) list->complete = 1;

  /* add the block to the list of known blocks */
  if (status == eslOK)
    {
      int inx;

      if (list->last >= list->size)
	{
	  void *tmp;
	  list->size += 500;
	  ESL_RALLOC(list->blocks, tmp, sizeof(SEQ_BLOCK) * list->size);
	}

      inx = list->last++;
      list->blocks[inx].offset = block->offset;
      list->blocks[inx].length = block->length;
      list->blocks[inx].count  = block->count;
    }

  return status;

 ERROR:
  return eslEMEM;
}

/* mpi_master()
 * The MPI version of hmmbuild.
 * Follows standard pattern for a master/worker load-balanced MPI program (J1/78-79).
 * 
 * A master can only return if it's successful. 
 * Errors in an MPI master come in two classes: recoverable and nonrecoverable.
 * 
 * Recoverable errors include all worker-side errors, and any
 * master-side error that do not affect MPI communication. Error
 * messages from recoverable messages are delayed until we've cleanly
 * shut down the workers.
 * 
 * Unrecoverable errors are master-side errors that may affect MPI
 * communication, meaning we cannot count on being able to reach the
 * workers and shut them down. Unrecoverable errors result in immediate
 * p7_Fail()'s, which will cause MPI to shut down the worker processes
 * uncleanly.
 */
static int
mpi_master(ESL_GETOPTS *go, struct cfg_s *cfg)
{
  FILE            *ofp      = stdout;            /* results output file (-o)                        */
  FILE            *afp      = NULL;              /* alignment output file (-A)                      */
  FILE            *tblfp    = NULL;              /* output stream for tabular per-seq (--tblout)    */
  FILE            *domtblfp = NULL;              /* output stream for tabular per-dom (--domtblout) */
  FILE            *pfamtblfp= NULL;              /* output stream for pfam-style tabular output  (--pfamtblout) */
  P7_BG           *bg       = NULL;	         /* null model                                      */
  P7_HMMFILE      *hfp      = NULL;              /* open input HMM file                             */
  ESL_SQFILE      *dbfp     = NULL;              /* open input sequence file                        */
  P7_HMM          *hmm      = NULL;              /* one HMM query                                   */
  ESL_SQ          *dbsq     = NULL;              /* one target sequence (digital)                   */
  ESL_ALPHABET    *abc      = NULL;              /* digital alphabet                                */
  int              dbfmt    = eslSQFILE_UNKNOWN; /* format code for sequence database file          */
  ESL_STOPWATCH   *w;
  int              textw    = 0;
  int              nquery   = 0;
  int              status   = eslOK;
  int              hstatus  = eslOK;
  int              sstatus  = eslOK;
  int              dest;

  char            *mpi_buf  = NULL;              /* buffer used to pack/unpack structures */
  int              mpi_size = 0;                 /* size of the allocated buffer */
  BLOCK_LIST      *list     = NULL;
  SEQ_BLOCK        block;

  int              i;
  int              size;
  MPI_Status       mpistatus;
  char             errbuf[eslERRBUFSIZE];

  int              n_targets;

  w = esl_stopwatch_Create();

  if (esl_opt_GetBoolean(go, "--notextw")) textw = 0;
  else                                     textw = esl_opt_GetInteger(go, "--textw");

  if (esl_opt_IsOn(go, "--tformat")) {
    dbfmt = esl_sqio_EncodeFormat(esl_opt_GetString(go, "--tformat"));
    if (dbfmt == eslSQFILE_UNKNOWN) mpi_failure("%s is not a recognized sequence database file format\n", esl_opt_GetString(go, "--tformat"));
  }

  /* Open the target sequence database */
  status = esl_sqfile_Open(cfg->dbfile, dbfmt, p7_SEQDBENV, &dbfp);
  if      (status == eslENOTFOUND) mpi_failure("Failed to open sequence file %s for reading\n",          cfg->dbfile);
  else if (status == eslEFORMAT)   mpi_failure("Sequence file %s is empty or misformatted\n",            cfg->dbfile);
  else if (status == eslEINVAL)    mpi_failure("Can't autodetect format of a stdin or .gz seqfile");
  else if (status != eslOK)        mpi_failure("Unexpected error %d opening sequence file %s\n", status, cfg->dbfile);  

  if (esl_opt_IsUsed(go, "--restrictdb_stkey") || esl_opt_IsUsed(go, "--restrictdb_n")) {
      if (esl_opt_IsUsed(go, "--ssifile"))
        esl_sqfile_OpenSSI(dbfp, esl_opt_GetString(go, "--ssifile"));
      else
        esl_sqfile_OpenSSI(dbfp, NULL);
  }


  /* Open the query profile HMM file */
  status = p7_hmmfile_Open(cfg->hmmfile, NULL, &hfp, errbuf);
  if      (status == eslENOTFOUND) mpi_failure("File existence/permissions problem in trying to open HMM file %s.\n%s\n", cfg->hmmfile, errbuf);
  else if (status == eslEFORMAT)   mpi_failure("File format problem in trying to open HMM file %s.\n%s\n",                cfg->hmmfile, errbuf);
  else if (status != eslOK)        mpi_failure("Unexpected error %d in opening HMM file %s.\n%s\n",               status, cfg->hmmfile, errbuf);  

  /* Open the results output files */
  if (esl_opt_IsOn(go, "-o") && (ofp = fopen(esl_opt_GetString(go, "-o"), "w")) == NULL)
    mpi_failure("Failed to open output file %s for writing\n",    esl_opt_GetString(go, "-o"));

  if (esl_opt_IsOn(go, "-A") && (afp = fopen(esl_opt_GetString(go, "-A"), "w")) == NULL) 
    mpi_failure("Failed to open alignment file %s for writing\n", esl_opt_GetString(go, "-A"));

  if (esl_opt_IsOn(go, "--tblout") && (tblfp = fopen(esl_opt_GetString(go, "--tblout"), "w")) == NULL)
    mpi_failure("Failed to open tabular per-seq output file %s for writing\n", esl_opt_GetString(go, "--tblout"));

  if (esl_opt_IsOn(go, "--domtblout") && (domtblfp = fopen(esl_opt_GetString(go, "--domtblout"), "w")) == NULL)
    mpi_failure("Failed to open tabular per-dom output file %s for writing\n", esl_opt_GetString(go, "--domtblout"));

  if (esl_opt_IsOn(go, "--pfamtblout") && (pfamtblfp = fopen(esl_opt_GetString(go, "--pfamtblout"), "w")) == NULL)
    mpi_failure("Failed to open pfam-style tabular output file %s for writing\n", esl_opt_GetString(go, "--pfamtblout"));

  ESL_ALLOC(list, sizeof(BLOCK_LIST));
  list->complete = 0;
  list->size     = 0;
  list->current  = 0;
  list->last     = 0;
  list->blocks   = NULL;

  /* <abc> is not known 'til first HMM is read. */
  hstatus = p7_hmmfile_Read(hfp, &abc, &hmm);
  if (hstatus == eslOK)
    {
      /* One-time initializations after alphabet <abc> becomes known */
      output_header(ofp, go, cfg->hmmfile, cfg->dbfile);
      dbsq = esl_sq_CreateDigital(abc);
      bg = p7_bg_Create(abc);
    }
  

  if ( cfg->firstseq_key != NULL ) { //it's tempting to want to do this once and capture the offset position for future passes, but ncbi files make this non-trivial, so this keeps it general
    sstatus = esl_sqfile_PositionByKey(dbfp, cfg->firstseq_key);
    if (sstatus != eslOK)
      p7_Fail("Failure setting restrictdb_stkey to %d\n", cfg->firstseq_key);
  }

  /* Outer loop: over each query HMM in <hmmfile>. */
  while (hstatus == eslOK) 
    {
      P7_PROFILE      *gm      = NULL;
      P7_OPROFILE     *om      = NULL;       /* optimized query profile                  */
      P7_PIPELINE     *pli     = NULL;
      P7_TOPHITS      *th      = NULL;
      int              seq_cnt = 0;
      nquery++;
      esl_stopwatch_Start(w);

      n_targets = cfg->n_targetseq;

      /* seqfile may need to be rewound (multiquery mode) */
      if (nquery > 1)   list->current = 0;

      if (fprintf(ofp, "Query:       %s  [M=%d]\n", hmm->name, hmm->M)  < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
      if (hmm->acc)  { if (fprintf(ofp, "Accession:   %s\n", hmm->acc)  < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed"); }
      if (hmm->desc) { if (fprintf(ofp, "Description: %s\n", hmm->desc) < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed"); }

      /* Convert to an optimized model */
      gm = p7_profile_Create (hmm->M, abc);
      om = p7_oprofile_Create(hmm->M, abc);
      p7_ProfileConfig(hmm, bg, gm, 100, p7_LOCAL);
      p7_oprofile_Convert(gm, om);

      /* Create processing pipeline and hit list */
      th  = p7_tophits_Create(); 
      pli = p7_pipeline_Create(go, hmm->M, 100, FALSE, p7_SEARCH_SEQS);
      p7_pli_NewModel(pli, om, bg);

      /* Main loop: */
      while ((n_targets==-1 || seq_cnt<=n_targets) && (sstatus = next_block(dbfp, dbsq, list, &block, n_targets-seq_cnt)) == eslOK )
      {
        seq_cnt += block.count;

        if (MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &mpistatus) != 0)
          mpi_failure("MPI error %d receiving message from %d\n", mpistatus.MPI_SOURCE);

        MPI_Get_count(&mpistatus, MPI_PACKED, &size);
        if (mpi_buf == NULL || size > mpi_size) {
          void *tmp;
          ESL_RALLOC(mpi_buf, tmp, sizeof(char) * size);
          mpi_size = size;
        }

        dest = mpistatus.MPI_SOURCE;
        MPI_Recv(mpi_buf, size, MPI_PACKED, dest, mpistatus.MPI_TAG, MPI_COMM_WORLD, &mpistatus);

        if (mpistatus.MPI_TAG == HMMER_ERROR_TAG)
          mpi_failure("MPI client %d raised error:\n%s\n", dest, mpi_buf);
        if (mpistatus.MPI_TAG != HMMER_READY_TAG)
          mpi_failure("Unexpected tag %d from %d\n", mpistatus.MPI_TAG, dest);

        MPI_Send(&block, 3, MPI_LONG_LONG_INT, dest, HMMER_BLOCK_TAG, MPI_COMM_WORLD);
      }

      if (n_targets!=-1 && seq_cnt==n_targets)
        sstatus = eslEOF;

      switch(sstatus)
      {
      case eslEFORMAT:
        mpi_failure("Parse failed (sequence file %s):\n%s\n", dbfp->filename, esl_sqfile_GetErrorBuf(dbfp));
        break;
      case eslEOF:
        break;
      default:
        mpi_failure("Unexpected error %d reading sequence file %s", sstatus, dbfp->filename);
      }

      block.offset = 0;
      block.length = 0;
      block.count  = 0;

      /* wait for all workers to finish up their work blocks */
      for (i = 1; i < cfg->nproc; ++i)
	{
	  if (MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &mpistatus) != 0) 
	    mpi_failure("MPI error %d receiving message from %d\n", mpistatus.MPI_SOURCE);

	  MPI_Get_count(&mpistatus, MPI_PACKED, &size);
	  if (mpi_buf == NULL || size > mpi_size) {
	    void *tmp;
	    ESL_RALLOC(mpi_buf, tmp, sizeof(char) * size);
	    mpi_size = size; 
	  }

	  dest = mpistatus.MPI_SOURCE;
	  MPI_Recv(mpi_buf, size, MPI_PACKED, dest, mpistatus.MPI_TAG, MPI_COMM_WORLD, &mpistatus);

	  if (mpistatus.MPI_TAG == HMMER_ERROR_TAG)
	    mpi_failure("MPI client %d raised error:\n%s\n", dest, mpi_buf);
	  if (mpistatus.MPI_TAG != HMMER_READY_TAG)
	    mpi_failure("Unexpected tag %d from %d\n", mpistatus.MPI_TAG, dest);
	}

      /* merge the results of the search results */
      for (dest = 1; dest < cfg->nproc; ++dest)
	{
	  P7_PIPELINE     *mpi_pli   = NULL;
	  P7_TOPHITS      *mpi_th    = NULL;

	  /* send an empty block to signal the worker they are done */
	  MPI_Send(&block, 3, MPI_LONG_LONG_INT, dest, HMMER_BLOCK_TAG, MPI_COMM_WORLD);

	  /* wait for the results */
	  if ((status = p7_tophits_MPIRecv(dest, HMMER_TOPHITS_TAG, MPI_COMM_WORLD, &mpi_buf, &mpi_size, &mpi_th)) != eslOK)
	    mpi_failure("Unexpected error %d receiving tophits from %d", status, dest);

	  if ((status = p7_pipeline_MPIRecv(dest, HMMER_PIPELINE_TAG, MPI_COMM_WORLD, &mpi_buf, &mpi_size, go, &mpi_pli)) != eslOK)
	    mpi_failure("Unexpected error %d receiving pipeline from %d", status, dest);

	  p7_tophits_Merge(th, mpi_th);
	  p7_pipeline_Merge(pli, mpi_pli);

	  p7_pipeline_Destroy(mpi_pli);
	  p7_tophits_Destroy(mpi_th);
	}

      /* Print the results.  */
      p7_tophits_SortBySortkey(th);
      p7_tophits_Threshold(th, pli);
      p7_tophits_Targets(ofp, th, pli, textw); if (fprintf(ofp, "\n\n") < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
      p7_tophits_Domains(ofp, th, pli, textw); if (fprintf(ofp, "\n\n") < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");

      if (tblfp)    p7_tophits_TabularTargets(tblfp,    hmm->name, hmm->acc, th, pli, (nquery == 1));
      if (domtblfp) p7_tophits_TabularDomains(domtblfp, hmm->name, hmm->acc, th, pli, (nquery == 1));
      if (pfamtblfp) p7_tophits_TabularXfam(pfamtblfp, hmm->name, hmm->acc, th, pli);

      esl_stopwatch_Stop(w);
      p7_pli_Statistics(ofp, pli, w);
      if (fprintf(ofp, "//\n") < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");

      /* Output the results in an MSA (-A option) */
      if (afp) {
	ESL_MSA *msa = NULL;

	if (p7_tophits_Alignment(th, abc, NULL, NULL, 0, p7_ALL_CONSENSUS_COLS, &msa) == eslOK)
	  {
	    esl_msa_SetName     (msa, hmm->name, -1);
	    esl_msa_SetAccession(msa, hmm->acc,  -1);
	    esl_msa_SetDesc     (msa, hmm->desc, -1);
	    esl_msa_FormatAuthor(msa, "hmmsearch (HMMER %s)", HMMER_VERSION);

	    if (textw > 0) esl_msafile_Write(afp, msa, eslMSAFILE_STOCKHOLM);
	    else           esl_msafile_Write(afp, msa, eslMSAFILE_PFAM);
	  
	    if (fprintf(ofp, "# Alignment of %d hits satisfying inclusion thresholds saved to: %s\n", msa->nseq, esl_opt_GetString(go, "-A")) < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
	  } 
	else { if (fprintf(ofp, "# No hits satisfy inclusion thresholds; no alignment saved\n") < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed"); }
	  
	esl_msa_Destroy(msa);
      }

      p7_pipeline_Destroy(pli);
      p7_tophits_Destroy(th);
      p7_hmm_Destroy(hmm);

      hstatus = p7_hmmfile_Read(hfp, &abc, &hmm);
    } /* end outer loop over query HMMs */

  switch(hstatus) {
  case eslEOD:       mpi_failure("read failed, HMM file %s may be truncated?", cfg->hmmfile);      break;
  case eslEFORMAT:   mpi_failure("bad file format in HMM file %s",             cfg->hmmfile);      break;
  case eslEINCOMPAT: mpi_failure("HMM file %s contains different alphabets",   cfg->hmmfile);      break;
  case eslEOF:       /* EOF is good, that's what we expect here */                                 break;
  default:           mpi_failure("Unexpected error (%d) in reading HMMs from %s", hstatus, cfg->hmmfile);
  }

  /* monitor all the workers to make sure they have ended */
  for (i = 1; i < cfg->nproc; ++i)
    {
      if (MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &mpistatus) != 0) 
	mpi_failure("MPI error %d receiving message from %d\n", mpistatus.MPI_SOURCE);

      MPI_Get_count(&mpistatus, MPI_PACKED, &size);
      if (mpi_buf == NULL || size > mpi_size) {
	void *tmp;
	ESL_RALLOC(mpi_buf, tmp, sizeof(char) * size);
	mpi_size = size; 
      }

      dest = mpistatus.MPI_SOURCE;
      MPI_Recv(mpi_buf, size, MPI_PACKED, dest, mpistatus.MPI_TAG, MPI_COMM_WORLD, &mpistatus);

      if (mpistatus.MPI_TAG == HMMER_ERROR_TAG)
	mpi_failure("MPI client %d raised error:\n%s\n", dest, mpi_buf);
      if (mpistatus.MPI_TAG != HMMER_TERMINATING_TAG)
	mpi_failure("Unexpected tag %d from %d\n", mpistatus.MPI_TAG, dest);
    }

  /* Terminate outputs... any last words?
   */
  if (tblfp)    p7_tophits_TabularTail(tblfp,     "hmmsearch", p7_SEARCH_SEQS, cfg->hmmfile, cfg->dbfile, go);
  if (domtblfp) p7_tophits_TabularTail(domtblfp,  "hmmsearch", p7_SEARCH_SEQS, cfg->hmmfile, cfg->dbfile, go);
  if (pfamtblfp)p7_tophits_TabularTail(pfamtblfp, "hmmsearch", p7_SEARCH_SEQS, cfg->hmmfile, cfg->dbfile, go);
  if (ofp)     { if (fprintf(ofp, "[ok]\n") < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed"); }

  /* Cleanup - prepare for exit
   */
  free(list);
  if (mpi_buf != NULL) free(mpi_buf);

  p7_hmmfile_Close(hfp);
  esl_sqfile_Close(dbfp);

  p7_bg_Destroy(bg);
  esl_sq_Destroy(dbsq);
  esl_stopwatch_Destroy(w);

  if (ofp != stdout) fclose(ofp);
  if (afp)           fclose(afp);
  if (tblfp)         fclose(tblfp);
  if (domtblfp)      fclose(domtblfp);
  if (pfamtblfp)     fclose(pfamtblfp);

  return eslOK;

 ERROR:
  return eslEMEM;
}


static int
mpi_worker(ESL_GETOPTS *go, struct cfg_s *cfg)
{
  P7_HMM          *hmm      = NULL;              /* one HMM query                                   */
  ESL_SQ          *dbsq     = NULL;              /* one target sequence (digital)                   */
  ESL_ALPHABET    *abc      = NULL;              /* digital alphabet                                */
  P7_BG           *bg       = NULL;	         /* null model                                      */
  P7_HMMFILE      *hfp      = NULL;              /* open input HMM file                             */
  ESL_SQFILE      *dbfp     = NULL;              /* open input sequence file                        */
  int              dbfmt    = eslSQFILE_UNKNOWN; /* format code for sequence database file          */
  ESL_STOPWATCH   *w;
  int              status   = eslOK;
  int              hstatus  = eslOK;
  int              sstatus  = eslOK;

  char            *mpi_buf  = NULL;              /* buffer used to pack/unpack structures           */
  int              mpi_size = 0;                 /* size of the allocated buffer                    */

  MPI_Status       mpistatus;
  char             errbuf[eslERRBUFSIZE];

  w = esl_stopwatch_Create();

  /* Open the target sequence database */
  status = esl_sqfile_Open(cfg->dbfile, dbfmt, p7_SEQDBENV, &dbfp);
  if      (status == eslENOTFOUND) mpi_failure("Failed to open sequence file %s for reading\n",          cfg->dbfile);
  else if (status == eslEFORMAT)   mpi_failure("Sequence file %s is empty or misformatted\n",            cfg->dbfile);
  else if (status == eslEINVAL)    mpi_failure("Can't autodetect format of a stdin or .gz seqfile");
  else if (status != eslOK)        mpi_failure("Unexpected error %d opening sequence file %s\n", status, cfg->dbfile);  

  /* Open the query profile HMM file */
  status = p7_hmmfile_Open(cfg->hmmfile, NULL, &hfp, errbuf);
  if      (status == eslENOTFOUND) mpi_failure("File existence/permissions problem in trying to open HMM file %s.\n%s\n", cfg->hmmfile, errbuf);
  else if (status == eslEFORMAT)   mpi_failure("File format problem in trying to open HMM file %s.\n%s\n",                cfg->hmmfile, errbuf);
  else if (status != eslOK)        mpi_failure("Unexpected error %d in opening HMM file %s.\n%s\n",               status, cfg->hmmfile, errbuf);  

  /* <abc> is not known 'til first HMM is read. */
  hstatus = p7_hmmfile_Read(hfp, &abc, &hmm);
  if (hstatus == eslOK)
    {
      /* One-time initializations after alphabet <abc> becomes known */
      dbsq = esl_sq_CreateDigital(abc);
      bg = p7_bg_Create(abc);
      esl_sqfile_SetDigital(dbfp, abc);
    }
  
  /* Outer loop: over each query HMM in <hmmfile>. */
  while (hstatus == eslOK) 
    {
      P7_PROFILE      *gm      = NULL;
      P7_OPROFILE     *om      = NULL;       /* optimized query profile                  */
      P7_PIPELINE     *pli     = NULL;
      P7_TOPHITS      *th      = NULL;

      SEQ_BLOCK        block;

      esl_stopwatch_Start(w);

      status = 0;
      MPI_Send(&status, 1, MPI_INT, 0, HMMER_READY_TAG, MPI_COMM_WORLD);

      /* Convert to an optimized model */
      gm = p7_profile_Create (hmm->M, abc);
      om = p7_oprofile_Create(hmm->M, abc);
      p7_ProfileConfig(hmm, bg, gm, 100, p7_LOCAL);
      p7_oprofile_Convert(gm, om);

      th  = p7_tophits_Create(); 
      pli = p7_pipeline_Create(go, om->M, 100, FALSE, p7_SEARCH_SEQS); /* L_hint = 100 is just a dummy for now */
      p7_pli_NewModel(pli, om, bg);

      /* receive a sequence block from the master */
      MPI_Recv(&block, 3, MPI_LONG_LONG_INT, 0, HMMER_BLOCK_TAG, MPI_COMM_WORLD, &mpistatus);
      while (block.count > 0)
	{
	  uint64_t length = 0;
	  uint64_t count  = block.count;

	  status = esl_sqfile_Position(dbfp, block.offset);
	  if (status != eslOK) mpi_failure("Cannot position sequence database to %ld\n", block.offset);

	  while (count > 0 && (sstatus = esl_sqio_Read(dbfp, dbsq)) == eslOK)
	    {
	      length = dbsq->eoff - block.offset + 1;

	      p7_pli_NewSeq(pli, dbsq);
	      p7_bg_SetLength(bg, dbsq->n);
	      p7_oprofile_ReconfigLength(om, dbsq->n);
      
	      p7_Pipeline(pli, om, bg, dbsq, NULL, th);

	      esl_sq_Reuse(dbsq);
	      p7_pipeline_Reuse(pli);

	      --count;
	    }

	  /* lets do a little bit of sanity checking here to make sure the blocks are the same */
	  if (count > 0)              mpi_failure("Block count mismatch - expected %ld found %ld at offset %ld\n",  block.count,  block.count - count, block.offset);
	  if (block.length != length) mpi_failure("Block length mismatch - expected %ld found %ld at offset %ld\n", block.length, length,              block.offset);

	  /* inform the master we need another block of sequences */
	  status = 0;
	  MPI_Send(&status, 1, MPI_INT, 0, HMMER_READY_TAG, MPI_COMM_WORLD);

	  /* wait for the next block of sequences */
	  MPI_Recv(&block, 3, MPI_LONG_LONG_INT, 0, HMMER_BLOCK_TAG, MPI_COMM_WORLD, &mpistatus);
	}

      esl_stopwatch_Stop(w);

      /* Send the top hits back to the master. */
      p7_tophits_MPISend(th, 0, HMMER_TOPHITS_TAG, MPI_COMM_WORLD,  &mpi_buf, &mpi_size);
      p7_pipeline_MPISend(pli, 0, HMMER_PIPELINE_TAG, MPI_COMM_WORLD,  &mpi_buf, &mpi_size);

      p7_pipeline_Destroy(pli);
      p7_tophits_Destroy(th);
      p7_oprofile_Destroy(om);
      p7_profile_Destroy(gm);
      p7_hmm_Destroy(hmm);

      hstatus = p7_hmmfile_Read(hfp, &abc, &hmm);
    } /* end outer loop over query HMMs */

  switch(hstatus)
    {
    case eslEOF:
      /* do nothing */
      break;
    case eslEFORMAT:
      mpi_failure("bad file format in HMM file %s", cfg->hmmfile);
      break;
    case eslEINCOMPAT:
      mpi_failure("HMM file %s contains different alphabets", cfg->hmmfile);
      break;
    default:
      mpi_failure("Unexpected error (%d) in reading HMMs from %s", hstatus, cfg->hmmfile);
    }

  status = 0;
  MPI_Send(&status, 1, MPI_INT, 0, HMMER_TERMINATING_TAG, MPI_COMM_WORLD);

  if (mpi_buf != NULL) free(mpi_buf);

  p7_hmmfile_Close(hfp);
  esl_sqfile_Close(dbfp);

  p7_bg_Destroy(bg);
  esl_sq_Destroy(dbsq);
  esl_stopwatch_Destroy(w);

  return eslOK;
}
#endif /*HMMER_MPI*/

static int
serial_loop(WORKER_INFO *info, ESL_SQFILE *dbfp, ESL_DSQDATA *dd, int n_targetseqs)
{
  int      sstatus;
  ESL_SQ   *dbsq     = NULL;   /* one target sequence (digital)  */
  int seq_cnt = 0;
  int status;
  int i;
  float nullsc;
  float usc;
  float seq_score;
  double P;
  char errbuf[eslERRBUFSIZE];
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

  gpu_search_batch_Init(&batch);
  if (info->cuda_engine != NULL) {
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
        t0 = hmmsearch_WallTime();
        p7_bg_NullOne(info->bg, dbsq->dsq, dbsq->n, &nullsc);
        info->pli->time_null += hmmsearch_WallTime() - t0;
        gpu_nullsc[i] = nullsc;
        usc = gpu_scores[i];
        seq_score = (usc - nullsc) / eslCONST_LOG2;
        P = (gpu_statuses[i] == eslERANGE) ? 0.0 : esl_gumbel_surv(seq_score, info->om->evparam[p7_MMU], info->om->evparam[p7_MLAMBDA]);
        if (P <= info->pli->F1) {
          int passed = FALSE;
          p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
          info->pli->n_past_msv++;
          t0 = hmmsearch_WallTime();
          if (info->pli->do_biasfilter) {
            seq_score = (usc + info->gpu_msv_slack - gpu_filtersc[i]) / eslCONST_LOG2;
            P = esl_gumbel_surv(seq_score, info->om->evparam[p7_MMU], info->om->evparam[p7_MLAMBDA]);
            if (P <= info->pli->F1) {
              info->pli->n_past_bias++;
              if (gpu_vit_active) {
                seq_score = (usc + info->gpu_msv_slack - gpu_filtersc[i]) / eslCONST_LOG2;
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
                                                &gpu_fb_n, &gpu_fb_alloc, i, nullsc, gpu_filtersc[i], fwdsc);
                      if (status != eslOK) goto ERROR;
                    } else {
                      status = gpu_PostFwd(info, dbsq, nullsc, gpu_filtersc[i], fwdsc, errbuf, sizeof(errbuf));
                      if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
                    }
                  }
                } else {
                  gpu_vit_idx[gpu_vit_n++] = i;
                }
              } else if (gpu_fwd_active) {
                if (info->gpu_vit_compare) gpu_vit_idx[gpu_vit_n++] = i;
                status = p7_Pipeline_PostMSVWithFilterPreFwd(info->pli, info->om, info->bg, dbsq, usc + info->gpu_msv_slack, gpu_filtersc[i], &passed);
                if (status != eslOK) goto ERROR;
                if (passed) gpu_fwd_idx[gpu_fwd_n++] = i;
              } else {
                if (info->gpu_vit_compare) gpu_vit_idx[gpu_vit_n++] = i;
                if (info->gpu_fb_parser) {
                  float fwdsc;
                  status = p7_Pipeline_PostMSVWithFilterPreFwd(info->pli, info->om, info->bg, dbsq, usc + info->gpu_msv_slack, gpu_filtersc[i], &passed);
                  if (status != eslOK) goto ERROR;
                  if (passed) {
                    p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
                    t0 = hmmsearch_WallTime();
                    p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &fwdsc);
                    info->pli->time_fwd += hmmsearch_WallTime() - t0;
                    if (info->gpu_fb_parser) {
                      status = gpu_fb_batch_Add(info, &gpu_fb_idx, &gpu_fb_nullsc, &gpu_fb_filtersc, &gpu_fb_fwdsc,
                                                &gpu_fb_n, &gpu_fb_alloc, i, nullsc, gpu_filtersc[i], fwdsc);
                      if (status != eslOK) goto ERROR;
                    } else {
                      status = gpu_PostFwd(info, dbsq, nullsc, gpu_filtersc[i], fwdsc, errbuf, sizeof(errbuf));
                      if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
                    }
                  }
                } else {
                  p7_Pipeline_PostMSVWithFilter(info->pli, info->om, info->bg, dbsq, NULL, info->th, nullsc, usc + info->gpu_msv_slack, gpu_filtersc[i]);
                }
              }
            }
          } else {
            info->pli->n_past_bias++;
            if (gpu_vit_active) {
              seq_score = (usc + info->gpu_msv_slack - nullsc) / eslCONST_LOG2;
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
                                              &gpu_fb_n, &gpu_fb_alloc, i, nullsc, nullsc, fwdsc);
                    if (status != eslOK) goto ERROR;
                  } else {
                    status = gpu_PostFwd(info, dbsq, nullsc, nullsc, fwdsc, errbuf, sizeof(errbuf));
                    if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
                  }
                }
              } else {
                gpu_vit_idx[gpu_vit_n++] = i;
              }
            } else if (gpu_fwd_active) {
              if (info->gpu_vit_compare) gpu_vit_idx[gpu_vit_n++] = i;
              status = p7_Pipeline_PostMSVWithFilterPreFwd(info->pli, info->om, info->bg, dbsq, usc + info->gpu_msv_slack, nullsc, &passed);
              if (status != eslOK) goto ERROR;
              if (passed) gpu_fwd_idx[gpu_fwd_n++] = i;
            } else {
              if (info->gpu_vit_compare) gpu_vit_idx[gpu_vit_n++] = i;
              if (info->gpu_fb_parser) {
                float fwdsc;
                status = p7_Pipeline_PostMSVWithFilterPreFwd(info->pli, info->om, info->bg, dbsq, usc + info->gpu_msv_slack, nullsc, &passed);
                if (status != eslOK) goto ERROR;
                if (passed) {
                  p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
                  t0 = hmmsearch_WallTime();
                  p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &fwdsc);
                  info->pli->time_fwd += hmmsearch_WallTime() - t0;
                  if (info->gpu_fb_parser) {
                    status = gpu_fb_batch_Add(info, &gpu_fb_idx, &gpu_fb_nullsc, &gpu_fb_filtersc, &gpu_fb_fwdsc,
                                              &gpu_fb_n, &gpu_fb_alloc, i, nullsc, nullsc, fwdsc);
                    if (status != eslOK) goto ERROR;
                  } else {
                    status = gpu_PostFwd(info, dbsq, nullsc, nullsc, fwdsc, errbuf, sizeof(errbuf));
                    if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
                  }
                }
              } else {
                p7_Pipeline_PostMSVWithFilter(info->pli, info->om, info->bg, dbsq, NULL, info->th, nullsc, usc + info->gpu_msv_slack, nullsc);
              }
            }
          }
          info->pli->time_gpu_survivor += hmmsearch_WallTime() - t0;
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
            p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
            cpu_status = p7_ViterbiFilter(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &cpu_vitsc);
            cpu_pass = (cpu_status == eslERANGE);
            if (cpu_status == eslOK) {
              seq_score = (cpu_vitsc - filtersc) / eslCONST_LOG2;
              cpu_pass = (esl_gumbel_surv(seq_score, info->om->evparam[p7_VMU], info->om->evparam[p7_VLAMBDA]) <= info->pli->F2);
            }
            if (cpu_status != gpu_vit_statuses[j] || cpu_pass != gpu_pass || (cpu_status == eslOK && fabsf(cpu_vitsc - gpu_vit_scores[j]) > 0.01f)) {
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
            float fwdsc;
            float cutoff_sc;
            const float grayzone_nats = 0.10f;
            int rerun_cpu = FALSE;
            seq_score = (gpu_fwd_scores[j] - filtersc) / eslCONST_LOG2;
            P = esl_exp_surv(seq_score, info->om->evparam[p7_FTAU], info->om->evparam[p7_FLAMBDA]);
            cutoff_sc = info->om->evparam[p7_FTAU] - log(info->pli->F3) / info->om->evparam[p7_FLAMBDA];
            if (seq_score >= cutoff_sc - grayzone_nats) rerun_cpu = TRUE;
            if (rerun_cpu) {
              p7_omx_GrowTo(info->pli->oxf, info->om->M, 0, dbsq->n);
              t0 = hmmsearch_WallTime();
              p7_ForwardParser(dbsq->dsq, dbsq->n, info->om, info->pli->oxf, &fwdsc);
              info->pli->time_fwd += hmmsearch_WallTime() - t0;
              seq_score = (fwdsc - filtersc) / eslCONST_LOG2;
              P = esl_exp_surv(seq_score, info->om->evparam[p7_FTAU], info->om->evparam[p7_FLAMBDA]);
              if (P <= info->pli->F3) {
                if (info->gpu_fb_parser) {
                  status = gpu_fb_batch_Add(info, &gpu_fb_idx, &gpu_fb_nullsc, &gpu_fb_filtersc, &gpu_fb_fwdsc,
                                            &gpu_fb_n, &gpu_fb_alloc, i, nullsc, filtersc, fwdsc);
                  if (status != eslOK) goto ERROR;
                } else {
                  status = gpu_PostFwd(info, dbsq, nullsc, filtersc, fwdsc, errbuf, sizeof(errbuf));
                  if (status != eslOK) p7_Fail("--gpu requested, but CUDA Forward/Backward parser failed: %s\n", errbuf);
                }
              }
            }
          } else if (gpu_fwd_statuses[j] == eslERANGE) {
            float filtersc = info->pli->do_biasfilter ? gpu_filtersc[i] : nullsc;
            float fwdsc;
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
    return sstatus;
  }

  dbsq = esl_sq_CreateDigital(info->om->abc);

  /* Main loop: */
  while ( (n_targetseqs==-1 || seq_cnt<n_targetseqs) &&  (sstatus = esl_sqio_Read(dbfp, dbsq)) == eslOK)
  {
      p7_pli_NewSeq(info->pli, dbsq);
      p7_bg_SetLength(info->bg, dbsq->n);
      p7_oprofile_ReconfigLength(info->om, dbsq->n);
      
      p7_Pipeline(info->pli, info->om, info->bg, dbsq, NULL, info->th);

      seq_cnt++;
      esl_sq_Reuse(dbsq);
      p7_pipeline_Reuse(info->pli);
  }

  if (n_targetseqs!=-1 && seq_cnt==n_targetseqs)
    sstatus = eslEOF;

  esl_sq_Destroy(dbsq);

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
  return eslEMEM;
}

#ifdef HMMER_THREADS
static int
thread_loop(ESL_THREADS *obj, ESL_WORK_QUEUE *queue, ESL_SQFILE *dbfp, int n_targetseqs)
{
  int  status  = eslOK;
  int  sstatus = eslOK;
  int  eofCount = 0;
  ESL_SQ_BLOCK *block;
  void         *newBlock;

  esl_workqueue_Reset(queue);
  esl_threads_WaitForStart(obj);

  status = esl_workqueue_ReaderUpdate(queue, NULL, &newBlock);
  if (status != eslOK) esl_fatal("Work queue reader failed");
      
  /* Main loop: */
  while (sstatus == eslOK )
    {
      block = (ESL_SQ_BLOCK *) newBlock;

      if (n_targetseqs == 0)
      {
        block->count = 0;
        sstatus = eslEOF;
      } else {
        sstatus = esl_sqio_ReadBlock(dbfp, block, -1, n_targetseqs, /*max_init_window=*/FALSE, FALSE);
        n_targetseqs -= block->count;
      }

      if (sstatus == eslEOF)
      {
        if (eofCount < esl_threads_GetWorkerCount(obj)) sstatus = eslOK;
        ++eofCount;
      }

      if (sstatus == eslOK)
      {
        status = esl_workqueue_ReaderUpdate(queue, block, &newBlock);
        if (status != eslOK) esl_fatal("Work queue reader failed");
      }
    }

  status = esl_workqueue_ReaderUpdate(queue, block, NULL);
  if (status != eslOK) esl_fatal("Work queue reader failed");

  if (sstatus == eslEOF)
    {
      /* wait for all the threads to complete */
      esl_threads_WaitForFinish(obj);
      esl_workqueue_Complete(queue);  
    }

  return sstatus;
}

static void 
pipeline_thread(void *arg)
{
  int i;
  int status;
  int workeridx;
  WORKER_INFO   *info;
  ESL_THREADS   *obj;

  ESL_SQ_BLOCK  *block = NULL;
  void          *newBlock;
  
  impl_Init();

  obj = (ESL_THREADS *) arg;
  esl_threads_Started(obj, &workeridx);

  info = (WORKER_INFO *) esl_threads_GetData(obj, workeridx);

  status = esl_workqueue_WorkerUpdate(info->queue, NULL, &newBlock);
  if (status != eslOK) esl_fatal("Work queue worker failed");

  /* loop until all blocks have been processed */
  block = (ESL_SQ_BLOCK *) newBlock;
  while (block->count > 0)
    {
      /* Main loop: */
      for (i = 0; i < block->count; ++i)
	{
	  ESL_SQ *dbsq = block->list + i;

	  p7_pli_NewSeq(info->pli, dbsq);
	  p7_bg_SetLength(info->bg, dbsq->n);
	  p7_oprofile_ReconfigLength(info->om, dbsq->n);
	  
	  p7_Pipeline(info->pli, info->om, info->bg, dbsq, NULL, info->th);
	  
	  esl_sq_Reuse(dbsq);
	  p7_pipeline_Reuse(info->pli);
	}

      status = esl_workqueue_WorkerUpdate(info->queue, block, &newBlock);
      if (status != eslOK) esl_fatal("Work queue worker failed");

      block = (ESL_SQ_BLOCK *) newBlock;
    }

  status = esl_workqueue_WorkerUpdate(info->queue, block, NULL);
  if (status != eslOK) esl_fatal("Work queue worker failed");

  esl_threads_Finished(obj, workeridx);
  return;
}
#endif   /* HMMER_THREADS */
 
