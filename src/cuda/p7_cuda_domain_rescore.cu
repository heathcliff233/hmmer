/* GPU domain rescoring: Full Forward/Backward + Decoding + OptimalAccuracy + OATrace
 * for batched domain envelopes identified by p7_domaindef_ByPosteriorHeuristics.
 *
 * Replaces rescore_isolated_domain() on CPU with GPU-batched equivalent.
 * All domains run in unihit mode (J loop disabled).
 */
#include "p7_cuda_internal.h"

/* ---- helpers (same indexing as p7_cuda_fb_parser.cu) ---- */

__device__ static inline int
dr_tfv_idx(int t, int q, int lane, int Q)
{
  return (t == p7O_DD) ? (((7 * Q) + q) * 4 + lane) : (((q * 7) + t) * 4 + lane);
}

__device__ static inline int
dr_cell_q(int c, int Q)
{
  return c % Q;
}

__device__ static inline int
dr_cell_lane(int c, int Q)
{
  return c / Q;
}

/* ================================================================
 * Kernel 1: Full Forward with full dpf matrix storage
 *
 * One block per domain. Single thread per block.
 * Unihit mode: xf_e_loop=0 (no J), xf_e_move=1 (E->C always).
 * Stores xmx special states AND full M/I/D cells per position.
 * ================================================================ */
__global__ static void
cuda_domain_fwd_full_kernel(const uint8_t **dsq_ptrs, const int *lengths,
                            const float **rfv_ptrs, const float *tfv,
                            int Q, int Kp,
                            float nj,
                            const size_t *dp_offsets,
                            const size_t *xmx_offsets,
                            float *dpf_out, float *xmx_out,
                            float *envsc_out, int *statuses_out)
{
  extern __shared__ float shmem[];
  int b = blockIdx.x;
  int N = Q * 4;
  float *prev = shmem;
  float *curr = prev + (size_t)N * 3;

  if (threadIdx.x != 0) return;

  const uint8_t *dsq = dsq_ptrs[b];
  int L = lengths[b];
  const float *rfv = rfv_ptrs[b];
  float *dpf = dpf_out + dp_offsets[b];
  float *xmx = xmx_out + xmx_offsets[b];

  /* Unihit mode: pmove/ploop are L-dependent; J loop disabled */
  float pmove = (2.0f + nj) / ((float)L + 2.0f + nj);
  float ploop = 1.0f - pmove;

  float xN = 1.0f, xJ = 0.0f, xC = 0.0f, xE = 0.0f;
  float xB = pmove;
  float totscale = 0.0f;

  size_t row_cells = (size_t)N * 3;

  for (int c = 0; c < N * 3; c++) prev[c] = 0.0f;
  for (int c = 0; c < (int)row_cells; c++) dpf[c] = 0.0f;

  xmx[p7X_E] = xE;
  xmx[p7X_N] = xN;
  xmx[p7X_J] = xJ;
  xmx[p7X_B] = xB;
  xmx[p7X_C] = xC;
  xmx[p7X_SCALE] = 1.0f;

  for (int i = 1; i <= L; i++) {
    uint8_t x = dsq[i];
    if (x >= Kp) { envsc_out[b] = 0.0f; statuses_out[b] = eslEINVAL; return; }

    xE = 0.0f;
    for (int c = 0; c < N; c++) {
      int q = dr_cell_q(c, Q);
      int lane = dr_cell_lane(c, Q);
      int cell = c * 3;
      float mpv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 0];
      float dpv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 1];
      float ipv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 2];
      float m = xB * tfv[dr_tfv_idx(p7O_BM, q, lane, Q)];
      m += mpv * tfv[dr_tfv_idx(p7O_MM, q, lane, Q)];
      m += ipv * tfv[dr_tfv_idx(p7O_IM, q, lane, Q)];
      m += dpv * tfv[dr_tfv_idx(p7O_DM, q, lane, Q)];
      m *= rfv[((int)x * Q + q) * 4 + lane];
      curr[cell + 0] = m;
      curr[cell + 2] = prev[cell + 0] * tfv[dr_tfv_idx(p7O_MI, q, lane, Q)]
                      + prev[cell + 2] * tfv[dr_tfv_idx(p7O_II, q, lane, Q)];
    }

    curr[1] = 0.0f;
    for (int c = 1; c < N; c++) {
      int pq = dr_cell_q(c - 1, Q);
      int plane = dr_cell_lane(c - 1, Q);
      curr[c * 3 + 1] = curr[(c - 1) * 3 + 0] * tfv[dr_tfv_idx(p7O_MD, pq, plane, Q)]
                       + curr[(c - 1) * 3 + 1] * tfv[dr_tfv_idx(p7O_DD, pq, plane, Q)];
    }

    for (int c = 0; c < N; c++) xE += curr[c * 3 + 0] + curr[c * 3 + 1];

    /* Unihit: E->C = 1.0, E->J = 0, J loop = 0, J->B = 0 */
    xN = xN * ploop;
    xC = (xC * ploop) + xE;      /* xf_e_move = 1.0 in unihit */
    xJ = 0.0f;                    /* J disabled in unihit */
    xB = xN * pmove;              /* only N->B path in unihit */

    float scale = 1.0f;
    if (xE > 1.0e4f) {
      scale = xE;
      float inv = 1.0f / xE;
      xN *= inv; xC *= inv; xJ *= inv; xB *= inv;
      for (int c = 0; c < N * 3; c++) curr[c] *= inv;
      totscale += logf(xE);
      xE = 1.0f;
    }

    float *dpf_row = dpf + (size_t)i * row_cells;
    for (int c = 0; c < (int)row_cells; c++) dpf_row[c] = curr[c];

    xmx[i * p7X_NXCELLS + p7X_E]     = xE;
    xmx[i * p7X_NXCELLS + p7X_N]     = xN;
    xmx[i * p7X_NXCELLS + p7X_J]     = xJ;
    xmx[i * p7X_NXCELLS + p7X_B]     = xB;
    xmx[i * p7X_NXCELLS + p7X_C]     = xC;
    xmx[i * p7X_NXCELLS + p7X_SCALE] = scale;

    float *tmp = prev; prev = curr; curr = tmp;
  }

  if (isnan(xC) || (L > 0 && xC == 0.0f) || isinf(xC)) {
    envsc_out[b] = 0.0f;
    statuses_out[b] = eslERANGE;
  } else {
    envsc_out[b] = totscale + logf(xC * pmove);
    statuses_out[b] = eslOK;
  }
}

/* ================================================================
 * Kernel 2: Full Backward with full dpf matrix storage
 * ================================================================ */
__global__ static void
cuda_domain_bck_full_kernel(const uint8_t **dsq_ptrs, const int *lengths,
                            const float **rfv_ptrs, const float *tfv,
                            int Q, int Kp,
                            float nj,
                            const size_t *dp_offsets,
                            const size_t *xmx_offsets,
                            float *dpf_out, const float *xmx_fwd,
                            float *xmx_out, float *bcksc_out, int *statuses_out)
{
  extern __shared__ float shmem[];
  int b = blockIdx.x;
  int N = Q * 4;
  float *next = shmem;
  float *curr = next + (size_t)N * 3;

  if (threadIdx.x != 0) return;

  const uint8_t *dsq = dsq_ptrs[b];
  int L = lengths[b];
  const float *rfv = rfv_ptrs[b];
  float *dpf = dpf_out + dp_offsets[b];
  float *xbck = xmx_out + xmx_offsets[b];
  const float *xfwd = xmx_fwd + xmx_offsets[b];
  size_t row_cells = (size_t)N * 3;

  /* Unihit mode */
  float pmove = (2.0f + nj) / ((float)L + 2.0f + nj);
  float ploop = 1.0f - pmove;
  float xJ = 0.0f, xB = 0.0f, xN = 0.0f;
  float xC = pmove;         /* xf_c_move in unihit = pmove */
  float xE = xC * 1.0f;    /* xf_e_move = 1.0 in unihit */
  float totscale = 0.0f;

  /* Initialize row L */
  for (int c = 0; c < N; c++) {
    next[c * 3 + 0] = xE;
    next[c * 3 + 1] = xE;
    next[c * 3 + 2] = 0.0f;
  }
  for (int c = N - 2; c >= 0; c--) {
    int q = dr_cell_q(c, Q);
    int lane = dr_cell_lane(c, Q);
    next[c * 3 + 1] += next[(c + 1) * 3 + 1] * tfv[dr_tfv_idx(p7O_DD, q, lane, Q)];
  }
  for (int c = N - 2; c >= 0; c--) {
    int q = dr_cell_q(c, Q);
    int lane = dr_cell_lane(c, Q);
    next[c * 3 + 0] += next[(c + 1) * 3 + 1] * tfv[dr_tfv_idx(p7O_MD, q, lane, Q)];
  }

  float scale = xfwd[L * p7X_NXCELLS + p7X_SCALE];
  if (scale > 1.0f) {
    float inv = 1.0f / scale;
    xE *= inv; xN *= inv; xC *= inv; xJ *= inv; xB *= inv;
    for (int c = 0; c < N * 3; c++) next[c] *= inv;
    totscale += logf(scale);
  }

  float *dpf_rowL = dpf + (size_t)L * row_cells;
  for (int c = 0; c < (int)row_cells; c++) dpf_rowL[c] = next[c];

  xbck[L * p7X_NXCELLS + p7X_E]     = xE;
  xbck[L * p7X_NXCELLS + p7X_N]     = xN;
  xbck[L * p7X_NXCELLS + p7X_J]     = xJ;
  xbck[L * p7X_NXCELLS + p7X_B]     = xB;
  xbck[L * p7X_NXCELLS + p7X_C]     = xC;
  xbck[L * p7X_NXCELLS + p7X_SCALE] = scale;

  for (int i = L - 1; i >= 1; i--) {
    uint8_t x = dsq[i + 1];
    if (x >= Kp) { bcksc_out[b] = 0.0f; statuses_out[b] = eslEINVAL; return; }

    xB = 0.0f;
    for (int c = 0; c < N; c++) {
      int q = dr_cell_q(c, Q);
      int lane = dr_cell_lane(c, Q);
      float mpv = next[c * 3 + 0] * rfv[((int)x * Q + q) * 4 + lane];
      xB += mpv * tfv[dr_tfv_idx(p7O_BM, q, lane, Q)];
    }
    for (int c = N - 1; c >= 0; c--) {
      int cell = c * 3;
      float mpv = 0.0f;
      if (c + 1 < N) {
        int nq = dr_cell_q(c + 1, Q);
        int nlane = dr_cell_lane(c + 1, Q);
        mpv = next[(c + 1) * 3 + 0] * rfv[((int)x * Q + nq) * 4 + nlane];
      }
      int q = dr_cell_q(c, Q);
      int lane = dr_cell_lane(c, Q);
      curr[cell + 2] = next[cell + 2] * tfv[dr_tfv_idx(p7O_II, q, lane, Q)];
      curr[cell + 0] = next[cell + 2] * tfv[dr_tfv_idx(p7O_MI, q, lane, Q)];
      if (c + 1 < N) {
        int nq = dr_cell_q(c + 1, Q);
        int nlane = dr_cell_lane(c + 1, Q);
        curr[cell + 2] += mpv * tfv[dr_tfv_idx(p7O_IM, nq, nlane, Q)];
        curr[cell + 1]  = mpv * tfv[dr_tfv_idx(p7O_DM, nq, nlane, Q)];
        curr[cell + 0] += mpv * tfv[dr_tfv_idx(p7O_MM, nq, nlane, Q)];
      } else {
        curr[cell + 1] = 0.0f;
      }
    }

    /* Unihit: E->J = 0, J = 0 */
    xC = xC * ploop;
    xJ = 0.0f;
    xN = (xB * pmove) + (xN * ploop);
    xE = xC * 1.0f;   /* xf_e_move = 1.0 */

    for (int c = N - 1; c >= 0; c--) {
      curr[c * 3 + 0] += xE;
      curr[c * 3 + 1] += xE;
      if (c + 1 < N) {
        int q = dr_cell_q(c, Q);
        int lane = dr_cell_lane(c, Q);
        curr[c * 3 + 1] += curr[(c + 1) * 3 + 1] * tfv[dr_tfv_idx(p7O_DD, q, lane, Q)];
      }
    }
    for (int c = N - 2; c >= 0; c--) {
      int q = dr_cell_q(c, Q);
      int lane = dr_cell_lane(c, Q);
      curr[c * 3 + 0] += curr[(c + 1) * 3 + 1] * tfv[dr_tfv_idx(p7O_MD, q, lane, Q)];
    }

    scale = xfwd[i * p7X_NXCELLS + p7X_SCALE];
    if (scale > 1.0f) {
      float inv = 1.0f / scale;
      xE *= inv; xN *= inv; xJ *= inv; xB *= inv; xC *= inv;
      for (int c = 0; c < N * 3; c++) curr[c] *= inv;
      totscale += logf(scale);
    }

    float *dpf_row = dpf + (size_t)i * row_cells;
    for (int c = 0; c < (int)row_cells; c++) dpf_row[c] = curr[c];

    xbck[i * p7X_NXCELLS + p7X_E]     = xE;
    xbck[i * p7X_NXCELLS + p7X_N]     = xN;
    xbck[i * p7X_NXCELLS + p7X_J]     = xJ;
    xbck[i * p7X_NXCELLS + p7X_B]     = xB;
    xbck[i * p7X_NXCELLS + p7X_C]     = xC;
    xbck[i * p7X_NXCELLS + p7X_SCALE] = scale;

    float *tmp = next; next = curr; curr = tmp;
  }

  if (L >= 1) {
    uint8_t x = dsq[1];
    if (x >= Kp) { bcksc_out[b] = 0.0f; statuses_out[b] = eslEINVAL; return; }
    xB = 0.0f;
    for (int c = 0; c < N; c++) {
      int q = dr_cell_q(c, Q);
      int lane = dr_cell_lane(c, Q);
      float mpv = next[c * 3 + 0] * rfv[((int)x * Q + q) * 4 + lane];
      xB += mpv * tfv[dr_tfv_idx(p7O_BM, q, lane, Q)];
    }
    xN = (xB * pmove) + (xN * ploop);
  }

  for (int c = 0; c < (int)row_cells; c++) dpf[c] = 0.0f;

  xbck[p7X_E] = 0.0f;
  xbck[p7X_N] = xN;
  xbck[p7X_J] = 0.0f;
  xbck[p7X_B] = xB;
  xbck[p7X_C] = 0.0f;
  xbck[p7X_SCALE] = 1.0f;

  if (isnan(xN) || (L > 0 && xN == 0.0f) || isinf(xN)) {
    bcksc_out[b] = 0.0f;
    statuses_out[b] = eslERANGE;
  } else {
    bcksc_out[b] = totscale + logf(xN);
    statuses_out[b] = eslOK;
  }
}

/* ================================================================
 * Kernel 3: Posterior decoding
 *
 * pp[i][c] = fwd[i][c] * bck[i][c] * scaleproduct
 * Writes to separate pp buffer (not in-place on bck).
 * ================================================================ */
__global__ static void
cuda_domain_decoding_kernel(const int *lengths, int Q,
                            const size_t *dp_offsets,
                            const size_t *xmx_offsets,
                            const float *dpf_fwd, const float *dpf_bck,
                            float *dpf_pp,
                            const float *xmx_fwd, const float *xmx_bck,
                            float *xmx_pp, float nj, int *statuses)
{
  int b = blockIdx.x;
  if (threadIdx.x != 0) return;

  int L = lengths[b];
  int N = Q * 4;
  size_t row_cells = (size_t)N * 3;
  const float *fwd = dpf_fwd + dp_offsets[b];
  const float *bck = dpf_bck + dp_offsets[b];
  float       *pp  = dpf_pp  + dp_offsets[b];
  const float *xf  = xmx_fwd + xmx_offsets[b];
  const float *xb  = xmx_bck + xmx_offsets[b];
  float       *xp  = xmx_pp  + xmx_offsets[b];

  float scaleproduct = 1.0f / xb[p7X_N];

  /* Unihit: N/C loop = ploop, J loop = 0 */
  float pmove = (2.0f + nj) / ((float)L + 2.0f + nj);
  float ploop = 1.0f - pmove;

  /* Row 0: all zeros */
  for (int c = 0; c < (int)row_cells; c++) pp[c] = 0.0f;
  xp[p7X_E] = 0.0f;
  xp[p7X_N] = 0.0f;
  xp[p7X_J] = 0.0f;
  xp[p7X_C] = 0.0f;
  xp[p7X_B] = 0.0f;
  xp[p7X_SCALE] = 1.0f;

  for (int i = 1; i <= L; i++) {
    const float *frow = fwd + (size_t)i * row_cells;
    const float *brow = bck + (size_t)i * row_cells;
    float       *prow = pp  + (size_t)i * row_cells;
    float totr = scaleproduct * xf[i * p7X_NXCELLS + p7X_SCALE];

    for (int c = 0; c < N; c++) {
      int cell = c * 3;
      prow[cell + 0] = frow[cell + 0] * brow[cell + 0] * totr;  /* M */
      prow[cell + 1] = 0.0f;                                      /* D: always 0 in pp */
      prow[cell + 2] = frow[cell + 2] * brow[cell + 2] * totr;  /* I */
    }

    xp[i * p7X_NXCELLS + p7X_E] = 0.0f;
    xp[i * p7X_NXCELLS + p7X_N] = xf[(i-1) * p7X_NXCELLS + p7X_N] * xb[i * p7X_NXCELLS + p7X_N] * ploop * scaleproduct;
    xp[i * p7X_NXCELLS + p7X_J] = 0.0f;  /* J loop = 0 in unihit */
    xp[i * p7X_NXCELLS + p7X_C] = xf[(i-1) * p7X_NXCELLS + p7X_C] * xb[i * p7X_NXCELLS + p7X_C] * ploop * scaleproduct;
    xp[i * p7X_NXCELLS + p7X_B] = 0.0f;
    xp[i * p7X_NXCELLS + p7X_SCALE] = 1.0f;
    /* Note: scaleproduct is NOT updated when backward uses forward's scales (has_own_scales=FALSE) */
  }

  if (isinf(scaleproduct)) statuses[b] = eslERANGE;
}

/* ================================================================
 * Kernel 4: Optimal Accuracy DP
 *
 * Max-sum DP over posterior probability matrix.
 * Uses tfv for transition boolean checks.
 * In unihit: J=off, E->C=1, N/B/C L-dependent.
 * ================================================================ */
__global__ static void
cuda_domain_optacc_kernel(const int *lengths, int Q,
                          const float *tfv,
                          const size_t *dp_offsets,
                          const size_t *xmx_offsets,
                          const float *dpf_pp, const float *xmx_pp,
                          float *dpf_oa, float *xmx_oa,
                          float *oasc_out, float nj)
{
  int b = blockIdx.x;
  if (threadIdx.x != 0) return;

  int L = lengths[b];
  int N = Q * 4;
  size_t row_cells = (size_t)N * 3;
  const float *pp   = dpf_pp  + dp_offsets[b];
  float       *oa   = dpf_oa  + dp_offsets[b];
  const float *xpp  = xmx_pp  + xmx_offsets[b];
  float       *xoa  = xmx_oa  + xmx_offsets[b];

  float NINF = -1e30f;
  float pmove = (2.0f + nj) / ((float)L + 2.0f + nj);
  float ploop = 1.0f - pmove;

  /* Row 0: all -inf except N=0, B=0 */
  for (int c = 0; c < (int)row_cells; c++) oa[c] = NINF;
  xoa[p7X_E] = NINF;
  xoa[p7X_N] = 0.0f;
  xoa[p7X_J] = NINF;
  xoa[p7X_B] = 0.0f;
  xoa[p7X_C] = NINF;
  xoa[p7X_SCALE] = 1.0f;

  for (int i = 1; i <= L; i++) {
    float *oa_curr = oa + (size_t)i * row_cells;
    float *oa_prev = oa + (size_t)(i-1) * row_cells;
    const float *pp_curr = pp + (size_t)i * row_cells;
    float xEv = NINF;
    float xBv = xoa[(i-1) * p7X_NXCELLS + p7X_B];

    /* First pass: M and I states, initial D from M->D */
    for (int c = 0; c < N; c++) {
      int q = dr_cell_q(c, Q);
      int lane = dr_cell_lane(c, Q);

      float mpv = (c == 0) ? NINF : oa_prev[(c - 1) * 3 + 0];
      float dpv = (c == 0) ? NINF : oa_prev[(c - 1) * 3 + 1];
      float ipv = (c == 0) ? NINF : oa_prev[(c - 1) * 3 + 2];

      /* M: check transitions as boolean gates (>0 means allowed) */
      float sv = NINF;
      if (tfv[dr_tfv_idx(p7O_BM, q, lane, Q)] > 0.0f && xBv > sv) sv = xBv;
      if (tfv[dr_tfv_idx(p7O_MM, q, lane, Q)] > 0.0f && mpv > sv) sv = mpv;
      if (tfv[dr_tfv_idx(p7O_IM, q, lane, Q)] > 0.0f && ipv > sv) sv = ipv;
      if (tfv[dr_tfv_idx(p7O_DM, q, lane, Q)] > 0.0f && dpv > sv) sv = dpv;
      sv += pp_curr[c * 3 + 0];
      oa_curr[c * 3 + 0] = sv;
      if (sv > xEv) xEv = sv;

      /* I: max(M(i-1,k) if t_MI>0, I(i-1,k) if t_II>0) + pp_I */
      float isv = NINF;
      float mik = oa_prev[c * 3 + 0];
      float iik = oa_prev[c * 3 + 2];
      if (tfv[dr_tfv_idx(p7O_MI, q, lane, Q)] > 0.0f && mik > isv) isv = mik;
      if (tfv[dr_tfv_idx(p7O_II, q, lane, Q)] > 0.0f && iik > isv) isv = iik;
      isv += pp_curr[c * 3 + 2];
      oa_curr[c * 3 + 2] = isv;

      /* D: initial from M->D chain */
      oa_curr[c * 3 + 1] = NINF;
    }

    /* D->D propagation: first pass */
    for (int c = 0; c < N; c++) {
      int q = dr_cell_q(c, Q);
      int lane = dr_cell_lane(c, Q);
      /* M->D from same row */
      float md = NINF;
      if (c > 0) {
        int pq = dr_cell_q(c-1, Q);
        int plane = dr_cell_lane(c-1, Q);
        if (tfv[dr_tfv_idx(p7O_MD, pq, plane, Q)] > 0.0f) md = oa_curr[(c-1) * 3 + 0];
      }
      /* D->D */
      float dd = NINF;
      if (c > 0) {
        int pq = dr_cell_q(c-1, Q);
        int plane = dr_cell_lane(c-1, Q);
        if (tfv[dr_tfv_idx(p7O_DD, pq, plane, Q)] > 0.0f) dd = oa_curr[(c-1) * 3 + 1];
      }
      float dv = (md > dd) ? md : dd;
      if (dv > oa_curr[c * 3 + 1]) oa_curr[c * 3 + 1] = dv;
    }

    /* Fully serialized D->D (3 more passes for 4-way striping) */
    for (int j = 1; j < 4; j++) {
      for (int c = 1; c < N; c++) {
        int pq = dr_cell_q(c-1, Q);
        int plane = dr_cell_lane(c-1, Q);
        if (tfv[dr_tfv_idx(p7O_DD, pq, plane, Q)] > 0.0f) {
          float dd = oa_curr[(c-1) * 3 + 1];
          if (dd > oa_curr[c * 3 + 1]) oa_curr[c * 3 + 1] = dd;
        }
      }
    }

    /* D->E paths */
    for (int c = 0; c < N; c++) {
      if (oa_curr[c * 3 + 1] > xEv) xEv = oa_curr[c * 3 + 1];
    }

    xoa[i * p7X_NXCELLS + p7X_E] = xEv;

    /* Unihit: J disabled, E->C = 1.0 */
    xoa[i * p7X_NXCELLS + p7X_J] = NINF;

    float t1, t2;
    t1 = (ploop == 0.0f) ? NINF : xoa[(i-1) * p7X_NXCELLS + p7X_C] + xpp[i * p7X_NXCELLS + p7X_C];
    t2 = xEv;  /* E->C = 1.0 in unihit */
    xoa[i * p7X_NXCELLS + p7X_C] = (t1 > t2) ? t1 : t2;

    xoa[i * p7X_NXCELLS + p7X_N] = (ploop == 0.0f) ? NINF : xoa[(i-1) * p7X_NXCELLS + p7X_N] + xpp[i * p7X_NXCELLS + p7X_N];

    /* B = N->B (J disabled in unihit) */
    xoa[i * p7X_NXCELLS + p7X_B] = (pmove == 0.0f) ? NINF : xoa[i * p7X_NXCELLS + p7X_N];

    xoa[i * p7X_NXCELLS + p7X_SCALE] = 1.0f;
  }

  oasc_out[b] = xoa[L * p7X_NXCELLS + p7X_C];
}

/* ================================================================
 * Kernel 5: OA Traceback
 * ================================================================ */
#define GPU_p7T_S  0
#define GPU_p7T_N  1
#define GPU_p7T_B  2
#define GPU_p7T_M  3
#define GPU_p7T_D  4
#define GPU_p7T_I  5
#define GPU_p7T_E  6
#define GPU_p7T_C  7
#define GPU_p7T_T  8
#define GPU_p7T_J  9

__device__ static float
gpu_get_postprob(const float *pp_dpf, const float *pp_xmx, int scur, int sprv,
                 int k, int i, int Q, size_t row_cells)
{
  if (scur == GPU_p7T_M && k >= 1) {
    int q = (k - 1) % Q;
    int r = (k - 1) / Q;
    return pp_dpf[(size_t)i * row_cells + (size_t)(r * Q + q) * 3 + 0];
  }
  if (scur == GPU_p7T_I && k >= 1) {
    int q = (k - 1) % Q;
    int r = (k - 1) / Q;
    return pp_dpf[(size_t)i * row_cells + (size_t)(r * Q + q) * 3 + 2];
  }
  if (scur == GPU_p7T_N && sprv == scur) return pp_xmx[i * p7X_NXCELLS + p7X_N];
  if (scur == GPU_p7T_C && sprv == scur) return pp_xmx[i * p7X_NXCELLS + p7X_C];
  if (scur == GPU_p7T_J && sprv == scur) return pp_xmx[i * p7X_NXCELLS + p7X_J];
  return 0.0f;
}

__device__ static int
gpu_select_e(const float *oa_dpf, int i, int Q, int M, size_t row_cells, int *ret_k)
{
  float best = -1e30f;
  int best_k = 1;
  int best_is_d = 0;

  for (int k = 1; k <= M; k++) {
    int q = (k - 1) % Q;
    int r = (k - 1) / Q;
    int c = r * Q + q;
    float mv = oa_dpf[(size_t)i * row_cells + (size_t)c * 3 + 0];
    float dv = oa_dpf[(size_t)i * row_cells + (size_t)c * 3 + 1];
    if (mv > best) { best = mv; best_k = k; best_is_d = 0; }
    if (dv > best) { best = dv; best_k = k; best_is_d = 1; }
  }
  *ret_k = best_k;
  return best_is_d ? GPU_p7T_D : GPU_p7T_M;
}

__device__ static int
gpu_select_m(const float *oa_dpf, const float *oa_xmx,
             const float *tfv, int i, int k, int Q, int M, size_t row_cells)
{
  float path[4];
  int   state[4] = { GPU_p7T_M, GPU_p7T_I, GPU_p7T_D, GPU_p7T_B };
  float NINF = -1e30f;

  int q = (k - 1) % Q;
  int r = (k - 1) / Q;

  path[3] = (tfv[dr_tfv_idx(p7O_BM, q, r, Q)] > 0.0f) ? oa_xmx[(i-1) * p7X_NXCELLS + p7X_B] : NINF;

  if (k > 1) {
    int pq = (k - 2) % Q;
    int pr = (k - 2) / Q;
    int pc = pr * Q + pq;
    path[0] = (tfv[dr_tfv_idx(p7O_MM, q, r, Q)] > 0.0f) ? oa_dpf[(size_t)(i-1) * row_cells + (size_t)pc * 3 + 0] : NINF;
    path[1] = (tfv[dr_tfv_idx(p7O_IM, q, r, Q)] > 0.0f) ? oa_dpf[(size_t)(i-1) * row_cells + (size_t)pc * 3 + 2] : NINF;
    path[2] = (tfv[dr_tfv_idx(p7O_DM, q, r, Q)] > 0.0f) ? oa_dpf[(size_t)(i-1) * row_cells + (size_t)pc * 3 + 1] : NINF;
  } else {
    path[0] = NINF; path[1] = NINF; path[2] = NINF;
  }

  int best = 0;
  for (int j = 1; j < 4; j++) if (path[j] > path[best]) best = j;
  return state[best];
}

__device__ static int
gpu_select_d(const float *oa_dpf, const float *tfv, int i, int k, int Q, size_t row_cells)
{
  if (k < 2) return -1;
  int pq = (k - 2) % Q;
  int pr = (k - 2) / Q;
  int pc = pr * Q + pq;
  float path_m = (tfv[dr_tfv_idx(p7O_MD, pq, pr, Q)] > 0.0f) ? oa_dpf[(size_t)i * row_cells + (size_t)pc * 3 + 0] : -1e30f;
  float path_d = (tfv[dr_tfv_idx(p7O_DD, pq, pr, Q)] > 0.0f) ? oa_dpf[(size_t)i * row_cells + (size_t)pc * 3 + 1] : -1e30f;
  return (path_m >= path_d) ? GPU_p7T_M : GPU_p7T_D;
}

__device__ static int
gpu_select_i(const float *oa_dpf, const float *tfv, int i, int k, int Q, size_t row_cells)
{
  int q = (k - 1) % Q;
  int r = (k - 1) / Q;
  int c = r * Q + q;
  float path_m = (tfv[dr_tfv_idx(p7O_MI, q, r, Q)] > 0.0f) ? oa_dpf[(size_t)(i-1) * row_cells + (size_t)c * 3 + 0] : -1e30f;
  float path_i = (tfv[dr_tfv_idx(p7O_II, q, r, Q)] > 0.0f) ? oa_dpf[(size_t)(i-1) * row_cells + (size_t)c * 3 + 2] : -1e30f;
  return (path_m >= path_i) ? GPU_p7T_M : GPU_p7T_I;
}

__global__ static void
cuda_domain_oatrace_kernel(const int *lengths, int Q, int M,
                           const float *tfv,
                           const size_t *dp_offsets,
                           const size_t *xmx_offsets,
                           const float *dpf_pp, const float *xmx_pp,
                           const float *dpf_oa, const float *xmx_oa,
                           float nj,
                           int8_t *trace_st, int *trace_k, int *trace_i,
                           float *trace_pp,
                           int *trace_N, int max_trace_len)
{
  int b = blockIdx.x;
  if (threadIdx.x != 0) return;

  int L = lengths[b];
  int N = Q * 4;
  size_t row_cells = (size_t)N * 3;
  const float *oa_dpf = dpf_oa + dp_offsets[b];
  const float *oa_xmx = xmx_oa + xmx_offsets[b];
  const float *pp_dpf = dpf_pp + dp_offsets[b];
  const float *pp_xmx = xmx_pp + xmx_offsets[b];

  int8_t  *st = trace_st + (size_t)b * max_trace_len;
  int     *tk = trace_k  + (size_t)b * max_trace_len;
  int     *ti = trace_i  + (size_t)b * max_trace_len;
  float   *tp = trace_pp + (size_t)b * max_trace_len;

  float pmove = (2.0f + nj) / ((float)L + 2.0f + nj);
  float ploop = 1.0f - pmove;

  int idx = 0;
  int cur_i = L;
  int cur_k = 0;

  if (idx < max_trace_len) { st[idx] = GPU_p7T_T; tk[idx] = 0; ti[idx] = cur_i; tp[idx] = 0.0f; idx++; }
  if (idx < max_trace_len) { st[idx] = GPU_p7T_C; tk[idx] = 0; ti[idx] = cur_i; tp[idx] = 0.0f; idx++; }

  int s0 = GPU_p7T_C;
  while (s0 != GPU_p7T_S && idx < max_trace_len) {
    int s1;
    switch (s0) {
      case GPU_p7T_M: s1 = gpu_select_m(oa_dpf, oa_xmx, tfv, cur_i, cur_k, Q, M, row_cells); cur_k--; cur_i--; break;
      case GPU_p7T_D: s1 = gpu_select_d(oa_dpf, tfv, cur_i, cur_k, Q, row_cells); cur_k--; break;
      case GPU_p7T_I: s1 = gpu_select_i(oa_dpf, tfv, cur_i, cur_k, Q, row_cells); cur_i--; break;
      case GPU_p7T_N: s1 = (cur_i == 0) ? GPU_p7T_S : GPU_p7T_N; break;
      case GPU_p7T_C: {
        /* select_c: unihit, E->C = 1.0 */
        float path_c = (ploop == 0.0f) ? -1e30f : oa_xmx[(cur_i-1) * p7X_NXCELLS + p7X_C] + pp_xmx[cur_i * p7X_NXCELLS + p7X_C];
        float path_e = oa_xmx[cur_i * p7X_NXCELLS + p7X_E];
        s1 = (path_c >= path_e) ? GPU_p7T_C : GPU_p7T_E;
        break;
      }
      case GPU_p7T_J: s1 = GPU_p7T_E; break; /* J disabled in unihit, always came from E */
      case GPU_p7T_E: s1 = gpu_select_e(oa_dpf, cur_i, Q, M, row_cells, &cur_k); break;
      case GPU_p7T_B: {
        /* select_b: unihit, only N->B */
        s1 = GPU_p7T_N;
        break;
      }
      default: s1 = -1; break;
    }
    if (s1 == -1) break;

    float postprob = gpu_get_postprob(pp_dpf, pp_xmx, s1, s0, cur_k, cur_i, Q, row_cells);
    st[idx] = (int8_t)s1;
    tk[idx] = cur_k;
    ti[idx] = cur_i;
    tp[idx] = postprob;
    idx++;

    if ((s1 == GPU_p7T_N || s1 == GPU_p7T_J || s1 == GPU_p7T_C) && s1 == s0) cur_i--;
    s0 = s1;
  }

  trace_N[b] = idx;
}

/* ================================================================
 * Kernel 6: Forward score-only for domcorrection
 * ================================================================ */
__global__ static void
cuda_domain_fwd_scoreonly_kernel(const uint8_t **dsq_ptrs, const int *lengths,
                                const float *rfv_orig, const float *tfv,
                                int Q, int Kp, float nj,
                                float *scores_out, int *statuses_out)
{
  extern __shared__ float shmem[];
  int b = blockIdx.x;
  int N = Q * 4;
  float *prev = shmem;
  float *curr = prev + (size_t)N * 3;

  if (threadIdx.x != 0) return;

  const uint8_t *dsq = dsq_ptrs[b];
  int L = lengths[b];

  float pmove = (2.0f + nj) / ((float)L + 2.0f + nj);
  float ploop = 1.0f - pmove;
  float xN = 1.0f, xJ = 0.0f, xC = 0.0f, xE = 0.0f;
  float xB = pmove;
  float totscale = 0.0f;

  for (int c = 0; c < N * 3; c++) prev[c] = 0.0f;

  for (int i = 1; i <= L; i++) {
    uint8_t x = dsq[i];
    if (x >= Kp) { scores_out[b] = 0.0f; statuses_out[b] = eslEINVAL; return; }

    xE = 0.0f;
    for (int c = 0; c < N; c++) {
      int q = dr_cell_q(c, Q);
      int lane = dr_cell_lane(c, Q);
      int cell = c * 3;
      float mpv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 0];
      float dpv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 1];
      float ipv = (c == 0) ? 0.0f : prev[(c - 1) * 3 + 2];
      float m = xB * tfv[dr_tfv_idx(p7O_BM, q, lane, Q)];
      m += mpv * tfv[dr_tfv_idx(p7O_MM, q, lane, Q)];
      m += ipv * tfv[dr_tfv_idx(p7O_IM, q, lane, Q)];
      m += dpv * tfv[dr_tfv_idx(p7O_DM, q, lane, Q)];
      m *= rfv_orig[((int)x * Q + q) * 4 + lane];
      curr[cell + 0] = m;
      curr[cell + 2] = prev[cell + 0] * tfv[dr_tfv_idx(p7O_MI, q, lane, Q)]
                      + prev[cell + 2] * tfv[dr_tfv_idx(p7O_II, q, lane, Q)];
    }

    curr[1] = 0.0f;
    for (int c = 1; c < N; c++) {
      int pq = dr_cell_q(c - 1, Q);
      int plane = dr_cell_lane(c - 1, Q);
      curr[c * 3 + 1] = curr[(c - 1) * 3 + 0] * tfv[dr_tfv_idx(p7O_MD, pq, plane, Q)]
                       + curr[(c - 1) * 3 + 1] * tfv[dr_tfv_idx(p7O_DD, pq, plane, Q)];
    }

    for (int c = 0; c < N; c++) xE += curr[c * 3 + 0] + curr[c * 3 + 1];

    xN = xN * ploop;
    xC = (xC * ploop) + xE;
    xJ = 0.0f;
    xB = xN * pmove;

    if (xE > 1.0e4f) {
      float inv = 1.0f / xE;
      xN *= inv; xC *= inv; xJ *= inv; xB *= inv;
      for (int c = 0; c < N * 3; c++) curr[c] *= inv;
      totscale += logf(xE);
    }

    float *tmp = prev; prev = curr; curr = tmp;
  }

  if (isnan(xC) || (L > 0 && xC == 0.0f) || isinf(xC)) {
    scores_out[b] = 0.0f;
    statuses_out[b] = eslERANGE;
  } else {
    scores_out[b] = totscale + logf(xC * pmove);
    statuses_out[b] = eslOK;
  }
}

/* Helper: grow a device buffer if current alloc < needed bytes.
 * Frees old, allocates new. Updates *ptr and *alloc. */
static int
dom_grow_d(void **ptr, size_t *alloc, size_t needed, char *errbuf, int errbuf_size, const char *label)
{
  int status;
  if (*alloc >= needed) return eslOK;
  if (*ptr) cudaFree(*ptr);
  *ptr = NULL;
  *alloc = 0;
  if ((status = cuda_status(cudaMalloc(ptr, needed), errbuf, errbuf_size, label)) != eslOK) return status;
  *alloc = needed;
  return eslOK;
}

/* ================================================================
 * Host entry point: p7_cuda_DomainRescoreBatch
 * ================================================================ */
extern "C" int
p7_cuda_DomainRescoreBatch(P7_CUDA_ENGINE *engine,
                           const P7_CUDA_MSVPROFILE *cuom,
                           int ndomains,
                           const uint8_t **h_dsq_ptrs,
                           const int      *h_lengths,
                           const float   **h_rfv_ptrs,
                           const float    *h_orig_rfv,
                           int Q, int Kp, float nj,
                           float   *h_envsc,
                           float   *h_domcorrection,
                           float   *h_oasc,
                           int8_t  *h_trace_st,
                           int     *h_trace_k,
                           int     *h_trace_i,
                           float   *h_trace_pp,
                           int     *h_trace_N,
                           int      max_trace_len,
                           int     *h_statuses,
                           char    *errbuf, int errbuf_size)
{
  int status;
  int N = Q * 4;
  size_t row_cells = (size_t)N * 3;
  size_t shmem_bytes = row_cells * 2 * sizeof(float);
  size_t total_dp_cells = 0;
  size_t total_xmx = 0;
  size_t total_dsq = 0;
  size_t rfv_size = (size_t)Kp * Q * 4 * sizeof(float);
  size_t off;
  size_t trace_total;
  int b;

  float    **h_d_rfv_ptrs = NULL;
  uint8_t  **h_d_dsq_ptrs = NULL;
  size_t    *h_dp_offsets = NULL;
  size_t    *h_xmx_offsets = NULL;

  if (ndomains == 0) return eslOK;

  h_dp_offsets  = (size_t *)malloc(sizeof(size_t) * ndomains);
  h_xmx_offsets = (size_t *)malloc(sizeof(size_t) * ndomains);
  h_d_rfv_ptrs  = (float **)malloc(sizeof(float *) * ndomains);
  h_d_dsq_ptrs  = (uint8_t **)malloc(sizeof(uint8_t *) * ndomains);
  if (!h_dp_offsets || !h_xmx_offsets || !h_d_rfv_ptrs || !h_d_dsq_ptrs)
    ESL_XFAIL(eslEMEM, errbuf, "malloc host arrays");

  for (b = 0; b < ndomains; b++) {
    h_dp_offsets[b] = total_dp_cells;
    total_dp_cells += (size_t)(h_lengths[b] + 1) * row_cells;
    h_xmx_offsets[b] = total_xmx;
    total_xmx += (size_t)(h_lengths[b] + 1) * p7X_NXCELLS;
    total_dsq += h_lengths[b] + 2;
  }

  trace_total = (size_t)ndomains * max_trace_len;

  /* ---- Grow engine buffers as needed ---- */
  /* DP matrices (4 copies: fwd, bck, pp, oa) — all same size, grow together */
  {
    size_t dp_bytes = sizeof(float) * total_dp_cells;
    if (dp_bytes > engine->dom_dp_alloc) {
      for (int i = 0; i < 4; i++) { if (engine->d_dom_dpf[i]) { cudaFree(engine->d_dom_dpf[i]); engine->d_dom_dpf[i] = NULL; } }
      for (int i = 0; i < 4; i++) {
        if ((status = cuda_status(cudaMalloc(&engine->d_dom_dpf[i], dp_bytes), errbuf, errbuf_size, "grow dpf")) != eslOK) goto ERROR;
      }
      engine->dom_dp_alloc = dp_bytes;
    }
  }
  /* XMX (4 copies) — all same size, grow together */
  {
    size_t xmx_bytes = sizeof(float) * total_xmx;
    if (xmx_bytes > engine->dom_xmx_alloc) {
      for (int i = 0; i < 4; i++) { if (engine->d_dom_xmx[i]) { cudaFree(engine->d_dom_xmx[i]); engine->d_dom_xmx[i] = NULL; } }
      for (int i = 0; i < 4; i++) {
        if ((status = cuda_status(cudaMalloc(&engine->d_dom_xmx[i], xmx_bytes), errbuf, errbuf_size, "grow xmx")) != eslOK) goto ERROR;
      }
      engine->dom_xmx_alloc = xmx_bytes;
    }
  }
  /* Per-domain metadata */
  if (ndomains > engine->dom_meta_alloc) {
    if (engine->d_dom_dsq_ptrs)    { cudaFree(engine->d_dom_dsq_ptrs);    engine->d_dom_dsq_ptrs = NULL; }
    if (engine->d_dom_rfv_ptrs)    { cudaFree(engine->d_dom_rfv_ptrs);    engine->d_dom_rfv_ptrs = NULL; }
    if (engine->d_dom_lengths)     { cudaFree(engine->d_dom_lengths);     engine->d_dom_lengths = NULL; }
    if (engine->d_dom_dp_offsets)  { cudaFree(engine->d_dom_dp_offsets);  engine->d_dom_dp_offsets = NULL; }
    if (engine->d_dom_xmx_offsets) { cudaFree(engine->d_dom_xmx_offsets); engine->d_dom_xmx_offsets = NULL; }
    if (engine->d_dom_envsc)       { cudaFree(engine->d_dom_envsc);       engine->d_dom_envsc = NULL; }
    if (engine->d_dom_bcksc)       { cudaFree(engine->d_dom_bcksc);       engine->d_dom_bcksc = NULL; }
    if (engine->d_dom_oasc)        { cudaFree(engine->d_dom_oasc);        engine->d_dom_oasc = NULL; }
    if (engine->d_dom_domcorr)     { cudaFree(engine->d_dom_domcorr);     engine->d_dom_domcorr = NULL; }
    if (engine->d_dom_statuses)    { cudaFree(engine->d_dom_statuses);    engine->d_dom_statuses = NULL; }
    if (engine->d_dom_trace_N)     { cudaFree(engine->d_dom_trace_N);     engine->d_dom_trace_N = NULL; }
    int alloc_n = ndomains;
    if ((status = cuda_status(cudaMalloc(&engine->d_dom_dsq_ptrs,    sizeof(uint8_t *) * alloc_n), errbuf, errbuf_size, "malloc ptrs")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc(&engine->d_dom_rfv_ptrs,    sizeof(float *)   * alloc_n), errbuf, errbuf_size, "malloc ptrs")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc(&engine->d_dom_lengths,     sizeof(int)       * alloc_n), errbuf, errbuf_size, "malloc")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc(&engine->d_dom_dp_offsets,  sizeof(size_t)    * alloc_n), errbuf, errbuf_size, "malloc")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc(&engine->d_dom_xmx_offsets, sizeof(size_t)    * alloc_n), errbuf, errbuf_size, "malloc")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc(&engine->d_dom_envsc,       sizeof(float)     * alloc_n), errbuf, errbuf_size, "malloc")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc(&engine->d_dom_bcksc,       sizeof(float)     * alloc_n), errbuf, errbuf_size, "malloc")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc(&engine->d_dom_oasc,        sizeof(float)     * alloc_n), errbuf, errbuf_size, "malloc")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc(&engine->d_dom_domcorr,     sizeof(float)     * alloc_n), errbuf, errbuf_size, "malloc")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc(&engine->d_dom_statuses,    sizeof(int)       * alloc_n), errbuf, errbuf_size, "malloc")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc(&engine->d_dom_trace_N,    sizeof(int)       * alloc_n), errbuf, errbuf_size, "malloc")) != eslOK) goto ERROR;
    engine->dom_meta_alloc = alloc_n;
  }
  /* Trace buffers (per-element arrays keyed on ndomains * max_trace_len) */
  if (trace_total > engine->dom_trace_alloc) {
    if (engine->d_dom_trace_st) { cudaFree(engine->d_dom_trace_st); engine->d_dom_trace_st = NULL; }
    if (engine->d_dom_trace_k)  { cudaFree(engine->d_dom_trace_k);  engine->d_dom_trace_k = NULL; }
    if (engine->d_dom_trace_i)  { cudaFree(engine->d_dom_trace_i);  engine->d_dom_trace_i = NULL; }
    if (engine->d_dom_trace_pp) { cudaFree(engine->d_dom_trace_pp); engine->d_dom_trace_pp = NULL; }
    if ((status = cuda_status(cudaMalloc(&engine->d_dom_trace_st, sizeof(int8_t) * trace_total), errbuf, errbuf_size, "malloc trace")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc(&engine->d_dom_trace_k,  sizeof(int)    * trace_total), errbuf, errbuf_size, "malloc trace")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc(&engine->d_dom_trace_i,  sizeof(int)    * trace_total), errbuf, errbuf_size, "malloc trace")) != eslOK) goto ERROR;
    if ((status = cuda_status(cudaMalloc(&engine->d_dom_trace_pp, sizeof(float)  * trace_total), errbuf, errbuf_size, "malloc trace")) != eslOK) goto ERROR;
    engine->dom_trace_alloc = trace_total;
  }
  /* DSQ */
  if ((status = dom_grow_d((void **)&engine->d_dom_dsq_all, &engine->dom_dsq_alloc,
                           total_dsq, errbuf, errbuf_size, "grow dsq")) != eslOK) goto ERROR;
  /* RFV per-domain */
  if ((status = dom_grow_d((void **)&engine->d_dom_rfv_all, &engine->dom_rfv_alloc,
                           rfv_size * ndomains, errbuf, errbuf_size, "grow rfv")) != eslOK) goto ERROR;
  /* Original RFV */
  if ((status = dom_grow_d((void **)&engine->d_dom_orig_rfv, &engine->dom_orig_rfv_alloc,
                           rfv_size, errbuf, errbuf_size, "grow orig_rfv")) != eslOK) goto ERROR;

  /* ---- Upload data ---- */
  /* RFV per domain: stage into contiguous host buffer then single H2D */
  {
    uint8_t *h_rfv_staging = (uint8_t *)malloc(rfv_size * ndomains);
    if (!h_rfv_staging) ESL_XFAIL(eslEMEM, errbuf, "malloc rfv staging");
    for (b = 0; b < ndomains; b++) {
      memcpy(h_rfv_staging + rfv_size * b, h_rfv_ptrs[b], rfv_size);
      h_d_rfv_ptrs[b] = (float *)((char *)engine->d_dom_rfv_all + rfv_size * b);
    }
    if ((status = cuda_status(cudaMemcpy(engine->d_dom_rfv_all, h_rfv_staging,
                              rfv_size * ndomains, cudaMemcpyHostToDevice), errbuf, errbuf_size, "H2D rfv bulk")) != eslOK) { free(h_rfv_staging); goto ERROR; }
    free(h_rfv_staging);
  }

  /* DSQ per domain: stage into contiguous host buffer then single H2D */
  {
    uint8_t *h_dsq_staging = (uint8_t *)malloc(total_dsq);
    if (!h_dsq_staging) ESL_XFAIL(eslEMEM, errbuf, "malloc dsq staging");
    off = 0;
    for (b = 0; b < ndomains; b++) {
      size_t len = h_lengths[b] + 2;
      memcpy(h_dsq_staging + off, h_dsq_ptrs[b], len);
      h_d_dsq_ptrs[b] = engine->d_dom_dsq_all + off;
      off += len;
    }
    if ((status = cuda_status(cudaMemcpy(engine->d_dom_dsq_all, h_dsq_staging,
                              total_dsq, cudaMemcpyHostToDevice), errbuf, errbuf_size, "H2D dsq bulk")) != eslOK) { free(h_dsq_staging); goto ERROR; }
    free(h_dsq_staging);
  }

  /* Upload pointer tables, metadata */
  if ((status = cuda_status(cudaMemcpy((void *)engine->d_dom_dsq_ptrs, h_d_dsq_ptrs, sizeof(uint8_t *) * ndomains, cudaMemcpyHostToDevice), errbuf, errbuf_size, "H2D")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy((void *)engine->d_dom_rfv_ptrs, h_d_rfv_ptrs, sizeof(float *)   * ndomains, cudaMemcpyHostToDevice), errbuf, errbuf_size, "H2D")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_dom_lengths,     h_lengths,     sizeof(int)    * ndomains, cudaMemcpyHostToDevice), errbuf, errbuf_size, "H2D")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_dom_dp_offsets,  h_dp_offsets,  sizeof(size_t) * ndomains, cudaMemcpyHostToDevice), errbuf, errbuf_size, "H2D")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(engine->d_dom_xmx_offsets, h_xmx_offsets, sizeof(size_t) * ndomains, cudaMemcpyHostToDevice), errbuf, errbuf_size, "H2D")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemset(engine->d_dom_statuses, 0, sizeof(int) * ndomains), errbuf, errbuf_size, "memset")) != eslOK) goto ERROR;

  /* Original rfv for domcorrection */
  if ((status = cuda_status(cudaMemcpy(engine->d_dom_orig_rfv, h_orig_rfv, rfv_size, cudaMemcpyHostToDevice), errbuf, errbuf_size, "H2D")) != eslOK) goto ERROR;

  /* ---- Kernel 1: Full Forward ---- */
  cuda_domain_fwd_full_kernel<<<ndomains, 1, shmem_bytes>>>(
    engine->d_dom_dsq_ptrs, engine->d_dom_lengths, engine->d_dom_rfv_ptrs, cuom->d_tfv,
    Q, Kp, nj, engine->d_dom_dp_offsets, engine->d_dom_xmx_offsets,
    engine->d_dom_dpf[0], engine->d_dom_xmx[0], engine->d_dom_envsc, engine->d_dom_statuses);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "fwd kernel")) != eslOK) goto ERROR;

  /* ---- Kernel 2: Full Backward ---- */
  cuda_domain_bck_full_kernel<<<ndomains, 1, shmem_bytes>>>(
    engine->d_dom_dsq_ptrs, engine->d_dom_lengths, engine->d_dom_rfv_ptrs, cuom->d_tfv,
    Q, Kp, nj, engine->d_dom_dp_offsets, engine->d_dom_xmx_offsets,
    engine->d_dom_dpf[1], engine->d_dom_xmx[0], engine->d_dom_xmx[1], engine->d_dom_bcksc, engine->d_dom_statuses);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "bck kernel")) != eslOK) goto ERROR;

  /* ---- Kernel 3: Posterior Decoding ---- */
  cuda_domain_decoding_kernel<<<ndomains, 1>>>(
    engine->d_dom_lengths, Q, engine->d_dom_dp_offsets, engine->d_dom_xmx_offsets,
    engine->d_dom_dpf[0], engine->d_dom_dpf[1], engine->d_dom_dpf[2],
    engine->d_dom_xmx[0], engine->d_dom_xmx[1], engine->d_dom_xmx[2], nj, engine->d_dom_statuses);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "decoding kernel")) != eslOK) goto ERROR;

  /* ---- Kernel 4: Optimal Accuracy ---- */
  cuda_domain_optacc_kernel<<<ndomains, 1>>>(
    engine->d_dom_lengths, Q, cuom->d_tfv, engine->d_dom_dp_offsets, engine->d_dom_xmx_offsets,
    engine->d_dom_dpf[2], engine->d_dom_xmx[2], engine->d_dom_dpf[3], engine->d_dom_xmx[3],
    engine->d_dom_oasc, nj);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "optacc kernel")) != eslOK) goto ERROR;

  /* ---- Kernel 5: OA Traceback ---- */
  cuda_domain_oatrace_kernel<<<ndomains, 1>>>(
    engine->d_dom_lengths, Q, cuom->M, cuom->d_tfv, engine->d_dom_dp_offsets, engine->d_dom_xmx_offsets,
    engine->d_dom_dpf[2], engine->d_dom_xmx[2], engine->d_dom_dpf[3], engine->d_dom_xmx[3], nj,
    engine->d_dom_trace_st, engine->d_dom_trace_k, engine->d_dom_trace_i,
    engine->d_dom_trace_pp, engine->d_dom_trace_N, max_trace_len);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "oatrace kernel")) != eslOK) goto ERROR;

  /* ---- Kernel 6: Domcorrection Forward ---- */
  cuda_domain_fwd_scoreonly_kernel<<<ndomains, 1, shmem_bytes>>>(
    engine->d_dom_dsq_ptrs, engine->d_dom_lengths, engine->d_dom_orig_rfv, cuom->d_tfv,
    Q, Kp, nj, engine->d_dom_domcorr, engine->d_dom_statuses);
  if ((status = cuda_status(cudaGetLastError(), errbuf, errbuf_size, "domcorr kernel")) != eslOK) goto ERROR;

  /* Synchronize and D2H */
  if ((status = cuda_status(cudaDeviceSynchronize(), errbuf, errbuf_size, "sync")) != eslOK) goto ERROR;

  if ((status = cuda_status(cudaMemcpy(h_envsc, engine->d_dom_envsc, sizeof(float) * ndomains, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "D2H")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(h_domcorrection, engine->d_dom_domcorr, sizeof(float) * ndomains, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "D2H")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(h_oasc, engine->d_dom_oasc, sizeof(float) * ndomains, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "D2H")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(h_statuses, engine->d_dom_statuses, sizeof(int) * ndomains, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "D2H")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(h_trace_st, engine->d_dom_trace_st, sizeof(int8_t) * ndomains * max_trace_len, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "D2H")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(h_trace_k, engine->d_dom_trace_k, sizeof(int) * ndomains * max_trace_len, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "D2H")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(h_trace_i, engine->d_dom_trace_i, sizeof(int) * ndomains * max_trace_len, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "D2H")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(h_trace_pp, engine->d_dom_trace_pp, sizeof(float) * ndomains * max_trace_len, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "D2H")) != eslOK) goto ERROR;
  if ((status = cuda_status(cudaMemcpy(h_trace_N, engine->d_dom_trace_N, sizeof(int) * ndomains, cudaMemcpyDeviceToHost), errbuf, errbuf_size, "D2H")) != eslOK) goto ERROR;

  status = eslOK;

 ERROR:
  free(h_dp_offsets);
  free(h_xmx_offsets);
  free(h_d_rfv_ptrs);
  free(h_d_dsq_ptrs);
  return status;
}
