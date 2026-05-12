/* nhmmer GPU pipeline: public outer loop (async 2-slot strand pipeline).
 *
 * Iterates over the target .nucdb (overlap geometry required, enforced here)
 * and drives a 2-slot double-buffered pipeline: the main thread runs the GPU
 * pipeline for strand N+1 while strand N's CPU workers finish in background
 * threads. At the end of each iteration the next slot (the one we're about
 * to reuse) is retired, merging its hits/pipeline into the shared info
 * structs before we touch them again.
 *
 * Plain C (no .cu); compiled by the host compiler. It #includes
 * cuda/p7_cuda.h and calls the kernel-dispatch APIs declared there.
 */
#include <p7_config.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

#ifdef HMMER_THREADS
#include <pthread.h>
#endif

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_sq.h"
#include "esl_vectorops.h"

#include "hmmer.h"
#include "p7_nucdb.h"
#include "nhmmer_internal.h"

#ifdef HMMER_CUDA
#include "cuda/p7_cuda.h"
#include "cuda/nhmmer_cuda_internal.h"


char nhmmer_empty_str[] = "";


int
nhmmer_gpu_nucdb_upload(NHMMER_GPU_INFO *info, P7_NUCDB *ndb,
                        char *errbuf, int errbuf_size)
{
  int64_t nucdb_data_size;

  if (!info || !ndb) return eslEINVAL;
  if (p7_cuda_engine_NucdbDevPtr(info->cuda_engine) != NULL) {
    info->nucdb_resident = TRUE;
    return eslOK;
  }

  /* Upload only the packed region (2-bit residues). The mask region
   * stays in the host mmap and is consulted only by CPU slice-fill. */
  if (ndb->hdr.mask_offset <= ndb->hdr.data_offset)
    return eslEFORMAT;
  nucdb_data_size = (int64_t)(ndb->hdr.mask_offset - ndb->hdr.data_offset);
  int status = p7_cuda_engine_UploadNucdb(info->cuda_engine, ndb->chunk_data, nucdb_data_size,
                                          errbuf, errbuf_size);
  if (status == eslOK) info->nucdb_resident = TRUE;
  return status;
}


/* Submit one strand through the slot pipeline. Rotates the slot index.
 * Responsibilities:
 *   1. Retire the slot we're about to reuse (joins prior workers, merges
 *      hits + pli into info).
 *   2. Install the shell ESL_SQ (caller-owned; the slot takes ownership only
 *      when take_ownership_of_sq == TRUE, meaning this is the LAST strand
 *      that uses it; otherwise we pass a clone or leave freeing to caller).
 *   3. Run the GPU pipeline.
 *   4. Launch workers (background). Slot now in flight; main thread returns.
 *
 * On error, the slot is left not-in-flight (either retired cleanly or
 * never entered flight), and the slot's shell sq is freed.
 */
static int
submit_strand(NHMMER_GPU_SLOT *slots, int *slot_idx_p,
              NHMMER_GPU_INFO *info, const P7_NUCDB *ndb,
              ESL_SQ *sq_shell_owned, int64_t seq_id, int complementarity,
              int chunk_start, int chunk_count,
              char *errbuf, int errbuf_size)
{
  int status;
  NHMMER_GPU_SLOT *slot = &slots[*slot_idx_p];

  status = nhmmer_gpu_slot_retire(slot, info);
  if (status != eslOK) {
    /* Prior slot had an error. We still own the new shell sq; free it. */
    if (sq_shell_owned) esl_sq_Destroy(sq_shell_owned);
    return status;
  }
  nhmmer_gpu_slot_reset(slot);

  slot->sq_shell        = sq_shell_owned;
  slot->ndb             = ndb;
  slot->seq_id          = seq_id;
  slot->complementarity = complementarity;

  status = nhmmer_gpu_run_strand_gpu_phase(info, slot, ndb,
                                           chunk_start, chunk_count,
                                           complementarity, errbuf, errbuf_size);
  if (status != eslOK) {
    /* Tear down any partial state (sq_shell owned by slot) */
    nhmmer_gpu_slot_reset(slot);
    return status;
  }

  if (!slot->has_work) {
    /* No FB survivors; nothing to dispatch. Free the shell and move on. */
    nhmmer_gpu_slot_reset(slot);
    *slot_idx_p = 1 - *slot_idx_p;
    return eslOK;
  }

  status = nhmmer_gpu_slot_launch_workers(slot, info);
  if (status != eslOK) {
    nhmmer_gpu_slot_reset(slot);
    return status;
  }

  *slot_idx_p = 1 - *slot_idx_p;
  return eslOK;
}


int
nhmmer_gpu_nucdb_loop(NHMMER_GPU_INFO *info, P7_NUCDB *ndb,
                      int strands, NHMMER_GPU_IDLEN_CB idlen_cb, void *idlen_data,
                      int *ret_nseqs, int64_t *ret_nres)
{
  int       status = eslOK;
  int64_t   nres   = 0;
  char      errbuf[eslERRBUFSIZE];
  P7_OPROFILE *om = info->om;
  P7_BG       *bg = info->bg;

  p7_bg_SetLength(bg, om->max_length);
  p7_oprofile_ReconfigMSVLength(om, om->max_length);
  p7_cuda_msvprofile_UpdateLength(info->cuda_msv, om, om->max_length, errbuf, sizeof(errbuf));

  status = nhmmer_gpu_nucdb_upload(info, ndb, errbuf, sizeof(errbuf));
  if (status != eslOK) {
    fprintf(stderr, "GPU nhmmer: failed to upload nucdb: %s\n", errbuf);
    return status;
  }

  int chunk_size = info->gpu_chunk_size > 0 ? info->gpu_chunk_size : NHMMER_GPU_CHUNK_SIZE;
  if (p7_cuda_engine_NucdbDevPtr(info->cuda_engine) == NULL ||
      (int64_t)ndb->hdr.overlap < (int64_t)om->max_length ||
      (int64_t)ndb->hdr.chunk_size != (int64_t)chunk_size)
  {
    fprintf(stderr,
            "GPU nhmmer requires an overlap .nucdb with chunk_size=%d overlap>=%d "
            "(got chunk_size=%" PRId64 " overlap=%" PRId64 "). "
            "Rebuild with `hmmnucdb` (defaults to --overlap 2001).\n",
            chunk_size, (int)om->max_length,
            (int64_t)ndb->hdr.chunk_size, (int64_t)ndb->hdr.overlap);
    return eslEINCOMPAT;
  }

  /* PHASE 1 GATE: GPU kernels have not yet been ported to the v2
   * 2-bit packed .nucdb layout. The host-side driver (this file) and
   * CPU slice-fill have been updated, but the device kernels still
   * assume byte-per-residue bytes. Running them would silently produce
   * wrong scores. Fail cleanly until Phase 2 lands. */
  fprintf(stderr,
          "GPU nhmmer: .nucdb v2 (2-bit packed) is not yet supported on the "
          "device side. Run with the CPU path for now; the GPU kernel port "
          "is tracked as Phase 2 of the v2 .nucdb migration.\n");
  return eslEINCOMPAT;

  /* Initialize the 2-slot ring. */
  NHMMER_GPU_SLOT slots[2];
  memset(slots, 0, sizeof(slots));

  int nworkers_max = info->ncpus;
  if (nworkers_max < 1) nworkers_max = 1;

  status = nhmmer_gpu_slot_init(&slots[0], info, nworkers_max);
  if (status != eslOK) return status;
  status = nhmmer_gpu_slot_init(&slots[1], info, nworkers_max);
  if (status != eslOK) { nhmmer_gpu_slot_destroy(&slots[0]); return status; }

  int slot_idx = 0;

  for (int64_t si = 0; si < (int64_t)ndb->hdr.nseq; si++) {
    P7_NUCDB_SEQ_IDX *sidx = &ndb->seq_idx[si];

    /* p7_pli_NewSeq touches info->pli. Before we mutate it, make sure both
     * slots' workers have finished merging into info->pli. Because the
     * loop only keeps ONE slot in flight at a time when we reach this
     * point (the outer iteration cycle submits fwd then rc, each of which
     * may overlap with the previously-submitted strand), we just need to
     * retire the next slot we'll submit into — any other in-flight slot
     * corresponds to the OTHER complementarity of the SAME seq si and
     * does not conflict with p7_pli_NewSeq. But to keep pipeline stats
     * consistent we conservatively retire whichever slot we're about to
     * reuse (slots[slot_idx]) before updating info->pli. */
    status = nhmmer_gpu_slot_retire(&slots[slot_idx], info);
    if (status != eslOK) goto ERROR;

    /* Build the shell sq. It gets referenced by workers for slice
     * materialization. One shell serves both strands (workers only read
     * name/length metadata and the complementarity flag). */
    ESL_SQ *sq_shell_fwd = NULL;
    status = nhmmer_gpu_nucdb_create_seq_shell(ndb, om->abc, si, &sq_shell_fwd);
    if (status != eslOK) goto ERROR;

    p7_pli_NewSeq(info->pli, sq_shell_fwd);

    /* Forward strand */
    if (strands != p7_STRAND_BOTTOMONLY) {
      nres += sq_shell_fwd->n;
      status = submit_strand(slots, &slot_idx, info, ndb,
                             sq_shell_fwd, si, p7_NOCOMPLEMENT,
                             sidx->chunk_start, sidx->chunk_count,
                             errbuf, sizeof(errbuf));
      if (status != eslOK) {
        fprintf(stderr, "GPU nhmmer nucdb forward strand failed: %s\n", errbuf);
        goto ERROR;
      }
      sq_shell_fwd = NULL;  /* slot owns it (or it's been freed on has_work=0) */
    } else {
      esl_sq_Destroy(sq_shell_fwd);
      sq_shell_fwd = NULL;
    }

    /* Reverse complement strand — reads the SAME forward chunk range; the
     * GPU kernels and slice-fill reverse the index and complement the
     * 2-bit code on the fly. Build its own shell (the forward one may
     * still be referenced by workers of the previous slot). */
    if (strands != p7_STRAND_TOPONLY && sidx->chunk_count > 0) {
      ESL_SQ *sq_shell_rc = NULL;
      status = nhmmer_gpu_nucdb_create_seq_shell(ndb, om->abc, si, &sq_shell_rc);
      if (status != eslOK) goto ERROR;
      nres += sq_shell_rc->n;

      status = submit_strand(slots, &slot_idx, info, ndb,
                             sq_shell_rc, si, p7_COMPLEMENT,
                             sidx->chunk_start, sidx->chunk_count,
                             errbuf, sizeof(errbuf));
      if (status != eslOK) {
        fprintf(stderr, "GPU nhmmer nucdb revcomp strand failed: %s\n", errbuf);
        goto ERROR;
      }
    }

    /* p7_pipeline_Reuse must happen AFTER all strands of seq si have
     * either run to completion or been handed off. Background workers
     * already merged into info->pli inside slot_retire (which we run at
     * the top of each iteration). The only strand potentially still in
     * flight is the one we just submitted; its merge will happen on the
     * next iteration's slot_retire. Reuse/nseqs++ here would race with
     * that merge, so defer them: we Reuse only if no slots are in flight.
     */
    int any_in_flight = (slots[0].in_flight || slots[1].in_flight);
    if (!any_in_flight) {
      p7_pipeline_Reuse(info->pli);
    }
    if (idlen_cb) idlen_cb(idlen_data, si, sidx->length);
    info->pli->nseqs++;
  }

  /* Drain any remaining in-flight slots */
  for (int i = 0; i < 2; i++) {
    status = nhmmer_gpu_slot_retire(&slots[i], info);
    if (status != eslOK) goto ERROR;
  }

  *ret_nseqs = (int)ndb->hdr.nseq;
  *ret_nres  = nres;

  nhmmer_gpu_slot_destroy(&slots[0]);
  nhmmer_gpu_slot_destroy(&slots[1]);
  return eslOK;

ERROR:
  /* Drain before destroying so we don't abort in destroy(). */
  for (int i = 0; i < 2; i++) {
    (void) nhmmer_gpu_slot_retire(&slots[i], info);
  }
  nhmmer_gpu_slot_destroy(&slots[0]);
  nhmmer_gpu_slot_destroy(&slots[1]);
  return status;
}


#endif /* HMMER_CUDA */
