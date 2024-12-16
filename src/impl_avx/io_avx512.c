/* Saving optimized profiles in two pieces: MSV part and the rest.
 * 
 * To accelerate hmmscan, which is limited by speed of HMM input,
 * hmmpress saves an optimized profile in two pieces. One file gets
 * a bare minimum of information needed to run the MSV filter.
 * The other file gets the rest of the profile. Both files are binary,
 * stored exactly as the <P7_OPROFILE> has the information internally.
 * 
 * By convention, hmmpress calls the two files <hmmfile>.h3f and
 * <hmmfile>.h3p, which nominally stand for "H3 filter" and "H3
 * profile".
 * 
 * Contents:
 *    1. Writing optimized profiles to two files.
 *    2. Reading optimized profiles in two stages.
 *    3. Utility routines.
 *    4. Benchmark driver.
 *    5. Unit tests.
 *    6. Test driver.
 *    7. Example.
 *    
 * TODO:
 *    - crossplatform binary compatibility (endedness and off_t)
 *    - Write() could save a tag (model #) instead of name for verifying
 *      that MSV and Rest parts match, saving a malloc for var-lengthed name
 *      in ReadRest().
 */
#include <p7_config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HMMER_THREADS
#include <pthread.h>
#endif

#include "easel.h"

#include "hmmer.h"
#include "impl_avx.h"
#ifdef eslENABLE_AVX512
#include <x86intrin.h>
#include "esl_avx512.h"
#endif
#ifdef eslENABLE_AVX512
static uint32_t  v3g_fmagic = 0xb3e7e6f3; /* 3/f binary MSV file, SSE:     "3gfs" = 0x 33 67 66 73  + 0x80808080 */
static uint32_t  v3g_pmagic = 0xb3e7f0f3; /* 3/f binary profile file, SSE: "3gps" = 0x 33 67 70 73  + 0x80808080 */

static uint32_t  v3f_fmagic = 0xb3e6e6f3; /* 3/f binary MSV file, SSE:     "3ffs" = 0x 33 66 66 73  + 0x80808080 */
static uint32_t  v3f_pmagic = 0xb3e6f0f3; /* 3/f binary profile file, SSE: "3fps" = 0x 33 66 70 73  + 0x80808080 */

static uint32_t  v3e_fmagic = 0xb3e5e6f3; /* 3/e binary MSV file, SSE:     "3efs" = 0x 33 65 66 73  + 0x80808080 */
static uint32_t  v3e_pmagic = 0xb3e5f0f3; /* 3/e binary profile file, SSE: "3eps" = 0x 33 65 70 73  + 0x80808080 */

static uint32_t  v3d_fmagic = 0xb3e4e6f3; /* 3/d binary MSV file, SSE:     "3dfs" = 0x 33 64 66 73  + 0x80808080 */
static uint32_t  v3d_pmagic = 0xb3e4f0f3; /* 3/d binary profile file, SSE: "3dps" = 0x 33 64 70 73  + 0x80808080 */

static uint32_t  v3c_fmagic = 0xb3e3e6f3; /* 3/c binary MSV file, SSE:     "3cfs" = 0x 33 63 66 73  + 0x80808080 */
static uint32_t  v3c_pmagic = 0xb3e3f0f3; /* 3/c binary profile file, SSE: "3cps" = 0x 33 63 70 73  + 0x80808080 */

static uint32_t  v3b_fmagic = 0xb3e2e6f3; /* 3/b binary MSV file, SSE:     "3bfs" = 0x 33 62 66 73  + 0x80808080 */
static uint32_t  v3b_pmagic = 0xb3e2f0f3; /* 3/b binary profile file, SSE: "3bps" = 0x 33 62 70 73  + 0x80808080 */

static uint32_t  v3a_fmagic = 0xe8b3e6f3; /* 3/a binary MSV file, SSE:     "h3fs" = 0x 68 33 66 73  + 0x80808080 */
static uint32_t  v3a_pmagic = 0xe8b3f0f3; /* 3/a binary profile file, SSE: "h3ps" = 0x 68 33 70 73  + 0x80808080 */

/*****************************************************************
 *# 1. Writing optimized profiles to two files.
 *****************************************************************/

/* Function:  p7_oprofile_Write()
 * Synopsis:  Write an optimized profile in two files.
 *
 * Purpose:   Write the MSV filter part of <om> to open binary stream
 *            <ffp>, and the rest of the model to <pfp>. These two
 *            streams will typically be <.h3f> and <.h3p> files 
 *            being created by hmmpress.
 *
 * Args:      ffp  - open binary stream for saving MSV filter part
 *            pfp  - open binary stream for saving rest of profile
 *            om   - optimized profile to save
 *
 * Returns:   <eslOK> on success.
 *
 * Throws:    <eslEWRITE> on any write failure, such as filling
 *            the disk.
 */

int
p7_oprofile_Write_avx512(P7_HMMFILE *hfp, P7_OPROFILE *om)
{
  int Q16  = p7O_NQB(om->M);
  int Q16x = p7O_NQB(om->M) + p7O_EXTRA_SB;
  int Q4_AVX   = p7O_NQF_AVX(om->M);
  int Q8_AVX   = p7O_NQW_AVX(om->M);
  int Q16_AVX  = p7O_NQB_AVX(om->M);
  int Q16x_AVX = p7O_NQB_AVX(om->M) + p7O_EXTRA_SB;
  int Q4_AVX512   = p7O_NQF_AVX512(om->M);
  int Q8_AVX512   = p7O_NQW_AVX512(om->M);
  int Q16_AVX512  = p7O_NQB_AVX512(om->M);
  int Q16x_AVX512 = p7O_NQB_AVX512(om->M) + p7O_EXTRA_SB;
  int n    = strlen(om->name);
  int x;
  int status;
  /* <hfp->ffp> is the part of the oprofile that MSVFilter() needs */
  //128-bit version
  if (fwrite((char *) &(v3g_fmagic),    sizeof(uint32_t), 1,           hfp->ffp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->M),         sizeof(int),      1,           hfp->ffp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->abc->type), sizeof(int),      1,           hfp->ffp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &n,               sizeof(int),      1,           hfp->ffp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) om->name,         sizeof(char),     n+1,         hfp->ffp) != n+1)         ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->max_length),sizeof(int),      1,           hfp->ffp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->tbm_b),     sizeof(uint8_t),  1,           hfp->ffp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->tec_b),     sizeof(uint8_t),  1,           hfp->ffp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->tjb_b),     sizeof(uint8_t),  1,           hfp->ffp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->scale_b),   sizeof(float),    1,           hfp->ffp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->base_b),    sizeof(uint8_t),  1,           hfp->ffp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->bias_b),    sizeof(uint8_t),  1,           hfp->ffp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  

  // 256-bit version
  if (fwrite((char *) &(v3g_fmagic),    sizeof(uint32_t), 1,           hfp->ffp256) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->M),         sizeof(int),      1,           hfp->ffp256) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->abc->type), sizeof(int),      1,           hfp->ffp256) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &n,               sizeof(int),      1,           hfp->ffp256) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) om->name,         sizeof(char),     n+1,         hfp->ffp256) != n+1)         ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->max_length),sizeof(int),      1,           hfp->ffp256) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->tbm_b),     sizeof(uint8_t),  1,           hfp->ffp256) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->tec_b),     sizeof(uint8_t),  1,           hfp->ffp256) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->tjb_b),     sizeof(uint8_t),  1,           hfp->ffp256) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->scale_b),   sizeof(float),    1,           hfp->ffp256) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->base_b),    sizeof(uint8_t),  1,           hfp->ffp256) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->bias_b),    sizeof(uint8_t),  1,           hfp->ffp256) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  

  //512-bit version
  if (fwrite((char *) &(v3g_fmagic),    sizeof(uint32_t), 1,           hfp->ffp512) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->M),         sizeof(int),      1,           hfp->ffp512) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->abc->type), sizeof(int),      1,           hfp->ffp512) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &n,               sizeof(int),      1,           hfp->ffp512) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) om->name,         sizeof(char),     n+1,         hfp->ffp512) != n+1)         ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->max_length),sizeof(int),      1,           hfp->ffp512) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->tbm_b),     sizeof(uint8_t),  1,           hfp->ffp512) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->tec_b),     sizeof(uint8_t),  1,           hfp->ffp512) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->tjb_b),     sizeof(uint8_t),  1,           hfp->ffp512) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->scale_b),   sizeof(float),    1,           hfp->ffp512) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->base_b),    sizeof(uint8_t),  1,           hfp->ffp512) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->bias_b),    sizeof(uint8_t),  1,           hfp->ffp512) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  

  uint8_t *buffer, *rbv_scalar, **rbv_unstriped;
  int buffer_size;
  // figure out the size of the largest buffer we'll use
  // sizeof(__m128i) * 4 = sizeof(__m512i) without requiring AVX-512 headers
  if((sizeof(__m128i)*4 * Q4_AVX512 * p7O_NTRANS )> (Q16x_AVX512 * sizeof(__m128i) * 4)){
    buffer_size = sizeof(__m128i)*4 * Q4_AVX512 * p7O_NTRANS ;
  }
  else{
    buffer_size = Q16x_AVX512 * sizeof(__m128i) * 4;
  }
  ESL_ALLOC(buffer, buffer_size); // should be enough space for any of the 
  
  ESL_ALLOC(rbv_unstriped, om->abc->Kp * sizeof(uint8_t *));
  for(int v = 0; v < om->abc->Kp; v++){
    ESL_ALLOC(rbv_unstriped[v], om->M * sizeof(uint8_t));
  }
  //structures

// First, the SSV (sbv) data
for (x = 0; x < om->abc->Kp; x++){
  rbv_scalar = (uint8_t *) om->sbv_avx512[x];
  //unstripe the sbv data.  Only unstripe the first M positions, we'll fix the padding rows later
  for(int hmmpos = 0; hmmpos < om->M; hmmpos++){
    int vector = hmmpos % Q16_AVX512;
    int within_vector = hmmpos /Q16_AVX512;
    int index = (vector * sizeof(__m512i))+within_vector;
    rbv_unstriped[x][hmmpos] = rbv_scalar[index];
  }
}
// Create and write 128-bit striped version
for (x = 0; x < om->abc->Kp; x++){ // First the first M positions and padding out to even vector
    int hmmpos;
    for(hmmpos = 0; hmmpos < om->M; hmmpos++){
      int vector = hmmpos % Q16;
      int within_vector = hmmpos /Q16;
      int index = (vector * sizeof(__m128i))+within_vector;
      buffer[index] = rbv_unstriped[x][hmmpos];
    }
    for(int padding = hmmpos; padding < Q16 * sizeof(__m128i); padding++){
      int vector = padding % Q16;
      int within_vector = padding /Q16;
      int index = (vector * sizeof(__m128i))+within_vector;
      buffer[index] = 127;  // value for elements of vector that don't correspond to HMM
      //positions
    }

    // Then the "wraparound" rows
    for(int wraparound_vector = Q16; wraparound_vector < Q16x; wraparound_vector++){
      for(int within_vector = 0; within_vector < sizeof(__m128i); within_vector++){
        buffer[wraparound_vector * sizeof(__m128i)+ within_vector] = buffer[(wraparound_vector %Q16)*sizeof(__m128i)+within_vector];
      }
    }

    if (fwrite( (char *) buffer, 1,  Q16x * sizeof(__m128i), hfp->ffp) != Q16x*sizeof(__m128i))         ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  }

//256-bit version
for (x = 0; x < om->abc->Kp; x++){ // First the first M positions and padding out to even vector
    int hmmpos;
    for(hmmpos = 0; hmmpos < om->M; hmmpos++){
      int vector = hmmpos % Q16_AVX;
      int within_vector = hmmpos /Q16_AVX;
      int index = (vector * sizeof(__m128i)*2)+within_vector;
      buffer[index] = rbv_unstriped[x][hmmpos];
    }
    for(int padding = hmmpos; padding < Q16_AVX * sizeof(__m128i)*2; padding++){
      int vector = padding % Q16_AVX;
      int within_vector = padding /Q16_AVX;
      int index = (vector * sizeof(__m128i)*2)+within_vector;
      buffer[index] = 127;  // value for elements of vector that don't correspond to HMM
      //positions
    }

    // Then the "wraparound" rows
    for(int wraparound_vector = Q16_AVX; wraparound_vector < Q16x_AVX; wraparound_vector++){
      for(int within_vector = 0; within_vector < sizeof(__m128i)*2; within_vector++){
        buffer[wraparound_vector * sizeof(__m128i)*2+ within_vector] = buffer[(wraparound_vector %Q16_AVX)*sizeof(__m128i)*2+within_vector];
      }
    }

    if (fwrite( (char *) buffer, 1,  Q16x_AVX * sizeof(__m128i)*2, hfp->ffp256) != Q16x_AVX*sizeof(__m128i)*2)         ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  }
//512-bit version
for (x = 0; x < om->abc->Kp; x++){ // First the first M positions and padding out to even vector
    int hmmpos;
    for(hmmpos = 0; hmmpos < om->M; hmmpos++){
      int vector = hmmpos % Q16_AVX512;
      int within_vector = hmmpos /Q16_AVX512;
      int index = (vector * sizeof(__m128i)*4)+within_vector;
      buffer[index] = rbv_unstriped[x][hmmpos];
    }
    for(int padding = hmmpos; padding < Q16_AVX512 * sizeof(__m128i)*4; padding++){
      int vector = padding % Q16_AVX512;
      int within_vector = padding /Q16_AVX512;
      int index = (vector * sizeof(__m128i)*4)+within_vector;
      buffer[index] = 127;  // value for elements of vector that don't correspond to HMM
      //positions
    }

    // Then the "wraparound" rows
    for(int wraparound_vector = Q16_AVX512; wraparound_vector < Q16x_AVX512; wraparound_vector++){
      for(int within_vector = 0; within_vector < sizeof(__m128i)*4; within_vector++){
        buffer[wraparound_vector * sizeof(__m128i)*4+ within_vector] = buffer[(wraparound_vector %Q16_AVX512)*sizeof(__m128i)*4+within_vector];
      }
    }

    if (fwrite( (char *) buffer, 1,  Q16x_AVX512 * sizeof(__m128i)*4, hfp->ffp512) != Q16x_AVX512*sizeof(__m128i)*4)         ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  }

// Next, the MSV (rbv) data
for (x = 0; x < om->abc->Kp; x++){
    rbv_scalar = (uint8_t *) om->rbv_avx512[x];
    //unstripe the rbv data
    for(int hmmpos = 0; hmmpos < om->M; hmmpos++){
      int vector = hmmpos % Q16_AVX512;
      int within_vector = hmmpos /Q16_AVX512;
      int index = (vector * sizeof(__m512i))+within_vector;
      rbv_unstriped[x][hmmpos] = rbv_scalar[index];
    }
  }
  // Write 128-bit striped version
  for (x = 0; x < om->abc->Kp; x++){
    int hmmpos;
    for(hmmpos = 0; hmmpos < om->M; hmmpos++){
      int vector = hmmpos % Q16;
      int within_vector = hmmpos /Q16;
      int index = (vector * sizeof(__m128i))+within_vector;
      buffer[index] = rbv_unstriped[x][hmmpos];
    }
    for(int padding = hmmpos; padding < Q16 * sizeof(__m128i); padding++){
      int vector = padding % Q16;
      int within_vector = padding /Q16;
      int index = (vector * sizeof(__m128i))+within_vector;
      buffer[index] = 255;  // value for elements of vector that don't correspond to HMM
      //positions
    }
    if (fwrite( (char *) buffer, 1,  Q16 * sizeof(__m128i), hfp->ffp) != Q16*sizeof(__m128i))         ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  }

  //Now, 256-bit striped version
  for (x = 0; x < om->abc->Kp; x++){
    int hmmpos;
    for(hmmpos = 0; hmmpos < om->M; hmmpos++){
      int vector = hmmpos % Q16_AVX;
      int within_vector = hmmpos /Q16_AVX;
      int index = (vector * sizeof(__m128i)*2)+within_vector;
      buffer[index] = rbv_unstriped[x][hmmpos];
    }
    for(int padding = hmmpos; padding < Q16_AVX * sizeof(__m128i) *2; padding++){
      int vector = padding % Q16_AVX;
      int within_vector = padding /Q16_AVX;
      int index = (vector * sizeof(__m128i)*2)+within_vector;
      buffer[index] = 255;  // value for elements of vector that don't correspond to HMM
      //positions
    }
    if (fwrite( (char *) buffer, 1,  Q16_AVX * sizeof(__m128i) * 2, hfp->ffp256) != Q16_AVX * sizeof(__m128i) * 2)         ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  }

  //then, 512-bit striped version
  for (x = 0; x < om->abc->Kp; x++){
    int hmmpos;
    for(hmmpos = 0; hmmpos < om->M; hmmpos++){
      int vector = hmmpos % Q16_AVX512;
      int within_vector = hmmpos /Q16_AVX512;
      int index = (vector * sizeof(__m128i)*4)+within_vector;
      buffer[index] = rbv_unstriped[x][hmmpos];
    }
    for(int padding = hmmpos; padding < Q16_AVX512 * sizeof(__m128i) *4; padding++){
      int vector = padding % Q16_AVX512;
      int within_vector = padding /Q16_AVX512;
      int index = (vector * sizeof(__m128i)*4)+within_vector;
      buffer[index] = 255;  // value for elements of vector that don't correspond to HMM
      //positions
    }
    if (fwrite( (char *) buffer, 1,  Q16_AVX512* sizeof(__m128i)*4, hfp->ffp512) !=  Q16_AVX512* sizeof(__m128i)*4)         ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  }

  for(x=0; x < om->abc->Kp; x++){
    free(rbv_unstriped[x]);
  }
  free(rbv_unstriped);
  // 128-bit version
  if (fwrite((char *) om->evparam,      sizeof(float),    p7_NEVPARAM, hfp->ffp) != p7_NEVPARAM) ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) om->offs,         sizeof(off_t),    p7_NOFFSETS, hfp->ffp) != p7_NOFFSETS) ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) om->compo,        sizeof(float),    p7_MAXABET,  hfp->ffp) != p7_MAXABET)  ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(v3g_fmagic),    sizeof(uint32_t), 1,           hfp->ffp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed"); /* sentinel */
  //256-bit version 
  if (fwrite((char *) om->evparam,      sizeof(float),    p7_NEVPARAM, hfp->ffp256) != p7_NEVPARAM) ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) om->offs,         sizeof(off_t),    p7_NOFFSETS, hfp->ffp256) != p7_NOFFSETS) ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) om->compo,        sizeof(float),    p7_MAXABET,  hfp->ffp256) != p7_MAXABET)  ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(v3g_fmagic),    sizeof(uint32_t), 1,           hfp->ffp256) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed"); /* sentinel */
  //512-bit version
  if (fwrite((char *) om->evparam,      sizeof(float),    p7_NEVPARAM, hfp->ffp512) != p7_NEVPARAM) ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) om->offs,         sizeof(off_t),    p7_NOFFSETS, hfp->ffp512) != p7_NOFFSETS) ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) om->compo,        sizeof(float),    p7_MAXABET,  hfp->ffp512) != p7_MAXABET)  ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(v3g_fmagic),    sizeof(uint32_t), 1,           hfp->ffp512) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed"); /* sentinel */
  /* <hfp->pfp> gets the rest of the oprofile */
  if (fwrite((char *) &(v3g_pmagic),    sizeof(uint32_t), 1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->M),         sizeof(int),      1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->abc->type), sizeof(int),      1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &n,               sizeof(int),      1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) om->name,         sizeof(char),     n+1,         hfp->pfp) != n+1)         ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");

  if (om->acc == NULL) {
    n = 0;
    if (fwrite((char *) &n,             sizeof(int),      1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  } else {
    n = strlen(om->acc);
    if (fwrite((char *) &n,             sizeof(int),      1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
    if (fwrite((char *) om->acc,        sizeof(char),     n+1,         hfp->pfp) != n+1)         ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  }

  if (om->desc == NULL) {
    n = 0;
    if (fwrite((char *) &n,             sizeof(int),      1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  } else {
    n = strlen(om->desc);
    if (fwrite((char *) &n,             sizeof(int),      1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
    if (fwrite((char *) om->desc,       sizeof(char),     n+1,         hfp->pfp) != n+1)         ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  }
  
  if (fwrite((char *) om->rf,           sizeof(char),     om->M+2,     hfp->pfp) != om->M+2)     ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) om->mm,           sizeof(char),     om->M+2,     hfp->pfp) != om->M+2)     ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) om->cs,           sizeof(char),     om->M+2,     hfp->pfp) != om->M+2)     ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) om->consensus,    sizeof(char),     om->M+2,     hfp->pfp) != om->M+2)     ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");

  /* ViterbiFilter part */
  uint16_t *short_pointer;
  short_pointer = (uint16_t *) om->twv_avx512;
  // Unstripe the twv.  Output format is M elements of each transition value in
  // transition order

  // everything but the DDs
  for (int t = 0; t <(p7O_NTRANS -1); t++){ /* this loop of 7 transitions depends on the order in p7o_tsc_e */
    for(int hmmpos = 0; hmmpos < om->M; hmmpos++){
      int vector = ((hmmpos%Q8_AVX512 ) * (p7O_NTRANS-1)) + t;
      int within_vector = hmmpos/Q8_AVX512;  
      int index = (vector * (sizeof(__m512i)/sizeof(uint16_t)))+within_vector;
      ((uint16_t *)buffer)[(t * om->M)+hmmpos] = short_pointer[index];
    }
  }
  // and the DDs
  for(int hmmpos = 0; hmmpos < om->M; hmmpos++){
    int vector = (hmmpos % Q8_AVX512) + (Q8_AVX512 * (p7O_NTRANS -1));
      int within_vector = hmmpos /Q8_AVX512;
      int index = (vector * (sizeof(__m512i)/sizeof(uint16_t)))+within_vector;
      ((uint16_t *) buffer)[((p7O_NTRANS-1)* om->M) +hmmpos] = short_pointer[index];
  }

  if (fwrite((char *) buffer, sizeof(uint16_t),  om->M * p7O_NTRANS,        hfp->pfp) != p7O_NTRANS*om->M)        ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");

  for (x = 0; x < om->abc->Kp; x++){
    short_pointer = (uint16_t *) om->rwv_avx512[x];
    for(int hmmpos = 0; hmmpos < om->M; hmmpos++){
      int vector = hmmpos % Q8_AVX512;
      int within_vector = hmmpos /Q8_AVX512;
      int index = (vector * (sizeof(__m512i)/sizeof(uint16_t)))+within_vector;
      ((uint16_t *) buffer)[hmmpos] = short_pointer[index];
    }
    if (fwrite( (char *) buffer, sizeof(uint16_t),  om->M,         hfp->pfp) != om->M)         ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  }


  for (x = 0; x < p7O_NXSTATES; x++)
    if (fwrite( (char *) om->xw[x],        sizeof(int16_t),  p7O_NXTRANS, hfp->pfp) != p7O_NXTRANS) ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->scale_w),      sizeof(float),    1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->base_w),       sizeof(int16_t),  1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");  
  if (fwrite((char *) &(om->ddbound_w),    sizeof(int16_t),  1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->ncj_roundoff), sizeof(float),    1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");

  /* Forward/Backward part */

  float *float_pointer;
  float_pointer = (float *)om->tfv_avx512;
  // everything but the DDs
  for (int t = 0; t <(p7O_NTRANS -1); t++){ /* this loop of 7 transitions depends on the order in p7o_tsc_e */
    for(int hmmpos = 0; hmmpos < om->M; hmmpos++){
      int vector = ((hmmpos%Q4_AVX512) * (p7O_NTRANS-1)) + t;
      int within_vector = hmmpos/Q4_AVX512;  
      int index = (vector * (sizeof(__m512i)/sizeof(float)))+within_vector;
      ((float *)buffer)[(t * om->M)+hmmpos] = float_pointer[index];
    }
  }
  // and the DDs
  for(int hmmpos = 0; hmmpos < om->M; hmmpos++){
    int vector = (hmmpos % Q4_AVX512) + (Q4_AVX512 * (p7O_NTRANS -1));
      int within_vector = hmmpos /Q4_AVX512;
      int index = (vector * (sizeof(__m512i)/sizeof(float)))+within_vector;
      ((float *) buffer)[((p7O_NTRANS-1)* om->M) +hmmpos] = float_pointer[index];
  }

    if (fwrite((char *) buffer, sizeof(float),  om->M * p7O_NTRANS,        hfp->pfp) != p7O_NTRANS*om->M)        ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");

  for (x = 0; x < om->abc->Kp; x++){
    float_pointer = (float *) om->rfv_avx512[x];
    for(int hmmpos = 0; hmmpos < om->M; hmmpos++){
      int vector = hmmpos % Q4_AVX512;
      int within_vector = hmmpos /Q4_AVX512;
      int index = (vector * (sizeof(__m512i)/sizeof(float)))+within_vector;
      ((float *)buffer)[hmmpos] = float_pointer[index];
    }
    if (fwrite( (char *) buffer, sizeof(float),  om->M,         hfp->pfp) != om->M)         ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  }


  for (x = 0; x < p7O_NXSTATES; x++)
    if (fwrite( (char *) om->xf[x],     sizeof(float),    p7O_NXTRANS, hfp->pfp) != p7O_NXTRANS) ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");

  if (fwrite((char *)   om->cutoff,     sizeof(float),    p7_NCUTOFFS, hfp->pfp) != p7_NCUTOFFS) ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->nj),        sizeof(float),    1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->mode),      sizeof(int),      1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(om->L)   ,      sizeof(int),      1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed");
  if (fwrite((char *) &(v3g_pmagic),    sizeof(uint32_t), 1,           hfp->pfp) != 1)           ESL_EXCEPTION_SYS(eslEWRITE, "oprofile write failed"); /* sentinel */
  free(buffer);
  return eslOK;

 ERROR:
  p7_Die("Unable to allocate memory in p7_oprofile_Write_sse()\n");
  return eslEMEM;
}
/*---------------- end, writing oprofile ------------------------*/


/*****************************************************************
 * 2. Reading optimized profiles in two stages.
 *****************************************************************/

/* Function:  p7_oprofile_ReadMSV()
 * Synopsis:  Read MSV filter part of an optimized profile.
 *
 * Purpose:   Read the MSV filter part of a profile from the
 *            <.h3f> file associated with an open HMM file <hfp>.
 *            Allocate a new model, populate it with this minimal
 *            MSV filter information, and return a pointer to it
 *            in <*ret_om>. 
 *            
 *            Our alphabet may get set by the first HMM we read.  If
 *            <*byp_abc> is <NULL> at start, create a new alphabet and
 *            return a pointer to it in <*byp_abc>. If <*byp_abc> is
 *            non-<NULL>, it is assumed to be a pointer to an existing
 *            alphabet; we verify that the HMM's alphabet matches it
 *            and <*ret_abc> isn't changed.  This is the same
 *            convention used by <p7_hmmfile_Read()>.
 *            
 *            The <.h3f> file was opened automatically, if it existed,
 *            when the HMM file was opened with <p7_hmmfile_Open()>.
 *            
 *            When no more HMMs remain in the file, return <eslEOF>.
 *
 * Args:      hfp     - open HMM file, with associated .h3p file
 *            byp_abc - BYPASS: <*byp_abc == ESL_ALPHABET *> if known; 
 *                              <*byp_abc == NULL> if desired; 
 *                              <NULL> if unwanted.
 *            ret_om  - RETURN: newly allocated <om> with MSV filter
 *                      data filled in.
 *            
 * Returns:   <eslOK> on success. <*ret_om> is allocated here;
 *            caller free's with <p7_oprofile_Destroy()>.
 *            <*byp_abc> is allocated here if it was requested;
 *            caller free's with <esl_alphabet_Destroy()>.
 *            
 *            Returns <eslEFORMAT> if <hfp> has no <.h3f> file open,
 *            or on any parsing error.
 *            
 *            Returns <eslEINCOMPAT> if the HMM we read is incompatible
 *            with the existing alphabet <*byp_abc> led us to expect.
 *            
 *            On any returned error, <hfp->errbuf> contains an
 *            informative error message.
 *
 * Throws:    <eslEMEM> on allocation error.
 */

int
p7_oprofile_ReadMSV_avx512(P7_HMMFILE *hfp, ESL_ALPHABET **byp_abc, P7_OPROFILE **ret_om)
{
  P7_OPROFILE  *om = NULL;
  ESL_ALPHABET *abc = NULL;
  uint32_t      magic;
  off_t         roff;
  int           M, Q16_AVX, Q16x_AVX;
  int           x,n;
  int           alphatype;
  int           status;

  hfp->errbuf[0] = '\0';  // do NOT touch rr_errbuf[]. In thread parallelization, master is exclusively using ReadMSV, workers are using ReadRest
  if (hfp->ffp == NULL) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "no MSV profile file; hmmpress probably wasn't run");
  if (feof(hfp->ffp))   { status = eslEOF; goto ERROR; }	/* normal EOF: no more profiles */
  
  /* keep track of the starting offset of the MSV model */
  roff = ftello(hfp->ffp);

  if (! fread( (char *) &magic,     sizeof(uint32_t), 1, hfp->ffp)) { status = eslEOF; goto ERROR; }
  if (magic == v3a_fmagic)  ESL_XFAIL(eslEFORMAT, hfp->errbuf, "binary auxfiles are in an outdated HMMER format (3/a); please hmmpress your HMM file again");
  if (magic == v3b_fmagic)  ESL_XFAIL(eslEFORMAT, hfp->errbuf, "binary auxfiles are in an outdated HMMER format (3/b); please hmmpress your HMM file again");
  if (magic == v3c_fmagic)  ESL_XFAIL(eslEFORMAT, hfp->errbuf, "binary auxfiles are in an outdated HMMER format (3/c); please hmmpress your HMM file again");
  if (magic == v3d_fmagic)  ESL_XFAIL(eslEFORMAT, hfp->errbuf, "binary auxfiles are in an outdated HMMER format (3/d); please hmmpress your HMM file again");
  if (magic == v3e_fmagic)  ESL_XFAIL(eslEFORMAT, hfp->errbuf, "binary auxfiles are in an outdated HMMER format (3/e); please hmmpress your HMM file again");
  if (magic == v3f_fmagic)  ESL_XFAIL(eslEFORMAT, hfp->errbuf, "binary auxfiles are in an outdated HMMER format (3/f); please hmmpress your HMM file again");
  if (magic != v3g_fmagic)  ESL_XFAIL(eslEFORMAT, hfp->errbuf, "bad magic; not an HMM database?");

  if (! fread( (char *) &M,         sizeof(int),      1, hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read model size M");
  if (! fread( (char *) &alphatype, sizeof(int),      1, hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read alphabet type");  
  Q16_AVX  = p7O_NQB_AVX(M);
  Q16x_AVX = p7O_NQB_AVX(M) + p7O_EXTRA_SB;
  /* Set or verify alphabet. */
  if (byp_abc == NULL || *byp_abc == NULL)	{	/* alphabet unknown: whether wanted or unwanted, make a new one */
    if ((abc = esl_alphabet_Create(alphatype)) == NULL)  ESL_XFAIL(eslEMEM, hfp->errbuf, "allocation failed: alphabet");
  } else {			/* alphabet already known: verify it against what we see in the HMM */
    abc = *byp_abc;
    if (abc->type != alphatype) 
      ESL_XFAIL(eslEINCOMPAT, hfp->errbuf, "Alphabet type mismatch: was %s, but current profile says %s", 
		esl_abc_DecodeType(abc->type), esl_abc_DecodeType(alphatype));
  }
  /* Now we know the sizes of things, so we can allocate. */
  if ((om = p7_oprofile_Create(M, abc)) == NULL)         ESL_XFAIL(eslEMEM, hfp->errbuf, "allocation failed: oprofile");
  om->M = M;
  om->roff = roff;
  if (! fread((char *) &n,               sizeof(int),     1,           hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read name length");
  ESL_ALLOC(om->name, sizeof(char) * (n+1));
  if (! fread((char *) om->name,         sizeof(char),    n+1,         hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read name");

  if (! fread((char *) &(om->max_length),sizeof(int),     1,           hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read max_length");
  if (! fread((char *) &(om->tbm_b),     sizeof(uint8_t), 1,           hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read tbm");
  if (! fread((char *) &(om->tec_b),     sizeof(uint8_t), 1,           hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read tec");
  if (! fread((char *) &(om->tjb_b),     sizeof(uint8_t), 1,           hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read tjb");
  if (! fread((char *) &(om->scale_b),   sizeof(float),   1,           hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read scale");
  if (! fread((char *) &(om->base_b),    sizeof(uint8_t), 1,           hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read base");
  if (! fread((char *) &(om->bias_b),    sizeof(uint8_t), 1,           hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read bias");

  for (x = 0; x < abc->Kp; x++){
    if (! fread((char *) om->sbv_avx[x],     sizeof(__m256i), Q16x_AVX,        hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read ssv scores at %d [residue %c]", x, abc->sym[x]); 
  //  p7_restripe_byte((char *) om->sbv[x], (char*) om->sbv_avx[x], padded_byte_vector_length, 128, 256, 255);
  }
  for (x = 0; x < abc->Kp; x++){
    if (! fread((char *) om->rbv_avx[x],     sizeof(__m256i), Q16_AVX,         hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read msv scores at %d [residue %c]", x, abc->sym[x]);
    //p7_restripe_byte((char *) om->rbv[x], (char *) om->rbv_avx[x], byte_vector_length, 128, 256, 255);
  }
  if (! fread((char *) om->evparam,      sizeof(float),   p7_NEVPARAM, hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read stat params");
  if (! fread((char *) om->offs,         sizeof(off_t),   p7_NOFFSETS, hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read hmmpfam offsets");
  if (! fread((char *) om->compo,        sizeof(float),   p7_MAXABET,  hfp->ffp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read model composition");

  /* record ends with magic sentinel, for detecting binary file corruption */
  if (! fread( (char *) &magic,     sizeof(uint32_t), 1, hfp->ffp))  ESL_XFAIL(eslEFORMAT, hfp->errbuf, "no sentinel magic: .h3f file corrupted?");
  if (magic != v3g_fmagic)                                           ESL_XFAIL(eslEFORMAT, hfp->errbuf, "bad sentinel magic; .h3f file corrupted?");

  /* keep track of the ending offset of the MSV model */
  om->eoff = ftello(hfp->ffp) - 1;

  if (byp_abc != NULL) *byp_abc = abc;  /* pass alphabet (whether new or not) back to caller, if caller wanted it */
  *ret_om = om;
  return eslOK;

 ERROR:
  if (abc && (byp_abc == NULL || *byp_abc == NULL)) esl_alphabet_Destroy(abc); /* destroy alphabet if we created it here */
  p7_oprofile_Destroy(om);
  *ret_om = NULL;
  return status;
}

/* Function:  p7_oprofile_ReadRest()
 * Synopsis:  Read the rest of an optimized profile.
 *
 * Purpose:   Read the rest of an optimized profile <om> from
 *            the <.h3p> file associated with an open HMM
 *            file <hfp>. 
 *            
 *            This is the second part of a two-part calling sequence.
 *            The <om> here must be the result of a previous
 *            successful <p7_oprofile_ReadMSV()> call on the same
 *            open <hfp>.
 *
 *            In thread-parallel hmmscan, the master is calling
 *            ReadMSV() and multiple workers are calling ReadRest().
 *            ReadRest() must be mutex-protected, and we must make
 *            sure that ReadMSV and ReadRest never touch the same
 *            data. ReadMSV only touches ffp (the MSV input data
 *            stream) and errbuf. We can't use the same errbuf
 *            in ReadRest; we work around by using hfp->rr_errbuf.
 
 *
 * Args:      hfp - open HMM file, from which we've previously
 *                  called <p7_oprofile_ReadMSV()>.
 *            om  - optimized profile that was successfully
 *                  returned by  <p7_oprofile_ReadMSV()>.
 *
 * Returns:   <eslOK> on success, and <om> is now a complete
 *            optimized profile.
 *            
 *            Returns <eslEFORMAT> if <hfp> has no <.h3p> file open,
 *            or on any parsing error, and set <hfp->rr_errbuf> to
 *            an informative error message.
 *
 * Throws:    <eslESYS> if an <fseek()> fails to reposition the
 *            binary <.h3p> file.
 *            
 *            <eslEMEM> on allocation error.
 */

int
p7_oprofile_ReadRest_avx512(P7_HMMFILE *hfp, P7_OPROFILE *om)
{
  uint32_t      magic;
  int           M, Q4_AVX, Q8_AVX;
  int           x,n;
  char         *name = NULL;
  int           alphatype;
  int           status;

#ifdef HMMER_THREADS
  /* lock the mutex to prevent other threads from reading from the optimized
   * profile at the same time.
   */
  if (hfp->syncRead)
    {
      if (pthread_mutex_lock (&hfp->readMutex) != 0) ESL_EXCEPTION(eslESYS, "mutex lock failed");
    }
#endif  
  hfp->rr_errbuf[0] = '\0';
  if (hfp->pfp == NULL) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "no MSV profile file; hmmpress probably wasn't run");
 
  /* Position the <hfp->pfp> using offset stored in <om> */
  if (fseeko(hfp->pfp, om->offs[p7_POFFSET], SEEK_SET) != 0)                       ESL_EXCEPTION(eslESYS, "fseeko() failed");
   
  if (! fread( (char *) &magic,          sizeof(uint32_t), 1,           hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read magic");
  if (magic == v3a_pmagic) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "binary auxfiles are in an outdated HMMER format (3/a); please hmmpress your HMM file again");
  if (magic == v3b_pmagic) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "binary auxfiles are in an outdated HMMER format (3/b); please hmmpress your HMM file again");
  if (magic == v3c_pmagic) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "binary auxfiles are in an outdated HMMER format (3/c); please hmmpress your HMM file again");
  if (magic == v3d_pmagic) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "binary auxfiles are in an outdated HMMER format (3/d); please hmmpress your HMM file again");
  if (magic == v3e_pmagic) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "binary auxfiles are in an outdated HMMER format (3/e); please hmmpress your HMM file again");
  if (magic == v3f_pmagic) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "binary auxfiles are in an outdated HMMER format (3/f); please hmmpress your HMM file again");
  if (magic != v3g_pmagic) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "bad magic; not an HMM database file?");

  if (! fread( (char *) &M,              sizeof(int),      1,           hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read model size M");
  if (! fread( (char *) &alphatype,      sizeof(int),      1,           hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read alphabet type");  
  if (! fread( (char *) &n,              sizeof(int),      1,           hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read name length");  
  if (M         != om->M)                                                          ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "p/f model length mismatch");
  if (alphatype != om->abc->type)                                                  ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "p/f alphabet type mismatch");

  ESL_ALLOC(name, sizeof(char) * (n+1));
  if (! fread( (char *) name,            sizeof(char),     n+1,         hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read name");  
  if (strcmp(name, om->name) != 0)                                                 ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "p/f name mismatch");  
  
  if (! fread((char *) &n,               sizeof(int),      1,           hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read accession length");
  if (n > 0) {
    ESL_ALLOC(om->acc, sizeof(char) * (n+1));
    if (! fread( (char *) om->acc,       sizeof(char),     n+1,         hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read accession");      
  }
  if (! fread((char *) &n,               sizeof(int),      1,           hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read description length");
  if (n > 0) {
    ESL_ALLOC(om->desc, sizeof(char) * (n+1));
    if (! fread( (char *) om->desc,      sizeof(char),     n+1,         hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read description");      
  }

  if (! fread((char *) om->rf,           sizeof(char),     M+2,         hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read rf annotation");
  if (! fread((char *) om->mm,           sizeof(char),     M+2,         hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read mm annotation");
  if (! fread((char *) om->cs,           sizeof(char),     M+2,         hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read cs annotation");
  if (! fread((char *) om->consensus,    sizeof(char),     M+2,         hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read consensus annotation");

  Q4_AVX  = p7O_NQF_AVX(om->M);
  Q8_AVX  = p7O_NQW_AVX(om->M);
  char *buffer;
  ESL_ALLOC(buffer, sizeof(__m256i) * Q4_AVX * p7O_NTRANS); // big enough for any row

  if (! fread((char *) buffer, sizeof(uint16_t),  p7O_NTRANS*om->M, hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read <tu>, vitfilter transitions");
  uint16_t *short_pointer;
  short_pointer = (uint16_t *) om->twv_avx;
  // everything but the DDs
  for (int t = p7O_BM; t <= p7O_II; t++){ /* this loop of 7 transitions depends on the order in p7o_tsc_e */
    for(int hmmpos = 0; hmmpos < om->M; hmmpos++){
      int vector = ((hmmpos%Q8_AVX) * (p7O_NTRANS-1)) + t;
      int within_vector = hmmpos/Q8_AVX;  
      int index = (vector * (sizeof(__m256i)/sizeof(uint16_t)))+within_vector;
      short_pointer[index] = ((uint16_t *) buffer)[(t * om->M)+hmmpos];
    }
    for(int padding = om->M; padding < (Q8_AVX * (sizeof(__m256i)/sizeof(uint16_t))); padding++){
      int vector = ((padding%Q8_AVX) * (p7O_NTRANS-1)) + t;
      int within_vector = padding/Q8_AVX;  
      int index = (vector * (sizeof(__m256i)/sizeof(uint16_t)))+within_vector;
      short_pointer[index] = -32768;
    }
  }
  // and the DDs
  for(int hmmpos = 0; hmmpos < om->M; hmmpos++){
    int vector = (hmmpos % Q8_AVX) + (Q8_AVX * (p7O_NTRANS -1));
      int within_vector = hmmpos /Q8_AVX;
      int index = (vector * (sizeof(__m256i)/sizeof(uint16_t)))+within_vector;
      short_pointer[index] = ((uint16_t *) buffer)[((p7O_NTRANS -1)*om->M) +hmmpos];
  }
  for(int padding = om->M; padding < (Q8_AVX * (sizeof(__m256i)/sizeof(uint16_t))); padding++){
      int vector = (padding % Q8_AVX) + (Q8_AVX * (p7O_NTRANS -1));
      int within_vector = padding /Q8_AVX;
      int index = (vector * (sizeof(__m256i)/sizeof(uint16_t)))+within_vector;
      short_pointer[index] = -32768;
  }

  for (x = 0; x < om->abc->Kp; x++){
    short_pointer = (uint16_t *) om->rwv_avx[x];
    if (! fread((char *) buffer,     sizeof(int16_t), om->M,         hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read viterbi scores at %d [residue %c]", x, om->abc->sym[x]);
    int hmmpos;
    for(hmmpos = 0; hmmpos < om->M; hmmpos++){
      int vector = hmmpos % Q8_AVX;
      int within_vector = hmmpos /Q8_AVX;
      int index = (vector * (sizeof(__m256i)/sizeof(uint16_t)))+within_vector;
      short_pointer[index] = ((uint16_t *)buffer)[hmmpos];
    }
    
    for(int padding = hmmpos; padding < (Q8_AVX * (sizeof(__m256i)/sizeof(uint16_t))); padding++){
      int vector = padding % Q8_AVX;
      int within_vector = padding /Q8_AVX;
      int index = (vector * (sizeof(__m256i)/sizeof(uint16_t)))+within_vector;
      short_pointer[index] = -32768;  // value for elements of vector that don't
        //correspond to HMM positions
    }
  }
  for (x = 0; x < p7O_NXSTATES; x++)
    if (! fread( (char *) om->xw[x],        sizeof(int16_t),  p7O_NXTRANS, hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read <xu>[%d], vitfilter special transitions", x);
  if (! fread((char *) &(om->scale_w),      sizeof(float),    1,           hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read scale_w");
  if (! fread((char *) &(om->base_w),       sizeof(int16_t),  1,           hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read base_w");
  if (! fread((char *) &(om->ddbound_w),    sizeof(int16_t),  1,           hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read ddbound_w");
  if (! fread((char *) &(om->ncj_roundoff), sizeof(float),    1,           hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read ddbound_w");

 if (! fread((char *) buffer, sizeof(float),  p7O_NTRANS*om->M, hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read <tf>, fwd/back transitions");
  float *float_pointer = (float *) om->tfv_avx;
  // everything but the DDs
  for (int t = p7O_BM; t <= p7O_II; t++){ /* this loop of 7 transitions depends on the order in p7o_tsc_e */
    for(int hmmpos = 0; hmmpos < om->M; hmmpos++){
      int vector = ((hmmpos%Q4_AVX) * (p7O_NTRANS-1)) + t;
      int within_vector = hmmpos/Q4_AVX;  
      int index = (vector * (sizeof(__m256i)/sizeof(float)))+within_vector;
      float_pointer[index] = ((float *) buffer)[(t * om->M)+hmmpos];
    }
    for(int padding = om->M; padding < (Q4_AVX * (sizeof(__m256i)/sizeof(float))); padding++){
      int vector = ((padding%Q4_AVX) * (p7O_NTRANS-1)) + t;
      int within_vector = padding/Q4_AVX;  
      int index = (vector * (sizeof(__m256i)/sizeof(float)))+within_vector;
      float_pointer[index] = 0;
    }
  }
  // and the DDs
  for(int hmmpos = 0; hmmpos < om->M; hmmpos++){
    int vector = (hmmpos % Q4_AVX) + (Q4_AVX * (p7O_NTRANS -1));
      int within_vector = hmmpos /Q4_AVX;
      int index = (vector * (sizeof(__m256i)/sizeof(float)))+within_vector;
      float_pointer[index] = ((float *) buffer)[((p7O_NTRANS -1)*om->M) +hmmpos];
  }
  for(int padding = om->M; padding < (Q4_AVX * (sizeof(__m256i)/sizeof(float))); padding++){
      int vector = (padding % Q4_AVX) + (Q4_AVX * (p7O_NTRANS -1));
      int within_vector = padding /Q4_AVX;
      int index = (vector * (sizeof(__m256i)/sizeof(float)))+within_vector;
      float_pointer[index] = 0;
  }

  for (x = 0; x < om->abc->Kp; x++){
    float_pointer = (float *) om->rfv_avx[x];
    if (! fread((char *) buffer,     sizeof(float), om->M,         hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->errbuf, "failed to read forward scores at %d [residue %c]", x, om->abc->sym[x]);
    int hmmpos;
    for(hmmpos = 0; hmmpos < om->M; hmmpos++){
      int vector = hmmpos % Q4_AVX;
      int within_vector = hmmpos /Q4_AVX;
      int index = (vector * (sizeof(__m256i)/sizeof(float)))+within_vector;
      float_pointer[index] = ((float *)buffer)[hmmpos];
    }
    for(int padding = hmmpos; padding < (Q4_AVX * (sizeof(__m256i)/sizeof(float))); padding++){
      int vector = padding % Q4_AVX;
      int within_vector = padding /Q4_AVX;
      int index = (vector * (sizeof(__m256i)/sizeof(float)))+within_vector;
      float_pointer[index] = 0;  // value for elements of vector that don't correspond to HMM
      //positions
    }
  }
  for (x = 0; x < p7O_NXSTATES; x++)
    if (! fread( (char *) om->xf[x],     sizeof(float),    p7O_NXTRANS, hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read <xf>[%d] special transitions", x);

  if (! fread((char *)   om->cutoff,     sizeof(float),    p7_NCUTOFFS, hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read Pfam score cutoffs");
  if (! fread((char *) &(om->nj),        sizeof(float),    1,           hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read nj");
  if (! fread((char *) &(om->mode),      sizeof(int),      1,           hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read mode");
  if (! fread((char *) &(om->L)   ,      sizeof(int),      1,           hfp->pfp)) ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "failed to read L");

  /* record ends with magic sentinel, for detecting binary file corruption */
  if (! fread( (char *) &magic,     sizeof(uint32_t), 1, hfp->pfp))  ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "no sentinel magic: .h3p file corrupted?");
  if (magic != v3g_pmagic)                                           ESL_XFAIL(eslEFORMAT, hfp->rr_errbuf, "bad sentinel magic; .h3p file corrupted?");

#ifdef HMMER_THREADS
  if (hfp->syncRead)
    {
      if (pthread_mutex_unlock (&hfp->readMutex) != 0) ESL_EXCEPTION(eslESYS, "mutex unlock failed");
    }
#endif
  free(buffer);
  free(name);
  return eslOK;

 ERROR:

#ifdef HMMER_THREADS
  if (hfp->syncRead)
    {
      if (pthread_mutex_unlock (&hfp->readMutex) != 0) ESL_EXCEPTION(eslESYS, "mutex unlock failed");
    }
#endif

  if (name != NULL) free(name);
  return status;
}
/*----------- end, reading optimized profiles -------------------*/
#endif //esl_ENABLEAVX512

#ifndef eslENABLE_AVX512  // stubs for compilers that can't handle AVX-512

int p7_oprofile_Write_avx512(P7_HMMFILE *hfp, P7_OPROFILE *om){
  return eslEUNSUPPORTEDISA;
}

int p7_oprofile_ReadMSV_avx512(P7_HMMFILE *hfp, ESL_ALPHABET **byp_abc, P7_OPROFILE **ret_om){
  return eslEUNSUPPORTEDISA;
}

int p7_oprofile_ReadRest_avx512(P7_HMMFILE *hfp, P7_OPROFILE *om){
  return eslEUNSUPPORTEDISA;
}

#endif