/* Debugging utilities foc striped (vectorized) data structures

1. Unstriping routines

*/
#include <stdlib.h>
#include "easel.h"
#include "hmmer.h"
#include "impl_avx.h"
/* Function:  p7_unstripe_float
 * Synopsis:  Creates and returns an L x M array of floats from a striped array of length
 *            vector_length_bits (which must be divisible by 32)
 * Incept:    NPC 1/13/24 [The Cushy Office]
 *
 * Purpose:   Unstripes a Farrar-striped array of floating-point values to allow easier comparison
 *            between data that has been striped with different SIMD widths
 *
 * Throws:    <NULL> on allocation error.
 */
float **p7_unstripe_float(float **striped_array, int M, int L, int vector_length_bits){
  float **unstriped_array;
  int status;
  // Create the unstriped array
  ESL_ALLOC(unstriped_array, L * sizeof(float *));
  for(int i = 0; i < L; i++){
    ESL_ALLOC(unstriped_array[i], M * sizeof(float));
  }
  int Q = ( ESL_MAX(2, ((((M)-1) / (vector_length_bits / (sizeof(float)*8)))  + 1))) ;
  for (int x = 0; x < L; x++){
    float *striped_row = (float *) striped_array[x];
    //unstripe a row.  Only unstripe M elements, ignore any padding to 
    for(int hmmpos = 0; hmmpos < M; hmmpos++){
      int vector = hmmpos % Q;
      int within_vector = hmmpos /Q;
      int index = (vector * (vector_length_bits / (sizeof(float)*8))) +within_vector;
      unstriped_array[x][hmmpos] = striped_row[index];
    }
  }
  ERROR:
  // Clean up any memory we allocated
  p7_unstriped_float_Destroy(unstriped_array, L);
  return NULL;
}

/* Function:  p7_unstripe_float_Compare
 * Synopsis:  Compares two 2-D arrays of floating-point numbers for equality
 * Incept:    NPC 1/13/24 [The Cushy Office]
 *
 * Purpose:   Compares two 2-d arrays of floating-point numbers for equality.  Returns
 *            eslOK if all elements match to within .01 tolerance.  Otherwise, prints
 *            information about the elements that did not match and returns eslFAIL.
 *
 * Throws:    none.
 */
int p7_unstriped_float_Compare(float **arr1, float **arr2, int L, int M){
  int found_mismatch = 0;
  for(int i = 0; i < L; i++){
    for(int j = 0; j<L; j++){
      if(esl_FCompare(arr1[i][j], arr2[i][j], 0.01, 0.01)!=eslOK){
        printf("Element miss-match at %d, %d, %f vs. %f\n", i, j, arr1[i][j], arr2[i][j]);
        found_mismatch = 1; 
      }
    }
  }
  if(found_mismatch != 0){
    return eslFAIL;
  }
  else{
    return eslOK;
  }
}

/* Function:  p7_unstriped_float_Destroy
 * Synopsis:  Frees a 2-D array of the type created by p7_unstripe_float
 * Incept:    NPC 1/13/24 [The Cushy Office]
 *
 * Purpose:   Unstripes a Farrar-striped array of floating-point values to allow easier comparison
 *            between data that has been striped with different SIMD widths
 *
 * Throws:    <NULL> on allocation error.
 */
void p7_unstriped_float_Destroy(float **unstriped_array, int L){
  if(unstriped_array != NULL){
    for(int i = 0; i < L; i++){
      if(unstriped_array[i] != NULL){
        free(unstriped_array[i]);
      }
    } 
    free(unstriped_array);
  }
}