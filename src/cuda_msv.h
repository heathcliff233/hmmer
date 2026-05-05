/* Compatibility wrapper for the CUDA acceleration API.
 *
 * New code should include "cuda/p7_cuda.h" directly.  This wrapper preserves
 * the original include path used by the existing MSV benchmark and pipeline
 * code while the CUDA subsystem lives under src/cuda/.
 */
#ifndef P7_CUDA_MSV_COMPAT_INCLUDED
#define P7_CUDA_MSV_COMPAT_INCLUDED

#include "cuda/p7_cuda.h"

#endif /*P7_CUDA_MSV_COMPAT_INCLUDED*/
