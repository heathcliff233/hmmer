# CUDA subsystem

This directory owns HMMER's CUDA implementation for the opt-in protein
`hmmsearch --gpu` path.

## Files

- `p7_cuda.h`: C-callable CUDA API used by HMMER C modules.
- `p7_cuda_internal.h`: private shared CUDA runtime/profile structures and
  internal helper declarations used across stage files.
- `p7_cuda_runtime.cu`: CUDA runtime availability checks, engine lifecycle,
  profile upload/update/destruction, and shared stats plumbing.
- `p7_cuda_msv.cu`: MSV kernels and the public MSV entry points.
- `p7_cuda_bias.cu`: bias-composition kernel and batched bias entry point.
- `p7_cuda_viterbi.cu`: Viterbi score prefilter kernel and batched entry
  point.
- `p7_cuda_forward.cu`: Forward score prefilter kernels and batched entry
  point.
- `p7_cuda_fb_parser.cu`: experimental Forward/Backward parser kernels and
  single/batched parser entry points.
- `p7_cuda_stub.c`: non-CUDA implementation of the same API. It lets normal
  non-CUDA builds link cleanly and report an explicit runtime diagnostic when
  `--gpu` is requested.

`../cuda_msv.h` is only a compatibility wrapper for older include sites. New
CUDA-facing code should include `cuda/p7_cuda.h` directly.

## Boundaries

The accepted GPU path is additive to the normal HMMER pipeline. CUDA currently
accelerates protein `hmmsearch --gpu` on target databases built by `hmmseqdb`
as Easel protein `dsqdata`; ordinary sequence files remain on the CPU path.

The selected CPU optimized implementation (`impl_sse`, `impl_neon`, or
`impl_vmx`) remains active. CPU code still owns profile construction,
thresholding, survivor continuation, domain definition, null2, hit storage,
and output. Experimental later-stage CUDA options must preserve final hit
parity before becoming default behavior.
