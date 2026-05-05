# cuda-hmm Reference Notes

Last updated: 2026-05-05

## Source Checked

- Repository: `https://github.com/divinrkz/cuda-hmm/`
- Commit inspected locally: `fd0b5ec correct config file`
- Local clone path during assessment: `/tmp/cuda-hmm`
- Relevant files:
  - `include/hmm_gpu.cuh`
  - `src/hmm/hmm_gpu.cu`
  - `src/hmm/hmm_impl.cpp`
  - `tests/test_benchmark.cpp`

No license file was present in the inspected checkout. Do not copy code from this repository into HMMER unless licensing is clarified. For the current GPU work, treat the repository only as a source of high-level CUDA DP design cues.

## What It Implements

The project implements textbook dense HMM algorithms:

- Forward: dense `N x N` transition matrix, dense `N x M` emissions, probability-space recursion.
- Backward: dense probability-space recursion.
- Viterbi: dense probability-space recursion plus a backpointer matrix and CPU path reconstruction.
- Baum-Welch: uses scaled Forward/Backward plus expectation/maximization kernels.

The CUDA mapping is one time step at a time, parallelizing across dense states within a time step. For Viterbi and Forward/Backward, each state thread loops over all previous/next states. The implementation copies model and observation arrays for each call and stores full trellis-sized device matrices.

## Why Its Code Does Not Directly Apply To HMMER

HMMER's protein search path is not a dense textbook HMM:

- The model is Plan7, with match/insert/delete states, special states, local/multihit transitions, and sparse structured recurrences.
- The current Viterbi filter is an optimized 16-bit striped SIMD approximation (`p7_ViterbiFilter()`), including overflow-as-pass behavior. A GPU replacement must preserve the score boundary, overflow handling, and E-value behavior.
- `p7_ForwardParser()` is probability-space but not just a final score. It also writes `P7_OMX` parser rows and scaling/state needed by `p7_BackwardParser()` and domain definition.
- Backward is tightly coupled to Forward scaling/state and domain definition. A standalone dense backward kernel would not produce the `P7_OMX` state needed by `p7_domaindef_ByPosteriorHeuristics()`.
- The external implementation is single-sequence oriented and dense in states. The current HMMER GPU path needs batched candidate handling from dsqdata chunks and must be measured against the accepted MSV+bias baseline on profmark.

Because of these differences, copying or mechanically porting the dense kernels would not be parity-safe and would likely repeat the rejected naive Viterbi direction: high overhead, wrong score/state boundary, and insufficient compatibility with HMMER's optimized pipeline.

## Usable Ideas

The repository is still useful as a reminder of broad CUDA DP patterns:

- Keep time steps sequential unless using a more advanced wavefront/parallel-prefix design.
- Parallelize state computations within a time step.
- Treat Forward/Backward scaling as first-class state, not as a final-score detail.
- Benchmark GPU kernels only after including host/device transfer, candidate packing, and downstream CPU rerun or state handoff costs.

These are design cues, not code to port or build.

## HMMER-Specific Implementation Notes

Do not start by porting the dense `cuda-hmm` kernels. Borrow ideas only where they fit HMMER's Plan7 pipeline. A safe HMMER path should be:

1. Build a parity harness before pipeline integration.
   - Input: post-bias survivor sequences from `hmmsearch --gpu`.
   - Compare CPU optimized Viterbi score/status to any GPU Viterbi prototype.
   - Compare CPU `p7_ForwardParser()` score and required parser/scaling state to any GPU Forward prototype, or explicitly include CPU Forward rerun for F3 survivors.

2. If retrying Viterbi, use HMMER's optimized profile layout and filter semantics.
   - Preserve `p7_ViterbiFilter()` 16-bit score approximation and `eslERANGE` high-score pass behavior.
   - Batch post-bias candidates; avoid one expensive launch per sequence.
   - Measure against the current MSV+bias baseline, not just CPU.

3. If trying Forward, start as a score prefilter only after CPU Viterbi.
   - Include CPU Forward rerun cost for F3 survivors unless the GPU path returns `P7_OMX`-equivalent state.
   - The all-13 profmark upper bound for a free score-only Forward prefilter was modest, so a real kernel must be batched and low-overhead.

4. Treat Forward/Backward/domain as a coupled state-transfer problem if moving beyond score-only filtering.
   - Backward alone is not useful in the current measurements.
   - Domain definition remains CPU-side unless a much larger matrix/state handoff is designed.

## Verification Notes

The external project was not built, and that is intentional for this task. We are borrowing high-level CUDA DP design cues only, not copying, linking, vendoring, or treating the external repository as a dependency. Source inspection was sufficient because the model and state representation are visibly incompatible with HMMER's Plan7 optimized pipeline.

Do not add CMake to this HMMER worktree just to evaluate or reuse ideas from `cuda-hmm`. HMMER remains an autotools project, and any GPU implementation that lands here should be native to the existing `configure`/`Makefile.in` flow. If a future task needs to reproduce an external benchmark, do that outside this repository and record only the relevant observations.

Current local environment check:

- `g++` is available.
- `nvcc` is available at `/usr/local/cuda/bin/nvcc` and reports CUDA 12.6.85. HMMER's `./configure --enable-cuda --with-cuda-arch=sm_89` finds that path even when `CUDA_HOME` is unset because the generated CUDA search path includes `/usr/local/cuda/bin`.
