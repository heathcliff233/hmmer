# cuda-hmm Reference Compatibility Note

The external `divinrkz/cuda-hmm` repository is reference material only, not a migration target.

Use `agents_docs/cuda-hmm-reference.md` for the current notes. HMMER GPU work should remain native to the existing autotools/CUDA build; do not add CMake, copy code, vendor sources, link external libraries, or build the external project inside this worktree.
