# HMMER Agent Docs

This directory contains agent-facing notes for navigating the HMMER worktree. The notes optimize for source ownership, data flow, and safe verification choices rather than user-facing tutorial material.

Start with:

1. `architecture.md`
2. `data-flow.md`
3. `program-families.md`
4. The topic-specific file that matches the change: `search-pipelines.md`, `nucleotide-and-fm-index.md`, `model-build-and-io.md`, `parallel-daemon-cache.md`, or `tests-docs-benchmarks.md`.

The parent worktree currently lacks `easel/`. Build and full test notes assume Easel is restored first.
