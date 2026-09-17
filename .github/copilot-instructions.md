# GitHub Copilot repository instructions

When reviewing a pull request, write concise comments in Simplified Chinese while preserving
code identifiers, paths, and commands in their original form.

Prioritize defects that can break evidence traceability, return unsupported claims, corrupt or
desynchronize corpus/index state, trust unsafe serialized data, escape configured filesystem
roots, expose sensitive document content, or silently change retrieval behavior. Inspect callers,
models, storage code, tests, and the documented compatibility modules before concluding that an
issue exists.

For every finding:

- Identify a concrete input, corpus state, or execution sequence that triggers the problem.
- Explain its effect on evidence quality, security, reproducibility, or supported behavior.
- Point to the smallest relevant changed line and propose the smallest viable correction.
- Distinguish demonstrated defects from questions and optional improvements.

Do not report personal style preferences, speculative scale requirements, or issues already
reported precisely by Ruff, pytest, or the package build. Do not require live LLM calls, MinerU
runs, embedding downloads, or large-model downloads in routine tests.

Apply `AGENTS.md` and applicable path instructions. Focus on untrusted document parsing, upload
and path containment, ingestion atomicity and deduplication, SQLite/FTS5/FAISS consistency,
explicit trust gates, retrieval-mode semantics, citation provenance, deterministic fallbacks, and
legacy compatibility.
