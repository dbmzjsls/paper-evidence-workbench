---
name: code-review
description: Review pull requests for this evidence-grounded Python, FastAPI, SQLite/FTS5, FAISS, MinerU, and LangChain workbench. Use this skill for every PR code review, especially changes involving untrusted document ingestion, parsing, corpus storage, indexes, retrieval modes, reranking, citations, LLM fallbacks, uploads, or legacy compatibility.
---

# Pull request review workflow

## Gather context

1. Read the PR title, description, linked issue, and complete diff.
2. Read `AGENTS.md`, `.github/copilot-instructions.md`, and applicable path instructions.
3. Inspect affected callers, models, storage transactions, parsers, tests, and compatibility shims.
4. Consult GitHub Actions results through the GitHub MCP tools when available.
5. Read `references/repository-checklist.md` and apply only the sections relevant to the diff.

## Analyze

Trace concrete documents, corpus states, and queries through the changed code. Give highest
priority to path escape, unsafe deserialization, partial ingestion, stale indexes, lost evidence
provenance, unsupported generated claims, changed retrieval semantics, nondeterministic ordering,
and hidden fallback behavior. Check that tests exercise failures and boundary states without
requiring credentials, network services, or large model downloads.

## Report

Create an inline finding only when the issue is introduced or exposed by the PR and can be
demonstrated from repository context. In Simplified Chinese, state the trigger, impact, and a
minimal correction. Keep identifiers and commands unchanged. If evidence is incomplete, ask a
clearly labeled question instead of asserting a defect. Do not produce a comment merely to prove
that the review ran.
