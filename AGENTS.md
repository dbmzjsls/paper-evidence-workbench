# Repository guidance

## Product boundaries

- This workbench produces evidence-grounded answers and screening reports. Preserve traceability
  from every claim and citation to the stored document, element, page, and quoted text.
- Treat imported files, parsed output, HTML, metadata, and local vector indexes as untrusted input.
- The no-key extractive fallback is supported behavior. Do not make basic ingestion and retrieval
  require an LLM credential or an external network call.

## Architecture invariants

- Keep SQLite/FTS5 records, parsed elements, assets, chunks, and FAISS metadata consistent across
  ingestion, deduplication, rebuild, and reindex operations.
- Never load a local FAISS index unless the explicit trust setting permits it. Stale or missing
  indexes must degrade safely and visibly.
- Preserve distinct keyword, vector, and ensemble retrieval semantics and stable citation ordering.
- Keep `src/bulid_db.py`, `src/parser.py`, and `src/rag_chain.py` compatible unless a change
  deliberately migrates their callers and tests.
- Restrict path ingestion to configured data/upload roots and prevent partial batches on upload
  validation failure.

## Validation

Run the checks relevant to the change. The CI-equivalent sequence is:

```text
uv sync --frozen --group dev
uv run ruff check --output-format=github .
uv run python -m pytest tests -q
uv build
```

Tests should use temporary storage and deterministic substitutes for model, parser, embedding,
and reranker calls whenever practical. Do not download large models or process real documents in
routine CI tests.
