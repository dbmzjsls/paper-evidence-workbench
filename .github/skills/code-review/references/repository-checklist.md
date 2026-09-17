# Repository review checklist

Use this as a relevance filter, not a requirement to comment on every category.

## Ingestion and storage

- Can filenames, parser-produced paths, symlinks, or requested ingest paths escape allowed roots?
- Can a failed multi-file batch leave accepted files or partial corpus records behind?
- Do deduplication and rebuild operations keep documents, elements, assets, chunks, and FTS rows aligned?
- Are raw documents and local filesystem paths kept out of errors and responses where inappropriate?

## Index safety

- Does stored FAISS metadata match the current corpus and embedding assumptions?
- Is local index loading still blocked unless `TRUST_LOCAL_FAISS_INDEX` explicitly permits it?
- Are empty, missing, corrupt, and stale index states handled visibly without breaking keyword search?

## Retrieval and evidence

- Are keyword, vector, and ensemble modes still distinct and normalized consistently?
- Are candidate limits, score combination, reranking, and tie ordering deterministic and bounded?
- Does each citation retain the correct document, quote, page, element, and asset provenance?
- Can an LLM answer make claims that are absent from the supplied evidence?

## Compatibility and tests

- Do legacy entry points continue to behave compatibly or have their callers been migrated?
- Does no-key operation still provide the documented extractive fallback?
- Do tests avoid credentials, external services, large downloads, and mutable shared data?
