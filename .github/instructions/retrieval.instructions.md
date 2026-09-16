---
applyTo: "api.py,main.py,src/config.py,src/retrieval.py,src/rag_chain.py,src/models.py,rag_eval/**/*.py,tests/**/*.py"
---

Review retrieval changes for preservation of keyword, vector, and ensemble semantics. Check empty
corpora, missing or stale indexes, disabled vector search, reranker failure, no-key operation,
candidate limits, deterministic ordering, and citation-to-source traceability.

Do not accept generated answers whose supporting quote, document identity, or page/element
metadata can drift from the retrieved evidence. Keep local FAISS loading behind the explicit trust
gate. Require deterministic tests with fake model, embedding, or reranker behavior rather than
network-dependent tests.
