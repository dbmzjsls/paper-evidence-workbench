---
applyTo: "api.py,src/indexing.py,src/storage.py,src/evidence.py,src/parser.py,src/bulid_db.py,src/parsers/**/*.py,tests/**/*.py"
---

Review ingestion and storage changes as processing of untrusted input. Check upload batch
atomicity, filename normalization, size/type validation, archive or parser output paths, root
containment, duplicate detection, cleanup after failure, and disclosure of local paths or raw
document content.

Verify that documents, elements, assets, chunks, FTS rows, and vector-index metadata cannot become
partially updated or stale without a visible status. Require focused temporary-directory tests for
new boundary and failure behavior; routine tests must not process large real documents.
