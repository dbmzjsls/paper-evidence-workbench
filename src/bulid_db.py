"""
Backward-compatible shim for the historical misspelled module name.

New code should import from src.indexing.
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.config import Config
from src.indexing import IndexingService, build_and_save_db, get_db_stats
from src.models import ContentElement, PaperDocument, ParsedDocument, utc_now_iso
from src.parsers.base import clean_text, document_id_from_hash, hash_file


def add_documents(new_docs: list[Document], embedding_model=None):
    """
    Compatibility helper for older callers that already hold LangChain documents.
    Documents are stored as a synthetic imported corpus, then the FAISS index is rebuilt.
    """
    service = IndexingService()
    imported = []
    for idx, doc in enumerate(new_docs, 1):
        source = str(doc.metadata.get("source", f"imported-{idx}"))
        pseudo_hash = document_id_from_hash(str(abs(hash((source, doc.page_content)))))
        document = PaperDocument(
            document_id=pseudo_hash,
            source_path=source,
            filename=source.split("\\")[-1].split("/")[-1],
            file_type="imported",
            title=doc.metadata.get("title") or source,
            sha256=pseudo_hash.replace("doc_", "").ljust(64, "0")[:64],
            parser="langchain-import",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            metadata=dict(doc.metadata),
        )
        element = ContentElement(
            element_id=f"{document.document_id}_el_000001",
            document_id=document.document_id,
            sequence=1,
            type="text",
            text=clean_text(doc.page_content),
            page_idx=doc.metadata.get("page"),
            metadata=dict(doc.metadata),
        )
        imported.append(ParsedDocument(document=document, elements=[element], assets=[]))

    for parsed in imported:
        service.save_parsed_document(parsed)
    service.rebuild_vector_index()
    return {"status": "ready", "imported": len(imported)}
