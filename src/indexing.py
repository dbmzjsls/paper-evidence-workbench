from __future__ import annotations

import json
import pickle
import uuid
from pathlib import Path
from typing import Iterable

from src.config import Config
from src.evidence import build_evidence_chunks
from src.models import EvidenceChunk, IngestionJob, ParsedDocument
from src.parsers import discover_files, parse_document
from src.parsers.base import hash_file
from src.storage import CorpusStorage


class IndexingService:
    def __init__(
        self,
        storage: CorpusStorage | None = None,
        build_vectors: bool = True,
    ):
        self.storage = storage or CorpusStorage()
        self.build_vectors = build_vectors

    def ingest_path(
        self,
        path: str | Path,
        recursive: bool = True,
        force: bool = False,
        job_id: str | None = None,
    ) -> IngestionJob:
        files = discover_files(path, recursive=recursive)
        return self.ingest_files(files, force=force, job_id=job_id)

    def ingest_files(
        self,
        files: Iterable[str | Path],
        force: bool = False,
        job_id: str | None = None,
    ) -> IngestionJob:
        paths = [Path(p) for p in files]
        job = IngestionJob(
            job_id=job_id or f"job_{uuid.uuid4().hex[:12]}",
            status="running",
            total_files=len(paths),
            metadata={"files": [str(p) for p in paths]},
        )
        self.storage.create_job(job)

        processed = 0
        failed = 0
        document_ids: list[str] = []
        failures: list[dict[str, str]] = []

        for path in paths:
            try:
                file_hash = hash_file(path)
                existing = self.storage.find_document_by_hash(file_hash)
                if existing and not force:
                    document_ids.append(existing.document_id)
                    processed += 1
                    self.storage.update_job(
                        job.job_id,
                        processed_files=processed,
                        failed_files=failed,
                        message=f"Skipped duplicate: {path.name}",
                        metadata={**job.metadata, "document_ids": document_ids, "failures": failures},
                    )
                    continue

                parsed = parse_document(path, force=force)
                existing = self.storage.find_document_by_hash(parsed.document.sha256)
                if existing and not force:
                    document_ids.append(existing.document_id)
                    processed += 1
                    self.storage.update_job(
                        job.job_id,
                        processed_files=processed,
                        failed_files=failed,
                        message=f"Skipped duplicate: {path.name}",
                        metadata={**job.metadata, "document_ids": document_ids, "failures": failures},
                    )
                    continue

                self.save_parsed_document(parsed)
                document_ids.append(parsed.document.document_id)
                processed += 1
                self.storage.update_job(
                    job.job_id,
                    processed_files=processed,
                    failed_files=failed,
                    message=f"Indexed: {path.name}",
                    metadata={**job.metadata, "document_ids": document_ids, "failures": failures},
                )
            except Exception as exc:
                failed += 1
                failures.append({"file": str(path), "error": str(exc)})
                self.storage.update_job(
                    job.job_id,
                    processed_files=processed,
                    failed_files=failed,
                    message=f"Failed: {path.name}",
                    metadata={**job.metadata, "document_ids": document_ids, "failures": failures},
                )

        if self.build_vectors:
            try:
                self.rebuild_vector_index()
            except Exception as exc:
                failures.append({"file": "vector_index", "error": str(exc)})
                failed += 1

        status = "completed" if failed == 0 else ("partial" if processed else "failed")
        self.storage.update_job(
            job.job_id,
            status=status,
            processed_files=processed,
            failed_files=failed,
            message=f"{processed} processed, {failed} failed",
            metadata={**job.metadata, "document_ids": document_ids, "failures": failures},
        )
        return self.storage.get_job(job.job_id) or job

    def rebuild_corpus(
        self,
        path: str | Path,
        recursive: bool = True,
        job_id: str | None = None,
    ) -> IngestionJob:
        self.storage.clear_corpus()
        self._remove_vector_files()
        return self.ingest_path(path, recursive=recursive, force=True, job_id=job_id)

    def save_parsed_document(self, parsed: ParsedDocument) -> list[EvidenceChunk]:
        chunks = build_evidence_chunks(parsed.document, parsed.elements)
        self.storage.save_document(parsed.document)
        self.storage.save_elements(parsed.document.document_id, parsed.elements)
        self.storage.save_assets(parsed.document.document_id, parsed.assets)
        self.storage.save_chunks(parsed.document.document_id, chunks)
        return chunks

    def rebuild_vector_index(self) -> dict:
        chunks = self.storage.get_all_chunks()
        if not chunks:
            self._remove_vector_files()
            return {"status": "empty", "chunk_count": 0}

        documents = [self._to_langchain_document(chunk) for chunk in chunks]
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": Config.EMBEDDING_DEVICE},
        )
        vectorstore = FAISS.from_documents(documents, embeddings)
        Path(Config.VECTOR_DIR).mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(Config.VECTOR_DIR)

        # Compatibility only: old evaluation scripts still load docs.pkl.
        with open(Config.DOCS_DIR, "wb") as f:
            pickle.dump(documents, f)
        metadata_path = Path(Config.VECTOR_DIR) / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunk_count": len(chunks),
                    "embedding_model": Config.EMBEDDING_MODEL,
                    "source": "sqlite",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return {"status": "ready", "chunk_count": len(chunks)}

    def stats(self) -> dict:
        stats = self.storage.stats()
        index_path = Path(Config.VECTOR_DIR) / "index.faiss"
        metadata_path = Path(Config.VECTOR_DIR) / "metadata.json"
        index_chunk_count = None
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    index_chunk_count = json.load(f).get("chunk_count")
            except (OSError, json.JSONDecodeError):
                index_chunk_count = None

        if stats["chunk_count"] == 0:
            status = "empty"
        elif not index_path.exists():
            status = "not_built"
        elif index_chunk_count != stats["chunk_count"]:
            status = "stale_index"
        else:
            status = "ready"

        vector_search_enabled = status == "ready" and Config.TRUST_LOCAL_FAISS_INDEX
        if status == "stale_index":
            vector_search_warning = "Local FAISS index is stale. Rebuild the index before vector search."
        elif status == "ready" and not Config.TRUST_LOCAL_FAISS_INDEX:
            vector_search_warning = (
                "Local FAISS index is present but vector search is disabled. "
                "Set TRUST_LOCAL_FAISS_INDEX=true only for trusted index files."
            )
        else:
            vector_search_warning = ""
        stats.update(
            {
                "status": status,
                "index_size_mb": round(index_path.stat().st_size / (1024 * 1024), 2)
                if index_path.exists()
                else 0,
                "index_chunk_count": index_chunk_count,
                "embedding_model": Config.EMBEDDING_MODEL,
                "sqlite_path": Config.SQLITE_PATH,
                "data_dir": Config.DATA_DIR,
                "vector_search_enabled": vector_search_enabled,
                "vector_search_warning": vector_search_warning,
            }
        )
        return stats

    def _to_langchain_document(self, chunk: EvidenceChunk):
        from langchain_core.documents import Document

        return Document(
            page_content=chunk.search_text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "element_ids": chunk.element_ids,
                "type": chunk.type,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section": chunk.section,
                **chunk.metadata,
            },
        )

    def _remove_vector_files(self) -> None:
        vector_dir = Path(Config.VECTOR_DIR)
        for filename in ("index.faiss", "index.pkl", "metadata.json"):
            target = vector_dir / filename
            if target.exists():
                target.unlink()
        docs = Path(Config.DOCS_DIR)
        if docs.exists():
            docs.unlink()


def build_and_save_db(rebuild: bool = False, use_mineru=None):
    service = IndexingService()
    if rebuild:
        return service.rebuild_corpus(Config.DATA_DIR)
    return service.ingest_path(Config.DATA_DIR)


def get_db_stats() -> dict:
    return IndexingService(build_vectors=False).stats()
