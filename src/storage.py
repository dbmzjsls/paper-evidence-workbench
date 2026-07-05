from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterable, Optional, TypeVar

from src.config import Config
from src.models import (
    AssetRef,
    ContentElement,
    EvidenceChunk,
    IngestionJob,
    PaperDocument,
    ScreeningReport,
    model_to_dict,
    utc_now_iso,
)

T = TypeVar("T")


def _json(data: Any) -> str:
    return json.dumps(data or {}, ensure_ascii=False)


def _loads(data: Optional[str], default: Any) -> Any:
    if not data:
        return default
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return default


class CorpusStorage:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or Config.SQLITE_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    parser TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS elements (
                    element_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    text TEXT NOT NULL DEFAULT '',
                    page_idx INTEGER,
                    section TEXT NOT NULL DEFAULT '',
                    bbox TEXT,
                    asset_path TEXT NOT NULL DEFAULT '',
                    html TEXT NOT NULL DEFAULT '',
                    latex TEXT NOT NULL DEFAULT '',
                    caption TEXT NOT NULL DEFAULT '',
                    footnote TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );

                CREATE INDEX IF NOT EXISTS idx_elements_document
                    ON elements(document_id, sequence);

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    element_ids TEXT NOT NULL DEFAULT '[]',
                    text TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    type TEXT NOT NULL,
                    page_start INTEGER,
                    page_end INTEGER,
                    section TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document
                    ON chunks(document_id);

                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    page_idx INTEGER,
                    bbox TEXT,
                    caption TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );

                CREATE INDEX IF NOT EXISTS idx_assets_document
                    ON assets(document_id);

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    total_files INTEGER NOT NULL DEFAULT 0,
                    processed_files INTEGER NOT NULL DEFAULT 0,
                    failed_files INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS reports (
                    report_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                """
            )

    def clear_corpus(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM reports")
            conn.execute("DELETE FROM assets")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM elements")
            conn.execute("DELETE FROM documents")

    def save_document(self, document: PaperDocument) -> None:
        data = model_to_dict(document)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    document_id, source_path, filename, file_type, title, sha256,
                    parser, status, summary, created_at, updated_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["document_id"],
                    data["source_path"],
                    data["filename"],
                    data["file_type"],
                    data["title"],
                    data["sha256"],
                    data["parser"],
                    data["status"],
                    data.get("summary", ""),
                    data["created_at"],
                    data["updated_at"],
                    _json(data.get("metadata")),
                ),
            )

    def delete_document_payload(self, document_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM assets WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM elements WHERE document_id = ?", (document_id,))

    def save_elements(self, document_id: str, elements: list[ContentElement]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM elements WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT OR REPLACE INTO elements (
                    element_id, document_id, sequence, type, text, page_idx, section,
                    bbox, asset_path, html, latex, caption, footnote, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.element_id,
                        e.document_id,
                        e.sequence,
                        e.type,
                        e.text,
                        e.page_idx,
                        e.section,
                        _json(e.bbox) if e.bbox is not None else None,
                        e.asset_path,
                        e.html,
                        e.latex,
                        e.caption,
                        e.footnote,
                        _json(e.metadata),
                    )
                    for e in elements
                ],
            )

    def save_chunks(self, document_id: str, chunks: list[EvidenceChunk]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id, document_id, element_ids, text, search_text, type,
                    page_start, page_end, section, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.chunk_id,
                        c.document_id,
                        _json(c.element_ids),
                        c.text,
                        c.search_text,
                        c.type,
                        c.page_start,
                        c.page_end,
                        c.section,
                        _json(c.metadata),
                    )
                    for c in chunks
                ],
            )

    def save_assets(self, document_id: str, assets: list[AssetRef]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM assets WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT OR REPLACE INTO assets (
                    asset_id, document_id, kind, path, page_idx, bbox, caption, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        a.asset_id,
                        a.document_id,
                        a.kind,
                        a.path,
                        a.page_idx,
                        _json(a.bbox) if a.bbox is not None else None,
                        a.caption,
                        _json(a.metadata),
                    )
                    for a in assets
                ],
            )

    def find_document_by_hash(self, sha256: str) -> Optional[PaperDocument]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return self._row_to_document(row) if row else None

    def get_document(self, document_id: str) -> Optional[PaperDocument]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(self, limit: int = 200, offset: int = 0) -> list[PaperDocument]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM documents
                ORDER BY updated_at DESC, filename ASC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_elements(self, document_id: str, limit: int = 1000) -> list[ContentElement]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM elements
                WHERE document_id = ?
                ORDER BY sequence ASC
                LIMIT ?
                """,
                (document_id, limit),
            ).fetchall()
        return [self._row_to_element(row) for row in rows]

    def get_assets(self, document_id: str) -> list[AssetRef]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM assets WHERE document_id = ? ORDER BY page_idx, asset_id",
                (document_id,),
            ).fetchall()
        return [self._row_to_asset(row) for row in rows]

    def get_chunks(self, document_id: str) -> list[EvidenceChunk]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_id",
                (document_id,),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_all_chunks(self) -> list[EvidenceChunk]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM chunks ORDER BY rowid ASC").fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> Optional[EvidenceChunk]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks_by_ids(self, chunk_ids: Iterable[str]) -> list[EvidenceChunk]:
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", ids
            ).fetchall()
        by_id = {row["chunk_id"]: self._row_to_chunk(row) for row in rows}
        return [by_id[i] for i in ids if i in by_id]

    def create_job(self, job: IngestionJob) -> None:
        data = model_to_dict(job)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO jobs (
                    job_id, status, message, total_files, processed_files,
                    failed_files, created_at, updated_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["job_id"],
                    data["status"],
                    data.get("message", ""),
                    data.get("total_files", 0),
                    data.get("processed_files", 0),
                    data.get("failed_files", 0),
                    data["created_at"],
                    data["updated_at"],
                    _json(data.get("metadata")),
                ),
            )

    def update_job(self, job_id: str, **updates: Any) -> None:
        allowed = {
            "status",
            "message",
            "total_files",
            "processed_files",
            "failed_files",
            "metadata",
        }
        updates = {k: v for k, v in updates.items() if k in allowed}
        updates["updated_at"] = utc_now_iso()
        if "metadata" in updates:
            updates["metadata"] = _json(updates["metadata"])
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [job_id]
        with self.connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {assignments} WHERE job_id = ?",
                values,
            )

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def save_report(self, report: ScreeningReport) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reports (report_id, topic, created_at, payload)
                VALUES (?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    report.topic,
                    report.created_at,
                    json.dumps(model_to_dict(report), ensure_ascii=False),
                ),
            )

    def stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            element_count = conn.execute("SELECT COUNT(*) FROM elements").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            asset_count = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
            job_count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        return {
            "document_count": document_count,
            "element_count": element_count,
            "chunk_count": chunk_count,
            "asset_count": asset_count,
            "job_count": job_count,
        }

    def _row_to_document(self, row: sqlite3.Row) -> PaperDocument:
        return PaperDocument(
            document_id=row["document_id"],
            source_path=row["source_path"],
            filename=row["filename"],
            file_type=row["file_type"],
            title=row["title"],
            sha256=row["sha256"],
            parser=row["parser"],
            status=row["status"],
            summary=row["summary"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_loads(row["metadata"], {}),
        )

    def _row_to_element(self, row: sqlite3.Row) -> ContentElement:
        return ContentElement(
            element_id=row["element_id"],
            document_id=row["document_id"],
            sequence=row["sequence"],
            type=row["type"],
            text=row["text"],
            page_idx=row["page_idx"],
            section=row["section"],
            bbox=_loads(row["bbox"], None),
            asset_path=row["asset_path"],
            html=row["html"],
            latex=row["latex"],
            caption=row["caption"],
            footnote=row["footnote"],
            metadata=_loads(row["metadata"], {}),
        )

    def _row_to_chunk(self, row: sqlite3.Row) -> EvidenceChunk:
        return EvidenceChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            element_ids=_loads(row["element_ids"], []),
            text=row["text"],
            search_text=row["search_text"],
            type=row["type"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            section=row["section"],
            metadata=_loads(row["metadata"], {}),
        )

    def _row_to_asset(self, row: sqlite3.Row) -> AssetRef:
        return AssetRef(
            asset_id=row["asset_id"],
            document_id=row["document_id"],
            kind=row["kind"],
            path=row["path"],
            page_idx=row["page_idx"],
            bbox=_loads(row["bbox"], None),
            caption=row["caption"],
            metadata=_loads(row["metadata"], {}),
        )

    def _row_to_job(self, row: sqlite3.Row) -> IngestionJob:
        return IngestionJob(
            job_id=row["job_id"],
            status=row["status"],
            message=row["message"],
            total_files=row["total_files"],
            processed_files=row["processed_files"],
            failed_files=row["failed_files"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_loads(row["metadata"], {}),
        )
