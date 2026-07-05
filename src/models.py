from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class AssetRef(BaseModel):
    asset_id: str
    document_id: str
    kind: str
    path: str
    page_idx: Optional[int] = None
    bbox: Optional[list[float]] = None
    caption: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperDocument(BaseModel):
    document_id: str
    source_path: str
    filename: str
    file_type: str
    title: str
    sha256: str
    parser: str
    status: str = "ready"
    summary: str = ""
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContentElement(BaseModel):
    element_id: str
    document_id: str
    sequence: int
    type: str
    text: str = ""
    page_idx: Optional[int] = None
    section: str = ""
    bbox: Optional[list[float]] = None
    asset_path: str = ""
    html: str = ""
    latex: str = ""
    caption: str = ""
    footnote: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceChunk(BaseModel):
    chunk_id: str
    document_id: str
    element_ids: list[str]
    text: str
    search_text: str
    type: str = "text"
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    citation_id: str
    document_id: str
    chunk_id: str
    title: str
    filename: str
    page: Optional[int] = None
    section: str = ""
    quote: str = ""
    element_ids: list[str] = Field(default_factory=list)
    asset_path: str = ""
    evidence_type: str = "text"


class QueryResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[EvidenceChunk] = Field(default_factory=list)


class ScreeningItem(BaseModel):
    document_id: str
    title: str
    filename: str
    relevance_score: float
    decision: str
    core_contribution: str
    methods_data: str
    main_findings: str
    limitations: str
    research_ideas: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class ScreeningReport(BaseModel):
    report_id: str
    topic: str
    created_at: str = Field(default_factory=utc_now_iso)
    items: list[ScreeningItem] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class IngestionJob(BaseModel):
    job_id: str
    status: str
    message: str = ""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    document: PaperDocument
    elements: list[ContentElement]
    assets: list[AssetRef] = Field(default_factory=list)


def model_to_json(model: BaseModel) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()
    return model.json()


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
