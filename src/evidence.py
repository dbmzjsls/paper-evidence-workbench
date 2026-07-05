from __future__ import annotations

import re

from src.config import Config
from src.models import ContentElement, EvidenceChunk, PaperDocument
from src.parsers.base import clean_text


VISUAL_ELEMENT_TYPES = {"image", "chart", "table", "equation", "code", "algorithm"}


def build_evidence_chunks(
    document: PaperDocument,
    elements: list[ContentElement],
    max_chars: int | None = None,
) -> list[EvidenceChunk]:
    max_chars = max_chars or max(Config.CHUNK_SIZE, 500)
    chunks: list[EvidenceChunk] = []
    current: list[ContentElement] = []
    previous_context = ""

    def flush() -> None:
        nonlocal current, previous_context
        if not current:
            return
        text = clean_text("\n".join(_element_text(e) for e in current if _element_text(e)))
        if not text:
            current = []
            return
        chunk = _make_chunk(document, current, text, len(chunks) + 1, "text")
        chunks.append(chunk)
        previous_context = text[-600:]
        current = []

    for element in sorted(elements, key=lambda e: e.sequence):
        text = _element_text(element)
        if not text and not element.asset_path:
            continue

        if element.type in VISUAL_ELEMENT_TYPES:
            flush()
            visual_text = _visual_text(element, previous_context)
            chunk = _make_chunk(document, [element], visual_text, len(chunks) + 1, element.type)
            chunks.append(chunk)
            previous_context = clean_text((previous_context + "\n" + visual_text)[-600:])
            continue

        candidate = clean_text("\n".join(_element_text(e) for e in current + [element]))
        if current and len(candidate) > max_chars:
            flush()
        current.append(element)

    flush()
    return chunks


def _make_chunk(
    document: PaperDocument,
    elements: list[ContentElement],
    text: str,
    index: int,
    chunk_type: str,
) -> EvidenceChunk:
    pages = [e.page_idx for e in elements if e.page_idx is not None]
    section = next((e.section for e in reversed(elements) if e.section), "")
    evidence_text = clean_text(text)
    search_text = clean_text(
        "\n".join(
            [
                f"Title: {document.title}",
                f"File: {document.filename}",
                f"Section: {section}" if section else "",
                evidence_text,
            ]
        )
    )
    asset_path = next((e.asset_path for e in elements if e.asset_path), "")
    return EvidenceChunk(
        chunk_id=f"{document.document_id}_chunk_{index:06d}",
        document_id=document.document_id,
        element_ids=[e.element_id for e in elements],
        text=evidence_text,
        search_text=search_text,
        type=chunk_type,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        section=section,
        metadata={
            "filename": document.filename,
            "title": document.title,
            "asset_path": asset_path,
            "element_types": [e.type for e in elements],
        },
    )


def _element_text(element: ContentElement) -> str:
    parts = []
    if element.type == "title" and element.text:
        parts.append(f"# {element.text}")
    elif element.text:
        parts.append(element.text)
    if element.latex and element.latex not in parts:
        parts.append(f"LaTeX: {element.latex}")
    if element.caption:
        parts.append(f"Caption: {element.caption}")
    if element.footnote:
        parts.append(f"Footnote: {element.footnote}")
    if element.html:
        parts.append(_html_to_text(element.html))
    return clean_text("\n".join(parts))


def _visual_text(element: ContentElement, previous_context: str) -> str:
    label = element.type.upper()
    parts = [
        previous_context[-400:] if previous_context else "",
        f"[{label}]",
        _element_text(element),
    ]
    if element.asset_path:
        parts.append(f"Asset: {element.asset_path}")
    return clean_text("\n".join(parts))


def _html_to_text(raw_html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return clean_text(text)
