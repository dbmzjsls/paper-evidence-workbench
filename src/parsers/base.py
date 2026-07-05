from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from src.models import ContentElement, PaperDocument, ParsedDocument, utc_now_iso


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_id_from_hash(sha256: str) -> str:
    return f"doc_{sha256[:20]}"


def stable_element_id(document_id: str, sequence: int) -> str:
    return f"{document_id}_el_{sequence:06d}"


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def flatten_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return clean_text(" ".join(flatten_text(item) for item in value if item is not None))
    if isinstance(value, dict):
        for key in (
            "text",
            "content",
            "value",
            "title_content",
            "paragraph_content",
            "math_content",
            "code_content",
            "item_content",
        ):
            if key in value:
                return flatten_text(value[key])
        return clean_text(" ".join(flatten_text(v) for v in value.values()))
    return str(value)


def first_non_empty(lines: Iterable[str], default: str) -> str:
    for line in lines:
        line = clean_text(line)
        if line:
            return line[:200]
    return default


class BaseParser:
    parser_name = "base"

    def parse(self, file_path: str | Path, force: bool = False) -> ParsedDocument:
        raise NotImplementedError

    def build_document(self, path: Path, title: str, parser: str, metadata=None) -> PaperDocument:
        sha = hash_file(path)
        now = utc_now_iso()
        return PaperDocument(
            document_id=document_id_from_hash(sha),
            source_path=str(path.resolve()),
            filename=path.name,
            file_type=path.suffix.lower().lstrip("."),
            title=title or path.stem,
            sha256=sha,
            parser=parser,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

    def text_element(
        self,
        document_id: str,
        sequence: int,
        text: str,
        element_type: str = "text",
        section: str = "",
        page_idx: int | None = None,
        metadata=None,
    ) -> ContentElement:
        return ContentElement(
            element_id=stable_element_id(document_id, sequence),
            document_id=document_id,
            sequence=sequence,
            type=element_type,
            text=clean_text(text),
            page_idx=page_idx,
            section=section,
            metadata=metadata or {},
        )
