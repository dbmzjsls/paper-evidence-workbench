from __future__ import annotations

import csv
import html
import re
from pathlib import Path

from src.models import ParsedDocument
from src.parsers.base import BaseParser, clean_text, first_non_empty


class PlainTextParser(BaseParser):
    parser_name = "plain"

    def parse(self, file_path: str | Path, force: bool = False) -> ParsedDocument:
        path = Path(file_path)
        suffix = path.suffix.lower()
        text = self._read_text(path, suffix)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        title = first_non_empty(paragraphs, path.stem)
        document = self.build_document(path, title=title, parser=self.parser_name)

        elements = []
        for idx, paragraph in enumerate(paragraphs or [text], 1):
            element_type = "table" if suffix == ".csv" else "text"
            elements.append(
                self.text_element(
                    document.document_id,
                    idx,
                    paragraph,
                    element_type=element_type,
                    metadata={"source_parser": self.parser_name},
                )
            )

        return ParsedDocument(document=document, elements=elements, assets=[])

    def _read_text(self, path: Path, suffix: str) -> str:
        if suffix == ".csv":
            return self._read_csv(path)

        raw = path.read_text(encoding="utf-8", errors="ignore")
        if suffix in {".html", ".htm"}:
            return self._strip_html(raw)
        return clean_text(raw)

    def _read_csv(self, path: Path) -> str:
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= 500:
                    rows.append(["..."])
                    break
                rows.append([cell.strip() for cell in row])

        if not rows:
            return ""

        width = max(len(row) for row in rows)
        padded = [row + [""] * (width - len(row)) for row in rows]
        lines = [" | ".join(row) for row in padded]
        return clean_text("\n".join(lines))

    def _strip_html(self, raw: str) -> str:
        raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
        raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
        raw = re.sub(r"(?i)</(p|div|section|article|h[1-6]|li|tr)>", "\n", raw)
        raw = re.sub(r"(?s)<[^>]+>", " ", raw)
        return clean_text(html.unescape(raw))


class PyPDFTextParser(BaseParser):
    parser_name = "pypdf"

    def parse(self, file_path: str | Path, force: bool = False) -> ParsedDocument:
        from pypdf import PdfReader

        path = Path(file_path)
        reader = PdfReader(str(path))
        page_texts = []
        for idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                page_texts.append((idx, clean_text(text)))

        title = ""
        try:
            title = (reader.metadata.title or "").strip() if reader.metadata else ""
        except Exception:
            title = ""
        title = title or first_non_empty((text for _, text in page_texts), path.stem)
        document = self.build_document(
            path,
            title=title,
            parser=self.parser_name,
            metadata={"fallback": True, "total_pages": len(reader.pages)},
        )
        elements = [
            self.text_element(
                document.document_id,
                i + 1,
                text,
                page_idx=page_idx,
                metadata={"source_parser": self.parser_name},
            )
            for i, (page_idx, text) in enumerate(page_texts)
        ]
        return ParsedDocument(document=document, elements=elements, assets=[])
