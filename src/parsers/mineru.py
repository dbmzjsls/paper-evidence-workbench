from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from src.config import Config
from src.models import AssetRef, ContentElement, ParsedDocument
from src.parsers.base import (
    BaseParser,
    clean_text,
    flatten_text,
    first_non_empty,
    stable_element_id,
)


VISUAL_TYPES = {
    "image",
    "chart",
    "table",
    "table_simple",
    "table_complex",
    "equation",
    "equation_interline",
    "interline_equation",
    "code",
    "algorithm",
}


class MinerUDocumentParser(BaseParser):
    parser_name = "mineru"

    def __init__(self, output_dir: str | Path | None = None, asset_dir: str | Path | None = None):
        self.output_dir = Path(output_dir or Config.MINERU_OUTPUT_DIR)
        self.asset_dir = Path(asset_dir or Config.ASSET_DIR)

    def parse(self, file_path: str | Path, force: bool = False) -> ParsedDocument:
        path = Path(file_path)
        document = self.build_document(
            path,
            title=path.stem,
            parser=self.parser_name,
            metadata={
                "mineru_backend": Config.MINERU_BACKEND,
                "mineru_method": Config.MINERU_METHOD,
            },
        )

        parse_dir = self._find_parse_dir(path)
        if force or parse_dir is None:
            self._run_mineru(path)
            parse_dir = self._find_parse_dir(path)

        if parse_dir is None:
            raise FileNotFoundError(f"MinerU output not found for {path}")

        elements, assets = self._load_elements(parse_dir, document.document_id, path)
        if not elements:
            raise ValueError(f"No readable content found in MinerU output: {parse_dir}")

        title = self._infer_title(elements, path.stem)
        document.title = title
        document.metadata.update({"parse_dir": str(parse_dir.resolve())})
        return ParsedDocument(document=document, elements=elements, assets=assets)

    def _run_mineru(self, path: Path) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "mineru",
            "-p",
            str(path),
            "-o",
            str(self.output_dir),
            "-b",
            Config.MINERU_BACKEND,
            "-m",
            Config.MINERU_METHOD,
            "-l",
            Config.MINERU_LANG,
            "-f",
            str(Config.MINERU_FORMULA).lower(),
            "-t",
            str(Config.MINERU_TABLE).lower(),
            "--image-analysis",
            str(Config.MINERU_IMAGE_ANALYSIS).lower(),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=Config.MINERU_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            stderr = result.stderr[-1000:] if result.stderr else ""
            raise RuntimeError(f"MinerU failed for {path.name}: {stderr}")

    def _find_parse_dir(self, path: Path) -> Optional[Path]:
        root = self.output_dir / path.stem
        if not root.exists():
            return None

        candidates = [
            root / "office",
            root / Config.MINERU_METHOD,
            root / f"hybrid_{Config.MINERU_METHOD}",
            root / "vlm",
        ]
        candidates.extend([p for p in root.iterdir() if p.is_dir()])

        for candidate in candidates:
            if not candidate.exists():
                continue
            if (
                list(candidate.glob("*_content_list_v2.json"))
                or list(candidate.glob("*_content_list.json"))
                or list(candidate.glob("*.md"))
            ):
                return candidate
        return None

    def _load_elements(
        self,
        parse_dir: Path,
        document_id: str,
        source_path: Path,
    ) -> tuple[list[ContentElement], list[AssetRef]]:
        v2_files = list(parse_dir.glob("*_content_list_v2.json"))
        if v2_files:
            data = json.loads(v2_files[0].read_text(encoding="utf-8"))
            return self._from_content_list_v2(data, parse_dir, document_id)

        v1_files = list(parse_dir.glob("*_content_list.json"))
        if v1_files:
            data = json.loads(v1_files[0].read_text(encoding="utf-8"))
            return self._from_content_list(data, parse_dir, document_id)

        md_files = [p for p in parse_dir.glob("*.md") if not p.name.endswith("_origin.md")]
        if md_files:
            text = md_files[0].read_text(encoding="utf-8", errors="ignore")
            element = self.text_element(document_id, 1, text, metadata={"parse_dir": str(parse_dir)})
            return [element], []

        # Last resort for a text-based source if MinerU produced nothing usable.
        if source_path.suffix.lower() in {".txt", ".md", ".markdown"}:
            text = source_path.read_text(encoding="utf-8", errors="ignore")
            element = self.text_element(document_id, 1, text)
            return [element], []

        return [], []

    def _from_content_list_v2(
        self,
        data: list[dict[str, Any]],
        parse_dir: Path,
        document_id: str,
    ) -> tuple[list[ContentElement], list[AssetRef]]:
        elements: list[ContentElement] = []
        assets: list[AssetRef] = []
        current_section = ""

        for idx, item in enumerate(data, 1):
            raw_type = str(item.get("type", "paragraph")).lower()
            content = item.get("content") or {}
            page_idx = self._page_idx(item)
            bbox = item.get("bbox")

            text = ""
            html = ""
            latex = ""
            asset_path = ""
            caption = ""
            footnote = ""
            element_type = self._normalize_type(raw_type)

            if raw_type == "title":
                text = flatten_text(content.get("title_content") or content)
                current_section = text or current_section
                element_type = "title"
            elif raw_type in {"paragraph", "page_header", "page_footer", "page_number"}:
                text = flatten_text(content.get("paragraph_content") or content)
                element_type = "text"
            elif "equation" in raw_type:
                latex = flatten_text(content.get("math_content"))
                text = latex
                asset_path = self._copy_asset(
                    parse_dir,
                    document_id,
                    self._image_path(content),
                )
                element_type = "equation"
            elif raw_type == "table":
                html = content.get("html", "") or ""
                caption = flatten_text(content.get("table_caption"))
                footnote = flatten_text(content.get("table_footnote"))
                text = clean_text(" ".join([caption, self._html_to_text(html), footnote]))
                asset_path = self._copy_asset(parse_dir, document_id, self._image_path(content))
                element_type = "table"
            elif raw_type in {"image", "chart"}:
                caption = flatten_text(content.get(f"{raw_type}_caption"))
                footnote = flatten_text(content.get(f"{raw_type}_footnote"))
                text = clean_text(" ".join([caption, footnote]))
                asset_path = self._copy_asset(parse_dir, document_id, self._image_path(content))
                element_type = raw_type
            elif raw_type in {"code", "algorithm"}:
                text = flatten_text(content.get("code_content") or content.get("algorithm_content") or content)
                caption = flatten_text(content.get("code_caption") or content.get("algorithm_caption"))
                footnote = flatten_text(content.get("code_footnote") or content.get("algorithm_footnote"))
                element_type = raw_type
            elif raw_type == "list":
                text = flatten_text(content.get("list_items") or content)
                element_type = "list"
            else:
                text = flatten_text(content or item)

            element = ContentElement(
                element_id=stable_element_id(document_id, idx),
                document_id=document_id,
                sequence=idx,
                type=element_type,
                text=clean_text(text),
                page_idx=page_idx,
                section=current_section,
                bbox=bbox,
                asset_path=asset_path,
                html=html,
                latex=latex,
                caption=caption,
                footnote=footnote,
                metadata={"raw_type": raw_type, "parse_format": "content_list_v2"},
            )
            if self._is_useful(element):
                elements.append(element)
                asset = self._asset_from_element(element)
                if asset:
                    assets.append(asset)

        return elements, assets

    def _from_content_list(
        self,
        data: list[dict[str, Any]],
        parse_dir: Path,
        document_id: str,
    ) -> tuple[list[ContentElement], list[AssetRef]]:
        elements: list[ContentElement] = []
        assets: list[AssetRef] = []
        current_section = ""

        for idx, item in enumerate(data, 1):
            raw_type = str(item.get("type", "text")).lower()
            element_type = self._normalize_type(raw_type)
            page_idx = self._page_idx(item)
            bbox = item.get("bbox")
            text = flatten_text(item.get("text") or item.get("content") or "")
            html = item.get("html") or item.get("table_body") or ""
            latex = text if element_type == "equation" and item.get("text_format") == "latex" else ""
            caption = flatten_text(
                item.get("image_caption")
                or item.get("table_caption")
                or item.get("chart_caption")
                or item.get("caption")
            )
            footnote = flatten_text(
                item.get("image_footnote")
                or item.get("table_footnote")
                or item.get("chart_footnote")
                or item.get("footnote")
            )
            asset_path = self._copy_asset(parse_dir, document_id, item.get("img_path"))

            if element_type == "title":
                current_section = text or current_section
            if element_type == "table":
                text = clean_text(" ".join([caption, text, self._html_to_text(html), footnote]))
            elif element_type in {"image", "chart"}:
                text = clean_text(" ".join([caption, text, footnote]))

            element = ContentElement(
                element_id=stable_element_id(document_id, idx),
                document_id=document_id,
                sequence=idx,
                type=element_type,
                text=clean_text(text),
                page_idx=page_idx,
                section=current_section,
                bbox=bbox,
                asset_path=asset_path,
                html=html,
                latex=latex,
                caption=caption,
                footnote=footnote,
                metadata={"raw_type": raw_type, "parse_format": "content_list"},
            )
            if self._is_useful(element):
                elements.append(element)
                asset = self._asset_from_element(element)
                if asset:
                    assets.append(asset)

        return elements, assets

    def _normalize_type(self, raw_type: str) -> str:
        raw_type = raw_type.lower()
        if "title" in raw_type:
            return "title"
        if "equation" in raw_type or "formula" in raw_type:
            return "equation"
        if "table" in raw_type:
            return "table"
        if "chart" in raw_type:
            return "chart"
        if "image" in raw_type:
            return "image"
        if "code" in raw_type:
            return "code"
        if "list" in raw_type:
            return "list"
        return "text"

    def _page_idx(self, item: dict[str, Any]) -> int | None:
        if "page_idx" in item:
            return item.get("page_idx")
        return item.get("page")

    def _image_path(self, content: dict[str, Any]) -> str:
        source = content.get("image_source") or {}
        if isinstance(source, dict):
            return source.get("path", "") or ""
        return ""

    def _copy_asset(self, parse_dir: Path, document_id: str, rel_path: Any) -> str:
        rel_path = flatten_text(rel_path)
        if not rel_path:
            return ""

        parse_root = parse_dir.resolve()
        source = Path(rel_path)
        source = source.resolve() if source.is_absolute() else (parse_root / source).resolve()
        try:
            source.relative_to(parse_root)
        except ValueError:
            return ""
        if not source.exists() or not source.is_file():
            return ""

        target_dir = self.asset_dir / document_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if not target.exists():
            shutil.copy2(source, target)
        return str(target.resolve())

    def _asset_from_element(self, element: ContentElement) -> Optional[AssetRef]:
        if not element.asset_path:
            return None
        return AssetRef(
            asset_id=f"{element.element_id}_asset",
            document_id=element.document_id,
            kind=element.type,
            path=element.asset_path,
            page_idx=element.page_idx,
            bbox=element.bbox,
            caption=element.caption or element.text[:300],
            metadata={"element_id": element.element_id},
        )

    def _is_useful(self, element: ContentElement) -> bool:
        return bool(element.text or element.html or element.latex or element.asset_path)

    def _infer_title(self, elements: list[ContentElement], default: str) -> str:
        titles = [e.text for e in elements if e.type == "title" and e.text]
        if titles:
            return titles[0][:200]
        return first_non_empty((e.text for e in elements), default)

    def _html_to_text(self, html: str) -> str:
        if not html:
            return ""
        import re

        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        return clean_text(text)
