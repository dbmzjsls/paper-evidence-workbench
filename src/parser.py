"""
Backward-compatible MinerU parser wrapper.

MinerU 3.x writes *_content_list_v2.json, *_content_list.json, *_middle.json,
Markdown, and images/. Older code in this project looked for *_content.json;
this wrapper keeps the old class name while using the new output contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.config import Config
from src.parsers.mineru import MinerUDocumentParser


class MinerUParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")
        self.pdf_name = self.pdf_path.stem
        self._parser = MinerUDocumentParser()

    @property
    def output_dir(self) -> Path:
        found = self._parser._find_parse_dir(self.pdf_path)
        return found or Path(Config.MINERU_OUTPUT_DIR) / self.pdf_name

    def parse_pdf(self) -> None:
        self._parser._run_mineru(self.pdf_path)

    def load_json(self) -> list:
        parse_dir = self._parser._find_parse_dir(self.pdf_path)
        if parse_dir is None:
            raise FileNotFoundError(
                f"No MinerU output found for {self.pdf_path}. Run parse_pdf() first."
            )
        for pattern in ("*_content_list_v2.json", "*_content_list.json", "*_middle.json"):
            files = list(parse_dir.glob(pattern))
            if files:
                return json.loads(files[0].read_text(encoding="utf-8"))
        raise FileNotFoundError(f"No MinerU JSON output found in {parse_dir}")

    def load_json_if_exists(self) -> Optional[list]:
        try:
            return self.load_json()
        except FileNotFoundError:
            return None

    @property
    def is_parsed(self) -> bool:
        return self._parser._find_parse_dir(self.pdf_path) is not None
