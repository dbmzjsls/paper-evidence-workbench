from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.models import ParsedDocument
from src.parsers.mineru import MinerUDocumentParser
from src.parsers.plain import PlainTextParser, PyPDFTextParser


class UnsupportedFileType(ValueError):
    pass


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in Config.SUPPORTED_EXTENSIONS


def discover_files(path: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root] if is_supported(root) else []

    pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in root.glob(pattern)
        if p.is_file() and is_supported(p)
    ]
    return sorted(files, key=lambda p: str(p).lower())


def parse_document(file_path: str | Path, force: bool = False) -> ParsedDocument:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in Config.SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(f"Unsupported file type: {suffix}")

    if suffix in Config.PLAIN_EXTENSIONS:
        return PlainTextParser().parse(path, force=force)

    if suffix in Config.MINERU_EXTENSIONS:
        try:
            return MinerUDocumentParser().parse(path, force=force)
        except Exception:
            if suffix == ".pdf":
                return PyPDFTextParser().parse(path, force=force)
            raise

    raise UnsupportedFileType(f"Unsupported file type: {suffix}")


__all__ = [
    "UnsupportedFileType",
    "discover_files",
    "is_supported",
    "parse_document",
]
