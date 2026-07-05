from __future__ import annotations

import json

from fastapi.testclient import TestClient

import api
from src.config import Config
from src.indexing import IndexingService
from src.models import ContentElement, PaperDocument, ParsedDocument, utc_now_iso
from src.parsers import parse_document
from src.parsers.base import clean_text
from src.parsers.mineru import MinerUDocumentParser
from src.retrieval import ResearchRAG
from src.storage import CorpusStorage


def test_mineru_content_list_v2_preserves_visual_evidence(tmp_path):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"not a real pdf; parser reads existing mineru output")

    parse_dir = tmp_path / "parsed" / "paper" / "auto"
    image_dir = parse_dir / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "eq.png").write_bytes(b"image")

    payload = [
        {
            "type": "title",
            "content": {"title_content": "A Useful Paper"},
            "page_idx": 0,
            "bbox": [0, 0, 100, 20],
        },
        {
            "type": "paragraph",
            "content": {"paragraph_content": "This paper proposes a robust method."},
            "page_idx": 0,
        },
        {
            "type": "equation_interline",
            "content": {
                "math_content": "E = mc^2",
                "math_type": "latex",
                "image_source": {"path": "images/eq.png"},
            },
            "page_idx": 1,
            "bbox": [10, 20, 80, 40],
        },
        {
            "type": "table",
            "content": {
                "table_caption": ["Table 1. Results"],
                "html": "<table><tr><td>Accuracy</td><td>0.91</td></tr></table>",
                "image_source": {"path": "images/eq.png"},
            },
            "page_idx": 2,
        },
    ]
    (parse_dir / "paper_content_list_v2.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    parsed = MinerUDocumentParser(
        output_dir=tmp_path / "parsed",
        asset_dir=tmp_path / "assets",
    ).parse(source)

    assert parsed.document.title == "A Useful Paper"
    assert [element.type for element in parsed.elements] == [
        "title",
        "text",
        "equation",
        "table",
    ]
    assert parsed.elements[0].page_idx == 0
    assert parsed.elements[1].page_idx == 0
    equation = parsed.elements[2]
    assert equation.latex == "E = mc^2"
    assert equation.page_idx == 1
    assert equation.bbox == [10, 20, 80, 40]
    assert equation.asset_path.endswith("eq.png")
    assert parsed.assets


def test_plain_csv_routes_to_table_element(tmp_path):
    source = tmp_path / "scores.csv"
    source.write_text("metric,value\naccuracy,0.91\n", encoding="utf-8")

    parsed = parse_document(source)

    assert parsed.document.file_type == "csv"
    assert parsed.elements[0].type == "table"
    assert "accuracy" in parsed.elements[0].text


def test_ingestion_deduplicates_before_parsing_and_retrieval_screens_without_vectors(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "paper.md"
    source.write_text(
        "# EV Review Study\n\n"
        "This paper proposes a method for analyzing negative online reviews.\n\n"
        "The findings show that negative word of mouth affects purchase intention.",
        encoding="utf-8",
    )
    storage = CorpusStorage(str(tmp_path / "corpus.sqlite3"))
    service = IndexingService(storage=storage, build_vectors=False)

    first = service.ingest_path(source)
    monkeypatch.setattr(
        "src.indexing.parse_document",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("duplicate parsed")),
    )
    second = service.ingest_path(source)

    assert first.status == "completed"
    assert second.status == "completed"
    assert storage.stats()["document_count"] == 1
    assert storage.stats()["chunk_count"] >= 1

    rag = ResearchRAG(storage=storage)
    answer = rag.answer("negative online reviews purchase intention")
    assert answer.citations
    assert "negative" in answer.citations[0].quote.lower()

    report = rag.screen("negative online reviews purchase intention", limit=3)
    assert report.items
    assert report.items[0].decision in {
        "值得优先精读",
        "可作为辅助阅读",
        "相关性较弱，先暂缓",
    }


def test_stats_distinguishes_empty_stale_and_ready_index(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "VECTOR_DIR", str(tmp_path / "vectors"))
    monkeypatch.setattr(Config, "DOCS_DIR", str(tmp_path / "docs.pkl"))
    storage = CorpusStorage(str(tmp_path / "corpus.sqlite3"))
    service = IndexingService(storage=storage, build_vectors=False)

    assert service.stats()["status"] == "empty"

    document = PaperDocument(
        document_id="doc_test",
        source_path=str(tmp_path / "paper.md"),
        filename="paper.md",
        file_type="md",
        title="Paper",
        sha256="a" * 64,
        parser="test",
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
    )
    element = ContentElement(
        element_id="doc_test_el_000001",
        document_id=document.document_id,
        sequence=1,
        type="text",
        text=clean_text("negative reviews affect purchase intention"),
    )
    service.save_parsed_document(ParsedDocument(document=document, elements=[element]))

    assert service.stats()["status"] == "not_built"

    vector_dir = tmp_path / "vectors"
    vector_dir.mkdir()
    (vector_dir / "index.faiss").write_bytes(b"fake")
    assert service.stats()["status"] == "stale_index"

    (vector_dir / "metadata.json").write_text(
        json.dumps({"chunk_count": storage.stats()["chunk_count"]}),
        encoding="utf-8",
    )
    assert service.stats()["status"] == "ready"


def test_upload_rejects_unsupported_batch_without_partial_files(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "UPLOAD_DIR", str(tmp_path / "uploads"))
    (tmp_path / "uploads").mkdir()
    client = TestClient(api.app)

    response = client.post(
        "/documents/upload",
        files=[
            ("files", ("paper.md", b"# ok", "text/markdown")),
            ("files", ("payload.exe", b"bad", "application/octet-stream")),
        ],
    )

    assert response.status_code == 400
    assert list((tmp_path / "uploads").iterdir()) == []
