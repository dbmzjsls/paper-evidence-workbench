from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from src.indexing import IndexingService
from src.models import AssetRef
from src.parsers import parse_document
from src.storage import CorpusStorage


def corpus_snapshot(storage):
    with storage.connect() as conn:
        return {
            name: [
                tuple(row)
                for row in conn.execute(f"SELECT * FROM {name} ORDER BY rowid")
            ]
            for name in ("documents", "elements", "assets", "chunks", "chunks_fts")
        }


@pytest.mark.parametrize("table", ["elements", "assets", "chunks"])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_document_write_rolls_back_and_can_retry(tmp_path, table, existing):
    source = tmp_path / "paper.md"
    source.write_text("original evidence", encoding="utf-8")
    storage = CorpusStorage(str(tmp_path / "corpus.sqlite3"))
    service = IndexingService(storage=storage, build_vectors=False)
    parsed = parse_document(source)
    parsed.assets = [
        AssetRef(
            asset_id="test_asset",
            document_id=parsed.document.document_id,
            kind="image",
            path="old.png",
        )
    ]
    if existing:
        service.save_parsed_document(parsed)

    before = corpus_snapshot(storage)
    with storage.connect() as conn:
        conn.execute(
            f"CREATE TRIGGER fail_payload BEFORE INSERT ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'injected payload failure'); END"
        )
    parsed.document.title = "replacement title"
    parsed.elements[0].text = "replacement evidence"
    with pytest.raises(sqlite3.IntegrityError, match="injected payload failure"):
        service.save_parsed_document(parsed)
    assert corpus_snapshot(storage) == before

    with storage.connect() as conn:
        conn.execute("DROP TRIGGER fail_payload")
    if existing:
        service.save_parsed_document(parsed)
        assert (
            storage.get_document(parsed.document.document_id).title
            == "replacement title"
        )
    else:
        job = service.ingest_path(source)
        assert job.status == "completed"
    assert storage.get_chunks(parsed.document.document_id)
    assert storage.search_chunks("evidence")


@pytest.mark.parametrize("broken_fts", [False, True])
def test_missing_fts_is_optional_but_broken_fts_rolls_back(tmp_path, broken_fts):
    source = tmp_path / "paper.md"
    source.write_text("searchable evidence", encoding="utf-8")
    storage = CorpusStorage(str(tmp_path / "corpus.sqlite3"))
    service = IndexingService(storage=storage, build_vectors=False)
    with storage.connect() as conn:
        conn.execute("DROP TABLE chunks_fts")
        if broken_fts:
            conn.execute("CREATE TABLE chunks_fts (document_id TEXT)")

    job = service.ingest_path(source)
    if broken_fts:
        assert job.status == "failed"
        assert storage.stats()["document_count"] == 0
        assert storage.stats()["element_count"] == 0
        assert storage.stats()["chunk_count"] == 0
    else:
        assert job.status == "completed"
        assert storage.search_chunks("searchable")


@pytest.mark.parametrize("existing", [False, True])
def test_partial_fts_failure_rolls_back_entire_corpus_and_retry_is_idempotent(
    tmp_path,
    monkeypatch,
    existing,
):
    source = tmp_path / "paper.md"
    source.write_text("original " * 80 + "\n\nsecond evidence", encoding="utf-8")
    storage = CorpusStorage(str(tmp_path / "corpus.sqlite3"))
    service = IndexingService(storage=storage, build_vectors=False)
    parsed = parse_document(source)
    parsed.assets = [
        AssetRef(
            asset_id="asset",
            document_id=parsed.document.document_id,
            kind="image",
            path="old.png",
        )
    ]
    if existing:
        service.save_parsed_document(parsed)
    before = corpus_snapshot(storage)
    parsed.document.title = "replacement title"
    parsed.elements[0].text = "replacement " * 80
    parsed.assets[0].path = "new.png"
    statements = []
    connections = []
    real_connect = sqlite3.connect
    with storage.connect() as conn:
        original_hits = conn.execute(
            "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'original' ORDER BY chunk_id"
        ).fetchall()

    class FailingConnection(sqlite3.Connection):
        def executemany(self, sql, parameters):
            if "INSERT INTO chunks_fts" in sql:
                rows = list(parameters)
                assert len(rows) == 2
                assert self.in_transaction
                super().execute(sql, rows[0])
                assert (
                    self.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 1
                )
                # Another connection must still see the old committed corpus.
                with closing(real_connect(storage.db_path)) as reader:
                    assert [
                        tuple(row)
                        for row in reader.execute(
                            "SELECT * FROM documents ORDER BY rowid"
                        )
                    ] == before["documents"]
                raise sqlite3.OperationalError("injected FTS write failure")
            return super().executemany(sql, parameters)

    def failing_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs, factory=FailingConnection)
        conn.set_trace_callback(statements.append)
        connections.append(conn)
        return conn

    with monkeypatch.context() as patcher:
        patcher.setattr(sqlite3, "connect", failing_connect)
        with pytest.raises(
            sqlite3.OperationalError, match="injected FTS write failure"
        ):
            service.save_parsed_document(parsed)
    assert len(connections) == 1
    assert sum(sql.startswith("BEGIN") for sql in statements) == 1
    assert "COMMIT" not in statements
    assert "ROLLBACK" in statements
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")
    assert corpus_snapshot(storage) == before
    with storage.connect() as conn:
        assert (
            conn.execute(
                "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'replacement'"
            ).fetchall()
            == []
        )
        assert conn.execute(
            "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'original' ORDER BY chunk_id"
        ).fetchall() == original_hits

    service.save_parsed_document(parsed)
    successful = corpus_snapshot(storage)
    service.save_parsed_document(parsed)
    assert corpus_snapshot(storage) == successful
    job = service.ingest_path(source)
    assert job.status == "completed"
    assert corpus_snapshot(storage) == successful
    assert [len(successful[name]) for name in successful] == [1, 2, 1, 2, 2]
    with storage.connect() as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert (
            conn.execute(
                "SELECT chunk_id FROM chunks_fts EXCEPT SELECT chunk_id FROM chunks"
            ).fetchall()
            == []
        )
        assert (
            conn.execute(
                "SELECT chunk_id FROM chunks EXCEPT SELECT chunk_id FROM chunks_fts"
            ).fetchall()
            == []
        )
    assert storage.search_chunks("replacement")


@pytest.mark.parametrize(
    "error",
    [
        "no such module: fts5",
        "database is locked",
        "disk I/O error",
        "attempt to write a readonly database",
    ],
)
def test_fts_initialization_only_falls_back_for_missing_module(
    tmp_path, monkeypatch, error
):
    real_connect = sqlite3.connect

    class MissingFTSConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if "CREATE VIRTUAL TABLE" in sql:
                raise sqlite3.OperationalError(error)
            return super().execute(sql, *args, **kwargs)

    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: real_connect(
            *args,
            **kwargs,
            factory=MissingFTSConnection,
        ),
    )
    db_path = str(tmp_path / "corpus.sqlite3")
    if error != "no such module: fts5":
        with pytest.raises(sqlite3.OperationalError, match=error):
            CorpusStorage(db_path)
        return
    storage = CorpusStorage(db_path)
    source = tmp_path / "paper.md"
    source.write_text("searchable evidence", encoding="utf-8")
    service = IndexingService(storage=storage, build_vectors=False)
    assert service.ingest_path(source).status == "completed"
    assert storage.search_chunks("searchable")


@pytest.mark.parametrize(
    "error", ["database is locked", "disk I/O error", "no such column: rank"]
)
def test_fts_search_does_not_hide_database_errors(tmp_path, monkeypatch, error):
    storage = CorpusStorage(str(tmp_path / "corpus.sqlite3"))
    real_connect = sqlite3.connect

    class FailingSearchConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if "WITH matches AS" in sql:
                raise sqlite3.OperationalError(error)
            return super().execute(sql, *args, **kwargs)

    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: real_connect(
            *args,
            **kwargs,
            factory=FailingSearchConnection,
        ),
    )
    with pytest.raises(sqlite3.OperationalError, match=error):
        storage.search_chunks("evidence")
