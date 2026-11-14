from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.models import DegradationReason, EvidenceChunk, QueryResult, ScreeningReport, model_to_dict
from src.config import Config
from src.retrieval import ResearchRAG, RetrievalExecution, RetrievedEvidence
from src.storage import CorpusStorage


@pytest.mark.parametrize("factory", [
    lambda **kwargs: QueryResult(question="问题", answer="回答", **kwargs),
    lambda **kwargs: ScreeningReport(report_id="report-1", topic="主题", **kwargs),
])
def test_execution_metadata_defaults_serialize_for_legacy_constructors(factory):
    payload = model_to_dict(factory())
    assert payload["retrieval_mode_requested"] == "ensemble"
    assert payload["retrieval_mode_used"] == "unknown"
    assert payload["degraded"] is False
    assert payload["degradation_reason"] == []


def test_degradation_reason_lists_are_independent():
    first = QueryResult(question="问题", answer="回答")
    second = QueryResult(question="问题", answer="回答")
    report = ScreeningReport(report_id="report-1", topic="主题")
    first.degradation_reason.append(DegradationReason(
        stage="vector", code="vector_index_missing", message="向量索引不存在。", fallback="keyword",
    ))
    assert second.degradation_reason == []
    assert report.degradation_reason == []


def test_multiple_degradation_reasons_serialize_in_order():
    reasons = [
        {"stage": "vector", "code": "vector_index_missing", "message": "向量索引不存在。", "fallback": "keyword"},
        {"stage": "llm", "code": "llm_failed", "message": "回答生成失败。", "fallback": "extractive"},
    ]
    result = QueryResult(question="问题", answer="回答", retrieval_mode_requested="rerank",
                         retrieval_mode_used="keyword", degraded=True, degradation_reason=reasons)
    payload = model_to_dict(result)
    assert payload["retrieval_mode_requested"] == "rerank"
    assert payload["retrieval_mode_used"] == "keyword"
    assert payload["degraded"] is True
    assert payload["degradation_reason"] == reasons


def test_degradation_reason_rejects_unknown_stage():
    with pytest.raises(ValidationError):
        QueryResult(question="问题", answer="回答", degradation_reason=[
            {"stage": "unknown", "code": "failure", "message": "失败。", "fallback": "none"},
        ])


def test_retrieval_execution_tracks_zero_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "VECTOR_DIR", str(tmp_path))
    (tmp_path / "index.faiss").touch()
    rag = ResearchRAG(CorpusStorage(str(tmp_path / "corpus.sqlite3")))
    rag._vectorstore = SimpleNamespace(similarity_search=lambda *args, **kwargs: [])
    execution = RetrievalExecution(" Vector ")
    assert rag.retrieve("example", mode=" Vector ", execution=execution) == []
    assert execution.metadata().retrieval_mode_used == "vector"
    assert not execution.metadata().degraded


@pytest.fixture
def retrieval_rig(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "VECTOR_DIR", str(tmp_path))
    monkeypatch.setattr(Config, "ENABLE_RERANK", False)
    monkeypatch.setattr(Config, "TRUST_LOCAL_FAISS_INDEX", True)
    return ResearchRAG(CorpusStorage(str(tmp_path / "corpus.sqlite3"))), tmp_path


def fail(*args, **kwargs):
    raise RuntimeError("private connection information")


@pytest.mark.parametrize("mode,fallback,used", [("vector", "none", "none"), ("ensemble", "keyword", "keyword")])
@pytest.mark.parametrize("failure,code", [
    ("missing", "vector_index_missing"), ("disabled", "vector_loading_disabled"),
    ("load", "vector_load_failed"), ("search", "vector_search_failed"),
])
def test_vector_failure_paths(retrieval_rig, monkeypatch, mode, fallback, used, failure, code):
    rag, path = retrieval_rig
    if failure != "missing":
        (path / "index.faiss").touch()
    if failure == "disabled":
        monkeypatch.setattr(Config, "TRUST_LOCAL_FAISS_INDEX", False)
    if failure == "load":
        monkeypatch.setattr(rag, "_load_vectorstore", fail)
    if failure == "search":
        rag._vectorstore = SimpleNamespace(similarity_search=fail)
    if mode == "vector":
        monkeypatch.setattr(rag.storage, "search_chunks", fail)
    execution = RetrievalExecution(mode)
    assert rag.retrieve("example", mode=mode, execution=execution) == []
    metadata = execution.metadata()
    assert metadata.retrieval_mode_used == used
    assert [(r.code, r.fallback) for r in metadata.degradation_reason] == [(code, fallback)]
    assert "private" not in metadata.degradation_reason[0].message


def test_lazy_load_and_cached_vector_ignore_disabled_loading(retrieval_rig, monkeypatch):
    rag, path = retrieval_rig
    (path / "index.faiss").touch()
    fake = SimpleNamespace(similarity_search=lambda *args, **kwargs: [])
    monkeypatch.setattr(rag, "_load_vectorstore", lambda: fake)
    execution = RetrievalExecution("vector", "vector")
    assert rag._vector_rank_scores("query", 2, execution=execution) == {}
    assert execution.metadata().retrieval_mode_used == "vector"
    assert not execution.metadata().degraded
    rag._vectorstore = fake
    monkeypatch.setattr(Config, "TRUST_LOCAL_FAISS_INDEX", False)
    assert rag._vector_rank_scores("query", 2, execution=execution) == {}
    assert not execution.metadata().degraded


def test_multiquery_partial_success_and_dedup(retrieval_rig):
    rag, path = retrieval_rig
    (path / "index.faiss").touch()
    responses = iter([RuntimeError(), [], RuntimeError()])
    def search(*args, **kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response
    rag._vectorstore = SimpleNamespace(similarity_search=search)
    execution = RetrievalExecution("multiquery")
    assert rag.retrieve("alpha beta gamma delta?", mode="multiquery", execution=execution) == []
    assert execution.query_variant_count == 3
    assert execution.metadata().retrieval_mode_used == "ensemble+multiquery"
    assert [r.code for r in execution.degradation_reason] == ["vector_search_failed"]


def test_unknown_mode_normalization_and_request_isolation(retrieval_rig):
    rag, _ = retrieval_rig
    execution = RetrievalExecution("unknown")
    rag.retrieve("example", mode="unknown", execution=execution)
    assert [r.code for r in execution.degradation_reason] == ["unsupported_mode", "vector_index_missing"]
    assert execution.metadata().retrieval_mode_requested == "unknown"
    assert RetrievalExecution().degradation_reason == []


@pytest.fixture
def candidates():
    return [RetrievedEvidence(EvidenceChunk(chunk_id=str(i), document_id="doc", element_ids=["element"],
                                          text="example", search_text="example"), score=1-i*.1)
            for i in range(2)]


@pytest.mark.parametrize("scores", [[1], [float("nan"), 1], [1, float("inf")], ["invalid", 1]])
def test_rerank_invalid_scores_preserve_original_order(retrieval_rig, monkeypatch, candidates, scores):
    rag, _ = retrieval_rig
    monkeypatch.setattr(Config, "RERANK_MODEL", "fake")
    monkeypatch.setattr(rag, "_rerank_scores", lambda *args: scores)
    execution = RetrievalExecution()
    assert rag._rerank("query", candidates, force=True, execution=execution) is candidates
    assert [r.code for r in execution.degradation_reason] == ["reranker_invalid_scores"]
    assert not execution.rerank_used


def test_rerank_invalid_prediction_preserves_original_candidates(retrieval_rig, monkeypatch, candidates):
    rag, _ = retrieval_rig
    monkeypatch.setattr(Config, "RERANK_MODEL", "fake")
    rag._reranker = SimpleNamespace(predict=lambda pairs: ["invalid", 1])
    original_scores = [candidate.score for candidate in candidates]
    execution = RetrievalExecution()
    assert rag._rerank("query", candidates, force=True, execution=execution) is candidates
    assert [candidate.chunk.chunk_id for candidate in candidates] == ["0", "1"]
    assert [candidate.score for candidate in candidates] == original_scores
    assert [reason.code for reason in execution.degradation_reason] == ["reranker_invalid_scores"]
    assert not execution.rerank_used


@pytest.mark.parametrize("failure,code", [("missing", "reranker_not_configured"),
                                           ("predict", "reranker_failed"), ("cached", "reranker_failed")])
def test_rerank_failure_preserves_original_order(retrieval_rig, monkeypatch, candidates, failure, code):
    rag, _ = retrieval_rig
    monkeypatch.setattr(Config, "RERANK_MODEL", "" if failure == "missing" else "fake")
    if failure == "predict":
        rag._reranker = SimpleNamespace(predict=fail)
    if failure == "cached":
        rag._reranker_unavailable = True
    execution = RetrievalExecution()
    assert rag._rerank("query", candidates, force=True, execution=execution) is candidates
    assert [r.code for r in execution.degradation_reason] == [code]


def test_rerank_success_and_natural_skips(retrieval_rig, monkeypatch, candidates):
    rag, _ = retrieval_rig
    monkeypatch.setattr(Config, "RERANK_MODEL", "fake")
    rag._reranker = SimpleNamespace(predict=lambda pairs: [0, 2])
    execution = RetrievalExecution()
    execution.keyword_used = True
    assert rag._rerank("query", candidates, execution=execution) is candidates
    assert rag._rerank("query", candidates[:1], force=True, execution=execution) == candidates[:1]
    assert not execution.degradation_reason
    ranked = rag._rerank("query", candidates, force=True, execution=execution)
    assert [r.chunk.chunk_id for r in ranked] == ["1", "0"]
    assert execution.metadata().retrieval_mode_used == "keyword+rerank"
    assert ranked[0].score == pytest.approx(2.0009)


def test_vector_success_retained_when_all_hits_filtered(retrieval_rig, monkeypatch, candidates):
    rag, path = retrieval_rig
    (path / "index.faiss").touch()
    rag._vectorstore = SimpleNamespace(similarity_search=lambda *args, **kwargs: [
        SimpleNamespace(metadata={"chunk_id": "0"})])
    monkeypatch.setattr(rag.storage, "get_chunks_by_ids", lambda ids: [candidates[0].chunk])
    execution = RetrievalExecution("vector")
    assert rag.retrieve("example", mode="vector", include_keywords=["absent"], execution=execution) == []
    assert execution.metadata().retrieval_mode_used == "vector"
    assert not execution.metadata().degraded


def test_reranker_loading_exception_is_recorded_and_cached(retrieval_rig, monkeypatch, candidates):
    import sys
    rag, _ = retrieval_rig
    monkeypatch.setattr(Config, "RERANK_MODEL", "fake")
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=fail))
    execution = RetrievalExecution()
    assert rag._rerank("query", candidates, force=True, execution=execution) is candidates
    assert rag._reranker_unavailable
    assert rag._rerank("query", candidates, force=True, execution=execution) is candidates
    assert [r.code for r in execution.degradation_reason] == ["reranker_failed"]


@pytest.fixture
def answer_rig(retrieval_rig, monkeypatch):
    import sys
    from src.models import PaperDocument
    rag, path = retrieval_rig
    rag.storage.save_document(PaperDocument(document_id="doc", source_path="paper.md",
        filename="paper.md", file_type="md", title="Study", sha256="hash", parser="plain"))
    rag.storage.save_chunks("doc", [EvidenceChunk(chunk_id=str(i), document_id="doc", element_ids=["element"],
        text=f"Example method findings {i}.", search_text=f"example method findings {i}") for i in range(2)])
    (path / "index.faiss").touch()
    rag._vectorstore = SimpleNamespace(similarity_search=lambda *args, **kwargs: [])
    monkeypatch.setattr(Config, "get_api_key", staticmethod(lambda: "fake-key"))
    monkeypatch.setattr(Config, "RERANK_MODEL", "fake")
    state = SimpleNamespace(content="Evidence answer [C1].", error=False)
    def invoke(prompt):
        if state.error:
            fail()
        return SimpleNamespace(content=state.content)
    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(
        ChatOpenAI=lambda **kwargs: SimpleNamespace(invoke=invoke)))
    return rag, state


@pytest.mark.parametrize("failure,code", [("key", "llm_not_configured"), ("error", "llm_failed"),
    ("blank", "llm_empty_response"), ("list", "llm_empty_response"), ("none", "llm_empty_response")])
def test_answer_llm_fallback_metadata(answer_rig, monkeypatch, failure, code):
    rag, state = answer_rig
    if failure == "key":
        monkeypatch.setattr(Config, "get_api_key", staticmethod(lambda: ""))
    state.error = failure == "error"
    state.content = {"blank": "  \n", "list": [], "none": None}.get(failure, state.content)
    result = rag.answer("example")
    assert result.answer == rag._extractive_answer("example", result.citations)
    assert len(result.citations) == len(result.contexts) == 2
    assert result.retrieval_mode_used == "ensemble"
    assert [(r.code, r.fallback) for r in result.degradation_reason] == [(code, "extractive")]
    assert "private" not in str(model_to_dict(result))
    assert rag._llm_answer("example", result.contexts, result.citations) == ""


def test_answer_success_and_consecutive_request_isolation(answer_rig):
    rag, state = answer_rig
    state.error = True
    first = rag.answer("example")
    state.error = False
    second = rag.answer("example")
    assert [r.code for r in first.degradation_reason] == ["llm_failed"]
    assert second.answer == "Evidence answer [C1]."
    assert second.retrieval_mode_used == "ensemble"
    assert not second.degraded and second.degradation_reason == []


def test_answer_no_evidence_has_only_retrieval_metadata(answer_rig):
    rag, state = answer_rig
    state.error = True
    result = rag.answer("unmatched", mode="vector")
    assert result.answer == "当前语料库中没有可支持回答的证据。请先导入论文，或换一个更具体的研究问题。"
    assert result.retrieval_mode_used == "vector"
    assert result.citations == result.contexts == result.degradation_reason == []
    assert not result.degraded


def test_blank_multiquery_answer_skips_backends_and_llm(answer_rig, monkeypatch):
    rag, _ = answer_rig
    calls = []
    def unexpected_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Blank multiquery must not call a backend or LLM")
    monkeypatch.setattr(rag.storage, "search_chunks", unexpected_call)
    monkeypatch.setattr(rag, "_vector_rank_scores", unexpected_call)
    monkeypatch.setattr(rag, "_llm_answer", unexpected_call)
    result = rag.answer("   ", mode="multiquery")
    assert calls == []
    assert result.retrieval_mode_requested == "multiquery"
    assert result.retrieval_mode_used == "none"
    assert result.answer == "当前语料库中没有可支持回答的证据。请先导入论文，或换一个更具体的研究问题。"
    assert result.citations == result.contexts == result.degradation_reason == []
    assert not result.degraded


def test_answer_accumulates_simultaneous_failures(answer_rig):
    rag, state = answer_rig
    rag._vectorstore = SimpleNamespace(similarity_search=fail)
    rag._reranker = SimpleNamespace(predict=fail)
    state.error = True
    result = rag.answer("example", mode="rerank")
    assert result.retrieval_mode_requested == "rerank"
    assert result.retrieval_mode_used == "keyword"
    assert [r.code for r in result.degradation_reason] == [
        "vector_search_failed", "reranker_failed", "llm_failed"]
    assert result.answer == rag._extractive_answer("example", result.citations)


def test_screen_persists_metadata_without_llm(answer_rig, monkeypatch):
    import json
    rag, _ = answer_rig
    monkeypatch.setattr(Config, "get_api_key", staticmethod(fail))
    rag._vectorstore = SimpleNamespace(similarity_search=fail)
    report = rag.screen("example")
    assert report.items and report.retrieval_mode_requested == "ensemble"
    assert report.retrieval_mode_used == "keyword"
    assert [r.code for r in report.degradation_reason] == ["vector_search_failed"]
    with rag.storage.connect() as conn:
        payload = json.loads(conn.execute("SELECT payload FROM reports WHERE report_id = ?",
                                         (report.report_id,)).fetchone()[0])
    assert payload == model_to_dict(report)
    empty = rag.screen("unmatched")
    assert empty.items == [] and empty.retrieval_mode_used == "keyword"


def test_interleaved_answer_requests_keep_separate_execution(answer_rig):
    rag, state = answer_rig
    nested = []
    def search(query, **kwargs):
        if query == "example":
            nested.append(rag.answer("unmatched", mode="vector"))
            fail()
        return []
    rag._vectorstore = SimpleNamespace(similarity_search=search)
    state.error = True
    outer = rag.answer("example")
    assert [r.code for r in outer.degradation_reason] == ["vector_search_failed", "llm_failed"]
    assert nested[0].retrieval_mode_used == "vector"
    assert not nested[0].degraded and nested[0].degradation_reason == []



def test_answer_missing_vector_index_keeps_early_return_reason(answer_rig, monkeypatch):
    from pathlib import Path
    rag, _ = answer_rig
    (Path(Config.VECTOR_DIR) / "index.faiss").unlink()
    monkeypatch.setattr(Config, "get_api_key", staticmethod(fail))
    result = rag.answer("example", mode="vector")
    assert result.retrieval_mode_requested == "vector"
    assert result.retrieval_mode_used == "none"
    assert [r.code for r in result.degradation_reason] == ["vector_index_missing"]
    assert result.degradation_reason[0].fallback == "none"
    assert result.citations == result.contexts == []
