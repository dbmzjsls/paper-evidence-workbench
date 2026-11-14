import json

import pytest
from fastapi.testclient import TestClient

import api
from src.models import ExecutionMetadata, QueryResult, ScreeningReport


@pytest.fixture(params=[False, True], ids=["normal", "degraded"])
def backend(monkeypatch, request):
    metadata = {
        "retrieval_mode_requested": ' "rerank"\n',
        "retrieval_mode_used": "keyword",
        "degraded": True,
        "degradation_reason": [
            {
                "stage": "vector",
                "code": "vector_index_missing",
                "message": "向量索引不存在。",
                "fallback": "keyword",
            }
        ],
    }
    if not request.param:
        metadata.update(
            retrieval_mode_used="ensemble+rerank", degraded=False, degradation_reason=[]
        )
    result = QueryResult(
        question="question", answer='\nfirst "quote"\r\nsecond\r\n\n', **metadata
    )
    report = ScreeningReport(
        report_id="report", topic="topic", assumptions=["assumption"], **metadata
    )

    class FakeRAG:
        def answer(self, question, mode="ensemble"):
            return result

        def screen(self, **kwargs):
            return report

    monkeypatch.setattr(api, "ResearchRAG", FakeRAG)
    return TestClient(api.app), result, report, metadata


def parse_events(text):
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        if not block:
            continue
        event = "message"
        data = []
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(" ") else value)
        events.append((event, "\n".join(data)))
    return events


def test_response_fields_and_openapi(backend):
    client, result, report, metadata = backend
    for path, body, expected in [
        ("/query", {"question": "question"}, result),
        ("/screen", {"topic": "topic"}, report),
    ]:
        response = client.post(path, json=body)
        assert response.status_code == 200
        assert response.json() == expected.model_dump(mode="json")
        assert {key: response.json()[key] for key in metadata} == metadata
        schema = client.get("/openapi.json").json()["paths"][path]["post"]["responses"][
            "200"
        ]["content"]["application/json"]["schema"]
        assert schema["$ref"].endswith(type(expected).__name__)


def test_stream_metadata_and_multiline_answer(backend):
    client, result, _, metadata = backend
    response = client.post("/query/stream", json={"question": "question"})
    assert response.status_code == 200
    events = parse_events(response.text)
    assert [event for event, _ in events] == ["metadata", "message", "message"]
    assert json.loads(events[0][1]) == metadata
    assert set(json.loads(events[0][1])) == set(ExecutionMetadata.model_fields)
    assert events[1][1] == result.answer.replace("\r\n", "\n").replace("\r", "\n")
    assert events[2][1] == "[DONE]"


def test_existing_validation_and_stream_error(backend, monkeypatch):
    client, _, _, _ = backend
    assert client.post("/query", json={"question": " "}).status_code == 400
    assert client.post("/screen", json={"topic": " "}).status_code == 400

    class FailingRAG:
        def answer(self, *args, **kwargs):
            raise RuntimeError("existing failure")

    monkeypatch.setattr(api, "ResearchRAG", FailingRAG)
    assert parse_events(client.post("/query/stream", json={"question": "q"}).text) == [
        ("message", "[ERROR] existing failure")
    ]
