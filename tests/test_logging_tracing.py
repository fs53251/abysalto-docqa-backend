from __future__ import annotations

import logging

from fastapi.testclient import TestClient


def _records_for_event(caplog, event_name: str, *, path: str | None = None):
    records = []
    for record in caplog.records:
        if record.name != "app.access":
            continue
        if getattr(record, "event", None) != event_name:
            continue
        if path is not None and getattr(record, "path", None) != path:
            continue
        records.append(record)
    return records


def test_access_logging_smoke_and_request_id_header(
    client: TestClient,
    caplog,
) -> None:
    caplog.set_level(logging.INFO)

    response = client.get("/health")
    assert response.status_code == 200, response.text
    assert "X-Request-Id" in response.headers
    assert response.headers["X-Request-Id"]

    start_records = _records_for_event(caplog, "request.start", path="/health")
    end_records = _records_for_event(caplog, "request.end", path="/health")

    assert start_records
    assert end_records

    end_record = end_records[-1]
    assert getattr(end_record, "method", None) == "GET"
    assert getattr(end_record, "status_code", None) == 200
    assert isinstance(getattr(end_record, "latency_ms", None), float)


def test_identity_context_propagates_into_access_logs(
    client: TestClient,
    register_and_login,
    caplog,
) -> None:
    user = register_and_login(email="trace-user@example.com")

    caplog.clear()
    caplog.set_level(logging.INFO)

    response = client.get("/documents", headers=user.headers)
    assert response.status_code == 200, response.text
    assert "X-Request-Id" in response.headers

    end_records = _records_for_event(caplog, "request.end", path="/documents")
    assert end_records
    assert any(
        str(getattr(record, "identity", "")).startswith("user:")
        for record in end_records
    )


def test_access_logging_middleware_does_not_break_handled_errors(
    client: TestClient,
    caplog,
) -> None:
    caplog.set_level(logging.INFO)

    response = client.post("/ask", json={})
    assert response.status_code == 400, response.text
    assert "X-Request-Id" in response.headers

    end_records = _records_for_event(caplog, "request.end", path="/ask")
    assert end_records
    assert any(getattr(record, "status_code", None) == 400 for record in end_records)
