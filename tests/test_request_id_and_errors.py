from __future__ import annotations

from fastapi.testclient import TestClient


def test_request_id_header_present_on_health(client: TestClient) -> None:
    res = client.get("/health")

    assert res.status_code == 200
    assert "X-Request-Id" in res.headers
    assert isinstance(res.headers["X-Request-Id"], str)
    assert len(res.headers["X-Request-Id"]) > 0


def test_validation_error_is_standardized_and_includes_request_id(
    client: TestClient,
) -> None:
    # /ask expects a body with at least the "question" field.
    res = client.post("/ask", json={})

    assert res.status_code == 400
    body = res.json()

    assert body["error_code"] == "invalid_input"
    assert body["message"] == "Invalid request"
    assert isinstance(body["request_id"], str) and body["request_id"]
    # In test/dev we expect details to be included
    assert isinstance(body["details"], list) and body["details"]
