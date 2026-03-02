import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    # Ensure details are returned in errors (details are omitted only in prod)
    monkeypatch.setattr(settings, "APP_ENV", "test", raising=False)
    yield


def test_request_id_header_present_on_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert "X-Request-Id" in r.headers
        assert isinstance(r.headers["X-Request-Id"], str)
        assert len(r.headers["X-Request-Id"]) > 0


def test_validation_error_is_standardized_and_includes_request_id():
    """
    Trigger validation error with invalid body.
    /ask expects a JSON body with at least a question field (existing API).
    Our exception handler converts validation errors to 400 with standard payload.
    """
    with TestClient(app) as client:
        r = client.post("/ask", json={})
        assert r.status_code == 400

        body = r.json()
        assert body["error_code"] == "invalid_input"
        assert "message" in body
        assert body["message"] == "Invalid request"
        assert "request_id" in body
        assert isinstance(body["request_id"], str)
        assert len(body["request_id"]) > 0

        # In test/dev we expect details to be included
        assert "details" in body
        assert isinstance(body["details"], list)
        assert len(body["details"]) >= 1
