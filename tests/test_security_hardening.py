from __future__ import annotations

from fastapi.testclient import TestClient
from starlette.requests import Request

from app.core.config import settings
from app.core.network import get_client_ip
from app.main import app


def test_security_headers_present_on_response(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200, response.text

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert (
        response.headers["Permissions-Policy"]
        == "geolocation=(), microphone=(), camera=()"
    )
    assert (
        response.headers["Content-Security-Policy"]
        == "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    )
    assert "Strict-Transport-Security" not in response.headers


def test_hsts_only_in_prod_over_https(monkeypatch) -> None:
    monkeypatch.setattr(settings, "APP_ENV", "prod", raising=False)

    with TestClient(app, base_url="https://testserver") as https_client:
        response = https_client.get("/health")
        assert response.status_code == 200, response.text
        assert (
            response.headers["Strict-Transport-Security"]
            == "max-age=31536000; includeSubDomains"
        )


def test_cors_allows_configured_local_origin(client: TestClient) -> None:
    response = client.get(
        "/health",
        headers={"Origin": "http://localhost:8501"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_client_ip_uses_forwarded_header_only_for_trusted_proxy(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "TRUSTED_PROXIES",
        ("127.0.0.1", "::1"),
        raising=False,
    )

    trusted_scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"x-forwarded-for", b"203.0.113.10, 127.0.0.1")],
        "client": ("127.0.0.1", 50000),
        "scheme": "http",
        "server": ("testserver", 80),
    }
    trusted_request = Request(trusted_scope)
    assert get_client_ip(trusted_request) == "203.0.113.10"

    untrusted_scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"x-forwarded-for", b"203.0.113.20, 10.0.0.1")],
        "client": ("198.51.100.5", 50000),
        "scheme": "http",
        "server": ("testserver", 80),
    }
    untrusted_request = Request(untrusted_scope)
    assert get_client_ip(untrusted_request) == "198.51.100.5"
