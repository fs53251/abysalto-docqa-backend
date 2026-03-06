from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient

import app.api.routes.ask as ask_route
from app.api.deps import (
    get_embedding_service,
    get_optional_cache,
    get_optional_ner_service,
    get_qa_service,
)
from app.core.config import settings
from app.core.security.session import load_session_cookie
from app.db.session import get_sessionmaker
from app.main import app
from app.repositories.documents import get_document_by_public_id
from app.services.retrieval.retriever import RetrievedChunk


def _upload_one_pdf(client: TestClient):
    return client.post(
        "/upload",
        files=[
            (
                "files",
                (
                    "session-doc.pdf",
                    BytesIO(b"%PDF-1.4 session test"),
                    "application/pdf",
                ),
            )
        ],
    )


def test_upload_without_cookie_sets_signed_session_cookie_and_persists_owner(
    client: TestClient,
    temp_data_dir,
):
    response = _upload_one_pdf(client)
    assert response.status_code == 200, response.text

    set_cookie = response.headers.get("set-cookie", "")
    assert f"{settings.SESSION_COOKIE_NAME}=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite=lax" in set_cookie
    assert "Max-Age=604800" in set_cookie
    assert "Secure" not in set_cookie

    signed_cookie = client.cookies.get(settings.SESSION_COOKIE_NAME)
    session_id = load_session_cookie(signed_cookie)
    assert session_id is not None

    doc_id = response.json()["documents"][0]["doc_id"]
    SessionLocal = get_sessionmaker()
    db = SessionLocal()
    try:
        doc = get_document_by_public_id(db, doc_id=doc_id)
        assert doc is not None
        assert doc.owner_session_id == session_id
    finally:
        db.close()


def test_same_session_reuses_cookie_for_documents_listing(
    client: TestClient,
    temp_data_dir,
):
    upload_response = _upload_one_pdf(client)
    assert upload_response.status_code == 200, upload_response.text
    doc_id = upload_response.json()["documents"][0]["doc_id"]

    list_response = client.get("/documents")
    assert list_response.status_code == 200, list_response.text
    payload = list_response.json()

    assert payload["count"] == 1
    assert payload["documents"][0]["doc_id"] == doc_id


def test_ask_uses_session_scoped_documents_and_other_session_cannot_access_them(
    client: TestClient,
    services,
    temp_data_dir,
    monkeypatch,
):
    upload_response = _upload_one_pdf(client)
    assert upload_response.status_code == 200, upload_response.text
    doc_id = upload_response.json()["documents"][0]["doc_id"]

    processed_dir = temp_data_dir / "processed" / doc_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "faiss.index").write_bytes(b"index")

    def fake_search(self, doc_id: str, query: str, top_k: int, query_emb=None):
        return [
            RetrievedChunk(
                doc_id=doc_id,
                chunk_id="chunk-1",
                score=0.99,
                page=1,
                chunk_index=0,
                text_snippet="This document belongs to the active session.",
            )
        ]

    monkeypatch.setattr(ask_route.RetrieverService, "search", fake_search)

    same_session = client.post("/ask", json={"question": "Whose document is this?"})
    assert same_session.status_code == 200, same_session.text
    assert same_session.json()["sources"][0]["doc_id"] == doc_id

    with TestClient(app) as other_client:
        other_client.app.dependency_overrides[get_embedding_service] = (
            lambda: services.embedding
        )
        other_client.app.dependency_overrides[get_qa_service] = lambda: services.qa
        other_client.app.dependency_overrides[get_optional_ner_service] = (
            lambda: services.ner
        )
        other_client.app.dependency_overrides[get_optional_cache] = (
            lambda: services.cache
        )

        other_docs = other_client.get("/documents")
        assert other_docs.status_code == 200, other_docs.text
        assert other_docs.json()["count"] == 0

        other_ask = other_client.post(
            "/ask", json={"question": "Whose document is this?"}
        )
        assert other_ask.status_code == 404, other_ask.text


def test_invalid_session_cookie_gets_rotated_to_new_session(
    services,
    temp_data_dir,
):
    app.dependency_overrides[get_embedding_service] = lambda: services.embedding
    app.dependency_overrides[get_qa_service] = lambda: services.qa
    app.dependency_overrides[get_optional_ner_service] = lambda: services.ner
    app.dependency_overrides[get_optional_cache] = lambda: services.cache

    try:
        with TestClient(app) as other_client:
            other_client.cookies.set(
                settings.SESSION_COOKIE_NAME,
                "tampered-cookie",
                domain="testserver.local",
                path="/",
            )

            response = other_client.get("/documents")
            assert response.status_code == 200, response.text
            assert response.json()["count"] == 0

            assert "set-cookie" in response.headers

            new_cookie = other_client.cookies.get(
                settings.SESSION_COOKIE_NAME,
                domain="testserver.local",
                path="/",
            )
            assert new_cookie is not None
            assert new_cookie != "tampered-cookie"
            assert load_session_cookie(new_cookie) is not None
    finally:
        app.dependency_overrides.clear()
