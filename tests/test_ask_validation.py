from __future__ import annotations

import uuid

import numpy as np
from fastapi.testclient import TestClient

from app.core.identifiers import document_public_id, generate_document_id
from app.db.session import get_sessionmaker
from app.repositories.documents import create_document


class DummyQAService:
    def answer(self, question: str, context: str):
        class Result:
            answer = "SCOPE ANSWER"
            score = 0.95

        return Result()


def _touch_index(temp_data_dir, doc_id: str) -> None:
    processed = temp_data_dir / "processed" / doc_id
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "faiss.index").write_bytes(b"index")


def _create_foreign_session_document(
    *, temp_data_dir, filename: str = "foreign-session.pdf"
) -> str:
    document_uuid = generate_document_id()
    public_doc_id = document_public_id(document_uuid)
    stored_path = temp_data_dir / "uploads" / public_doc_id / "original" / filename
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    stored_path.touch(exist_ok=True)

    session_local = get_sessionmaker()
    db = session_local()
    try:
        create_document(
            db,
            doc_id=document_uuid,
            filename=filename,
            content_type="application/pdf",
            stored_path=str(stored_path),
            owner_session_id=str(uuid.uuid4()),
            status="indexed",
        )
    finally:
        db.close()
    return public_doc_id


def test_ask_rejects_invalid_top_k(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "hello", "top_k": 999})
    assert response.status_code == 400, response.text
    assert response.json()["error_code"] == "invalid_input"


def test_ask_rejects_question_too_long(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "x" * 2001, "top_k": 1})
    assert response.status_code == 400, response.text
    assert response.json()["error_code"] == "invalid_input"


def test_ask_rejects_invalid_doc_id(client: TestClient) -> None:
    response = client.post(
        "/ask",
        json={
            "question": "What is this?",
            "scope": "docs",
            "doc_ids": ["not-a-valid-doc-id"],
            "top_k": 1,
        },
    )
    assert response.status_code == 400, response.text
    assert response.json()["error_code"] == "invalid_input"


def test_anon_session_ask_uses_only_session_documents(
    client: TestClient,
    services,
    temp_data_dir,
    create_owned_document,
    monkeypatch,
) -> None:
    services.qa = DummyQAService()
    monkeypatch.setattr(
        services.embedding,
        "encode_texts",
        lambda texts: np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )

    import app.services.retrieval.retriever as retr_mod
    from app.services.retrieval.retriever import RetrievedChunk

    owned = create_owned_document(
        client, filename="owned-session.pdf", status="indexed"
    )
    foreign_doc_id = _create_foreign_session_document(
        temp_data_dir=temp_data_dir, filename="foreign-session.pdf"
    )
    _touch_index(temp_data_dir, owned.doc_id)
    _touch_index(temp_data_dir, foreign_doc_id)

    searched_doc_ids: list[str] = []

    def fake_search(self, doc_id, query, top_k, query_emb=None):
        searched_doc_ids.append(doc_id)
        return [
            RetrievedChunk(
                doc_id=doc_id,
                chunk_id="chunk-1",
                score=0.99,
                page=1,
                chunk_index=0,
                text_snippet="Owned session snippet",
                text="Owned session snippet",
            )
        ]

    monkeypatch.setattr(retr_mod.RetrieverService, "search", fake_search)

    response = client.post(
        "/ask", json={"question": "Which docs are visible?", "top_k": 1}
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert searched_doc_ids == [owned.doc_id]
    assert payload["sources"][0]["doc_id"] == owned.doc_id
    assert payload["sources"][0]["filename"] == "owned-session.pdf"


def test_auth_user_ask_uses_only_user_documents(
    client: TestClient,
    services,
    temp_data_dir,
    register_and_login,
    create_user_owned_document,
    monkeypatch,
) -> None:
    services.qa = DummyQAService()
    monkeypatch.setattr(
        services.embedding,
        "encode_texts",
        lambda texts: np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )

    import app.services.retrieval.retriever as retr_mod
    from app.services.retrieval.retriever import RetrievedChunk

    owner = register_and_login(email="scope-owner@example.com")
    other = register_and_login(email="scope-other@example.com")

    owned = create_user_owned_document(
        user_id=owner.user_id, filename="owned-user.pdf", status="indexed"
    )
    foreign = create_user_owned_document(
        user_id=other.user_id, filename="foreign-user.pdf", status="indexed"
    )
    _touch_index(temp_data_dir, owned.doc_id)
    _touch_index(temp_data_dir, foreign.doc_id)

    searched_doc_ids: list[str] = []

    def fake_search(self, doc_id, query, top_k, query_emb=None):
        searched_doc_ids.append(doc_id)
        return [
            RetrievedChunk(
                doc_id=doc_id,
                chunk_id="chunk-1",
                score=0.99,
                page=1,
                chunk_index=0,
                text_snippet="Owned user snippet",
                text="Owned user snippet",
            )
        ]

    monkeypatch.setattr(retr_mod.RetrieverService, "search", fake_search)

    response = client.post(
        "/ask",
        json={"question": "Which docs are visible?", "top_k": 1},
        headers=owner.headers,
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert searched_doc_ids == [owned.doc_id]
    assert payload["sources"][0]["doc_id"] == owned.doc_id
    assert payload["sources"][0]["filename"] == "owned-user.pdf"
