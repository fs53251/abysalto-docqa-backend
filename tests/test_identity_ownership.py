from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("faiss")


def _write_text_json(temp_data_dir: Path, doc_id: str) -> None:
    processed = temp_data_dir / "processed" / doc_id
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "text.json").write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "page_count": 1,
                "pages": [
                    {
                        "page": 1,
                        "text": "owned content for chunking",
                        "source": "pymupdf",
                        "confidence": None,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_chunks_and_embeddings(temp_data_dir: Path, doc_id: str) -> None:
    processed = temp_data_dir / "processed" / doc_id
    processed.mkdir(parents=True, exist_ok=True)

    (processed / "chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "chunk-1",
                "doc_id": doc_id,
                "page": 1,
                "chunk_index": 0,
                "text": "owned content for search",
                "char_start": 0,
                "char_end": 22,
                "source": "pymupdf",
                "confidence": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    np.save(
        processed / "embeddings.npy",
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    (processed / "embeddings_meta.jsonl").write_text(
        json.dumps(
            {
                "row": 0,
                "chunk_id": "chunk-1",
                "doc_id": doc_id,
                "page": 1,
                "chunk_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (processed / "embeddings_info.json").write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "row_count": 1,
                "dim": 3,
                "embedding_model": "dummy",
                "normalize": True,
                "batch_size": 1,
                "chunking_version": "testver",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_authenticated_upload_and_documents_listing_use_user_identity(
    client: TestClient,
    register_and_login,
) -> None:
    auth_user = register_and_login(email="owner@example.com")

    upload_res = client.post(
        "/upload",
        headers=auth_user.headers,
        files=[
            (
                "files",
                ("owned.pdf", BytesIO(b"%PDF-1.4 fake pdf content"), "application/pdf"),
            )
        ],
    )
    assert upload_res.status_code == 200, upload_res.text
    uploaded_doc_id = upload_res.json()["documents"][0]["doc_id"]

    docs_as_user = client.get("/documents", headers=auth_user.headers)
    assert docs_as_user.status_code == 200, docs_as_user.text
    docs_payload = docs_as_user.json()
    assert docs_payload["count"] == 1
    assert docs_payload["documents"][0]["doc_id"] == uploaded_doc_id

    docs_as_session = client.get("/documents")
    assert docs_as_session.status_code == 200, docs_as_session.text
    assert docs_as_session.json()["count"] == 0


def test_user_owner_can_use_direct_document_endpoints_and_other_user_gets_404(
    client: TestClient,
    services,
    temp_data_dir: Path,
    register_and_login,
    create_user_owned_document,
    monkeypatch,
) -> None:
    owner = register_and_login(email="user-a@example.com")
    other = register_and_login(email="user-b@example.com")

    owned = create_user_owned_document(user_id=owner.user_id)
    _write_text_json(temp_data_dir, owned.doc_id)
    _write_chunks_and_embeddings(temp_data_dir, owned.doc_id)

    monkeypatch.setattr(
        services.embedding,
        "encode_texts",
        lambda texts: np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )

    owner_chunk = client.post(
        f"/documents/{owned.doc_id}/chunk",
        headers=owner.headers,
    )
    assert owner_chunk.status_code == 200, owner_chunk.text

    owner_embed = client.post(
        f"/documents/{owned.doc_id}/embed",
        params={"force": True},
        headers=owner.headers,
    )
    assert owner_embed.status_code == 200, owner_embed.text

    owner_index = client.post(
        f"/documents/{owned.doc_id}/index",
        headers=owner.headers,
    )
    assert owner_index.status_code == 200, owner_index.text

    owner_search = client.post(
        f"/documents/{owned.doc_id}/search",
        json={"query": "owned", "top_k": 1},
        headers=owner.headers,
    )
    assert owner_search.status_code == 200, owner_search.text

    denied_chunk = client.post(
        f"/documents/{owned.doc_id}/chunk",
        headers=other.headers,
    )
    assert denied_chunk.status_code == 404, denied_chunk.text

    denied_embed = client.post(
        f"/documents/{owned.doc_id}/embed",
        headers=other.headers,
    )
    assert denied_embed.status_code == 404, denied_embed.text

    denied_index = client.post(
        f"/documents/{owned.doc_id}/index",
        headers=other.headers,
    )
    assert denied_index.status_code == 404, denied_index.text

    denied_search = client.post(
        f"/documents/{owned.doc_id}/search",
        json={"query": "owned", "top_k": 1},
        headers=other.headers,
    )
    assert denied_search.status_code == 404, denied_search.text


def test_ask_with_explicit_foreign_doc_id_returns_403(
    client: TestClient,
    services,
    temp_data_dir: Path,
    register_and_login,
    create_user_owned_document,
    monkeypatch,
) -> None:
    owner = register_and_login(email="ask-owner@example.com")
    other = register_and_login(email="ask-other@example.com")

    owned = create_user_owned_document(user_id=owner.user_id)
    _write_chunks_and_embeddings(temp_data_dir, owned.doc_id)

    monkeypatch.setattr(
        services.embedding,
        "encode_texts",
        lambda texts: np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )

    build_response = client.post(
        f"/documents/{owned.doc_id}/index",
        headers=owner.headers,
    )
    assert build_response.status_code == 200, build_response.text

    ask_response = client.post(
        "/ask",
        json={
            "question": "What is in the document?",
            "scope": "docs",
            "doc_ids": [owned.doc_id],
            "top_k": 1,
        },
        headers=other.headers,
    )
    assert ask_response.status_code == 403, ask_response.text
    assert ask_response.json()["error_code"] == "doc_forbidden"
