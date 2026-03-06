from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.deps import (
    get_embedding_service,
    get_optional_cache,
    get_optional_ner_service,
    get_qa_service,
)
from app.main import app

fitz = pytest.importorskip("fitz")
pytest.importorskip("faiss")


def _make_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    payload = doc.tobytes()
    doc.close()
    return payload


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
        temp_data_dir / "processed" / doc_id / "embeddings.npy",
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


def test_other_session_cannot_extract_or_read_text(
    client: TestClient,
    services,
    temp_data_dir: Path,
):
    pdf_bytes = _make_pdf_bytes("owner only")
    upload_response = client.post(
        "/upload",
        files=[("files", ("owner.pdf", BytesIO(pdf_bytes), "application/pdf"))],
    )
    assert upload_response.status_code == 200, upload_response.text
    doc_id = upload_response.json()["documents"][0]["doc_id"]

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

        denied_extract = other_client.post(f"/documents/{doc_id}/extract-text")
        assert denied_extract.status_code == 404, denied_extract.text

        denied_text = other_client.get(f"/documents/{doc_id}/text")
        assert denied_text.status_code == 404, denied_text.text

        other_client.app.dependency_overrides.clear()


def test_other_session_cannot_use_chunk_embed_index_or_search(
    client: TestClient,
    services,
    temp_data_dir: Path,
    create_owned_document,
    monkeypatch,
):
    doc_id = uuid.uuid4().hex
    create_owned_document(client, doc_id=doc_id)
    _write_text_json(temp_data_dir, doc_id)
    _write_chunks_and_embeddings(temp_data_dir, doc_id)

    monkeypatch.setattr(
        services.embedding,
        "encode_texts",
        lambda texts: np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )

    assert client.post(f"/documents/{doc_id}/chunk").status_code == 200
    assert (
        client.post(f"/documents/{doc_id}/embed", params={"force": True}).status_code
        == 200
    )
    assert client.post(f"/documents/{doc_id}/index").status_code == 200
    assert (
        client.post(
            f"/documents/{doc_id}/search",
            json={"query": "owned", "top_k": 1},
        ).status_code
        == 200
    )

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

        assert other_client.post(f"/documents/{doc_id}/chunk").status_code == 404
        assert other_client.post(f"/documents/{doc_id}/embed").status_code == 404
        assert other_client.post(f"/documents/{doc_id}/index").status_code == 404
        assert (
            other_client.post(
                f"/documents/{doc_id}/search",
                json={"query": "owned", "top_k": 1},
            ).status_code
            == 404
        )

        other_client.app.dependency_overrides.clear()
