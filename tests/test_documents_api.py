from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.db.session import get_sessionmaker
from app.repositories.documents import get_document


def _write_document_artifacts(
    temp_data_dir: Path,
    doc_id: str,
    *,
    page_count: int = 2,
    chunk_count: int = 3,
) -> None:
    processed_dir = temp_data_dir / "processed" / doc_id
    uploads_dir = temp_data_dir / "uploads" / doc_id

    processed_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    (uploads_dir / "metadata.json").write_text(
        json.dumps({"doc_id": doc_id}, indent=2),
        encoding="utf-8",
    )
    original_dir = uploads_dir / "original"
    original_dir.mkdir(parents=True, exist_ok=True)
    (original_dir / "source.pdf").write_bytes(b"%PDF-1.4")

    (processed_dir / "text.json").write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "page_count": page_count,
                "pages": [
                    {"page": idx + 1, "text": f"Page {idx + 1}"}
                    for idx in range(page_count)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (processed_dir / "chunk_map.json").write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "chunks": [
                    {"chunk_id": f"chunk-{idx + 1}", "page": 1, "chunk_index": idx}
                    for idx in range(chunk_count)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (processed_dir / "chunks.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "chunk_id": f"chunk-{idx + 1}",
                    "doc_id": doc_id,
                    "page": 1,
                    "chunk_index": idx,
                    "text": f"Chunk {idx + 1}",
                }
            )
            for idx in range(chunk_count)
        )
        + "\n",
        encoding="utf-8",
    )

    (processed_dir / "embeddings.npy").write_bytes(b"fake")
    (processed_dir / "embeddings_meta.jsonl").write_text(
        json.dumps({"row": 0, "chunk_id": "chunk-1"}) + "\n",
        encoding="utf-8",
    )
    (processed_dir / "embeddings_info.json").write_text(
        json.dumps({"row_count": chunk_count, "dim": 3}, indent=2),
        encoding="utf-8",
    )
    (processed_dir / "faiss.index").write_bytes(b"fake-index")
    (processed_dir / "faiss_meta.json").write_text(
        json.dumps({"row_count": chunk_count, "dim": 3}, indent=2),
        encoding="utf-8",
    )


def test_documents_list_returns_only_own_documents(
    client: TestClient,
    temp_data_dir: Path,
    create_owned_document,
) -> None:
    owned = create_owned_document(client, filename="owned.pdf")
    _write_document_artifacts(temp_data_dir, owned.doc_id, page_count=4, chunk_count=6)

    payload = client.get("/documents").json()
    assert payload["count"] == 1
    assert payload["documents"][0]["doc_id"] == owned.doc_id
    assert payload["documents"][0]["filename"] == "owned.pdf"
    assert payload["documents"][0]["pages"] == 4
    assert payload["documents"][0]["chunks"] == 6

    from app.main import app
    from app.api.deps import (
        get_embedding_service,
        get_optional_cache,
        get_optional_ner_service,
        get_qa_service,
    )

    with TestClient(app) as other_client:
        other_client.app.dependency_overrides[get_embedding_service] = (
            client.app.dependency_overrides[get_embedding_service]
        )
        other_client.app.dependency_overrides[get_qa_service] = (
            client.app.dependency_overrides[get_qa_service]
        )
        other_client.app.dependency_overrides[get_optional_ner_service] = (
            client.app.dependency_overrides[get_optional_ner_service]
        )
        other_client.app.dependency_overrides[get_optional_cache] = (
            client.app.dependency_overrides[get_optional_cache]
        )

        other_payload = other_client.get("/documents").json()
        assert other_payload["count"] == 0

        other_client.app.dependency_overrides.clear()


def test_get_document_detail_returns_metadata_and_artifact_flags(
    client: TestClient,
    temp_data_dir: Path,
    create_owned_document,
) -> None:
    owned = create_owned_document(client, filename="detail.pdf", status="indexed")
    _write_document_artifacts(temp_data_dir, owned.doc_id, page_count=2, chunk_count=5)

    response = client.get(f"/documents/{owned.doc_id}")
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["doc_id"] == owned.doc_id
    assert payload["filename"] == "detail.pdf"
    assert payload["owner_type"] == "session"
    assert payload["status"] == "indexed"
    assert payload["page_count"] == 2
    assert payload["chunk_count"] == 5
    assert payload["artifacts"]["has_metadata"] is True
    assert payload["artifacts"]["has_original"] is True
    assert payload["artifacts"]["has_text"] is True
    assert payload["artifacts"]["has_chunks"] is True
    assert payload["artifacts"]["has_embeddings"] is True
    assert payload["artifacts"]["has_index"] is True


def test_direct_access_to_foreign_document_returns_404(
    client: TestClient,
    create_owned_document,
) -> None:
    owned = create_owned_document(client)

    from app.main import app
    from app.api.deps import (
        get_embedding_service,
        get_optional_cache,
        get_optional_ner_service,
        get_qa_service,
    )

    with TestClient(app) as other_client:
        other_client.app.dependency_overrides[get_embedding_service] = (
            client.app.dependency_overrides[get_embedding_service]
        )
        other_client.app.dependency_overrides[get_qa_service] = (
            client.app.dependency_overrides[get_qa_service]
        )
        other_client.app.dependency_overrides[get_optional_ner_service] = (
            client.app.dependency_overrides[get_optional_ner_service]
        )
        other_client.app.dependency_overrides[get_optional_cache] = (
            client.app.dependency_overrides[get_optional_cache]
        )

        response = other_client.get(f"/documents/{owned.doc_id}")
        assert response.status_code == 404, response.text

        other_client.app.dependency_overrides.clear()


def test_delete_own_document_removes_db_record_listing_and_disk(
    client: TestClient,
    temp_data_dir: Path,
    create_owned_document,
) -> None:
    owned = create_owned_document(client, filename="delete-me.pdf")
    _write_document_artifacts(temp_data_dir, owned.doc_id, page_count=1, chunk_count=2)

    upload_dir = temp_data_dir / "uploads" / owned.doc_id
    processed_dir = temp_data_dir / "processed" / owned.doc_id
    assert upload_dir.exists()
    assert processed_dir.exists()

    delete_response = client.delete(f"/documents/{owned.doc_id}")
    assert delete_response.status_code == 200, delete_response.text
    assert delete_response.json() == {"doc_id": owned.doc_id, "status": "deleted"}

    list_response = client.get("/documents")
    assert list_response.status_code == 200, list_response.text
    assert list_response.json()["count"] == 0

    assert not upload_dir.exists()
    assert not processed_dir.exists()

    session_local = get_sessionmaker()
    db = session_local()
    try:
        assert get_document(db, doc_id=owned.doc_id.replace("-", "")) is None
    except Exception:
        # fallback below because repository expects UUID, not public string
        pass
    finally:
        db.close()

    from app.core.identifiers import parse_document_public_id

    session_local = get_sessionmaker()
    db = session_local()
    try:
        deleted = get_document(db, doc_id=parse_document_public_id(owned.doc_id))
        assert deleted is None
    finally:
        db.close()


def test_delete_foreign_document_returns_404(
    client: TestClient,
    create_owned_document,
) -> None:
    owned = create_owned_document(client)

    from app.main import app
    from app.api.deps import (
        get_embedding_service,
        get_optional_cache,
        get_optional_ner_service,
        get_qa_service,
    )

    with TestClient(app) as other_client:
        other_client.app.dependency_overrides[get_embedding_service] = (
            client.app.dependency_overrides[get_embedding_service]
        )
        other_client.app.dependency_overrides[get_qa_service] = (
            client.app.dependency_overrides[get_qa_service]
        )
        other_client.app.dependency_overrides[get_optional_ner_service] = (
            client.app.dependency_overrides[get_optional_ner_service]
        )
        other_client.app.dependency_overrides[get_optional_cache] = (
            client.app.dependency_overrides[get_optional_cache]
        )

        response = other_client.delete(f"/documents/{owned.doc_id}")
        assert response.status_code == 404, response.text

        other_client.app.dependency_overrides.clear()
