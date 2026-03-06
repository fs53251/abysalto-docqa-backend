import json
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("faiss")


class DummyEmbeddingService:
    def encode_texts(self, texts):
        raise NotImplementedError


def write_chunks_and_embeddings(temp_data_dir: Path, doc_id: str):
    processed = temp_data_dir / "processed" / doc_id
    processed.mkdir(parents=True, exist_ok=True)

    chunks_path = processed / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "chunk_id": "chunk_a",
                    "doc_id": doc_id,
                    "page": 1,
                    "chunk_index": 0,
                    "text": "alpha content about invoices",
                    "char_start": 0,
                    "char_end": 10,
                    "source": "pymupdf",
                    "confidence": None,
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "chunk_id": "chunk_b",
                    "doc_id": doc_id,
                    "page": 2,
                    "chunk_index": 1,
                    "text": "beta content about contracts",
                    "char_start": 0,
                    "char_end": 10,
                    "source": "pymupdf",
                    "confidence": None,
                }
            )
            + "\n"
        )

    emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    np.save(processed / "embeddings.npy", emb)

    meta_path = processed / "embeddings_meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "row": 0,
                    "chunk_id": "chunk_a",
                    "doc_id": doc_id,
                    "page": 1,
                    "chunk_index": 0,
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "row": 1,
                    "chunk_id": "chunk_b",
                    "doc_id": doc_id,
                    "page": 2,
                    "chunk_index": 1,
                }
            )
            + "\n"
        )

    (processed / "embeddings_info.json").write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "row_count": 2,
                "dim": 3,
                "embedding_model": "dummy",
                "normalize": True,
                "batch_size": 2,
                "chunking_version": "testver",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_build_index_and_search_returns_expected_chunk(
    client: TestClient,
    temp_data_dir: Path,
    monkeypatch,
    create_owned_document,
) -> None:
    doc_id = uuid.uuid4().hex
    create_owned_document(client, doc_id=doc_id)
    write_chunks_and_embeddings(temp_data_dir, doc_id)

    client.app.state.embedding_service = DummyEmbeddingService()

    def fake_encode_texts(texts):
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(
        client.app.state.embedding_service,
        "encode_texts",
        fake_encode_texts,
    )

    build_response = client.post(f"/documents/{doc_id}/index")
    assert build_response.status_code == 200, build_response.text
    assert build_response.json()["status"] in ("indexed", "already_indexed")

    search_response = client.post(
        f"/documents/{doc_id}/search",
        json={"query": "invoice", "top_k": 1},
    )
    assert search_response.status_code == 200, search_response.text
    hits = search_response.json()["hits"]
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "chunk_a"
