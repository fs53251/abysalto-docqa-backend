import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class DummyEmbeddingService:
    def encode_texts(self, texts):
        raise NotImplementedError


def write_chunks_and_embeddings(temp_data_dir: Path, doc_id: str):
    processed = temp_data_dir / "processed" / doc_id
    processed.mkdir(parents=True, exist_ok=True)

    # chunks.jsonl (2 chunks)
    chunks_path = processed / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        f.write(
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
        f.write(
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

    # embeddings.npy (2 x 3) - normalized vectors for IP search
    emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    np.save(processed / "embeddings.npy", emb)

    # embeddings_meta.jsonl row mapping
    meta_path = processed / "embeddings_meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"row": 0, "chunk_id": "chunk_a", "doc_id": doc_id, "page": 1, "chunk_index": 0}
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {"row": 1, "chunk_id": "chunk_b", "doc_id": doc_id, "page": 2, "chunk_index": 1}
            )
            + "\n"
        )

    # embeddings_info.json
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


def test_build_index_and_search_returns_expected_chunk(temp_data_dir: Path, monkeypatch):
    doc_id = "f" * 32
    write_chunks_and_embeddings(temp_data_dir, doc_id)

    # Attach dummy embedding service to app
    client.app.state.embedding_service = DummyEmbeddingService()

    # Query embedding equals [1,0,0] -> should retrieve chunk_a first
    def fake_encode_texts(texts):
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(client.app.state.embedding_service, "encode_texts", fake_encode_texts)

    # Build index
    r1 = client.post(f"/documents/{doc_id}/index")
    assert r1.status_code == 200, r1.text
    assert r1.json()["status"] in ("indexed", "already_indexed")

    # Search
    r2 = client.post(f"/documents/{doc_id}/search", json={"query": "invoice", "top_k": 1})
    assert r2.status_code == 200, r2.text
    hits = r2.json()["hits"]
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "chunk_a"
