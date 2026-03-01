import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class DummyEmbeddingService:
    def encode_texts(self, texts):
        raise NotImplementedError


class DummyQAService:
    def answer(self, question: str, context: str):
        class R:
            answer = "INTEGRATION ANSWER"
            score = 0.8

        return R()


def write_chunks_and_embeddings(temp_data_dir: Path, doc_id: str):
    processed = temp_data_dir / "processed" / doc_id
    processed.mkdir(parents=True, exist_ok=True)

    (processed / "chunks.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chunk_id": "chunk_a",
                        "doc_id": doc_id,
                        "page": 1,
                        "chunk_index": 0,
                        "text": "alpha invoice total is 100 EUR",
                        "char_start": 0,
                        "char_end": 10,
                        "source": "pymupdf",
                        "confidence": None,
                    }
                ),
                json.dumps(
                    {
                        "chunk_id": "chunk_b",
                        "doc_id": doc_id,
                        "page": 2,
                        "chunk_index": 1,
                        "text": "beta contract starts on monday",
                        "char_start": 0,
                        "char_end": 10,
                        "source": "pymupdf",
                        "confidence": None,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    np.save(processed / "embeddings.npy", emb)

    (processed / "embeddings_meta.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {"row": 0, "chunk_id": "chunk_a", "doc_id": doc_id, "page": 1, "chunk_index": 0}
                ),
                json.dumps(
                    {"row": 1, "chunk_id": "chunk_b", "doc_id": doc_id, "page": 2, "chunk_index": 1}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
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


def test_ask_happy_path_with_index(temp_data_dir: Path, monkeypatch):
    doc_id = "f" * 32
    write_chunks_and_embeddings(temp_data_dir, doc_id)

    client.app.state.embedding_service = DummyEmbeddingService()
    client.app.state.qa_service = DummyQAService()

    monkeypatch.setattr(
        client.app.state.embedding_service,
        "encode_texts",
        lambda texts: np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )

    r1 = client.post(f"/documents/{doc_id}/index")
    assert r1.status_code == 200, r1.text

    r2 = client.post(
        "/ask",
        json={
            "question": "What is the invoice total?",
            "scope": "docs",
            "doc_ids": [doc_id],
            "top_k": 1,
        },
    )
    assert r2.status_code == 200, r2.text
    data = r2.json()
    assert data["answer"] == "INTEGRATION ANSWER"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["doc_id"] == doc_id
    assert data["sources"][0]["chunk_id"] == "chunk_a"
