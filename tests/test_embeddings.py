import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class DummyEmbeddingService:
    def encode_texts(self, texts):
        raise NotImplementedError


def write_chunks_jsonl(temp_data_dir: Path, doc_id: str, n: int = 3):
    processed = temp_data_dir / "processed" / doc_id
    processed.mkdir(parents=True, exist_ok=True)

    p = processed / "chunks.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "chunk_id": f"chunk_{i}",
                        "doc_id": doc_id,
                        "page": 1,
                        "chunk_index": i,
                        "text": f"chunk text {i}",
                        "char_start": 0,
                        "char_end": 10,
                        "source": "pymupdf",
                        "confidence": None,
                    }
                )
                + "\n"
            )
    return p


def test_embed_endpoint_creates_embeddings_files(temp_data_dir: Path, monkeypatch):
    doc_id = "d" * 32
    write_chunks_jsonl(temp_data_dir, doc_id, n=4)

    # Attach a dummy embedding service to the FastAPI app used by the client
    client.app.state.embedding_service = DummyEmbeddingService()

    def fake_encode_texts(texts):
        # deterministic embeddings: N x 5
        n = len(texts)
        out = np.zeros((n, 5), dtype=np.float32)
        for i in range(n):
            out[i, 0] = float(i)
        return out

    monkeypatch.setattr(client.app.state.embedding_service, "encode_texts", fake_encode_texts)

    r = client.post(f"/documents/{doc_id}/embed")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "embedded"
    assert data["row_count"] == 4
    assert data["dim"] == 5

    npy = Path(data["embeddings_npy"])
    meta = Path(data["embeddings_meta_jsonl"])
    info = Path(data["embeddings_info"])

    assert npy.exists()
    assert meta.exists()
    assert info.exists()

    m = np.load(npy)
    assert m.shape == (4, 5)

    lines = meta.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    first = json.loads(lines[0])
    assert first["row"] == 0
    assert first["chunk_id"] == "chunk_0"


def test_embed_requires_chunks_first(temp_data_dir: Path):
    doc_id = "e" * 32
    client.app.state.embedding_service = DummyEmbeddingService()

    r = client.post(f"/documents/{doc_id}/embed")
    assert r.status_code == 404
