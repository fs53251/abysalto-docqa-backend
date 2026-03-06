from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


class DummyEmbeddingService:
    """Deterministic fake for embedding pipeline tests."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        # Map each text to a simple, deterministic vector based on length.
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, 0] = float(len(t))
        return out


def _write_chunks_jsonl(temp_data_dir: Path, doc_id: str, n: int = 3) -> None:
    processed = temp_data_dir / "processed" / doc_id
    processed.mkdir(parents=True, exist_ok=True)

    chunks_path = processed / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "chunk_id": f"c{i}",
                        "page": 1,
                        "text": f"hello {i}",
                        "start_char": i * 10,
                        "end_char": i * 10 + 5,
                    }
                )
                + "\n"
            )


def test_embed_document_builds_artifacts(
    client: TestClient, services, temp_data_dir: Path
) -> None:
    # Arrange
    doc_id = "a" * 16
    _write_chunks_jsonl(temp_data_dir, doc_id, n=3)
    services.embedding = DummyEmbeddingService(dim=8)

    # Act
    res = client.post(f"/documents/{doc_id}/embed", params={"force": True})

    # Assert
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "embedded"
    assert body["row_count"] == 3
    assert body["dim"] == 8
