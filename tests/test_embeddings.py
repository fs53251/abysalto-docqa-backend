from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


class DummyEmbeddingService:
    """Deterministic fake for embedding pipeline tests."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for index, text in enumerate(texts):
            out[index, 0] = float(len(text))
        return out


def _write_chunks_jsonl(temp_data_dir: Path, doc_id: str, n: int = 3) -> None:
    processed = temp_data_dir / "processed" / doc_id
    processed.mkdir(parents=True, exist_ok=True)

    chunks_path = processed / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for index in range(n):
            handle.write(
                json.dumps(
                    {
                        "chunk_id": f"c{index}",
                        "page": 1,
                        "text": f"hello {index}",
                        "start_char": index * 10,
                        "end_char": index * 10 + 5,
                    }
                )
                + "\n"
            )


def test_embed_document_builds_artifacts(
    client: TestClient,
    services,
    temp_data_dir: Path,
    create_owned_document,
) -> None:
    doc_id = uuid.uuid4().hex
    create_owned_document(client, doc_id=doc_id)
    _write_chunks_jsonl(temp_data_dir, doc_id, n=3)
    services.embedding = DummyEmbeddingService(dim=8)

    response = client.post(f"/documents/{doc_id}/embed", params={"force": True})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "embedded"
    assert body["row_count"] == 3
    assert body["dim"] == 8
