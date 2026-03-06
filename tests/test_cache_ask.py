from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np
from fastapi.testclient import TestClient


class FakeCache:
    def __init__(self):
        self.kv = {}

    def get_json(self, key):
        value = self.kv.get(key)
        return type("R", (), {"hit": value is not None, "value": value})

    def set_json(self, key, value, ttl):
        self.kv[key] = value

    def get_embedding(self, key):
        value = self.kv.get(key)
        return type("R", (), {"hit": value is not None, "value": value})

    def set_embedding(self, key, emb, ttl):
        self.kv[key] = np.asarray(emb, dtype=np.float32).reshape(-1)


class DummyEmb:
    def encode_texts(self, texts):
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


class DummyQA:
    def answer(self, question, context):
        class Result:
            answer = "ANSWER"
            score = 0.9

        return Result()


def test_answer_cache_hit(
    client: TestClient,
    services,
    temp_data_dir: Path,
    create_owned_document,
    monkeypatch,
):
    services.embedding = DummyEmb()
    services.qa = DummyQA()
    services.ner = None
    services.cache = FakeCache()

    import app.services.retrieval.retriever as retr_mod
    from app.services.retrieval.retriever import RetrievedChunk

    doc_id = uuid.uuid4().hex
    create_owned_document(client, doc_id=doc_id)
    processed = temp_data_dir / "processed" / doc_id
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "faiss.index").write_bytes(b"index")

    def fake_search(self, doc_id, query, top_k, query_emb=None):
        return [
            RetrievedChunk(
                doc_id=doc_id,
                chunk_id="chunk_1",
                score=0.99,
                page=1,
                chunk_index=0,
                text_snippet="Some context",
            )
        ]

    monkeypatch.setattr(retr_mod.RetrieverService, "search", fake_search)

    first = client.post(
        "/ask",
        json={"question": "What is it?", "scope": "all", "top_k": 1},
    )
    assert first.status_code == 200
    assert first.json()["answer"] == "ANSWER"

    second = client.post(
        "/ask",
        json={"question": "What is it?", "scope": "all", "top_k": 1},
    )
    assert second.status_code == 200
    assert second.json()["answer"] == "ANSWER"
