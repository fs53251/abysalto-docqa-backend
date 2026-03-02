import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class FakeCache:
    def __init__(self):
        self.kv = {}

    def get_json(self, k):
        v = self.kv.get(k)
        return type("R", (), {"hit": v is not None, "value": v})

    def set_json(self, k, v, ttl):
        self.kv[k] = v

    def get_embedding(self, k):
        v = self.kv.get(k)
        return type("R", (), {"hit": v is not None, "value": v})

    def set_embedding(self, k, emb, ttl):
        self.kv[k] = np.asarray(emb, dtype=np.float32).reshape(-1)


class DummyEmb:
    def encode_texts(self, texts):
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


class DummyQA:
    def answer(self, question, context):
        class R:
            answer = "ANSWER"
            score = 0.9

        return R()


def test_answer_cache_hit(monkeypatch):
    client.app.state.embedding_service = DummyEmb()
    client.app.state.qa_service = DummyQA()
    client.app.state.ner_service = None
    client.app.state.cache = FakeCache()

    import app.api.routes.ask as ask_route
    import app.services.retrieval.retriever as retr_mod
    from app.services.retrieval.retriever import RetrievedChunk

    monkeypatch.setattr(ask_route, "list_indexed_docs", lambda _: ["a" * 32])
    monkeypatch.setattr(ask_route, "validate_doc_ids", lambda ids: ids)

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

    r1 = client.post("/ask", json={"question": "What is it?", "scope": "all", "top_k": 1})
    assert r1.status_code == 200
    assert r1.json()["answer"] == "ANSWER"

    r2 = client.post("/ask", json={"question": "What is it?", "scope": "all", "top_k": 1})
    assert r2.status_code == 200
    assert r2.json()["answer"] == "ANSWER"
