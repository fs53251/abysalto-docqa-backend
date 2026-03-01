from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class DummyEmbeddingService:
    def encode_texts(self, texts):
        raise NotImplementedError


class DummyQAService:
    def answer(self, question: str, context: str):
        class R:
            answer = "MOCK ANSWER"
            score = 0.9

        return R()


class DummyNerService:
    def extract_entities(self, answer, sources):
        return [
            {
                "text": "John Doe",
                "label": "PERSON",
                "start": 0,
                "end": 8,
                "source": "answer",
                "doc_id": None,
                "page": None,
                "chunk_id": None,
            }
        ]


def test_ask_returns_answer_and_sources(monkeypatch):
    client.app.state.embedding_service = DummyEmbeddingService()
    client.app.state.qa_service = DummyQAService()
    client.app.state.ner_service = DummyNerService()

    import app.api.routes.ask as ask_route
    import app.services.retrieval.retriever as retr_mod
    from app.services.retrieval.retriever import RetrievedChunk

    monkeypatch.setattr(ask_route, "list_indexed_docs", lambda _: ["a" * 32])
    monkeypatch.setattr(ask_route, "validate_doc_ids", lambda ids: ids)

    def fake_search(self, doc_id, query, top_k):
        return [
            RetrievedChunk(
                doc_id=doc_id,
                chunk_id="chunk_1",
                score=0.99,
                page=1,
                chunk_index=0,
                text_snippet="This is relevant context.",
            )
        ]

    monkeypatch.setattr(retr_mod.RetrieverService, "search", fake_search)

    r = client.post("/ask", json={"question": "What is it?", "scope": "all", "top_k": 1})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["answer"] == "MOCK ANSWER"
    assert data["confidence"] == 0.9
    assert len(data["sources"]) == 1
    assert data["sources"][0]["doc_id"] == "a" * 32
    assert data["sources"][0]["chunk_id"] == "chunk_1"
    assert len(data["entities"]) == 1
    assert data["entities"][0]["text"] == "John Doe"
