from __future__ import annotations

import uuid

import numpy as np
from fastapi.testclient import TestClient


class DummyEmbeddingService:
    def encode_texts(self, texts):
        raise NotImplementedError


class DummyQAService:
    def answer(self, question: str, context: str):
        class Result:
            answer = "MOCK ANSWER"
            score = 0.9

        return Result()


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


def test_ask_returns_answer_sources_filename_and_grounding(
    client: TestClient,
    services,
    temp_data_dir,
    create_owned_document,
    monkeypatch,
):
    services.embedding = DummyEmbeddingService()
    monkeypatch.setattr(
        services.embedding,
        "encode_texts",
        lambda texts: np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    services.qa = DummyQAService()
    services.ner = DummyNerService()

    import app.services.retrieval.retriever as retr_mod
    from app.services.retrieval.retriever import RetrievedChunk

    doc_id = uuid.uuid4().hex
    create_owned_document(
        client, doc_id=doc_id, filename="unit-doc.pdf", status="indexed"
    )
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
                text_snippet="This is relevant context for the answer.",
                text="This is relevant context for the answer.",
                semantic_score=0.95,
                lexical_score=0.62,
                combined_score=0.99,
            )
        ]

    monkeypatch.setattr(retr_mod.RetrieverService, "search", fake_search)

    response = client.post(
        "/ask", json={"question": "What is it?", "scope": "all", "top_k": 1}
    )
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["answer"] == "MOCK ANSWER"
    assert data["grounded"] is True
    assert data["confidence"] == 0.9
    assert data["confidence_label"] == "high"
    assert (
        data["message"]
        == "Structured answer generated from retrieved document evidence."
    )
    assert len(data["sources"]) == 1
    assert data["sources"][0]["doc_id"] == doc_id
    assert data["sources"][0]["filename"] == "unit-doc.pdf"
    assert data["sources"][0]["chunk_id"] == "chunk_1"
    assert data["sources"][0]["semantic_score"] == 0.95
    assert data["sources"][0]["lexical_score"] == 0.62
    assert len(data["entities"]) == 1
    assert data["entities"][0]["text"] == "John Doe"
