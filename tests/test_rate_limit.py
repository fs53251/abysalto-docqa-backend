from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.core.config import settings


class FakeRedisClient:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.ttls: dict[str, int] = {}
        self.seen_keys: list[str] = []

    def ping(self):
        return True

    def get(self, key: str):
        return None

    def set(self, key: str, value, ex: int | None = None):
        return True

    def incr(self, key: str) -> int:
        self.seen_keys.append(key)
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    def expire(self, key: str, seconds: int):
        self.ttls[key] = seconds
        return True

    def ttl(self, key: str) -> int:
        return self.ttls.get(key, -1)


class DummyQAService:
    def answer(self, question: str, context: str):
        class Result:
            answer = "RATE LIMITED ANSWER"
            score = 0.9

        return Result()


def test_ask_rate_limit_returns_429_and_retry_after(
    client: TestClient,
    services,
    temp_data_dir: Path,
    create_owned_document,
    monkeypatch,
) -> None:
    fake_redis = FakeRedisClient()
    services.redis_client = fake_redis
    services.cache = None
    services.qa = DummyQAService()

    monkeypatch.setattr(settings, "ASK_RATE_LIMIT_PER_MIN", 1, raising=False)
    monkeypatch.setattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 60, raising=False)
    monkeypatch.setattr(
        services.embedding,
        "encode_texts",
        lambda texts: np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )

    import app.services.retrieval.retriever as retr_mod
    from app.services.retrieval.retriever import RetrievedChunk

    owned = create_owned_document(
        client,
        filename="ask-rate-limit.pdf",
        status="indexed",
    )
    processed = temp_data_dir / "processed" / owned.doc_id
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "faiss.index").write_bytes(b"index")

    def fake_search(self, doc_id, query, top_k, query_emb=None):
        return [
            RetrievedChunk(
                doc_id=doc_id,
                chunk_id="chunk-1",
                score=0.99,
                page=1,
                chunk_index=0,
                text_snippet="context",
            )
        ]

    monkeypatch.setattr(retr_mod.RetrieverService, "search", fake_search)

    first = client.post(
        "/ask",
        json={"question": "hello", "top_k": 1},
    )
    assert first.status_code == 200, first.text

    second = client.post(
        "/ask",
        json={"question": "hello again", "top_k": 1},
    )
    assert second.status_code == 429, second.text
    assert second.json()["error_code"] == "rate_limit_exceeded"
    assert second.headers["Retry-After"] == "60"
    assert any(key.startswith("rl:ask:sess:") for key in fake_redis.seen_keys)


def test_upload_rate_limit_returns_429(
    client: TestClient,
    services,
    monkeypatch,
) -> None:
    fake_redis = FakeRedisClient()
    services.redis_client = fake_redis

    monkeypatch.setattr(settings, "UPLOAD_RATE_LIMIT_PER_MIN", 1, raising=False)
    monkeypatch.setattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 60, raising=False)
    monkeypatch.setattr(settings, "UPLOAD_AUTO_PROCESS", False, raising=False)

    first = client.post(
        "/upload",
        files=[
            (
                "files",
                ("first.pdf", BytesIO(b"%PDF-1.4 rate limit"), "application/pdf"),
            )
        ],
    )
    assert first.status_code == 200, first.text

    second = client.post(
        "/upload",
        files=[
            (
                "files",
                ("second.pdf", BytesIO(b"%PDF-1.4 rate limit"), "application/pdf"),
            )
        ],
    )
    assert second.status_code == 429, second.text
    assert second.json()["error_code"] == "rate_limit_exceeded"
    assert second.headers["Retry-After"] == "60"
    assert any(key.startswith("rl:upload:sess:") for key in fake_redis.seen_keys)


def test_login_rate_limit_returns_429_and_uses_login_key(
    client: TestClient,
    services,
    monkeypatch,
) -> None:
    fake_redis = FakeRedisClient()
    services.redis_client = fake_redis

    monkeypatch.setattr(settings, "LOGIN_RATE_LIMIT_PER_MIN", 1, raising=False)
    monkeypatch.setattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 60, raising=False)

    register_res = client.post(
        "/auth/register",
        json={"email": "rate-limit@example.com", "password": "supersecret123"},
    )
    assert register_res.status_code == 201, register_res.text

    first = client.post(
        "/auth/login",
        json={"email": "rate-limit@example.com", "password": "supersecret123"},
    )
    assert first.status_code == 200, first.text

    second = client.post(
        "/auth/login",
        json={"email": "rate-limit@example.com", "password": "supersecret123"},
    )
    assert second.status_code == 429, second.text
    assert second.json()["error_code"] == "rate_limit_exceeded"
    assert second.headers["Retry-After"] == "60"
    assert any(key.startswith("rl:login:testclient:") for key in fake_redis.seen_keys)
