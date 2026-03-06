from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

# We override these dependency providers:
from app.api.deps import (
    get_embedding_service,
    get_qa_service,
    get_optional_ner_service,
    get_optional_cache,
)

# --------- Default lightweight test doubles ---------


class DummyEmbeddingService:
    def load(self) -> None:
        return

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        # deterministic 3-dim vector per text
        out = np.zeros((len(texts), 3), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, 0] = float(len(t))
        return out


class DummyQAService:
    model_name = "dummy-qa"

    def load(self) -> None:
        return

    def answer(self, question: str, context: str):
        # minimal QaResult-like shape used downstream
        from app.services.interfaces import QaResult

        return QaResult(answer="dummy", score=0.99)


@dataclass
class TestServices:
    """
    A mutable container. Tests can override these fields without touching app.state.
    """

    embedding: Any
    qa: Any
    ner: Any | None = None
    cache: Any | None = None


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """
    Force deterministic settings for tests.
    Also ensures factories skip heavy model loading (APP_ENV == test).
    """
    monkeypatch.setattr(settings, "APP_ENV", "test", raising=False)
    monkeypatch.setattr(settings, "ENABLE_CACHE", False, raising=False)
    yield


@pytest.fixture()
def temp_data_dir(tmp_path: Path) -> Iterator[Path]:
    """
    Use an isolated DATA_DIR for tests.
    """
    old = settings.DATA_DIR
    settings.DATA_DIR = str(tmp_path)
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    yield tmp_path
    settings.DATA_DIR = old


@pytest.fixture()
def services() -> TestServices:
    """
    Default services for tests (fast, no heavy ML loads).
    Individual tests can mutate:
      services.embedding = CustomEmb()
      services.qa = CustomQA()
      services.cache = FakeCache()
    """
    return TestServices(
        embedding=DummyEmbeddingService(),
        qa=DummyQAService(),
        ner=None,
        cache=None,
    )


@pytest.fixture()
def client(services: TestServices) -> Iterator[TestClient]:
    """
    TestClient that uses FastAPI dependency overrides,
    so tests do NOT depend on app.state.
    """
    # Set dependency overrides
    app.dependency_overrides[get_embedding_service] = lambda: services.embedding
    app.dependency_overrides[get_qa_service] = lambda: services.qa
    app.dependency_overrides[get_optional_ner_service] = lambda: services.ner
    app.dependency_overrides[get_optional_cache] = lambda: services.cache

    with TestClient(app) as c:
        yield c

    # Cleanup
    app.dependency_overrides.clear()
