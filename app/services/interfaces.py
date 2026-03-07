from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingServicePort(Protocol):
    def load(self) -> None: ...
    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class QaResult:
    answer: str
    score: float | None


@runtime_checkable
class QaServicePort(Protocol):
    model_name: str

    def load(self) -> None: ...
    def answer(self, question: str, context: str) -> QaResult: ...


@runtime_checkable
class NerServicePort(Protocol):
    model_name: str

    def load(self) -> None: ...
    def extract_entities(
        self, answer: str, sources: list[Any]
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class CacheGetResult:
    hit: bool
    value: Any | None


@runtime_checkable
class CachePort(Protocol):
    def get_json(self, key: str) -> CacheGetResult: ...
    def set_json(self, key: str, value: Any, ttl: int) -> None: ...

    def get_embedding(self, key: str) -> CacheGetResult: ...
    def set_embedding(self, key: str, emb: np.ndarray, ttl: int) -> None: ...


@runtime_checkable
class RedisClientPort(Protocol):
    def ping(self) -> Any: ...
    def get(self, key: str) -> Any: ...
    def set(self, key: str, value: Any, ex: int | None = None) -> Any: ...
    def incr(self, key: str) -> int: ...
    def expire(self, key: str, seconds: int) -> Any: ...
    def ttl(self, key: str) -> int: ...
