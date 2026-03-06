from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.services.interfaces import RedisClientPort
from app.services.redis_client import create_redis_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheGetResult:
    hit: bool
    value: Any | None


class RedisCache:
    def __init__(self, client: RedisClientPort):
        self.client = client

    @staticmethod
    def connect(url: str) -> RedisClientPort:
        return create_redis_client(url)

    def get_json(self, key: str) -> CacheGetResult:
        raw = self.client.get(key)
        if raw is None:
            return CacheGetResult(hit=False, value=None)

        try:
            return CacheGetResult(hit=True, value=json.loads(raw.decode("utf-8")))
        except Exception:
            return CacheGetResult(hit=False, value=None)

    def set_json(self, key: str, value: Any, ttl: int) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.client.set(key, payload, ex=ttl)

    def get_embedding(self, key: str) -> CacheGetResult:
        raw = self.client.get(key)
        if raw is None:
            return CacheGetResult(hit=False, value=None)

        try:
            arr = np.frombuffer(raw, dtype=np.float32)
            return CacheGetResult(hit=True, value=arr)
        except Exception:
            return CacheGetResult(hit=False, value=None)

    def set_embedding(self, key: str, emb: np.ndarray, ttl: int) -> None:
        arr = np.asarray(emb, dtype=np.float32).reshape(-1)
        self.client.set(key, arr.tobytes(), ex=ttl)
