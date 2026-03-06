from __future__ import annotations

import redis

from app.services.interfaces import RedisClientPort


def create_redis_client(url: str) -> RedisClientPort:
    pool = redis.ConnectionPool.from_url(url, decode_responses=False)
    return redis.Redis(connection_pool=pool)
