from __future__ import annotations

import hashlib
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeAlias

from fastapi import Depends, Request

from app.api.deps import CurrentIdentity, get_optional_redis_client
from app.core.config import settings
from app.core.errors import ApiError
from app.core.network import get_client_ip
from app.core.identity import RequestIdentity
from app.services.interfaces import RedisClientPort

logger = logging.getLogger(__name__)

IntProvider: TypeAlias = int | Callable[[], int]
RateLimitKeyFn: TypeAlias = Callable[
    [Request, RequestIdentity],
    str | Awaitable[str],
]


def _resolve_int(value: IntProvider) -> int:
    if callable(value):
        return int(value())
    return int(value)


async def _maybe_await(value: str | Awaitable[str]) -> str:
    if inspect.isawaitable(value):
        return str(await value)
    return str(value)


class RedisRateLimiter:
    def __init__(self, client: RedisClientPort) -> None:
        self.client = client

    def hit(
        self,
        *,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[int, int]:
        bucket = int(time.time()) // window_seconds
        redis_key = f"rl:{key}:{bucket}"

        count = int(self.client.incr(redis_key))
        if count == 1:
            self.client.expire(redis_key, window_seconds)

        ttl_raw = self.client.ttl(redis_key)
        try:
            ttl = int(ttl_raw)
        except (TypeError, ValueError):
            ttl = window_seconds

        if ttl <= 0:
            ttl = max(1, window_seconds - (int(time.time()) % window_seconds))

        return count, ttl


def identity_rate_limit_key(namespace: str) -> RateLimitKeyFn:
    def _key(request: Request, identity: RequestIdentity) -> str:
        del request
        return f"{namespace}:{identity.log_identity}"

    return _key


def login_rate_limit_key(namespace: str = "login") -> RateLimitKeyFn:
    async def _key(request: Request, identity: RequestIdentity) -> str:
        del identity

        email = ""
        try:
            raw_body = await request.body()
            if raw_body:
                payload = json.loads(raw_body)
                if isinstance(payload, dict):
                    email = str(payload.get("email", "")).strip().lower()
        except Exception:
            email = ""

        email_hash = (
            hashlib.sha256(email.encode("utf-8")).hexdigest()[:16]
            if email
            else "unknown"
        )
        return f"{namespace}:{get_client_ip(request)}:{email_hash}"

    return _key


def rate_limit(
    *,
    limit: IntProvider,
    window_seconds: IntProvider,
    key_fn: RateLimitKeyFn,
):
    async def _dependency(
        request: Request,
        identity: CurrentIdentity,
        redis_client: RedisClientPort | None = Depends(get_optional_redis_client),
    ) -> None:
        if not settings.ENABLE_RATE_LIMITING:
            return

        if redis_client is None:
            return

        limit_value = _resolve_int(limit)
        window_value = _resolve_int(window_seconds)

        if limit_value <= 0 or window_value <= 0:
            return

        try:
            key = await _maybe_await(key_fn(request, identity))
            limiter = RedisRateLimiter(redis_client)
            count, retry_after = limiter.hit(
                key=key,
                limit=limit_value,
                window_seconds=window_value,
            )
        except Exception:
            logger.warning(
                "Rate limit unavailable; allowing request",
                exc_info=True,
                extra={"path": request.url.path, "method": request.method},
            )
            return

        if count > limit_value:
            raise ApiError(
                status_code=429,
                error_code="rate_limit_exceeded",
                message="Rate limit exceeded.",
                headers={"Retry-After": str(retry_after)},
            )

    return _dependency
