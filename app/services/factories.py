from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI

from app.core.config import settings
from app.services.cache.redis_cache import RedisCache
from app.services.indexing.embedding_service import default_embedding_service
from app.services.ner.ner_service import default_ner_service
from app.services.qa.qa_service import default_qa_service
from app.services.redis_client import create_redis_client

logger = logging.getLogger(__name__)


def _should_skip_service_init() -> bool:
    return settings.APP_ENV == "test" or "PYTEST_CURRENT_TEST" in os.environ


def _set_service_status(
    app: FastAPI,
    name: str,
    *,
    ready: bool,
    detail: str,
    extra: dict[str, Any] | None = None,
) -> None:
    statuses = getattr(app.state, "service_statuses", {})
    statuses[name] = {
        "ready": ready,
        "detail": detail,
        **(extra or {}),
    }
    app.state.service_statuses = statuses


def init_embedding_service(app: FastAPI) -> None:
    if _should_skip_service_init():
        app.state.embedding_service = None
        _set_service_status(
            app,
            "embedding",
            ready=False,
            detail="skipped in test runtime",
        )
        logger.info(
            "Embedding service skipped in test runtime (use dependency overrides)."
        )
        return

    try:
        svc = default_embedding_service()
        svc.load()
        app.state.embedding_service = svc
        _set_service_status(
            app,
            "embedding",
            ready=True,
            detail="initialized",
            extra={"model": svc.cfg.model_name},
        )
        logger.info("Embedding service ready: %s", svc.cfg.model_name)
    except Exception as e:
        app.state.embedding_service = None
        _set_service_status(
            app,
            "embedding",
            ready=False,
            detail="initialization failed",
            extra={"error": type(e).__name__},
        )
        logger.exception("Embedding service init failed (disabled): %s", e)


def init_qa_service(app: FastAPI) -> None:
    if _should_skip_service_init():
        app.state.qa_service = None
        _set_service_status(
            app,
            "qa",
            ready=False,
            detail="skipped in test runtime",
        )
        logger.info("QA service skipped in test runtime (use dependency overrides).")
        return

    try:
        qa = default_qa_service()
        qa.load()
        app.state.qa_service = qa
        _set_service_status(
            app,
            "qa",
            ready=True,
            detail=getattr(qa, "status_detail", "initialized"),
            extra={
                "model": qa.model_name,
                "backend": getattr(qa, "backend", "unknown"),
            },
        )
        logger.info(
            "QA service ready: %s (%s)",
            qa.model_name,
            getattr(qa, "backend", "unknown"),
        )
    except Exception as e:
        app.state.qa_service = None
        _set_service_status(
            app,
            "qa",
            ready=False,
            detail="initialization failed",
            extra={"error": type(e).__name__},
        )
        logger.exception("QA service init failed (disabled): %s", e)


def init_ner_service(app: FastAPI) -> None:
    if _should_skip_service_init():
        app.state.ner_service = None
        _set_service_status(
            app,
            "ner",
            ready=False,
            detail="skipped in test runtime",
        )
        logger.info("NER service skipped in test runtime (use dependency overrides).")
        return

    try:
        ner = default_ner_service()
        ner.load()
        app.state.ner_service = ner
        _set_service_status(
            app,
            "ner",
            ready=True,
            detail="initialized",
            extra={"model": ner.model_name},
        )
        logger.info("NER service ready: %s", ner.model_name)
    except Exception as e:
        app.state.ner_service = None
        _set_service_status(
            app,
            "ner",
            ready=False,
            detail="initialization failed",
            extra={"error": type(e).__name__},
        )
        logger.exception("NER service init failed (disabled): %s", e)


def init_redis_client(app: FastAPI) -> None:
    if _should_skip_service_init():
        app.state.redis_client = None
        _set_service_status(
            app,
            "redis",
            ready=False,
            detail="skipped in test runtime",
        )
        logger.info("Redis client skipped in test runtime (use dependency overrides).")
        return

    redis_needed = bool(settings.REDIS_URL) and (
        settings.ENABLE_CACHE or settings.ENABLE_RATE_LIMITING
    )

    if not redis_needed:
        app.state.redis_client = None
        _set_service_status(
            app,
            "redis",
            ready=False,
            detail="disabled by config",
        )
        logger.info("Redis client disabled by config.")
        return

    try:
        client = create_redis_client(settings.REDIS_URL)
        client.ping()
        app.state.redis_client = client
        _set_service_status(
            app,
            "redis",
            ready=True,
            detail="connected",
            extra={"url": settings.REDIS_URL},
        )
        logger.info("Redis client ready: %s", settings.REDIS_URL)
    except Exception as e:
        app.state.redis_client = None
        _set_service_status(
            app,
            "redis",
            ready=False,
            detail="connection failed",
            extra={"error": type(e).__name__},
        )
        logger.exception("Redis client init failed (disabled): %s", e)


def init_cache(app: FastAPI) -> None:
    if _should_skip_service_init():
        app.state.cache = None
        _set_service_status(
            app,
            "cache",
            ready=False,
            detail="skipped in test runtime",
        )
        logger.info("Cache skipped in test runtime (use dependency overrides).")
        return

    if not settings.ENABLE_CACHE:
        app.state.cache = None
        _set_service_status(
            app,
            "cache",
            ready=False,
            detail="disabled by config",
        )
        logger.info("Cache disabled by config.")
        return

    client = getattr(app.state, "redis_client", None)
    if client is None:
        app.state.cache = None
        _set_service_status(
            app,
            "cache",
            ready=False,
            detail="redis unavailable",
        )
        logger.info("Cache unavailable because Redis client is unavailable.")
        return

    try:
        app.state.cache = RedisCache(client)
        _set_service_status(
            app,
            "cache",
            ready=True,
            detail="initialized",
        )
        logger.info("Redis cache ready: %s", settings.REDIS_URL)
    except Exception as e:
        app.state.cache = None
        _set_service_status(
            app,
            "cache",
            ready=False,
            detail="initialization failed",
            extra={"error": type(e).__name__},
        )
        logger.exception("Redis cache init failed (disabled): %s", e)


def init_app_services(app: FastAPI) -> None:
    app.state.service_statuses = {}
    init_embedding_service(app)
    init_qa_service(app)
    init_ner_service(app)
    init_redis_client(app)
    init_cache(app)
