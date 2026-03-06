from __future__ import annotations

import logging

from fastapi import FastAPI

from app.core.config import settings
from app.services.indexing.embedding_service import default_embedding_service
from app.services.ner.ner_service import default_ner_service
from app.services.qa.qa_service import default_qa_service

logger = logging.getLogger(__name__)


def _disabled_in_test(name: str) -> bool:
    return settings.APP_ENV == "test"


def init_embedding_service(app: FastAPI) -> None:
    if _disabled_in_test("embedding"):
        app.state.embedding_service = None
        logger.info("Embedding service skipped in test env (use dependency overrides).")
        return

    try:
        svc = default_embedding_service()
        svc.load()
        app.state.embedding_service = svc
        logger.info("Embedding service ready: %s", svc.cfg.model_name)
    except Exception as e:
        app.state.embedding_service = None
        logger.exception("Embedding service init failed (disabled): %s", e)


def init_qa_service(app: FastAPI) -> None:
    if _disabled_in_test("qa"):
        app.state.qa_service = None
        logger.info("QA service skipped in test env (use dependency overrides).")
        return

    try:
        qa = default_qa_service()
        qa.load()
        app.state.qa_service = qa
        logger.info("QA service ready: %s", qa.model_name)
    except Exception as e:
        app.state.qa_service = None
        logger.exception("QA service init failed (disabled): %s", e)


def init_ner_service(app: FastAPI) -> None:
    if _disabled_in_test("ner"):
        app.state.ner_service = None
        logger.info("NER service skipped in test env (use dependency overrides).")
        return

    try:
        ner = default_ner_service()
        ner.load()
        app.state.ner_service = ner
        logger.info("NER service ready: %s", ner.model_name)
    except Exception as e:
        app.state.ner_service = None
        logger.exception("NER service init failed (disabled): %s", e)


def init_cache(app: FastAPI) -> None:
    # Cache u testovima tipično overrideamo fake-om; defaultno isključeno
    if settings.APP_ENV == "test":
        app.state.cache = None
        logger.info("Cache skipped in test env (use dependency overrides).")
        return

    if not settings.ENABLE_CACHE or not settings.REDIS_URL:
        app.state.cache = None
        logger.info("Cache disabled by config.")
        return

    try:
        from app.services.cache.redis_cache import RedisCache

        client = RedisCache.connect(settings.REDIS_URL)
        client.ping()
        app.state.cache = RedisCache(client)
        logger.info("Redis cache ready: %s", settings.REDIS_URL)
    except Exception as e:
        app.state.cache = None
        logger.exception("Redis cache init failed (disabled): %s", e)


def init_app_services(app: FastAPI) -> None:
    """
    Single entry point for all service initialization.
    """
    init_embedding_service(app)
    init_qa_service(app)
    init_ner_service(app)
    init_cache(app)
