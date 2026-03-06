from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.errors import ServiceUnavailable
from app.services.interfaces import (
    CachePort,
    EmbeddingServicePort,
    NerServicePort,
    QaServicePort,
)


def get_embedding_service(request: Request) -> EmbeddingServicePort:
    svc = getattr(request.app.state, "embedding_service", None)
    if svc is None:
        raise ServiceUnavailable("Embedding service unavailable (not initialized).")
    return svc


def get_qa_service(request: Request) -> QaServicePort:
    svc = getattr(request.app.state, "qa_service", None)
    if svc is None:
        raise ServiceUnavailable("QA service unavailable (not initialized).")
    return svc


def get_optional_ner_service(request: Request) -> NerServicePort | None:
    return getattr(request.app.state, "ner_service", None)


def get_optional_cache(request: Request) -> CachePort | None:
    return getattr(request.app.state, "cache", None)


EmbeddingSvc = Annotated[EmbeddingServicePort, Depends(get_embedding_service)]
QaSvc = Annotated[QaServicePort, Depends(get_qa_service)]
OptNerSvc = Annotated[NerServicePort | None, Depends(get_optional_ner_service)]
OptCache = Annotated[CachePort | None, Depends(get_optional_cache)]
