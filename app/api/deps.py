from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Path, Request
from sqlalchemy.orm import Session

from app.core.errors import InvalidInput, ServiceUnavailable
from app.core.identifiers import parse_document_public_id
from app.core.security.session import generate_session_id
from app.db.models import Document
from app.db.session import get_db
from app.repositories.documents import get_document_for_session
from app.services.interfaces import (
    CachePort,
    EmbeddingServicePort,
    NerServicePort,
    QaServicePort,
)


def get_session_id(request: Request) -> str:
    session_id = getattr(request.state, "session_id", None)
    if session_id:
        return session_id

    session_id = generate_session_id()
    request.state.session_id = session_id
    return session_id


def get_document_id(doc_id: str = Path(...)) -> str:
    try:
        parse_document_public_id(doc_id)
    except ValueError as exc:
        raise InvalidInput("Invalid doc_id format.") from exc
    return doc_id.lower()


def get_owned_document(
    doc_id: Annotated[str, Depends(get_document_id)],
    db: Session = Depends(get_db),
    session_id: str = Depends(get_session_id),
) -> Document:
    parsed_doc_id = parse_document_public_id(doc_id)
    return get_document_for_session(db, doc_id=parsed_doc_id, session_id=session_id)


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


DbSession = Annotated[Session, Depends(get_db)]
SessionId = Annotated[str, Depends(get_session_id)]
OwnedDocument = Annotated[Document, Depends(get_owned_document)]
EmbeddingSvc = Annotated[EmbeddingServicePort, Depends(get_embedding_service)]
QaSvc = Annotated[QaServicePort, Depends(get_qa_service)]
OptNerSvc = Annotated[NerServicePort | None, Depends(get_optional_ner_service)]
OptCache = Annotated[CachePort | None, Depends(get_optional_cache)]
