from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Path, Request
from fastapi.security.utils import get_authorization_scheme_param
from sqlalchemy.orm import Session

from app.core.errors import ApiError, InvalidInput, ServiceUnavailable
from app.core.identifiers import parse_document_public_id
from app.core.identity import RequestIdentity
from app.core.request_context import set_identity
from app.core.security.jwt import TokenExpiredError, TokenInvalidError, decode_token
from app.core.security.session import generate_session_id
from app.db.models import Document, User
from app.db.session import get_db
from app.repositories.documents import get_document_for_identity
from app.repositories.users import get_user
from app.services.interfaces import (
    CachePort,
    EmbeddingServicePort,
    NerServicePort,
    QaServicePort,
)


def _auth_error(error_code: str, message: str) -> ApiError:
    return ApiError(
        status_code=401,
        error_code=error_code,
        message=message,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _extract_bearer_token(request: Request) -> str | None:
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None

    scheme, token = get_authorization_scheme_param(authorization)
    if not scheme and not token:
        return None
    if scheme.lower() != "bearer" or not token:
        raise _auth_error("auth_invalid_header", "Invalid authorization header.")

    return token


def get_session_id(request: Request) -> str:
    session_id = getattr(request.state, "session_id", None)
    if session_id:
        return session_id

    session_id = generate_session_id()
    request.state.session_id = session_id
    return session_id


def get_optional_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User | None:
    token = _extract_bearer_token(request)
    if token is None:
        request.state.user_id = None
        return None

    try:
        payload = decode_token(token)
    except TokenExpiredError as exc:
        raise _auth_error("auth_token_expired", "Access token expired.") from exc
    except TokenInvalidError as exc:
        raise _auth_error("auth_invalid_token", "Invalid access token.") from exc

    try:
        user_id = uuid.UUID(str(payload["sub"]))
    except (TypeError, ValueError, KeyError) as exc:
        raise _auth_error("auth_invalid_token", "Invalid access token.") from exc

    user = get_user(db, user_id=user_id)
    if user is None or not user.is_active:
        raise _auth_error(
            "auth_invalid_user",
            "Authenticated user is missing or inactive.",
        )

    request.state.user_id = user.id
    identity = RequestIdentity.for_user(user.id)
    request.state.identity = identity
    set_identity(identity.log_identity)
    return user


def get_current_user(
    request: Request,
    current_user: User | None = Depends(get_optional_current_user),
) -> User:
    if current_user is None:
        request.state.user_id = None
        raise _auth_error("auth_required", "Authentication required.")
    return current_user


def get_identity(
    request: Request,
    session_id: str = Depends(get_session_id),
    current_user: User | None = Depends(get_optional_current_user),
) -> RequestIdentity:
    if current_user is not None:
        identity = RequestIdentity.for_user(current_user.id)
    else:
        request.state.user_id = None
        identity = RequestIdentity.for_session(session_id)

    request.state.identity = identity
    set_identity(identity.log_identity)
    return identity


def get_document_id(doc_id: str = Path(...)) -> str:
    try:
        parse_document_public_id(doc_id)
    except ValueError as exc:
        raise InvalidInput("Invalid doc_id format.") from exc
    return doc_id.lower()


def get_owned_document(
    doc_id: Annotated[str, Depends(get_document_id)],
    db: Session = Depends(get_db),
    identity: RequestIdentity = Depends(get_identity),
) -> Document:
    parsed_doc_id = parse_document_public_id(doc_id)
    return get_document_for_identity(db, doc_id=parsed_doc_id, identity=identity)


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

CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalCurrentUser = Annotated[User | None, Depends(get_optional_current_user)]
CurrentIdentity = Annotated[RequestIdentity, Depends(get_identity)]
