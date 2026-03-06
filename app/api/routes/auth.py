from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.exc import IntegrityError

from app.api.deps import CurrentIdentity, CurrentUser, DbSession, SessionId
from app.core.config import settings
from app.core.errors import Conflict, http_error
from app.core.log_safety import hash_text
from app.core.security.jwt import create_access_token
from app.core.security.passwords import hash_password, verify_password
from app.models.auth import (
    AuthUserResponse,
    IdentityResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)
from app.repositories.documents import claim_session_documents_for_user
from app.repositories.users import create_user, get_user_by_email
from app.services.rate_limit import login_rate_limit_key, rate_limit

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)

login_rate_limit = rate_limit(
    limit=lambda: settings.LOGIN_RATE_LIMIT_PER_MIN,
    window_seconds=lambda: settings.RATE_LIMIT_WINDOW_SECONDS,
    key_fn=login_rate_limit_key("login"),
)


def _token_response_for_user(user_id: str) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(sub=user_id),
        token_type="bearer",
        expires_in=settings.JWT_EXP_MIN * 60,
    )


@router.post(
    "/register",
    response_model=AuthUserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    body: RegisterRequest,
    db: DbSession,
    session_id: SessionId,
) -> AuthUserResponse:
    email_hash = hash_text(str(body.email))

    if get_user_by_email(db, email=str(body.email)) is not None:
        logger.warning(
            "auth register failed",
            extra={
                "event": "auth.register.failed",
                "email_hash": email_hash,
                "outcome": "duplicate_email",
                "identity": f"sess:{session_id}",
            },
        )
        raise Conflict("A user with this email already exists.")

    try:
        user = create_user(
            db,
            email=str(body.email),
            password_hash=hash_password(body.password),
            is_active=True,
        )
    except IntegrityError as exc:
        logger.warning(
            "auth register failed",
            extra={
                "event": "auth.register.failed",
                "email_hash": email_hash,
                "outcome": "duplicate_email",
                "identity": f"sess:{session_id}",
            },
        )
        raise Conflict("A user with this email already exists.") from exc

    claimed = claim_session_documents_for_user(
        db, session_id=session_id, user_id=user.id
    )

    logger.info(
        "auth register succeeded",
        extra={
            "event": "auth.register.succeeded",
            "email_hash": email_hash,
            "outcome": "created",
            "identity": f"user:{user.id}",
            "doc_ids_count": claimed,
        },
    )
    return AuthUserResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse)
def login(
    body: LoginRequest,
    db: DbSession,
    session_id: SessionId,
    request: Request,
    _rate_limit: None = Depends(login_rate_limit),
) -> TokenResponse:
    del request
    del _rate_limit

    email_hash = hash_text(str(body.email))
    user = get_user_by_email(db, email=str(body.email))

    if user is None or not verify_password(body.password, user.password_hash):
        logger.warning(
            "auth login failed",
            extra={
                "event": "auth.login.failed",
                "email_hash": email_hash,
                "outcome": "invalid_credentials",
                "identity": f"sess:{session_id}",
            },
        )
        raise http_error(
            401,
            "invalid_credentials",
            "Invalid email or password.",
        )

    if not user.is_active:
        logger.warning(
            "auth login failed",
            extra={
                "event": "auth.login.failed",
                "email_hash": email_hash,
                "outcome": "inactive_user",
                "identity": f"user:{user.id}",
            },
        )
        raise http_error(
            401,
            "inactive_user",
            "User account is inactive.",
        )

    claimed = claim_session_documents_for_user(
        db, session_id=session_id, user_id=user.id
    )

    logger.info(
        "auth login succeeded",
        extra={
            "event": "auth.login.succeeded",
            "email_hash": email_hash,
            "outcome": "ok",
            "identity": f"user:{user.id}",
            "doc_ids_count": claimed,
        },
    )
    return _token_response_for_user(str(user.id))


@router.get("/me", response_model=AuthUserResponse)
def me(current_user: CurrentUser) -> AuthUserResponse:
    return AuthUserResponse.model_validate(current_user)


@router.get("/identity", response_model=IdentityResponse)
def auth_identity(identity: CurrentIdentity) -> IdentityResponse:
    return IdentityResponse(
        kind=identity.kind,
        user_id=identity.user_id,
        session_id=identity.session_id,
        log_identity=identity.log_identity,
    )
