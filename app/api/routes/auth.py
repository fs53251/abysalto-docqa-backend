from __future__ import annotations

from fastapi import APIRouter, status
from sqlalchemy.exc import IntegrityError

from app.api.deps import CurrentIdentity, CurrentUser, DbSession, SessionId
from app.core.config import settings
from app.core.errors import Conflict, http_error
from app.core.security.jwt import create_access_token
from app.core.security.passwords import hash_password, verify_password
from app.models.auth import (
    AuthUserResponse,
    IdentityResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)
from app.repositories.users import create_user, get_user_by_email

router = APIRouter(prefix="/auth", tags=["auth"])


def _token_response_for_user(user_id: str) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(sub=user_id),
        token_type="bearer",
        expires_in=settings.JWT_EXP_MIN * 60,
    )


@router.post(
    "/register", response_model=AuthUserResponse, status_code=status.HTTP_201_CREATED
)
def register(
    body: RegisterRequest,
    db: DbSession,
    session_id: SessionId,
) -> AuthUserResponse:
    del session_id  # reserved for later session -> user claim flow

    if get_user_by_email(db, email=str(body.email)) is not None:
        raise Conflict("A user with this email already exists.")

    try:
        user = create_user(
            db,
            email=str(body.email),
            password_hash=hash_password(body.password),
            is_active=True,
        )
    except IntegrityError as exc:
        raise Conflict("A user with this email already exists.") from exc

    return AuthUserResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse)
def login(
    body: LoginRequest,
    db: DbSession,
    session_id: SessionId,
) -> TokenResponse:
    del session_id  # reserved for later session -> user claim flow

    user = get_user_by_email(db, email=str(body.email))
    if user is None or not verify_password(body.password, user.password_hash):
        raise http_error(
            401,
            "invalid_credentials",
            "Invalid email or password.",
        )

    if not user.is_active:
        raise http_error(
            401,
            "inactive_user",
            "User account is inactive.",
        )

    return _token_response_for_user(str(user.id))


@router.get("/me", response_model=AuthUserResponse)
def me(current_user: CurrentUser) -> AuthUserResponse:
    return AuthUserResponse.model_validate(current_user)


@router.get("/identity", response_model=IdentityResponse)
def auth_identiry(identity: CurrentIdentity) -> IdentityResponse:
    return IdentityResponse(
        kind=identity.kind,
        user_id=identity.user_id,
        session_id=identity.session_id,
        log_identity=identity.log_identity,
    )
