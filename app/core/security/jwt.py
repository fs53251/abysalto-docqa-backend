from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

from app.core.config import settings

# Stateless auth tokens


class TokenError(ValueError):
    """
    Token related failures
    """

    pass


class TokenExpiredError(TokenError):
    """
    Used when the token is valid but expired
    """

    pass


class TokenInvalidError(TokenError):
    """
    Used when the token is missing or invalid
    """

    pass


# JWT std:
#   - sub: subject (user id)...UUID user_id
#   - exp: expiration time
#   - iat: issued at
#   - iss: issuer (not used here)
#   - aud: audience (not used here)


def create_access_token(
    *,
    sub: str | UUID,
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    expires_at = now + (expires_delta or timedelta(minutes=settings.JWT_EXP_MIN))

    payload: dict[str, Any] = {
        "sub": str(sub),
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "type": "access",
    }

    if extra_claims:
        payload.update(extra_claims)

    # signs the payload and returns JWT string
    # alg - HS256 - HMAC-SHA256
    #   - the same secret for signing token
    #   - the same secret for verifying token
    # JWT: header.payload.signature
    return jwt.encode(
        payload,
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_token(token: str) -> dict[str, Any]:
    if not token:
        raise TokenInvalidError("TOKEN_REQUIRED")

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except ExpiredSignatureError as exc:
        raise TokenExpiredError("TOKEN_EXPIRED") from exc
    except InvalidTokenError as exc:
        raise TokenInvalidError("TOKEN_INVALID") from exc

    sub = payload.get("sub")

    if not isinstance(sub, str) or not sub.strip():
        raise TokenInvalidError("TOKEN_SUB_INVALID")

    try:
        UUID(sub)
    except ValueError as exc:
        raise TokenInvalidError("TOKEN_SUB_INVALID") from exc

    if payload.get("type") not in {None, "access"}:
        raise TokenInvalidError("TOKEN_TYPE_INVALID")

    return payload
