from __future__ import annotations

import base64
import hashlib
import hmac
import uuid
from typing import Final

from app.core.config import settings

COOKIE_VERSION: Final[str] = "v1"


def _session_secret() -> bytes:
    secret = settings.SESSION_COOKIE_SECRET or settings.JWT_SECRET
    if not secret:
        raise RuntimeError("SESSION_COOKIE_SECRET_OR_JWT_SECRET_REQUIRED")
    return secret.encode("utf-8")


def generate_session_id() -> str:
    """Create a canonical UUIDv4 session id."""
    return str(uuid.uuid4())


def is_valid_session_id(value: str) -> bool:
    try:
        parsed = uuid.UUID(value)
    except (TypeError, ValueError, AttributeError):
        return False
    return parsed.version == 4 and str(parsed) == value.lower()


def _sign(session_id: str) -> str:
    mac = hmac.new(
        _session_secret(),
        msg=f"{COOKIE_VERSION}:{session_id}".encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(mac).rstrip(b"=").decode("ascii")


def dump_session_cookie(session_id: str) -> str:
    if not is_valid_session_id(session_id):
        raise ValueError("INVALID_SESSION_ID")
    return f"{COOKIE_VERSION}.{session_id}.{_sign(session_id)}"


def load_session_cookie(cookie_value: str | None) -> str | None:
    if not cookie_value:
        return None

    try:
        version, session_id, signature = cookie_value.split(".", 2)
    except ValueError:
        return None

    if version != COOKIE_VERSION:
        return None
    if not is_valid_session_id(session_id):
        return None
    if not hmac.compare_digest(signature, _sign(session_id)):
        return None

    return session_id
