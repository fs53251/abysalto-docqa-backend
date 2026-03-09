from __future__ import annotations

import uuid

from itsdangerous import BadSignature, URLSafeSerializer

from app.core.config import settings

# Handle creating, signing and reading session cookies


def _serializer() -> URLSafeSerializer:
    """
    Signs and verifies cookie values.
    """
    return URLSafeSerializer(
        secret_key=settings.SESSION_COOKIE_SECRET,
        salt=f"{settings.APP_NAME}:session",
    )


def generate_session_id() -> str:
    return uuid.uuid4().hex


def dump_session_cookie(session_id: str) -> str:
    """
    Raw session ID -> signed cookie value
    """
    return _serializer().dumps({"sid": session_id})


def sign_session_cookie(session_id: str) -> str:
    return dump_session_cookie(session_id)


def load_session_cookie(cookie_value: str | None) -> str | None:
    if not cookie_value:
        return None

    try:
        payload = _serializer().loads(cookie_value)
    except BadSignature:
        return None

    session_id = payload.get("sid")
    if not isinstance(session_id, str) or not session_id.strip():
        return None

    return session_id.strip()
