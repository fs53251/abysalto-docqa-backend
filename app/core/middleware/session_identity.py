from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings
from app.core.security.session import (
    dump_session_cookie,
    generate_session_id,
    load_session_cookie,
)


class SessionIdentityMiddleware(BaseHTTPMiddleware):
    """
    Ensure every anonymous request has a stable signed session cookie.

    - On missing or invalid cookie, generate a fresh UUIDv4 session id.
    - Expose the raw session id on request.state.session_id.
    - Set a signed cookie only when a new session had to be issued.
    """

    async def dispatch(self, request: Request, call_next):
        raw_cookie = request.cookies.get(settings.SESSION_COOKIE_NAME)
        session_id = load_session_cookie(raw_cookie)
        must_set_cookie = False

        if session_id is None:
            session_id = generate_session_id()
            must_set_cookie = True

        request.state.session_id = session_id
        request.state.new_session_issued = must_set_cookie

        response: Response = await call_next(request)

        # 7 day cookie session
        # httponly - javascript in browser cannot read it
        # secure - HTTPS
        # Set-Cookie: docqa_session=<signed abc-123>; HttpOnly; Path=/; Max-Age=...
        if must_set_cookie:
            response.set_cookie(
                key=settings.SESSION_COOKIE_NAME,
                value=dump_session_cookie(session_id),
                max_age=settings.SESSION_COOKIE_MAX_AGE_SECONDS,
                httponly=True,
                samesite=settings.SESSION_COOKIE_SAMESITE,
                secure=settings.SESSION_COOKIE_SECURE,
                path="/",
            )

        return response
