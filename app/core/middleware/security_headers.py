from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings

# Middleware that adds security-realted HTTP headers
# to every response!
# request -> app -> response -> add security headers -> send to client

# OWASP top 10:
#   - MIME sniffing
#   - clickjacking
#   - referrer leakage
#   - unsafe content loading


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        # These are messages to BROWSER!!!

        # Do not try to guess the content type (message to browser)
        # Browser uses only declared Content-Type!
        response.headers.setdefault("X-Content-Type-Options", "nosniff")

        # Do not allow this page to be embedded inside a frame
        response.headers.setdefault("X-Frame-Options", "DENY")

        # Don't send referrer info when navigating away
        response.headers.setdefault("Referrer-Policy", "no-referrer")

        # Page is not allowed to use these features.
        response.headers.setdefault(
            "Permissions-Policy",
            "geolocation=(), microphone=(), camera=()",
        )

        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )

        # Tells the browser for how long to use HTTPS
        if settings.APP_ENV == "prod" and request.url.scheme == "https":
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )

        return response
