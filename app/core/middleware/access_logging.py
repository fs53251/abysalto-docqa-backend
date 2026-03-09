from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Middleware wraps every HTTP request
# Logs when it starts and ends
# Measures latency, attaches request identity and request ID

# Creating logger, mentioned in core config
logger = logging.getLogger("app.access")


def _request_identity(request: Request) -> str:
    identity = getattr(request.state, "identity", None)
    if identity is not None:
        log_identity = getattr(identity, "log_identity", None)
        if isinstance(log_identity, str) and log_identity:
            return log_identity

    user_id = getattr(request.state, "user_id", None)
    if user_id is not None:
        return f"user:{user_id}"

    session_id = getattr(request.state, "session_id", None)
    if session_id:
        return f"sess:{session_id}"

    return "-"


# request -> middleware -> route handler -> middleware -> response
class AccessLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        started_at = time.perf_counter()

        initial_request_id = getattr(
            request.state, "request_id", None
        ) or request.headers.get("X-Request-Id", "-")

        logger.info(
            "request start",
            extra={
                "event": "request.start",
                "request_id": initial_request_id,
                "identity": _request_identity(request),
                "path": request.url.path,
                "method": request.method,
            },
        )

        try:
            response = await call_next(request)
        except Exception:
            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            logger.exception(
                "request end with unhandled error",
                extra={
                    "event": "request.end",
                    "request_id": getattr(
                        request.state, "request_id", initial_request_id
                    ),
                    "identity": _request_identity(request),
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": 500,
                    "latency_ms": latency_ms,
                },
            )
            raise

        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.info(
            "request end",
            extra={
                "event": "request.end",
                "request_id": getattr(request.state, "request_id", initial_request_id),
                "identity": _request_identity(request),
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        return response
