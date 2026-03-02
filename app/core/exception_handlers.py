from __future__ import annotations

import logging
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.errors import ApiError
from app.core.request_context import get_request_id

logger = logging.getLogger(__name__)


def _error_response(
    *,
    status_code: int,
    error_code: str,
    message: str,
    details: Any | None = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "error_code": error_code,
        "message": message,
        "request_id": get_request_id(),
    }
    if settings.APP_ENV != "prod" and details is not None:
        payload["details"] = details
    return JSONResponse(status_code=status_code, content=payload)


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    # Maps my implementation of ApiError into standard HTTPException
    if isinstance(exc, ApiError):
        return _error_response(
            status_code=exc.status_code,
            error_code=exc.error_code,
            message=str(exc.detail),
            details=getattr(exc, "details", None),
        )

    return _error_response(
        status_code=exc.status_code,
        error_code=f"http_{exc.status_code}",
        message=str(exc.detail),
        details=None,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # 422 is FastAPI default for validation, but assignment expects 400 for invalid input.
    return _error_response(
        status_code=400,
        error_code="invalid_input",
        message="Invalid request",
        details=exc.errors(),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Never leak stack traces to clients!!!
    # I still log server-side!!
    logger.exception(
        "Unhandled error",
        extra={"path": request.url.path, "request_id": get_request_id()},
    )
    return _error_response(
        status_code=500,
        error_code="internal_server_error",
        message="Internal Server Error",
        details=None,
    )
