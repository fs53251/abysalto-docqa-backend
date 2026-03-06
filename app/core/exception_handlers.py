from __future__ import annotations

import logging
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.errors import ApiError, DomainError, from_domain_error
from app.core.request_context import get_request_id

logger = logging.getLogger(__name__)


def _sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(item) for item in value]

    if isinstance(value, BaseException):
        return str(value)

    return str(value)


def _error_response(
    *,
    status_code: int,
    error_code: str,
    message: str,
    details: Any | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "error_code": error_code,
        "message": message,
        "request_id": get_request_id(),
    }
    if settings.APP_ENV != "prod" and details is not None:
        payload["details"] = _sanitize_for_json(details)

    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers=headers,
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    if isinstance(exc, ApiError):
        return _error_response(
            status_code=exc.status_code,
            error_code=exc.error_code,
            message=str(exc.detail),
            details=getattr(exc, "details", None),
            headers=getattr(exc, "headers", None),
        )

    return _error_response(
        status_code=exc.status_code,
        error_code=f"http_{exc.status_code}",
        message=str(exc.detail),
        details=None,
        headers=getattr(exc, "headers", None),
    )


async def domain_exception_handler(
    request: Request,
    exc: DomainError,
) -> JSONResponse:
    api_exc = from_domain_error(exc)
    return _error_response(
        status_code=api_exc.status_code,
        error_code=api_exc.error_code,
        message=str(api_exc.detail),
        details=getattr(api_exc, "details", None),
        headers=getattr(api_exc, "headers", None),
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return _error_response(
        status_code=400,
        error_code="invalid_input",
        message="Invalid request",
        details=exc.errors(),
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception(
        "Unhandled error",
        extra={
            "path": request.url.path,
            "method": request.method,
            "request_id": get_request_id(),
        },
    )
    return _error_response(
        status_code=500,
        error_code="internal_server_error",
        message="Internal Server Error",
        details=None,
    )
