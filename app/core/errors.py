from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

# This file creates a clean bridge between
# internal business errors and 
# HTTP responses in a FastAPI application.


#########################################
##      DOMAIN ERROR HANDLING          ##
#########################################
class DomainError(Exception):
    """
    Base class for non-HTTP,
    business/domain level errors.
    For internal error handling.
    """

    error_code: str = "domain_error"
    status_code: int = 400

    def __init__(self, message: str, *, details: Any | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class InvalidInput(DomainError):
    error_code = "invalid_input"
    status_code = 400


class NotFound(DomainError):
    error_code = "not_found"
    status_code = 404


class Conflict(DomainError):
    error_code = "conflict"
    status_code = 409


class PayloadTooLarge(DomainError):
    error_code = "payload_too_large"
    status_code = 413


class UnsupportedMediaType(DomainError):
    error_code = "unsupported_media_type"
    status_code = 415


class ServiceUnavailable(DomainError):
    error_code = "service_unavailable"
    status_code = 503


class InternalError(DomainError):
    error_code = "internal_error"
    status_code = 500


class ExternalDependencyMissing(ServiceUnavailable):
    """
    Raised when an optional dependency is not installed.
    """

    error_code = "dependency_missing"

    def __init__(self, dependency: str, *, details: Any | None = None) -> None:
        super().__init__(
            f"Required dependency is missing: {dependency}", details=details
        )
        self.dependency = dependency


#########################################
##         API ERROR HANDLING          ##
#########################################
@dataclass(frozen=True)
class ErrorPayload:
    """
    Standard error payload returned by the API.
    """

    error_code: str
    message: str
    details: Any | None = None

class ApiError(HTTPException):
    """
    HTTPException enriched with a stable error_code
      and optional details.
    """

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        *,
        details: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message, headers=headers)
        self.error_code = error_code
        self.details = details


def http_error(
    status_code: int,
    error_code: str,
    message: str,
    *,
    details: Any | None = None,
) -> ApiError:
    return ApiError(
        status_code=status_code,
        error_code=error_code,
        message=message,
        details=details,
    )


def from_domain_error(err: DomainError) -> ApiError:
    return http_error(
        status_code=err.status_code,
        error_code=err.error_code,
        message=err.message,
        details=err.details,
    )
