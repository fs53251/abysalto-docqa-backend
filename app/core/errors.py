from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException


@dataclass(frozen=True)
class ErrorPayload:
    error_code: str
    message: str
    details: Any | None = None


class ApiError(HTTPException):
    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
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
    details: Any | None = None,
) -> ApiError:
    return ApiError(
        status_code=status_code,
        error_code=error_code,
        message=message,
        details=details,
    )