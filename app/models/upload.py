from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class UploadItemResponse(BaseModel):
    filename: str
    status: str
    status_detail: str | None = None
    ready_to_ask: bool = False

    doc_id: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    owner_type: Literal["user", "session"] | None = None

    error_detail: str | None = None


class UploadResponse(BaseModel):
    documents: list[UploadItemResponse]
    has_errors: bool
