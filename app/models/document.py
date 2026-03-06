from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class DocumentListItem(BaseModel):
    doc_id: str
    filename: str
    content_type: str | None = None
    size_bytes: int | None = None
    status: str
    created_at: datetime
    indexed_at: datetime | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
    count: int
