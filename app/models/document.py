from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class DocumentArtifacts(BaseModel):
    has_metadata: bool
    has_original: bool
    has_text: bool
    has_chunks: bool
    has_embeddings: bool
    has_index: bool


class DocumentListItem(BaseModel):
    doc_id: str
    filename: str
    content_type: str | None = None
    size_bytes: int | None = None
    status: str
    status_detail: str | None = None
    ready_to_ask: bool = False
    created_at: datetime
    indexed_at: datetime | None = None
    pages: int | None = None
    chunks: int | None = None
    owner_type: Literal["user", "session"]
    owner_id: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
    count: int


class DocumentDetailResponse(BaseModel):
    doc_id: str
    filename: str
    content_type: str | None = None
    size_bytes: int | None = None
    status: str
    status_detail: str | None = None
    ready_to_ask: bool = False
    created_at: datetime
    indexed_at: datetime | None = None
    owner_type: Literal["user", "session"]
    owner_id: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    artifacts: DocumentArtifacts


class DocumentDeleteResponse(BaseModel):
    doc_id: str
    status: Literal["deleted"] = "deleted"
