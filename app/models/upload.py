from typing import Optional

from pydantic import BaseModel


class UploadItemResponse(BaseModel):
    filename: str
    status: str

    doc_id: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None

    error_detail: Optional[str] = None


class UploadResponse(BaseModel):
    documents: list[UploadItemResponse]
    has_errors: bool
