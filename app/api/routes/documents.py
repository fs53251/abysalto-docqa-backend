from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import DbSession, SessionId
from app.core.identifiers import document_public_id
from app.models.document import DocumentListItem, DocumentListResponse
from app.repositories.documents import list_documents_for_session

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(db: DbSession, session_id: SessionId) -> DocumentListResponse:
    docs = list_documents_for_session(db, session_id=session_id)
    items = [
        DocumentListItem(
            doc_id=document_public_id(doc.id),
            filename=doc.filename,
            content_type=doc.content_type,
            size_bytes=doc.size_bytes,
            status=doc.status,
            created_at=doc.created_at,
            indexed_at=doc.indexed_at,
        )
        for doc in docs
    ]
    return DocumentListResponse(documents=items, count=len(items))
