from __future__ import annotations

import logging

from fastapi import APIRouter

from app.api.deps import CurrentIdentity, DbSession, OwnedDocument
from app.core.identifiers import document_public_id
from app.models.document import (
    DocumentArtifacts,
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentListItem,
    DocumentListResponse,
)
from app.repositories.documents import (
    delete_document_record,
    list_documents_for_identity,
)
from app.services.documents.metadata import (
    build_document_artifact_state,
    delete_document_storage,
)

router = APIRouter(tags=["documents"])
logger = logging.getLogger(__name__)


def _owner_type(document) -> str:
    return "user" if document.owner_user_id is not None else "session"


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    db: DbSession,
    identity: CurrentIdentity,
) -> DocumentListResponse:
    docs = list_documents_for_identity(db, identity=identity)
    items: list[DocumentListItem] = []

    for doc in docs:
        public_doc_id = document_public_id(doc.id)
        artifacts = build_document_artifact_state(public_doc_id)

        items.append(
            DocumentListItem(
                doc_id=public_doc_id,
                filename=doc.filename,
                content_type=doc.content_type,
                size_bytes=doc.size_bytes,
                status=doc.status,
                created_at=doc.created_at,
                indexed_at=doc.indexed_at,
                pages=artifacts.page_count,
                chunks=artifacts.chunk_count,
            )
        )

    return DocumentListResponse(documents=items, count=len(items))


@router.get("/documents/{doc_id}", response_model=DocumentDetailResponse)
def get_document_detail(document: OwnedDocument) -> DocumentDetailResponse:
    public_doc_id = document_public_id(document.id)
    artifacts = build_document_artifact_state(public_doc_id)

    return DocumentDetailResponse(
        doc_id=public_doc_id,
        filename=document.filename,
        content_type=document.content_type,
        size_bytes=document.size_bytes,
        status=document.status,
        created_at=document.created_at,
        indexed_at=document.indexed_at,
        owner_type=_owner_type(document),
        page_count=artifacts.page_count,
        chunk_count=artifacts.chunk_count,
        artifacts=DocumentArtifacts(
            has_metadata=artifacts.has_metadata,
            has_original=artifacts.has_original,
            has_text=artifacts.has_text,
            has_chunks=artifacts.has_chunks,
            has_embeddings=artifacts.has_embeddings,
            has_index=artifacts.has_index,
        ),
    )


@router.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
def delete_document(
    document: OwnedDocument,
    db: DbSession,
) -> DocumentDeleteResponse:
    public_doc_id = document_public_id(document.id)
    owner_type = _owner_type(document)

    delete_document_storage(public_doc_id)
    delete_document_record(db, document=document)

    logger.info(
        "document deleted",
        extra={
            "event": "document.deleted",
            "doc_id": public_doc_id,
            "owner_type": owner_type,
        },
    )

    return DocumentDeleteResponse(doc_id=public_doc_id)
