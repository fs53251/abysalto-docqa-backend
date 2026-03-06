from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import and_, delete, select, update
from sqlalchemy.orm import Session

from app.core.errors import http_error
from app.core.identity import RequestIdentity
from app.core.identifiers import generate_document_id, parse_document_public_id
from app.db.models import Document


def _validate_owner_identity(
    *,
    owner_user_id: uuid.UUID | None,
    owner_session_id: str | None,
) -> None:
    if owner_user_id is None and not owner_session_id:
        raise ValueError("DOCUMENT_OWNER_REQUIRED")
    if owner_user_id is not None and owner_session_id is not None:
        raise ValueError("DOCUMENT_OWNER_AMBIGUOUS")


def _ownership_filter(identity: RequestIdentity):
    if identity.kind == "user":
        return Document.owner_user_id == identity.user_id

    return and_(
        Document.owner_user_id.is_(None),
        Document.owner_session_id == identity.session_id,
    )


def create_document(
    db: Session,
    *,
    filename: str,
    content_type: str | None = None,
    size_bytes: int | None = None,
    sha256: str | None = None,
    stored_path: str | None = None,
    doc_id: uuid.UUID | None = None,
    owner_user_id: uuid.UUID | None = None,
    owner_session_id: str | None = None,
    status: str = "uploaded",
) -> Document:
    _validate_owner_identity(
        owner_user_id=owner_user_id,
        owner_session_id=owner_session_id,
    )

    document_id = doc_id or generate_document_id()
    if document_id.version != 4:
        raise ValueError("DOCUMENT_ID_MUST_BE_UUID4")

    doc = Document(
        id=document_id,
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        sha256=sha256,
        stored_path=stored_path,
        owner_user_id=owner_user_id,
        owner_session_id=owner_session_id,
        status=status,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_document(db: Session, *, doc_id: uuid.UUID) -> Optional[Document]:
    stmt = select(Document).where(Document.id == doc_id)
    return db.execute(stmt).scalars().first()


def get_document_by_public_id(db: Session, *, doc_id: str) -> Optional[Document]:
    try:
        parsed = parse_document_public_id(doc_id)
    except ValueError:
        return None
    return get_document(db, doc_id=parsed)


def get_document_for_identity(
    db: Session,
    *,
    doc_id: uuid.UUID,
    identity: RequestIdentity,
) -> Document:
    stmt = select(Document).where(
        Document.id == doc_id,
        _ownership_filter(identity),
    )
    doc = db.execute(stmt).scalars().first()
    if doc is None:
        raise http_error(404, "doc_not_found", "Document not found")
    return doc


def list_documents_for_identity(
    db: Session,
    *,
    identity: RequestIdentity,
) -> list[Document]:
    stmt = (
        select(Document)
        .where(_ownership_filter(identity))
        .order_by(Document.created_at.desc())
    )
    return list(db.execute(stmt).scalars().all())


def assert_documents_owned_by_identity(
    db: Session,
    *,
    doc_ids: Iterable[uuid.UUID],
    identity: RequestIdentity,
) -> list[Document]:
    ordered_ids: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()

    for doc_id in doc_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ordered_ids.append(doc_id)

    if not ordered_ids:
        return []

    stmt = select(Document).where(
        Document.id.in_(ordered_ids),
        _ownership_filter(identity),
    )
    owned_documents = list(db.execute(stmt).scalars().all())
    owned_by_id = {document.id: document for document in owned_documents}

    if len(owned_by_id) != len(ordered_ids):
        raise http_error(
            403,
            "doc_forbidden",
            "One or more requested documents are not accessible.",
        )

    return [owned_by_id[doc_id] for doc_id in ordered_ids]


def get_document_for_session(
    db: Session,
    *,
    doc_id: uuid.UUID,
    session_id: str,
) -> Document:
    return get_document_for_identity(
        db,
        doc_id=doc_id,
        identity=RequestIdentity.for_session(session_id),
    )


def list_documents_for_user(db: Session, *, user_id: uuid.UUID) -> list[Document]:
    return list_documents_for_identity(
        db,
        identity=RequestIdentity.for_user(user_id),
    )


def list_documents_for_session(db: Session, *, session_id: str) -> list[Document]:
    return list_documents_for_identity(
        db,
        identity=RequestIdentity.for_session(session_id),
    )


def claim_session_documents_for_user(
    db: Session,
    *,
    session_id: str,
    user_id: uuid.UUID,
) -> int:
    stmt = (
        update(Document)
        .where(
            Document.owner_session_id == session_id,
            Document.owner_user_id.is_(None),
        )
        .values(owner_user_id=user_id, owner_session_id=None)
    )
    result = db.execute(stmt.execution_options(synchronize_session=False))
    db.commit()
    return int(result.rowcount or 0)


def mark_document_processing(db: Session, *, document: Document) -> Document:
    document.status = "processing"
    document.indexed_at = None
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def mark_document_failed(db: Session, *, document: Document) -> Document:
    document.status = "failed"
    document.indexed_at = None
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def mark_document_indexed(db: Session, *, document: Document) -> Document:
    document.status = "indexed"
    if document.indexed_at is None:
        document.indexed_at = datetime.now(timezone.utc)
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def delete_document_record(db: Session, *, document: Document) -> None:
    db.delete(document)
    db.commit()


def delete_expired_session_documents(
    db: Session, *, ttl_days: int, now: datetime | None = None
) -> int:
    if now is None:
        now = datetime.now(timezone.utc)

    cutoff = now - timedelta(days=ttl_days)
    cutoff_cmp = cutoff.replace(tzinfo=None) if cutoff.tzinfo is not None else cutoff

    stmt = delete(Document).where(
        and_(
            Document.owner_session_id.is_not(None),
            Document.owner_user_id.is_(None),
            Document.created_at < cutoff_cmp,
        )
    )

    result = db.execute(stmt.execution_options(synchronize_session=False))
    db.commit()
    return int(result.rowcount or 0)
