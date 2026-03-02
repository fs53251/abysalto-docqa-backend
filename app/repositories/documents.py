from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import and_, delete, select
from sqlalchemy.orm import Session

from app.core.errors import http_error
from app.db.models import Document


def create_document(
    db: Session,
    *,
    filename: str,
    content_type: str | None = None,
    size_bytes: int | None = None,
    sha256: str | None = None,
    stored_path: str | None = None,
    owner_user_id: uuid.UUID | None = None,
    owner_session_id: str | None = None,
    status: str = "uploaded",
) -> Document:
    doc = Document(
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


def list_documents_for_user(db: Session, *, user_id: uuid.UUID) -> list[Document]:
    stmt = (
        select(Document)
        .where(Document.owner_user_id == user_id)
        .order_by(Document.created_at.desc())
    )
    return list(db.execute(stmt).scalars().all())


def list_documents_for_session(db: Session, *, session_id: str) -> list[Document]:
    stmt = (
        select(Document)
        .where(Document.owner_session_id == session_id)
        .order_by(Document.created_at.desc())
    )
    return list(db.execute(stmt).scalars().all())


def assert_document_owner(
    db: Session,
    *,
    doc_id: uuid.UUID,
    user_id: uuid.UUID | None = None,
    session_id: str | None = None,
) -> Document:
    """
    Ensures doc exists and belongs to the provided identity.
    - If user_id provided: must match owner_user_id
    - Else if session_id provided: must match owner_session_id
    """
    doc = get_document(db, doc_id=doc_id)
    if doc is None:
        raise http_error(404, "doc_not_found", "Document not found")

    if user_id is not None:
        if doc.owner_user_id != user_id:
            raise http_error(
                403, "forbidden", "You do not have access to this document"
            )
        return doc

    if session_id is not None:
        if doc.owner_session_id != session_id:
            raise http_error(
                403, "forbidden", "You do not have access to this document"
            )
        return doc

    raise http_error(400, "invalid_identity", "Missing user/session identity")


def delete_expired_session_documents(
    db: Session, *, ttl_days: int, now: datetime | None = None
) -> int:
    """
    Deletes DB rows for anonymous/session documents older than ttl_days.
    Returns number of deleted rows.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    cutoff = now - timedelta(days=ttl_days)

    if cutoff.tzinfo is not None:
        cutoff_cmp = cutoff.replace(tzinfo=None)
    else:
        cutoff_cmp = cutoff

    stmt = delete(Document).where(
        and_(
            Document.owner_session_id.is_not(None),
            Document.owner_user_id.is_(None),
            Document.created_at < cutoff_cmp,
        )
    )

    # Prevent in-Python evaluation of criteria against ORM instances.
    result = db.execute(stmt.execution_options(synchronize_session=False))
    db.commit()
    return int(result.rowcount or 0)
