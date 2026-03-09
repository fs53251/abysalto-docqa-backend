from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid_type():
    return PG_UUID(as_uuid=True)


class User(Base):
    """
    Application user account.

    Relationships:
        documents: Documents owned by this user.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        _uuid_type(), primary_key=True, default=uuid.uuid4
    )

    email: Mapped[str] = mapped_column(
        String(320), unique=True, index=True, nullable=False
    )

    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=text("true"),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    documents: Mapped[list["Document"]] = relationship(back_populates="owner_user")
    """
    Documents owned by this user.

    This relationship links a user to all documents uploaded under their
    authenticated account.
    """


class Document(Base):
    """
    Uploaded document metadata.

    This model represents a file uploaded to the system and stores ownership,
    storage, integrity, and processing metadata.

    A document may belong either to:
    - an authenticated user via 'owner_user_id', or
    - an anonymous session via 'owner_session_id'.

    Relationships:
        owner_user: The authenticated user who owns the document, if any.
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        _uuid_type(), primary_key=True, default=uuid.uuid4
    )

    owner_user_id: Mapped[uuid.UUID | None] = mapped_column(
        _uuid_type(),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    """
    Optional foreign key to the owning user.
    If the owning user is deleted, this value is set to 'NULL' so the
    document record can remain in the database.
    """

    owner_session_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    """
    Optional anonymous session identifier for session-owned documents.
    This is used when a document is uploaded by a non-authenticated client.
    """

    filename: Mapped[str] = mapped_column(String(512), nullable=False)

    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)

    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    stored_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    status: Mapped[str] = mapped_column(String(32), default="uploaded", nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    indexed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    owner_user: Mapped[User | None] = relationship(back_populates="documents")
    """
    The authenticated user who owns this document.
    """
