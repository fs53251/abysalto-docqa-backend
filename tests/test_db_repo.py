from datetime import datetime, timedelta, timezone
import uuid

import pytest

from app.db.base import Base
from app.db.session import get_engine, get_sessionmaker
from app.repositories.documents import (
    create_document,
    delete_expired_session_documents,
    get_document_for_session,
    list_documents_for_session,
    list_documents_for_user,
    mark_document_indexed,
)
from app.repositories.users import create_user, get_user, get_user_by_email

sqlalchemy = pytest.importorskip("sqlalchemy")


def _reset_db_engine(monkeypatch, sqlite_url: str):
    import app.db.session as db_session_module
    from app.core.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", sqlite_url, raising=False)
    monkeypatch.setattr(settings, "APP_ENV", "test", raising=False)
    db_session_module._engine = None
    db_session_module._SessionLocal = None


@pytest.fixture()
def db_session(monkeypatch, tmp_path):
    db_file = tmp_path / "test.db"
    sqlite_url = f"sqlite:///{db_file}"

    _reset_db_engine(monkeypatch, sqlite_url)

    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    session_local = get_sessionmaker()
    db = session_local()
    try:
        yield db
    finally:
        db.close()


def test_user_repository_create_and_get(db_session):
    user = create_user(db_session, email="a@example.com", password_hash="hash123")
    assert user.id is not None
    assert user.email == "a@example.com"
    assert user.is_active is True

    by_email = get_user_by_email(db_session, email="a@example.com")
    assert by_email is not None
    assert by_email.id == user.id

    by_id = get_user(db_session, user_id=user.id)
    assert by_id is not None
    assert by_id.email == "a@example.com"


def test_documents_repository_user_ownership(db_session):
    user_a = create_user(db_session, email="a@example.com", password_hash="hashA")
    user_b = create_user(db_session, email="b@example.com", password_hash="hashB")

    doc = create_document(
        db_session,
        filename="doc.pdf",
        content_type="application/pdf",
        size_bytes=123,
        stored_path="/tmp/uploads/x/original/doc.pdf",
        owner_user_id=user_a.id,
    )

    docs_a = list_documents_for_user(db_session, user_id=user_a.id)
    assert len(docs_a) == 1
    assert docs_a[0].id == doc.id

    docs_b = list_documents_for_user(db_session, user_id=user_b.id)
    assert len(docs_b) == 0


def test_documents_repository_session_ownership_and_cleanup(db_session):
    session_id = str(uuid.uuid4())
    other_session_id = str(uuid.uuid4())

    recent = create_document(
        db_session,
        filename="recent.pdf",
        owner_session_id=session_id,
        stored_path="/tmp/uploads/recent/original/recent.pdf",
    )

    old = create_document(
        db_session,
        filename="old.pdf",
        owner_session_id=session_id,
        stored_path="/tmp/uploads/old/original/old.pdf",
    )

    old.created_at = datetime.now(timezone.utc) - timedelta(days=30)
    db_session.add(old)
    db_session.commit()

    docs = list_documents_for_session(db_session, session_id=session_id)
    assert {document.filename for document in docs} == {"recent.pdf", "old.pdf"}

    owned = get_document_for_session(
        db_session, doc_id=recent.id, session_id=session_id
    )
    assert owned.id == recent.id

    with pytest.raises(Exception):
        get_document_for_session(
            db_session, doc_id=recent.id, session_id=other_session_id
        )

    deleted = delete_expired_session_documents(db_session, ttl_days=7)
    assert deleted == 1

    remaining = list_documents_for_session(db_session, session_id=session_id)
    assert len(remaining) == 1
    assert remaining[0].filename == "recent.pdf"


def test_mark_document_indexed_updates_status_and_timestamp(db_session):
    session_id = str(uuid.uuid4())
    document = create_document(
        db_session,
        filename="indexed.pdf",
        owner_session_id=session_id,
        stored_path="/tmp/uploads/indexed/original/indexed.pdf",
    )

    assert document.status == "uploaded"
    assert document.indexed_at is None

    updated = mark_document_indexed(db_session, document=document)
    assert updated.status == "indexed"
    assert updated.indexed_at is not None
