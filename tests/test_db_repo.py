from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import settings
from app.db.base import Base
from app.db.session import get_engine, get_sessionmaker
from app.repositories.documents import (
    assert_document_owner,
    create_document,
    delete_expired_session_documents,
    list_documents_for_session,
    list_documents_for_user,
)
from app.repositories.users import create_user, get_user, get_user_by_email


def _reset_db_engine(monkeypatch, sqlite_url: str):
    """
    app.db.session caches engine/sessionmaker in module globals.
    We reset them for clean tests and point DATABASE_URL to our temp sqlite.
    """
    import app.db.session as db_session_module

    monkeypatch.setattr(settings, "DATABASE_URL", sqlite_url, raising=False)
    monkeypatch.setattr(settings, "APP_ENV", "test", raising=False)

    # reset cached engine/sessionmaker
    db_session_module._engine = None
    db_session_module._SessionLocal = None


@pytest.fixture()
def db_session(monkeypatch, tmp_path):
    db_file = tmp_path / "test.db"
    sqlite_url = f"sqlite:///{db_file}"

    _reset_db_engine(monkeypatch, sqlite_url)

    # Create tables
    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    SessionLocal = get_sessionmaker()
    db = SessionLocal()
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

    # listing works
    docs_a = list_documents_for_user(db_session, user_id=user_a.id)
    assert len(docs_a) == 1
    assert docs_a[0].id == doc.id

    docs_b = list_documents_for_user(db_session, user_id=user_b.id)
    assert len(docs_b) == 0

    # assert owner ok for user_a
    ok = assert_document_owner(db_session, doc_id=doc.id, user_id=user_a.id)
    assert ok.id == doc.id

    # assert owner forbidden for user_b
    with pytest.raises(Exception) as excinfo:
        assert_document_owner(db_session, doc_id=doc.id, user_id=user_b.id)

    # Our http_error returns ApiError which is an HTTPException; just check message
    assert (
        "access" in str(excinfo.value).lower()
        or "forbidden" in str(excinfo.value).lower()
    )


def test_documents_repository_session_ownership_and_cleanup(db_session, monkeypatch):
    session_id = "sess-123"
    other_session_id = "sess-999"

    # create a recent session doc
    recent = create_document(
        db_session,
        filename="recent.pdf",
        owner_session_id=session_id,
        stored_path="/tmp/uploads/recent/original/recent.pdf",
    )

    # create an old session doc (manually backdate created_at)
    old = create_document(
        db_session,
        filename="old.pdf",
        owner_session_id=session_id,
        stored_path="/tmp/uploads/old/original/old.pdf",
    )

    old.created_at = datetime.now(timezone.utc) - timedelta(days=30)
    db_session.add(old)
    db_session.commit()

    # list for session
    docs = list_documents_for_session(db_session, session_id=session_id)
    assert {d.filename for d in docs} == {"recent.pdf", "old.pdf"}

    # other session should not own
    with pytest.raises(Exception):
        assert_document_owner(db_session, doc_id=recent.id, session_id=other_session_id)

    # cleanup with ttl_days=7 should delete the old one only
    deleted = delete_expired_session_documents(db_session, ttl_days=7)
    assert deleted == 1

    remaining = list_documents_for_session(db_session, session_id=session_id)
    assert len(remaining) == 1
    assert remaining[0].filename == "recent.pdf"
