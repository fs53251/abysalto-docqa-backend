from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.base import Base

# SQLAlchemy database setup module
#   1) create the database engine
#   2) crate a session factory
#   3) provide a DB session to the app

# lazy-init cached singletons, loaded once!
_engine = None
_SessionLocal = None


def get_engine():
    """
    Apps database connection infrastructure.
    Connects to local DB (SQLite, PostgreSql)
    """
    global _engine
    if _engine is None:
        _engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

    return _engine


def get_sessionmaker():
    """
    Factory conf for making sessions.

    autocommit=False:
        - must explicitly call: db.commit()!!!
    connects session factory to engine
    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )

    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    Creates one db session, yields it and close at end.
    FastAPI start the generator, get yielded db and
    use it in request, than continue to finally.
    """
    db = get_sessionmaker()()

    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> None:
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))


def init_db_dev_failsafe() -> None:
    """
    Automatically create tables, only dev/test
    """
    if settings.APP_ENV in {"dev", "test"}:
        Base.metadata.create_all(bind=get_engine())
