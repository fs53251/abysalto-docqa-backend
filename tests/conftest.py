from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.deps import (
    get_embedding_service,
    get_optional_cache,
    get_optional_ner_service,
    get_qa_service,
)
from app.core.config import settings
from app.core.identifiers import (
    document_public_id,
    generate_document_id,
    parse_document_public_id,
)
from app.core.security.session import load_session_cookie
from app.db.base import Base
from app.db.session import get_engine, get_sessionmaker
from app.main import app
from app.repositories.documents import create_document


class DummyEmbeddingService:
    def load(self) -> None:
        return

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 3), dtype=np.float32)
        for index, text in enumerate(texts):
            out[index, 0] = float(len(text))
        return out


class DummyQAService:
    model_name = "dummy-qa"

    def load(self) -> None:
        return

    def answer(self, question: str, context: str):
        from app.services.interfaces import QaResult

        return QaResult(answer="dummy", score=0.99)


@dataclass
class TestServices:
    embedding: Any
    qa: Any
    ner: Any | None = None
    cache: Any | None = None


@dataclass(frozen=True)
class SessionOwnedDocument:
    doc_id: str
    session_id: str


@dataclass(frozen=True)
class UserOwnedDocument:
    doc_id: str
    user_id: str


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    email: str
    access_token: str

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(settings, "APP_ENV", "test", raising=False)
    monkeypatch.setattr(settings, "ENABLE_CACHE", False, raising=False)
    yield


@pytest.fixture(autouse=True)
def _isolate_test_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[None]:
    sqlite_url = f"sqlite:///{tmp_path / 'app.db'}"

    import app.db.session as db_session_module

    monkeypatch.setattr(settings, "DATABASE_URL", sqlite_url, raising=False)
    db_session_module._engine = None
    db_session_module._SessionLocal = None

    Base.metadata.create_all(bind=get_engine())

    yield

    db_session_module._engine = None
    db_session_module._SessionLocal = None


@pytest.fixture()
def temp_data_dir(tmp_path: Path) -> Iterator[Path]:
    old = settings.DATA_DIR
    settings.DATA_DIR = str(tmp_path)
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    yield tmp_path
    settings.DATA_DIR = old


@pytest.fixture()
def services() -> TestServices:
    return TestServices(
        embedding=DummyEmbeddingService(),
        qa=DummyQAService(),
        ner=None,
        cache=None,
    )


@pytest.fixture()
def client(services: TestServices) -> Iterator[TestClient]:
    app.dependency_overrides[get_embedding_service] = lambda: services.embedding
    app.dependency_overrides[get_qa_service] = lambda: services.qa
    app.dependency_overrides[get_optional_ner_service] = lambda: services.ner
    app.dependency_overrides[get_optional_cache] = lambda: services.cache

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture()
def session_id_for_client() -> Callable[[TestClient], str]:
    def _session_id_for_client(client: TestClient) -> str:
        client.get("/documents")
        signed_cookie = client.cookies.get(settings.SESSION_COOKIE_NAME)
        assert signed_cookie is not None
        session_id = load_session_cookie(signed_cookie)
        assert session_id is not None
        return session_id

    return _session_id_for_client


@pytest.fixture()
def register_and_login(
    client: TestClient,
) -> Callable[..., AuthenticatedUser]:
    def _register_and_login(
        *,
        email: str | None = None,
        password: str = "supersecret123",
    ) -> AuthenticatedUser:
        normalized_email = (
            email.strip().lower()
            if email is not None
            else f"user-{uuid.uuid4().hex}@example.com"
        )

        register_res = client.post(
            "/auth/register",
            json={"email": normalized_email, "password": password},
        )
        assert register_res.status_code == 201, register_res.text
        user_id = register_res.json()["id"]

        login_res = client.post(
            "/auth/login",
            json={"email": normalized_email, "password": password},
        )
        assert login_res.status_code == 200, login_res.text
        access_token = login_res.json()["access_token"]

        return AuthenticatedUser(
            user_id=user_id,
            email=normalized_email,
            access_token=access_token,
        )

    return _register_and_login


@pytest.fixture()
def create_owned_document(
    temp_data_dir: Path,
    session_id_for_client: Callable[[TestClient], str],
) -> Callable[..., SessionOwnedDocument]:
    def _create_owned_document(
        client: TestClient,
        *,
        doc_id: str | None = None,
        filename: str = "test.pdf",
        content_type: str = "application/pdf",
        status: str = "uploaded",
        stored_path: str | None = None,
    ) -> SessionOwnedDocument:
        session_id = session_id_for_client(client)
        document_uuid = (
            generate_document_id()
            if doc_id is None
            else parse_document_public_id(doc_id)
        )
        public_doc_id = document_public_id(document_uuid)
        if stored_path is None:
            stored_path = str(
                temp_data_dir / "uploads" / public_doc_id / "original" / filename
            )
        path = Path(stored_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

        session_local = get_sessionmaker()
        db = session_local()
        try:
            create_document(
                db,
                doc_id=document_uuid,
                filename=filename,
                content_type=content_type,
                stored_path=str(path),
                owner_session_id=session_id,
                status=status,
            )
        finally:
            db.close()

        return SessionOwnedDocument(doc_id=public_doc_id, session_id=session_id)

    return _create_owned_document


@pytest.fixture()
def create_user_owned_document(
    temp_data_dir: Path,
) -> Callable[..., UserOwnedDocument]:
    def _create_user_owned_document(
        *,
        user_id: str,
        doc_id: str | None = None,
        filename: str = "user-test.pdf",
        content_type: str = "application/pdf",
        status: str = "uploaded",
        stored_path: str | None = None,
    ) -> UserOwnedDocument:
        owner_user_id = uuid.UUID(str(user_id))
        document_uuid = (
            generate_document_id()
            if doc_id is None
            else parse_document_public_id(doc_id)
        )
        public_doc_id = document_public_id(document_uuid)

        if stored_path is None:
            stored_path = str(
                temp_data_dir / "uploads" / public_doc_id / "original" / filename
            )

        path = Path(stored_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

        session_local = get_sessionmaker()
        db = session_local()
        try:
            create_document(
                db,
                doc_id=document_uuid,
                filename=filename,
                content_type=content_type,
                stored_path=str(path),
                owner_user_id=owner_user_id,
                status=status,
            )
        finally:
            db.close()

        return UserOwnedDocument(doc_id=public_doc_id, user_id=str(owner_user_id))

    return _create_user_owned_document
