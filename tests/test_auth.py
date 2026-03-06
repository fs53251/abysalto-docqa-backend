from __future__ import annotations

from datetime import timedelta
from io import BytesIO

from fastapi.testclient import TestClient

from app.core.security.jwt import create_access_token
from app.core.security.passwords import hash_password
from app.db.session import get_sessionmaker
from app.repositories.users import create_user, get_user_by_email


def test_register_and_login_happy_path(client: TestClient) -> None:
    register_res = client.post(
        "/auth/register",
        json={"email": "  USER@Example.com ", "password": "supersecret123"},
    )
    assert register_res.status_code == 201, register_res.text

    register_body = register_res.json()
    assert register_body["email"] == "user@example.com"
    assert "password_hash" not in register_body
    assert register_body["is_active"] is True
    assert register_body["id"]

    login_res = client.post(
        "/auth/login",
        json={"email": "USER@example.com", "password": "supersecret123"},
    )
    assert login_res.status_code == 200, login_res.text

    login_body = login_res.json()
    assert login_body["token_type"] == "bearer"
    assert isinstance(login_body["access_token"], str) and login_body["access_token"]
    assert login_body["expires_in"] > 0

    me_res = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {login_body['access_token']}"},
    )
    assert me_res.status_code == 200, me_res.text
    assert me_res.json()["email"] == "user@example.com"


def test_register_rejects_duplicate_email(client: TestClient) -> None:
    first = client.post(
        "/auth/register",
        json={"email": "dupe@example.com", "password": "supersecret123"},
    )
    assert first.status_code == 201, first.text

    second = client.post(
        "/auth/register",
        json={"email": "  DUPE@example.com ", "password": "supersecret123"},
    )
    assert second.status_code == 409, second.text
    assert second.json()["error_code"] == "conflict"


def test_login_invalid_password_returns_401(client: TestClient) -> None:
    client.post(
        "/auth/register",
        json={"email": "user@example.com", "password": "supersecret123"},
    )

    res = client.post(
        "/auth/login",
        json={"email": "user@example.com", "password": "wrong-password"},
    )
    assert res.status_code == 401, res.text
    assert res.json()["error_code"] == "invalid_credentials"


def test_expired_token_returns_401(client: TestClient) -> None:
    client.post(
        "/auth/register",
        json={"email": "expired@example.com", "password": "supersecret123"},
    )

    session_local = get_sessionmaker()
    db = session_local()
    try:
        user = get_user_by_email(db, email="expired@example.com")
        assert user is not None
        token = create_access_token(sub=user.id, expires_delta=timedelta(seconds=-1))
    finally:
        db.close()

    res = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 401, res.text
    assert res.json()["error_code"] == "auth_token_expired"


def test_inactive_user_returns_401(client: TestClient) -> None:
    session_local = get_sessionmaker()
    db = session_local()
    try:
        create_user(
            db,
            email="inactive@example.com",
            password_hash=hash_password("supersecret123"),
            is_active=False,
        )
    finally:
        db.close()

    res = client.post(
        "/auth/login",
        json={"email": "inactive@example.com", "password": "supersecret123"},
    )
    assert res.status_code == 401, res.text
    assert res.json()["error_code"] == "inactive_user"


def test_protected_route_requires_token(client: TestClient) -> None:
    res = client.get("/auth/me")
    assert res.status_code == 401, res.text
    assert res.json()["error_code"] == "auth_required"


def test_login_claims_existing_session_documents(client: TestClient) -> None:
    upload_res = client.post(
        "/upload",
        files=[
            (
                "files",
                (
                    "claim-me.pdf",
                    BytesIO(b"%PDF-1.4 fake pdf content"),
                    "application/pdf",
                ),
            )
        ],
    )
    assert upload_res.status_code == 200, upload_res.text
    uploaded_doc_id = upload_res.json()["documents"][0]["doc_id"]

    register_res = client.post(
        "/auth/register",
        json={"email": "claim@example.com", "password": "supersecret123"},
    )
    assert register_res.status_code == 201, register_res.text

    login_res = client.post(
        "/auth/login",
        json={"email": "claim@example.com", "password": "supersecret123"},
    )
    assert login_res.status_code == 200, login_res.text
    token = login_res.json()["access_token"]

    docs_res = client.get(
        "/documents",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert docs_res.status_code == 200, docs_res.text
    docs = docs_res.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["doc_id"] == uploaded_doc_id
