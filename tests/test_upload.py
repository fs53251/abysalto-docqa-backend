from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.identifiers import parse_document_public_id
from app.core.security.session import load_session_cookie
from app.db.session import get_sessionmaker
from app.repositories.documents import get_document

fitz = pytest.importorskip("fitz")
pytest.importorskip("faiss")


def _make_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    payload = doc.tobytes()
    doc.close()
    return payload


def test_upload_multiple_ok_runs_processing_pipeline(
    client: TestClient,
    temp_data_dir: Path,
):
    files = [
        (
            "files",
            (
                "first.pdf",
                BytesIO(_make_pdf_bytes("First valid PDF content for pipeline.")),
                "application/pdf",
            ),
        ),
        (
            "files",
            (
                "second.pdf",
                BytesIO(_make_pdf_bytes("Second valid PDF content for pipeline.")),
                "application/pdf",
            ),
        ),
    ]

    response = client.post("/upload", files=files)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["has_errors"] is False
    assert len(data["documents"]) == 2
    assert all(item["status"] == "indexed" for item in data["documents"])
    assert all(item["owner_type"] == "session" for item in data["documents"])

    for item in data["documents"]:
        doc_id = item["doc_id"]
        doc_dir = temp_data_dir / "uploads" / doc_id
        processed_dir = temp_data_dir / "processed" / doc_id

        assert (doc_dir / "metadata.json").exists()
        assert (doc_dir / "original").exists()
        assert (processed_dir / "text.json").exists()
        assert (processed_dir / "chunks.jsonl").exists()
        assert (processed_dir / "chunk_map.json").exists()
        assert (processed_dir / "embeddings.npy").exists()
        assert (processed_dir / "embeddings_meta.jsonl").exists()
        assert (processed_dir / "embeddings_info.json").exists()
        assert (processed_dir / "faiss.index").exists()
        assert (processed_dir / "faiss_meta.json").exists()


def test_upload_rejects_wrong_extension_best_effort(
    client: TestClient,
    temp_data_dir: Path,
):
    files = [
        (
            "files",
            (
                "ok.pdf",
                BytesIO(_make_pdf_bytes("Valid PDF for mixed upload.")),
                "application/pdf",
            ),
        ),
        ("files", ("evil.exe", BytesIO(b"nope"), "application/octet-stream")),
    ]

    response = client.post("/upload", files=files)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["has_errors"] is True
    assert len(data["documents"]) == 2
    assert data["documents"][0]["status"] == "indexed"
    assert data["documents"][0]["owner_type"] == "session"
    assert data["documents"][1]["status"] == "error"


def test_upload_rejects_wrong_mime(client: TestClient, temp_data_dir: Path):
    files = [
        (
            "files",
            (
                "test.pdf",
                BytesIO(_make_pdf_bytes("Valid PDF bytes but wrong MIME.")),
                "text/plain",
            ),
        )
    ]
    response = client.post("/upload", files=files)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["has_errors"] is True
    assert data["documents"][0]["status"] == "error"


def test_upload_rejects_magic_bytes_mismatch(client: TestClient, temp_data_dir: Path):
    files = [("files", ("test.pdf", BytesIO(b"NOTPDF"), "application/pdf"))]

    response = client.post("/upload", files=files)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["has_errors"] is True
    assert data["documents"][0]["status"] == "error"
    assert "Magic-bytes" in data["documents"][0]["error_detail"]


def test_upload_rejects_too_large_file(
    client: TestClient,
    temp_data_dir: Path,
    monkeypatch,
):
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)

    payload = b"%PDF-1.4\n" + (b"a" * (1024 * 1024 + 10))
    big = BytesIO(payload)
    files = [("files", ("big.pdf", big, "application/pdf"))]

    response = client.post("/upload", files=files)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["has_errors"] is True
    assert data["documents"][0]["status"] == "error"
    assert "max size" in data["documents"][0]["error_detail"].lower()


def test_upload_sanitizes_filename_no_path_traversal(
    client: TestClient,
    temp_data_dir: Path,
):
    files = [
        (
            "files",
            (
                "../../etc/passwd.pdf",
                BytesIO(_make_pdf_bytes("Sanitized filename upload.")),
                "application/pdf",
            ),
        )
    ]

    response = client.post("/upload", files=files)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["has_errors"] is False
    doc_id = data["documents"][0]["doc_id"]

    assert not (temp_data_dir / "etc").exists()

    original_dir = temp_data_dir / "uploads" / doc_id / "original"
    assert original_dir.exists()

    saved_files = list(original_dir.glob("*"))
    assert len(saved_files) == 1

    upload_root = (temp_data_dir / "uploads").resolve()
    assert saved_files[0].resolve().is_relative_to(upload_root)


def test_anon_upload_sets_session_owner_and_returns_owner_type(
    client: TestClient,
    temp_data_dir: Path,
):
    response = client.post(
        "/upload",
        files=[
            (
                "files",
                (
                    "anon.pdf",
                    BytesIO(_make_pdf_bytes("Anonymous upload document.")),
                    "application/pdf",
                ),
            )
        ],
    )
    assert response.status_code == 200, response.text

    item = response.json()["documents"][0]
    assert item["owner_type"] == "session"
    assert item["status"] == "indexed"

    signed_cookie = client.cookies.get(settings.SESSION_COOKIE_NAME)
    assert signed_cookie is not None
    session_id = load_session_cookie(signed_cookie)
    assert session_id is not None

    session_local = get_sessionmaker()
    db = session_local()
    try:
        document = get_document(db, doc_id=parse_document_public_id(item["doc_id"]))
        assert document is not None
        assert document.owner_session_id == session_id
        assert document.owner_user_id is None
        assert document.status == "indexed"
    finally:
        db.close()


def test_auth_upload_sets_user_owner_and_returns_owner_type(
    client: TestClient,
    temp_data_dir: Path,
    register_and_login,
):
    auth_user = register_and_login(email="upload-owner@example.com")

    response = client.post(
        "/upload",
        headers=auth_user.headers,
        files=[
            (
                "files",
                (
                    "user-owned.pdf",
                    BytesIO(_make_pdf_bytes("Authenticated upload document.")),
                    "application/pdf",
                ),
            )
        ],
    )
    assert response.status_code == 200, response.text

    item = response.json()["documents"][0]
    assert item["owner_type"] == "user"
    assert item["status"] == "indexed"

    session_local = get_sessionmaker()
    db = session_local()
    try:
        document = get_document(db, doc_id=parse_document_public_id(item["doc_id"]))
        assert document is not None
        assert str(document.owner_user_id) == auth_user.user_id
        assert document.owner_session_id is None
        assert document.status == "indexed"
    finally:
        db.close()


def test_upload_processing_failure_marks_document_failed(
    client: TestClient,
    temp_data_dir: Path,
):
    response = client.post(
        "/upload",
        files=[
            (
                "files",
                (
                    "broken.pdf",
                    BytesIO(b"%PDF-1.4 definitely-not-a-real-pdf"),
                    "application/pdf",
                ),
            )
        ],
    )
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["has_errors"] is True

    item = data["documents"][0]
    assert item["status"] == "failed"
    assert item["owner_type"] == "session"
    assert item["doc_id"] is not None
    assert item["error_detail"]

    session_local = get_sessionmaker()
    db = session_local()
    try:
        document = get_document(db, doc_id=parse_document_public_id(item["doc_id"]))
        assert document is not None
        assert document.status == "failed"
        assert document.owner_session_id is not None
        assert document.owner_user_id is None
    finally:
        db.close()
