from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_upload_multiple_ok(temp_data_dir: Path):
    files = [
        ("files", ("test.pdf", BytesIO(b"%PDF-1.4 fake pdf content"), "application/pdf")),
        ("files", ("image.png", BytesIO(b"\x89PNG\r\n\x1a\nfake"), "image/png")),
    ]

    r = client.post("/upload", files=files)
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["has_errors"] is False
    assert len(data["documents"]) == 2
    assert all(d["status"] == "ok" for d in data["documents"])

    for d in data["documents"]:
        doc_id = d["doc_id"]
        doc_dir = temp_data_dir / "uploads" / doc_id
        assert (doc_dir / "metadata.json").exists()
        assert (doc_dir / "original").exists()


def test_upload_rejects_wrong_extension_best_effort(temp_data_dir: Path):
    files = [
        ("files", ("ok.pdf", BytesIO(b"%PDF-1.4 fake"), "application/pdf")),
        ("files", ("evil.exe", BytesIO(b"nope"), "application/octet-stream")),
    ]
    r = client.post("/upload", files=files)
    assert r.status_code == 200, r.text

    data = r.json()
    assert data["has_errors"] is True
    assert len(data["documents"]) == 2
    assert data["documents"][0]["status"] == "ok"
    assert data["documents"][1]["status"] == "error"


def test_upload_rejects_wrong_mime(temp_data_dir: Path):
    # PDF extension but mime text/plain
    files = [("files", ("test.pdf", BytesIO(b"%PDF-1.4 fake"), "text/plain"))]
    r = client.post("/upload", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["has_errors"] is True
    assert data["documents"][0]["status"] == "error"


def test_upload_rejects_magic_bytes_mismatch(temp_data_dir: Path):
    # says pdf but bytes not %PDF
    files = [("files", ("test.pdf", BytesIO(b"NOTPDF"), "application/pdf"))]
    r = client.post("/upload", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["has_errors"] is True
    assert data["documents"][0]["status"] == "error"
    assert "Magic-bytes" in data["documents"][0]["error_detail"]


def test_upload_rejects_too_large_file(temp_data_dir: Path, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)  # 1MB

    # Must pass magic-bytes check first (%PDF...)
    payload = b"%PDF-1.4\n" + (b"a" * (1024 * 1024 + 10))
    big = BytesIO(payload)

    files = [("files", ("big.pdf", big, "application/pdf"))]
    r = client.post("/upload", files=files)

    assert r.status_code == 200, r.text
    data = r.json()
    assert data["has_errors"] is True
    assert data["documents"][0]["status"] == "error"
    assert "max size" in data["documents"][0]["error_detail"].lower()


def test_upload_sanitizes_filename_no_path_traversal(temp_data_dir: Path):
    files = [("files", ("../../etc/passwd.pdf", BytesIO(b"%PDF-1.4 fake"), "application/pdf"))]
    r = client.post("/upload", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["has_errors"] is False
    doc_id = data["documents"][0]["doc_id"]

    # ensure no unexpected dirs created
    assert not (temp_data_dir / "etc").exists()

    original_dir = temp_data_dir / "uploads" / doc_id / "original"
    assert original_dir.exists()
    # there should be exactly one file inside original/
    saved_files = list(original_dir.glob("*"))
    assert len(saved_files) == 1
