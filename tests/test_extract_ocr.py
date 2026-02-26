from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import fitz
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_pdf_bytes_empty_page() -> bytes:
    doc = fitz.open()
    doc.new_page()
    b = doc.tobytes()
    doc.close()
    return b


def test_pdf_ocr_fallback_used_when_empty(temp_data_dir: Path, monkeypatch):
    # Upload empty PDF (valid)
    pdf_bytes = make_pdf_bytes_empty_page()
    files = [("files", ("empty.pdf", BytesIO(pdf_bytes), "application/pdf"))]
    r = client.post("/upload", files=files)
    assert r.status_code == 200, r.text
    doc_id = r.json()["documents"][0]["doc_id"]

    # Mock OCR in pdf_text module
    from app.services.ingestion import pdf_text as pdf_text_mod

    class DummyOcr:
        text = "MOCK OCR"
        confidence = 0.9
        lines = 1

    monkeypatch.setattr(pdf_text_mod, "ocr_image_bytes", lambda _b: DummyOcr())

    r2 = client.post(f"/documents/{doc_id}/extract-text?ocr_fallback=true")
    assert r2.status_code == 200, r2.text

    tj = client.get(f"/documents/{doc_id}/text").json()
    assert tj["pages"][0]["source"] == "easyocr"
    assert tj["pages"][0]["confidence"] == 0.9
    assert "MOCK OCR" in tj["pages"][0]["text"]


def test_image_extract_calls_mocked_route_function(temp_data_dir: Path, monkeypatch):
    # Use a real-looking PNG header to satisfy upload magic check,
    # but we will NOT let OCR/PIL run because we patch route-level function.
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 128
    files = [("files", ("img.png", BytesIO(png), "image/png"))]
    r = client.post("/upload", files=files)
    assert r.status_code == 200, r.text
    doc_id = r.json()["documents"][0]["doc_id"]

    # IMPORTANT: patch the symbol used by the route module (extract.py),
    # because extract.py imports extract_image_text directly.
    import app.api.routes.extract as extract_route

    dummy = SimpleNamespace(doc_id=doc_id, text="IMG MOCK", confidence=0.66)
    monkeypatch.setattr(
        extract_route,
        "extract_image_text",
        lambda doc_id, image_path: dummy,
    )

    # Also patch save_image_text_json to write deterministic output without OCR
    def fake_save_image_text_json(extracted):
        import json

        from app.storage.files import ensure_dir
        from app.storage.processed import get_text_json_path

        p = get_text_json_path(doc_id)
        ensure_dir(p.parent)
        p.write_text(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "page_count": 1,
                    "pages": [
                        {
                            "page": 1,
                            "text": extracted.text,
                            "char_count": len(extracted.text),
                            "is_empty": False,
                            "source": "easyocr",
                            "confidence": extracted.confidence,
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return p

    monkeypatch.setattr(extract_route, "save_image_text_json", fake_save_image_text_json)

    r2 = client.post(f"/documents/{doc_id}/extract-text")
    assert r2.status_code == 200, r2.text

    tj = client.get(f"/documents/{doc_id}/text").json()
    assert tj["pages"][0]["source"] == "easyocr"
    assert tj["pages"][0]["confidence"] == 0.66
    assert "IMG MOCK" in tj["pages"][0]["text"]
