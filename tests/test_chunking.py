import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def write_text_json(temp_data_dir: Path, doc_id: str, pages: list[dict]):
    processed = temp_data_dir / "processed" / doc_id
    processed.mkdir(parents=True, exist_ok=True)
    p = processed / "text.json"
    p.write_text(
        json.dumps({"doc_id": doc_id, "page_count": len(pages), "pages": pages}, indent=2),
        encoding="utf-8",
    )
    return p


def test_chunk_endpoint_creates_chunks_and_map(temp_data_dir: Path):
    doc_id = "a" * 32
    pages = [
        {
            "page": 1,
            "text": "Paragraph one.\n\nParagraph two is a bit longer. " * 30,
            "source": "pymupdf",
            "confidence": None,
        },
        {
            "page": 2,
            "text": "Second page content. " * 60,
            "source": "easyocr",
            "confidence": 0.8,
        },
    ]
    write_text_json(temp_data_dir, doc_id, pages)

    r = client.post(f"/documents/{doc_id}/chunk")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "chunked"
    assert data["chunk_count"] > 0

    chunks_path = Path(data["chunks_jsonl"])
    map_path = Path(data["chunk_map"])

    assert chunks_path.exists()
    assert map_path.exists()

    # chunks.jsonl lines are valid JSON and contain required keys
    first_line = chunks_path.read_text(encoding="utf-8").splitlines()[0]
    obj = json.loads(first_line)
    assert obj["doc_id"] == doc_id
    assert "chunk_id" in obj
    assert "page" in obj
    assert "text" in obj

    # chunk_map has mapping entries
    m = json.loads(map_path.read_text(encoding="utf-8"))
    assert m["doc_id"] == doc_id
    assert "chunks" in m
    assert len(m["chunks"]) == data["chunk_count"]


def test_chunk_endpoint_is_idempotent(temp_data_dir: Path):
    doc_id = "b" * 32
    pages = [
        {
            "page": 1,
            "text": "Some content. " * 200,
            "source": "pymupdf",
            "confidence": None,
        }
    ]
    write_text_json(temp_data_dir, doc_id, pages)

    r1 = client.post(f"/documents/{doc_id}/chunk")
    assert r1.status_code == 200
    c1 = r1.json()["chunk_count"]

    r2 = client.post(f"/documents/{doc_id}/chunk")
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["status"] == "already_chunked"
    assert data2["chunk_count"] == c1


def test_chunk_requires_text_json(temp_data_dir: Path):
    doc_id = "c" * 32
    r = client.post(f"/documents/{doc_id}/chunk")
    assert r.status_code == 404
