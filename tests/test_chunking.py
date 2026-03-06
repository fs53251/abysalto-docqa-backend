import json
import uuid
from pathlib import Path

from fastapi.testclient import TestClient


def write_text_json(temp_data_dir: Path, doc_id: str, pages: list[dict]):
    processed = temp_data_dir / "processed" / doc_id
    processed.mkdir(parents=True, exist_ok=True)
    path = processed / "text.json"
    path.write_text(
        json.dumps(
            {"doc_id": doc_id, "page_count": len(pages), "pages": pages}, indent=2
        ),
        encoding="utf-8",
    )
    return path


def test_chunk_endpoint_creates_chunks_and_map(
    client: TestClient,
    temp_data_dir: Path,
    create_owned_document,
):
    doc_id = uuid.uuid4().hex
    create_owned_document(client, doc_id=doc_id)
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

    response = client.post(f"/documents/{doc_id}/chunk")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "chunked"
    assert data["chunk_count"] > 0

    chunks_path = Path(data["chunks_jsonl"])
    map_path = Path(data["chunk_map"])
    assert chunks_path.exists()
    assert map_path.exists()

    first_line = chunks_path.read_text(encoding="utf-8").splitlines()[0]
    obj = json.loads(first_line)
    assert obj["doc_id"] == doc_id
    assert "chunk_id" in obj
    assert "page" in obj
    assert "text" in obj

    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    assert mapping["doc_id"] == doc_id
    assert "chunks" in mapping
    assert len(mapping["chunks"]) == data["chunk_count"]


def test_chunk_endpoint_is_idempotent(
    client: TestClient,
    temp_data_dir: Path,
    create_owned_document,
):
    doc_id = uuid.uuid4().hex
    create_owned_document(client, doc_id=doc_id)
    pages = [
        {
            "page": 1,
            "text": "Some content. " * 200,
            "source": "pymupdf",
            "confidence": None,
        }
    ]
    write_text_json(temp_data_dir, doc_id, pages)

    first = client.post(f"/documents/{doc_id}/chunk")
    assert first.status_code == 200
    chunk_count = first.json()["chunk_count"]

    second = client.post(f"/documents/{doc_id}/chunk")
    assert second.status_code == 200
    data = second.json()
    assert data["status"] == "already_chunked"
    assert data["chunk_count"] == chunk_count


def test_chunk_requires_text_json(
    client: TestClient,
    temp_data_dir: Path,
    create_owned_document,
):
    doc_id = uuid.uuid4().hex
    create_owned_document(client, doc_id=doc_id)
    response = client.post(f"/documents/{doc_id}/chunk")
    assert response.status_code == 404
