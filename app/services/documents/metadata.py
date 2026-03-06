from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.core.config import processed_root, upload_root
from app.storage.chunks import get_chunk_map_path, get_chunks_jsonl_path
from app.storage.embeddings import (
    get_embeddings_info_path,
    get_embeddings_meta_jsonl_path,
    get_embeddings_npy_path,
)
from app.storage.faiss_store import get_faiss_index_path, get_faiss_meta_path
from app.storage.files import ensure_path_under_root
from app.storage.processed import get_text_json_path
from app.storage.upload_registry import get_metadata_path


@dataclass(frozen=True)
class DocumentArtifactState:
    page_count: int | None
    chunk_count: int | None
    has_metadata: bool
    has_original: bool
    has_text: bool
    has_chunks: bool
    has_embeddings: bool
    has_index: bool


def _safe_read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_original_dir(doc_id: str) -> Path:
    return upload_root() / doc_id / "original"


def _has_original_file(doc_id: str) -> bool:
    original_dir = _get_original_dir(doc_id)
    if not original_dir.exists() or not original_dir.is_dir():
        return False

    return any(path.is_file() for path in original_dir.iterdir())


def _read_page_count(doc_id: str) -> int | None:
    payload = _safe_read_json(get_text_json_path(doc_id))
    if not isinstance(payload, dict):
        return None

    page_count = payload.get("page_count")
    if isinstance(page_count, int) and page_count >= 0:
        return page_count

    pages = payload.get("pages")
    if isinstance(pages, list):
        return len(pages)

    return None


def _read_chunk_count(doc_id: str) -> int | None:
    chunk_map_payload = _safe_read_json(get_chunk_map_path(doc_id))
    if isinstance(chunk_map_payload, dict):
        chunks = chunk_map_payload.get("chunks")
        if isinstance(chunks, list):
            return len(chunks)

    faiss_meta_payload = _safe_read_json(get_faiss_meta_path(doc_id))
    if isinstance(faiss_meta_payload, dict):
        row_count = faiss_meta_payload.get("row_count")
        if isinstance(row_count, int) and row_count >= 0:
            return row_count

    chunks_jsonl = get_chunks_jsonl_path(doc_id)
    if chunks_jsonl.exists():
        try:
            with chunks_jsonl.open("r", encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())
        except Exception:
            return None

    return None


def build_document_artifact_state(doc_id: str) -> DocumentArtifactState:
    metadata_path = get_metadata_path(doc_id)
    text_path = get_text_json_path(doc_id)
    chunk_map_path = get_chunk_map_path(doc_id)
    chunks_jsonl_path = get_chunks_jsonl_path(doc_id)
    embeddings_npy_path = get_embeddings_npy_path(doc_id)
    embeddings_meta_path = get_embeddings_meta_jsonl_path(doc_id)
    embeddings_info_path = get_embeddings_info_path(doc_id)
    faiss_index_path = get_faiss_index_path(doc_id)
    faiss_meta_path = get_faiss_meta_path(doc_id)

    return DocumentArtifactState(
        page_count=_read_page_count(doc_id),
        chunk_count=_read_chunk_count(doc_id),
        has_metadata=metadata_path.exists(),
        has_original=_has_original_file(doc_id),
        has_text=text_path.exists(),
        has_chunks=chunk_map_path.exists() or chunks_jsonl_path.exists(),
        has_embeddings=(
            embeddings_npy_path.exists()
            and embeddings_meta_path.exists()
            and embeddings_info_path.exists()
        ),
        has_index=faiss_index_path.exists() and faiss_meta_path.exists(),
    )


def delete_document_storage(doc_id: str) -> None:
    uploads_root = upload_root()
    processed_root_path = processed_root()

    upload_dir = ensure_path_under_root(uploads_root / doc_id, uploads_root)
    processed_dir = ensure_path_under_root(
        processed_root_path / doc_id, processed_root_path
    )

    shutil.rmtree(upload_dir, ignore_errors=True)
    shutil.rmtree(processed_dir, ignore_errors=True)
