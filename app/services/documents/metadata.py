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
from app.storage.files import ensure_dir, ensure_path_under_root
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

    @property
    def ready_to_ask(self) -> bool:
        return (
            self.has_text and self.has_chunks and self.has_embeddings and self.has_index
        )


REQUIRED_PROCESSED_FILES = (
    "text.json",
    "chunks.jsonl",
    "chunk_map.json",
    "embeddings.npy",
    "embeddings_meta.jsonl",
    "embeddings_info.json",
    "faiss.index",
    "faiss_meta.json",
)


def _safe_read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_original_dir(doc_id: str) -> Path:
    return upload_root() / doc_id / "original"


def _get_processed_dir(doc_id: str) -> Path:
    return processed_root() / doc_id


def _has_original_file(doc_id: str) -> bool:
    original_dir = _get_original_dir(doc_id)
    return (
        original_dir.exists()
        and original_dir.is_dir()
        and any(path.is_file() for path in original_dir.iterdir())
    )


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
    if isinstance(chunk_map_payload, dict) and isinstance(
        chunk_map_payload.get("chunks"), list
    ):
        return len(chunk_map_payload["chunks"])

    faiss_meta_payload = _safe_read_json(get_faiss_meta_path(doc_id))
    if isinstance(faiss_meta_payload, dict):
        row_count = faiss_meta_payload.get("row_count")
        if isinstance(row_count, int) and row_count >= 0:
            return row_count

    chunks_jsonl = get_chunks_jsonl_path(doc_id)
    if chunks_jsonl.exists():
        with chunks_jsonl.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
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
        has_embeddings=embeddings_npy_path.exists()
        and embeddings_meta_path.exists()
        and embeddings_info_path.exists(),
        has_index=faiss_index_path.exists() and faiss_meta_path.exists(),
    )


def document_status_detail(status: str, artifacts: DocumentArtifactState) -> str:
    normalized = (status or "uploaded").lower()
    if normalized == "indexed" and artifacts.ready_to_ask:
        return "Ready to answer questions."
    if normalized == "indexed" and not artifacts.ready_to_ask:
        return "Indexed flag is set, but one or more retrieval artifacts are missing."
    if normalized == "processing":
        return "Document is being extracted, chunked, embedded, and indexed."
    if normalized == "failed":
        return "Processing failed. Re-upload the file or inspect logs for details."
    if normalized == "uploaded":
        return "Upload finished, but processing or indexing has not completed yet."
    return "Document state available."


def _rewrite_json_file(path: Path, transform) -> None:
    payload = _safe_read_json(path)
    if payload is None:
        return
    rewritten = transform(payload)
    path.write_text(
        json.dumps(rewritten, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _rewrite_jsonl_file(path: Path, transform) -> None:
    if not path.exists():
        return
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            lines.append(json.dumps(transform(payload), ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _patch_cloned_artifacts(target_doc_id: str) -> None:
    processed_dir = _get_processed_dir(target_doc_id)

    _rewrite_json_file(
        processed_dir / "text.json",
        lambda payload: (
            {
                **payload,
                "doc_id": target_doc_id,
            }
            if isinstance(payload, dict)
            else payload
        ),
    )

    _rewrite_json_file(
        processed_dir / "chunk_map.json",
        lambda payload: (
            {
                **payload,
                "doc_id": target_doc_id,
            }
            if isinstance(payload, dict)
            else payload
        ),
    )

    _rewrite_json_file(
        processed_dir / "embeddings_info.json",
        lambda payload: (
            {
                **payload,
                "doc_id": target_doc_id,
            }
            if isinstance(payload, dict)
            else payload
        ),
    )

    _rewrite_json_file(
        processed_dir / "faiss_meta.json",
        lambda payload: (
            {
                **payload,
                "doc_id": target_doc_id,
            }
            if isinstance(payload, dict)
            else payload
        ),
    )

    _rewrite_jsonl_file(
        processed_dir / "chunks.jsonl",
        lambda payload: (
            {
                **payload,
                "doc_id": target_doc_id,
            }
            if isinstance(payload, dict)
            else payload
        ),
    )

    _rewrite_jsonl_file(
        processed_dir / "embeddings_meta.jsonl",
        lambda payload: (
            {
                **payload,
                "doc_id": target_doc_id,
            }
            if isinstance(payload, dict)
            else payload
        ),
    )


def clone_processed_artifacts(source_doc_id: str, target_doc_id: str) -> bool:
    source_dir = _get_processed_dir(source_doc_id)
    target_dir = _get_processed_dir(target_doc_id)
    if not source_dir.exists() or not source_dir.is_dir():
        return False
    for filename in REQUIRED_PROCESSED_FILES:
        if not (source_dir / filename).exists():
            return False

    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    ensure_dir(target_dir)

    for filename in REQUIRED_PROCESSED_FILES:
        shutil.copy2(source_dir / filename, target_dir / filename)

    _patch_cloned_artifacts(target_doc_id)
    return True


def delete_document_storage(doc_id: str) -> None:
    uploads_root = upload_root()
    processed_root_path = processed_root()

    upload_dir = ensure_path_under_root(uploads_root / doc_id, uploads_root)
    processed_dir = ensure_path_under_root(
        processed_root_path / doc_id, processed_root_path
    )

    shutil.rmtree(upload_dir, ignore_errors=True)
    shutil.rmtree(processed_dir, ignore_errors=True)
