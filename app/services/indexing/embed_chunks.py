from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

import numpy as np

from app.core.config import settings
from app.services.interfaces import EmbeddingServicePort
from app.storage.chunks import get_chunks_jsonl_path
from app.storage.embeddings import (
    get_embeddings_info_path,
    get_embeddings_meta_jsonl_path,
    get_embeddings_npy_path,
)
from app.storage.files import ensure_dir


@dataclass(frozen=True)
class EmbedResult:
    doc_id: str
    row_count: int
    dim: int
    embeddings_npy: str
    embeddings_meta_jsonl: str
    embeddings_info: str


def chunking_version() -> str:
    raw = (
        f"{settings.CHUNK_SIZE_CHARS}:"
        f"{settings.CHUNK_OVERLAP_CHARS}:"
        f"{settings.CHUNK_MIN_CHARS}:"
        f"{settings.CHUNK_SEPARATORS}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _iter_chunks(doc_id: str) -> Iterator[dict[str, Any]]:
    path = get_chunks_jsonl_path(doc_id)
    if not path.exists():
        raise FileNotFoundError("CHUNKS_NOT_FOUND")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _batched(iterable: Iterator[dict[str, Any]], batch_size: int):
    batch: list[dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def embed_document_chunks(doc_id: str, svc: EmbeddingServicePort) -> EmbedResult:
    ensure_dir(get_embeddings_npy_path(doc_id).parent)

    meta_path = get_embeddings_meta_jsonl_path(doc_id)
    npy_path = get_embeddings_npy_path(doc_id)
    info_path = get_embeddings_info_path(doc_id)

    matrices: list[np.ndarray] = []
    row_count = 0
    dim = 0

    with meta_path.open("w", encoding="utf-8") as meta_handle:
        for batch in _batched(_iter_chunks(doc_id), settings.EMBEDDING_BATCH_SIZE):
            texts = [str(item.get("text") or "") for item in batch]
            if row_count + len(texts) > settings.MAX_CHUNKS_TO_EMBED:
                raise ValueError("TOO_MANY_CHUNKS_TO_EMBED")

            embeddings = svc.encode_texts(texts)
            if embeddings.ndim != 2:
                raise ValueError("INVALID_EMBEDDING_SHAPE")

            if dim == 0:
                dim = int(embeddings.shape[1])
            matrices.append(np.asarray(embeddings, dtype=np.float32))

            for index, item in enumerate(batch):
                meta_handle.write(
                    json.dumps(
                        {
                            "row": row_count + index,
                            "chunk_id": item.get("chunk_id"),
                            "doc_id": item.get("doc_id"),
                            "page": item.get("page"),
                            "chunk_index": item.get("chunk_index"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            row_count += len(batch)

    matrix = (
        np.vstack(matrices).astype(np.float32)
        if matrices
        else np.zeros((0, 0), dtype=np.float32)
    )
    np.save(npy_path, matrix)

    info = {
        "doc_id": doc_id,
        "row_count": row_count,
        "dim": dim,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "normalize": settings.EMBEDDING_NORMALIZE,
        "batch_size": settings.EMBEDDING_BATCH_SIZE,
        "chunking_version": chunking_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    return EmbedResult(
        doc_id=doc_id,
        row_count=row_count,
        dim=dim,
        embeddings_npy=str(npy_path),
        embeddings_meta_jsonl=str(meta_path),
        embeddings_info=str(info_path),
    )


def embedding_cache_key(text: str) -> str:
    digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    return f"embed:{settings.EMBEDDING_MODEL_NAME}:{chunking_version()}:{digest}"
