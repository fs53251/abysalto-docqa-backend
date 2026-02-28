from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

import numpy as np

from app.core.config import settings
from app.services.indexing.embedding_service import EmbeddingService
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


def _chunking_version() -> str:
    """
    Hashing the chunking strategy!!!
    If chunking parameters change, the version string changes.
    Any cached data based on old chunking becomes invalid.
    Example:
        I chunk documents for embeddings
        Then I cache those chunks
        If chunk size changes (or any other param), I must rebuild the chunks
    """
    raw = f"{settings.CHUNK_SIZE_CHARS}:{settings.CHUNK_OVERLAP_CHARS}:{settings.CHUNK_SEPARATORS}"

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# Optimization, jsonl, do not load all to RAM, load per request!
def _iter_chunks(doc_id: str) -> Iterator[dict[str, Any]]:
    p = get_chunks_jsonl_path(doc_id)
    if not p.exists():
        raise FileNotFoundError("CHUNKS_NOT_FOUND")

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _batched(iterable, batch_size: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def embed_document_chunks(doc_id: str, svc: EmbeddingService) -> EmbedResult:
    """
    Reads chunks.jsonl
    batch-encode texts
    save embeddings.npy + meta
    """
    ensure_dir(get_embeddings_npy_path(doc_id).parent)

    meta_path = get_embeddings_meta_jsonl_path(doc_id)
    npy_path = get_embeddings_npy_path(doc_id)
    info_path = get_embeddings_info_path(doc_id)

    rows: list[np.ndarray] = []
    row_count = 0
    dim = 0

    # Possible runaway memory usage
    # accumulate in a list and vstack at the end
    # TODO: for huge docs -> memmap
    with meta_path.open("w", encoding="utf-8") as mf:
        for batch in _batched(_iter_chunks(doc_id), settings.EMBEDDING_BATCH_SIZE):
            texts = [str(it.get("text") or "") for it in batch]

            if row_count + len(texts) > settings.MAX_CHUNKS_TO_EMBED:
                raise ValueError("TOO_MANY_CHUNKS_TO_EMBED")

            emb = svc.encode_texts(texts)  # (B x D)
            if emb.ndim != 2:
                raise ValueError("INVALID_EMBEDDING_SHAPE")

            if dim == 0:
                dim = int(emb.shape[1])

            rows.append(emb)

            # meta lines matching embeddings row order
            for i, it in enumerate(batch):
                row_meta = {
                    "row": row_count + i,
                    "chunk_id": it.get("chunk_id"),
                    "doc_id": it.get("doc_id"),
                    "page": it.get("page"),
                    "chunk_index": it.get("chunk_index"),
                }
                mf.write(json.dumps(row_meta, ensure_ascii=False) + "\n")

            row_count += len(batch)

    matrix = np.vstack(rows).astype(np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
    np.save(npy_path, matrix)

    info = {
        "doc_id": doc_id,
        "row_count": row_count,
        "dim": dim,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "normalize": settings.EMBEDDING_NORMALIZE,
        "batch_size": settings.EMBEDDING_BATCH_SIZE,
        "chunking_version": _chunking_version(),
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
    """
    Prepare cache key for embeddings.
    embed:{model}:{chunking_version}:{sha256(text)}
    """
    hv = hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    return f"embed:{settings.EMBEDDING_MODEL_NAME}:{_chunking_version()}:{hv}"
