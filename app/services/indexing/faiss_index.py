from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import faiss
import numpy as np

from app.core.config import settings
from app.services.indexing.embed_cunks import _chunking_version
from app.storage.embeddings import (
    get_embeddings_info_path,
    get_embeddings_meta_jsonl_path,
    get_embeddings_npy_path,
)
from app.storage.faiss_store import get_faiss_index_path, get_faiss_meta_path
from app.storage.files import ensure_dir


@dataclass(frozen=True)
class FaissBuildResult:
    doc_id: str
    dim: int
    row_count: int
    index_path: str
    meta_path: str


def _read_embeddings_info(doc_id: str) -> dict[str, Any]:
    p = get_embeddings_info_path(doc_id)
    if not p.exists():
        raise FileNotFoundError("EMBEDDINGS_INFO_NOT_FOUND")

    return json.loads(p.read_text(encoding="utf-8"))


def _load_embeddings_matrix(doc_id: str) -> np.ndarray:
    p = get_embeddings_npy_path(doc_id)
    if not p.exists():
        raise FileNotFoundError("EMBEDDINGS_NPY_NOT_FOUND")

    mat = np.load(p)
    if mat.ndim != 2:
        raise ValueError("INVALID_EMBEDDINGS_SHAPE")

    return mat.astype(np.float32)


def build_faiss_index(doc_id: str) -> FaissBuildResult:
    """
    Build FAISS index from embeddings.npy and save to disk.
    Uses IP if normalized embeddings, else L2.
    """
    ensure_dir(get_faiss_index_path(doc_id).parent)

    info = _read_embeddings_info(doc_id)
    row_count = int(info.get("row_count") or 0)
    dim = int(info.get("dim") or 0)
    normalize = bool(info.get("normalize", settings.EMBEDDING_NORMALIZE))
    model_name = str(info.get("embedding_model", settings.EMBEDDING_MODEL_NAME))

    mat = _load_embeddings_matrix(doc_id)
    if row_count != mat.shape[0]:
        row_count = mat.shape[0]
    if dim != mat.shape[1]:
        dim = mat.shape[1]

    if dim <= 0:
        raise ValueError("INVALID_DIM")

    # Choose index type (IP - inner product cosine, L2 - euclidian norm)
    if normalize:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(mat)

    index_path = get_faiss_index_path(doc_id)
    faiss.write_index(index, str(index_path))

    meta = {
        "doc_id": doc_id,
        "row_count": row_count,
        "dim": dim,
        "index_type": "IndexFlatIP" if normalize else "IndexFlatL2",
        "embedding_model": model_name,
        "normalize": normalize,
        "chunking_version": str(info.get("chunking_version") or _chunking_version()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "embeddings_npy": str(get_embeddings_npy_path(doc_id)),
            "embeddings_meta_jsonl": str(get_embeddings_meta_jsonl_path(doc_id)),
        },
    }

    meta_path = get_faiss_meta_path(doc_id)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return FaissBuildResult(
        doc_id=doc_id,
        dim=dim,
        row_count=row_count,
        index_path=str(index_path),
        meta_path=str(meta_path),
    )


def load_faiss_index(doc_id: str) -> faiss.Index:
    p = get_faiss_index_path(doc_id)
    if not p.exists():
        raise FileNotFoundError("FAISS_INDEX_NOT_FOUND")

    return faiss.read_index(str(p))


def search_index(
    index: faiss.Index, query_vec: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    query_vec: shape (1, D)
    returns: (scores, indices) both shape (1, top_k)
    """
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    query_vec = query_vec.astype(np.float32)

    scores, idx = index.search(query_vec, top_k)

    return scores, idx
