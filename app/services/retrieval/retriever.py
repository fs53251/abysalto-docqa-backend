from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.config import settings
from app.services.indexing.faiss_index import load_faiss_index, search_index
from app.storage.chunks import get_chunks_jsonl_path
from app.storage.embeddings import get_embeddings_meta_jsonl_path


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    score: float
    page: int | None
    chunk_index: int | None
    text_snippet: str


def _load_row_to_chunk_id(doc_id: str) -> list[str]:
    """
    embeddings_meta.jsonl defines row order.
    Return a list where index = row => chunk_id

    Faiss returns [(score1, 0), (score2, 1) ...]
    I don't have mapping from index to chunk_id!!!

    Example of meta_jsonl:
        {"row": 0, "chunk_id": "chunk_a", "doc_id": "...", "page": 1, "chunk_index": 0}
    """
    p = get_embeddings_meta_jsonl_path(doc_id)
    if not p.exists():
        raise FileNotFoundError("EMBEDDINGS_META_NOT_FOUND")

    row_to_id: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            row = int(obj["row"])
            cid = str(obj["chunk_id"])

            # ensure list size
            if row == len(row_to_id):
                row_to_id.append(cid)
            else:
                # handle out-of-order rows
                while len(row_to_id) < row:
                    row_to_id.append("")
                row_to_id.append(cid)

    return row_to_id


def _load_chunk_text_map(doc_id: str) -> dict[str, dict[str, Any]]:
    """
    load chunks.jsonl into memory for snippet lookup.
    """
    p = get_chunks_jsonl_path(doc_id)
    if not p.exists():
        raise FileNotFoundError("CHUNKS_NOT_FOUND")

    out: dict[str, dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out[str(obj["chunk_id"])] = obj

    return out


class RetrieverService:
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service

    def search(
        self, doc_id: str, query: str, top_k: int, query_emb: np.ndarray | None = None
    ) -> list[RetrievedChunk]:

        if not query or not query.strip():
            return []

        if top_k < 1:
            top_k = 1
        if top_k > settings.MAX_TOP_K:
            top_k = settings.MAX_TOP_K

        index = load_faiss_index(doc_id)
        row_to_id = _load_row_to_chunk_id(doc_id)
        chunk_map = _load_chunk_text_map(doc_id)

        if query_emb is None:
            q_emb: np.ndarray = self.embedding_service.encode_texts([query])  # (1, D)
        else:
            q_emb = query_emb

        scores, idx = search_index(index, q_emb, top_k)

        results: list[RetrievedChunk] = []

        for j in range(idx.shape[1]):
            row = int(idx[0, j])
            score = float(scores[0, j])

            if row < 0 or row >= len(row_to_id):
                continue
            chunk_id = row_to_id[row]
            if not chunk_id:
                continue

            ch = chunk_map.get(chunk_id, {})
            text = str(ch.get("text") or "")
            snippet = text[:400]

            results.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    score=score,
                    page=int(ch.get("page")) if ch.get("page") is not None else None,
                    chunk_index=(
                        int(ch.get("chunk_index")) if ch.get("chunk_index") is not None else None
                    ),
                    text_snippet=snippet,
                )
            )

        return results
