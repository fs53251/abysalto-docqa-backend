from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import numpy as np

from app.core.config import settings
from app.services.indexing.faiss_index import load_faiss_index, search_index
from app.storage.chunks import get_chunks_jsonl_path
from app.storage.embeddings import get_embeddings_meta_jsonl_path

TOKEN_RE = re.compile(r"[a-z0-9]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    score: float
    page: int | None
    chunk_index: int | None
    text_snippet: str
    text: str | None = None
    semantic_score: float | None = None
    lexical_score: float | None = None
    combined_score: float | None = None


@dataclass(frozen=True)
class _ChunkRow:
    chunk_id: str
    page: int | None
    chunk_index: int | None
    text: str


def _query_terms(query: str) -> list[str]:
    terms = [
        token
        for token in TOKEN_RE.findall((query or "").lower())
        if token not in STOPWORDS
    ]
    return terms or TOKEN_RE.findall((query or "").lower())


def _load_row_to_chunk_id(doc_id: str) -> list[str]:
    path = get_embeddings_meta_jsonl_path(doc_id)
    if not path.exists():
        raise FileNotFoundError("EMBEDDINGS_META_NOT_FOUND")

    row_to_chunk: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            row = int(item["row"])
            chunk_id = str(item["chunk_id"])
            while len(row_to_chunk) < row:
                row_to_chunk.append("")
            if row == len(row_to_chunk):
                row_to_chunk.append(chunk_id)
            else:
                row_to_chunk[row] = chunk_id
    return row_to_chunk


def _load_chunk_map(doc_id: str) -> dict[str, _ChunkRow]:
    path = get_chunks_jsonl_path(doc_id)
    if not path.exists():
        raise FileNotFoundError("CHUNKS_NOT_FOUND")

    out: dict[str, _ChunkRow] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            chunk_id = str(item.get("chunk_id"))
            out[chunk_id] = _ChunkRow(
                chunk_id=chunk_id,
                page=int(item["page"]) if item.get("page") is not None else None,
                chunk_index=(
                    int(item["chunk_index"])
                    if item.get("chunk_index") is not None
                    else None
                ),
                text=str(item.get("text") or ""),
            )
    return out


def _lexical_score(text: str, query_terms: list[str]) -> float:
    if not text or not query_terms:
        return 0.0
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0

    token_counts: dict[str, int] = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    matched_terms = 0
    weighted_hits = 0.0
    for term in query_terms:
        count = token_counts.get(term, 0)
        if count > 0:
            matched_terms += 1
            weighted_hits += 1.0 + math.log1p(count)

    coverage = matched_terms / max(1, len(set(query_terms)))
    density = min(1.0, weighted_hits / max(3.0, len(tokens) / 24.0))
    return round((coverage * 0.7) + (density * 0.3), 4)


def _excerpt(text: str, query_terms: list[str], max_chars: int) -> str:
    clean = (text or "").strip()
    if not clean:
        return ""

    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_RE.split(clean)
        if sentence.strip()
    ]
    if not sentences:
        return clean[:max_chars].strip()

    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        score = _lexical_score(sentence, query_terms)
        scored.append((score, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    chosen: list[str] = []
    total = 0
    for _, sentence in scored[: settings.RETRIEVAL_MAX_SENTENCES_PER_CHUNK]:
        if sentence in chosen:
            continue
        extra = len(sentence) + (1 if chosen else 0)
        if total + extra > max_chars:
            break
        chosen.append(sentence)
        total += extra

    excerpt = " ".join(chosen).strip() or clean[:max_chars].strip()
    return excerpt[:max_chars].strip()


class RetrieverService:
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service

    def search(
        self,
        doc_id: str,
        query: str,
        top_k: int,
        query_emb: np.ndarray | None = None,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []

        top_k = max(1, min(int(top_k), settings.MAX_TOP_K))
        candidate_k = max(
            top_k,
            min(
                settings.MAX_TOP_K * settings.RETRIEVAL_CANDIDATE_MULTIPLIER,
                top_k * settings.RETRIEVAL_CANDIDATE_MULTIPLIER,
            ),
        )

        index = load_faiss_index(doc_id)
        row_to_chunk_id = _load_row_to_chunk_id(doc_id)
        chunk_map = _load_chunk_map(doc_id)
        terms = _query_terms(query)

        embedding = (
            query_emb
            if query_emb is not None
            else self.embedding_service.encode_texts([query])
        )
        scores, ids = search_index(index, embedding, candidate_k)

        results: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()

        for col in range(ids.shape[1]):
            row = int(ids[0, col])
            if row < 0 or row >= len(row_to_chunk_id):
                continue

            chunk_id = row_to_chunk_id[row]
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue

            semantic_score = float(scores[0, col])
            semantic_score = max(0.0, semantic_score)
            lexical_score = _lexical_score(chunk.text, terms)
            if (
                lexical_score < settings.RETRIEVAL_MIN_LEXICAL_SCORE
                and len(terms) >= 2
                and semantic_score < 0.12
            ):
                continue

            combined_score = round((semantic_score * 0.78) + (lexical_score * 0.22), 4)
            results.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=chunk.chunk_id,
                    score=combined_score,
                    page=chunk.page,
                    chunk_index=chunk.chunk_index,
                    text_snippet=_excerpt(
                        chunk.text, terms, settings.RETRIEVAL_EXCERPT_CHARS
                    ),
                    text=chunk.text,
                    semantic_score=round(semantic_score, 4),
                    lexical_score=lexical_score,
                    combined_score=combined_score,
                )
            )

        results.sort(
            key=lambda item: (
                item.combined_score or item.score,
                item.lexical_score or 0.0,
                item.semantic_score or 0.0,
            ),
            reverse=True,
        )
        return results[:top_k]
