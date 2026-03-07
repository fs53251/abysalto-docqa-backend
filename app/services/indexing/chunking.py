from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.storage.chunks import get_chunk_map_path, get_chunks_jsonl_path
from app.storage.files import ensure_dir
from app.storage.processed import get_text_json_path

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
MULTI_BLANK_RE = re.compile(r"\n{2,}")
WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    source: str
    confidence: float | None


def _normalize_page_text(value: str) -> str:
    value = str(value or "").replace("\x00", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [paragraph.strip() for paragraph in MULTI_BLANK_RE.split(value)]
    paragraphs = [
        WS_RE.sub(" ", paragraph).strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]
    return "\n\n".join(paragraphs).strip()


def _stable_chunk_id(doc_id: str, page: int, chunk_index: int, text: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"{doc_id}:{page}:{chunk_index}:".encode("utf-8"))
    hasher.update(text[:300].encode("utf-8", errors="ignore"))
    return hasher.hexdigest()[:24]


def _split_long_text(text: str, max_len: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if len(sentences) <= 1:
        words = text.split()
        chunks: list[str] = []
        current_words: list[str] = []
        current_len = 0
        for word in words:
            extra = len(word) + (1 if current_words else 0)
            if current_words and current_len + extra > max_len:
                chunks.append(" ".join(current_words))
                current_words = [word]
                current_len = len(word)
            else:
                current_words.append(word)
                current_len += extra
        if current_words:
            chunks.append(" ".join(current_words))
        return chunks

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if current and len(candidate) > max_len:
            chunks.extend(_split_long_text(current, max_len))
            current = sentence
        else:
            current = candidate
    if current:
        chunks.extend(
            _split_long_text(current, max_len) if len(current) > max_len else [current]
        )
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _paragraph_chunks(text: str, max_len: int) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if current and len(candidate) > max_len:
            chunks.append(current.strip())
            current = paragraph
        else:
            current = candidate

        if len(current) > max_len:
            chunks.extend(_split_long_text(current, max_len))
            current = ""

    if current:
        chunks.append(current.strip())

    normalized: list[str] = []
    for chunk in chunks:
        if len(chunk) < settings.CHUNK_MIN_CHARS and normalized:
            merged = f"{normalized[-1]}\n\n{chunk}".strip()
            if len(merged) <= max_len + settings.CHUNK_OVERLAP_CHARS:
                normalized[-1] = merged
                continue
        normalized.append(chunk)
    return normalized


def _tail_overlap(text: str, overlap: int) -> str:
    if overlap <= 0 or not text:
        return ""
    if len(text) <= overlap:
        return text.strip()
    tail = text[-overlap:]
    first_space = tail.find(" ")
    if first_space > 0:
        tail = tail[first_space + 1 :]
    return tail.strip()


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if not chunks or overlap <= 0:
        return chunks

    overlapped = [chunks[0]]
    for current in chunks[1:]:
        tail = _tail_overlap(overlapped[-1], overlap)
        merged = current if not tail else f"{tail}\n\n{current}".strip()
        overlapped.append(merged)
    return overlapped


def build_chunks_for_doc(doc_id: str) -> tuple[list[Chunk], dict[str, Any]]:
    text_path = get_text_json_path(doc_id)
    if not text_path.exists():
        raise FileNotFoundError("TEXT_JSON_NOT_FOUND")

    payload = json.loads(text_path.read_text(encoding="utf-8"))
    pages = payload.get("pages", [])
    if not isinstance(pages, list):
        raise ValueError("INVALID_TEXT_JSON")

    all_chunks: list[Chunk] = []
    chunk_map: dict[str, Any] = {
        "doc_id": doc_id,
        "chunking": {
            "chunk_size_chars": settings.CHUNK_SIZE_CHARS,
            "chunk_overlap_chars": settings.CHUNK_OVERLAP_CHARS,
            "chunk_min_chars": settings.CHUNK_MIN_CHARS,
            "separators": list(settings.CHUNK_SEPARATORS),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": [],
    }

    chunk_counter = 0
    for page_obj in pages:
        page_num = int(page_obj.get("page", 0) or 0)
        if page_num <= 0:
            continue

        page_text = _normalize_page_text(page_obj.get("text", ""))
        if not page_text:
            continue

        source = str(page_obj.get("source", "unknown") or "unknown")
        confidence_raw = page_obj.get("confidence")
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else None
        except (TypeError, ValueError):
            confidence = None

        base_chunks = _paragraph_chunks(page_text, max_len=settings.CHUNK_SIZE_CHARS)
        page_chunks = _apply_overlap(base_chunks, overlap=settings.CHUNK_OVERLAP_CHARS)

        cursor = 0
        for text in page_chunks:
            chunk_text = text.strip()
            if not chunk_text:
                continue

            found = page_text.find(chunk_text, cursor)
            if found == -1:
                char_start = cursor
                char_end = min(cursor + len(chunk_text), len(page_text))
            else:
                char_start = found
                char_end = found + len(chunk_text)
                cursor = char_end

            chunk_id = _stable_chunk_id(doc_id, page_num, chunk_counter, chunk_text)
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                page=page_num,
                chunk_index=chunk_counter,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                source=source,
                confidence=confidence,
            )
            all_chunks.append(chunk)
            chunk_map["chunks"].append(
                {
                    "chunk_id": chunk_id,
                    "page": page_num,
                    "chunk_index": chunk_counter,
                    "char_start": char_start,
                    "char_end": char_end,
                }
            )
            chunk_counter += 1
            if chunk_counter > settings.MAX_CHUNKS_PER_DOC:
                raise ValueError("TOO_MANY_CHUNKS")

    return all_chunks, chunk_map


def save_chunks(
    doc_id: str, chunks: list[Chunk], chunk_map: dict[str, Any]
) -> dict[str, str]:
    chunks_path = get_chunks_jsonl_path(doc_id)
    map_path = get_chunk_map_path(doc_id)
    ensure_dir(chunks_path.parent)

    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "page": chunk.page,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "source": chunk.source,
                        "confidence": chunk.confidence,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    map_path.write_text(
        json.dumps(chunk_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"chunks_jsonl": str(chunks_path), "chunk_map": str(map_path)}
