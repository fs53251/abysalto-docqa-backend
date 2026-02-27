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


def _normalize(s: str) -> str:
    """
    I want to normalize, but also want to keep newlines for better splitting into chunks.
    Paragraphs are semantic units.
    """
    # Ensure we always return a string (never None)
    if s is None:
        return ""

    # replace null bytes (can appear in OCR / extracted text)
    s = str(s).replace("\x00", " ")
    s = s.strip()
    lines = [line.strip() for line in s.splitlines()]
    s = "\n".join(lines)

    return s


def _stable_chunk_id(doc_id: str, page: int, chunk_index: int, text: str) -> str:
    h = hashlib.sha256()

    # update chiper with some content from chunk
    # I am including a short slice of text
    # chunk_id -> hash sha256 of some content
    h.update(f"{doc_id}:{page}:{chunk_index}:".encode("utf-8"))

    h.update(text[:200].encode("utf-8", errors="ignore"))

    return h.hexdigest()[:24]


# Implementing my own logic for chunking, don't want to use langchain-text-splitters
# LangChain does not provide page, char_start, char_end, source or confidence!!


def _recursive_split(text: str | None, max_len: int, seps: tuple[str, ...]) -> list[str]:
    """
    Try separators in order.
    If still too long, fall back to hard split.

        - max_len: maximal chunk length, example: 1000 characters
    """
    if text is None:
        return []

    text = text.strip()
    if not text:
        return []

    if len(text) <= max_len:
        return [text]

    sep = seps[0]
    rest_seps = seps[1:]

    # Hard split fallback, chunks are whole max_len size documents
    if sep == "":
        parts = []
        i = 0
        while i < len(text):
            parts.append(text[i : i + max_len].strip())
            i += max_len

        return [p for p in parts if p]

    parts_raw = text.split(sep)

    # If split doesn't help, try next separator
    # example: if can not separate by newline, try separating
    # by sentences...
    if len(parts_raw) == 1:
        return _recursive_split(text, max_len, rest_seps)

    chunks: list[str] = []
    current = ""

    # Are those chunks valid (chunk size <= max_len)?
    # If not, recursive split until valid
    for part in parts_raw:
        part = part.strip()
        if not part:
            continue

        # accumulate chunks until size is less than max_len, then split by recursion
        candidate = part if not current else current + sep + part
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.extend(_recursive_split(current, max_len, rest_seps))
            current = part

    if current:
        chunks.extend(_recursive_split(current, max_len, rest_seps))

    return [c.strip() for c in chunks if c.strip()]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """
    Applies overlap by prefixing each chunk with tail of previous.
    First chunk is not included.
    Uses character overlap.
    """
    if not chunks or overlap <= 0:
        return chunks

    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = out[-1]
        tail = prev[-overlap:] if len(prev) > overlap else prev

        combined = (tail + " " + chunks[i]).strip()
        out.append(combined)

    return out


def build_chunks_for_doc(doc_id: str) -> tuple[list[Chunk], dict[str, Any]]:
    """
    Reads processed/{doc_id}/text.json
    Produces chunks + chunk_map
    """
    text_path = get_text_json_path(doc_id)
    if not text_path.exists():
        raise FileNotFoundError("TEXT_JSON_NOT_FOUND")

    # load text.json
    payload = json.loads(text_path.read_text(encoding="utf-8"))
    pages = payload.get("pages", [])
    if not isinstance(pages, list):
        raise ValueError("INVALID_TEXT_JSON")

    max_len = settings.CHUNK_SIZE_CHARS
    overlap = settings.CHUNK_OVERLAP_CHARS
    seps = settings.CHUNK_SEPARATORS

    all_chunks: list[Chunk] = []
    chunk_map: dict[str, Any] = {
        "doc_id": doc_id,
        "chunking": {
            "chunk_size_chars": max_len,
            "chunk_overlap_chars": overlap,
            "separators": list(seps),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": [],  # list of mapping entries
    }

    chunk_counter = 0

    for page_obj in pages:
        page_num = int(page_obj.get("page", 0) or 0)
        page_text = _normalize(str(page_obj.get("text", "") or ""))
        source = str(page_obj.get("source", "unknown") or "unknown")
        confidence = page_obj.get("confidence", None)

        if confidence is not None:
            try:
                confidence = float(confidence)
            except Exception:
                confidence = None

        if page_num <= 0:
            continue

        # Split within page
        base_chunk = _recursive_split(page_text, max_len=max_len, seps=seps)
        overlapped_chunks = _apply_overlap(base_chunk, overlap=overlap)

        # approximate char_start and char_end via sequential search
        cursor = 0
        for idx_in_page, ctext in enumerate(overlapped_chunks):
            if not ctext:
                continue

            # Find location of your chunk in page_text
            found = page_text.find(ctext.strip(), cursor)
            if found == -1:
                char_start = cursor
                char_end = min(cursor + len(ctext), len(page_text))
            else:
                char_start = found
                char_end = found + len(ctext)
                cursor = char_end

            chunk_id = _stable_chunk_id(doc_id, page_num, chunk_counter, ctext)

            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page=page_num,
                    chunk_index=chunk_counter,
                    text=ctext,
                    char_start=char_start,
                    char_end=char_end,
                    source=source,
                    confidence=confidence,
                )
            )

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


def save_chunks(doc_id: str, chunks: list[Chunk], chunk_map: dict[str, Any]) -> dict[str, str]:
    """
    Write chunks into 2 documents:
        1. chunks.jsonl
        2. chunk_map.json
    """
    chunks_path = get_chunks_jsonl_path(doc_id)
    map_path = get_chunk_map_path(doc_id)
    ensure_dir(chunks_path.parent)

    # Here I am writing chunks.jsonl (one JSON per line)
    # Using .jsonl for chunks:
    #   One JSON object per line
    #   No need to load whole document into RAM while reading
    #   Load into RAM batch of JSONS!!!
    with chunks_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": ch.chunk_id,
                        "doc_id": ch.doc_id,
                        "page": ch.page,
                        "chunk_index": ch.chunk_index,
                        "text": ch.text,
                        "char_start": ch.char_start,
                        "char_end": ch.char_end,
                        "source": ch.source,
                        "confidence": ch.confidence,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    map_path.write_text(json.dumps(chunk_map, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"chunks_jsonl": str(chunks_path), "chunk_map": str(map_path)}
