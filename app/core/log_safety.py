from __future__ import annotations

import hashlib
import re

_WHITESPACE_RE = re.compile(r"\s+")


def safe_excerpt(text: str | None, *, max_chars: int = 120) -> str:
    if not text:
        return ""

    normalized = _WHITESPACE_RE.sub(" ", text).strip()
    if len(normalized) <= max_chars:
        return normalized

    if max_chars <= 1:
        return "…"

    return normalized[: max_chars - 1].rstrip() + "…"


def hash_text(value: str | None, *, prefix_len: int = 16) -> str:
    normalized = (value or "").strip().lower()
    if not normalized:
        return "unknown"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:prefix_len]
