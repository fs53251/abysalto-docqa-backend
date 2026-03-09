from __future__ import annotations

import hashlib
import re

# Safe logging and text processing:
#   - safe_excerpt - short readable preview
#   - hash_text    - non-readable stable identifier

# regex for: space, tab, newline
_WHITESPACE_RE = re.compile(r"\s+")


def safe_excerpt(text: str | None, *, max_chars: int = 120) -> str:
    """
    Normalize and cut text from chunk.
    Add '...' at the end.
    """
    if not text:
        return ""

    normalized = _WHITESPACE_RE.sub(" ", text).strip()
    if len(normalized) <= max_chars:
        return normalized

    if max_chars <= 1:
        return "…"

    return normalized[: max_chars - 1].rstrip() + "…"


def hash_text(value: str | None, *, prefix_len: int = 16) -> str:
    """
    Calculate hash from given text.
    SHA-256, and take only 'prefix_len' hex chars
    """
    normalized = (value or "").strip().lower()
    if not normalized:
        return "unknown"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:prefix_len]
