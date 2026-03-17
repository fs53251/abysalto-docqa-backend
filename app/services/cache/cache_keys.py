from __future__ import annotations

import hashlib
import re

from app.core.config import settings
from app.services.indexing.embed_chunks import chunking_version

# Utility for creating stable cache keys

# spaces, tabs, newlines
WS_RE = re.compile(r"\s+")

# \€ euro, [\d,]+ one or more digit or comma
# (?:\.\d+)? optional decimal part .40
RE_MONEY = re.compile(r"\€[\d,]+(?:\.\d+)?")
RE_YEAR = re.compile(r"\b(19|20)\d{2}\b")
RE_PERCENT = re.compile(r"\b\d+(\.\d+)?%")
RE_EMAIL = re.compile(r"\b\S+@\S+\b")
RE_NUMBER = re.compile(r"\b\d+(\.\d+)?\b")


def normalize_question(value: str) -> str:
    return WS_RE.sub(" ", (value or "").strip())


def mask_entities(question: str) -> str:
    masked = normalize_question(question)
    masked = RE_EMAIL.sub("[EMAIL]", masked)
    masked = RE_MONEY.sub("[AMOUNT]", masked)
    masked = RE_PERCENT.sub("[PERCENT]", masked)
    masked = RE_YEAR.sub("[YEAR]", masked)
    masked = RE_NUMBER.sub("[NUMBER]", masked)
    return masked


# Hashing helper
def sha256_hex(value: str) -> str:
    """
    Hash a string into SHA-256 hex string
    """
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


# Caching keys:


def qemb_key(question: str) -> str:
    """
    Cache for a question embedding.
    Caching id: embedding model and chunking version.
    """
    return f"qemb:{settings.EMBEDDING_MODEL_NAME}:{chunking_version()}:{sha256_hex(normalize_question(question))}"


def retr_key(
    scope: str, index_version: str, doc_id: str, question: str, top_k: int
) -> str:
    """
    Cache for retrieval results.
    Caching id: scope, doc_id, hash(q) and top-k
    """
    return f"retr:{scope}:{index_version}:{doc_id}:{sha256_hex(normalize_question(question))}:{top_k}"


def ans_key(scope: str, pipeline_version: str, question: str, top_k: int) -> str:
    """
    Cache for a final answer.
    Caching id: scope, pipeline version, hash(q) and top-k
    """
    return f"ans:{scope}:{pipeline_version}:{sha256_hex(normalize_question(question))}:{top_k}"


def sem_key(scope: str, pipeline_version: str, question: str, top_k: int) -> str:
    """
    Cache for semantic cache.
    Caching id: scope, pipeline version, hash(masked(q)) and top-k
    """
    return (
        f"sem:{scope}:{pipeline_version}:{sha256_hex(mask_entities(question))}:{top_k}"
    )
