from __future__ import annotations

import hashlib
import re

from app.core.config import settings
from app.services.indexing.embed_cunks import _chunking_version

WS_RE = re.compile(r"\s+")
RE_MONEY = re.compile(r"\$[\d,]+(?:\.\d+)?")  # Money like "$1,234"
RE_YEAR = re.compile(r"\b(19|20)\d{2}\b")  # 4-digit years 1900-2099
RE_PERCENT = re.compile(r"\b\d+(\.\d+)?%")  # percentages "15%" or "12.5%"
RE_EMAIL = re.compile(r"\b\S+@\S+\b")  # email pattern "smt@gmail"
RE_NUMBER = re.compile(r"\b\d+(\.\d+)?\b")  # number "43" or "3.24"


def normalize_question(q: str) -> str:
    q = (q or "").strip()
    q = WS_RE.sub(" ", q)

    return q


def mask_entities(q: str) -> str:
    """
    Entity masking to reduce cache fragmentation when values change.
    Inspired by the semantic cache:
        replace variable values with placeholders.
    """
    q = normalize_question(q)
    q = RE_EMAIL.sub("[EMAIL]", q)
    q = RE_MONEY.sub("[AMOUNT]", q)
    q = RE_PERCENT.sub("[PERCENT]", q)
    q = RE_YEAR.sub("[YEAR]", q)
    q = RE_NUMBER.sub("[NUMBER]", q)

    return q


def sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def qemb_key(question: str) -> str:
    qn = normalize_question(question)

    return f"qemb:{settings.EMBEDDING_MODEL_NAME}:{
        _chunking_version()}:{sha256_hex(qn)}"


def retr_key(scope: str, index_version: str, doc_id: str, question: str, top_k: int) -> str:
    qn = normalize_question(question)

    return f"retr:{scope}:{index_version}:{doc_id}:{sha256_hex(qn)}:{top_k}"


def ans_key(scope: str, pipeline_version: str, question: str, top_k: int) -> str:
    qn = normalize_question(question)

    return f"ans:{scope}:{pipeline_version}:{
        sha256_hex(qn)}:{top_k}"


def sem_key(scope: str, pipeline_version: str, question: str, top_k: int) -> str:
    """
    Key for semantic chache bucket.
    exact matching is done on masked query
    """
    mq = mask_entities(question)

    return f"sem:{scope}:{pipeline_version}:{
        sha256_hex(mq)}:{top_k}"
