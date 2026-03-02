from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import settings
from app.services.qa.qa_service import QaResult, QAService
from app.services.retrieval.retriever import RetrievedChunk

WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class AskPipelineResult:
    answer: str
    confidence: float | None
    sources: list[RetrievedChunk]


def clean_question(q: str) -> str:
    q = (q or "").strip()
    q = WS_RE.sub(" ", q)

    return q


def truncate_context(ctx: str, max_chars: int) -> str:
    if len(ctx) <= max_chars:
        return ctx

    return ctx[:max_chars].rstrip()


def build_context(sources: list[RetrievedChunk]) -> str:
    """
    Context is deterministic and includes provenance headers
    so later I can highlight sources easily.
    """
    parts: list[str] = []

    for s in sources:
        header = f"[doc_id = {
            s.doc_id} page = {s.page} chunk_id = {s.chunk_id} score = {s.score:.4f}]"
        parts.append(header)
        parts.append(s.text_snippet)
        parts.append("")

    return "\n".join(parts).strip()


def answer_with_sources(
    *, question: str, sources: list[RetrievedChunk], qa: QAService
) -> AskPipelineResult:
    """
    QA over already-retrieved sources
    """
    q = clean_question(question)

    if not q:
        return AskPipelineResult(answer="", confidence=None, sources=[])

    if len(q) > settings.MAX_QUESTION_CHARS:
        q = q[: settings.MAX_QUESTION_CHARS]

    ctx = build_context(sources)
    ctx = truncate_context(ctx, settings.QA_MAX_CONTENT_CHARS)

    qa_res: QaResult = qa.answer(q, ctx)

    if not qa_res.answer or (qa_res.score is not None and qa_res.score < settings.QA_MIN_SCORE):
        return AskPipelineResult(
            answer="I don't know based on the provided documents.",
            confidence=qa_res.score,
            sources=sources,
        )

    return AskPipelineResult(answer=qa_res.answer, confidence=qa_res.score, sources=sources)
