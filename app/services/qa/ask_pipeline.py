from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import settings
from app.services.qa.qa_service import QaResult, QAService
from app.services.retrieval.retriever import RetrievedChunk

WS_RE = re.compile(r"\s+")
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
GENERIC_ANSWERS = {"[cls]", "[sep]", "[pad]", "the", "a", "an"}


@dataclass(frozen=True)
class AskPipelineResult:
    answer: str
    confidence: float | None
    grounded: bool
    message: str | None
    sources: list[RetrievedChunk]


@dataclass(frozen=True)
class _EvidenceSentence:
    source: RetrievedChunk
    sentence: str
    score: float


def clean_question(question: str) -> str:
    return WS_RE.sub(" ", (question or "").strip())


def _query_terms(question: str) -> list[str]:
    terms = [
        token for token in TOKEN_RE.findall(question.lower()) if token not in STOPWORDS
    ]
    return terms or TOKEN_RE.findall(question.lower())


def _split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_RE.split(text)
        if sentence.strip()
    ]


def _sentence_match_score(sentence: str, query_terms: list[str]) -> float:
    if not sentence or not query_terms:
        return 0.0
    normalized = sentence.lower()
    matched = sum(1 for term in set(query_terms) if term in normalized)
    coverage = matched / max(1, len(set(query_terms)))
    return round(coverage, 4)


def _collect_evidence(
    question: str, sources: list[RetrievedChunk]
) -> list[_EvidenceSentence]:
    query_terms = _query_terms(question)
    evidence: list[_EvidenceSentence] = []

    for source in sources:
        text = (source.text or source.text_snippet or "").strip()
        if not text:
            continue
        for sentence in _split_sentences(text)[:10]:
            sent_score = _sentence_match_score(sentence, query_terms)
            combined = round((sent_score * 0.7) + ((source.score or 0.0) * 0.3), 4)
            if combined <= 0:
                continue
            evidence.append(
                _EvidenceSentence(source=source, sentence=sentence, score=combined)
            )

    evidence.sort(key=lambda item: item.score, reverse=True)

    deduped: list[_EvidenceSentence] = []
    seen_sentences: set[str] = set()
    for item in evidence:
        key = item.sentence.lower()
        if key in seen_sentences:
            continue
        seen_sentences.add(key)
        deduped.append(item)
        if len(deduped) >= settings.QA_MAX_EVIDENCE_SENTENCES:
            break
    return deduped


def _extractive_candidate(
    question: str, sources: list[RetrievedChunk], qa: QAService
) -> QaResult | None:
    best: QaResult | None = None
    for source in sources[: settings.QA_EXTRACTIVE_DOC_LIMIT]:
        context = (source.text or source.text_snippet or "").strip()
        if not context:
            continue
        context = context[: settings.QA_MAX_CONTENT_CHARS]
        result = qa.answer(question, context)
        answer = (result.answer or "").strip()
        if (
            not answer
            or answer.lower() in GENERIC_ANSWERS
            or len(answer) < settings.QA_MIN_ANSWER_CHARS
        ):
            continue
        if best is None or (result.score or 0.0) > (best.score or 0.0):
            best = QaResult(answer=answer, score=result.score)
    return best


def _synthesize_answer(evidence: list[_EvidenceSentence]) -> str:
    chosen: list[str] = []
    for item in evidence:
        sentence = item.sentence.strip()
        if not sentence or sentence in chosen:
            continue
        chosen.append(sentence)
        if len(chosen) >= 3:
            break
    return " ".join(chosen).strip()


def answer_with_sources(
    *, question: str, sources: list[RetrievedChunk], qa: QAService
) -> AskPipelineResult:
    normalized_question = clean_question(question)
    if not normalized_question:
        return AskPipelineResult(
            answer="",
            confidence=None,
            grounded=False,
            message="Question must not be empty.",
            sources=[],
        )

    if len(normalized_question) > settings.MAX_QUESTION_CHARS:
        normalized_question = normalized_question[: settings.MAX_QUESTION_CHARS]

    if not sources:
        return AskPipelineResult(
            answer="I couldn't find any indexed document content to search.",
            confidence=0.0,
            grounded=False,
            message="No indexed sources were available for this question.",
            sources=[],
        )

    evidence = _collect_evidence(normalized_question, sources)
    evidence_score = evidence[0].score if evidence else 0.0
    extractive = _extractive_candidate(normalized_question, sources, qa)

    if (
        extractive
        and (extractive.score or 0.0) >= settings.QA_MIN_SCORE
        and evidence_score >= settings.QA_MIN_EVIDENCE_SCORE
    ):
        return AskPipelineResult(
            answer=extractive.answer.strip(),
            confidence=extractive.score,
            grounded=True,
            message=None,
            sources=sources,
        )

    if evidence and evidence_score >= settings.QA_MIN_EVIDENCE_SCORE:
        synthesized = _synthesize_answer(evidence)
        if synthesized:
            confidence = round(
                min(0.95, max(evidence_score, extractive.score if extractive else 0.0)),
                4,
            )
            return AskPipelineResult(
                answer=synthesized,
                confidence=confidence,
                grounded=True,
                message="Answer synthesized from the strongest matching passages.",
                sources=sources,
            )

    if evidence:
        partial = _synthesize_answer(evidence)
        return AskPipelineResult(
            answer=partial
            or "I found related passages, but they do not clearly answer the question.",
            confidence=round(evidence_score, 4) if evidence_score else 0.0,
            grounded=False,
            message="The retrieved passages were related, but the support was weak or incomplete.",
            sources=sources,
        )

    return AskPipelineResult(
        answer="I couldn't find support for that answer in the indexed documents.",
        confidence=0.0,
        grounded=False,
        message="Try narrowing the scope or asking about content that appears explicitly in the uploaded files.",
        sources=sources,
    )
