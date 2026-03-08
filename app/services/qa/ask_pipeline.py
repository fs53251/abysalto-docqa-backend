from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

from app.core.config import settings
from app.services.interfaces import QaServicePort
from app.services.retrieval.retriever import RetrievedChunk

WS_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CODE_BLOCK_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
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
SUMMARY_PATTERNS = (
    re.compile(r"\bwhat(?:'s| is) this document about\b", re.I),
    re.compile(r"\bsummar(?:y|ize|ise)\b", re.I),
    re.compile(r"\boverview\b", re.I),
    re.compile(r"\bdescribe (?:this|the) document\b", re.I),
)
PURPOSE_PATTERNS = (
    re.compile(r"\bused for\b", re.I),
    re.compile(r"\buse of this document\b", re.I),
    re.compile(r"\bpurpose\b", re.I),
    re.compile(r"\bwhat can this document be used for\b", re.I),
    re.compile(r"\bwhy (?:would|might) (?:someone|a user) use\b", re.I),
)
FIELD_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "total_due": (
        re.compile(r"\btotal (?:invoice )?(?:amount|price|due|cost)\b", re.I),
        re.compile(r"\bamount due\b", re.I),
        re.compile(r"\btotal invoice\b", re.I),
    ),
    "subtotal": (
        re.compile(r"\bsubtotal\b", re.I),
        re.compile(r"\bbefore vat\b", re.I),
        re.compile(r"\bbefore tax\b", re.I),
    ),
    "vat": (
        re.compile(r"\bvat\b", re.I),
        re.compile(r"\btax\b", re.I),
    ),
    "due_date": (
        re.compile(r"\bdue date\b", re.I),
        re.compile(r"\bwhen is (?:the )?payment due\b", re.I),
    ),
    "issue_date": (
        re.compile(r"\bissue date\b", re.I),
        re.compile(r"\binvoice date\b", re.I),
    ),
    "invoice_number": (
        re.compile(r"\binvoice (?:number|no\.?|#)\b", re.I),
        re.compile(r"\breference number\b", re.I),
        re.compile(r"\breference\b", re.I),
    ),
    "currency": (
        re.compile(r"\bcurrency\b", re.I),
        re.compile(r"\bwhat currency\b", re.I),
    ),
}
GENERIC_ANSWERS = {"[cls]", "[sep]", "[pad]", "the", "a", "an"}
MoneyField = Literal["total_due", "subtotal", "vat"]
DateField = Literal["due_date", "issue_date"]
LookupField = Literal[
    "total_due",
    "subtotal",
    "vat",
    "due_date",
    "issue_date",
    "invoice_number",
    "currency",
]
QuestionIntent = Literal["summary", "purpose", "field_lookup", "freeform"]


@dataclass(frozen=True)
class AskPipelineResult:
    answer: str
    confidence: float | None
    confidence_label: str | None
    grounded: bool
    message: str | None
    sources: list[RetrievedChunk]


@dataclass(frozen=True)
class _EvidenceSentence:
    source: RetrievedChunk
    sentence: str
    score: float


@dataclass(frozen=True)
class InvoiceFields:
    looks_like_invoice: bool
    invoice_number: str | None = None
    reference_number: str | None = None
    issue_date: str | None = None
    due_date: str | None = None
    subtotal: str | None = None
    vat: str | None = None
    total_due: str | None = None
    currency: str | None = None
    vendor: str | None = None
    customer: str | None = None


def clean_question(question: str) -> str:
    return WS_RE.sub(" ", (question or "").strip())


def _clean_text(text: str) -> str:
    return WS_RE.sub(" ", (text or "").strip())


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


def _classify_intent(question: str) -> QuestionIntent:
    normalized = clean_question(question).lower()
    if any(pattern.search(normalized) for pattern in SUMMARY_PATTERNS):
        return "summary"
    if any(pattern.search(normalized) for pattern in PURPOSE_PATTERNS):
        return "purpose"
    if _requested_field(normalized) is not None:
        return "field_lookup"
    return "freeform"


def _requested_field(question: str) -> LookupField | None:
    normalized = clean_question(question).lower()
    for field_name, patterns in FIELD_PATTERNS.items():
        if any(pattern.search(normalized) for pattern in patterns):
            return field_name  # type: ignore[return-value]
    return None


def _extract_first(patterns: list[re.Pattern[str]], text: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return _clean_text(match.group(1))
    return None


def _format_amount(amount: str | None, currency: str | None = None) -> str | None:
    if not amount:
        return None
    cleaned_amount = re.sub(r"\s+", " ", amount).strip(" .,:;")
    cleaned_currency = (currency or "").strip().upper()
    if cleaned_currency and cleaned_currency not in cleaned_amount.upper():
        return f"{cleaned_amount} {cleaned_currency}".strip()
    return cleaned_amount


def _extract_money_field(label: str, text: str) -> str | None:
    patterns = [
        re.compile(
            rf"{label}\s*(?:\([^)]*\))?\s*[:=\-]?\s*([0-9][0-9., ]{{0,24}})\s*([A-Z]{{3}})?",
            re.I,
        ),
        re.compile(
            rf"{label}\s*(?:\([^)]*\))?\s*[:=\-]?\s*([A-Z]{{3}})\s*([0-9][0-9., ]{{0,24}})",
            re.I,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        first = _clean_text(match.group(1))
        second = (
            _clean_text(match.group(2))
            if match.lastindex and match.lastindex >= 2 and match.group(2)
            else None
        )
        if first.isalpha():
            return _format_amount(second, first)
        return _format_amount(first, second)
    return None


def _extract_invoice_fields(sources: list[RetrievedChunk]) -> InvoiceFields:
    combined_text = "\n".join(
        _clean_text(source.text or source.text_snippet or "")
        for source in sources
        if (source.text or source.text_snippet)
    )
    uppercase_text = combined_text.upper()
    looks_like_invoice = (
        sum(
            token in uppercase_text
            for token in ("INVOICE", "TOTAL DUE", "SUBTOTAL", "VAT", "PAYMENT TERMS")
        )
        >= 2
    )

    currency = _extract_first(
        [
            re.compile(r"\bCurrency\s*[:=]?\s*([A-Z]{3})\b", re.I),
            re.compile(r"\b([A-Z]{3})\s+Currency\b", re.I),
            re.compile(r"\bTotal Due\b.*?\b([A-Z]{3})\b", re.I),
        ],
        combined_text,
    )
    issue_date = _extract_first(
        [re.compile(r"\bIssue Date\s*[:=]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", re.I)],
        combined_text,
    )
    due_date = _extract_first(
        [re.compile(r"\bDue Date\s*[:=]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", re.I)],
        combined_text,
    )
    invoice_number = _extract_first(
        [
            re.compile(
                r"\bInvoice\s*(?:No\.?|Number|#)\s*[:=]?\s*([A-Z0-9][A-Z0-9\-/]{2,})",
                re.I,
            ),
            re.compile(r"\b([A-Z]{2,}-\d{4}-\d{2,})\b"),
        ],
        combined_text,
    )
    reference_number = _extract_first(
        [re.compile(r"\bReference\s*[:=#]?\s*([A-Z0-9][A-Z0-9\-/]{2,})", re.I)],
        combined_text,
    )
    vendor = _extract_first(
        [
            re.compile(r"\bSERVICE INVOICE\s+([^\n;]{3,80})", re.I),
            re.compile(r"\bFrom\s*[:=]?\s*([^\n;]{3,80})", re.I),
        ],
        combined_text,
    )
    customer = _extract_first(
        [
            re.compile(r"\bClient\s*[:=]?\s*([^\n;]{3,80})", re.I),
            re.compile(r"\bBill To\s*[:=]?\s*([^\n;]{3,80})", re.I),
        ],
        combined_text,
    )

    return InvoiceFields(
        looks_like_invoice=looks_like_invoice,
        invoice_number=invoice_number,
        reference_number=reference_number,
        issue_date=issue_date,
        due_date=due_date,
        subtotal=_extract_money_field("Subtotal", combined_text),
        vat=_extract_money_field("VAT", combined_text),
        total_due=_extract_money_field("Total Due", combined_text),
        currency=currency,
        vendor=vendor,
        customer=customer,
    )


def _field_value(fields: InvoiceFields, field_name: LookupField) -> str | None:
    if field_name == "invoice_number":
        return fields.invoice_number or fields.reference_number
    return getattr(fields, field_name, None)


def _format_field_answer(field_name: LookupField, value: str) -> str:
    templates = {
        "total_due": f"The total amount due is {value}.",
        "subtotal": f"The subtotal is {value}.",
        "vat": f"The VAT amount is {value}.",
        "due_date": f"The due date is {value}.",
        "issue_date": f"The issue date is {value}.",
        "invoice_number": f"The invoice reference is {value}.",
        "currency": f"The document uses {value} as the currency.",
    }
    return templates[field_name]


def _fallback_summary(fields: InvoiceFields) -> str:
    if fields.looks_like_invoice:
        details: list[str] = ["This document appears to be an invoice."]
        if fields.customer:
            details.append(f"It is addressed to {fields.customer}.")
        if fields.total_due:
            details.append(f"The total amount due is {fields.total_due}.")
        elif fields.invoice_number:
            details.append(f"The invoice reference is {fields.invoice_number}.")
        return " ".join(details[: settings.QA_SUMMARY_MAX_SENTENCES]).strip()
    return "This document contains business information, but the extracted evidence is limited for a better summary."


def _fallback_purpose(fields: InvoiceFields) -> str:
    if fields.looks_like_invoice:
        return (
            "It can be used for billing, payment processing, bookkeeping, and audit or "
            "compliance records."
        )
    return "It can be used as a reference document for the information captured in the file."


def _fallback_freeform(evidence: list[_EvidenceSentence]) -> str:
    chosen: list[str] = []
    for item in evidence:
        sentence = item.sentence.strip()
        if not sentence or sentence in chosen:
            continue
        chosen.append(sentence)
        if len(chosen) >= 2:
            break
    return " ".join(chosen).strip()


def _trim_sentences(text: str, *, max_sentences: int) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text.strip()
    return " ".join(sentences[:max_sentences]).strip()


def _answer_looks_like_dump(answer: str) -> bool:
    normalized = _clean_text(answer)
    if len(normalized) > 320:
        return True
    separators = normalized.count(":") + normalized.count(";") + normalized.count("|")
    digit_ratio = sum(char.isdigit() for char in normalized) / max(1, len(normalized))
    return separators >= 5 or digit_ratio > 0.22


def _sanitize_answer(answer: str) -> str:
    cleaned = CODE_BLOCK_RE.sub("", (answer or "").strip()).strip()
    return _clean_text(cleaned)


def _build_generation_context(
    *,
    intent: QuestionIntent,
    requested_field: LookupField | None,
    direct_answer: str | None,
    fields: InvoiceFields,
    evidence: list[_EvidenceSentence],
    sources: list[RetrievedChunk],
) -> str:
    if intent == "summary":
        style = "Return a concise summary in at most three sentences."
    elif intent == "purpose":
        style = "Return one or two sentences explaining what the document is used for."
    elif intent == "field_lookup":
        style = "Answer with the requested field first, in one sentence. Add one short follow-up sentence only if helpful."
    else:
        style = "Answer directly in one to three sentences using only supported facts from the evidence."

    field_dump = {key: value for key, value in asdict(fields).items() if value}
    evidence_lines = []
    for index, item in enumerate(
        evidence[: settings.QA_MAX_EVIDENCE_SENTENCES], start=1
    ):
        evidence_lines.append(
            f"[{index}] page={item.source.page or '-'} score={item.score} text={item.sentence}"
        )
    if not evidence_lines:
        for index, source in enumerate(sources[:3], start=1):
            snippet = _clean_text(source.text_snippet or source.text or "")
            if snippet:
                evidence_lines.append(
                    f"[{index}] page={source.page or '-'} score={source.score} text={snippet}"
                )

    document_type = "invoice" if fields.looks_like_invoice else "general_document"
    lines = [
        f"intent={intent}",
        f"document_type={document_type}",
        f"requested_field={requested_field or ''}",
        style,
        "Do not quote raw OCR blocks unless the user asked for a quote.",
        "Do not list every extracted field unless the user asked for details.",
        "If the answer is about a numeric field, keep the answer short and exact.",
    ]
    if direct_answer:
        lines.append(f"preferred_direct_answer={direct_answer}")
    if field_dump:
        lines.append(f"structured_fields={field_dump}")
    if evidence_lines:
        lines.append("evidence:")
        lines.extend(evidence_lines)
    return "\n".join(lines)


def _fallback_answer(
    *,
    intent: QuestionIntent,
    requested_field: LookupField | None,
    direct_answer: str | None,
    fields: InvoiceFields,
    evidence: list[_EvidenceSentence],
) -> str:
    if direct_answer:
        return direct_answer
    if intent == "summary":
        return _fallback_summary(fields)
    if intent == "purpose":
        return _fallback_purpose(fields)
    if intent == "field_lookup" and requested_field:
        return (
            "I found related invoice content, but not a reliable value for that field."
        )
    return (
        _fallback_freeform(evidence)
        or "I found related passages, but they do not clearly answer the question."
    )


def _confidence_from_evidence(
    *,
    evidence_score: float,
    direct_answer: str | None,
    intent: QuestionIntent,
    model_score: float | None = None,
) -> tuple[float, str]:
    base = max(evidence_score, float(model_score or 0.0))
    if direct_answer and intent == "field_lookup":
        base = max(base, 0.82)
    elif (
        intent in {"summary", "purpose"}
        and evidence_score >= settings.QA_MIN_EVIDENCE_SCORE
    ):
        base = max(base, 0.68)
    confidence = round(min(0.98, max(0.0, base)), 4)
    if confidence >= 0.75:
        label = "high"
    elif confidence >= 0.45:
        label = "medium"
    else:
        label = "low"
    return confidence, label


def answer_with_sources(
    *, question: str, sources: list[RetrievedChunk], qa: QaServicePort
) -> AskPipelineResult:
    normalized_question = clean_question(question)
    if not normalized_question:
        return AskPipelineResult(
            answer="",
            confidence=None,
            confidence_label=None,
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
            confidence_label="low",
            grounded=False,
            message="No indexed sources were available for this question.",
            sources=[],
        )

    intent = _classify_intent(normalized_question)
    requested_field = _requested_field(normalized_question)
    evidence = _collect_evidence(normalized_question, sources)
    evidence_score = evidence[0].score if evidence else 0.0
    fields = _extract_invoice_fields(sources)
    direct_answer = None
    if requested_field is not None:
        field_value = _field_value(fields, requested_field)
        if field_value:
            direct_answer = _format_field_answer(requested_field, field_value)

    context = _build_generation_context(
        intent=intent,
        requested_field=requested_field,
        direct_answer=direct_answer,
        fields=fields,
        evidence=evidence,
        sources=sources,
    )

    generated_answer = ""
    generated_score: float | None = None
    if evidence_score >= settings.QA_MIN_EVIDENCE_SCORE or direct_answer:
        try:
            generated = qa.answer(normalized_question, context)
            generated_answer = _sanitize_answer(generated.answer)
            generated_score = generated.score
        except Exception:
            generated_answer = ""
            generated_score = None

    answer = generated_answer or _fallback_answer(
        intent=intent,
        requested_field=requested_field,
        direct_answer=direct_answer,
        fields=fields,
        evidence=evidence,
    )

    if intent == "summary":
        answer = _trim_sentences(
            answer, max_sentences=settings.QA_SUMMARY_MAX_SENTENCES
        )
    elif intent == "purpose":
        answer = _trim_sentences(
            answer, max_sentences=settings.QA_PURPOSE_MAX_SENTENCES
        )
    elif intent == "field_lookup":
        answer = _trim_sentences(answer, max_sentences=settings.QA_FIELD_MAX_SENTENCES)
    else:
        answer = _trim_sentences(answer, max_sentences=settings.QA_MAX_SENTENCES)

    if _answer_looks_like_dump(answer):
        answer = _fallback_answer(
            intent=intent,
            requested_field=requested_field,
            direct_answer=direct_answer,
            fields=fields,
            evidence=evidence,
        )

    if not answer or answer.lower() in GENERIC_ANSWERS:
        answer = _fallback_answer(
            intent=intent,
            requested_field=requested_field,
            direct_answer=direct_answer,
            fields=fields,
            evidence=evidence,
        )

    grounded = bool(evidence_score >= settings.QA_MIN_EVIDENCE_SCORE or direct_answer)
    confidence, confidence_label = _confidence_from_evidence(
        evidence_score=evidence_score,
        direct_answer=direct_answer,
        intent=intent,
        model_score=generated_score,
    )

    if grounded:
        message = "Structured answer generated from retrieved document evidence."
    else:
        message = "The answer is based on limited supporting evidence from the retrieved passages."

    return AskPipelineResult(
        answer=answer,
        confidence=confidence,
        confidence_label=confidence_label,
        grounded=grounded,
        message=message,
        sources=sources,
    )
