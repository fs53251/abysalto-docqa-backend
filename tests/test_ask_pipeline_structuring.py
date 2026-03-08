from __future__ import annotations

from app.services.qa.ask_pipeline import answer_with_sources
from app.services.retrieval.retriever import RetrievedChunk


class NoisyDummyQA:
    def answer(self, question: str, context: str):
        class Result:
            answer = (
                "Total Due: 1,600.00 EUR Payment Terms: 7 days. VAT (25%) = 320.00 EUR "
                "Subtotal: 1,280.00 EUR Invoice No: NW-2026-021 Issue Date: 2026-02-26"
            )
            score = 0.91

        return Result()


class EmptyDummyQA:
    def answer(self, question: str, context: str):
        class Result:
            answer = ""
            score = None

        return Result()


def _invoice_sources() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            score=0.62,
            page=1,
            chunk_index=0,
            text_snippet=(
                "SERVICE INVOICE. Invoice No: NW-2026-021. Issue Date: 2026-02-26. "
                "Due Date: 2026-03-05."
            ),
            text=(
                "SERVICE INVOICE Northwind Studio Ltd. Invoice No: NW-2026-021. "
                "Issue Date: 2026-02-26. Due Date: 2026-03-05."
            ),
            semantic_score=0.62,
            lexical_score=0.55,
            combined_score=0.62,
        ),
        RetrievedChunk(
            doc_id="doc-1",
            chunk_id="chunk-2",
            score=0.71,
            page=1,
            chunk_index=1,
            text_snippet=(
                "Subtotal: 1,280.00 EUR. VAT (25%) = 320.00 EUR. Total Due: 1,600.00 EUR."
            ),
            text=(
                "Subtotal: 1,280.00 EUR. VAT (25%) = 320.00 EUR. Total Due: 1,600.00 EUR. "
                "Payment Terms: 7 days."
            ),
            semantic_score=0.71,
            lexical_score=0.67,
            combined_score=0.71,
        ),
    ]


def test_invoice_field_question_prefers_structured_direct_answer() -> None:
    result = answer_with_sources(
        question="What is the price of the total invoice?",
        sources=_invoice_sources(),
        qa=NoisyDummyQA(),
    )

    assert result.grounded is True
    assert result.answer == "The total amount due is 1,600.00 EUR."
    assert result.confidence_label == "high"


def test_invoice_purpose_question_falls_back_to_concise_business_use_answer() -> None:
    result = answer_with_sources(
        question="What could this document be used for?",
        sources=_invoice_sources(),
        qa=EmptyDummyQA(),
    )

    assert result.grounded is True
    assert "billing" in result.answer.lower()
    assert "bookkeeping" in result.answer.lower()
