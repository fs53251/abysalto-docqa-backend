from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.core.identifiers import parse_document_public_id
from app.models.ner import Entity


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=settings.MAX_QUESTION_CHARS)
    doc_ids: list[str] | None = None
    scope: Literal["all", "docs"] = "all"
    top_k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=settings.MAX_TOP_K)

    @field_validator("question", mode="before")
    @classmethod
    def normalize_question(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("doc_ids")
    @classmethod
    def validate_doc_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None

        normalized: list[str] = []
        for item in value:
            doc_id = str(item).strip().lower()
            parse_document_public_id(doc_id)
            normalized.append(doc_id)

        return normalized


class AskSource(BaseModel):
    doc_id: str
    filename: str | None = None
    page: int | None = None
    chunk_id: str
    score: float
    semantic_score: float | None = None
    lexical_score: float | None = None
    text_excerpt: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool = True
    confidence: float | None = None
    message: str | None = None
    sources: list[AskSource]
    entities: list[Entity]
