from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    doc_ids: list[str] | None = None
    scope: str | None = "all"  # "all" or "docs"
    top_k: int = 5


class AskSource(BaseModel):
    doc_id: str
    page: int | None
    chunk_id: str
    score: float
    text_excerpt: str


class AskResponse(BaseModel):
    answer: str
    confidence: float | None
    sources: list[AskSource]
    entities: list[dict]
