from pydantic import BaseModel


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    source: str  # "answer" | "chunk"
    doc_id: str | None = None
    page: int | None = None
    chunk_id: str | None = None
