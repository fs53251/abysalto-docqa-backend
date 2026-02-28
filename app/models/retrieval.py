from pydantic import BaseModel


class BuildIndexResponse(BaseModel):
    doc_id: str
    status: str
    dim: int
    row_count: int
    index_path: str
    meta_path: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchHit(BaseModel):
    chunk_id: str
    score: float
    page: int | None
    chunk_index: int | None
    text_snippet: str


class SearchResponse(BaseModel):
    doc_id: str
    query: str
    top_k: int
    hits: list[SearchHit]
