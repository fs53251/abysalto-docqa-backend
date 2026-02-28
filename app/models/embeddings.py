from pydantic import BaseModel


class EmbedBuildResponse(BaseModel):
    doc_id: str
    status: str
    row_count: int
    dim: int
    embeddings_npy: str
    embeddings_meta_jsonl: str
    embeddings_info: str
