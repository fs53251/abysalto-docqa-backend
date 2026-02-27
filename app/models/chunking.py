from pydantic import BaseModel


class ChunkBuildResponse(BaseModel):
    doc_id: str
    status: str
    chunk_count: int
    chunks_jsonl: str
    chunk_map: str
