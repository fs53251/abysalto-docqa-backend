from pathlib import Path

from app.core.config import settings


def get_chunks_jsonl_path(doc_id: str) -> Path:
    return Path(settings.DATA_DIR) / "processed" / doc_id / "chunks.jsonl"


def get_chunk_map_path(doc_id: str) -> Path:
    return Path(settings.DATA_DIR) / "processed" / doc_id / "chunk_map.json"
