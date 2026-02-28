from pathlib import Path

from app.core.config import settings


def get_embeddings_npy_path(doc_id: str) -> Path:
    return Path(settings.DATA_DIR) / "processed" / doc_id / "embeddings.npy"


def get_embeddings_meta_jsonl_path(doc_id: str) -> Path:
    return Path(settings.DATA_DIR) / "processed" / doc_id / "embeddings_meta.jsonl"


def get_embeddings_info_path(doc_id: str) -> Path:
    return Path(settings.DATA_DIR) / "processed" / doc_id / "embeddings_info.json"
