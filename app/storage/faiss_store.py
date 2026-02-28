from pathlib import Path

from app.core.config import settings


def get_faiss_index_path(doc_id: str) -> Path:
    return Path(settings.DATA_DIR) / "processed" / doc_id / "faiss.index"


def get_faiss_meta_path(doc_id: str) -> Path:
    return Path(settings.DATA_DIR) / "processed" / doc_id / "faiss_meta.json"
