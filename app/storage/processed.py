from pathlib import Path

from app.core.config import settings


def get_processed_root() -> Path:
    return Path(settings.DATA_DIR) / "processed"


def get_text_json_path(doc_id: str) -> Path:
    return get_processed_root() / doc_id / "text.json"
