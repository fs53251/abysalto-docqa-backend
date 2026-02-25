import json
from pathlib import Path
from typing import Any

from app.storage.files import get_upload_root


def get_metadata_path(doc_id: str) -> Path:
    return get_upload_root() / doc_id / "metadata.json"


def read_metadata(doc_id: str) -> dict[str, Any]:
    p = get_metadata_path(doc_id)
    if not p.exists():
        raise FileNotFoundError("DOC_NOT_FOUND")

    return json.loads(p.read_text(encoding="utf-8"))


def get_original_file_path(doc_id: str) -> Path:
    md = read_metadata(doc_id)

    stored_path = md.get("stored_path")
    if not stored_path:
        raise FileNotFoundError("MISSING_STORED_PATH")

    return Path(stored_path)
