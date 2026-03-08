from __future__ import annotations

import json
from pathlib import Path

from app.storage.files import ensure_dir, get_upload_root


def _index_path() -> Path:
    root = get_upload_root()
    ensure_dir(root)
    return root / "sha256_index.json"


def _read_index() -> dict[str, list[str]]:
    path = _index_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8") or "{}")
    normalized: dict[str, list[str]] = {}
    for sha256, value in raw.items():
        if isinstance(value, str):
            normalized[sha256] = [value]
        elif isinstance(value, list):
            normalized[sha256] = [str(item) for item in value if str(item).strip()]
    return normalized


def _write_index(data: dict[str, list[str]]) -> None:
    path = _index_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def find_existing_doc_ids(sha256: str) -> list[str]:
    return list(_read_index().get(sha256, []))


def find_existing_doc_id(sha256: str) -> str | None:
    doc_ids = find_existing_doc_ids(sha256)
    return doc_ids[0] if doc_ids else None


def find_reusable_doc_id(
    sha256: str, *, exclude_doc_id: str | None = None
) -> str | None:
    for doc_id in find_existing_doc_ids(sha256):
        if exclude_doc_id and doc_id == exclude_doc_id:
            continue
        return doc_id
    return None


def upsert_hash(sha256: str, doc_id: str) -> None:
    data = _read_index()
    current = data.get(sha256, [])
    if doc_id not in current:
        current.append(doc_id)
    data[sha256] = current
    _write_index(data)


def remove_doc_id(doc_id: str, sha256: str | None = None) -> None:
    data = _read_index()
    keys = [sha256] if sha256 else list(data.keys())
    changed = False
    for key in keys:
        doc_ids = [item for item in data.get(key, []) if item != doc_id]
        if doc_ids:
            data[key] = doc_ids
        elif key in data:
            del data[key]
        changed = True
    if changed:
        _write_index(data)
