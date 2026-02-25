import json
from pathlib import Path
from typing import Optional

from app.storage.files import ensure_dir, get_upload_root


def _index_path() -> Path:
    """
    Returns the path to the SHA-256 index file.

    The index file stores a mapping:
        sha256_hash -> document_id

    The file is located inside the upload root directory:
        <upload_root>/sha256_index.json

    Ensures the upload root directory exists before returning the path.
    """
    root = get_upload_root()
    ensure_dir(root)
    return root / "sha256_index.json"


def find_existing_doc_id(sha256: str) -> Optional[str]:
    """
    Look up an existing document ID by its SHA-256 hash.

    Parameters
    ----------
    sha256 : str
        The SHA-256 hash of a file.

    Returns
    -------
    Optional[str]
        - The associated document ID if the hash exists.
        - None if the hash is not found or index file does not exist.

    Example
    -------
    >>> find_existing_doc_id("abc123")
    "doc_42"

    >>> find_existing_doc_id("nonexistent")
    None
    """
    p = _index_path()
    if not p.exists():
        return None

    data = json.loads(p.read_text(encoding="utf-8") or "{}")
    return data.get(sha256)


def upsert_hash(sha256: str, doc_id: str) -> None:
    """
    Insert or update a SHA-256 -> document ID mapping.

    If the hash already exists, its document ID will be overwritten.
    If it does not exist, it will be added.

    The update is written atomically:
        - Data is first written to a temporary file (.tmp)
        - The temporary file replaces the original index file

    This prevents corruption if the process crashes during write.

    Parameters
    ----------
    sha256 : str
        The SHA-256 hash of the file.
    doc_id : str
        The internal document ID associated with the file.

    Example
    -------
    >>> upsert_hash("abc123", "doc_42")

    The JSON file may then look like:
    {
        "abc123": "doc_42"
    }
    """
    p = _index_path()
    data = {}

    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8") or "{}")

    data[sha256] = doc_id

    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)
