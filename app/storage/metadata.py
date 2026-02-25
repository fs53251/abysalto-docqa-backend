import json

from app.storage.files import SavedFile, ensure_dir, get_upload_root


def write_metadata(saved: SavedFile, *, magic_verified: bool) -> str:
    """
    Creates metadata.json next to the uploaded file (per doc_id)
    So I want structure like this one:
        ...data/uploads/file123/original
        ...data/uploads/file123/metadata.json
    """
    doc_root = get_upload_root() / saved.doc_id
    ensure_dir(doc_root)

    metadata_path = doc_root / "metadata.json"
    payload = {
        "doc_id": saved.doc_id,
        "original_filename": saved.original_filename,
        "stored_filename": saved.stored_filename,
        "content_type": saved.content_type,
        "size_bytes": saved.size_bytes,
        "sha256": saved.sha256,
        "stored_path": saved.stored_path,
        "magic_verified": magic_verified,
        "created_at": saved.created_at,
    }

    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return str(metadata_path)
