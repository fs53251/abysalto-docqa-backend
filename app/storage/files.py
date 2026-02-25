import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from fastapi import UploadFile

from app.core.config import settings

FILENAME_SAFE_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class SavedFile:
    doc_id: str
    original_filename: str
    stored_filename: str
    content_type: str
    size_bytes: int
    sha256: str
    stored_path: str
    created_at: str


def sanitize_filename(filename: str) -> str:
    filename = os.path.basename((filename or "").strip())
    filename = FILENAME_SAFE_RE.sub("_", filename)
    return filename or "file"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_upload_root() -> Path:
    return Path(settings.DATA_DIR) / "uploads"


def _unique_dest_path(dest_dir: Path, stored_filename: str) -> Path:
    """
    If filename already exists, add a short suffix before extension.
    """
    candidate = dest_dir / stored_filename
    if not candidate.exists():
        return candidate

    # file123.txt -> stem(file123), suffix(.txt)
    # I need to add some random string between to avoid overwriting
    stem = candidate.stem
    suffix = candidate.suffix
    for _ in range(50):
        extra = uuid.uuid4().hex[:8]
        candidate2 = dest_dir / f"{stem}_{extra}{suffix}"
        if not candidate2.exists():
            return candidate2

    return dest_dir / f"{stem}_{uuid.uuid4().hex}{suffix}"


def sniff_magic(content_type: str, first_bytes: bytes) -> bool:
    """
    OWASP uploading document risk, prevent from spoofing
    Basic 'magic bytes' verification.
    Significantly better than trusting content_type alone.
    """
    ct = (content_type or "").lower()

    if ct == "application/pdf":
        return first_bytes.startswith(b"%PDF")

    if ct == "image/png":
        return first_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    if ct == "image/jpeg":
        return first_bytes.startswith(b"\xff\xd8\xff")

    if ct == "image/tiff":
        # TIFF: II*\x00 or MM\x00*
        return first_bytes.startswith(b"II*\x00") or first_bytes.startswith(b"MM\x00*")

    return False


async def read_first_bytes(upload_file: UploadFile, n: int = 16) -> bytes:
    """
    Read first n bytes and reset pointer.
    """
    await upload_file.seek(0)
    b = await upload_file.read(n)
    await upload_file.seek(0)
    return b


async def save_upload_file_streaming(
    *,
    upload_file: UploadFile,
    doc_id: str,
    max_bytes: int,
) -> SavedFile:
    """
    Streams UploadFile to disk in chunks, calculates sha256, enforces max_bytes.
    Uses atomic write: write to temp -> rename to final file.
    Avoids overwriting by making filename unique.
    """
    upload_root = get_upload_root()
    dest_dir = upload_root / doc_id / "original"
    ensure_dir(dest_dir)

    original_filename = upload_file.filename or "file"
    stored_filename = sanitize_filename(original_filename)

    final_path = _unique_dest_path(dest_dir, stored_filename)
    tmp_path = final_path.with_name(final_path.name + f".tmp_{uuid.uuid4().hex}")

    hasher = hashlib.sha256()
    total = 0
    chunk_size = 1024 * 1024  # 1MB

    await upload_file.seek(0)

    try:
        with tmp_path.open("wb") as f:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("FILE_TOO_LARGE")
                hasher.update(chunk)
                f.write(chunk)

        tmp_path.replace(final_path)

    except Exception:
        # remove tmp if exists
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        # remove final if partially created (shouldn't happen with replace, but safe)
        if final_path.exists() and final_path.stat().st_size == 0:
            final_path.unlink(missing_ok=True)
        raise
    finally:
        await upload_file.close()

    created_at = datetime.now(timezone.utc).isoformat()
    content_type = upload_file.content_type or "application/octet-stream"

    return SavedFile(
        doc_id=doc_id,
        original_filename=original_filename,
        stored_filename=final_path.name,
        content_type=content_type,
        size_bytes=total,
        sha256=hasher.hexdigest(),
        stored_path=str(final_path),
        created_at=created_at,
    )
