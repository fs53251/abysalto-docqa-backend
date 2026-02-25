from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.models.upload import UploadItemResponse, UploadResponse
from app.storage.dedup import find_existing_doc_id, upsert_hash
from app.storage.files import read_first_bytes, save_upload_file_streaming, sniff_magic
from app.storage.metadata import write_metadata

router = APIRouter(tags=["documents"])


def _validate_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{suffix}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )


def _validate_mime(upload_file: UploadFile) -> None:
    content_type = (upload_file.content_type or "").lower()
    if content_type not in settings.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type '{content_type}'. "
                f"Allowed: {settings.ALLOWED_MIME_TYPES}"
            ),
        )


@router.post("/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...)) -> UploadResponse:
    """
    Upload one or more files (PDF/images), validate, store on disk, return per-file results.

    Extra added features:
    - One bad file won't fail the whole request
    - size limit per file
    - max files per request
    - magic-bytes sniff
    - atomic write + safe filenames
    - optional sha256 deduplication index
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    if len(files) > settings.MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Max allowed: {settings.MAX_FILES_PER_REQUEST}.",
        )

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    results: list[UploadItemResponse] = []
    has_errors = False

    for f in files:
        filename = f.filename or "file"

        # Validation (per-file)
        try:
            _validate_extension(filename)
            _validate_mime(f)

            # magic sniff
            first = await read_first_bytes(f, 16)
            magic_ok = sniff_magic(f.content_type or "", first)
            if not magic_ok:
                raise HTTPException(
                    status_code=415,
                    detail=f"Magic-bytes verification failed for '{filename}'.",
                )

            # Save
            doc_id = uuid.uuid4().hex
            saved = await save_upload_file_streaming(
                upload_file=f,
                doc_id=doc_id,
                max_bytes=max_bytes,
            )

            # If deduplcation, I am keeping the newly saved file but also provide dedup mapping.
            if settings.ENABLE_DEDUP:
                existing = find_existing_doc_id(saved.sha256)
                if existing and existing != doc_id:
                    # Keep saved file, but return the existing doc_id as canonical
                    canonical_doc_id = existing
                else:
                    canonical_doc_id = doc_id
                    upsert_hash(saved.sha256, canonical_doc_id)
            else:
                canonical_doc_id = doc_id

            write_metadata(saved, magic_verified=True)

            results.append(
                UploadItemResponse(
                    filename=saved.original_filename,
                    status="ok",
                    doc_id=canonical_doc_id,
                    content_type=saved.content_type,
                    size_bytes=saved.size_bytes,
                    sha256=saved.sha256,
                )
            )

        except HTTPException as e:
            has_errors = True
            results.append(
                UploadItemResponse(
                    filename=filename,
                    status="error",
                    error_detail=str(e.detail),
                )
            )
        except ValueError as e:
            # file too large from storage layer
            has_errors = True
            if str(e) == "FILE_TOO_LARGE":
                results.append(
                    UploadItemResponse(
                        filename=filename,
                        status="error",
                        error_detail=f"File exceeds max size {settings.MAX_UPLOAD_MB} MB.",
                    )
                )
            else:
                results.append(
                    UploadItemResponse(
                        filename=filename,
                        status="error",
                        error_detail="Upload failed due to an internal validation error.",
                    )
                )

    return UploadResponse(documents=results, has_errors=has_errors)
