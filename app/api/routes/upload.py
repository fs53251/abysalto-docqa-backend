from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.api.deps import CurrentIdentity, DbSession
from app.core.config import settings
from app.core.errors import InvalidInput, PayloadTooLarge, UnsupportedMediaType
from app.core.identifiers import document_public_id, generate_document_id
from app.models.upload import UploadItemResponse, UploadResponse
from app.repositories.documents import create_document
from app.storage.dedup import find_existing_doc_id, upsert_hash
from app.storage.files import read_first_bytes, save_upload_file_streaming, sniff_magic
from app.storage.metadata import write_metadata

router = APIRouter(tags=["documents"])


def _validate_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise InvalidInput(
            f"Unsupported file extension '{suffix}'. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )


def _validate_mime(upload_file: UploadFile) -> None:
    content_type = (upload_file.content_type or "").lower()
    if content_type not in settings.ALLOWED_MIME_TYPES:
        raise UnsupportedMediaType(
            f"Unsupported content type '{content_type}'. Allowed: {settings.ALLOWED_MIME_TYPES}"
        )


@router.post("/upload", response_model=UploadResponse)
async def upload(
    db: DbSession,
    identity: CurrentIdentity,
    files: list[UploadFile] = File(...),
) -> UploadResponse:
    if not files:
        raise InvalidInput("No files provided.")

    if len(files) > settings.MAX_FILES_PER_REQUEST:
        raise InvalidInput(
            f"Too many files. Max allowed: {settings.MAX_FILES_PER_REQUEST}."
        )

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    results: list[UploadItemResponse] = []
    has_errors = False

    for upload_file in files:
        filename = upload_file.filename or "file"

        try:
            _validate_extension(filename)
            _validate_mime(upload_file)

            first = await read_first_bytes(upload_file, 16)
            if not sniff_magic(upload_file.content_type or "", first):
                raise UnsupportedMediaType(
                    f"Magic-bytes verification failed for '{filename}'."
                )

            doc_uuid = generate_document_id()
            public_doc_id = document_public_id(doc_uuid)

            saved = await save_upload_file_streaming(
                upload_file=upload_file,
                doc_id=public_doc_id,
                max_bytes=max_bytes,
            )

            if settings.ENABLE_DEDUP:
                existing = find_existing_doc_id(saved.sha256)
                if existing is None:
                    upsert_hash(saved.sha256, public_doc_id)

            write_metadata(saved, magic_verified=True)

            owner_user_id = identity.user_id if identity.kind == "user" else None
            owner_session_id = (
                identity.session_id if identity.kind == "session" else None
            )

            create_document(
                db,
                doc_id=doc_uuid,
                filename=saved.original_filename,
                content_type=saved.content_type,
                size_bytes=saved.size_bytes,
                sha256=saved.sha256,
                stored_path=saved.stored_path,
                owner_user_id=owner_user_id,
                owner_session_id=owner_session_id,
                status="uploaded",
            )

            results.append(
                UploadItemResponse(
                    filename=saved.original_filename,
                    status="ok",
                    doc_id=public_doc_id,
                    content_type=saved.content_type,
                    size_bytes=saved.size_bytes,
                    sha256=saved.sha256,
                )
            )
        except (InvalidInput, UnsupportedMediaType, PayloadTooLarge) as exc:
            has_errors = True
            results.append(
                UploadItemResponse(
                    filename=filename,
                    status="error",
                    error_detail=str(exc),
                )
            )
        except ValueError as exc:
            has_errors = True
            if str(exc) == "FILE_TOO_LARGE":
                detail = f"File exceeds max size {settings.MAX_UPLOAD_MB} MB."
            else:
                detail = "Upload failed due to an internal validation error."
            results.append(
                UploadItemResponse(
                    filename=filename,
                    status="error",
                    error_detail=detail,
                )
            )

    return UploadResponse(documents=results, has_errors=has_errors)
