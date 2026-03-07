from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile

from app.api.deps import CurrentIdentity, DbSession, EmbeddingSvc
from app.core.config import settings
from app.core.errors import InvalidInput, PayloadTooLarge, UnsupportedMediaType
from app.core.identifiers import document_public_id, generate_document_id
from app.models.upload import UploadItemResponse, UploadResponse
from app.repositories.documents import create_document, mark_document_processing
from app.services.documents.pipeline import (
    process_uploaded_document,
    process_uploaded_document_task,
    try_reuse_processed_document,
)
from app.services.rate_limit import identity_rate_limit_key, rate_limit
from app.storage.dedup import find_existing_doc_ids, upsert_hash
from app.storage.files import read_first_bytes, save_upload_file_streaming, sniff_magic
from app.storage.metadata import write_metadata

router = APIRouter(tags=["documents"])
logger = logging.getLogger(__name__)

upload_rate_limit = rate_limit(
    limit=lambda: settings.UPLOAD_RATE_LIMIT_PER_MIN,
    window_seconds=lambda: settings.RATE_LIMIT_WINDOW_SECONDS,
    key_fn=identity_rate_limit_key("upload"),
)


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
    background_tasks: BackgroundTasks,
    db: DbSession,
    identity: CurrentIdentity,
    emb_svc: EmbeddingSvc,
    files: list[UploadFile] = File(...),
    _rate_limit: None = Depends(upload_rate_limit),
) -> UploadResponse:
    del _rate_limit

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

            first_bytes = await read_first_bytes(upload_file, 16)
            if not sniff_magic(upload_file.content_type or "", first_bytes):
                raise UnsupportedMediaType(
                    f"Magic-bytes verification failed for '{filename}'."
                )

            doc_uuid = generate_document_id()
            public_doc_id = document_public_id(doc_uuid)
            saved = await save_upload_file_streaming(
                upload_file=upload_file, doc_id=public_doc_id, max_bytes=max_bytes
            )
            write_metadata(saved, magic_verified=True)

            dedup_candidates: list[str] = []
            if settings.ENABLE_DEDUP:
                dedup_candidates = [
                    candidate
                    for candidate in reversed(find_existing_doc_ids(saved.sha256))
                    if candidate != public_doc_id
                ]
                upsert_hash(saved.sha256, public_doc_id)

            owner_user_id = identity.user_id if identity.kind == "user" else None
            owner_session_id = (
                identity.session_id if identity.kind == "session" else None
            )

            document = create_document(
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
            logger.info(
                "upload created",
                extra={
                    "event": "upload.created",
                    "doc_id": public_doc_id,
                    "document_filename": saved.original_filename,
                    "owner_type": identity.kind,
                },
            )

            item_status = "uploaded"
            error_detail: str | None = None
            status_detail = "Upload stored successfully."
            ready_to_ask = False

            if settings.UPLOAD_AUTO_PROCESS:
                reused = None
                for candidate_doc_id in dedup_candidates:
                    reused = try_reuse_processed_document(
                        db=db,
                        document=document,
                        source_doc_id=candidate_doc_id,
                    )
                    if reused is not None:
                        break

                if reused is not None:
                    item_status = reused.status
                    ready_to_ask = reused.status == "indexed"
                    status_detail = (
                        "Ready to ask. "
                        f"Reused processed artifacts from duplicate file {reused.reused_from_doc_id}."
                    )
                elif settings.UPLOAD_PROCESSING_MODE == "background":
                    mark_document_processing(db, document=document)
                    background_tasks.add_task(
                        process_uploaded_document_task,
                        public_doc_id=public_doc_id,
                        emb_svc=emb_svc,
                    )
                    item_status = "processing"
                    status_detail = (
                        "Upload accepted. Document is processing in the background."
                    )
                else:
                    process_result = process_uploaded_document(
                        db=db, document=document, emb_svc=emb_svc
                    )
                    item_status = process_result.status
                    error_detail = process_result.error_detail
                    ready_to_ask = process_result.status == "indexed"
                    if process_result.status == "failed":
                        has_errors = True
                    status_detail = (
                        "Ready to ask."
                        if process_result.status == "indexed"
                        else error_detail or "Processing did not complete."
                    )

            results.append(
                UploadItemResponse(
                    filename=saved.original_filename,
                    status=item_status,
                    status_detail=status_detail,
                    ready_to_ask=ready_to_ask,
                    doc_id=public_doc_id,
                    content_type=saved.content_type,
                    size_bytes=saved.size_bytes,
                    sha256=saved.sha256,
                    owner_type=identity.kind,
                    error_detail=error_detail,
                )
            )
        except (InvalidInput, UnsupportedMediaType, PayloadTooLarge) as exc:
            has_errors = True
            results.append(
                UploadItemResponse(
                    filename=filename,
                    status="error",
                    status_detail=str(exc),
                    owner_type=identity.kind,
                    error_detail=str(exc),
                )
            )
        except ValueError as exc:
            has_errors = True
            message = str(exc)
            if message == "FILE_TOO_LARGE":
                detail = f"File exceeds max size {settings.MAX_UPLOAD_MB} MB."
            elif message == "PATH_TRAVERSAL_DETECTED":
                detail = "Unsafe upload path detected."
            else:
                detail = "Upload failed due to an internal validation error."
            results.append(
                UploadItemResponse(
                    filename=filename,
                    status="error",
                    status_detail=detail,
                    owner_type=identity.kind,
                    error_detail=detail,
                )
            )

    return UploadResponse(documents=results, has_errors=has_errors)
