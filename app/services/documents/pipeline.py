from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.core.errors import DomainError, InternalError, InvalidInput, PayloadTooLarge
from app.core.identifiers import document_public_id, parse_document_public_id
from app.db.models import Document
from app.db.session import get_sessionmaker
from app.repositories.documents import (
    get_document,
    mark_document_failed,
    mark_document_indexed,
    mark_document_processing,
)
from app.services.indexing.chunking import build_chunks_for_doc, save_chunks
from app.services.indexing.embed_cunks import embed_document_chunks
from app.services.indexing.faiss_index import build_faiss_index
from app.services.ingestion.image_text import extract_image_text, save_image_text_json
from app.services.ingestion.pdf_text import extract_pdf_text_per_page, save_text_json
from app.services.interfaces import EmbeddingServicePort
from app.storage.upload_registry import get_original_file_path, read_metadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UploadProcessingResult:
    doc_id: str
    status: str
    error_detail: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    row_count: int | None = None
    dim: int | None = None


def _extract_document(*, document: Document) -> int:
    doc_id = document_public_id(document.id)

    try:
        metadata = read_metadata(doc_id)
    except FileNotFoundError as exc:
        raise InternalError("Upload metadata is missing.") from exc

    content_type = (metadata.get("content_type") or document.content_type or "").lower()
    original_path: Path = get_original_file_path(doc_id)
    if not original_path.exists():
        raise InternalError("Original uploaded file is missing on disk.")

    if content_type == "application/pdf":
        try:
            extracted = extract_pdf_text_per_page(
                doc_id=doc_id,
                pdf_path=original_path,
                ocr_fallback=bool(settings.OCR_FALLBACK_ENABLED),
            )
            save_text_json(extracted)
            return extracted.page_count
        except ValueError as exc:
            message = str(exc)
            if message.startswith("INVALID_PDF"):
                raise InvalidInput("Invalid or corrupted PDF.") from exc
            if message == "ENCRYPTED_PDF":
                raise InvalidInput("Encrypted PDF is not supported.") from exc
            if message == "PDF_TOO_MANY_PAGES":
                raise PayloadTooLarge(
                    "PDF exceeds maximum allowed page count."
                ) from exc
            raise InternalError("Extraction failed unexpectedly.") from exc

    if content_type in {"image/png", "image/jpeg", "image/tiff"}:
        try:
            extracted_img = extract_image_text(doc_id=doc_id, image_path=original_path)
            save_image_text_json(extracted_img)
            return 1
        except ValueError as exc:
            if str(exc) == "IMAGE_TOO_LARGE":
                raise PayloadTooLarge("Image too large to OCR safely.") from exc
            raise InternalError("OCR failed unexpectedly.") from exc

    raise InvalidInput("Unsupported content type for extraction.")


def _chunk_document(*, document: Document) -> int:
    doc_id = document_public_id(document.id)

    try:
        chunks, chunk_map = build_chunks_for_doc(doc_id)
    except FileNotFoundError as exc:
        raise InternalError("text.json not found after extraction.") from exc
    except ValueError as exc:
        if str(exc) == "TOO_MANY_CHUNKS":
            raise PayloadTooLarge(
                "Too many chunks generated; adjust settings."
            ) from exc
        raise InvalidInput("Invalid text.json format.") from exc

    save_chunks(doc_id, chunks, chunk_map)
    return len(chunks)


def _embed_document(
    *, document: Document, emb_svc: EmbeddingServicePort
) -> tuple[int, int]:
    doc_id = document_public_id(document.id)

    try:
        result = embed_document_chunks(doc_id, emb_svc)
    except FileNotFoundError as exc:
        raise InternalError("chunks.jsonl not found after chunking.") from exc
    except ValueError as exc:
        if str(exc) == "TOO_MANY_CHUNKS_TO_EMBED":
            raise PayloadTooLarge("Too many chunks to embed; adjust settings.") from exc
        raise InvalidInput("Embedding failed due to invalid input.") from exc

    return result.row_count, result.dim


def _index_document(*, document: Document) -> tuple[int, int]:
    doc_id = document_public_id(document.id)

    try:
        result = build_faiss_index(doc_id)
    except FileNotFoundError as exc:
        raise InternalError("Embeddings not found after embedding.") from exc
    except ValueError as exc:
        raise InvalidInput("Failed to build FAISS index.") from exc

    return result.row_count, result.dim


def process_uploaded_document(
    *,
    db,
    document: Document,
    emb_svc: EmbeddingServicePort,
) -> UploadProcessingResult:
    doc_id = document_public_id(document.id)

    page_count: int | None = None
    chunk_count: int | None = None
    row_count: int | None = None
    dim: int | None = None

    mark_document_processing(db, document=document)
    logger.info(
        "upload processing started",
        extra={
            "event": "upload.processing.started",
            "doc_id": doc_id,
            "document_filename": document.filename,
        },
    )

    try:
        page_count = _extract_document(document=document)
        chunk_count = _chunk_document(document=document)
        row_count, dim = _embed_document(document=document, emb_svc=emb_svc)
        row_count, dim = _index_document(document=document)
        mark_document_indexed(db, document=document)

        logger.info(
            "upload processing finished",
            extra={
                "event": "upload.processing.finished",
                "doc_id": doc_id,
                "pages": page_count,
                "chunks": chunk_count,
                "rows": row_count,
                "dim": dim,
            },
        )

        return UploadProcessingResult(
            doc_id=doc_id,
            status="indexed",
            page_count=page_count,
            chunk_count=chunk_count,
            row_count=row_count,
            dim=dim,
        )
    except DomainError as exc:
        mark_document_failed(db, document=document)
        logger.warning(
            "upload processing failed",
            extra={
                "event": "upload.processing.failed",
                "doc_id": doc_id,
                "pages": page_count,
                "chunks": chunk_count,
                "rows": row_count,
                "dim": dim,
                "outcome": exc.error_code,
            },
        )
        return UploadProcessingResult(
            doc_id=doc_id,
            status="failed",
            error_detail=exc.message,
            page_count=page_count,
            chunk_count=chunk_count,
            row_count=row_count,
            dim=dim,
        )
    except Exception:
        mark_document_failed(db, document=document)
        logger.exception(
            "upload processing failed unexpectedly",
            extra={
                "event": "upload.processing.failed",
                "doc_id": doc_id,
                "pages": page_count,
                "chunks": chunk_count,
                "rows": row_count,
                "dim": dim,
                "outcome": "unexpected_error",
            },
        )
        return UploadProcessingResult(
            doc_id=doc_id,
            status="failed",
            error_detail="Upload processing failed unexpectedly.",
            page_count=page_count,
            chunk_count=chunk_count,
            row_count=row_count,
            dim=dim,
        )


def process_uploaded_document_task(
    *,
    public_doc_id: str,
    emb_svc: EmbeddingServicePort,
) -> None:
    session_local = get_sessionmaker()
    db = session_local()

    try:
        parsed_doc_id = parse_document_public_id(public_doc_id)
        document = get_document(db, doc_id=parsed_doc_id)
        if document is None:
            logger.warning(
                "background upload pipeline skipped missing document",
                extra={
                    "event": "upload.processing.skipped",
                    "doc_id": public_doc_id,
                    "outcome": "missing_document",
                },
            )
            return

        process_uploaded_document(
            db=db,
            document=document,
            emb_svc=emb_svc,
        )
    finally:
        db.close()
