from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.api.deps import OwnedDocument
from app.core.config import settings
from app.core.errors import InternalError, InvalidInput, NotFound, PayloadTooLarge
from app.core.identifiers import document_public_id
from app.services.ingestion.image_text import extract_image_text, save_image_text_json
from app.services.ingestion.pdf_text import extract_pdf_text_per_page, save_text_json
from app.storage.processed import get_text_json_path
from app.storage.upload_registry import get_original_file_path, read_metadata

router = APIRouter(tags=["documents"])


@router.post("/documents/{doc_id}/extract-text")
def extract_text(
    document: OwnedDocument,
    force: bool = Query(False),
    ocr_fallback: bool = Query(True),
) -> JSONResponse:
    doc_id = document_public_id(document.id)
    out_path = get_text_json_path(doc_id)
    if out_path.exists() and not force:
        return JSONResponse(
            status_code=200,
            content={
                "doc_id": doc_id,
                "status": "already_extracted",
                "text_path": str(out_path),
            },
        )

    try:
        metadata = read_metadata(doc_id)
    except FileNotFoundError as exc:
        raise NotFound("Document not found.") from exc

    content_type = (metadata.get("content_type") or "").lower()
    original_path: Path = get_original_file_path(doc_id)
    if not original_path.exists():
        raise NotFound("Original file missing on disk.")

    if content_type == "application/pdf":
        try:
            extracted = extract_pdf_text_per_page(
                doc_id=doc_id,
                pdf_path=original_path,
                ocr_fallback=bool(ocr_fallback and settings.OCR_FALLBACK_ENABLED),
            )
            saved_path = save_text_json(extracted)
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

        return JSONResponse(
            status_code=200,
            content={
                "doc_id": doc_id,
                "status": "extracted",
                "page_count": extracted.page_count,
                "empty_pages": sum(1 for page in extracted.pages if page.is_empty),
                "text_path": str(saved_path),
            },
        )

    if content_type in {"image/png", "image/jpeg", "image/tiff"}:
        try:
            extracted_img = extract_image_text(doc_id=doc_id, image_path=original_path)
            saved_path = save_image_text_json(extracted_img)
        except ValueError as exc:
            if str(exc) == "IMAGE_TOO_LARGE":
                raise PayloadTooLarge("Image too large to OCR safely.") from exc
            raise InternalError("OCR failed unexpectedly.") from exc

        return JSONResponse(
            status_code=200,
            content={
                "doc_id": doc_id,
                "status": "extracted",
                "page_count": 1,
                "empty_pages": 1 if extracted_img.text.strip() == "" else 0,
                "text_path": str(saved_path),
            },
        )

    raise InvalidInput("Unsupported content type for extraction.")


@router.get("/documents/{doc_id}/text")
def get_text(document: OwnedDocument) -> JSONResponse:
    doc_id = document_public_id(document.id)
    path = get_text_json_path(doc_id)
    if not path.exists():
        raise NotFound("text.json not found. Run extraction first.")
    return JSONResponse(status_code=200, content=_json_load(path))


def _json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
