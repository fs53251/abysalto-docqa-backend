import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.services.ingestion.image_text import extract_image_text, save_image_text_json
from app.services.ingestion.pdf_text import extract_pdf_text_per_page, save_text_json
from app.storage.processed import get_text_json_path
from app.storage.upload_registry import get_original_file_path, read_metadata

router = APIRouter(tags=["documents"])

DOC_ID_RE = re.compile(r"^[a-f0-9]{16,64}$")


def _validate_doc_id(doc_id: str) -> None:
    if not DOC_ID_RE.match(doc_id):
        raise HTTPException(status_code=400, detail="Invalid doc_id format.")


@router.post("/documents/{doc_id}/extract-text")
def extract_text(
    doc_id: str,
    force: bool = Query(False),
    ocr_fallback: bool = Query(True),
) -> JSONResponse:
    _validate_doc_id(doc_id)

    out_path = get_text_json_path(doc_id)
    if out_path.exists() and not force:
        return JSONResponse(
            status_code=200,
            content={"doc_id": doc_id, "status": "already_extracted", "text_path": str(out_path)},
        )

    try:
        md = read_metadata(doc_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found.")

    ct = (md.get("content_type") or "").lower()

    original_path: Path = get_original_file_path(doc_id)
    if not original_path.exists():
        raise HTTPException(status_code=404, detail="Original file missing on disk.")

    # PDF: PyMuPDF per-page + optional OCR fallback
    if ct == "application/pdf":
        try:
            extracted = extract_pdf_text_per_page(
                doc_id=doc_id,
                pdf_path=original_path,
                ocr_fallback=bool(ocr_fallback and settings.OCR_FALLBACK_ENABLED),
            )
            saved_path = save_text_json(extracted)
        except ValueError as e:
            msg = str(e)
            if msg.startswith("INVALID_PDF"):
                raise HTTPException(status_code=400, detail="Invalid or corrupted PDF.")
            if msg == "ENCRYPTED_PDF":
                raise HTTPException(status_code=400, detail="Encrypted PDF is not supported.")
            if msg == "PDF_TOO_MANY_PAGES":
                raise HTTPException(
                    status_code=413, detail="PDF exceeds maximum allowed page count."
                )
            raise HTTPException(status_code=500, detail="Extraction failed unexpectedly.")

        return JSONResponse(
            status_code=200,
            content={
                "doc_id": doc_id,
                "status": "extracted",
                "page_count": extracted.page_count,
                "empty_pages": sum(1 for p in extracted.pages if p.is_empty),
                "text_path": str(saved_path),
            },
        )

    # Images: EasyOCR
    if ct in ("image/png", "image/jpeg", "image/tiff"):
        try:
            extracted_img = extract_image_text(doc_id=doc_id, image_path=original_path)
            saved_path = save_image_text_json(extracted_img)
        except ValueError as e:
            msg = str(e)
            if msg == "IMAGE_TOO_LARGE":
                raise HTTPException(status_code=413, detail="Image too large to OCR safely.")
            raise HTTPException(status_code=500, detail="OCR failed unexpectedly.")

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

    raise HTTPException(status_code=400, detail="Unsupported content type for extraction.")


@router.get("/documents/{doc_id}/text")
def get_text(doc_id: str) -> JSONResponse:
    _validate_doc_id(doc_id)

    p = get_text_json_path(doc_id)
    if not p.exists():
        raise HTTPException(
            status_code=404,
            detail="text.json not found. Run extraction first.",
        )

    return JSONResponse(status_code=200, content=json_load(p))


def json_load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))
