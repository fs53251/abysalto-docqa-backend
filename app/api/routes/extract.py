import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.services.ingestion.pdf_text import extract_pdf_text_per_page, save_text_json
from app.storage.processed import get_text_json_path
from app.storage.upload_registry import get_original_file_path, read_metadata

router = APIRouter(tags=["documents"])

# doc_id is a lowercase hexadecimal string between 16 and 64 characters long
DOC_ID_RE = re.compile(r"^[a-f0-9]{16,64}$")


def _validate_doc_id(doc_id: str) -> None:
    if not DOC_ID_RE.match(doc_id):
        raise HTTPException(status_code=400, detail="Invalid doc_id format.")


@router.post("/documents/{doc_id}/extract-text")
def extract_text(doc_id: str, force: bool = Query(False)) -> JSONResponse:
    _validate_doc_id(doc_id)

    out_path = get_text_json_path(doc_id)
    if out_path.exists() and not force:
        return JSONResponse(
            status_code=200,
            content={"doc_id": doc_id, "status": "already_extracted", "text-path": str(out_path)},
        )

    try:
        md = read_metadata(doc_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Only PDF for now
    ct = (md.get("content_type") or "").lower()
    if ct != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF extraction is supported for now.")

    pdf_path: Path = get_original_file_path(doc_id)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Original file missing on disk.")

    try:
        extracted = extract_pdf_text_per_page(doc_id=doc_id, pdf_path=pdf_path)
        saved_path = save_text_json(extracted)
    except ValueError as e:
        msg = str(e)

        if msg.startswith("INVALID_PDF"):
            raise HTTPException(status_code=400, detail="Invalid or corrupted PDF.")
        if msg == "ENCRYPTED_PDF":
            raise HTTPException(status_code=400, detail="Encrypted PDF is not supported.")
        if msg == "PDF_TOO_MANY_PAGES":
            raise HTTPException(status_code=413, detail="PDF exceeds maximum allowed page count.")
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
    import json

    return json.loads(p.read_text(encoding="utf-8"))
