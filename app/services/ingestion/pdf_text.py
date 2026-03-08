from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from app.core.config import settings
from app.core.errors import ExternalDependencyMissing
from app.services.ingestion.ocr import ocr_image_bytes
from app.storage.files import ensure_dir
from app.storage.processed import get_text_json_path

INLINE_WS_RE = re.compile(r"[ \t\x0b\x0c\r]+")
MULTI_BLANK_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class PageText:
    page: int
    text: str
    char_count: int
    is_empty: bool
    source: str
    confidence: float | None


@dataclass(frozen=True)
class ExtractedText:
    doc_id: str
    pages: list[PageText]
    page_count: int


def normalize_text(value: str) -> str:
    value = str(value or "").replace("\x00", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")

    paragraphs: list[str] = []
    current: list[str] = []

    for raw_line in value.split("\n"):
        line = INLINE_WS_RE.sub(" ", raw_line).strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        current.append(line)

    if current:
        paragraphs.append(" ".join(current).strip())

    normalized = "\n\n".join(paragraph for paragraph in paragraphs if paragraph)
    normalized = MULTI_BLANK_RE.sub("\n\n", normalized).strip()
    return normalized


def _render_page_png_bytes(page: fitz.Page, dpi: int) -> bytes:
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")


def extract_pdf_text_per_page(
    *, doc_id: str, pdf_path: Path, ocr_fallback: bool
) -> ExtractedText:
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"INVALID_PDF: {exc}") from exc

    if doc.is_encrypted:
        ok = doc.authenticate("")
        if not ok and doc.is_encrypted:
            doc.close()
            raise ValueError("ENCRYPTED_PDF")

    page_count = doc.page_count
    if page_count > settings.MAX_PDF_PAGES or page_count > settings.MAX_OCR_PAGES:
        doc.close()
        raise ValueError("PDF_TOO_MANY_PAGES")

    pages: list[PageText] = []
    for index in range(page_count):
        page = doc.load_page(index)
        extracted = normalize_text(page.get_text("text"))
        char_count = len(extracted)
        is_empty = char_count < settings.TEXT_EMPTY_MIN_CHARS

        if is_empty and ocr_fallback:
            png_bytes = _render_page_png_bytes(page, dpi=settings.OCR_DPI)
            try:
                ocr = ocr_image_bytes(png_bytes)
                ocr_text = normalize_text(ocr.text)
                pages.append(
                    PageText(
                        page=index + 1,
                        text=ocr_text,
                        char_count=len(ocr_text),
                        is_empty=len(ocr_text) < settings.TEXT_EMPTY_MIN_CHARS,
                        source="easyocr",
                        confidence=ocr.confidence,
                    )
                )
                continue
            except ExternalDependencyMissing:
                pass

        pages.append(
            PageText(
                page=index + 1,
                text=extracted,
                char_count=char_count,
                is_empty=is_empty,
                source="pymupdf",
                confidence=None,
            )
        )

    doc.close()
    return ExtractedText(doc_id=doc_id, pages=pages, page_count=page_count)


def save_text_json(extracted: ExtractedText) -> Path:
    out_path = get_text_json_path(extracted.doc_id)
    ensure_dir(out_path.parent)

    payload: dict[str, Any] = {
        "doc_id": extracted.doc_id,
        "page_count": extracted.page_count,
        "pages": [
            {
                "page": page.page,
                "text": page.text,
                "char_count": page.char_count,
                "is_empty": page.is_empty,
                "source": page.source,
                "confidence": page.confidence,
            }
            for page in extracted.pages
        ],
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path
