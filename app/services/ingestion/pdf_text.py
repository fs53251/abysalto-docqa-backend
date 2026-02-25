import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from app.core.config import settings
from app.storage.files import ensure_dir
from app.storage.processed import get_text_json_path

# regex for tab or blank space
WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PageText:
    page: int
    text: str
    char_count: int
    is_empty: bool
    source: str  # "pymupdf"


@dataclass(frozen=True)
class ExtractedText:
    doc_id: str
    pages: list[PageText]
    page_count: int


def normalize_text(s: str) -> str:
    """
    Normalize a text string by cleaning whitespace and null characters.
    """
    s = s.replace("\x00", " ")
    s = WS_RE.sub(" ", s).strip()

    return s


def extract_pdf_text_per_page(*, doc_id: str, pdf_path: Path) -> ExtractedText:
    # Open PDF
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise ValueError(f"INVALID_PDF: {e}") from e

    # Handle encrypted PDFs
    if doc.is_encrypted:
        # Try empty password; if fails, reject
        ok = doc.authenticate("")
        if not ok and doc.is_encrypted:
            doc.close()
            raise ValueError("ENCRYPTED_PDF")

    page_count = doc.page_count
    if page_count > settings.MAX_PDF_PAGES:
        doc.close()
        raise ValueError("PDF_TOO_MANY_PAGES")

    pages: list[PageText] = []
    for i in range(page_count):
        page = doc.load_page(i)
        raw = page.get_text("text")
        text = normalize_text(raw)
        char_count = len(text)
        is_empty = char_count < settings.TEXT_EMPTY_MIN_CHARS

        pages.append(
            PageText(
                page=i + 1,
                text=text,
                char_count=char_count,
                is_empty=is_empty,
                source="pymupdf",
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
                "page": p.page,
                "text": p.text,
                "char_count": p.char_count,
                "is_empty": p.is_empty,
                "source": p.source,
            }
            for p in extracted.pages
        ],
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_path
