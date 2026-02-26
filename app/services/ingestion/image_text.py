import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.ingestion.ocr import ocr_image_bytes
from app.storage.files import ensure_dir
from app.storage.processed import get_text_json_path


@dataclass(frozen=True)
class ExtractedImageText:
    doc_id: str
    text: str
    confidence: float | None


def extract_image_text(*, doc_id: str, image_path: Path) -> ExtractedImageText:
    image_bytes = image_path.read_bytes()
    ocr = ocr_image_bytes(image_bytes)

    return ExtractedImageText(doc_id=doc_id, text=ocr.text, confidence=ocr.confidence)


def save_image_text_json(extracted: ExtractedImageText) -> Path:
    out_path = get_text_json_path(extracted.doc_id)
    ensure_dir(out_path.parent)

    payload: dict[str, Any] = {
        "doc_id": extracted.doc_id,
        "page_count": 1,
        "pages": [
            {
                "page": 1,
                "text": extracted.text,
                "char_count": len(extracted.text),
                "is_empty": len(extracted.text.strip()) == 0,
                "source": "easyocr",
                "confidence": extracted.confidence,
            }
        ],
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_path
