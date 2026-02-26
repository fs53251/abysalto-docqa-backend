from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from app.core.config import settings


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float | None
    lines: int


# I've implemented this for optimization, OCR is slow
@lru_cache(maxsize=1)
def get_easyocr_reader():
    """
    Singleton EasyOCR Reader.
    Expensive to have more than one reader!
    """
    import easyocr

    return easyocr.Reader(list(settings.EASYOCR_LANGS), gpu=settings.EASYOCR_GPU)


# buffered bytes, in-memory
def _io_bytes(b: bytes):
    import io

    return io.BytesIO(b)


def _safe_pil_open(image_bytes: bytes) -> Image.Image:
    img = Image.open(_io_bytes(image_bytes))
    img.load()

    if img.width * img.height > settings.MAX_IMAGE_PIXELS:
        raise ValueError("IMAGE_TOO_LARGE")

    return img.convert("RGB")


def ocr_image_bytes(image_bytes: bytes) -> OcrResult:
    """
    Runs EasyOCR on an image (bytes).

    Returns joined text + avg confidence.
    """
    img = _safe_pil_open(image_bytes)
    arr = np.array(img)

    # fetch singleton reader object
    reader = get_easyocr_reader()

    # EasyOCR returns something like this:
    # [(bbox, text, conf), ...]
    results = reader.readtext(arr)

    texts: list[str] = []
    confs: list[float] = []

    # item[0] -> bbox
    # item[1] -> text
    # item[2] -> confidence value
    for item in results:
        text = (item[1] or "").strip()
        conf = float(item[2] if item[2] is not None else None)

        if text:
            texts.append(text)
            if conf is not None:
                confs.append(conf)

    joined = "\n".join(texts).strip()
    avg_conf = (sum(confs) / len(confs)) if confs else None

    return OcrResult(text=joined, confidence=avg_conf, lines=len(texts))
