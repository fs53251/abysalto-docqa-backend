from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.errors import ExternalDependencyMissing


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float | None
    lines: int


@dataclass(frozen=True)
class _OcrLine:
    text: str
    confidence: float | None
    top: float
    left: float


@lru_cache(maxsize=1)
def get_easyocr_reader():
    try:
        import easyocr
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ExternalDependencyMissing("easyocr") from exc

    return easyocr.Reader(list(settings.EASYOCR_LANGS), gpu=settings.EASYOCR_GPU)


def _io_bytes(payload: bytes):
    import io

    return io.BytesIO(payload)


def _safe_pil_open(image_bytes: bytes) -> Image.Image:
    img = Image.open(_io_bytes(image_bytes))
    img.load()

    if img.width * img.height > settings.MAX_IMAGE_PIXELS:
        raise ValueError("IMAGE_TOO_LARGE")

    return img.convert("RGB")


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_line(item: object) -> _OcrLine | None:
    if not isinstance(item, (list, tuple)) or len(item) < 3:
        return None

    bbox = item[0]
    text = str(item[1] or "").strip()
    confidence = _safe_float(item[2])
    if not text:
        return None

    try:
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
        top = min(ys)
        left = min(xs)
    except Exception:
        top = 0.0
        left = 0.0

    return _OcrLine(text=text, confidence=confidence, top=top, left=left)


def ocr_image_bytes(image_bytes: bytes) -> OcrResult:
    img = _safe_pil_open(image_bytes)
    arr = np.array(img)

    reader = get_easyocr_reader()
    raw_results = reader.readtext(arr)

    lines = [
        parsed for item in raw_results if (parsed := _parse_line(item)) is not None
    ]
    lines.sort(key=lambda line: (round(line.top / 12), line.top, line.left))

    texts = [line.text for line in lines]
    confs = [line.confidence for line in lines if line.confidence is not None]

    joined = "\n".join(texts).strip()
    avg_conf = (sum(confs) / len(confs)) if confs else None
    return OcrResult(text=joined, confidence=avg_conf, lines=len(texts))
