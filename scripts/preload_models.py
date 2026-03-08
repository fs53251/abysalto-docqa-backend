from __future__ import annotations

import os
from pathlib import Path

from app.core.config import settings


def _ensure_dir(path: str | None) -> None:
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def _prepare_cache_dirs() -> None:
    _ensure_dir(os.getenv("HF_HOME"))
    _ensure_dir(os.getenv("TRANSFORMERS_CACHE"))
    _ensure_dir(os.getenv("SENTENCE_TRANSFORMERS_HOME"))
    _ensure_dir(os.getenv("TORCH_HOME"))
    _ensure_dir(os.getenv("EASYOCR_MODULE_PATH"))
    _ensure_dir(settings.CACHE_ROOT)


def preload_embedding_model() -> None:
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(
        settings.EMBEDDING_MODEL_NAME,
        cache_folder=os.getenv("SENTENCE_TRANSFORMERS_HOME"),
    )


def preload_spacy_model() -> None:
    import spacy
    from spacy.cli import download as spacy_download

    try:
        spacy.load(settings.NER_MODEL_NAME)
    except OSError:
        spacy_download(settings.NER_MODEL_NAME)
        spacy.load(settings.NER_MODEL_NAME)


def preload_easyocr_models() -> None:
    import easyocr

    easyocr.Reader(list(settings.EASYOCR_LANGS), gpu=settings.EASYOCR_GPU)


def main() -> None:
    _prepare_cache_dirs()
    preload_embedding_model()
    preload_spacy_model()
    preload_easyocr_models()
    print(
        "Model caches are ready. Answer generation uses the OpenAI API at request time."
    )


if __name__ == "__main__":
    main()
