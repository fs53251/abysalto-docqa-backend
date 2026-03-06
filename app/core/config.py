from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "DocQA API"

    APP_ENV: Literal["dev", "test", "prod"] = Field(default="dev", alias="ENV")
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["text", "json"] = "text"
    LOG_JSON_INCLUDE_EXC_INFO: bool = True

    DATA_DIR: str = "./data"
    UPLOAD_ROOT: str | None = None
    CACHE_ROOT: str | None = None

    JWT_SECRET: str = "e802c66c7ac92c1ab9ac994e41aa9923353d1283ae5129f9ce30a12245e63026"
    SESSION_COOKIE_SECRET: str = (
        "4e1dba6937f23b741e9cbf8796a4d4f1f04f4d0dbd47d7a4f7f5a5a9b1ef82ac"
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_EXP_MIN: int = 60
    PASSWORD_MIN_LENGTH: int = 8

    SESSION_COOKIE_NAME: str = "docqa_session"
    SESSION_TTL_DAYS: int = 7

    DATABASE_URL: str = "sqlite:///./data/app.db"

    CORS_ALLOW_ORIGINS: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    )
    TRUSTED_PROXIES: tuple[str, ...] = (
        "127.0.0.1",
        "::1",
    )

    MAX_UPLOAD_MB: int = 25
    MAX_FILES_PER_REQUEST: int = 10
    ALLOWED_EXTENSIONS: tuple[str, ...] = (
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    )
    ALLOWED_MIME_TYPES: tuple[str, ...] = (
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/tiff",
    )
    UPLOAD_AUTO_PROCESS: bool = True
    UPLOAD_PROCESSING_MODE: Literal["sync", "background"] = "sync"

    ENABLE_DEDUP: bool = True

    MAX_PDF_PAGES: int = 250
    TEXT_EMPTY_MIN_CHARS: int = 20

    EASYOCR_LANGS: tuple[str, ...] = ("en",)
    EASYOCR_GPU: bool = False
    OCR_FALLBACK_ENABLED: bool = True
    OCR_DPI: int = 200
    MAX_OCR_PAGES: int = 250
    MAX_IMAGE_PIXELS: int = 20000000

    CHUNK_SIZE_CHARS: int = 1000
    CHUNK_OVERLAP_CHARS: int = 200
    MAX_CHUNKS_PER_DOC: int = 5000
    CHUNK_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_NORMALIZE: bool = True
    MAX_CHUNKS_TO_EMBED: int = 5000

    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20

    QA_MODEL_NAME: str = "distilbert-base-cased-distilled-squad"
    QA_MAX_CONTENT_CHARS: int = 4000
    QA_MIN_SCORE: float = 0.15

    MAX_QUESTION_LEN: int = 2000
    MAX_QUESTION_CHARS: int = Field(default=2000, validation_alias="MAX_QUESTION_LEN")

    HF_TOKEN: str | None = None

    NER_MODEL_NAME: str = "en_core_web_sm"
    MAX_ENTITIES: int = 50

    REDIS_URL: str = "redis://localhost:6379/0"
    ENABLE_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 3600
    ENABLE_SEMANTIC_CACHE: bool = True
    SEMANTIC_CACHE_THRESHOLD: float = 0.75

    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    ASK_RATE_LIMIT_PER_MIN: int = 60
    UPLOAD_RATE_LIMIT_PER_MIN: int = 10
    LOGIN_RATE_LIMIT_PER_MIN: int = 10


settings = Settings()


def data_root() -> Path:
    return Path(settings.DATA_DIR)


def upload_root() -> Path:
    if settings.UPLOAD_ROOT:
        return Path(settings.UPLOAD_ROOT)
    return data_root() / "uploads"


def cache_root() -> Path:
    if settings.CACHE_ROOT:
        return Path(settings.CACHE_ROOT)
    return data_root() / "cache"


def processed_root() -> Path:
    return data_root() / "processed"
