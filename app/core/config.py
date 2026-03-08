from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_SECRET_VALUES = {
    "CHANGE_ME",
    "CHANGE_ME_TOO",
    "CHANGE_ME_TO_A_LONG_RANDOM_SECRET",
    "CHANGE_ME_TO_ANOTHER_LONG_RANDOM_SECRET",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = "DocQA API"
    APP_ENV: Literal["dev", "test", "prod"] = Field(
        default="dev",
        validation_alias=AliasChoices("APP_ENV", "ENV"),
    )

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["text", "json"] = "text"
    LOG_JSON_INCLUDE_EXC_INFO: bool = False

    DATA_DIR: str = "./data"
    UPLOAD_ROOT: str | None = None
    CACHE_ROOT: str | None = None

    DATABASE_URL: str = "sqlite:///./data/app.db"

    SESSION_COOKIE_NAME: str = "docqa_session"
    SESSION_COOKIE_SECRET: str = "CHANGE_ME_TOO"
    SESSION_COOKIE_SECURE: bool = False
    SESSION_COOKIE_SAMESITE: Literal["lax", "strict", "none"] = "lax"
    SESSION_COOKIE_MAX_AGE_SECONDS: int = 7 * 24 * 60 * 60
    SESSION_TTL_DAYS: int | None = None

    JWT_SECRET: str = "CHANGE_ME"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60,
        validation_alias=AliasChoices(
            "JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
            "JWT_EXP_MIN",
        ),
    )
    JWT_EXP_MIN: int | None = None

    PASSWORD_MIN_LENGTH: int = 8

    CORS_ALLOW_ORIGINS: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    )
    TRUSTED_PROXIES: tuple[str, ...] = ("127.0.0.1", "::1")

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
    MAX_IMAGE_PIXELS: int = 20_000_000

    CHUNK_SIZE_CHARS: int = 1100
    CHUNK_OVERLAP_CHARS: int = 180
    MAX_CHUNKS_PER_DOC: int = 5000
    CHUNK_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")
    CHUNK_MIN_CHARS: int = 180

    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_NORMALIZE: bool = True
    MAX_CHUNKS_TO_EMBED: int = 5000

    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = 4
    RETRIEVAL_EXCERPT_CHARS: int = 420
    RETRIEVAL_MAX_SENTENCES_PER_CHUNK: int = 3
    RETRIEVAL_MIN_LEXICAL_SCORE: float = 0.05

    QA_MODEL_NAME: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("QA_MODEL_NAME", "OPENAI_ANSWER_MODEL"),
    )
    QA_USE_OPENAI: bool = True
    QA_MAX_CONTENT_CHARS: int = 4000
    QA_MIN_SCORE: float = 0.18
    QA_EXTRACTIVE_DOC_LIMIT: int = 2
    QA_MIN_EVIDENCE_SCORE: float = 0.20
    QA_MAX_EVIDENCE_SENTENCES: int = 5
    QA_MIN_ANSWER_CHARS: int = 3
    QA_MAX_SENTENCES: int = 4
    QA_SUMMARY_MAX_SENTENCES: int = 3
    QA_PURPOSE_MAX_SENTENCES: int = 2
    QA_FIELD_MAX_SENTENCES: int = 2

    MAX_QUESTION_CHARS: int = Field(
        default=2000,
        validation_alias=AliasChoices("MAX_QUESTION_CHARS", "MAX_QUESTION_LEN"),
    )
    MAX_QUESTION_LEN: int | None = None

    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    OPENAI_ORGANIZATION: str | None = None
    OPENAI_TIMEOUT_SECONDS: int = 60
    OPENAI_MAX_OUTPUT_TOKENS: int = 350

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

    HEALTH_READY_REQUIRED_SERVICES: tuple[str, ...] = ("embedding", "qa", "ner")

    STREAMLIT_API_BASE_URL: str = "http://127.0.0.1:8000"
    STREAMLIT_REQUEST_TIMEOUT_SEC: int = 60

    @model_validator(mode="after")
    def _sync_legacy_compat_fields(self) -> Settings:
        if self.JWT_EXP_MIN is None:
            self.JWT_EXP_MIN = self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        else:
            self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = self.JWT_EXP_MIN

        if self.SESSION_TTL_DAYS is None:
            self.SESSION_TTL_DAYS = max(1, self.SESSION_COOKIE_MAX_AGE_SECONDS // 86400)
        else:
            self.SESSION_COOKIE_MAX_AGE_SECONDS = self.SESSION_TTL_DAYS * 24 * 60 * 60

        if self.MAX_QUESTION_LEN is None:
            self.MAX_QUESTION_LEN = self.MAX_QUESTION_CHARS
        else:
            self.MAX_QUESTION_CHARS = self.MAX_QUESTION_LEN

        return self

    @model_validator(mode="after")
    def _validate_runtime_constraints(self) -> Settings:
        if self.CHUNK_OVERLAP_CHARS >= self.CHUNK_SIZE_CHARS:
            raise ValueError(
                "CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS"
            )
        if self.DEFAULT_TOP_K > self.MAX_TOP_K:
            raise ValueError("DEFAULT_TOP_K must be less than or equal to MAX_TOP_K")
        if self.MAX_FILES_PER_REQUEST < 1:
            raise ValueError("MAX_FILES_PER_REQUEST must be at least 1")
        if self.MAX_UPLOAD_MB < 1:
            raise ValueError("MAX_UPLOAD_MB must be at least 1")
        if self.SESSION_COOKIE_SAMESITE == "none" and not self.SESSION_COOKIE_SECURE:
            raise ValueError(
                "SESSION_COOKIE_SECURE must be true when SESSION_COOKIE_SAMESITE='none'"
            )

        if self.APP_ENV == "prod":
            if self.JWT_SECRET in _DEFAULT_SECRET_VALUES:
                raise ValueError("JWT_SECRET must be overridden in production")
            if self.SESSION_COOKIE_SECRET in _DEFAULT_SECRET_VALUES:
                raise ValueError(
                    "SESSION_COOKIE_SECRET must be overridden in production"
                )
            if not self.SESSION_COOKIE_SECURE:
                raise ValueError(
                    "SESSION_COOKIE_SECURE must be true in production deployments"
                )

        return self


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


def ensure_runtime_dirs() -> None:
    for path in (data_root(), upload_root(), processed_root(), cache_root()):
        path.mkdir(parents=True, exist_ok=True)
