from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "DocQA API"
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    DATA_DIR: str = "./data"
    JWT_SECRET: str = "CHANGE_ME"

    # Upload config
    MAX_UPLOAD_MB: int = 25
    MAX_FILES_PER_REQUEST: int = 10
    ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
    ALLOWED_MIME_TYPES: tuple[str, ...] = (
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/tiff",
    )

    # Deduplication using sha256
    ENABLE_DEDUP: bool = True

    # Extract text from PDF
    MAX_PDF_PAGES: int = 250
    TEXT_EMPTY_MIN_CHARS: int = 20  # heuristic: < 20 chars extracted

    # OCR
    EASYOCR_LANGS: tuple[str, ...] = ("en",)
    EASYOCR_GPU: bool = False

    OCR_FALLBACK_ENABLED: bool = True
    OCR_DPI: int = 200
    MAX_OCR_PAGES: int = 250
    MAX_IMAGE_PIXELS: int = 20000000

    # Chunking
    CHUNK_SIZE_CHARS: int = 1000
    CHUNK_OVERLAP_CHARS: int = 200
    MAX_CHUNKS_PER_DOC: int = 5000
    CHUNK_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    # Embedding
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_NORMALIZE: bool = True
    MAX_CHUNKS_TO_EMBED: int = 5000

    # FAISS / Retrieval
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20

    # QA
    QA_MODEL_NAME: str = "distilbert-base-cased-distilled-squad"
    QA_MAX_CONTENT_CHARS: int = 4000
    QA_MIN_SCORE: float = 0.15

    MAX_QUESTION_CHARS: int = 500

    HF_TOKEN: str | None = None

    # NER
    NER_MODEL_NAME: str = "en_core_web_sm"
    MAX_ENTITIES: int = 50

    # Redis / Cache
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600
    ENABLE_CACHE: bool = True

    # Semantic cache
    ENABLE_SEMANTIC_CACHE: bool = True
    SEMANTIC_CACHE_THRESHOLD: float = 0.75


settings = Settings()
