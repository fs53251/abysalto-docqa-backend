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


settings = Settings()
