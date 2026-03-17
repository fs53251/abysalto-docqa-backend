import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes.ask import router as ask_router
from app.api.routes.auth import router as auth_router
from app.api.routes.chunk import router as chunk_router
from app.api.routes.documents import router as documents_router
from app.api.routes.embeddings import router as embeddings_router
from app.api.routes.extract import router as extract_router
from app.api.routes.health import router as health_router
from app.api.routes.upload import router as upload_router
from app.api.routes.vectorstore import router as vectorstore_router
from app.core.config import ensure_runtime_dirs, settings
from app.core.errors import DomainError
from app.core.exception_handlers import (
    domain_exception_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.core.logging import configure_logging
from app.core.middleware.access_logging import AccessLoggingMiddleware
from app.core.middleware.request_id import RequestIdMiddleware
from app.core.middleware.security_headers import SecurityHeadersMiddleware
from app.core.middleware.session_identity import SessionIdentityMiddleware
from app.services.factories import init_app_services

configure_logging()
logger = logging.getLogger(__name__)


# Runs before app starts serving requests
# startup/shutdown logic in one place!
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup logic...
    if settings.HF_TOKEN:
        os.environ["HF_TOKEN"] = settings.HF_TOKEN
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = settings.HF_TOKEN

    ensure_runtime_dirs()

    # DB initialization
    try:
        from app.db.session import init_db_dev_failsafe

        init_db_dev_failsafe()
        logger.info(
            "DB init ok (env=%s, url=%s)", settings.APP_ENV, settings.DATABASE_URL
        )
    except Exception as e:
        logger.exception("DB init failed: %s", e)

    init_app_services(app)

    # sleep
    yield

    # shutdown logic...


# FastAPI use my custom startup logic: lifespan
app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

# Instructions to browser, who is allowed to cal my API
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.CORS_ALLOW_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-Id"],
    expose_headers=["X-Request-Id", "Retry-After"],
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(SessionIdentityMiddleware)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(AccessLoggingMiddleware)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(documents_router)
app.include_router(extract_router)
app.include_router(chunk_router)
app.include_router(embeddings_router)
app.include_router(vectorstore_router)
app.include_router(ask_router)

app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(DomainError, domain_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)
