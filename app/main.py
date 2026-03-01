import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.ask import router as ask_router
from app.api.routes.chunk import router as chunk_router
from app.api.routes.embeddings import router as embeddings_router
from app.api.routes.extract import router as extract_router
from app.api.routes.health import router as health_router
from app.api.routes.upload import router as upload_router
from app.api.routes.vectorstore import router as vectorstore_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.services.indexing.embedding_service import default_embedding_service
from app.services.qa.qa_service import default_qa_service

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize huggingface token
    if settings.HF_TOKEN:
        os.environ["HF_TOKEN"] = settings.HF_TOKEN
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = settings.HF_TOKEN

    # Initialize embedding model singleton once (fail-soft)
    try:
        svc = default_embedding_service()
        svc.load()
        app.state.embedding_service = svc
        logger.info("Embedding service loaded: %s", svc.cfg.model_name)
    except Exception as e:
        app.state.embedding_service = None
        logger.exception("Embedding model load failed (service disabled): %s", e)

    # Initialize QA model singleton once (fail-soft)
    try:
        qa = default_qa_service()
        qa.load()
        app.state.qa_service = qa
        logger.info("QA service loaded: %s", qa.model_name)
    except Exception as e:
        app.state.qa_service = None
        logger.exception("QA model load failed (service disabled): %s", e)

    yield


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(extract_router)
app.include_router(chunk_router)
app.include_router(embeddings_router)
app.include_router(vectorstore_router)
app.include_router(ask_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error", extra={"path": request.url.path})

    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
