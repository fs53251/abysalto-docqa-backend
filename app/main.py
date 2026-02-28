import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.chunk import router as chunk_router
from app.api.routes.embeddings import router as embeddings_router
from app.api.routes.extract import router as extract_router
from app.api.routes.health import router as health_router
from app.api.routes.upload import router as upload_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.services.indexing.embedding_service import default_embedding_service

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize embedding model singleton once
    svc = default_embedding_service()
    svc.load()
    app.state.embedding_service = svc
    yield


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(extract_router)
app.include_router(chunk_router)
app.include_router(embeddings_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error", extra={"path": request.url.path})

    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
