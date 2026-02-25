import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.extract import router as extract_router
from app.api.routes.health import router as health_router
from app.api.routes.upload import router as upload_router
from app.core.config import settings
from app.core.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_NAME)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(extract_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error", extra={"path": request.url.path})

    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
