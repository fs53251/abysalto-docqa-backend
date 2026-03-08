from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.db.session import check_db_connection

router = APIRouter(tags=["health"])


def _db_check() -> dict[str, object]:
    try:
        check_db_connection()
        return {"ready": True, "detail": "reachable"}
    except Exception as exc:
        return {
            "ready": False,
            "detail": "unreachable",
            "error": type(exc).__name__,
        }


def _redis_check(request: Request) -> dict[str, object]:
    if settings.APP_ENV == "test":
        return {"ready": True, "detail": "skipped in test env"}

    redis_needed = bool(settings.REDIS_URL) and (
        settings.ENABLE_CACHE or settings.ENABLE_RATE_LIMITING
    )
    if not redis_needed:
        return {"ready": True, "detail": "disabled by config"}

    client = getattr(request.app.state, "redis_client", None)
    if client is None:
        return {"ready": False, "detail": "not initialized"}

    try:
        client.ping()
        return {"ready": True, "detail": "reachable"}
    except Exception as exc:
        return {
            "ready": False,
            "detail": "unreachable",
            "error": type(exc).__name__,
        }


def _service_counts_as_ready(info: dict[str, object]) -> bool:
    if bool(info.get("ready")):
        return True

    detail = str(info.get("detail", ""))
    if settings.APP_ENV == "test" and detail == "skipped in test env":
        return True

    return False


def _service_checks(request: Request) -> tuple[bool, dict[str, object]]:
    statuses = getattr(request.app.state, "service_statuses", {})
    checks: dict[str, object] = {}
    required_ready = True

    for name in settings.HEALTH_READY_REQUIRED_SERVICES:
        info = statuses.get(name) or {"ready": False, "detail": "not initialized"}
        checks[name] = info
        required_ready = required_ready and _service_counts_as_ready(info)

    return required_ready, checks


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/health/ready")
def health_ready(request: Request):
    db = _db_check()
    redis = _redis_check(request)
    services_ready, service_checks = _service_checks(request)

    ready = bool(db["ready"]) and bool(redis["ready"]) and services_ready
    payload = {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "checks": {
            "db": db,
            "redis": redis,
            "services_initialized": {
                "ready": services_ready,
                "required": list(settings.HEALTH_READY_REQUIRED_SERVICES),
                "services": service_checks,
            },
        },
    }

    return JSONResponse(
        status_code=(
            status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        content=payload,
    )
