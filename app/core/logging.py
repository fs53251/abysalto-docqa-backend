from __future__ import annotations

import json
import logging
import logging.config
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.core.request_context import get_identity, get_request_id


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class _LogContext:
    request_id: str | None
    identity: str | None


class ContextFilter(logging.Filter):
    """Inject request-scoped fields into LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 (filter)
        ctx = _LogContext(request_id=get_request_id(), identity=get_identity())
        # Do not overwrite if caller already set custom values
        record.request_id = getattr(record, "request_id", None) or ctx.request_id
        record.identity = getattr(record, "identity", None) or ctx.identity
        return True


class JsonFormatter(logging.Formatter):
    """A small JSON formatter without external dependencies."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _iso_utc_now(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "identity": getattr(record, "identity", None),
        }

        # Common extras that are useful for debugging
        for key in (
            "path",
            "method",
            "status_code",
            "latency_ms",
            "doc_id",
            "cache_hit",
            "layer",
            "sim",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if settings.LOG_JSON_INCLUDE_EXC_INFO and record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """Configure application logging.

    - Supports text or JSON logs.
    - Propagates request_id/identity via a Filter.
    - Harmonizes noisy third-party loggers.
    """

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Reduce noise
    for noisy in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
        "uvicorn.error",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if settings.LOG_FORMAT == "json":
        formatter = {
            "()": "app.core.logging.JsonFormatter",
        }
    else:
        formatter = {
            "format": (
                "%(asctime)s | %(levelname)s | %(name)s | "
                "rid=%(request_id)s | id=%(identity)s | %(message)s"
            )
        }

    cfg: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "context": {"()": "app.core.logging.ContextFilter"},
        },
        "formatters": {
            "default": formatter,
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "filters": ["context"],
                "formatter": "default",
                "level": level,
            }
        },
        "root": {
            "handlers": ["stdout"],
            "level": level,
        },
        # Keep uvicorn access logs aligned with app logs
        "loggers": {
            "uvicorn.access": {
                "handlers": ["stdout"],
                "level": level,
                "propagate": False,
            }
        },
    }

    logging.config.dictConfig(cfg)
