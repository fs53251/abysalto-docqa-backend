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

# Logging configuration modul:
#   1) UTC timestamp helper
#   2) context object for log metadata
#   3) injects request_id and identity into every log record
#   4) custom JSON log formatter
#   5) logging based on app settings


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class _LogContext:
    request_id: str | None
    identity: str | None


# Filter inspect/modify each LogRecord before it is formatted and emitted.
class ContextFilter(logging.Filter):
    """
    Inject request-id and identity field into LogRecord.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        This method is called for each log entry!!!
        """
        ctx = _LogContext(request_id=get_request_id(), identity=get_identity())
        record.request_id = getattr(record, "request_id", None) or ctx.request_id
        record.identity = getattr(record, "identity", None) or ctx.identity
        return True


class JsonFormatter(logging.Formatter):
    """
    A small JSON formatter.
    Each log entry converts into JSON.

    Example: '20:21:01 | user123 | logged_in' converts
    into {date: '20:21:01', user_email: 'user123', ...}

    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _iso_utc_now(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "identity": getattr(record, "identity", None),
        }

        for key in (
            "event",
            "path",
            "method",
            "status_code",
            "latency_ms",
            "doc_id",
            "doc_ids_count",
            "source_count",
            "top_k",
            "cache_hit",
            "layer",
            "sim",
            "document_filename",
            "owner_type",
            "email_hash",
            "question_excerpt",
            "outcome",
            "pages",
            "chunks",
            "rows",
            "dim",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if settings.LOG_JSON_INCLUDE_EXC_INFO and record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


# logger -> filter -> formatter -> handler -> stdout
def configure_logging() -> None:
    """
    Build full logging configuration.
    """

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Warning levels: DEBUG -> INFO -> WARNING -> ERROR -> CRITICAL
    # Set next libraries from INFO -> WARNING!!!
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
        # Adding my custom logging filter, adds (req_id, identity)
        "filters": {
            "context": {"()": "app.core.logging.ContextFilter"},
        },
        # Adding my custom formatter (basic/JSON)
        "formatters": {
            "default": formatter,
        },
        # Handling logger that writes to stdout
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "filters": ["context"],
                "formatter": "default",
                "level": level,
            }
        },
        # The main default behaviour for my app
        "root": {
            "handlers": ["stdout"],
            "level": level,
        },
        "loggers": {
            # 127.0.0.1:12345 - "GET /health HTTP/1.1" 200
            # propagete=True -> passes records to root logger
            "uvicorn.access": {
                "level": level,
                "propagate": True,
            },
            "app.access": {
                "level": level,
                "propagate": True,
            },
        },
    }

    logging.config.dictConfig(cfg)
