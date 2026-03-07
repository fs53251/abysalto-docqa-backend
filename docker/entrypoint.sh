#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  /app/data/uploads \
  /app/data/processed \
  /cache/app \
  /cache/huggingface \
  /cache/huggingface/transformers \
  /cache/sentence-transformers \
  /cache/torch \
  /cache/easyocr

wait_for_db() {
  python - <<'PY'
import os
import sys
import time

from sqlalchemy import create_engine, text

database_url = os.getenv("DATABASE_URL", "")
timeout = int(os.getenv("WAIT_FOR_DEPENDENCIES_TIMEOUT", "90"))

if not database_url or database_url.startswith("sqlite"):
    print("DB wait skipped.")
    raise SystemExit(0)

deadline = time.time() + timeout
last_error = None

while time.time() < deadline:
    try:
        engine = create_engine(database_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Database is reachable.")
        raise SystemExit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(2)

print(f"Database did not become reachable within {timeout}s: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

wait_for_redis() {
  python - <<'PY'
import os
import sys
import time

import redis

def is_enabled(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}

redis_url = os.getenv("REDIS_URL", "")
timeout = int(os.getenv("WAIT_FOR_DEPENDENCIES_TIMEOUT", "90"))
redis_needed = bool(redis_url) and (
    is_enabled("ENABLE_CACHE", "true") or is_enabled("ENABLE_RATE_LIMITING", "true")
)

if not redis_needed:
    print("Redis wait skipped.")
    raise SystemExit(0)

deadline = time.time() + timeout
last_error = None

while time.time() < deadline:
    try:
        client = redis.from_url(redis_url)
        client.ping()
        print("Redis is reachable.")
        raise SystemExit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(2)

print(f"Redis did not become reachable within {timeout}s: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

wait_for_db
wait_for_redis

alembic upgrade head

if [[ "${PRELOAD_MODELS_ON_STARTUP:-false}" == "true" ]]; then
  python scripts/preload_models.py
fi

exec "$@"