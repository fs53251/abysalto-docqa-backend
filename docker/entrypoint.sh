#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  /app/data/uploads \
  /app/data/processed \
  /app/data/cache \
  /cache/app \
  /cache/huggingface \
  /cache/huggingface/transformers \
  /cache/sentence-transformers \
  /cache/torch \
  /cache/easyocr

bool_env() {
  local name="$1"
  local default_value="${2:-false}"
  local raw="${!name:-$default_value}"
  raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  [[ "$raw" == "1" || "$raw" == "true" || "$raw" == "yes" || "$raw" == "on" ]]
}

wait_for_db() {
  if ! bool_env WAIT_FOR_DB true; then
    echo "DB wait skipped by WAIT_FOR_DB=false"
    return 0
  fi

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
  if ! bool_env WAIT_FOR_REDIS true; then
    echo "Redis wait skipped by WAIT_FOR_REDIS=false"
    return 0
  fi

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

print_qa_mode() {
  python - <<'PY'
import os

use_openai = os.getenv("QA_USE_OPENAI", "true").strip().lower() in {"1", "true", "yes", "on"}
api_key = bool(os.getenv("OPENAI_API_KEY", "").strip())

if use_openai and api_key:
    print("QA mode: OpenAI-backed answer synthesis enabled.")
elif use_openai and not api_key:
    print("QA mode: heuristic fallback only (OPENAI_API_KEY missing).")
else:
    print("QA mode: heuristic fallback only (QA_USE_OPENAI=false).")
PY
}

run_migrations_if_enabled() {
  if bool_env RUN_MIGRATIONS true; then
    alembic upgrade head
  else
    echo "Skipping migrations because RUN_MIGRATIONS=false"
  fi
}

wait_for_db
wait_for_redis
print_qa_mode
run_migrations_if_enabled

if bool_env PRELOAD_MODELS_ON_STARTUP false; then
  python scripts/preload_models.py
fi

exec "$@"
