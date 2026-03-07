FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.4 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY scripts ./scripts
COPY ui ./ui

RUN ./.venv/bin/python -m spacy download en_core_web_sm

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    DATA_DIR=/app/data \
    CACHE_ROOT=/cache/app \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/cache/sentence-transformers \
    TORCH_HOME=/cache/torch \
    EASYOCR_MODULE_PATH=/cache/easyocr

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
        libstdc++6 \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /home/app app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/alembic.ini /app/alembic.ini
COPY --from=builder /app/alembic /app/alembic
COPY --from=builder /app/app /app/app
COPY --from=builder /app/scripts /app/scripts
COPY --from=builder /app/ui /app/ui
COPY --from=builder /app/README.md /app/README.md
COPY docker/entrypoint.sh /app/docker/entrypoint.sh

RUN mkdir -p /app/data/uploads /app/data/processed /cache/app /cache/huggingface /cache/sentence-transformers /cache/torch /cache/easyocr \
    && chmod +x /app/docker/entrypoint.sh \
    && chown -R app:app /app /cache /home/app

USER app

EXPOSE 8000

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
