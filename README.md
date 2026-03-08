# Abysalto · AI-driven Document Insight Service

Production-style FastAPI backend for uploading PDFs/images, extracting text, indexing document content, and answering user questions over owned documents. The project also includes a small Streamlit UI, Redis-backed caching/rate limiting, JWT authentication, document ownership isolation, structured logging, Docker support, and CI.

## What this solution implements

### Core requirements
- `POST /upload` for one or more PDF/image documents
- Session-based and user-based document ownership
- Text extraction from:
  - native PDFs via **PyMuPDF**
  - scanned PDFs / images via **EasyOCR** fallback
- `POST /ask` to answer questions over uploaded documents
- Full Dockerization with API, UI, Redis, and Postgres services

### Optional enhancements included
- **RAG with embeddings + FAISS** for retrieval
- **NER** on generated answers and retrieved evidence
- **Redis caching** for answer and embedding reuse
- **JWT authentication**
- **Rate limiting**
- **Structured logging + request IDs**
- **Streamlit UI**
- **CI pipeline**

## Architecture

The application is intentionally split into a deterministic retrieval layer and an answer-synthesis layer:

1. **Ingestion**
   - upload validation
   - file persistence
   - metadata creation
   - ownership persistence

2. **Extraction**
   - text extraction from PDFs via PyMuPDF
   - OCR fallback for scans and images via EasyOCR

3. **Indexing**
   - chunking
   - local embeddings via Sentence Transformers
   - FAISS index generation

4. **Question answering**
   - retrieve top chunks for the active identity
   - build structured evidence
   - answer with either:
     - OpenAI synthesis when configured, or
     - heuristic grounded fallback from retrieved evidence when no OpenAI key is provided

5. **Security / operations**
   - JWT login flow
   - session cookies for guests
   - rate limiting via Redis
   - cache layer via Redis
   - structured logs with request IDs
   - health/readiness endpoints

## Why these tools

- **FastAPI**: clean API design, validation, async-friendly request handling
- **PyMuPDF**: reliable native PDF text extraction
- **EasyOCR**: simple OCR fallback for scans/images
- **Sentence Transformers + FAISS**: practical local retrieval stack with low operational complexity
- **spaCy**: lightweight NER enhancement
- **Redis**: good fit for short-lived cache entries and rate limiting counters
- **SQLAlchemy + Alembic**: mature persistence and migrations
- **Docker Compose**: reproducible local deployment

## Repository layout

```text
app/
  api/                # routes and DI dependencies
  core/               # config, errors, logging, middleware, security helpers
  db/                 # SQLAlchemy base, models, session
  models/             # request/response schemas
  repositories/       # DB access layer
  services/           # ingestion, indexing, retrieval, QA, cache, NER
  storage/            # file-system path helpers and metadata helpers
  main.py             # FastAPI application wiring
alembic/              # database migrations
scripts/              # cleanup, preloading, smoke testing
ui/                   # Streamlit demo frontend
data/test_docs/       # dummy/sample documents kept in the repo
```

## Environment management

`app/core/config.py` is the single source of truth for configuration.

Use these files as follows:

- `.env.example` → local/manual development template
- `.env.docker.example` → Docker Compose template
- `.env` → your untracked local secrets/config file

### Secrets policy

Never commit these values:
- `JWT_SECRET`
- `SESSION_COOKIE_SECRET`
- `OPENAI_API_KEY`
- database passwords
- any real production credentials

Safe-to-commit files:
- `.env.example`
- `.env.docker.example`
- `README.md`
- `docker-compose.yml`
- `app/core/config.py`

## Quick start — manual local run

### 1. Prerequisites
- Python 3.12+
- Poetry
- Redis running locally

### 2. Install dependencies

```bash
poetry install
```

### 3. Prepare environment

```bash
cp .env.example .env
```

For a completely local run without OpenAI, keep this:

```env
QA_USE_OPENAI="false"
OPENAI_API_KEY=""
```

That mode still answers questions using the built-in heuristic grounded fallback over retrieved evidence.

### 4. Run API

```bash
make run
```

Or directly:

```bash
poetry run uvicorn app.main:app --reload
```

### 5. Run UI

```bash
make run-ui
```

## Docker run

### 1. Prepare Docker environment

```bash
cp .env.docker.example .env
```

Then replace all placeholder secrets.

### 2. Start the stack

```bash
docker compose up --build
```

Services:
- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`

### 3. Stop the stack

```bash
docker compose down
```

## Health endpoints

- `GET /health`
- `GET /health/ready`

## API usage examples

### 1. Register user

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "demo@example.com",
    "password": "supersecret123"
  }'
```

Example response:

```json
{
  "id": "0c1f5c3b-3f3f-4d9d-b19e-5d0fcdcb5a9f",
  "email": "demo@example.com"
}
```

### 2. Login

```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "demo@example.com",
    "password": "supersecret123"
  }'
```

Example response:

```json
{
  "access_token": "<jwt>",
  "token_type": "bearer"
}
```

### 3. Upload document as guest session

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@data/test_docs/northwind_invoice_scanned.pdf" \
  -c cookies.txt
```

Example response:

```json
{
  "documents": [
    {
      "filename": "northwind_invoice_scanned.pdf",
      "status": "indexed",
      "status_detail": "Ready to ask.",
      "ready_to_ask": true,
      "doc_id": "doc_7f3df1d4c60745a995d1590b4a0c3f43",
      "content_type": "application/pdf",
      "size_bytes": 123456,
      "sha256": "...",
      "owner_type": "session",
      "error_detail": null
    }
  ],
  "has_errors": false
}
```

### 4. Ask a question over the active session documents

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{
    "question": "What is the invoice number?"
  }'
```

Example response:

```json
{
  "answer": "The invoice reference is INV-2025-014.",
  "confidence": 0.82,
  "confidence_label": "high",
  "grounded": true,
  "message": "Structured answer generated from retrieved document evidence.",
  "sources": [
    {
      "doc_id": "doc_7f3df1d4c60745a995d1590b4a0c3f43",
      "chunk_id": "chunk-0",
      "page": 1,
      "score": 0.94,
      "text_snippet": "Invoice Number: INV-2025-014 ..."
    }
  ],
  "entities": []
}
```

### 5. List owned documents

```bash
curl -X GET http://localhost:8000/documents -b cookies.txt
```

## Developer commands

```bash
make install
make test
make lint
make format
make check
make smoke
```

## Tests

Run:

```bash
poetry run pytest -q
```

At the time of preparing this final version, the test suite in this repository passes locally.

## Production-readiness improvements included in the final refactor

- removed tracked runtime state from the deliverable (`.env`, SQLite db, Redis dump, generated upload/index artifacts)
- clarified environment management with separate local and Docker templates
- improved `.gitignore` / `.dockerignore`
- made Docker startup more robust with optional migrations and dependency waits
- made OpenAI synthesis optional so the app can run without an API key
- added stricter configuration validation for production secrets and cookie security
- preserved sample documents in the repository for demonstration and testing

## Notes

- OpenAI is optional in this version.
- When OpenAI is disabled or not configured, answers still work through retrieval-backed heuristic synthesis.
- For the strongest answer fluency, set `QA_USE_OPENAI=true` and provide `OPENAI_API_KEY`.

## Demo documents

Stored in `data/test_docs/`:
- `northwind_invoice_scanned.pdf`
- `northwind_invoice_scanned.png`
- assignment PDF for reference
