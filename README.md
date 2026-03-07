# DocQA API

Backend service for document ingestion and grounded Q&A, built with FastAPI and Poetry.

## What is included

- upload and process PDF / image documents
- extract text, chunk, embed and index documents
- ask grounded questions against indexed documents
- auth with register / login / bearer token flow
- anonymous session flow backed by a stable cookie identity
- document ownership, listing, detail inspection and deletion
- caching, rate limiting and structured error responses
- Streamlit demo UI for showcasing the full product flow
- Docker + Compose setup for API, Redis and Postgres

## Local development

Requirements:

- Python 3.12+
- Poetry
- Redis

Install dependencies:

