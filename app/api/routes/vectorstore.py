from __future__ import annotations

import json

from fastapi import APIRouter, Query

from app.api.deps import DbSession, EmbeddingSvc, OwnedDocument
from app.core.errors import InternalError, InvalidInput, NotFound
from app.core.identifiers import document_public_id
from app.models.retrieval import (
    BuildIndexResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.repositories.documents import mark_document_indexed
from app.services.indexing.faiss_index import build_faiss_index
from app.services.retrieval.retriever import RetrieverService
from app.storage.faiss_store import get_faiss_index_path, get_faiss_meta_path

router = APIRouter(tags=["indexing"])


@router.post("/documents/{doc_id}/index", response_model=BuildIndexResponse)
def build_index(
    document: OwnedDocument,
    db: DbSession,
    force: bool = Query(False),
) -> BuildIndexResponse:
    doc_id = document_public_id(document.id)
    idx_path = get_faiss_index_path(doc_id)
    meta_path = get_faiss_meta_path(doc_id)

    if idx_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mark_document_indexed(db, document=document)
        return BuildIndexResponse(
            doc_id=doc_id,
            status="already_indexed",
            dim=int(meta.get("dim") or 0),
            row_count=int(meta.get("row_count") or 0),
            index_path=str(idx_path),
            meta_path=str(meta_path),
        )

    try:
        result = build_faiss_index(doc_id)
    except FileNotFoundError as exc:
        raise NotFound("Embeddings not found. Run /embed first.") from exc
    except ValueError as exc:
        raise InvalidInput("Failed to build FAISS index.") from exc

    mark_document_indexed(db, document=document)
    return BuildIndexResponse(
        doc_id=doc_id,
        status="indexed",
        dim=result.dim,
        row_count=result.row_count,
        index_path=result.index_path,
        meta_path=result.meta_path,
    )


@router.post("/documents/{doc_id}/search", response_model=SearchResponse)
def search_doc(
    document: OwnedDocument,
    body: SearchRequest,
    emb_svc: EmbeddingSvc,
) -> SearchResponse:
    doc_id = document_public_id(document.id)
    retriever = RetrieverService(emb_svc)

    try:
        hits = retriever.search(doc_id=doc_id, query=body.query, top_k=body.top_k)
    except FileNotFoundError as exc:
        if "FAISS_INDEX_NOT_FOUND" in str(exc):
            raise NotFound("FAISS index not found. Run /index first.") from exc
        raise NotFound("Required artifacts missing.") from exc
    except Exception as exc:
        raise InternalError("Search failed unexpectedly.") from exc

    return SearchResponse(
        doc_id=doc_id,
        query=body.query,
        top_k=body.top_k,
        hits=[SearchHit(**hit.__dict__) for hit in hits],
    )
