from __future__ import annotations

import json

from fastapi import APIRouter, Query

from app.api.deps import EmbeddingSvc, OwnedDocument
from app.core.errors import InvalidInput, NotFound, PayloadTooLarge
from app.core.identifiers import document_public_id
from app.models.embeddings import EmbedBuildResponse
from app.services.indexing.embed_chunks import embed_document_chunks
from app.storage.embeddings import (
    get_embeddings_info_path,
    get_embeddings_meta_jsonl_path,
    get_embeddings_npy_path,
)

router = APIRouter(tags=["indexing"])


@router.post("/documents/{doc_id}/embed", response_model=EmbedBuildResponse)
def embed_document(
    document: OwnedDocument, emb_svc: EmbeddingSvc, force: bool = Query(False)
) -> EmbedBuildResponse:
    doc_id = document_public_id(document.id)
    npy_path = get_embeddings_npy_path(doc_id)
    meta_path = get_embeddings_meta_jsonl_path(doc_id)
    info_path = get_embeddings_info_path(doc_id)

    if npy_path.exists() and meta_path.exists() and info_path.exists() and not force:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        return EmbedBuildResponse(
            doc_id=doc_id,
            status="already_embedded",
            row_count=int(info.get("row_count") or 0),
            dim=int(info.get("dim") or 0),
            embeddings_npy=str(npy_path),
            embeddings_meta_jsonl=str(meta_path),
            embeddings_info=str(info_path),
        )

    try:
        result = embed_document_chunks(doc_id, emb_svc)
    except FileNotFoundError as exc:
        raise NotFound("chunks.jsonl not found. Run chunking first.") from exc
    except ValueError as exc:
        if str(exc) == "TOO_MANY_CHUNKS_TO_EMBED":
            raise PayloadTooLarge("Too many chunks to embed; adjust settings.") from exc
        raise InvalidInput("Embedding failed due to invalid input.") from exc

    return EmbedBuildResponse(
        doc_id=result.doc_id,
        status="embedded",
        row_count=result.row_count,
        dim=result.dim,
        embeddings_npy=result.embeddings_npy,
        embeddings_meta_jsonl=result.embeddings_meta_jsonl,
        embeddings_info=result.embeddings_info,
    )
