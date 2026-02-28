import re

from fastapi import APIRouter, HTTPException, Query, Request

from app.models.embeddings import EmbedBuildResponse
from app.services.indexing.embed_cunks import embed_document_chunks
from app.storage.embeddings import (
    get_embeddings_info_path,
    get_embeddings_meta_jsonl_path,
    get_embeddings_npy_path,
)

router = APIRouter(tags=["indexing"])

DOC_ID_RE = re.compile(r"^[a-f0-9]{16,64}$")


def _validate_doc_id(doc_id: str) -> None:
    if not DOC_ID_RE.match(doc_id):
        raise HTTPException(status_code=400, detail="Invalid doc_id format.")


@router.post("/documents/{doc_id}/embed", response_model=EmbedBuildResponse)
def embed_document(
    request: Request,
    doc_id: str,
    force: bool = Query(False),
) -> EmbedBuildResponse:
    """
    Building embeddings for processed chunks.jsonl
    Creates embeddings.npy + embeddings_meta.jsonl + embeddings_info.json
    """
    _validate_doc_id(doc_id)

    npy_path = get_embeddings_npy_path(doc_id)
    meta_path = get_embeddings_meta_jsonl_path(doc_id)
    info_path = get_embeddings_info_path(doc_id)

    if npy_path.exists() and meta_path.exists() and info_path.exists() and not force:
        # read info quickly
        import json

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

    # use sungleton from app.state (which is loaded once at startup)
    svc = getattr(request.app.state, "embedding_service", None)
    if svc is None:
        raise HTTPException(status_code=500, detail="Embedding service not initialized.")

    try:
        res = embed_document_chunks(doc_id, svc)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="chunks.jsonl not found. Run chunking first.")
    except ValueError as e:
        if str(e) == "TOO_MANY_CHUNKS_TO_EMBED":
            raise HTTPException(
                status_code=413, detail="Too many chunks to embed; adjust settings."
            )
        raise HTTPException(status_code=400, detail="Embedding failed due to invalid input.")

    return EmbedBuildResponse(
        doc_id=res.doc_id,
        status="embedded",
        row_count=res.row_count,
        dim=res.dim,
        embeddings_npy=res.embeddings_npy,
        embeddings_meta_jsonl=res.embeddings_meta_jsonl,
        embeddings_info=res.embeddings_info,
    )
