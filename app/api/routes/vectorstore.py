import re

from fastapi import APIRouter, Query

from app.api.deps import EmbeddingSvc
from app.core.errors import InternalError, InvalidInput, NotFound
from app.models.retrieval import (
    BuildIndexResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.services.indexing.faiss_index import build_faiss_index
from app.services.retrieval.retriever import RetrieverService
from app.storage.faiss_store import get_faiss_index_path, get_faiss_meta_path

router = APIRouter(tags=["indexing"])

DOC_ID_RE = re.compile(r"^[a-f0-9]{16,64}$")


def _validate_doc_id(doc_id: str) -> None:
    if not DOC_ID_RE.match(doc_id):
        raise InvalidInput("Invalid doc_id format.")


@router.post("/documents/{doc_id}/index", response_model=BuildIndexResponse)
def build_index(doc_id: str, force: bool = Query(False)) -> BuildIndexResponse:
    _validate_doc_id(doc_id)

    idx_path = get_faiss_index_path(doc_id)
    meta_path = get_faiss_meta_path(doc_id)
    if idx_path.exists() and meta_path.exists() and not force:
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return BuildIndexResponse(
            doc_id=doc_id,
            status="already_indexed",
            dim=int(meta.get("dim") or 0),
            row_count=int(meta.get("row_count") or 0),
            index_path=str(idx_path),
            meta_path=str(meta_path),
        )

    try:
        res = build_faiss_index(doc_id)
    except FileNotFoundError:
        raise NotFound("Embeddings not found. Run /embed first.")
    except ValueError:
        raise InvalidInput("Failed to build FAISS index.")

    return BuildIndexResponse(
        doc_id=doc_id,
        status="indexed",
        dim=res.dim,
        row_count=res.row_count,
        index_path=res.index_path,
        meta_path=res.meta_path,
    )


@router.post("/documents/{doc_id}/search", response_model=SearchResponse)
def search_doc(
    doc_id: str, body: SearchRequest, emb_svc: EmbeddingSvc
) -> SearchResponse:
    _validate_doc_id(doc_id)

    retriever = RetrieverService(emb_svc)

    try:
        hits = retriever.search(doc_id=doc_id, query=body.query, top_k=body.top_k)
    except FileNotFoundError as e:
        msg = str(e)
        if "FAISS_INDEX_NOT_FOUND" in msg:
            raise NotFound("FAISS index not found. Run /index first.")
        raise NotFound("Required artifacts missing.")
    except Exception:
        raise InternalError("Search failed unexpectedly.")

    return SearchResponse(
        doc_id=doc_id,
        query=body.query,
        top_k=body.top_k,
        hits=[SearchHit(**h.__dict__) for h in hits],
    )
