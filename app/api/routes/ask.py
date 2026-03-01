import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.core.config import settings
from app.models.ask import AskRequest, AskResponse, AskSource
from app.services.qa.ask_pipeline import ask_over_docs
from app.services.retrieval.retriever import RetrieverService
from app.storage.faiss_store import get_faiss_index_path

router = APIRouter(tags=["qa"])

DOC_ID_RE = re.compile(r"^[a-f0-9]{16,64}$")


def list_indexed_docs(data_dir: str) -> list[str]:
    """
    Scope = all: list processed/* directories that contain faiss.index.
    """
    root = Path(data_dir) / "processed"

    if not root.exists():
        return []

    out: list[str] = []
    for child in root.iterdir():
        if child.is_dir() and DOC_ID_RE.match(child.name):
            if (child / "faiss.index").exists():
                out.append(child.name)

    return out


def validate_doc_ids(doc_ids: list[str]) -> list[str]:
    valid: list[str] = []
    for did in doc_ids:
        if not DOC_ID_RE.match(did):
            continue
        if get_faiss_index_path(did).exists():
            valid.append(did)

    return valid


@router.post("/ask", response_model=AskResponse)
def ask(request: Request, body: AskRequest) -> AskResponse:
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if len(question) > settings.MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail="Question too long.")

    if body.scope == "docs":
        if not body.doc_ids:
            raise HTTPException(status_code=400, detail="doc_ids required when scope='docs'.")
        doc_ids = body.doc_ids
    else:
        doc_ids = list_indexed_docs(settings.DATA_DIR)

    doc_ids = validate_doc_ids(doc_ids)
    if not doc_ids:
        raise HTTPException(
            status_code=404, detail="No indexed documents available. Run /index first."
        )

    emb_svc = getattr(request.app.state, "embedding_service", None)
    qa_svc = getattr(request.app.state, "qa_service", None)
    if emb_svc is None:
        raise HTTPException(
            status_code=503, detail="Embedding service unavailable (model not loaded)."
        )
    if qa_svc is None:
        raise HTTPException(status_code=503, detail="QA service unavailable (model not loaded).")

    retriever = RetrieverService(emb_svc)
    res = ask_over_docs(
        question=question,
        doc_ids=doc_ids,
        top_k=body.top_k,
        retriever=retriever,
        qa=qa_svc,
    )

    ner_svc = getattr(request.app.state, "ner_service", None)
    entities = []
    if ner_svc is not None:
        try:
            entities = ner_svc.extract_entities(res.answer, res.sources)
        except Exception:
            entities = []

    sources: list[AskSource] = [
        AskSource(
            doc_id=s.doc_id,
            page=s.page,
            chunk_id=s.chunk_id,
            score=s.score,
            text_excerpt=s.text_snippet,
        )
        for s in res.sources
    ]

    return AskResponse(
        answer=res.answer,
        confidence=res.confidence,
        sources=sources,
        entities=entities,
    )
