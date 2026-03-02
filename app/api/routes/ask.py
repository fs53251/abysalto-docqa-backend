import logging
import re
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.core.config import settings
from app.models.ask import AskRequest, AskResponse, AskSource
from app.services.cache.cache_keys import (
    ans_key,
    mask_entities,
    normalize_question,
    qemb_key,
    retr_key,
    sem_key,
)
from app.services.qa.ask_pipeline import answer_with_sources
from app.services.retrieval.retriever import RetrievedChunk, RetrieverService
from app.storage.faiss_store import get_faiss_index_path

router = APIRouter(tags=["qa"])
logger = logging.getLogger(__name__)

DOC_ID_RE = re.compile(r"^[a-f0-9]{16,64}$")

# Caching / pipeline ordering rationale:
#
# Goal: minimize expensive work (embeddings + FAISS + QA + NER) for repeated queries,
# while keeping results correct and provenance-safe.
#
# 1) Semantic cache — BEFORE exact answer cache
#    - Semantic cache can return a response even when the question text changes slightly
#      (e.g., numbers/dates/emails differ).
#    - We first "mask" variable entities (AMOUNT/YEAR/NUMBER/EMAIL) to reduce fragmentation,
#      embed the masked question, and compare with a cached masked embedding.
#    - If cosine similarity >= SEMANTIC_CACHE_THRESHOLD (e.g., 0.75), we treat it as a safe hit
#      and return the cached full response (answer + sources + entities).
#    - This potentially skips ALL downstream work.
#    - Note: In strict production, you can place exact cache check before semantic to avoid
#      computing the masked embedding when an exact hit exists. We keep semantic first here
#      to prioritize semantic reuse; swapping the two is a valid optimization.
#
# 2) Exact final answer cache — fastest and safest
#    - Keyed by: scope + pipeline_version + normalized_question + top_k
#    - If hit: return full response immediately (no embedding, no retrieval, no QA, no NER).
#    - This is the biggest latency saver for identical repeated questions.
#
# 3) Query embedding cache — reuse the query vector
#    - FAISS retrieval requires a query embedding.
#    - Embedding is model inference (expensive on CPU); cache it by normalized question hash.
#    - We compute it once and reuse for all doc_ids in the scope (multi-doc).
#
# 4) Retrieval cache (per-doc) — skip FAISS + disk I/O
#    - Cache per doc_id and per index_version so that rebuilding an index invalidates cache.
#    - Value contains top-k hits with provenance (doc_id/page/chunk_id/score/snippet).
#    - If hit: skip index loading/search and snippet lookup.
#
# 5) QA over retrieved sources (answer_with_sources)
#    - Only after we have the final top-k sources.
#    - Context is constructed deterministically from sources and truncated to QA_MAX_CONTEXT_CHARS
#      to avoid model context limits.
#    - No-answer policy: if model score < QA_MIN_SCORE (or empty answer) -> return "I don't know..."
#
# 6) NER post-processing
#    - Extract entities from the final answer AND from retrieved snippets.
#    - Done after QA because entities depend on the final answer and chosen sources.
#    - Fail-soft: NER failure must not break /ask.
#
# 7) Store final caches
#    - After producing the final response, store:
#      - exact answer cache (always when cache enabled)
#      - semantic cache entry (masked embedding + response) when semantic cache enabled
#    - TTL is applied so caches expire and don't serve stale results forever.


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
    t0 = time.perf_counter()

    question_raw = (body.question or "").strip()
    if not question_raw:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if len(question_raw) > settings.MAX_QUESTION_CHARS:
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

    # services
    emb_svc = getattr(request.app.state, "embedding_service", None)
    qa_svc = getattr(request.app.state, "qa_service", None)

    if emb_svc is None:
        raise HTTPException(
            status_code=503, detail="Embedding service unavailable (model not loaded)."
        )
    if qa_svc is None:
        raise HTTPException(status_code=503, detail="QA service unavailable (model not loaded).")

    ner_svc = getattr(request.app.state, "ner_service", None)
    cache = getattr(request.app.state, "cache", None)

    # Versions of cache invalidation
    scope = body.scope
    pipeline_version = (
        f"qa={settings.QA_MODEL_NAME}|emb={settings.EMBEDDING_MODEL_NAME}|"
        f"ner={settings.NER_MODEL_NAME}|chunk={settings.CHUNK_SIZE_CHARS}-{settings.CHUNK_OVERLAP_CHARS}"
    )
    index_version = "v1"

    qn = normalize_question(question_raw)
    top_k = body.top_k

    cache_hit = False

    # 1) Semantic cache - BEFORE exact answer cache
    if cache is not None and settings.ENABLE_CACHE and settings.ENABLE_SEMANTIC_CACHE:
        mk = sem_key(scope, pipeline_version, qn, top_k)
        emb_hit = cache.get_embedding(mk + ":emb")
        resp_hit = cache.get_json(mk + ":resp")

        if emb_hit.hit and resp_hit.hit:
            mq = mask_entities(qn)
            cur = emb_svc.encode_texts([mq]).reshape(-1).astype(np.float32)
            prev = emb_hit.value.reshape(-1).astype(np.float32)

            if cur.shape == prev.shape:
                sim = float(np.dot(cur, prev) / (np.linalg.norm(cur) * np.linalg.norm(prev) + 1e-8))
                if sim >= settings.SEMANTIC_CACHE_THRESHOLD:
                    cache_hit = True
                    dt = (time.perf_counter() - t0) * 1000
                    logger.info("ask cache_hit=1 layer=semantic sim=%.3f latency_ms=%.2f", sim, dt)
                    return AskResponse(**resp_hit.value)

    # 2) Exact final answer cache
    if cache is not None and settings.ENABLE_CACHE:
        ak = ans_key(scope, pipeline_version, qn, top_k)
        ans_cached = cache.get_json(ak)

        if ans_cached.hit:
            cache_hit = True
            dt = (time.perf_counter() - t0) * 1000
            logger.info("ask cache_hit=1 layer=answer latency_ms=%.2f", dt)

            return AskResponse(**ans_cached.value)

    # 3) Query embedding cache
    q_emb = None
    if cache is not None and settings.ENABLE_CACHE:
        ek = qemb_key(qn)
        emb_cached = cache.get_embedding(ek)

        if emb_cached.hit:
            q_emb = emb_cached.value.reshape(1, -1)
        else:
            q_emb = emb_svc.encode_texts([qn])
            cache.set_embedding(ek, q_emb, settings.CACHE_TTL_SECONDS)
    else:
        q_emb = emb_svc.encode_texts([qn])

    # 4) Retrieval cache (per-doc)
    retriever = RetrieverService(emb_svc)
    all_hits: list[RetrievedChunk] = []

    for did in doc_ids:
        if cache is not None and settings.ENABLE_CACHE:
            rk = retr_key(scope, index_version, did, qn, top_k)
            rc = cache.get_json(rk)

            if rc.hit:
                for h in rc.value:
                    all_hits.append(
                        RetrievedChunk(
                            doc_id=h["doc_id"],
                            chunk_id=h["chunk_id"],
                            score=float(h["score"]),
                            page=h.get("page"),
                            chunk_index=h.get("chunk_index"),
                            text_snippet=h["text_snippet"],
                        )
                    )
            else:
                hits = retriever.search(doc_id=did, query=qn, top_k=top_k, query_emb=q_emb)
                cache.set_json(rk, [h.__dict__ for h in hits], settings.CACHE_TTL_SECONDS)
                all_hits.extend(hits)
        else:
            hits = retriever.search(doc_id=did, query=qn, top_k=top_k, query_emb=q_emb)
            all_hits.extend(hits)

    all_hits = sorted(all_hits, key=lambda x: x.score, reverse=True)[
        : max(1, min(top_k, settings.MAX_TOP_K))
    ]

    # 5) QA + no-answer policy handled inside answer_with_sources
    res = answer_with_sources(question=qn, sources=all_hits, qa=qa_svc)

    # 6) NER
    entities = []
    if ner_svc is not None:
        try:
            entities = ner_svc.extract_entities(res.answer, res.sources)
        except Exception:
            entities = []

    sources = [
        AskSource(
            doc_id=s.doc_id,
            page=s.page,
            chunk_id=s.chunk_id,
            score=s.score,
            text_excerpt=s.text_snippet,
        )
        for s in res.sources
    ]

    response_obj = AskResponse(
        answer=res.answer,
        confidence=res.confidence,
        sources=sources,
        entities=entities,
    )

    # 7) Store final caches
    if cache is not None and settings.ENABLE_CACHE:
        ak = ans_key(scope, pipeline_version, qn, top_k)
        cache.set_json(ak, response_obj.model_dump(), settings.CACHE_TTL_SECONDS)

        if settings.ENABLE_SEMANTIC_CACHE:
            mk = sem_key(scope, pipeline_version, qn, top_k)
            mq = mask_entities(qn)
            m_emb = emb_svc.encode_texts([mq])
            cache.set_embedding(mk + ":emb", m_emb, settings.CACHE_TTL_SECONDS)
            cache.set_json(mk + ":resp", response_obj.model_dump(), settings.CACHE_TTL_SECONDS)

    dt = (time.perf_counter() - t0) * 1000
    logger.info("ask cache_hit=%d latency_ms=%.2f", 1 if cache_hit else 0, dt)

    return response_obj
