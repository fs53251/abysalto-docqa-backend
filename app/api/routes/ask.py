from __future__ import annotations

import hashlib
import logging
import time

import numpy as np
from fastapi import APIRouter, Depends

from app.api.deps import (
    CurrentIdentity,
    DbSession,
    EmbeddingSvc,
    OptCache,
    OptNerSvc,
    QaSvc,
)
from app.core.config import settings
from app.core.errors import InvalidInput, NotFound
from app.core.identity import RequestIdentity
from app.core.identifiers import document_public_id, parse_document_public_id
from app.core.log_safety import safe_excerpt
from app.models.ask import AskRequest, AskResponse, AskSource
from app.repositories.documents import (
    assert_documents_owned_by_identity,
    list_documents_for_identity,
)
from app.services.cache.cache_keys import (
    ans_key,
    mask_entities,
    normalize_question,
    qemb_key,
    retr_key,
    sem_key,
)
from app.services.qa.ask_pipeline import answer_with_sources
from app.services.rate_limit import identity_rate_limit_key, rate_limit
from app.services.retrieval.retriever import RetrievedChunk, RetrieverService
from app.storage.faiss_store import get_faiss_index_path

router = APIRouter(tags=["qa"])
logger = logging.getLogger(__name__)

ask_rate_limit = rate_limit(
    limit=lambda: settings.ASK_RATE_LIMIT_PER_MIN,
    window_seconds=lambda: settings.RATE_LIMIT_WINDOW_SECONDS,
    key_fn=identity_rate_limit_key("ask"),
)


def _docs_digest(doc_ids: list[str]) -> str:
    joined = ",".join(sorted(set(doc_ids)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _scope_cache_key(
    identity: RequestIdentity,
    scope_mode: str,
    doc_ids: list[str],
) -> str:
    identity_hash = hashlib.sha256(identity.log_identity.encode("utf-8")).hexdigest()[
        :12
    ]
    return f"{scope_mode}:{identity_hash}:{_docs_digest(doc_ids)}"


def _resolve_identity_indexed_scope(
    db,
    identity: RequestIdentity,
) -> tuple[list[str], dict[str, str]]:
    doc_ids: list[str] = []
    filename_by_doc_id: dict[str, str] = {}

    for document in list_documents_for_identity(db, identity=identity):
        public_id = document_public_id(document.id)
        if not get_faiss_index_path(public_id).exists():
            continue

        doc_ids.append(public_id)
        filename_by_doc_id[public_id] = document.filename

    return doc_ids, filename_by_doc_id


def _resolve_requested_scope(
    db,
    identity: RequestIdentity,
    requested_doc_ids: list[str],
) -> tuple[list[str], dict[str, str]]:
    parsed_doc_ids = []
    seen: set[str] = set()

    for raw_doc_id in requested_doc_ids:
        try:
            parsed_doc_id = parse_document_public_id(raw_doc_id)
        except ValueError as exc:
            raise InvalidInput("One or more doc_ids are invalid.") from exc

        public_id = document_public_id(parsed_doc_id)
        if public_id in seen:
            continue

        seen.add(public_id)
        parsed_doc_ids.append(parsed_doc_id)

    owned_documents = assert_documents_owned_by_identity(
        db,
        doc_ids=parsed_doc_ids,
        identity=identity,
    )

    doc_ids: list[str] = []
    filename_by_doc_id: dict[str, str] = {}

    for document in owned_documents:
        public_id = document_public_id(document.id)
        if not get_faiss_index_path(public_id).exists():
            continue

        doc_ids.append(public_id)
        filename_by_doc_id[public_id] = document.filename

    return doc_ids, filename_by_doc_id


@router.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    db: DbSession,
    identity: CurrentIdentity,
    emb_svc: EmbeddingSvc,
    qa_svc: QaSvc,
    ner_svc: OptNerSvc,
    cache: OptCache,
    _rate_limit: None = Depends(ask_rate_limit),
) -> AskResponse:
    del _rate_limit

    started_at = time.perf_counter()

    question_raw = body.question
    if not question_raw:
        raise InvalidInput("Question must not be empty.")

    if body.doc_ids:
        doc_ids, filename_by_doc_id = _resolve_requested_scope(
            db,
            identity,
            body.doc_ids,
        )
        scope_mode = "docs"
    elif body.scope == "docs":
        raise InvalidInput("doc_ids required when scope='docs'.")
    else:
        doc_ids, filename_by_doc_id = _resolve_identity_indexed_scope(db, identity)
        scope_mode = "identity"

    if not doc_ids:
        raise NotFound(
            "No indexed documents available for this identity. Run /index first."
        )

    cache_scope = _scope_cache_key(identity, scope_mode, doc_ids)
    pipeline_version = (
        f"qa={settings.QA_MODEL_NAME}|emb={settings.EMBEDDING_MODEL_NAME}|"
        f"ner={settings.NER_MODEL_NAME}|"
        f"chunk={settings.CHUNK_SIZE_CHARS}-{settings.CHUNK_OVERLAP_CHARS}"
    )
    index_version = "v1"
    normalized_question = normalize_question(question_raw)
    question_excerpt = safe_excerpt(normalized_question, max_chars=120)
    top_k = body.top_k
    cache_hit = False

    if cache is not None and settings.ENABLE_CACHE and settings.ENABLE_SEMANTIC_CACHE:
        semantic_key = sem_key(
            cache_scope,
            pipeline_version,
            normalized_question,
            top_k,
        )
        emb_hit = cache.get_embedding(semantic_key + ":emb")
        resp_hit = cache.get_json(semantic_key + ":resp")

        if emb_hit.hit and resp_hit.hit:
            masked_question = mask_entities(normalized_question)
            current = (
                emb_svc.encode_texts([masked_question]).reshape(-1).astype(np.float32)
            )
            previous = emb_hit.value.reshape(-1).astype(np.float32)

            if current.shape == previous.shape:
                sim = float(
                    np.dot(current, previous)
                    / (np.linalg.norm(current) * np.linalg.norm(previous) + 1e-8)
                )
                if sim >= settings.SEMANTIC_CACHE_THRESHOLD:
                    cache_hit = True
                    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
                    logger.info(
                        "ask completed from semantic cache",
                        extra={
                            "event": "ask.completed",
                            "cache_hit": 1,
                            "layer": "semantic",
                            "sim": sim,
                            "latency_ms": latency_ms,
                            "doc_ids_count": len(doc_ids),
                            "top_k": top_k,
                            "question_excerpt": question_excerpt,
                            "source_count": len(resp_hit.value.get("sources", [])),
                        },
                    )
                    return AskResponse(**resp_hit.value)

    if cache is not None and settings.ENABLE_CACHE:
        answer_key = ans_key(
            cache_scope,
            pipeline_version,
            normalized_question,
            top_k,
        )
        ans_cached = cache.get_json(answer_key)
        if ans_cached.hit:
            cache_hit = True
            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            logger.info(
                "ask completed from answer cache",
                extra={
                    "event": "ask.completed",
                    "cache_hit": 1,
                    "layer": "answer",
                    "latency_ms": latency_ms,
                    "doc_ids_count": len(doc_ids),
                    "top_k": top_k,
                    "question_excerpt": question_excerpt,
                    "source_count": len(ans_cached.value.get("sources", [])),
                },
            )
            return AskResponse(**ans_cached.value)

    if cache is not None and settings.ENABLE_CACHE:
        query_embedding_key = qemb_key(normalized_question)
        emb_cached = cache.get_embedding(query_embedding_key)
        if emb_cached.hit:
            query_embedding = emb_cached.value.reshape(1, -1)
        else:
            query_embedding = emb_svc.encode_texts([normalized_question])
            cache.set_embedding(
                query_embedding_key,
                query_embedding,
                settings.CACHE_TTL_SECONDS,
            )
    else:
        query_embedding = emb_svc.encode_texts([normalized_question])

    retriever = RetrieverService(emb_svc)
    all_hits: list[RetrievedChunk] = []

    for doc_id in doc_ids:
        if cache is not None and settings.ENABLE_CACHE:
            retrieval_key = retr_key(
                cache_scope,
                index_version,
                doc_id,
                normalized_question,
                top_k,
            )
            retrieval_cached = cache.get_json(retrieval_key)

            if retrieval_cached.hit:
                for hit in retrieval_cached.value:
                    all_hits.append(
                        RetrievedChunk(
                            doc_id=hit["doc_id"],
                            chunk_id=hit["chunk_id"],
                            score=float(hit["score"]),
                            page=hit.get("page"),
                            chunk_index=hit.get("chunk_index"),
                            text_snippet=hit["text_snippet"],
                        )
                    )
            else:
                hits = retriever.search(
                    doc_id=doc_id,
                    query=normalized_question,
                    top_k=top_k,
                    query_emb=query_embedding,
                )
                cache.set_json(
                    retrieval_key,
                    [hit.__dict__ for hit in hits],
                    settings.CACHE_TTL_SECONDS,
                )
                all_hits.extend(hits)
        else:
            hits = retriever.search(
                doc_id=doc_id,
                query=normalized_question,
                top_k=top_k,
                query_emb=query_embedding,
            )
            all_hits.extend(hits)

    all_hits = sorted(all_hits, key=lambda hit: hit.score, reverse=True)[
        : max(1, min(top_k, settings.MAX_TOP_K))
    ]

    result = answer_with_sources(
        question=normalized_question,
        sources=all_hits,
        qa=qa_svc,
    )

    entities = []
    if ner_svc is not None:
        try:
            entities = ner_svc.extract_entities(result.answer, result.sources)
        except Exception:
            entities = []

    response_obj = AskResponse(
        answer=result.answer,
        confidence=result.confidence,
        sources=[
            AskSource(
                doc_id=source.doc_id,
                filename=filename_by_doc_id.get(source.doc_id),
                page=source.page,
                chunk_id=source.chunk_id,
                score=source.score,
                text_excerpt=source.text_snippet,
            )
            for source in result.sources
        ],
        entities=entities,
    )

    if cache is not None and settings.ENABLE_CACHE:
        answer_key = ans_key(
            cache_scope,
            pipeline_version,
            normalized_question,
            top_k,
        )
        cache.set_json(
            answer_key,
            response_obj.model_dump(),
            settings.CACHE_TTL_SECONDS,
        )

        if settings.ENABLE_SEMANTIC_CACHE:
            semantic_key = sem_key(
                cache_scope,
                pipeline_version,
                normalized_question,
                top_k,
            )
            masked_question = mask_entities(normalized_question)
            masked_embedding = emb_svc.encode_texts([masked_question])
            cache.set_embedding(
                semantic_key + ":emb",
                masked_embedding,
                settings.CACHE_TTL_SECONDS,
            )
            cache.set_json(
                semantic_key + ":resp",
                response_obj.model_dump(),
                settings.CACHE_TTL_SECONDS,
            )

    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    logger.info(
        "ask completed",
        extra={
            "event": "ask.completed",
            "cache_hit": 1 if cache_hit else 0,
            "latency_ms": latency_ms,
            "doc_ids_count": len(doc_ids),
            "top_k": top_k,
            "question_excerpt": question_excerpt,
            "source_count": len(response_obj.sources),
        },
    )
    return response_obj
