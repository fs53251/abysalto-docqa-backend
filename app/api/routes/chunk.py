from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import OwnedDocument
from app.core.errors import InvalidInput, NotFound, PayloadTooLarge
from app.core.identifiers import document_public_id
from app.models.chunking import ChunkBuildResponse
from app.services.indexing.chunking import build_chunks_for_doc, save_chunks
from app.storage.chunks import get_chunk_map_path, get_chunks_jsonl_path

router = APIRouter(tags=["indexing"])


@router.post("/documents/{doc_id}/chunk", response_model=ChunkBuildResponse)
def chunk_document(
    document: OwnedDocument, force: bool = Query(False)
) -> ChunkBuildResponse:
    doc_id = document_public_id(document.id)
    chunks_path = get_chunks_jsonl_path(doc_id)
    map_path = get_chunk_map_path(doc_id)

    if chunks_path.exists() and map_path.exists() and not force:
        count = sum(1 for _ in chunks_path.open("r", encoding="utf-8"))
        return ChunkBuildResponse(
            doc_id=doc_id,
            status="already_chunked",
            chunk_count=count,
            chunks_jsonl=str(chunks_path),
            chunk_map=str(map_path),
        )

    try:
        chunks, chunk_map = build_chunks_for_doc(doc_id)
    except FileNotFoundError as exc:
        raise NotFound("text.json not found. Run extraction first.") from exc
    except ValueError as exc:
        if str(exc) == "TOO_MANY_CHUNKS":
            raise PayloadTooLarge(
                "Too many chunks generated; adjust settings."
            ) from exc
        raise InvalidInput("Invalid text.json format.") from exc

    paths = save_chunks(doc_id, chunks, chunk_map)
    return ChunkBuildResponse(
        doc_id=doc_id,
        status="chunked",
        chunk_count=len(chunks),
        chunks_jsonl=paths["chunks_jsonl"],
        chunk_map=paths["chunk_map"],
    )
