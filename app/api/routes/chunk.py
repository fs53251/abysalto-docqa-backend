import re

from fastapi import APIRouter, HTTPException, Query

from app.models.chunking import ChunkBuildResponse
from app.services.indexing.chunking import build_chunks_for_doc, save_chunks
from app.storage.chunks import get_chunk_map_path, get_chunks_jsonl_path

router = APIRouter(tags=["indexing"])

DOC_ID_RE = re.compile(r"^[a-f0-9]{16,64}$")


def _validate_doc_id(doc_id: str) -> None:
    if not DOC_ID_RE.match(doc_id):
        raise HTTPException(status_code=400, detail="Invalid doc_id format.")


@router.post("/documents/{doc_id}/chunk", response_model=ChunkBuildResponse)
def chunk_document(doc_id: str, force: bool = Query(False)) -> ChunkBuildResponse:
    """
    Create chunks.jsonl + chunk_map.json from processed/{doc_id}/text.json
    chunks.jsonl has content while chunk_map is something like metadata
    """
    _validate_doc_id(doc_id)

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
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="text.json not found. Run extraction first.")
    except ValueError as e:
        if str(e) == "TOO_MANY_CHUNKS":
            raise HTTPException(
                status_code=413, detail="Too many chunks generated; adjust settings."
            )
        raise HTTPException(status_code=400, detail="Invalid text.json format.")

    paths = save_chunks(doc_id, chunks, chunk_map)

    return ChunkBuildResponse(
        doc_id=doc_id,
        status="chunked",
        chunk_count=len(chunks),
        chunks_jsonl=paths["chunks_jsonl"],
        chunk_map=paths["chunk_map"],
    )
