#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
TEST_DIR="${TEST_DIR:-data/test_docs}"
JQ="${JQ:-jq}"

if ! command -v curl >/dev/null 2>&1; then
  echo "❌ curl not found"; exit 1
fi
if ! command -v "$JQ" >/dev/null 2>&1; then
  echo "❌ jq not found (install: sudo apt install jq)"; exit 1
fi

echo "== Health =="
curl -s -i "$BASE_URL/health" | head -n 20
echo

echo "== Files =="
ls -lah "$TEST_DIR"
echo

req() {
  local method="$1"; local url="$2"; shift 2
  echo "-> $method $url"
  curl -s -X "$method" "$url" "$@" -H "Accept: application/json"
}

shopt -s nullglob
FILES=("$TEST_DIR"/*)
if [ ${#FILES[@]} -eq 0 ]; then
  echo "❌ No files found in $TEST_DIR"; exit 1
fi

for f in "${FILES[@]}"; do
  [ -d "$f" ] && continue

  echo "============================================================"
  echo "FILE: $f"
  echo "------------------------------------------------------------"

  # 1) Upload
  UPLOAD_JSON=$(curl -s -X POST "$BASE_URL/upload" -F "files=@$f")
  echo "$UPLOAD_JSON" | $JQ .
  DOC_ID=$(echo "$UPLOAD_JSON" | $JQ -r '.documents[0].doc_id // empty')
  STATUS=$(echo "$UPLOAD_JSON" | $JQ -r '.documents[0].status // empty')

  if [ -z "$DOC_ID" ] || [ "$STATUS" != "ok" ]; then
    echo "❌ Upload failed for $f (status=$STATUS doc_id=$DOC_ID)"
    continue
  fi

  echo "✅ doc_id=$DOC_ID"
  echo

  # 2) Extract with OCR fallback
  EXTRACT_JSON=$(req POST "$BASE_URL/documents/$DOC_ID/extract-text?ocr_fallback=true")
  echo "$EXTRACT_JSON" | $JQ .
  echo

  # 3) Chunk
  CHUNK_JSON=$(req POST "$BASE_URL/documents/$DOC_ID/chunk")
  echo "$CHUNK_JSON" | $JQ .
  CHUNK_COUNT=$(echo "$CHUNK_JSON" | $JQ -r '.chunk_count // 0')
  echo

  if [ "$CHUNK_COUNT" -eq 0 ]; then
    echo "⚠️ chunk_count=0 -> skipping embed/index/ask"
    continue
  fi

  # 4) Embed
  EMBED_JSON=$(req POST "$BASE_URL/documents/$DOC_ID/embed")
  echo "$EMBED_JSON" | $JQ .
  DIM=$(echo "$EMBED_JSON" | $JQ -r '.dim // 0')
  ROWS=$(echo "$EMBED_JSON" | $JQ -r '.row_count // 0')
  echo

  if [ "$DIM" -eq 0 ] || [ "$ROWS" -eq 0 ]; then
    echo "❌ Embeddings look empty (dim=$DIM rows=$ROWS)"
    continue
  fi

  # 5) Index
  INDEX_JSON=$(req POST "$BASE_URL/documents/$DOC_ID/index")
  echo "$INDEX_JSON" | $JQ .
  echo

  # 6) Ask #1 (summary)
  ASK1=$(curl -s -X POST "$BASE_URL/ask" -H "Content-Type: application/json" \
    -d "{\"question\":\"Summarize this document in 2-3 sentences.\",\"scope\":\"docs\",\"doc_ids\":[\"$DOC_ID\"],\"top_k\":3}")
  echo "$ASK1" | $JQ .
  echo

  # 7) Ask #2 (invoice-specific / contract-specific)
  ASK2=$(curl -s -X POST "$BASE_URL/ask" -H "Content-Type: application/json" \
    -d "{\"question\":\"Extract key fields (dates, parties, totals, invoice number if present).\",\"scope\":\"docs\",\"doc_ids\":[\"$DOC_ID\"],\"top_k\":5}")
  echo "$ASK2" | $JQ .
  echo

  # 8) Repeat Ask #2 to check cache hit (if enabled)
  ASK3=$(curl -s -X POST "$BASE_URL/ask" -H "Content-Type: application/json" \
    -d "{\"question\":\"Extract key fields (dates, parties, totals, invoice number if present).\",\"scope\":\"docs\",\"doc_ids\":[\"$DOC_ID\"],\"top_k\":5}")
  echo "$ASK3" | $JQ .
  echo "✅ repeated ask (watch server logs for cache_hit=1)"
  echo
done

echo "✅ Manual E2E finished."
