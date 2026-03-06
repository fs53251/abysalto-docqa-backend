#!/usr/bin/env bash

set -e

BASE_URL="http://localhost:8000"
FILE="data/test_docs/northwind_invoice_scanned.pdf"

COOKIE_JAR=$(mktemp)

echo "--------------------------------------"
echo "1) Upload document"
echo "--------------------------------------"

UPLOAD_RESPONSE=$(curl -s \
  -X POST "$BASE_URL/upload" \
  -F "files=@$FILE" \
  -c "$COOKIE_JAR")

echo "$UPLOAD_RESPONSE" | jq

DOC_ID=$(echo "$UPLOAD_RESPONSE" | jq -r '.documents[0].doc_id')

if [ "$DOC_ID" == "null" ]; then
  echo "Upload failed"
  exit 1
fi

echo
echo "DOC_ID=$DOC_ID"

echo
echo "--------------------------------------"
echo "2) Session cookie check"
echo "--------------------------------------"

cat "$COOKIE_JAR"

echo
echo "--------------------------------------"
echo "3) List documents"
echo "--------------------------------------"

curl -s \
  -X GET "$BASE_URL/documents" \
  -b "$COOKIE_JAR" \
  | jq

echo
echo "--------------------------------------"
echo "4) Extract text"
echo "--------------------------------------"

curl -s \
  -X POST "$BASE_URL/documents/$DOC_ID/extract-text" \
  -b "$COOKIE_JAR" \
  | jq

echo
echo "--------------------------------------"
echo "5) Chunk text"
echo "--------------------------------------"

curl -s \
  -X POST "$BASE_URL/documents/$DOC_ID/chunk" \
  -b "$COOKIE_JAR" \
  | jq

echo
echo "--------------------------------------"
echo "6) Embed chunks"
echo "--------------------------------------"

curl -s \
  -X POST "$BASE_URL/documents/$DOC_ID/embed" \
  -b "$COOKIE_JAR" \
  | jq

echo
echo "--------------------------------------"
echo "7) Build index"
echo "--------------------------------------"

curl -s \
  -X POST "$BASE_URL/documents/$DOC_ID/index" \
  -b "$COOKIE_JAR" \
  | jq

echo
echo "--------------------------------------"
echo "8) Search"
echo "--------------------------------------"

curl -s \
  -X POST "$BASE_URL/documents/$DOC_ID/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"invoice number","top_k":3}' \
  -b "$COOKIE_JAR" \
  | jq

echo
echo "--------------------------------------"
echo "9) Ask question"
echo "--------------------------------------"

curl -s \
  -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -b "$COOKIE_JAR" \
  -d '{
        "question":"What is the invoice number?",
        "doc_ids":["'"$DOC_ID"'"]
      }' \
  | jq

echo
echo "--------------------------------------"
echo "E2E test finished"
echo "--------------------------------------"