#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://localhost:8000}
ASSETS_DIR=${ASSETS_DIR:-./tests/assets}

GREEN_VIDEO=${ASSETS_DIR}/vid_green.mp4
TEST_VIDEO=${ASSETS_DIR}/vid_testsrc.mp4
QUERY_IMG=${ASSETS_DIR}/green.png

if [[ ! -f "$GREEN_VIDEO" || ! -f "$TEST_VIDEO" || ! -f "$QUERY_IMG" ]]; then
  echo "Missing test assets in ${ASSETS_DIR}" >&2
  exit 1
fi

echo "Indexing green video via API..."
JOB=$(curl -s -F "file=@${GREEN_VIDEO}" -F "enable_tiles=false" ${API_URL}/index/video)
JOB_ID=$(python - <<'PY'
import json,sys
sys.stdout.write(json.load(sys.stdin)["job_id"])
PY
<<< "$JOB")

./scripts/job_watch.sh "$JOB_ID"

echo "Indexing testsrc video via API..."
JOB2=$(curl -s -F "file=@${TEST_VIDEO}" -F "enable_tiles=true" ${API_URL}/index/video)
JOB_ID2=$(python - <<'PY'
import json,sys
sys.stdout.write(json.load(sys.stdin)["job_id"])
PY
<<< "$JOB2")

./scripts/job_watch.sh "$JOB_ID2"

echo "Text query..."
curl -s -H "Content-Type: application/json" -d '{"text":"green field"}' "${API_URL}/query/text?top_k=3&search_type=frame" | python -m json.tool

echo "Image query..."
curl -s -F "file=@${QUERY_IMG}" -F "top_k=3" -F "search_type=both" ${API_URL}/query/image | python -m json.tool

if [[ -n "${INDEX_DIR_PATH:-}" ]]; then
  echo "Directory precheck..."
  ./scripts/precheck_dir.sh "$INDEX_DIR_PATH" true true
  echo "Directory index..."
  ./scripts/index_dir.sh "$INDEX_DIR_PATH" false
fi
