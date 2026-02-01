#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://localhost:8000}
VIDEO_PATH=${1:-}
IMAGE_PATH=${2:-}

if [[ -z "$VIDEO_PATH" ]]; then
  echo "Usage: $0 /path/to/video.mp4 /path/to/image.jpg" >&2
  exit 1
fi

echo "Indexing video..."
JOB=$(curl -s -F "file=@${VIDEO_PATH}" -F "enable_tiles=true" ${API_URL}/index/video)
JOB_ID=$(python - <<'PY'
import json,sys
sys.stdout.write(json.load(sys.stdin)["job_id"])
PY
<<< "$JOB")

echo "Job: $JOB_ID"

for i in {1..30}; do
  STATUS=$(curl -s ${API_URL}/jobs/${JOB_ID})
  echo "$STATUS"
  STATUS_VAL=$(python - <<'PY'
import json,sys
sys.stdout.write(json.load(sys.stdin)["status"])
PY
<<< "$STATUS")
  if [[ "$STATUS_VAL" == "finished" || "$STATUS_VAL" == "error" ]]; then
    break
  fi
  sleep 2
 done

if [[ -n "$IMAGE_PATH" ]]; then
  echo "Image query..."
  curl -s -F "file=@${IMAGE_PATH}" -F "top_k=5" -F "search_type=both" ${API_URL}/query/image | python -m json.tool
fi

echo "Text query..."
curl -s -H "Content-Type: application/json" -d '{"text":"green field"}' "${API_URL}/query/text?top_k=5&search_type=both" | python -m json.tool
