#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://localhost:8000}
URL=${1:-}
ENABLE_TILES=${2:-true}

if [[ -z "$URL" ]]; then
  echo "Usage: $0 https://example.com/video.mp4 [true|false]" >&2
  exit 1
fi

curl -s -F "url=${URL}" -F "enable_tiles=${ENABLE_TILES}" ${API_URL}/index/url | python -m json.tool
