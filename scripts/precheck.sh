#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://localhost:8000}
MODE=${1:-}
TARGET=${2:-}

if [[ -z "$MODE" || -z "$TARGET" ]]; then
  echo "Usage: $0 file /path/to/video.mp4 | path /path/to/video.mp4 | url https://..." >&2
  exit 1
fi

case "$MODE" in
  file)
    curl -s -F "file=@${TARGET}" ${API_URL}/index/precheck | python -m json.tool
    ;;
  path)
    curl -s -F "path=${TARGET}" ${API_URL}/index/precheck | python -m json.tool
    ;;
  url)
    curl -s -F "url=${TARGET}" ${API_URL}/index/precheck | python -m json.tool
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
 esac
