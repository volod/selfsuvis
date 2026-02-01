#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://localhost:8000}
DIR_PATH=${1:-}
ENABLE_TILES=${2:-true}

if [[ -z "$DIR_PATH" ]]; then
  echo "Usage: $0 /path/to/dir [true|false]" >&2
  exit 1
fi

curl -s -F "path=${DIR_PATH}" -F "enable_tiles=${ENABLE_TILES}" ${API_URL}/index/dir | python -m json.tool
