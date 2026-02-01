#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://localhost:8000}
DIR_PATH=${1:-}
ENQUEUE=${2:-false}
ENABLE_TILES=${3:-true}

if [[ -z "$DIR_PATH" ]]; then
  echo "Usage: $0 /path/to/dir [enqueue=true|false] [enable_tiles=true|false]" >&2
  exit 1
fi

curl -s -F "path=${DIR_PATH}" -F "enqueue=${ENQUEUE}" -F "enable_tiles=${ENABLE_TILES}" ${API_URL}/index/precheck_dir | python -m json.tool
