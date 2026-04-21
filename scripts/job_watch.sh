#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_HELPER="${REPO_ROOT}/src/selfsuvis/scripts/shell_helpers.py"

API_URL=${API_URL:-http://localhost:8000}
JOB_ID=${1:-}

if [[ -z "$JOB_ID" ]]; then
  echo "Usage: $0 <job_id>" >&2
  exit 1
fi

while true; do
  STATUS=$(curl -s ${API_URL}/jobs/${JOB_ID})
  echo "$STATUS" | python3 "$PY_HELPER" pretty-json
  STATE=$(python3 "$PY_HELPER" json-field --field status <<< "$STATUS")
  if [[ "$STATE" == "finished" || "$STATE" == "error" ]]; then
    break
  fi
  sleep 2
 done
