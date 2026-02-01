#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://localhost:8000}
JOB_ID=${1:-}

if [[ -z "$JOB_ID" ]]; then
  echo "Usage: $0 <job_id>" >&2
  exit 1
fi

while true; do
  STATUS=$(curl -s ${API_URL}/jobs/${JOB_ID})
  echo "$STATUS" | python -m json.tool
  STATE=$(python - <<'PY'
import json,sys
sys.stdout.write(json.load(sys.stdin).get("status", ""))
PY
<<< "$STATUS")
  if [[ "$STATE" == "finished" || "$STATE" == "error" ]]; then
    break
  fi
  sleep 2
 done
