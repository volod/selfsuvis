#!/usr/bin/env bash
# Wraps: sslm dashboard
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${SSLM_VENV:-$ROOT/.venv-sslm}"

if [[ ! -x "$VENV/bin/sslm" ]]; then
  echo "sslm venv not found — run: scripts/sslm/setup-venv.sh eval,dashboard" >&2
  exit 1
fi

cd "$ROOT"
exec "$VENV/bin/sslm" dashboard \
  --results-dir "$ROOT/.data/sslm/results" \
  "$@"
