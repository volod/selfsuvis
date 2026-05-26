#!/usr/bin/env bash
# Wraps: sslm sequential  (see src/sslm/README.md for suite reference)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${SSLM_VENV:-$ROOT/.venv-sslm}"

if [[ ! -x "$VENV/bin/sslm" ]]; then
  echo "sslm venv not found at $VENV — run: scripts/sslm/setup-venv.sh eval,dashboard" >&2
  exit 1
fi

export SSLM_PROJECT_ROOT="${SSLM_PROJECT_ROOT:-$ROOT}"
export SSLM_HF_CACHE="${SSLM_HF_CACHE:-$ROOT/.data/sslm/hf-cache}"
cd "$ROOT"

exec "$VENV/bin/sslm" sequential \
  --models zaya1-8b,qwen3-8b \
  --suite open_llm_v2 \
  --compose-file "$ROOT/.data/sslm/docker-compose.generated.yml" \
  --results-dir "$ROOT/.data/sslm/results" \
  "$@"
