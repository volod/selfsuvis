#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${SSLM_VENV:-$ROOT/.venv-sslm}"

export SSLM_PROJECT_ROOT="${SSLM_PROJECT_ROOT:-$ROOT}"
export SSLM_HF_CACHE="${SSLM_HF_CACHE:-$ROOT/.data/sslm/hf-cache}"
cd "$ROOT"

exec "$VENV/bin/sslm" render-compose \
  --output "$ROOT/.data/sslm/docker-compose.generated.yml" \
  "$@"
