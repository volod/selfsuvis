#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${SSLM_VENV:-$ROOT/.venv-sslm}"
EXTRA="${1:-eval,dashboard}"

if [[ ! -d "$VENV" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV"
    "$VENV/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
  else
    python3 -m venv "$VENV"
  fi
else
  echo "Venv exists at $VENV — skipping creation, updating packages."
fi

"$VENV/bin/python" -m pip install --upgrade pip
if [[ "$EXTRA" == "none" ]]; then
  "$VENV/bin/python" -m pip install -e "$ROOT/src/sslm"
else
  "$VENV/bin/python" -m pip install -e "$ROOT/src/sslm[$EXTRA]"
fi

echo "SSLM venv ready: $VENV"

