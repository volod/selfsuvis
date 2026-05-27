#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${SSLM_VENV:-$ROOT/.venv-sslm}"
EXTRA="${1:-eval,dashboard}"
STAMP="$VENV/.sslm-install-stamp"
PYPROJECT="$ROOT/src/sslm/pyproject.toml"

if [[ ! -d "$VENV" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV"
    "$VENV/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
  else
    python3 -m venv "$VENV"
  fi
fi

needs_install=0
if [[ ! -f "$STAMP" ]]; then
  needs_install=1
elif [[ "$PYPROJECT" -nt "$STAMP" ]]; then
  needs_install=1
elif [[ "${SSLM_FORCE_INSTALL:-0}" == "1" ]]; then
  needs_install=1
fi

if [[ "$needs_install" == 1 ]]; then
  "$VENV/bin/python" -m pip install --upgrade pip
  if [[ "$EXTRA" == "none" ]]; then
    "$VENV/bin/python" -m pip install -e "$ROOT/src/sslm"
  else
    "$VENV/bin/python" -m pip install -e "$ROOT/src/sslm[$EXTRA]"
  fi
  touch "$STAMP"
else
  echo "SSLM venv up-to-date (pyproject.toml unchanged) -- skipping pip install."
fi

echo "SSLM venv ready: $VENV"
