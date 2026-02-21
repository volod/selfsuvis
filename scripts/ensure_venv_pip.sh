#!/usr/bin/env bash
# Ensure pip is installed into the project .venv. Prefer uv if available.
# Usage: ./scripts/ensure_venv_pip.sh [.venv]
set -euo pipefail

VENV_PATH=${1:-.venv}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Error: Virtual environment not found at $VENV_PATH" >&2
  echo "Create it first with: uv venv $VENV_PATH  or  python3 -m venv $VENV_PATH" >&2
  exit 1
fi

if "$VENV_PATH/bin/python" -c "import pip" 2>/dev/null; then
  echo "pip is already available in $VENV_PATH. Nothing to do."
  exit 0
fi

echo "Installing pip into $VENV_PATH..."

if command -v uv >/dev/null 2>&1; then
  echo "  Using uv to install pip..."
  uv pip install --python "$VENV_PATH" pip
  echo "Done. pip is now available in $VENV_PATH."
  exit 0
fi

if "$VENV_PATH/bin/python" -c "import ensurepip" 2>/dev/null; then
  echo "  Using ensurepip (standard library)..."
  "$VENV_PATH/bin/python" -m ensurepip --upgrade
  echo "Done. pip is now available in $VENV_PATH."
  exit 0
fi

echo "  Downloading get-pip.py and bootstrapping pip..."
GET_PIP="$REPO_ROOT/.get-pip.py"
curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$GET_PIP"
"$VENV_PATH/bin/python" "$GET_PIP" --quiet
rm -f "$GET_PIP"
echo "Done. pip is now available in $VENV_PATH."
