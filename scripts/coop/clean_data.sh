#!/usr/bin/env bash
# Remove all data from bind-mounted directories.
# Fixes ownership first so user can remove without sudo when possible.
# NOTE: data/.env is a dotfile and is preserved by the glob rm -rf data/*.
#
# Usage:
#   ./scripts/clean_data.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="./data"
if [[ -f "$ROOT_DIR/data/.env" ]]; then
  set -a
  source "$ROOT_DIR/data/.env"
  set +a
  DATA_DIR="${DATA_DIR:-./data}"
fi

if [[ "$DATA_DIR" != /* ]]; then
  DATA_DIR="$ROOT_DIR/${DATA_DIR#./}"
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Data directory does not exist: $DATA_DIR"
  exit 0
fi

echo "Stopping containers..."
"$ROOT_DIR/scripts/coop/compose.sh" down 2>/dev/null || true

echo "Removing data under $DATA_DIR (preserving .env) ..."
if rm -rf "$DATA_DIR"/* 2>/dev/null; then
  echo "OK: data removed"
else
  echo "Fixing ownership of container-owned files..."
  sudo chown -R "$(id -u):$(id -g)" "$DATA_DIR"
  rm -rf "$DATA_DIR"/*
  echo "OK: data removed"
fi
