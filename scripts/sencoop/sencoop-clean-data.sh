#!/usr/bin/env bash
# Remove coop stack bind-mounted data under `$DATA_DIR` while preserving `.data/.env`.
#
# Usage:
#   ./scripts/coop-clean-data.sh

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../shared/common.sh"
project_cd_root

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/coop-clean-data.sh

Stops the coop stack and deletes bind-mounted runtime data under `$DATA_DIR`.
Preserves `.data/.env`.
EOF
  exit 0
fi

DATA_DIR="$(project_data_dir)"

if [[ ! -d "$DATA_DIR" ]]; then
  project_log "Data directory does not exist: $DATA_DIR"
  exit 0
fi

project_log "Stopping containers"
"$PROJECT_ROOT_DIR/scripts/sencoop/sencoop-compose.sh" down 2>/dev/null || true

project_log "Removing data under $DATA_DIR (preserving .env)"
if rm -rf "$DATA_DIR"/* 2>/dev/null; then
  project_log "Data removed"
else
  project_warn "Fixing ownership of container-owned files"
  sudo chown -R "$(id -u):$(id -g)" "$DATA_DIR"
  rm -rf "$DATA_DIR"/*
  project_log "Data removed"
fi
