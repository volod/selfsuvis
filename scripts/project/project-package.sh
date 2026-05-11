#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/project/project-package.sh

Creates a tarball of the current repository while excluding generated runtime data and secrets.
Optional:
  PROJECT_PACKAGE_NAME=<name>
EOF
  exit 0
fi

PROJ_NAME="${PROJECT_PACKAGE_NAME:-$(basename "$PROJECT_ROOT_DIR")}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${PROJECT_ROOT_DIR}/${PROJ_NAME}-${TS}.tar.gz"
project_cd_root
tar \
  --exclude=".git" \
  --exclude="*.tar.gz" \
  --exclude="env/*.local.env" \
  --exclude="data/" \
  --exclude="config/coop/mosquitto/pwfile" \
  --exclude="config/coop/mosquitto/pwfile.tmp" \
  --exclude="config/coop/mosquitto/certs/*.key" \
  -czf "$OUT" \
  -C "$(dirname "$PROJECT_ROOT_DIR")" \
  "$PROJ_NAME"
sha256sum "$OUT"
project_log "Created package: $OUT"
