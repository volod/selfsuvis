#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJ_NAME="coop-stack-a-pilot"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT_DIR}/${PROJ_NAME}-${TS}.tar.gz"
cd "$ROOT_DIR"
tar \
  --exclude=".git" \
  --exclude="*.tar.gz" \
  --exclude="env/*.local.env" \
  --exclude="data/" \
  --exclude="config/coop/mosquitto/pwfile" \
  --exclude="config/coop/mosquitto/pwfile.tmp" \
  --exclude="config/coop/mosquitto/certs/*.key" \
  -czf "$OUT" \
  -C "$(dirname "$ROOT_DIR")" \
  "$PROJ_NAME"
sha256sum "$OUT"
echo "OK: $OUT"
