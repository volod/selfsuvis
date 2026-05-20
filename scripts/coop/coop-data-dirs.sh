#!/usr/bin/env bash
# Create coop stack bind-mount directories under `$DATA_DIR`.
#
# Usage:
#   ./scripts/coop/coop-data-dirs.sh

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../shared/common.sh"
project_cd_root

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/coop-data-dirs.sh

Creates coop bind-mount directories and fixes Mosquitto dir ownership.
EOF
  exit 0
fi

DATA_DIR="$(project_data_dir)"

# postgresql and manager use named volumes, not bind mounts
DIRS=(
  "$DATA_DIR/proxy"
  "$DATA_DIR/mosquitto/data"
  "$DATA_DIR/mosquitto/log"
  "$DATA_DIR/coop/mosquitto/certs"
  "$DATA_DIR/coop/frigate"
  "$DATA_DIR/chirpstack-postgres"
  "$DATA_DIR/chirpstack-redis"
  "$DATA_DIR/frigate-media"
  "$DATA_DIR/prometheus"
)

for d in "${DIRS[@]}"; do
  if [[ ! -d "$d" ]]; then
    mkdir -p "$d"
    project_log "Created: $d"
  fi
done

# Ensure mosquitto data/log dirs are owned by current user (PUID) for non-root container
OWNER_UID=$(id -u)
OWNER_GID=$(id -g)
for d in "$DATA_DIR/mosquitto/data" "$DATA_DIR/mosquitto/log"; do
  if [[ -d "$d" ]] && [[ -O "$d" ]]; then
    :  # Already owned by us
  elif [[ -d "$d" ]]; then
    project_log "Fixing ownership of $d for non-root mosquitto"
    chown -R "$OWNER_UID:$OWNER_GID" "$d" 2>/dev/null || sudo chown -R "$OWNER_UID:$OWNER_GID" "$d"
  fi
  chmod 0777 "$d"
done

project_log "Data directories ready under: $DATA_DIR"
