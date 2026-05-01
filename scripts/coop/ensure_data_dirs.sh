#!/usr/bin/env bash
# Create data directories for bind-mounted volumes.
# Run before first `docker compose up` so the user running compose owns the dirs.
#
# Usage:
#   ./scripts/ensure_data_dirs.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Load DATA_DIR from data/.env if present (PUID/PGID are set dynamically by compose.sh)
DATA_DIR="./data"
if [[ -f "$ROOT_DIR/data/.env" ]]; then
  set -a
  source "$ROOT_DIR/data/.env"
  set +a
  DATA_DIR="${DATA_DIR:-./data}"
fi

# Resolve relative path from project root
if [[ "$DATA_DIR" != /* ]]; then
  DATA_DIR="$ROOT_DIR/${DATA_DIR#./}"
fi

# postgresql and manager use named volumes, not bind mounts
DIRS=(
  "$DATA_DIR/proxy"
  "$DATA_DIR/mosquitto/data"
  "$DATA_DIR/mosquitto/log"
  "$DATA_DIR/chirpstack-postgres"
  "$DATA_DIR/chirpstack-redis"
  "$DATA_DIR/frigate-media"
  "$DATA_DIR/prometheus"
)

for d in "${DIRS[@]}"; do
  if [[ ! -d "$d" ]]; then
    mkdir -p "$d"
    echo "Created: $d"
  fi
done

# Ensure mosquitto data/log dirs are owned by current user (PUID) for non-root container
OWNER_UID=$(id -u)
OWNER_GID=$(id -g)
for d in "$DATA_DIR/mosquitto/data" "$DATA_DIR/mosquitto/log"; do
  if [[ -d "$d" ]] && [[ -O "$d" ]]; then
    :  # Already owned by us
  elif [[ -d "$d" ]]; then
    echo "Fixing ownership of $d for non-root mosquitto..."
    chown -R "$OWNER_UID:$OWNER_GID" "$d" 2>/dev/null || sudo chown -R "$OWNER_UID:$OWNER_GID" "$d"
  fi
  chmod 0777 "$d"
done

echo "Data directories ready under: $DATA_DIR"
