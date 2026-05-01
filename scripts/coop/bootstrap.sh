#!/usr/bin/env bash
# Bootstrap: generate data/.env if missing, create data dirs, TLS certs,
# MQTT users, then start the stack. PUID/PGID are set dynamically from the
# current user (never stored in data/.env).
#
# Usage:
#   ./scripts/bootstrap.sh [compose-args...]
#   ./scripts/bootstrap.sh                       # up -d (prod)
#   APP_ENV=dev ./scripts/bootstrap.sh up -d     # dev environment
#   ./scripts/bootstrap.sh logs -f

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV="${APP_ENV:-prod}"

# Generate data/.env from template if missing
if [[ ! -f "$ROOT_DIR/data/.env" ]]; then
  echo "Creating data/.env from env/${ENV}.env (first run)..."
  "$ROOT_DIR/scripts/coop/gen-env.sh" "$ENV"
fi

# Ensure bind-mount directories exist
"$ROOT_DIR/scripts/coop/ensure_data_dirs.sh"

# Generate Mosquitto TLS certs if missing
if [[ ! -f "$ROOT_DIR/config/coop/mosquitto/certs/server.key" ]]; then
  echo "Generating Mosquitto TLS certs..."
  HOST="localhost"
  if [[ -f "$ROOT_DIR/data/.env" ]]; then
    set -a; source "$ROOT_DIR/data/.env"; set +a
    HOST="${OR_HOSTNAME:-localhost}"
  fi
  "$ROOT_DIR/scripts/coop/gen_mosquitto_selfsigned_tls.sh" "$HOST"
fi

# Initialise MQTT users if pwfile is missing
if [[ ! -f "$ROOT_DIR/config/coop/mosquitto/pwfile" ]]; then
  echo "Initializing Mosquitto users..."
  "$ROOT_DIR/scripts/coop/init_mosquitto_users.sh"
fi

# Start the stack
if [[ $# -eq 0 ]]; then
  exec "$ROOT_DIR/scripts/coop/compose.sh" up -d
else
  exec "$ROOT_DIR/scripts/coop/compose.sh" "$@"
fi
