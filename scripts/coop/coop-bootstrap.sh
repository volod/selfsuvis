#!/usr/bin/env bash
# Bootstrap the coop stack: ensure `data/.env`, create bind-mount dirs,
# generate Mosquitto TLS + MQTT users if missing, then run docker compose.
#
# Usage:
#   ./scripts/coop-bootstrap.sh
#   APP_ENV=dev ./scripts/coop-bootstrap.sh up -d
#   ./scripts/coop-bootstrap.sh logs -f

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"
project_cd_root

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/coop-bootstrap.sh [compose-args...]

Bootstrap the coop stack and then delegate to coop-compose.
If no arguments are passed, this runs: `./scripts/coop/coop-compose.sh up -d`

Examples:
  ./scripts/coop-bootstrap.sh
  APP_ENV=dev ./scripts/coop-bootstrap.sh up -d
  ./scripts/coop-bootstrap.sh logs -f
EOF
  exit 0
fi

ENV="$(project_default_app_env)"

# Generate data/.env from template if missing
if [[ ! -f "$(project_env_file)" ]]; then
  project_log "Creating data/.env from env/${ENV}.env (first run)"
  "$PROJECT_ROOT_DIR/scripts/coop/coop-env.sh" "$ENV"
fi

# Ensure bind-mount directories exist
"$PROJECT_ROOT_DIR/scripts/coop/coop-data-dirs.sh"

# Generate Mosquitto TLS certs if missing
if [[ ! -f "$PROJECT_ROOT_DIR/config/coop/mosquitto/certs/server.key" ]]; then
  project_log "Generating Mosquitto TLS certs"
  HOST="localhost"
  project_load_env_optional
  HOST="${OR_HOSTNAME:-localhost}"
  "$PROJECT_ROOT_DIR/scripts/coop/coop-mosquitto-tls.sh" "$HOST"
fi

# Initialise MQTT users if pwfile is missing
if [[ ! -f "$PROJECT_ROOT_DIR/config/coop/mosquitto/pwfile" ]]; then
  project_log "Initializing Mosquitto users"
  "$PROJECT_ROOT_DIR/scripts/coop/coop-mqtt-users.sh"
fi

# Start the stack
if [[ $# -eq 0 ]]; then
  exec "$PROJECT_ROOT_DIR/scripts/coop/coop-compose.sh" up -d
else
  exec "$PROJECT_ROOT_DIR/scripts/coop/coop-compose.sh" "$@"
fi
