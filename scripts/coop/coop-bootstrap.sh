#!/usr/bin/env bash
# Bootstrap the coop stack: ensure `.data/.env`, create bind-mount dirs,
# generate Mosquitto TLS + MQTT users if missing, then run docker compose.
#
# Usage:
#   ./scripts/coop-bootstrap.sh
#   APP_ENV=dev ./scripts/coop-bootstrap.sh up -d
#   ./scripts/coop-bootstrap.sh logs -f

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../shared/common.sh"
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

# Generate .data/.env from template if missing
if [[ ! -f "$(project_env_file)" ]]; then
  project_log "Creating .data/.env from env/${ENV}.env (first run)"
  "$PROJECT_ROOT_DIR/scripts/coop/coop-env.sh" "$ENV"
fi

# Ensure bind-mount directories exist
"$PROJECT_ROOT_DIR/scripts/coop/coop-data-dirs.sh"

project_load_env_optional
_DATA_DIR="$(project_data_dir)"

# Copy Frigate config template to .data/coop/frigate/ on first run
_FRIGATE_LIVE="$_DATA_DIR/coop/frigate/config.yml"
if [[ ! -f "$_FRIGATE_LIVE" ]]; then
  project_log "Copying Frigate config template to $_FRIGATE_LIVE"
  mkdir -p "$(dirname "$_FRIGATE_LIVE")"
  cp "$PROJECT_ROOT_DIR/config/coop/frigate/config.yml" "$_FRIGATE_LIVE"
  # Also copy go2rtc homekit config if present
  _GO2RTC="$PROJECT_ROOT_DIR/config/coop/frigate/go2rtc_homekit.yml"
  [[ -f "$_GO2RTC" ]] && cp "$_GO2RTC" "$_DATA_DIR/coop/frigate/go2rtc_homekit.yml"
fi

# Generate Mosquitto TLS certs if missing
if [[ ! -f "$_DATA_DIR/coop/mosquitto/certs/server.key" ]]; then
  project_log "Generating Mosquitto TLS certs"
  HOST="${OR_HOSTNAME:-localhost}"
  "$PROJECT_ROOT_DIR/scripts/coop/coop-mosquitto-tls.sh" "$HOST"
fi

# Initialise MQTT users if pwfile is missing
if [[ ! -f "$_DATA_DIR/coop/mosquitto/pwfile" ]]; then
  project_log "Initializing Mosquitto users"
  "$PROJECT_ROOT_DIR/scripts/coop/coop-mqtt-users.sh"
fi

# Start the stack
if [[ $# -eq 0 ]]; then
  exec "$PROJECT_ROOT_DIR/scripts/coop/coop-compose.sh" up -d
else
  exec "$PROJECT_ROOT_DIR/scripts/coop/coop-compose.sh" "$@"
fi
