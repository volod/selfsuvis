#!/usr/bin/env bash
# Generate coop stack `data/.env` from the coop env templates.
#
# Usage:
#   ./scripts/coop-env.sh [prod|dev|test] [--force]

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"

ENV="prod"
FORCE=false

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/coop/coop-env.sh [prod|dev|test] [--force]

Generates `data/.env` for the coop stack only.
This is separate from root `.env` used by the main selfsuvis app setup.
EOF
  exit 0
fi

for arg in "$@"; do
  case "$arg" in
    prod|dev|test) ENV="$arg" ;;
    --force) FORCE=true ;;
    *) echo "Usage: $0 [prod|dev|test] [--force]" >&2; exit 1 ;;
  esac
done

TEMPLATE="$PROJECT_ROOT_DIR/src/selfsuvis/coop_pilot/env/${ENV}.env"
OUTPUT="$(project_env_file)"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "ERROR: Template not found: $TEMPLATE" >&2
  exit 1
fi

if [[ -f "$OUTPUT" ]] && [[ "$FORCE" != true ]]; then
  echo "ERROR: data/.env already exists. Pass --force to overwrite." >&2
  exit 1
fi

generate_password() {
  local len="${1:-16}"
  openssl rand -base64 "$len" | tr -d '\n/+=' | head -c "$len"
}

# data/ must exist before we can write into it
mkdir -p "$PROJECT_ROOT_DIR/data"

cp "$TEMPLATE" "$OUTPUT"

# Inject generated secrets — only replaces REPLACE_ME placeholders
sed -i "s|OR_ADMIN_PASSWORD=REPLACE_ME|OR_ADMIN_PASSWORD=$(generate_password 16)|"                                 "$OUTPUT"
sed -i "s|MOSQUITTO_HEALTH_PASSWORD=REPLACE_ME|MOSQUITTO_HEALTH_PASSWORD=$(generate_password 16)|"                 "$OUTPUT"
sed -i "s|CHIRPSTACK_PG_PASSWORD=REPLACE_ME|CHIRPSTACK_PG_PASSWORD=$(generate_password 16)|"                       "$OUTPUT"
sed -i "s|CHIRPSTACK_API_SECRET=REPLACE_ME|CHIRPSTACK_API_SECRET=$(openssl rand -base64 32 | tr -d '\n')|"         "$OUTPUT"
sed -i "s|CHIRPSTACK_MQTT_PASSWORD=REPLACE_ME|CHIRPSTACK_MQTT_PASSWORD=$(generate_password 16)|"                   "$OUTPUT"
sed -i "s|CHIRPSTACK_GWBRIDGE_MQTT_PASSWORD=REPLACE_ME|CHIRPSTACK_GWBRIDGE_MQTT_PASSWORD=$(generate_password 16)|" "$OUTPUT"
sed -i "s|FRIGATE_MQTT_PASSWORD=REPLACE_ME|FRIGATE_MQTT_PASSWORD=$(generate_password 16)|"                         "$OUTPUT"

project_log "Generated data/.env from env/${ENV}.env (APP_ENV=${ENV})"
project_log "Print credentials with: ./scripts/coop/coop-credentials.sh --list"
