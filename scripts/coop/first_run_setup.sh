#!/usr/bin/env bash
# Print a credentials summary. Delegates data/.env generation to gen-env.sh.
#
# Usage:
#   ./scripts/first_run_setup.sh           # generate data/.env (prod) + print summary
#   ./scripts/first_run_setup.sh --list    # print credentials from existing data/.env

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="$ROOT_DIR/data/.env"

print_credentials() {
  echo ""
  echo "=============================================="
  echo "  Stack A Pilot - Credentials Summary"
  echo "=============================================="
  echo ""
  echo "Service URLs:"
  echo "  OpenRemote:     https://${OR_HOSTNAME:-localhost}"
  echo "  ChirpStack:     http://localhost:${CHIRPSTACK_UI_PORT:-8080}"
  echo "  ChirpStack API: http://localhost:${CHIRPSTACK_REST_PORT:-8090}"
  echo "  Frigate:        http://localhost:${FRIGATE_PORT:-8971}"
  echo "  MQTT (TLS):     localhost:${MOSQUITTO_MQTTS_PORT:-8883}"
  echo ""
  echo "Credentials:"
  echo "  OpenRemote admin:    ${OR_ADMIN_PASSWORD:-<not set>}"
  echo "  MQTT health user:    ${MOSQUITTO_HEALTH_USER:-health} / ${MOSQUITTO_HEALTH_PASSWORD:-<not set>}"
  echo "  ChirpStack MQTT:     ${CHIRPSTACK_MQTT_USERNAME:-chirpstack} / ${CHIRPSTACK_MQTT_PASSWORD:-<not set>}"
  echo "  ChirpStack GW MQTT:  ${CHIRPSTACK_GWBRIDGE_MQTT_USERNAME:-chirpstack_gw} / ${CHIRPSTACK_GWBRIDGE_MQTT_PASSWORD:-<not set>}"
  echo "  Frigate MQTT:        frigate / ${FRIGATE_MQTT_PASSWORD:-<not set>}"
  echo "  ChirpStack API key:  ${CHIRPSTACK_API_SECRET:0:8}... (base64)"
  echo "  ChirpStack DB:       ${CHIRPSTACK_PG_USER:-chirpstack} / ${CHIRPSTACK_PG_PASSWORD:-<not set>}"
  echo ""
  echo "Passwords stored in: data/.env"
  echo "=============================================="
  echo ""
}

if [[ "${1:-}" == "--list" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: data/.env not found. Run './scripts/gen-env.sh' to generate." >&2
    exit 1
  fi
  set -a; source "$ENV_FILE"; set +a
  print_credentials
  exit 0
fi

# Generate data/.env if missing
if [[ ! -f "$ENV_FILE" ]]; then
  "$ROOT_DIR/scripts/coop/gen-env.sh" "${APP_ENV:-prod}"
fi

set -a; source "$ENV_FILE"; set +a
print_credentials
