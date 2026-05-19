#!/usr/bin/env bash
# Print coop stack credentials from `.data/.env`.
#
# Usage:
#   ./scripts/coop-credentials.sh
#   ./scripts/coop-credentials.sh --list

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../shared/common.sh"
project_cd_root

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/coop-credentials.sh [--list]

Prints the current coop stack credentials summary.
If `.data/.env` is missing, it is generated with prod defaults first.
EOF
  exit 0
fi

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
  echo "Passwords stored in: .data/.env"
  echo "=============================================="
  echo ""
}

if [[ "${1:-}" == "--list" ]]; then
  project_load_env_required
  print_credentials
  exit 0
fi

# Generate .data/.env if missing
if [[ ! -f "$(project_env_file)" ]]; then
  "$PROJECT_ROOT_DIR/scripts/coop/coop-env.sh" "$(project_default_app_env)"
fi

project_load_env_required
print_credentials
