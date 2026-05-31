#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../shared/common.sh"
project_cd_root

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/coop-mqtt-users.sh

Builds `.data/coop/mosquitto/pwfile` from credentials in `.data/.env`.
EOF
  exit 0
fi

project_load_env_required
project_require_cmd docker

_SECRETS_DIR="$(project_data_dir)/coop/mosquitto"
mkdir -p "$_SECRETS_DIR"
PWFILE="$_SECRETS_DIR/pwfile"
TMP="$_SECRETS_DIR/pwfile.tmp"
rm -f "$TMP"

add_user() {
  local user="$1"
  local pass="$2"
  if [[ -z "${pass}" || "${pass}" == REPLACE_* ]]; then
    echo "ERROR: password for user '$user' not set" >&2
    exit 1
  fi
  if [[ ! -f "$TMP" ]]; then
    docker run --rm --user "$(id -u):$(id -g)" -v "$_SECRETS_DIR:/mosquitto/secrets" eclipse-mosquitto:2 \
      mosquitto_passwd -b -c /mosquitto/secrets/pwfile.tmp "$user" "$pass"
  else
    docker run --rm --user "$(id -u):$(id -g)" -v "$_SECRETS_DIR:/mosquitto/secrets" eclipse-mosquitto:2 \
      mosquitto_passwd -b /mosquitto/secrets/pwfile.tmp "$user" "$pass"
  fi
}

add_user "${MOSQUITTO_HEALTH_USER:-health}" "${MOSQUITTO_HEALTH_PASSWORD:-}"
add_user "${CHIRPSTACK_MQTT_USERNAME:-chirpstack}" "${CHIRPSTACK_MQTT_PASSWORD:-}"
add_user "${CHIRPSTACK_GWBRIDGE_MQTT_USERNAME:-chirpstack_gw}" "${CHIRPSTACK_GWBRIDGE_MQTT_PASSWORD:-}"
add_user "frigate" "${FRIGATE_MQTT_PASSWORD:-}"

mv "$TMP" "$PWFILE"
chmod 0644 "$PWFILE"
project_log "Generated $PWFILE"
