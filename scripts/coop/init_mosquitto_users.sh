#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$ROOT_DIR/data/.env" ]]; then
  echo "ERROR: data/.env not found. Run './scripts/gen-env.sh' to generate." >&2
  exit 1
fi
set -a
source "$ROOT_DIR/data/.env"
set +a

mkdir -p config/coop/mosquitto
PWFILE="config/coop/mosquitto/pwfile"
TMP="config/coop/mosquitto/pwfile.tmp"
rm -f "$TMP"

add_user() {
  local user="$1"
  local pass="$2"
  if [[ -z "${pass}" || "${pass}" == REPLACE_* ]]; then
    echo "ERROR: password for user '$user' not set" >&2
    exit 1
  fi
  if [[ ! -f "$TMP" ]]; then
    docker run --rm --user "$(id -u):$(id -g)" -v "$ROOT_DIR/config/coop/mosquitto:/mosquitto/config" eclipse-mosquitto:2 \
      mosquitto_passwd -b -c /mosquitto/config/pwfile.tmp "$user" "$pass"
  else
    docker run --rm --user "$(id -u):$(id -g)" -v "$ROOT_DIR/config/coop/mosquitto:/mosquitto/config" eclipse-mosquitto:2 \
      mosquitto_passwd -b /mosquitto/config/pwfile.tmp "$user" "$pass"
  fi
}

add_user "${MOSQUITTO_HEALTH_USER:-health}" "${MOSQUITTO_HEALTH_PASSWORD:-}"
add_user "${CHIRPSTACK_MQTT_USERNAME:-chirpstack}" "${CHIRPSTACK_MQTT_PASSWORD:-}"
add_user "${CHIRPSTACK_GWBRIDGE_MQTT_USERNAME:-chirpstack_gw}" "${CHIRPSTACK_GWBRIDGE_MQTT_PASSWORD:-}"
add_user "frigate" "${FRIGATE_MQTT_PASSWORD:-}"

mv "$TMP" "$PWFILE"
chmod 0644 "$PWFILE"
echo "OK: generated $PWFILE"
