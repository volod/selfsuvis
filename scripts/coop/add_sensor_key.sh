#!/usr/bin/env bash
# Provision a sensor API key. Usage: ./scripts/add_sensor_key.sh --sensor-id <id> [--scopes ingest]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../shared/common.sh"
python -m selfsuvis.scripts.add_sensor_key "$@"
