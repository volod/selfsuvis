#!/usr/bin/env bash
# Run the coop docker-compose stack with runtime `PUID`/`PGID`.
#
# Usage:
#   ./scripts/coop-compose.sh up -d
#   APP_ENV=dev ./scripts/coop-compose.sh up -d
#   APP_ENV=test ./scripts/coop-compose.sh down

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../shared/common.sh"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -eq 0 ]]; then
  cat <<'EOF'
Usage: ./scripts/sencoop/sencoop-compose.sh <docker-compose-args...>

Examples:
  ./scripts/sencoop/sencoop-compose.sh up -d
  ./scripts/sencoop/sencoop-compose.sh logs -f
  APP_ENV=test ./scripts/sencoop/sencoop-compose.sh down
EOF
  [[ $# -eq 0 ]] && exit 1 || exit 0
fi

[[ -f "$(project_env_file)" ]] || project_die ".data/.env not found. Run './scripts/sencoop/sencoop-env.sh' or './scripts/sencoop/sencoop-bootstrap.sh' first."
project_coop_compose "$@"
