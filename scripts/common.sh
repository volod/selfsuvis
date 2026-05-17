#!/usr/bin/env bash

# Shared helpers for project shell scripts.
# Source this file from scripts/*.sh.

if [[ -n "${SELFSUVIS_SCRIPTS_COMMON_SOURCED:-}" ]]; then
  return 0
fi
readonly SELFSUVIS_SCRIPTS_COMMON_SOURCED=1

readonly PROJECT_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT_DIR="$(cd "$PROJECT_SCRIPTS_DIR/.." && pwd)"
readonly PROJECT_ENV_FILE="$PROJECT_ROOT_DIR/data/.env"
readonly PROJECT_COOP_COMPOSE_FILE="$PROJECT_ROOT_DIR/docker/coop/docker-compose.coop.yml"

project_root_dir() {
  printf '%s\n' "$PROJECT_ROOT_DIR"
}

project_env_file() {
  printf '%s\n' "$PROJECT_ENV_FILE"
}

project_coop_compose_file() {
  printf '%s\n' "$PROJECT_COOP_COMPOSE_FILE"
}

project_cd_root() {
  cd "$PROJECT_ROOT_DIR"
}

project_log() {
  printf '[scripts] %s\n' "$*"
}

project_warn() {
  printf '[scripts] WARN: %s\n' "$*" >&2
}

project_die() {
  printf '[scripts] ERROR: %s\n' "$*" >&2
  exit 1
}

project_have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

project_require_cmd() {
  project_have_cmd "$1" || project_die "Required command not found: $1"
}

project_load_env_optional() {
  if [[ -f "$PROJECT_ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$PROJECT_ENV_FILE"
    set +a
  fi
}

project_load_env_required() {
  [[ -f "$PROJECT_ENV_FILE" ]] || project_die "data/.env not found. Run './scripts/coop-env.sh' first."
  project_load_env_optional
}

project_default_app_env() {
  printf '%s\n' "${APP_ENV:-prod}"
}

project_data_dir() {
  local data_dir="./data"
  project_load_env_optional
  data_dir="${DATA_DIR:-$data_dir}"
  if [[ "$data_dir" != /* ]]; then
    data_dir="$PROJECT_ROOT_DIR/${data_dir#./}"
  fi
  printf '%s\n' "$data_dir"
}

project_export_runtime_ids() {
  export PUID
  export PGID
  PUID="$(id -u)"
  PGID="$(id -g)"
}

project_coop_compose() {
  project_export_runtime_ids
  exec docker compose \
    --project-directory "$PROJECT_ROOT_DIR" \
    --env-file "$PROJECT_ENV_FILE" \
    -f "$PROJECT_COOP_COMPOSE_FILE" \
    "$@"
}

project_python_bin() {
  if [[ -x "$PROJECT_ROOT_DIR/.venv/bin/python" ]]; then
    printf '%s\n' "$PROJECT_ROOT_DIR/.venv/bin/python"
  elif project_have_cmd python3; then
    printf '%s\n' "python3"
  elif project_have_cmd python; then
    printf '%s\n' "python"
  else
    project_die "Python interpreter not found. Create .venv or install python3."
  fi
}

project_run_python_module() {
  local module="$1"
  shift
  local python_bin
  python_bin="$(project_python_bin)"
  PYTHONPATH="$PROJECT_ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" -m "$module" "$@"
}
