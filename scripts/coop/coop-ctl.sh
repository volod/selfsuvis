#!/usr/bin/env bash
# Control the coop-pilot IoT edge stack.
# Works from the project repo (dev) and as /usr/local/bin/coop-ctl (symlink).
#
# Usage:
#   coop-ctl <command> [args]
#
# Commands:
#   start            Start all services (reads COOP_COMPOSE_PROFILES from .env)
#   stop             Stop all services
#   restart          Stop then start all services
#   status           Container status + live resource usage snapshot
#   logs [service]   Stream logs (optional: filter to one service)
#   ps               List containers
#   shell <service>  Open interactive shell inside a container
#   update           Pull latest images and recreate changed containers
#   config           Show resolved docker-compose configuration
#   env              Print active .env path and key settings
#
# Environment:
#   COOP_INSTALL_DIR   Override the install directory (set by systemd unit)

set -euo pipefail

# ── Locate real script dir (follow symlink) ───────────────────────────────────
_SELF="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null \
        || realpath "${BASH_SOURCE[0]}" 2>/dev/null \
        || echo "${BASH_SOURCE[0]}")"
_SCRIPTS_DIR="$(cd "$(dirname "$_SELF")" && pwd)"

# COOP_INSTALL_DIR override (useful for multi-install or systemd ExecStart env)
if [[ -n "${COOP_INSTALL_DIR:-}" ]]; then
  _SCRIPTS_DIR="${COOP_INSTALL_DIR}/scripts/coop"
fi

# shellcheck source=scripts/shared/common.sh
source "$_SCRIPTS_DIR/../shared/common.sh"

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { printf '[coop-ctl] %s\n' "$*"; }
die()  { printf '[coop-ctl] ERROR: %s\n' "$*" >&2; exit 1; }

_require_env() {
  [[ -f "$(project_env_file)" ]] \
    || die ".env not found at $(project_env_file). Run coop-bootstrap.sh first."
}

# ── Compose profile auto-detection ───────────────────────────────────────────
# Read COOP_COMPOSE_PROFILES from .env and export as COMPOSE_PROFILES.
# Falls back to COOP_METRICS_ENABLED for installs predating the bundle config.
_apply_compose_profiles() {
  _CTL_EF="$(project_env_file)"
  if [[ ! -f "$_CTL_EF" ]]; then return; fi
  _CTL_PROFILES="$(grep -E '^COOP_COMPOSE_PROFILES=' "$_CTL_EF" \
                   | cut -d= -f2 | tr -d '[:space:]' || true)"
  if [[ -n "$_CTL_PROFILES" ]]; then
    export COMPOSE_PROFILES="$_CTL_PROFILES"
    return
  fi
  # Backward compat: legacy flag written by older installs
  _CTL_METRICS="$(grep -E '^COOP_METRICS_ENABLED=' "$_CTL_EF" \
                  | cut -d= -f2 | tr -d '[:space:]' || true)"
  if [[ "$_CTL_METRICS" == "true" ]]; then
    export COMPOSE_PROFILES=metrics
  fi
}

# ── Compose wrapper (no exec -- retains process control for multi-step cmds) ──
_compose() {
  docker compose \
    --project-directory "$PROJECT_ROOT_DIR" \
    --env-file "$PROJECT_ENV_FILE" \
    -f "$PROJECT_COOP_COMPOSE_FILE" \
    "$@"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
CMD="${1:-}"
shift || true

case "$CMD" in

  start)
    _require_env
    _apply_compose_profiles
    log "Starting coop stack..."
    _compose up -d "$@"
    log "Stack running. Use 'coop-ctl status' or 'coop-ctl logs'."
    ;;

  stop)
    _apply_compose_profiles
    log "Stopping coop stack..."
    _compose down "$@"
    ;;

  restart)
    _require_env
    _apply_compose_profiles
    log "Restarting coop stack..."
    _compose down
    _compose up -d "$@"
    log "Stack restarted."
    ;;

  status)
    _apply_compose_profiles
    _compose ps
    printf '\n'
    # shellcheck disable=SC2046  # word splitting on ps -q output is intentional
    docker stats --no-stream \
      --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}" \
      $(_compose ps -q 2>/dev/null) 2>/dev/null || true
    ;;

  logs)
    _apply_compose_profiles
    _compose logs -f --tail=100 "$@"
    ;;

  ps)
    _apply_compose_profiles
    _compose ps "$@"
    ;;

  shell)
    [[ -n "${1:-}" ]] || die "Usage: coop-ctl shell <service>"
    _apply_compose_profiles
    _compose exec "$1" /bin/sh
    ;;

  update)
    _require_env
    _apply_compose_profiles
    log "Pulling latest images..."
    _compose pull
    log "Recreating changed containers..."
    _compose up -d --remove-orphans
    log "Update complete."
    ;;

  config)
    _apply_compose_profiles
    _compose config "$@"
    ;;

  env)
    _CTL_ENV_FILE="$(project_env_file)"
    log "Active env: $_CTL_ENV_FILE"
    [[ -f "$_CTL_ENV_FILE" ]] || die "Env file not found: $_CTL_ENV_FILE"
    # Show all non-secret lines; mask passwords and secrets
    grep -v -E '(PASSWORD|SECRET|TOKEN)=' "$_CTL_ENV_FILE" || true
    printf '\n'
    log "(Secrets masked. Use: sudo cat $_CTL_ENV_FILE)"
    ;;

  help|-h|--help)
    sed -n '2,21p' "$_SELF"
    exit 0
    ;;

  "")
    sed -n '2,21p' "$_SELF"
    exit 1
    ;;

  *)
    die "Unknown command: $CMD. Run 'coop-ctl help' for usage."
    ;;

esac
