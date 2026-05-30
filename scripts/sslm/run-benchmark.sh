#!/usr/bin/env bash
# Wraps: sslm sequential  (see src/sslm/README.md for suite reference)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${SSLM_VENV:-$ROOT/.venv-sslm}"

if [[ ! -x "$VENV/bin/sslm" ]]; then
  echo "sslm venv not found at $VENV -- run: scripts/sslm/setup-venv.sh eval,dashboard" >&2
  exit 1
fi

# Load .env from project root so HF_TOKEN and other secrets are available.
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT/.env"
  set +a
fi

# HuggingFace Hub accepts both HF_TOKEN (new) and HUGGING_FACE_HUB_TOKEN (legacy).
# Normalise: if only one is set, mirror it to the other so both the host-side
# prefetch (uses huggingface_hub) and the container (vllm) can authenticate.
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi

export SSLM_PROJECT_ROOT="${SSLM_PROJECT_ROOT:-$ROOT}"
# Default to the system HF cache so pipeline-cached models are reused.
SYSTEM_HF_CACHE="$HOME/.cache/huggingface"
export SSLM_HF_CACHE="${SSLM_HF_CACHE:-$SYSTEM_HF_CACHE}"
cd "$ROOT"

# ── cleanup prompt ────────────────────────────────────────────────────────────
# Dangling images (untagged layers left by prior builds) accumulate at ~40 GB
# per image version.  Offer to prune them before starting a new run.
_maybe_prune_dangling() {
  local ids
  ids=$(docker images -f "dangling=true" -q 2>/dev/null || true)
  [[ -z "$ids" ]] && return 0

  local count
  count=$(printf '%s\n' "$ids" | wc -l | tr -d ' ')
  printf '\n[sslm] %s dangling Docker image(s) from previous builds:\n' "$count"
  docker images -f "dangling=true" \
    --format "  {{.ID}}  created {{.CreatedSince}}  {{.Size}}" 2>/dev/null || true
  printf '\nRemove them to free disk space? [y/N] '
  read -r _answer
  case "$_answer" in
    [yY]|[yY][eE][sS])
      docker image prune -f
      printf '[sslm] Dangling images removed.\n\n'
      ;;
    *)
      printf '[sslm] Keeping dangling images.\n\n'
      ;;
  esac
}

_maybe_prune_dangling

# Inject --limit from SSLM_QUICK_LIMIT env var (set in .env).
# Placed before $@ so an explicit --limit on the command line takes precedence.
_LIMIT_ARGS=()
if [[ -n "${SSLM_QUICK_LIMIT:-}" ]]; then
  _LIMIT_ARGS=(--limit "${SSLM_QUICK_LIMIT}")
fi

exec "$VENV/bin/sslm" sequential \
  --models zaya1-8b,qwen3-8b \
  --suite open_llm_v2 \
  --compose-file "$ROOT/.data/sslm/docker-compose.generated.yml" \
  --results-dir "$ROOT/.data/sslm/results" \
  "${_LIMIT_ARGS[@]}" \
  "$@"
