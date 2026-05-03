#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper: keep the legacy entrypoint, but default to the same
# fully-wired local setup path, including optional Utilyze installation unless
# the caller explicitly passes --no-utilyze.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/selfsuvis-setup.sh" "$@"
