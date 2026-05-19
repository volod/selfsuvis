#!/usr/bin/env bash
# Download geronimobasso/drone-audio-detection-samples and split into
# train / val / test subdirectories under data/drone-audio-data/.
#
# Usage:
#   ./scripts/split_drone_audio_data.sh [options]
#
# Options (passed through to the Python script):
#   --data-dir PATH         Output directory (default: data/drone-audio-data)
#   --sr N                  Target sample rate Hz (default: 22050)
#   --val-frac F            Fraction of data for validation (default: 0.15)
#   --test-frac F           Fraction of data for test (default: 0.10)
#   --max-per-class N       Max samples per class (0 = unlimited)
#
# Requirements:
#   pip install datasets soundfile
#   or:  pip install "selfsuvis[audio]"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../shared/common.sh"

project_run_python_module selfsuvis.scripts.split_drone_audio_data "$@"
