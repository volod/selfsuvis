#!/usr/bin/env bash
# Simulate and play (or save) physically-realistic drone audio.
#
# Models a drone as a moving point source applying:
#   - Inverse-square-law amplitude decay
#   - Atmospheric absorption
#   - Doppler pitch shift (emission-time interpolation)
#
# Usage:
#   ./scripts/play_drone_sound.sh [options]
#
# Scenarios:
#   flyover    Drone flies straight line; closest approach = --distance (default)
#   approach   Drone approaches from --distance, passes at 5 m, recedes
#   hover      Drone hovers at fixed --distance directly above mic
#   circle     Drone circles mic at orbit radius = --distance
#
# Examples:
#   # Drone flies over from 200 m at 10 m/s (classic counter-UAS test case)
#   ./scripts/play_drone_sound.sh --scenario flyover --distance 200 --speed 10
#
#   # Hovering drone at 30 m altitude
#   ./scripts/play_drone_sound.sh --scenario hover --distance 30 --duration 15
#
#   # High-speed approach from 500 m
#   ./scripts/play_drone_sound.sh --scenario approach --distance 500 --speed 20
#
#   # Save to file for offline testing / model validation
#   ./scripts/play_drone_sound.sh --scenario flyover --distance 100 --speed 15 \
#       --output data/reports/sim_flyover_100m.wav
#
# Options:
#   --scenario   flyover|approach|hover|circle  (default: flyover)
#   --distance   closest / hover / orbit distance in m (default: 200)
#   --speed      drone speed m/s (default: 10; ignored for hover)
#   --duration   simulation length in seconds (default: 30)
#   --source-db  drone dBSPL at 1 m reference (default: 85)
#   --speaker-ref-db estimated speaker dBSPL at full-scale / 100% OS volume (default: 85)
#   --system-volume  current OS output volume 0.0..1.0, or auto-detect (default: auto)
#   --mic-type    measurement|acoustic|embedded|headset|phone|unknown (default: unknown)
#   --player-type single-speaker|stereo-speakers|laptop|phone|headphones|unknown (default: unknown)
#   --probe-distance-m speaker-to-probe-mic distance during acoustic check (default: 1.0)
#   --placement-help print mic/speaker placement guidance and exit
#   --c          speed of sound m/s (default: 343)
#   --atm-db     atmospheric absorption dB/100m (default: 0.5)
#   --sample     path to a drone WAV file (default: first file in data/drone-audio-data)
#   --output     save to WAV file instead of playing
#   --data-dir   dataset cache directory (default: data/drone-audio-data)
#
# Playback requires:  pip install sounddevice
# Save to WAV uses scipy (already a project dependency)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

project_run_python_module selfsuvis.scripts.play_drone_sound "$@"
