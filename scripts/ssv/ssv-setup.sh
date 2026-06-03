#!/usr/bin/env bash
# =============================================================================
# ssv-setup.sh — one-shot bootstrap for a complete local run.
#
# WHAT THIS SCRIPT DOES (in order):
#
#   Step 0 — Check system prerequisites (ffmpeg, curl, git, uv)
#   Step 1 — Create Python virtual environment (.venv) and install all deps
#   Step 1a— Install sensor-specific Python packages (open3d, filterpy, …)
#   Step 2 — Download HuggingFace model weights for each pipeline step
#   Step 3 — Install Ollama and pull LLM/VLM sidecar models
#   Step 4 — Create .data/ layout, download test video, generate sensor sidecars
#   Step 5 — Print instructions to start Docker stack + run DB migration
#   Step 6 — Confirm test video path for run-command summary
#   Summary— Print the exact run command(s) for your configuration
#
# This script is SETUP ONLY — it does not start any services or containers.
# Docker and PostgreSQL are started separately after setup completes.
# Use --with-docker to also start the Docker stack in this script.
#
# USAGE:
#   ./scripts/ssv/ssv-setup.sh [flags]
#
# FLAGS:
#   (none)                Full setup — installs deps, models, test data
#   --with-docker         Also start Docker stack (Qdrant + PostgreSQL) and migrate
#   --no-ollama           Skip Ollama install and model pulls (use HF weights)
#   --with-utilyze        Install optional Utilyze GPU profiler (default)
#   --no-utilyze          Skip Utilyze install
#   --sensor-data-only    Download/generate sensor sidecars only; skip models
#
# ENVIRONMENT VARIABLES (set before running):
#   HF_TOKEN              HuggingFace API token
#                         Required for: Gemma weights (gated model)
#                         Not needed if you use Ollama for Gemma
#                         Get one at: https://huggingface.co/settings/tokens
#
#   OLLAMA_HOST           Ollama server base URL (default: http://localhost:11434)
#                         Override if Ollama runs on a different host/port
#
#   VLLM_PORT_QWEN        Port for Qwen2.5-VL vLLM server (default: 8010)
#   VLLM_PORT_UNIDRIVE    Port for UniDriveVLA vLLM server (default: 8030)
#
# EXAMPLES:
#   # Standard first-time setup on a GPU machine (no containers started):
#   ./scripts/ssv/ssv-setup.sh
#
#   # Setup + start Docker stack in one go:
#   ./scripts/ssv/ssv-setup.sh --with-docker
#
#   # Setup without Ollama (no GPU inference sidecars):
#   ./scripts/ssv/ssv-setup.sh --no-ollama
#
#   # Setup plus default Utilyze profiler install:
#   ./scripts/ssv/ssv-setup.sh
#
#   # Skip Utilyze on unsupported hosts:
#   ./scripts/ssv/ssv-setup.sh --no-utilyze
#
#   # Re-download sensor sample data only (already have models):
#   ./scripts/ssv/ssv-setup.sh --sensor-data-only
#
#   # With HuggingFace token for gated Gemma weights:
#   HF_TOKEN=hf_xxxx ./scripts/ssv/ssv-setup.sh
#
# AFTER SETUP:
#   Start services manually (or use --with-docker above):
#     make up
#     python -m selfsuvis.scripts.migrate_postgres
#   Then run the pipeline:
#     selfsuvis --mode local --input .data/videos/drone_mission.mp4 --no-qdrant
#
# TROUBLESHOOTING:
#   "Permission denied" on Docker:
#     sudo usermod -aG docker $USER   (then log out and back in)
#   "uv: command not found":
#     pip install uv
#   Ollama pull fails (timeout):
#     ollama pull <model>   (run manually, then re-run this script)
#   PostgreSQL migration fails on first try:
#     python -m selfsuvis.scripts.migrate_postgres   (postgres may need a few more seconds)
#   HuggingFace 401 on Gemma:
#     Visit https://huggingface.co/google/gemma-3-4b-it and accept the licence
#     then set HF_TOKEN and re-run
# =============================================================================

set -euo pipefail

# shellcheck source=scripts/shared/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../shared/common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,70p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

# -- Flags ---------------------------------------------------------------------
WITH_DOCKER=false
NO_OLLAMA=false
SENSOR_DATA_ONLY=false
WITH_UTILYZE=true

for arg in "$@"; do
  case "$arg" in
    --with-docker)      WITH_DOCKER=true ;;
    --no-docker)        WITH_DOCKER=false ;;   # kept for backwards compat; now the default
    --no-ollama)        NO_OLLAMA=true ;;
    --with-utilyze)     WITH_UTILYZE=true ;;
    --no-utilyze)       WITH_UTILYZE=false ;;
    --sensor-data-only) SENSOR_DATA_ONLY=true ;;
    *)
      echo "Unknown flag: $arg"
      echo "Valid flags: --with-docker  --no-ollama  --with-utilyze  --no-utilyze  --sensor-data-only"
      exit 1
      ;;
  esac
done

# -- Terminal colors -----------------------------------------------------------
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

log()     { echo -e "${GREEN}[setup]${RESET} $*"; }
section() {
  echo -e "\n${BOLD}${CYAN}======================================================${RESET}"
  echo -e "${BOLD}${CYAN}  $*${RESET}"
  echo -e "${BOLD}${CYAN}======================================================${RESET}"
}
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }

# Always run from the repo root regardless of where the script is invoked from
project_cd_root

# -- Load project .env files in precedence order --------------------------------
# Load all three layers so that DATA_DIR and HF_TOKEN are always resolved
# from the correct source.  Later files win over earlier ones.
#
#   .env              — committed project defaults (DATA_DIR, model names, etc.)
#   .data/.env        — stack overrides (sencoop / test infra; may not exist)
#   .data/.env.local  — machine-local dev overrides written by `make env`;
#                       highest precedence; never committed
#
# Variables already exported in the shell are NOT overwritten by any of these
# files — set them in the calling shell to force an override.
for _env_layer in ".env" ".data/.env" ".data/.env.local"; do
  if [[ -f "$_env_layer" ]]; then
    set -a   # export every variable defined from this point
    # shellcheck source=/dev/null
    source "$_env_layer"
    set +a
  fi
done
unset _env_layer

# -- Resolve best CUDA toolkit for the current GPU -----------------------------
# When multiple CUDA toolkits are installed (e.g. /usr/bin/nvcc at 12.0 and
# /usr/local/cuda-13.2/bin/nvcc at 13.2), the system PATH may point at an older
# one that does not support the GPU's compute capability (e.g. sm_120 Blackwell).
# This function finds the highest-version toolkit that satisfies the GPU's minimum.
_resolve_cuda_home() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo ""; return; }
  local gpu_cc
  gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
           | head -1 | tr -d ' ')
  local cc_major="${gpu_cc%%.*}"

  # Minimum CUDA toolkit version required per compute-capability major.
  # Older GPUs (sm_50–sm_90) are satisfied by any toolkit ≥ 10.
  local min_maj min_min
  case "$cc_major" in
    12) min_maj=12; min_min=8 ;;  # Blackwell sm_120+
    13) min_maj=13; min_min=0 ;;  # future
    *)  min_maj=10; min_min=0 ;;  # any modern toolkit is fine
  esac

  local best="" best_maj=0 best_min=0
  for cuda_dir in /usr/local/cuda-* /usr/local/cuda; do
    local nvcc_bin="${cuda_dir}/bin/nvcc"
    [[ -x "$nvcc_bin" ]] || continue
    local ver
    ver=$("$nvcc_bin" --version 2>/dev/null \
          | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | head -1)
    [[ -z "$ver" ]] && continue
    local maj min
    read -r maj min <<< "$ver"
    # Accept only if it satisfies the minimum AND is the best found so far.
    if { [[ "$maj" -gt "$min_maj" ]] || \
         { [[ "$maj" -eq "$min_maj" ]] && [[ "${min:-0}" -ge "$min_min" ]]; }; } && \
       { [[ "$maj" -gt "$best_maj" ]] || \
         { [[ "$maj" -eq "$best_maj" ]] && [[ "${min:-0}" -gt "$best_min" ]]; }; }; then
      best="$cuda_dir"
      best_maj="$maj"
      best_min="${min:-0}"
    fi
  done
  echo "$best"
}

_BEST_CUDA_HOME=$(_resolve_cuda_home)
if [[ -n "$_BEST_CUDA_HOME" ]]; then
  _NVCC_VER=$("${_BEST_CUDA_HOME}/bin/nvcc" --version 2>/dev/null \
              | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
  if [[ "${CUDA_HOME:-}" != "$_BEST_CUDA_HOME" ]]; then
    log "CUDA toolkit resolved: $_BEST_CUDA_HOME  (nvcc $_NVCC_VER — overrides PATH default)"
    export CUDA_HOME="$_BEST_CUDA_HOME"
    export PATH="$_BEST_CUDA_HOME/bin:$PATH"
  fi
fi
unset _BEST_CUDA_HOME _NVCC_VER

# -- Configuration --------------------------------------------------------------
# These can be overridden by environment variables before running the script.
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
VLLM_PORT_QWEN="${VLLM_PORT_QWEN:-8010}"
VLLM_PORT_UNIDRIVE="${VLLM_PORT_UNIDRIVE:-8030}"
HF_TOKEN="${HF_TOKEN:-}"

# Shortcuts — resolved after .venv is created
PYTHON=".venv/bin/python"
PIP=".venv/bin/pip"

# Data directory — honoured by all sub-steps below
_DATA_DIR="$(project_data_dir)"

# Derive and export all cache-related dirs from DATA_DIR so every subprocess
# (uv, pip, huggingface_hub, torch.hub, prepare_models.py) stores artefacts
# under DATA_DIR instead of falling back to repo-level .data/.cache or ~/.cache.
# Per-key .env overrides are honoured via the ${VAR:-default} expansion.
CACHE_DIR="${CACHE_DIR:-${_DATA_DIR}/.cache}"
export CACHE_DIR
export HF_HOME="${HF_HOME:-${CACHE_DIR}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_DIR}/torch}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_DIR}/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_DIR}}"

# =============================================================================
# SENSOR-DATA-ONLY SHORT-CIRCUIT
# When --sensor-data-only is passed, skip all model and environment steps.
# Useful when you already have .venv and model weights and only want fresh
# sensor test sidecars (e.g. after adding a new sensor modality).
# =============================================================================
if $SENSOR_DATA_ONLY; then
  section "Sensor data only mode — skipping model and environment setup"
  mkdir -p "$_DATA_DIR/videos" "$_DATA_DIR/sensors" "$_DATA_DIR/frames" \
           "$_DATA_DIR/tiles" "$_DATA_DIR/maps" "$_DATA_DIR/reports" "$_DATA_DIR/cache_test"
  bash scripts/ssv/ssv-prepare-sensor-data.sh "$_DATA_DIR/sensors"
  echo ""
  log "Done. Sensor sample data is in $_DATA_DIR/sensors/"
  _VB="$(ls "$_DATA_DIR/videos"/*.mp4 "$_DATA_DIR/videos"/*.mov 2>/dev/null | head -1 || true)"
  _VB="${_VB:+$(basename "${_VB%.*}")}"
  _VB="${_VB:-<video-basename>}"
  log "Sidecars are named after your video: $_DATA_DIR/sensors/step16_imu/${_VB}.imu.jsonl"
  exit 0
fi

# =============================================================================
# STEP 0: SYSTEM PREREQUISITES
#
# Check that the tools the pipeline depends on are available on PATH.
# Nothing is installed here automatically except 'uv' (the fast pip/venv tool
# used instead of plain pip for speed and reproducibility).
#
# Install missing tools on Ubuntu/Debian:
#   sudo apt-get install -y ffmpeg curl git
# =============================================================================
section "Step 0 — System prerequisites"

for cmd in ffmpeg curl git; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    warn "'$cmd' not found. Install it: sudo apt-get install -y $cmd"
  else
    log "$cmd OK: $(command -v "$cmd")"
  fi
done

# uv is the recommended Python package manager for this project.
# It is significantly faster than pip and creates isolated venvs correctly.
# Official install: https://github.com/astral-sh/uv
if ! command -v uv >/dev/null 2>&1; then
  log "uv not found — installing via pip..."
  pip install --quiet uv \
    || error "Failed to install uv. Install manually: pip install uv"
fi
log "uv OK: $(uv --version)"

if $WITH_UTILYZE; then
  section "Step 0a — Optional Utilyze profiler"
  bash scripts/install/install_utilyze.sh
fi

# =============================================================================
# STEP 1: PYTHON VIRTUAL ENVIRONMENT
#
# Creates .venv in the repo root and installs all production + dev dependencies
# from pyproject.toml extras (`vision,dev`). The install script auto-detects
# your CUDA version and installs the matching PyTorch wheels.
#
# If .venv already exists this step is skipped to avoid reinstalling everything
# on subsequent runs.  To force a clean reinstall:
#   rm -rf .venv && ./scripts/ssv/ssv-setup.sh
#
# Expected duration: 3–10 minutes on first run (downloads ~2–4 GB of wheels).
# =============================================================================
section "Step 1 — Python virtual environment (.venv)"

if [[ ! -d .venv ]]; then
  log "Creating .venv..."
  uv venv .venv

  # install_requirements.sh installs deps and selects the correct PyTorch wheel
  # for your CUDA version (falls back to CPU-only torch if no GPU found)
  log "Installing Python dependencies (may take 5–10 minutes)..."
  bash scripts/install/install_requirements.sh vision,dev .venv
  log ".venv ready."
else
  # Check whether the installed PyTorch supports the current GPU architecture.
  # If not (e.g. old cu121/cu126 wheels on a new Blackwell sm_120 GPU),
  # reinstall torch wheels even though the venv already exists.
  _TORCH_ARCH_OK=true
  if "$PYTHON" -c "
import sys, warnings
warnings.filterwarnings('ignore')
try:
    import torch
    if not torch.cuda.is_available():
        sys.exit(0)
    cap = torch.cuda.get_device_capability(0)
    arch = 'sm_{}{}'.format(cap[0], cap[1])
    if arch not in torch.cuda.get_arch_list():
        sys.exit(1)
except Exception:
    pass
sys.exit(0)
" 2>/dev/null; then
    log ".venv already exists and torch supports current GPU — skipping. (rm -rf .venv to reinstall)"
  else
    warn ".venv exists but installed PyTorch does not include kernels for current GPU."
    warn "Re-running install_requirements.sh to install matching torch wheels..."
    bash scripts/install/install_requirements.sh vision,dev .venv
    log "Torch wheels updated."
  fi
fi

# =============================================================================
# STEP 1a: SENSOR-SPECIFIC PYTHON PACKAGES
#
# These packages are not in pyproject.toml because they are optional —
# the core pipeline runs without them.  Install them here so all 35 sensor
# steps work out of the box.
#
# Packages and what each enables:
#   filterpy          — EKF/UKF/particle filter (Step 20: sensor fusion IMU+GPS)
#   open3d            — LiDAR point cloud I/O, ICP, visualisation (Step 13)
#   pyroomacoustics   — microphone array beamforming, DoA, GCC-PHAT (Step 19)
#   librosa           — audio feature extraction: MFCC, spectrograms (Step 19)
#   torchaudio        — PyTorch audio transforms, wav2vec2, HuBERT (Step 19)
#   sounddevice       — real-time audio capture from microphone (Step 19)
#   soundfile         — WAV/FLAC read-write (used by audio dataset prep, Step 32)
#   datasets          — HuggingFace datasets client (drone audio download, Step 32)
#   scipy             — signal processing: FFTs, CFAR, peak detection (Steps 9, 14)
#   rasterio          — multispectral GeoTIFF read/write (Step 11)
#   pyproj            — geodetic ↔ UTM/ENU coordinate transforms (Steps 15, 20)
#   metpy             — meteorological calculations: dew point, turbulence (Step 17)
#   smbus2            — I2C sensor communication: SCD41, SGP41, SHT45 (Step 18)
#   pyserial          — serial sensor communication: OPC-N3, miniPID (Step 18)
#   scikit-gstat      — variogram fitting and kriging for contamination maps (Step 18)
#   tonic             — PyTorch Dataset wrappers for event camera data (Step 12)
# =============================================================================
section "Step 1a — Sensor-specific Python packages"

log "Installing sensor modality libraries..."
# Use uv (not pip) so the existing numpy>=2 constraint from pyproject.toml is
# respected during resolution.  tonic and scikit-gstat pin numpy<2 in their
# metadata; installing them with uv alongside an explicit numpy>=2 forces uv
# to pick a compatible solution instead of silently downgrading numpy.
uv pip install --python .venv \
  "numpy>=2" \
  filterpy \
  "open3d>=0.18" \
  pyroomacoustics \
  librosa \
  torchaudio \
  sounddevice \
  "soundfile>=0.12" \
  "datasets>=2.18" \
  scipy \
  rasterio \
  pyproj \
  metpy \
  smbus2 \
  pyserial \
  scikit-gstat \
  tonic \
  || warn "Some sensor packages failed — non-fatal; sensor steps degrade gracefully."

# Some packages ignore the numpy>=2 pin and downgrade to 1.x at install time.
# opencv-python 4.13+ requires numpy>=2; restore it here unconditionally.
log "Enforcing numpy>=2 (required by opencv-python 4.13+)..."
uv pip install --python .venv "numpy>=2" \
  || warn "numpy>=2 enforcement failed — opencv-python may have import errors."

# onnxruntime-gpu (installed on CUDA machines by install_requirements.sh) replaces
# the 'onnxruntime' pip package by name, so pip's resolver may warn that
# 'onnxruntime>=1.18' is not satisfied even though the module is importable.
# Reinstall whichever variant is needed to make the import work cleanly.
if ! "$PYTHON" -c "import onnxruntime" 2>/dev/null; then
  log "onnxruntime not importable — reinstalling..."
  if "$PYTHON" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    uv pip install --python .venv "onnxruntime-gpu>=1.18.0" \
      || uv pip install --python .venv "onnxruntime>=1.18.0" \
      || warn "onnxruntime install failed — ONNX inference may not work."
  else
    uv pip install --python .venv "onnxruntime>=1.18.0" \
      || warn "onnxruntime install failed — ONNX inference may not work."
  fi
fi

# OpenRadar: FMCW radar signal processing (range-Doppler, CFAR, angle estimation)
# Installed separately because it is a git-only package not on PyPI.
# Required for Step 14 (radar).  Non-fatal if it fails.
if ! "$PYTHON" -c "import mmwave" 2>/dev/null; then
  log "Installing OpenRadar (FMCW radar signal processing)..."
  "$PIP" install --quiet \
    "git+https://github.com/PreSenseRadar/OpenRadar.git" 2>/dev/null \
    || warn "OpenRadar install failed — Step 14 (radar) will have limited functionality."
fi

log "Sensor packages installed."

# RF-DETR object detector / tracker — installed separately to catch venv-age gaps
if ! "$PYTHON" -c "import rfdetr" 2>/dev/null; then
  log "Installing RF-DETR..."
  "$PIP" install --quiet "rfdetr>=1.1.0" \
    || warn "RF-DETR install failed — Step P3 (Gemma tracking) will run detection only."
fi

# SAM3 (preferred) with SAM2 fallback — installed separately because:
#   1. sam3 pulls in decord/pycocotools which can fail on some systems
#   2. If either failed during the bulk dependency install, they need a retry
#   3. sam3 is the preferred backend; sam2 is the open fallback
if ! "$PYTHON" -c "import sam3" 2>/dev/null; then
  log "Installing SAM3 (preferred SAM backend)..."
  if "$PIP" install --quiet sam3 2>/dev/null; then
    log "  SAM3 ready."
  else
    warn "SAM3 install failed — installing SAM2 fallback..."
    "$PIP" install --quiet sam2 \
      || warn "SAM2 also failed — SAM segmentation will be disabled. Install manually: pip install sam2"
  fi
fi
# Ensure SAM2 is also available (used as explicit fallback in pipeline)
if ! "$PYTHON" -c "import sam2" 2>/dev/null; then
  log "Installing SAM2 (SAM fallback backend)..."
  "$PIP" install --quiet sam2 \
    || warn "SAM2 install failed — only SAM3 will be available."
fi

# Compile SAM2 CUDA extension (_C) if not already built.
# The PyPI wheel ships the source (csrc/connected_components.cu) but not the
# compiled .so; without it SAM2 logs a UserWarning about skipped post-processing.
if ! "$PYTHON" -c "
import sys, warnings
warnings.filterwarnings('ignore')
try:
    from sam2 import _C
    import torch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        arch = 'sm_{}{}'.format(cap[0], cap[1])
        if arch not in torch.cuda.get_arch_list():
            sys.exit(1)  # compiled against wrong arch — recompile
except ImportError:
    sys.exit(1)
" 2>/dev/null; then
  if "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log "Compiling SAM2 CUDA extension (_C)..."
    "$PYTHON" - <<'PYEOF' && log "  ✓ SAM2 _C compiled" || warn "SAM2 _C compilation failed — post-processing will be limited"
import os
from torch.utils.cpp_extension import load
sam2_dir = os.path.dirname(__import__('sam2').__file__)
csrc = os.path.join(sam2_dir, 'csrc', 'connected_components.cu')
load(name='_C', sources=[csrc], extra_cuda_cflags=['-O2', '-DCUDA_HAS_FP16=1'],
     build_directory=sam2_dir, verbose=False)
PYEOF
  else
    log "  Skipping SAM2 CUDA extension (no GPU available)"
  fi
fi

# =============================================================================
# STEP 2: HUGGINGFACE MODEL WEIGHTS
#
# Downloads model weights for each vision/language pipeline step.
# Models are cached in ~/.cache/huggingface/ (or HF_HOME if set).
# Re-running this step is safe — already-cached models are skipped.
#
# Steps and their models:
#   Core    — OpenCLIP ViT-H-14 (LAION-2B) + DINOv3 ViT-B/14
#   Step  4 — Florence-2-large (captioning)             ~1.5 GiB
#   Step  5 — Whisper large-v3-turbo (ASR)              ~1.6 GiB
#   Step  6 — OCR model (auto-selected by VRAM)         ~1–4 GiB
#   Step  7 — Apple DepthPro (monocular depth)          ~2.5 GiB
#   Step  8 — HF RT-DETR / Grounding DINO               ~1–2 GiB
#   Step 21 — YOLO11l detection                         ~48 MB
#   Step 21 — SAM3 / SAM2 segmentation                  ~2.5 GiB
#   Step 23 — Cosmos-1.0 world model (video embeddings) ~9 GiB  (gated)
#   Step 25 — UniDriveVLA expert analysis               ~4 GiB
#   Step 3/22 — Gemma 4 open-weight (gated; needs HF_TOKEN)  ~8 GiB
#
# Total: approximately 35–50 GiB on first run.
# Subsequent runs only re-download missing files.
#
# HuggingFace token setup (required for Gemma + Cosmos):
#   1. Go to https://huggingface.co/settings/tokens
#   2. Create a token with "read" scope
#   3. Accept the model licence on the model card page
#   4. Set: export HF_TOKEN=hf_xxxx
# =============================================================================
section "Step 2 — HuggingFace model weights"

# Core: OpenCLIP + DINOv3 (always required — used by Steps 1–2 for all embeddings)
log "Downloading OpenCLIP + DINOv3 (core embeddings)..."
"$PYTHON" -m selfsuvis.scripts.prepare_models --clip --dino

# Step 4: Florence-2-large for per-keyframe scene captioning.
# Loads locally into the same GPU process as DINOv3 / CLIP.
# If Ollama is running and VRAM is tight, the pipeline evicts Ollama before loading.
log "Downloading Florence-2 (Step 4 — scene captioning)..."
"$PYTHON" -m selfsuvis.scripts.prepare_models --florence

# Step 5: Whisper large-v3-turbo for audio transcription.
# Aligned subtitle text is injected into Qwen captioning prompts (Step 24).
log "Downloading Whisper ASR (Step 5 — speech-to-text)..."
"$PYTHON" -m selfsuvis.scripts.prepare_models --whisper

# Step 6: OCR model auto-selected by available VRAM.
# On 8 GB VRAM → Phi-3.5-mini-instruct; on 16+ GB → DeepSeek-VL.
log "Downloading OCR model (Step 6 — text extraction from frames)..."
"$PYTHON" -m selfsuvis.scripts.prepare_models --ocr

# Step 7: Apple DepthPro for monocular per-pixel depth estimation.
# Depth percentiles are stored in frame_facts_json["depth"] and used by
# sensor fusion (Step 20) to cross-validate LiDAR range measurements.
log "Downloading Depth model (Step 7 — monocular depth estimation)..."
"$PYTHON" -m selfsuvis.scripts.prepare_models --depth

# Step 8: HuggingFace RT-DETR or Grounding DINO for open-vocabulary detection.
# This is a second detection pass separate from YOLO (Step 21).
log "Downloading RT-DETR / Grounding DINO (Step 8 — HF object detection)..."
"$PYTHON" -m selfsuvis.scripts.prepare_models --detection

# Step 21: YOLO11l — priority-aware detection (human > vehicle > artificial > other).
# Weights are tiny (~48 MB) but needed before SAM mask refinement can run.
log "Downloading YOLO11l (Step 21 — object detection)..."
"$PYTHON" -m selfsuvis.scripts.prepare_models --yolo

# Step 21: SAM3 / SAM2 — refines each YOLO bounding box into a pixel-level mask.
# Also used by Gemma directed tracking (Step 22) for SAM-prompted segmentation.
log "Downloading SAM3/SAM2 (Step 21 — segmentation masks)..."
"$PYTHON" -m selfsuvis.scripts.prepare_models --sam

# Step 23: Cosmos-1.0 world model — encodes entire video clips as temporal embeddings.
# Used for scene-level search and training data selection.
# NOTE: This model is gated on HuggingFace. If HF_TOKEN is not set, this step
# will prompt you to accept the licence and authenticate.
log "Downloading Cosmos world model (Step 23 — video clip embeddings)..."
"$PYTHON" -m selfsuvis.scripts.prepare_models --world-model \
  || warn "World model download failed — Step 23 will be skipped. Re-run with HF_TOKEN set."

# Step 25: UniDriveVLA — expert autonomous-driving analysis.
# Produces four blocks: understanding / perception / planning / mixture_of_experts.
# Source: https://github.com/xiaomi-research/unidrivevla
log "Downloading UniDriveVLA (Step 25 — expert driving analysis)..."
"$PYTHON" -m selfsuvis.scripts.prepare_models --unidrive \
  || warn "UniDriveVLA download failed — Step 25 will be skipped."

# Step 3 / 22: Gemma 4 open-weight (gated, requires HF_TOKEN).
# This downloads the weights for local embedding mode (GemmaEmbedder).
# If you prefer to use Ollama for generative mode, you can skip this —
# Ollama pulls are handled in Step 3 and do not require HF_TOKEN.
if [[ -n "$HF_TOKEN" ]]; then
  log "Downloading Gemma 4 open-weight (Steps 3, 22) — requires HF_TOKEN (~8 GiB)..."
  HF_TOKEN="$HF_TOKEN" "$PYTHON" -m selfsuvis.scripts.prepare_models --gemma
else
  warn "HF_TOKEN not set — skipping direct Gemma weight download."
  warn "Gemma will run via Ollama (Step 3 below).  To enable HF weights:"
  warn "  export HF_TOKEN=hf_xxxx  &&  ./scripts/ssv/ssv-setup.sh"
fi

log "Model weights done."

# =============================================================================
# STEP 3: OLLAMA SIDECAR LLM/VLM SERVERS
#
# Ollama serves the generative (vision) models as OpenAI-compatible endpoints.
# The pipeline calls these via HTTP — models run in a separate process from
# the Python pipeline, so VRAM is shared between them.
#
# Models pulled here and which pipeline steps use them:
#
#   gemma4:e4b    — ~5 GiB VRAM (4-bit quant, efficient)
#                   Step  3: Gemma multimodal frame analysis (scene change,
#                            clustering, CLIP+DINO comparison)
#                   Step 22: Gemma directed tracking structured JSON
#                   Step 35: Agentic flow audit / final reasoning
#
#   qwen2.5vl:7b  — ~5 GiB VRAM
#                   Step 24: Qwen detailed per-frame captioning with ASR context
#
# Both models fit simultaneously on 16 GB VRAM with 8-bit quantisation, or
# they can time-share a single GPU via Ollama's keep_alive mechanism.
#
# If you want higher quality at the cost of VRAM:
#   ollama pull gemma4:12b    (~13 GiB)
#   ollama pull gemma4:31b    (~20 GiB INT4, can offload layers to RAM)
#   ollama pull qwen2.5vl:72b (~45 GiB — requires 2+ GPUs or large RAM)
#
# Skip this step with --no-ollama if:
#   - You downloaded Gemma weights via HF_TOKEN above (step J runs locally)
#   - You want to run vLLM instead (see README vLLM section)
#   - You have no GPU and want CPU-only mode
# =============================================================================
section "Step 3 — Ollama sidecar LLM/VLM servers"

if $NO_OLLAMA; then
  warn "--no-ollama specified — skipping Ollama install and model pulls."
  warn "Steps 3, 22, 24, 35 will use direct HF weights or be skipped."
else
  # Install Ollama if not already present on this machine.
  # The installer script detects Linux/macOS and installs the correct binary.
  # Manual install: https://ollama.com/download
  if ! command -v ollama >/dev/null 2>&1; then
    log "Installing Ollama (https://ollama.com)..."
    curl -fsSL https://ollama.com/install.sh | sh
  else
    log "Ollama already installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
  fi

  # Start the Ollama daemon in the background if it is not already running.
  # The daemon listens on OLLAMA_HOST (default: http://localhost:11434).
  # It stays alive after this script exits — stop with: pkill ollama
  if ! curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    log "Starting Ollama daemon in background (log: /tmp/ollama.log)..."
    ollama serve &>/tmp/ollama.log &
    sleep 3   # give the daemon time to bind the port
  fi

  if curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    log "Ollama is running at ${OLLAMA_HOST}"
  else
    warn "Ollama does not appear to be running."
    warn "Check: curl ${OLLAMA_HOST}/api/tags"
    warn "Logs:  cat /tmp/ollama.log"
  fi

  # -- Gemma 4 4B (efficient multimodal, vision-capable) ----------------------
  # gemma4:e4b is the 4-bit efficient variant — best balance of quality and VRAM.
  # Vision-capable: accepts image + text input for frame analysis.
  # Used by: Step 3 (multimodal analysis), Step 22 (directed tracking), Step 35 (audit).
  log "Pulling gemma4:e4b (~5 GiB) — Steps 3, 22, 35..."
  ollama pull gemma4:e4b \
    || warn "gemma4:e4b pull failed. Retry manually: ollama pull gemma4:e4b"

  # -- Qwen2.5-VL 7B (detailed structured captioning) -------------------------
  # Qwen2.5-VL produces structured per-frame scene analysis with ASR subtitle
  # context injected into the prompt.  Used by Step 24 (detailed captioning).
  # If VRAM is tight, both Gemma and Qwen share the GPU via Ollama time-slicing.
  log "Pulling qwen2.5vl:7b (~5 GiB) — Step 24..."
  ollama pull qwen2.5vl:7b \
    || warn "qwen2.5vl:7b pull failed. Retry manually: ollama pull qwen2.5vl:7b"

  log "Ollama models ready. Verify: ollama list"
fi

# =============================================================================
# STEP 4: TEST DATA — DIRECTORIES, VIDEO, AND SENSOR SIDECARS
#
# Creates the full data/ layout expected by the pipeline, downloads a
# public-domain outdoor test video, and generates sensor sidecar files keyed
# to that video's basename.
#
# Directory layout created (all under DATA_DIR, default .data/):
#   .data/videos/      — input video(s)
#   .data/sensors/     — per-step sensor sidecars
#   .data/frames/      — keyframe output (written by pipeline)
#   .data/tiles/       — tile output (written by pipeline)
#   .data/maps/        — 3DGS / splat output (written by pipeline)
#   .data/reports/     — HTML mission summaries (written by pipeline)
#   .data/cache_test/  — integration-test cache volume
#
# Test video (auto-downloaded if .data/videos/ is empty):
#   Primary:  US Highway 60 drone flyover — real vehicles on a divided highway,
#             desert terrain, trees, road markings (~27 MB, archive.org, public domain)
#   Fallback: Archer Lodge suburban aerial — roads, buildings, trees, carpark (~15 MB)
#   Last:     15-second synthetic video generated by ffmpeg (network-free fallback)
#
# Sensor sidecars (Steps 9–19) are generated with the video's basename so they
# are immediately usable without renaming:
#   .data/sensors/step16_imu/<video>.imu.jsonl
#   .data/sensors/step17_atmospheric/<video>.env.jsonl
#   … etc.
#
# Steps requiring manual dataset download (free, registration only):
#   Step  9  — DeepSig RadioML 2018.01a    https://www.deepsig.ai/datasets
#   Step 10  — FLIR ADAS Thermal           https://www.flir.com/oem/adas/
#   Step 12  — N-Caltech101 / DSEC         https://www.garrickorchard.com/datasets/n-caltech101
#   Step 13  — KITTI odometry velodyne     https://www.cvlibs.net/datasets/kitti/
#   Step 14  — RADIATE radar               https://pro.hw.ac.uk/radiate/
#   Step 15  — CYGNSS GNSS-R DDMs          https://podaac.jpl.nasa.gov/dataset/CYGNSS_L1_V3.1
# =============================================================================
section "Step 4 — Test data (directories + video + sensor sidecars)"

# -- 4a: Create the full data/ directory layout --------------------------------
log "Creating $_DATA_DIR/ layout..."
mkdir -p \
  "$_DATA_DIR/videos" \
  "$_DATA_DIR/sensors" \
  "$_DATA_DIR/frames" \
  "$_DATA_DIR/tiles" \
  "$_DATA_DIR/maps" \
  "$_DATA_DIR/reports" \
  "$_DATA_DIR/cache_test"
log "Directories ready."

# -- 4b: Download a test video if .data/videos/ is empty -------------------------
_FOUND_VIDEO="$(ls "$_DATA_DIR/videos"/*.mp4 "$_DATA_DIR/videos"/*.mov \
                   "$_DATA_DIR/videos"/*.avi "$_DATA_DIR/videos"/*.mkv \
                   2>/dev/null | head -1 || true)"

if [[ -n "$_FOUND_VIDEO" ]]; then
  log "Test video found: $_FOUND_VIDEO"
else
  _DEST="$_DATA_DIR/videos/drone_mission.mp4"
  _DOWNLOADED=false

  # Primary and fallback: real drone footage from Internet Archive (public domain).
  #   1. US Highway 60 drone flyover — vehicles, divided highway, desert terrain (~27 MB)
  #   2. Archer Lodge suburban aerial — roads, buildings, trees, carpark (~15 MB)
  # Both are served via archive.org CDN with no authentication required.
  for _URL in \
    "https://archive.org/download/t11az-US60_Drone_Footage_-_May_21_2022/US60_Drone_Footage_-_May_21_2022.mp4" \
    "https://archive.org/download/pmpnc-Archer_Lodge_Aerial_Drone_Video__uEw2MMaDiA/Archer_Lodge_Aerial_Drone_Video__uEw2MMaDiA.mp4"
  do
    log "Trying: $(basename "$_URL") ..."
    if curl -fL --progress-bar --max-time 180 --retry 2 -o "$_DEST" "$_URL" 2>/dev/null \
        && [[ -s "$_DEST" ]]; then
      _FOUND_VIDEO="$_DEST"
      _DOWNLOADED=true
      break
    fi
    rm -f "$_DEST"
  done

  # Last resort: generate a short synthetic video with ffmpeg.
  if ! $_DOWNLOADED; then
    warn "CDN unavailable — generating synthetic outdoor video with ffmpeg (10 s, 1280×720)..."
    if command -v ffmpeg >/dev/null 2>&1; then
      ffmpeg -y -loglevel error \
        -f lavfi -i "color=c=0x4a7c3f:size=1280x720:rate=25,format=yuv420p" \
        -f lavfi -i "color=c=0x888888:size=1280x180:rate=25,format=yuv420p" \
        -filter_complex "[0][1]overlay=x=0:y=270" \
        -t 10 -c:v libx264 -preset fast \
        "$_DEST" 2>/dev/null \
        && { _FOUND_VIDEO="$_DEST"; } \
        || {
          ffmpeg -y -loglevel error \
            -f lavfi -i "testsrc=size=1280x720:rate=25,format=yuv420p" \
            -t 10 -c:v libx264 -preset fast \
            "$_DEST" 2>/dev/null \
            && { _FOUND_VIDEO="$_DEST"; } \
            || warn "ffmpeg generation failed — add a .mp4 to $_DATA_DIR/videos/ and re-run."
        }
    else
      warn "ffmpeg not found — cannot generate a synthetic video."
      warn "Add any .mp4 or .mov to $_DATA_DIR/videos/ and re-run."
    fi
  fi
fi

# -- Trim any video longer than 10 s down to exactly 10 s -----------------------
# Applies whether the video was just downloaded or was already present from a
# previous run. -ss 5 skips the first 5 s (often static hover/dark frames),
# then -t 10 takes the next 10 s. -c copy makes the trim instant (no re-encode).
if [[ -n "$_FOUND_VIDEO" ]] && command -v ffprobe >/dev/null 2>&1; then
  _DUR="$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$_FOUND_VIDEO" 2>/dev/null || echo 0)"
  _DUR_INT="${_DUR%.*}"
  if [[ "$_DUR_INT" -gt 10 ]]; then
    log "Trimming $(basename "$_FOUND_VIDEO") (${_DUR_INT} s) to 10 s ..."
    _TRIM_TMP="${_FOUND_VIDEO%.mp4}_trimtmp.mp4"
    mv "$_FOUND_VIDEO" "$_TRIM_TMP"
    if ffmpeg -y -loglevel error -ss 5 -i "$_TRIM_TMP" -t 10 -c copy "$_FOUND_VIDEO" 2>/dev/null \
        && [[ -s "$_FOUND_VIDEO" ]]; then
      rm -f "$_TRIM_TMP"
      log "Trimmed: $_FOUND_VIDEO ($(du -sh "$_FOUND_VIDEO" | cut -f1))"
    else
      mv "$_TRIM_TMP" "$_FOUND_VIDEO"
      warn "Trim failed — keeping full-length video."
    fi
  else
    log "Video is already ≤10 s — no trim needed."
  fi
  log "Test video: $_FOUND_VIDEO ($(du -sh "$_FOUND_VIDEO" | cut -f1))"
fi

# -- 4c: Generate sensor sidecars keyed to the video's basename -----------------
# prepare_sensor_data.sh auto-detects the video in .data/videos/ and names
# all generated sidecar files after its basename (e.g. drone_mission.imu.jsonl).
# Steps requiring manual download print instructions and create empty placeholder dirs.
log "Generating sensor sample data (Steps 9–19)..."
bash scripts/ssv/ssv-prepare-sensor-data.sh "$_DATA_DIR/sensors"
log "Sensor data ready in $_DATA_DIR/sensors/"

# -- 4d: Drone audio dataset (Step 32 - DroneAudioCNN training) -----------------
# Downloads geronimobasso/drone-audio-detection-samples from HuggingFace and
# splits it into .data/drone-audio-data/{train,val,test}/{drone,no_drone}/.
# Step 32 in the local pipeline runner trains DroneAudioCNN from this cache.
# Re-running is safe — already-split files are skipped.
#
# Requires: pip install datasets soundfile  (installed in Step 1a above)
# Dataset:  https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples
section "Step 4d — Drone audio dataset (Step 32 training cache)"

log "Preparing drone audio dataset → $_DATA_DIR/drone-audio-data/ ..."
"$PYTHON" -m selfsuvis.scripts.prepare_audio_data \
  --data-dir "$_DATA_DIR/drone-audio-data" \
  || warn "Drone audio dataset prep failed — Step 32 will download on first run instead."
log "Drone audio dataset ready."

# =============================================================================
# STEP 5: DOCKER SERVICES (Qdrant + PostgreSQL)
#
# By default this step only prints the commands to start services.
# Pass --with-docker to actually start the stack from this script.
#
# After containers start:
#   - Qdrant is available at http://localhost:6333
#   - PostgreSQL is available at localhost:5432 (user/pass from .env or defaults)
#   - migrate_postgres.py creates all required tables on first run
#
# If 'make up' fails:
#   make docker-check          # verify Docker daemon is reachable
#   sudo usermod -aG docker $USER  # fix permission denied
#   make fix-data              # fix root-owned data/ directory
# =============================================================================
section "Step 5 — Docker services (Qdrant + PostgreSQL)"

if $WITH_DOCKER; then
  if ! command -v docker >/dev/null 2>&1; then
    warn "Docker not found — skipping stack."
    warn "Install Docker: https://docs.docker.com/engine/install/"
  else
    log "Starting Docker stack (Qdrant + PostgreSQL)..."
    make docker-check 2>/dev/null || true

    _STACK_OK=false
    if make up 2>/dev/null; then
      _STACK_OK=true
    else
      warn "make up failed — Docker stack did not start."
      warn "Fix:  make docker-check"
      warn "      sudo usermod -aG docker \$USER   # if permission denied"
      warn "      make fix-data                    # if .data/ is root-owned"
    fi

    if $_STACK_OK; then
      _PG_HOST="localhost"
      _PG_PORT="5432"
      _PG_USER="selfsuvis"

      log "Waiting for PostgreSQL to be ready at ${_PG_HOST}:${_PG_PORT} ..."
      _PG_READY=false
      for _i in $(seq 1 15); do
        if docker compose -f docker/core/docker-compose.yml exec -T postgres \
              pg_isready -U "$_PG_USER" -q 2>/dev/null; then
          _PG_READY=true; break
        elif command -v pg_isready >/dev/null 2>&1 && \
             pg_isready -h "$_PG_HOST" -p "$_PG_PORT" -U "$_PG_USER" -q 2>/dev/null; then
          _PG_READY=true; break
        elif nc -z "$_PG_HOST" "$_PG_PORT" 2>/dev/null; then
          _PG_READY=true; break
        fi
        sleep 2
      done

      if ! $_PG_READY; then
        warn "PostgreSQL did not become ready within 30 s."
        warn "Retry: DATABASE_URL=postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis python -m selfsuvis.scripts.migrate_postgres"
      else
        log "PostgreSQL is ready — running database migration..."
        DATABASE_URL="postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis" \
          "$PYTHON" -m selfsuvis.scripts.migrate_postgres \
          && log "Migration complete." \
          || warn "Migration failed — retry manually (see above)."
      fi
    fi
  fi
else
  log "Docker stack not started (setup-only mode)."
  log "Start services manually when ready:"
  echo ""
  echo "    make up"
  echo "    python -m selfsuvis.scripts.migrate_postgres"
  echo ""
  log "Or re-run this script with --with-docker to do it automatically."
fi

# =============================================================================
# STEP 6: CONFIRM TEST VIDEO PATH FOR SUMMARY
#
# The video was downloaded / located in Step 4.  This step just resolves the
# final path so the summary section can print an accurate run command.
#
# To use your own footage instead:
#   cp /path/to/drone_mission.mp4 .data/videos/
#   ./scripts/ssv/ssv-setup.sh --sensor-data-only   # regenerate sidecars
#
# Free outdoor footage (no login):
#   Mixkit:  https://mixkit.co/free-stock-video/nature/
#   NASA:    https://images.nasa.gov/
# =============================================================================
section "Step 6 — Confirm test video"

TEST_VIDEO="$(ls "$_DATA_DIR/videos"/*.mp4 "$_DATA_DIR/videos"/*.mov \
               "$_DATA_DIR/videos"/*.avi "$_DATA_DIR/videos"/*.mkv \
               2>/dev/null | head -1 || true)"

if [[ -n "$TEST_VIDEO" ]]; then
  log "Test video: $TEST_VIDEO"
else
  warn "No video found in $_DATA_DIR/videos/ — run commands below will need --input updated."
  TEST_VIDEO="$_DATA_DIR/videos/drone_mission.mp4"
fi

# =============================================================================
# SUMMARY — PRINT RUN COMMANDS
#
# Based on what was set up, print the exact command(s) to run the pipeline.
# =============================================================================
section "Setup complete"

echo ""
echo "------------------------------------------------------"
echo ""
echo -e "${BOLD}Quick start:${RESET}"
echo ""
echo "   APP_ENV=dev .venv/bin/ssv --mode local \\"
echo "     --videos-dir $_DATA_DIR/videos \\"
echo "     --no-qdrant --no-sfm --no-gsplat"
echo ""
echo "Full run variants, flags reference, and sidecar naming:"
echo "   docs/quickstart.md — Step 6"
echo ""
if $WITH_UTILYZE; then
  echo -e "${BOLD}GPU profiling:${RESET}"
  echo "   make utlz"
  echo "   make utlz-endpoints"
  echo ""
fi
echo "------------------------------------------------------"
