#!/usr/bin/env bash
# =============================================================================
# setup_local_full.sh — one-shot bootstrap for a complete local run
#                        of all 35 selfsuvis pipeline steps.
#
# WHAT THIS SCRIPT DOES (in order):
#
#   Step 0 — Check system prerequisites (ffmpeg, curl, git, uv)
#   Step 1 — Create Python virtual environment (.venv) and install all deps
#   Step 1a— Install sensor-specific Python packages (open3d, filterpy, …)
#   Step 2 — Download HuggingFace model weights for each pipeline step
#   Step 3 — Install Ollama and pull LLM/VLM sidecar models
#   Step 4 — Create data_test/ layout, download test video, generate sensor sidecars
#   Step 5 — Start Docker stack (Qdrant + PostgreSQL) and run DB migration
#   Step 6 — Confirm test video path for run-command summary
#   Summary— Print the exact run command(s) for your configuration
#
# USAGE:
#   bash scripts/setup_local_full.sh [flags]
#
# FLAGS:
#   (none)                Full setup — all steps
#   --no-docker           Skip Docker services; add --no-qdrant to run cmd
#   --no-ollama           Skip Ollama install and model pulls (use HF weights)
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
#   # Standard first-time setup on a GPU machine:
#   bash scripts/setup_local_full.sh
#
#   # CPU-only machine (no Docker, no Ollama GPU inference):
#   bash scripts/setup_local_full.sh --no-docker --no-ollama
#
#   # Re-download sensor sample data only (already have models):
#   bash scripts/setup_local_full.sh --sensor-data-only
#
#   # With HuggingFace token for gated Gemma weights:
#   HF_TOKEN=hf_xxxx bash scripts/setup_local_full.sh
#
# AFTER SETUP:
#   The script prints the exact run command at the end.
#   Minimal run (Steps 1–9):
#     .venv/bin/python main.py --mode local --input <video.mp4> --no-qdrant
#   Full run (all 35 steps, Ollama + sensors):
#     SENSOR_FUSION_ENABLED=true RF_ENABLED=true ... .venv/bin/python main.py ...
#
# TROUBLESHOOTING:
#   "Permission denied" on Docker:
#     sudo usermod -aG docker $USER   (then log out and back in)
#   "uv: command not found":
#     pip install uv
#   Ollama pull fails (timeout):
#     ollama pull <model>   (run manually, then re-run this script)
#   PostgreSQL migration fails on first try:
#     python scripts/migrate_postgres.py   (postgres may need a few more seconds)
#   HuggingFace 401 on Gemma:
#     Visit https://huggingface.co/google/gemma-3-4b-it and accept the licence
#     then set HF_TOKEN and re-run
# =============================================================================

set -euo pipefail

# ── Flags ─────────────────────────────────────────────────────────────────────
NO_DOCKER=false
NO_OLLAMA=false
SENSOR_DATA_ONLY=false

for arg in "$@"; do
  case "$arg" in
    --no-docker)        NO_DOCKER=true ;;
    --no-ollama)        NO_OLLAMA=true ;;
    --sensor-data-only) SENSOR_DATA_ONLY=true ;;
    *)
      echo "Unknown flag: $arg"
      echo "Valid flags: --no-docker  --no-ollama  --sensor-data-only"
      exit 1
      ;;
  esac
done

# ── Terminal colours ──────────────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

log()     { echo -e "${GREEN}[setup]${RESET} $*"; }
section() {
  echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════════${RESET}"
  echo -e "${BOLD}${CYAN}  $*${RESET}"
  echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════${RESET}"
}
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }

# Always run from the repo root regardless of where the script is invoked from
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Load .env if present ──────────────────────────────────────────────────────
# Copy .env.sample to .env and set HF_TOKEN there — the script picks it up here.
if [[ -f .env ]]; then
  set -a   # export every variable defined from this point
  # shellcheck source=/dev/null
  source .env
  set +a
fi

# ── Configuration ─────────────────────────────────────────────────────────────
# These can be overridden by environment variables before running the script.
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
VLLM_PORT_QWEN="${VLLM_PORT_QWEN:-8010}"
VLLM_PORT_UNIDRIVE="${VLLM_PORT_UNIDRIVE:-8030}"
HF_TOKEN="${HF_TOKEN:-}"

# Shortcuts — resolved after .venv is created
PYTHON=".venv/bin/python"
PIP=".venv/bin/pip"

# =============================================================================
# SENSOR-DATA-ONLY SHORT-CIRCUIT
# When --sensor-data-only is passed, skip all model and environment steps.
# Useful when you already have .venv and model weights and only want fresh
# sensor test sidecars (e.g. after adding a new sensor modality).
# =============================================================================
if $SENSOR_DATA_ONLY; then
  section "Sensor data only mode — skipping model and environment setup"
  mkdir -p data_test/videos data_test/sensors data_test/frames \
           data_test/tiles data_test/maps data_test/reports cache_test
  bash scripts/prepare_sensor_data.sh data_test/sensors
  echo ""
  log "Done. Sensor sample data is in data_test/sensors/"
  _VB="$(ls data_test/videos/*.mp4 data_test/videos/*.mov 2>/dev/null | head -1 || true)"
  _VB="${_VB:+$(basename "${_VB%.*}")}"
  _VB="${_VB:-<video-basename>}"
  log "Sidecars are named after your video: data_test/sensors/step16_imu/${_VB}.imu.jsonl"
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

# =============================================================================
# STEP 1: PYTHON VIRTUAL ENVIRONMENT
#
# Creates .venv in the repo root and installs all production + dev dependencies
# from requirements/requirements_dev.txt.  The install script auto-detects your
# CUDA version and installs the matching PyTorch wheels.
#
# If .venv already exists this step is skipped to avoid reinstalling everything
# on subsequent runs.  To force a clean reinstall:
#   rm -rf .venv && bash scripts/setup_local_full.sh
#
# Expected duration: 3–10 minutes on first run (downloads ~2–4 GB of wheels).
# =============================================================================
section "Step 1 — Python virtual environment (.venv)"

if [[ ! -d .venv ]]; then
  log "Creating .venv..."
  uv venv .venv

  # ensure_venv_pip.sh adds pip into the uv-created venv (uv omits it by default)
  bash scripts/ensure_venv_pip.sh .venv

  # install_requirements.sh installs deps and selects the correct PyTorch wheel
  # for your CUDA version (falls back to CPU-only torch if no GPU found)
  log "Installing Python dependencies (may take 5–10 minutes)..."
  bash scripts/install_requirements.sh requirements/requirements_dev.txt .venv
  log ".venv ready."
else
  log ".venv already exists — skipping. (rm -rf .venv to reinstall)"
fi

# =============================================================================
# STEP 1a: SENSOR-SPECIFIC PYTHON PACKAGES
#
# These packages are not in requirements_dev.txt because they are optional —
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
"$PIP" install --quiet \
  filterpy \
  open3d \
  pyroomacoustics \
  librosa \
  torchaudio \
  sounddevice \
  scipy \
  rasterio \
  pyproj \
  metpy \
  smbus2 \
  pyserial \
  scikit-gstat \
  tonic \
  || warn "Some sensor packages failed — check output above. Non-fatal: sensor steps degrade gracefully."

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
#   2. If either failed during the bulk requirements install, they need a retry
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
if ! "$PYTHON" -c "from sam2 import _C" 2>/dev/null; then
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
"$PYTHON" scripts/prepare_models.py

# Step 4: Florence-2-large for per-keyframe scene captioning.
# Loads locally into the same GPU process as DINOv3 / CLIP.
# If Ollama is running and VRAM is tight, the pipeline evicts Ollama before loading.
log "Downloading Florence-2 (Step 4 — scene captioning)..."
"$PYTHON" scripts/prepare_models.py --florence

# Step 5: Whisper large-v3-turbo for audio transcription.
# Aligned subtitle text is injected into Qwen captioning prompts (Step 24).
log "Downloading Whisper ASR (Step 5 — speech-to-text)..."
"$PYTHON" scripts/prepare_models.py --whisper

# Step 6: OCR model auto-selected by available VRAM.
# On 8 GB VRAM → Phi-3.5-mini-instruct; on 16+ GB → DeepSeek-VL.
log "Downloading OCR model (Step 6 — text extraction from frames)..."
"$PYTHON" scripts/prepare_models.py --ocr

# Step 7: Apple DepthPro for monocular per-pixel depth estimation.
# Depth percentiles are stored in frame_facts_json["depth"] and used by
# sensor fusion (Step 20) to cross-validate LiDAR range measurements.
log "Downloading Depth model (Step 7 — monocular depth estimation)..."
"$PYTHON" scripts/prepare_models.py --depth

# Step 8: HuggingFace RT-DETR or Grounding DINO for open-vocabulary detection.
# This is a second detection pass separate from YOLO (Step 21).
log "Downloading RT-DETR / Grounding DINO (Step 8 — HF object detection)..."
"$PYTHON" scripts/prepare_models.py --detection

# Step 21: YOLO11l — priority-aware detection (human > vehicle > artificial > other).
# Weights are tiny (~48 MB) but needed before SAM mask refinement can run.
log "Downloading YOLO11l (Step 21 — object detection)..."
"$PYTHON" scripts/prepare_models.py --yolo

# Step 21: SAM3 / SAM2 — refines each YOLO bounding box into a pixel-level mask.
# Also used by Gemma directed tracking (Step 22) for SAM-prompted segmentation.
log "Downloading SAM3/SAM2 (Step 21 — segmentation masks)..."
"$PYTHON" scripts/prepare_models.py --sam

# Step 23: Cosmos-1.0 world model — encodes entire video clips as temporal embeddings.
# Used for scene-level search and training data selection.
# NOTE: This model is gated on HuggingFace. If HF_TOKEN is not set, this step
# will prompt you to accept the licence and authenticate.
log "Downloading Cosmos world model (Step 23 — video clip embeddings)..."
"$PYTHON" scripts/prepare_models.py --world-model \
  || warn "World model download failed — Step 23 will be skipped. Re-run with HF_TOKEN set."

# Step 25: UniDriveVLA — expert autonomous-driving analysis.
# Produces four blocks: understanding / perception / planning / mixture_of_experts.
# Source: https://github.com/xiaomi-research/unidrivevla
log "Downloading UniDriveVLA (Step 25 — expert driving analysis)..."
"$PYTHON" scripts/prepare_models.py --unidrive \
  || warn "UniDriveVLA download failed — Step 25 will be skipped."

# Step 3 / 22: Gemma 4 open-weight (gated, requires HF_TOKEN).
# This downloads the weights for local embedding mode (GemmaEmbedder).
# If you prefer to use Ollama for generative mode, you can skip this —
# Ollama pulls are handled in Step 3 and do not require HF_TOKEN.
if [[ -n "$HF_TOKEN" ]]; then
  log "Downloading Gemma 4 open-weight (Steps 3, 22) — requires HF_TOKEN (~8 GiB)..."
  HF_TOKEN="$HF_TOKEN" "$PYTHON" scripts/prepare_models.py --gemma
else
  warn "HF_TOKEN not set — skipping direct Gemma weight download."
  warn "Gemma will run via Ollama (Step 3 below).  To enable HF weights:"
  warn "  export HF_TOKEN=hf_xxxx  &&  bash scripts/setup_local_full.sh"
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

  # ── Gemma 4 4B (efficient multimodal, vision-capable) ─────────────────────
  # gemma4:e4b is the 4-bit efficient variant — best balance of quality and VRAM.
  # Vision-capable: accepts image + text input for frame analysis.
  # Used by: Step 3 (multimodal analysis), Step 22 (directed tracking), Step 35 (audit).
  log "Pulling gemma4:e4b (~5 GiB) — Steps 3, 22, 35..."
  ollama pull gemma4:e4b \
    || warn "gemma4:e4b pull failed. Retry manually: ollama pull gemma4:e4b"

  # ── Qwen2.5-VL 7B (detailed structured captioning) ────────────────────────
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
# Creates the full data_test/ layout expected by the pipeline, downloads a
# public-domain outdoor test video, and generates sensor sidecar files keyed
# to that video's basename.
#
# Directory layout created:
#   data_test/videos/     — input video(s)
#   data_test/sensors/    — per-step sensor sidecars
#   data_test/frames/     — keyframe output (written by pipeline)
#   data_test/tiles/      — tile output (written by pipeline)
#   data_test/maps/       — 3DGS / splat output (written by pipeline)
#   data_test/reports/    — HTML mission summaries (written by pipeline)
#   cache_test/           — integration-test cache volume
#
# Test video (auto-downloaded if data_test/videos/ is empty):
#   Primary:  US Highway 60 drone flyover — real vehicles on a divided highway,
#             desert terrain, trees, road markings (~27 MB, archive.org, public domain)
#   Fallback: Archer Lodge suburban aerial — roads, buildings, trees, carpark (~15 MB)
#   Last:     15-second synthetic video generated by ffmpeg (network-free fallback)
#
# Sensor sidecars (Steps 9–19) are generated with the video's basename so they
# are immediately usable without renaming:
#   data_test/sensors/step16_imu/<video>.imu.jsonl
#   data_test/sensors/step17_atmospheric/<video>.env.jsonl
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

# ── 4a: Create the full data_test directory layout ───────────────────────────
log "Creating data_test/ layout..."
mkdir -p \
  data_test/videos \
  data_test/sensors \
  data_test/frames \
  data_test/tiles \
  data_test/maps \
  data_test/reports \
  cache_test
log "Directories ready."

# ── 4b: Download a test video if data_test/videos/ is empty ─────────────────
_FOUND_VIDEO="$(ls data_test/videos/*.mp4 data_test/videos/*.mov \
                   data_test/videos/*.avi data_test/videos/*.mkv \
                   2>/dev/null | head -1 || true)"

if [[ -n "$_FOUND_VIDEO" ]]; then
  log "Test video found: $_FOUND_VIDEO"
else
  _DEST="data_test/videos/drone_mission.mp4"
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
            || warn "ffmpeg generation failed — add a .mp4 to data_test/videos/ and re-run."
        }
    else
      warn "ffmpeg not found — cannot generate a synthetic video."
      warn "Add any .mp4 or .mov to data_test/videos/ and re-run."
    fi
  fi
fi

# ── Trim any video longer than 10 s down to exactly 10 s ─────────────────────
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

# ── 4c: Generate sensor sidecars keyed to the video's basename ───────────────
# prepare_sensor_data.sh auto-detects the video in data_test/videos/ and names
# all generated sidecar files after its basename (e.g. drone_mission.imu.jsonl).
# Steps requiring manual download print instructions and create empty placeholder dirs.
log "Generating sensor sample data (Steps 9–19)..."
bash scripts/prepare_sensor_data.sh data_test/sensors
log "Sensor data ready in data_test/sensors/"

# =============================================================================
# STEP 5: DOCKER SERVICES (Qdrant + PostgreSQL)
#
# The production pipeline writes vectors to Qdrant and metadata to PostgreSQL.
# For pure local runs you can skip this with --no-docker and pass --no-qdrant
# to main.py (which then falls back to in-memory cosine similarity search).
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

if $NO_DOCKER; then
  warn "--no-docker specified — skipping Docker stack."
  warn "Add --no-qdrant to your run command (in-memory vector search will be used)."
  warn "You will also need a local PostgreSQL instance; set DATABASE_URL accordingly."
else
  if ! command -v docker >/dev/null 2>&1; then
    warn "Docker not found — skipping stack."
    warn "Install Docker: https://docs.docker.com/engine/install/"
    warn "Then re-run without --no-docker."
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
      warn "      make fix-data                    # if data/ is root-owned"
      warn "Re-run setup after fixing Docker, or pass --no-docker to skip."
    fi

    if $_STACK_OK; then
      # Wait for PostgreSQL to accept connections.
      # Uses pg_isready (same tool as the Docker healthcheck) with a 30-second
      # timeout and 2-second retry interval — no blind sleep needed.
      # DATABASE_URL inside containers uses the Docker hostname 'postgres', but
      # from the host the container is reachable at localhost:5432 (port-mapped).
      _PG_HOST="localhost"
      _PG_PORT="5432"
      _PG_USER="selfsuvis"

      log "Waiting for PostgreSQL to be ready at ${_PG_HOST}:${_PG_PORT} ..."
      _PG_READY=false
      for _i in $(seq 1 15); do
        # Prefer pg_isready from the postgres container (always available there).
        # Fall back to nc or a TCP socket check if the client tools aren't installed.
        if docker compose -f docker/docker-compose.yml exec -T postgres \
              pg_isready -U "$_PG_USER" -q 2>/dev/null; then
          _PG_READY=true
          break
        elif command -v pg_isready >/dev/null 2>&1 && \
             pg_isready -h "$_PG_HOST" -p "$_PG_PORT" -U "$_PG_USER" -q 2>/dev/null; then
          _PG_READY=true
          break
        elif nc -z "$_PG_HOST" "$_PG_PORT" 2>/dev/null; then
          _PG_READY=true
          break
        fi
        sleep 2
      done

      if ! $_PG_READY; then
        warn "PostgreSQL did not become ready within 30 s."
        warn "Retry migration manually: DATABASE_URL=postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis python scripts/migrate_postgres.py"
      else
        log "PostgreSQL is ready — running database migration..."
        # Force localhost so the migration works from the host regardless of
        # what DATABASE_URL is set to in .env (which may use 'postgres:5432',
        # the Docker-internal hostname only resolvable inside the container network).
        DATABASE_URL="postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis" \
          "$PYTHON" scripts/migrate_postgres.py \
          && log "Migration complete." \
          || warn "Migration failed — retry: DATABASE_URL=postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis python scripts/migrate_postgres.py"
      fi
    fi
  fi
fi

# =============================================================================
# STEP 6: CONFIRM TEST VIDEO PATH FOR SUMMARY
#
# The video was downloaded / located in Step 4.  This step just resolves the
# final path so the summary section can print an accurate run command.
#
# To use your own footage instead:
#   cp /path/to/drone_mission.mp4 data_test/videos/
#   bash scripts/setup_local_full.sh --sensor-data-only   # regenerate sidecars
#
# Free outdoor footage (no login):
#   Mixkit:  https://mixkit.co/free-stock-video/nature/
#   NASA:    https://images.nasa.gov/
# =============================================================================
section "Step 6 — Confirm test video"

TEST_VIDEO="$(ls data_test/videos/*.mp4 data_test/videos/*.mov \
               data_test/videos/*.avi data_test/videos/*.mkv \
               2>/dev/null | head -1 || true)"

if [[ -n "$TEST_VIDEO" ]]; then
  log "Test video: $TEST_VIDEO"
else
  warn "No video found in data_test/videos/ — run commands below will need --input updated."
  TEST_VIDEO="data_test/videos/drone_mission.mp4"
fi

# =============================================================================
# SUMMARY — PRINT RUN COMMANDS
#
# Based on what was set up, print the exact command(s) to run the pipeline.
# =============================================================================
section "Setup complete — run commands"

# Detect whether Ollama is reachable and build the sidecar flag string
OLLAMA_FLAG=""
if ! $NO_OLLAMA && curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  OLLAMA_FLAG="--gemma-api-url ${OLLAMA_HOST}/v1 --qwen-api-url ${OLLAMA_HOST}/v1"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${BOLD}1. Minimal run (Steps 1–9, no sidecar servers needed):${RESET}"
echo "   .venv/bin/python main.py --mode local \\"
echo "     --input $TEST_VIDEO \\"
echo "     --no-qdrant"
echo ""

echo -e "${BOLD}2. Full run — Ollama sidecars + all sensor steps${RESET} (sensor steps on by default):"
echo "   .venv/bin/python main.py --mode local \\"
echo "     --input $TEST_VIDEO \\"
if [[ -n "$OLLAMA_FLAG" ]]; then
  echo "     $OLLAMA_FLAG \\"
else
  echo "     --gemma-api-url http://localhost:11434/v1 \\"
  echo "     --qwen-api-url  http://localhost:11434/v1 \\"
fi
echo "     --rfdetr-model base"
echo ""

echo -e "${BOLD}3. Full run — vLLM sidecars (Qwen2.5-VL + UniDriveVLA):${RESET}"
echo "   # Terminal 1 — Qwen2.5-VL (Step 24, port ${VLLM_PORT_QWEN}):"
echo "   python -m vllm.entrypoints.openai.api_server \\"
echo "     --model Qwen/Qwen2.5-VL-7B-Instruct \\"
echo "     --port ${VLLM_PORT_QWEN} --max-model-len 8192"
echo ""
echo "   # Terminal 2 — UniDriveVLA (Step 25, port ${VLLM_PORT_UNIDRIVE}):"
echo "   python -m vllm.entrypoints.openai.api_server \\"
echo "     --model owl10/UniDriveVLA_Nusc_Base_Stage3 \\"
echo "     --port ${VLLM_PORT_UNIDRIVE} --max-model-len 4096"
echo ""
echo "   # Terminal 3 — pipeline:"
echo "   .venv/bin/python main.py --mode local \\"
echo "     --input $TEST_VIDEO \\"
echo "     --gemma-api-url    ${OLLAMA_HOST}/v1 \\"
echo "     --qwen-api-url     http://localhost:${VLLM_PORT_QWEN}/v1 \\"
echo "     --unidrive-api-url http://localhost:${VLLM_PORT_UNIDRIVE}/v1"
echo ""

echo -e "${BOLD}Sensor sidecar naming convention${RESET} (place next to video):"
echo "   ${TEST_VIDEO%.mp4}.iq              # Step  9 — RF/SDR IQ (float32)"
echo "   ${TEST_VIDEO%.mp4}.thermal.mp4     # Step 10 — FLIR LWIR video"
echo "   ${TEST_VIDEO%.mp4}.multispectral/  # Step 11 — per-band GeoTIFF dir"
echo "   ${TEST_VIDEO%.mp4}.events.raw      # Step 12 — Prophesee event stream"
echo "   ${TEST_VIDEO%.mp4}.lidar.pcd       # Step 13 — LiDAR point cloud"
echo "   ${TEST_VIDEO%.mp4}.radar.bin       # Step 14 — radar ADC IQ"
echo "   ${TEST_VIDEO%.mp4}.adsb.jsonl      # Step 15 — ADS-B aircraft log"
echo "   ${TEST_VIDEO%.mp4}.imu.jsonl       # Step 16 — IMU (200 Hz)"
echo "   ${TEST_VIDEO%.mp4}.baro.jsonl      # Step 16 — barometer (5 Hz)"
echo "   ${TEST_VIDEO%.mp4}.wind.jsonl      # Step 16 — anemometer (1 Hz)"
echo "   ${TEST_VIDEO%.mp4}.env.jsonl       # Step 17 — atmospheric"
echo "   ${TEST_VIDEO%.mp4}.gas.jsonl       # Step 18 — gas/radiation"
echo "   ${TEST_VIDEO%.mp4}.audio.wav       # Step 19 — acoustic (48 kHz)"
echo ""
echo "   Generated sample sidecars are in data_test/sensors/"
echo "   Copy them to data_test/videos/ and rename to match your video basename."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Reference: docs/learning_path.md — Local Full Run Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
