#!/usr/bin/env bash
set -euo pipefail

REQ_FILE=${1:-requirements/requirements_dev.txt}
VENV_PATH=${2:-.venv}

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Virtual environment not found at $VENV_PATH" >&2
  exit 1
fi

# Ensure pip is available in the venv (uv-created venvs do not include it by default)
uv pip install --python "$VENV_PATH" pip

detect_cuda_version() {
  local cuda_version=""

  # Parse regular nvidia-smi text output (--query-gpu=cuda_version is not a valid field)
  if command -v nvidia-smi >/dev/null 2>&1; then
    cuda_version=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)
  fi

  # Fallback: nvcc reports the installed toolkit version
  if [[ -z "$cuda_version" ]] && command -v nvcc >/dev/null 2>&1; then
    cuda_version=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)
  fi

  # Fallback: /usr/local/cuda version files
  if [[ -z "$cuda_version" ]] && [[ -d /usr/local/cuda ]]; then
    if [[ -f /usr/local/cuda/version.json ]]; then
      cuda_version=$(python3 -c "import json,sys; d=json.load(open('/usr/local/cuda/version.json')); print(d.get('cuda',{}).get('version','').rsplit('.',1)[0])" 2>/dev/null)
    fi
    if [[ -z "$cuda_version" ]] && [[ -f /usr/local/cuda/version.txt ]]; then
      cuda_version=$(sed -n 's/.*CUDA Version \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' /usr/local/cuda/version.txt | head -n1)
    fi
  fi

  echo "$cuda_version"
}

map_cuda_to_torch_index() {
  local cuda_version="$1"
  case "$cuda_version" in
    13.*) echo "cu126" ;;
    12.9*) echo "cu126" ;;
    12.8*) echo "cu126" ;;
    12.7*) echo "cu126" ;;
    12.6*) echo "cu126" ;;
    12.5*) echo "cu126" ;;
    12.4*) echo "cu124" ;;
    12.3*) echo "cu121" ;;
    12.2*) echo "cu121" ;;
    12.1*) echo "cu121" ;;
    12.0*) echo "cu121" ;;
    11.8*) echo "cu118" ;;
    11.7*) echo "cu118" ;;
    11.6*) echo "cu118" ;;
    *) echo "" ;;
  esac
}

# Install deps; opencv-python>=4.10 supports numpy 2.x (see requirements_prod.txt)
uv pip install --python "$VENV_PATH" -r "$REQ_FILE"

CUDA_VERSION=$(detect_cuda_version)

# Allow caller to force a specific CUDA index (e.g. FORCE_CUDA=cu126 or FORCE_CUDA=1 for auto-latest)
if [[ -n "${FORCE_CUDA:-}" ]]; then
  if [[ "$FORCE_CUDA" == "1" ]]; then
    TORCH_CUDA_INDEX="cu126"
  else
    TORCH_CUDA_INDEX="$FORCE_CUDA"
  fi
else
  TORCH_CUDA_INDEX=$(map_cuda_to_torch_index "$CUDA_VERSION")
fi

if [[ -n "$TORCH_CUDA_INDEX" ]]; then
  echo "CUDA $CUDA_VERSION detected → installing torch with $TORCH_CUDA_INDEX wheels"
  if ! uv pip install --python "$VENV_PATH" --upgrade --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_INDEX" torch==2.10.0 torchvision==0.25.0; then
    echo "CUDA wheel install failed, falling back to CPU"
    uv pip install --python "$VENV_PATH" --upgrade --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
  fi
else
  echo "No CUDA detected → installing CPU-only torch"
  uv pip install --python "$VENV_PATH" --upgrade --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
fi

# flash-attn must be installed AFTER torch (its setup.py imports torch at build time).
# --no-build-isolation lets it find the torch headers already in the active environment.
# Skipped on CPU-only machines; non-fatal on failure (models fall back to sdpa attention).
PYTHON_BIN="$VENV_PATH/bin/python"
if "$PYTHON_BIN" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "Installing flash-attn (CUDA available — uses prebuilt wheel or compiles from source) …"
  if "$PYTHON_BIN" -m pip install flash-attn --no-build-isolation -q; then
    echo "flash-attn installed."
  else
    echo "WARNING: flash-attn build failed. Run later with:"
    echo "  python scripts/prepare_models.py --flash-attn"
    echo "Models will use sdpa (PyTorch built-in SDPA) until then."
  fi
else
  echo "No CUDA GPU detected — skipping flash-attn (CPU-only mode)."
fi

# ── TensorRT + onnxruntime-gpu ─────────────────────────────────────────────────
# onnxruntime-gpu replaces the CPU-only onnxruntime installed from requirements.
# It adds CUDAExecutionProvider and TensorrtExecutionProvider, giving the fastest
# ONNX inference on NVIDIA GPUs.  TensorRT must also be present at runtime for
# TensorrtExecutionProvider to activate; we install it from NVIDIA's PyPI index.
#
# TensorRT CUDA compatibility:
#   CUDA 12.x → TensorRT 10.x  (tensorrt>=10.0 from pypi.nvidia.com)
#   CUDA 11.8 → TensorRT 8.x   (tensorrt<9; use apt or older wheel)
#
# Non-fatal: falls back to CPU onnxruntime if GPU install fails.
if [[ -n "$TORCH_CUDA_INDEX" ]]; then
  echo ""
  echo "── onnxruntime-gpu + TensorRT ────────────────────────────────────────────────"

  # Uninstall CPU-only onnxruntime first to avoid package conflict.
  "$PYTHON_BIN" -m pip uninstall -y onnxruntime 2>/dev/null || true

  echo "Installing onnxruntime-gpu …"
  if "$PYTHON_BIN" -m pip install "onnxruntime-gpu>=1.18.0" -q; then
    echo "  onnxruntime-gpu installed (CUDAExecutionProvider enabled)."
  else
    echo "  WARNING: onnxruntime-gpu install failed — reinstalling CPU onnxruntime."
    "$PYTHON_BIN" -m pip install "onnxruntime>=1.18.0" -q || true
  fi

  # TensorRT wheels are published by NVIDIA at https://pypi.nvidia.com.
  # Version selection:
  #   CUDA 12.x → tensorrt>=10 (cu12 variant)
  #   CUDA 11.8 → tensorrt<9   (cu11 bindings are not on pypi.nvidia.com; use apt)
  echo "Installing TensorRT Python package (enables TensorrtExecutionProvider) …"
  if [[ "$TORCH_CUDA_INDEX" == cu118 ]]; then
    echo "  CUDA 11.8 detected — TensorRT 10+ is not available for CUDA 11."
    echo "  Install TensorRT 8.x via apt from the NVIDIA TensorRT apt repo, then"
    echo "  the TensorrtExecutionProvider will activate automatically."
    echo "  See: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
  else
    # CUDA 12.x — use NVIDIA's PyPI index (pypi.nvidia.com publishes cu12 wheels).
    if "$PYTHON_BIN" -m pip install \
        "tensorrt>=10.0" \
        "tensorrt-cu12-bindings" \
        "tensorrt-cu12-libs" \
        --extra-index-url https://pypi.nvidia.com \
        -q; then
      echo "  TensorRT installed (TensorrtExecutionProvider will be active)."
    else
      echo "  WARNING: TensorRT pip install failed. Possible fixes:"
      echo "    pip install tensorrt --extra-index-url https://pypi.nvidia.com"
      echo "    sudo apt-get install -y tensorrt  (requires nvidia-tensorrt apt repo)"
      echo "    See: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
      echo "  Without TensorRT, onnxruntime-gpu will use CUDAExecutionProvider only."
    fi
  fi
else
  echo ""
  echo "No CUDA detected — keeping CPU-only onnxruntime (no TensorRT)."
fi
