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
  if command -v nvidia-smi >/dev/null 2>&1; then
    cuda_version=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
  fi

  if [[ -z "$cuda_version" ]] && command -v nvcc >/dev/null 2>&1; then
    cuda_version=$(nvcc --version 2>/dev/null | sed -n 's/.*release \\([0-9]\\+\\.[0-9]\\+\\).*/\\1/p' | head -n1)
  fi

  if [[ -z "$cuda_version" ]] && [[ -d /usr/local/cuda ]]; then
    if [[ -f /usr/local/cuda/version.txt ]]; then
      cuda_version=$(sed -n 's/.*CUDA Version \\([0-9]\\+\\.[0-9]\\+\\).*/\\1/p' /usr/local/cuda/version.txt | head -n1)
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
