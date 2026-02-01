#!/usr/bin/env bash
set -euo pipefail

REQ_FILE=${1:-requirements/requirements_dev.txt}
VENV_PATH=${2:-.venv}

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Virtual environment not found at $VENV_PATH" >&2
  exit 1
fi

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
    12.4*) echo "cu124" ;;
    12.3*) echo "cu121" ;;
    12.2*) echo "cu121" ;;
    12.1*) echo "cu121" ;;
    12.0*) echo "cu121" ;;
    11.8*) echo "cu118" ;;
    11.7*) echo "cu117" ;;
    11.6*) echo "cu116" ;;
    11.5*) echo "cu115" ;;
    11.4*) echo "cu113" ;;
    *) echo "" ;;
  esac
}

uv pip install --python "$VENV_PATH" -r "$REQ_FILE"

CUDA_VERSION=$(detect_cuda_version)
TORCH_CUDA_INDEX=$(map_cuda_to_torch_index "$CUDA_VERSION")

if [[ -n "$TORCH_CUDA_INDEX" ]]; then
  if ! uv pip install --python "$VENV_PATH" --upgrade --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_INDEX" torch==2.2.0 torchvision==0.17.0; then
    uv pip install --python "$VENV_PATH" --upgrade --index-url https://download.pytorch.org/whl/cpu torch==2.2.0 torchvision==0.17.0
  fi
else
  uv pip install --python "$VENV_PATH" --upgrade --index-url https://download.pytorch.org/whl/cpu torch==2.2.0 torchvision==0.17.0
fi
