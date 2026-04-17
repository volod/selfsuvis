#!/usr/bin/env bash
# Install system libraries and tools required to run the video semantic search
# solution locally on Linux (ffmpeg, OpenCV runtime deps, optional Python).
# Usage: sudo ./scripts/install_system_deps.sh [--with-python]
set -euo pipefail

WITH_PYTHON=false
for arg in "$@"; do
  case "$arg" in
    --with-python) WITH_PYTHON=true ;;
    -h|--help)
      echo "Usage: $0 [--with-python]"
      echo ""
      echo "Installs system packages needed to run the project locally (no Docker):"
      echo "  - ffmpeg (video decoding)"
      echo "  - OpenCV runtime libs (libgl1, libglib2.0-0, libsm, libxext, libxrender)"
      echo ""
      echo "Options:"
      echo "  --with-python  Also install Python 3 and python3-venv / python3-pip."
      echo ""
      echo "After running this script, create a venv and install Python deps:"
      echo "  make venv"
      exit 0
      ;;
  esac
done

echo "=============================================="
echo "  System dependencies for Video Semantic Search"
echo "=============================================="
echo "This script installs ffmpeg and OpenCV-related libraries so you can run"
echo "the app locally (without Docker). Run with sudo."
echo ""

# Detect distro
if [[ -f /etc/os-release ]]; then
  # shellcheck source=/dev/null
  source /etc/os-release
  echo "Detected OS: $ID ${VERSION_ID:-}"
else
  echo "Cannot detect OS: /etc/os-release not found." >&2
  exit 1
fi
echo ""

install_debian() {
  echo "Installing packages with apt (Debian/Ubuntu)..."
  apt-get update
  echo "  Installing ffmpeg and OpenCV runtime libraries..."
  apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1
  if [[ "$WITH_PYTHON" == true ]]; then
    echo "  Installing Python 3, venv, and pip..."
    apt-get install -y --no-install-recommends \
      python3 \
      python3-venv \
      python3-pip
  fi
  echo "  Done."
}

install_fedora() {
  echo "Installing packages with dnf (Fedora/RHEL)..."
  echo "  Installing ffmpeg and OpenCV runtime libraries..."
  dnf install -y \
    ffmpeg \
    mesa-libGL \
    glib2 \
    libSM \
    libXext \
    libXrender
  if [[ "$WITH_PYTHON" == true ]]; then
    echo "  Installing Python 3 and pip..."
    dnf install -y \
      python3 \
      python3-pip
  fi
  echo "  Done."
}

install_arch() {
  echo "Installing packages with pacman (Arch)..."
  echo "  Installing ffmpeg and OpenCV runtime libraries..."
  pacman -Sy --noconfirm --needed \
    ffmpeg \
    mesa \
    glib2 \
    libsm \
    libxext \
    libxrender
  if [[ "$WITH_PYTHON" == true ]]; then
    echo "  Installing Python and pip..."
    pacman -Sy --noconfirm --needed \
      python \
      python-pip
  fi
  echo "  Done."
}

case "${ID:-}" in
  debian|ubuntu)
    install_debian
    ;;
  fedora|rhel|centos|rocky|almalinux)
    install_fedora
    ;;
  arch|artix)
    install_arch
    ;;
  *)
    case "${ID_LIKE:-}" in
      *debian*|*ubuntu*)
        install_debian
        ;;
      *fedora*|*rhel*)
        install_fedora
        ;;
      *)
        echo "Unsupported OS: ID=$ID ID_LIKE=$ID_LIKE" >&2
        exit 1
        ;;
    esac
    ;;
esac

echo ""
echo "=============================================="
echo "  System dependencies installed"
echo "=============================================="
if [[ "$WITH_PYTHON" == true ]]; then
  echo "Python was installed. Next steps:"
  echo "  make venv"
else
  echo "Next steps: ensure Python 3.10+ is installed, then create a venv and install deps:"
  echo "  make venv"
fi
echo ""
echo "To run the full stack with Docker instead: make up"
