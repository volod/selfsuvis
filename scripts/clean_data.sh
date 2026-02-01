#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-./data}

read -p "Remove frames/tiles caches under ${ROOT}? [y/N] " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Canceled"
  exit 0
fi

rm -rf "${ROOT}/frames" "${ROOT}/tiles"
mkdir -p "${ROOT}/frames" "${ROOT}/tiles"

echo "Cleaned frames and tiles under ${ROOT}" 
