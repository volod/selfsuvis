#!/usr/bin/env bash
set -euo pipefail

QDRANT_URL=${QDRANT_URL:-http://localhost:6333}
COLLECTION=${COLLECTION:-video_semantic}

read -p "Delete Qdrant collection '${COLLECTION}' at ${QDRANT_URL}? [y/N] " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Canceled"
  exit 0
fi

curl -s -X DELETE "${QDRANT_URL}/collections/${COLLECTION}" | python -m json.tool
