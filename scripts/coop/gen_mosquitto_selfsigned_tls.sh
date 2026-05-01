#!/usr/bin/env bash
# Generate self-signed TLS certs for Mosquitto.
# Run as the same user who runs docker compose (not sudo) so mosquitto (PUID) can read them.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CERT_DIR="$ROOT_DIR/config/coop/mosquitto/certs"
mkdir -p "$CERT_DIR"

HOST="${1:-}"
if [[ -z "$HOST" ]]; then
  echo "Usage: $0 <hostname>" >&2
  exit 1
fi

CA_KEY="$CERT_DIR/ca.key"
CA_CERT="$CERT_DIR/ca.crt"
SERVER_KEY="$CERT_DIR/server.key"
SERVER_CSR="$CERT_DIR/server.csr"
SERVER_CERT="$CERT_DIR/server.crt"
SERVER_EXT="$CERT_DIR/server.ext"

openssl genrsa -out "$CA_KEY" 4096
openssl req -x509 -new -nodes -key "$CA_KEY" -sha256 -days 3650 \
  -subj "/C=UA/O=Cooperative/CN=Coop Pilot CA" \
  -out "$CA_CERT"

openssl genrsa -out "$SERVER_KEY" 4096
openssl req -new -key "$SERVER_KEY" \
  -subj "/C=UA/O=Cooperative/CN=$HOST" \
  -out "$SERVER_CSR"

cat > "$SERVER_EXT" <<EOF
subjectAltName=DNS:$HOST,IP:127.0.0.1
extendedKeyUsage=serverAuth
keyUsage=digitalSignature,keyEncipherment
EOF

openssl x509 -req -in "$SERVER_CSR" -CA "$CA_CERT" -CAkey "$CA_KEY" -CAcreateserial \
  -out "$SERVER_CERT" -days 730 -sha256 -extfile "$SERVER_EXT"

rm -f "$SERVER_CSR" "$SERVER_EXT" "$CERT_DIR/ca.srl" || true
chmod 600 "$CA_KEY"
chmod 0644 "$SERVER_KEY"
chmod 644 "$CA_CERT" "$SERVER_CERT"
echo "OK: generated certs in $CERT_DIR"
