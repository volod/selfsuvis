#!/usr/bin/env bash
# Generate self-signed TLS certs for Mosquitto.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
CERT_DIR="$PROJECT_ROOT_DIR/config/coop/mosquitto/certs"
mkdir -p "$CERT_DIR"

HOST="${1:-}"
if [[ "$HOST" == "-h" || "$HOST" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/coop-mosquitto-tls.sh <hostname>

Generates a local self-signed CA and server certificate for the coop Mosquitto broker.
Run as the same user who runs docker compose so the broker can read the files.
EOF
  exit 0
fi
if [[ -z "$HOST" ]]; then
  project_die "Usage: $0 <hostname>"
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
project_log "Generated certs in $CERT_DIR"
