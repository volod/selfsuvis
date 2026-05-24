#!/usr/bin/env bash
# Build a self-contained offline installation bundle for the coop-pilot IoT edge stack.
# Run on a machine WITH Docker and internet access (the build machine, not the target).
#
# Usage:
#   ./scripts/coop/coop-release.sh [options]
#
# Options:
#   --version VERSION      Bundle version tag (default: git describe --tags --always)
#   --arch ARCH            Target CPU architecture: amd64 or arm64 (default: amd64)
#   --output-dir DIR       Output directory for the bundle (default: ./dist)
#   --bundle BUNDLE        Service bundle to include (default: standard):
#                            min      -- MQTT hub + LoRaWAN (ChirpStack only, no video)
#                            standard -- min + Frigate NVR video surveillance
#                            video    -- MQTT hub + Frigate NVR (no LoRaWAN/ChirpStack)
#   --with-metrics         Include Prometheus, Grafana, cAdvisor, node-exporter images
#   --no-images            Skip pulling and saving Docker images (faster, configs only)
#   --no-docker-pkgs       Skip downloading offline Docker Engine packages
#   --yes                  Skip confirmation prompt
#
# Examples:
#   ./scripts/coop/coop-release.sh --version 1.2.0 --arch amd64
#   ./scripts/coop/coop-release.sh --version 1.2.0 --arch arm64 --bundle min
#   ./scripts/coop/coop-release.sh --bundle standard --with-metrics --yes
#   ./scripts/coop/coop-release.sh --no-images --no-docker-pkgs  # config-only test bundle

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Pinned versions (update deliberately) ────────────────────────────────────
DOCKER_ENGINE_VERSION="27.5.1"
DOCKER_COMPOSE_VERSION="2.33.1"

# ── Image groups ──────────────────────────────────────────────────────────────
# mosquitto is the MQTT hub; always included in every bundle.
BASE_IMAGES=(
  "eclipse-mosquitto:2"
)

# ChirpStack LoRaWAN Network Server (bundle: min, standard)
LORAWAN_IMAGES=(
  "postgres:14-alpine"
  "redis:7-alpine"
  "chirpstack/chirpstack:4"
  "chirpstack/chirpstack-gateway-bridge:4"
  "chirpstack/chirpstack-rest-api:4"
)

# Frigate NVR + object detection (bundle: standard, video)
VIDEO_IMAGES=(
  "ghcr.io/blakeblackshear/frigate:stable"
)

# Observability stack (added by --with-metrics to any bundle)
METRICS_IMAGES=(
  "prom/prometheus:latest"
  "grafana/grafana:11-alpine"
  "gcr.io/cadvisor/cadvisor:v0.49.1"
  "prom/node-exporter:latest"
)

# ── Scripts to bundle from scripts/ (paths relative to scripts/; structure preserved in bundle) ──
BUNDLE_SCRIPTS=(
  shared/common.sh
  coop/coop-ctl.sh
  coop/coop-bootstrap.sh
  coop/coop-compose.sh
  coop/coop-credentials.sh
  coop/coop-data-dirs.sh
  coop/coop-env.sh
  coop/coop-mosquitto-tls.sh
  coop/coop-mqtt-users.sh
)

log()  { printf '[release] %s\n' "$*"; }
die()  { printf '[release] ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf '[release] WARN:  %s\n' "$*"; }

# ── Defaults ──────────────────────────────────────────────────────────────────
VERSION="${VERSION:-$(git -C "$PROJECT_ROOT" describe --tags --always --dirty 2>/dev/null || echo "0.0.0-dev")}"
ARCH="amd64"
OUTPUT_DIR="$PROJECT_ROOT/dist"
BUNDLE_CONFIG="standard"
WITH_METRICS=false
SKIP_IMAGES=false
SKIP_DOCKER_PKGS=false
YES=false

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)        VERSION="$2";         shift 2 ;;
    --arch)           ARCH="$2";            shift 2 ;;
    --output-dir)     OUTPUT_DIR="$2";      shift 2 ;;
    --bundle)         BUNDLE_CONFIG="$2";   shift 2 ;;
    --with-metrics)   WITH_METRICS=true;    shift   ;;
    --no-images)      SKIP_IMAGES=true;     shift   ;;
    --no-docker-pkgs) SKIP_DOCKER_PKGS=true; shift  ;;
    --yes|-y)         YES=true;             shift   ;;
    -h|--help)        sed -n '2,26p' "$0"; exit 0  ;;
    *) die "Unknown option: $1. Run with --help for usage." ;;
  esac
done

[[ "$BUNDLE_CONFIG" == "min" || "$BUNDLE_CONFIG" == "standard" || "$BUNDLE_CONFIG" == "video" ]] \
  || die "Invalid --bundle: $BUNDLE_CONFIG (use min, standard, or video)"

[[ "$ARCH" == "amd64" || "$ARCH" == "arm64" ]] \
  || die "Unsupported arch: $ARCH (use amd64 or arm64)"

case "$ARCH" in
  amd64) DOCKER_ARCH="x86_64";  COMPOSE_ARCH="linux-x86_64"  ;;
  arm64) DOCKER_ARCH="aarch64"; COMPOSE_ARCH="linux-aarch64" ;;
esac

BUNDLE_NAME="coop-edge-${VERSION}-${ARCH}-${BUNDLE_CONFIG}"
BUNDLE_DIR="$OUTPUT_DIR/$BUNDLE_NAME"
IMAGES_DIR="$BUNDLE_DIR/images"
PKGS_DIR="$BUNDLE_DIR/packages/${ARCH}"

# ── Pre-flight ────────────────────────────────────────────────────────────────
command -v docker >/dev/null 2>&1 || die "Docker is required to build the bundle."
command -v curl   >/dev/null 2>&1 || die "curl is required to download packages."

log "Bundle:       $BUNDLE_NAME"
log "Bundle config:$BUNDLE_CONFIG"
log "Target arch:  $ARCH"
log "Output:       $OUTPUT_DIR"
log "With metrics: $WITH_METRICS"
log "Skip images:  $SKIP_IMAGES"
log "Skip pkgs:    $SKIP_DOCKER_PKGS"
printf '\n'

if [[ "$YES" != true ]]; then
  printf '[release] Proceed? [y/N] '
  read -r _ans
  [[ "${_ans,,}" == "y" ]] || { log "Aborted."; exit 0; }
fi

# ── Directory scaffold ────────────────────────────────────────────────────────
log "Creating bundle directory..."
rm -rf "$BUNDLE_DIR"
mkdir -p "$IMAGES_DIR" "$PKGS_DIR" \
  "$BUNDLE_DIR/scripts" \
  "$BUNDLE_DIR/docker/coop" \
  "$BUNDLE_DIR/config" \
  "$BUNDLE_DIR/env"

# ── Docker images ─────────────────────────────────────────────────────────────
if [[ "$SKIP_IMAGES" == true ]]; then
  warn "Skipping image export (--no-images)."
else
  # Assemble image list from bundle config
  ALL_IMAGES=("${BASE_IMAGES[@]}")
  case "$BUNDLE_CONFIG" in
    min)      ALL_IMAGES+=("${LORAWAN_IMAGES[@]}") ;;
    standard) ALL_IMAGES+=("${LORAWAN_IMAGES[@]}"); ALL_IMAGES+=("${VIDEO_IMAGES[@]}") ;;
    video)    ALL_IMAGES+=("${VIDEO_IMAGES[@]}") ;;
  esac
  [[ "$WITH_METRICS" == true ]] && ALL_IMAGES+=("${METRICS_IMAGES[@]}")

  log "Pulling ${#ALL_IMAGES[@]} images for linux/${ARCH}..."
  for IMAGE in "${ALL_IMAGES[@]}"; do
    log "  pull  $IMAGE"
    docker pull --platform "linux/${ARCH}" "$IMAGE"
  done

  log "Saving images (may take several minutes)..."
  for IMAGE in "${ALL_IMAGES[@]}"; do
    # Sanitise: strip registry prefix, replace / and : with _
    FNAME="$(printf '%s' "$IMAGE" | sed 's|.*/||; s|[:/]|_|g').tar.gz"
    log "  save  $IMAGE -> images/$FNAME"
    docker save "$IMAGE" | gzip -9 > "$IMAGES_DIR/$FNAME"
  done
fi

# ── Docker Engine offline packages ────────────────────────────────────────────
if [[ "$SKIP_DOCKER_PKGS" == true ]]; then
  warn "Skipping Docker package download (--no-docker-pkgs)."
else
  log "Downloading Docker Engine ${DOCKER_ENGINE_VERSION} static binary (${DOCKER_ARCH})..."
  DOCKER_TGZ="docker-${DOCKER_ENGINE_VERSION}.tgz"
  curl -fsSL --retry 3 --progress-bar \
    "https://download.docker.com/linux/static/stable/${DOCKER_ARCH}/${DOCKER_TGZ}" \
    -o "$PKGS_DIR/$DOCKER_TGZ"

  log "Downloading Docker Compose v${DOCKER_COMPOSE_VERSION} (${COMPOSE_ARCH})..."
  curl -fsSL --retry 3 --progress-bar \
    "https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-${COMPOSE_ARCH}" \
    -o "$PKGS_DIR/docker-compose"
  chmod +x "$PKGS_DIR/docker-compose"

  # Embed systemd unit files required for static Docker install
  log "Writing systemd unit templates..."
  cat > "$PKGS_DIR/containerd.service" << 'UNIT'
[Unit]
Description=containerd container runtime
After=network.target

[Service]
ExecStartPre=-/sbin/modprobe overlay
ExecStart=/usr/local/bin/containerd
Type=notify
Delegate=yes
KillMode=process
Restart=always
RestartSec=5
LimitNPROC=infinity
LimitCORE=infinity
LimitNOFILE=1048576
TasksMax=infinity

[Install]
WantedBy=multi-user.target
UNIT

  cat > "$PKGS_DIR/docker.service" << 'UNIT'
[Unit]
Description=Docker Application Container Engine
After=network-online.target containerd.service
Requires=docker.socket containerd.service

[Service]
Type=notify
ExecStart=/usr/local/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock
ExecReload=/bin/kill -s HUP $MAINPID
TimeoutStartSec=0
RestartSec=2
Restart=always
LimitNOFILE=infinity
LimitNPROC=infinity
LimitCORE=infinity
Delegate=yes
KillMode=process
OOMScoreAdjust=-500

[Install]
WantedBy=multi-user.target
UNIT

  cat > "$PKGS_DIR/docker.socket" << 'UNIT'
[Unit]
Description=Docker Socket for the API

[Socket]
ListenStream=/var/run/docker.sock
SocketMode=0660
SocketUser=root
SocketGroup=docker

[Install]
WantedBy=sockets.target
UNIT
fi

# ── Copy project files ────────────────────────────────────────────────────────
log "Copying project files..."

cp "$PROJECT_ROOT/docker/coop/docker-compose.coop.yml" "$BUNDLE_DIR/docker/coop/"
cp -r "$PROJECT_ROOT/config/coop" "$BUNDLE_DIR/config/"
cp "$PROJECT_ROOT/src/selfsuvis/coop/env/prod.env" "$BUNDLE_DIR/env/"
cp "$PROJECT_ROOT/src/selfsuvis/coop/env/dev.env"  "$BUNDLE_DIR/env/"
cp "$PROJECT_ROOT/src/selfsuvis/coop/env/test.env" "$BUNDLE_DIR/env/"

for SCRIPT in "${BUNDLE_SCRIPTS[@]}"; do
  SRC="$PROJECT_ROOT/scripts/$SCRIPT"
  DEST="$BUNDLE_DIR/scripts/$SCRIPT"
  if [[ -f "$SRC" ]]; then
    mkdir -p "$(dirname "$DEST")"
    cp "$SRC" "$DEST"
    chmod +x "$DEST"
  else
    warn "Script not found, skipping: $SCRIPT"
  fi
done

# install.sh at bundle root is a copy of coop-install.sh
cp "$PROJECT_ROOT/scripts/coop/coop-install.sh" "$BUNDLE_DIR/install.sh"
chmod +x "$BUNDLE_DIR/install.sh"

# ── Release manifest ──────────────────────────────────────────────────────────
log "Generating release.manifest..."
{
  printf 'bundle:          %s\n' "$BUNDLE_NAME"
  printf 'version:         %s\n' "$VERSION"
  printf 'arch:            %s\n' "$ARCH"
  printf 'bundle_config:   %s\n' "$BUNDLE_CONFIG"
  printf 'with_metrics:    %s\n' "$WITH_METRICS"
  printf 'built_at:        %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'docker_engine:   %s\n' "$DOCKER_ENGINE_VERSION"
  printf 'docker_compose:  %s\n' "$DOCKER_COMPOSE_VERSION"
  printf '\nimages:\n'
  for F in "$IMAGES_DIR"/*.tar.gz; do
    [[ -f "$F" ]] && printf '  %-40s  %s\n' \
      "$(basename "$F")" "$(sha256sum "$F" | cut -d' ' -f1)"
  done
  printf '\npackages:\n'
  for F in "$PKGS_DIR"/*; do
    [[ -f "$F" ]] && printf '  %-40s  %s\n' \
      "$(basename "$F")" "$(sha256sum "$F" | cut -d' ' -f1)"
  done
} > "$BUNDLE_DIR/release.manifest"

# ── Tarball ───────────────────────────────────────────────────────────────────
TARBALL="$OUTPUT_DIR/${BUNDLE_NAME}.tar.gz"
log "Packing tarball..."
tar -czf "$TARBALL" -C "$OUTPUT_DIR" "$BUNDLE_NAME"

SHA="$(sha256sum "$TARBALL" | cut -d' ' -f1)"
printf '%s  %s\n' "$SHA" "${BUNDLE_NAME}.tar.gz" > "${TARBALL}.sha256"

log ""
log "Bundle ready"
log "  File:   $TARBALL"
log "  SHA256: $SHA"
log ""
log "Deploy to target:"
log "  scp $TARBALL user@target:~/"
log "  ssh user@target"
log "  tar -xzf ${BUNDLE_NAME}.tar.gz"
log "  cd ${BUNDLE_NAME}"
log "  sudo ./install.sh --bundle ${BUNDLE_CONFIG} [--hw-profile min|mid|high] [--storage-dev /dev/sdX]"
