#!/usr/bin/env bash
# Install the coop-pilot IoT edge stack on a target system without internet access.
# This script is the entry point inside the extracted release bundle (as install.sh)
# and can also be run directly from the repo for development.
#
# Usage:
#   sudo ./install.sh [options]                    # from extracted bundle
#   sudo ./scripts/coop/coop-install.sh [options]  # from repo root
#
# Options:
#   --install-dir DIR      Installation root (default: /opt/coop)
#   --data-dir DIR         Data storage root (default: INSTALL_DIR/.data)
#   --storage-dev DEV      Block device for data storage (optional).
#                          Formats ext4, mounts at DATA_DIR, adds /etc/fstab entry.
#                          Examples: /dev/sdb  /dev/mmcblk0  /dev/nvme0n1
#   --bundle BUNDLE        Service bundle to activate (default: standard):
#                            min      -- MQTT hub + LoRaWAN (sensors, mechanical control)
#                            standard -- min + Frigate NVR video surveillance
#                            video    -- MQTT hub + Frigate NVR (no LoRaWAN)
#   --hw-profile PROFILE   Hardware resource profile for Docker limits (default: min):
#                            min  -- 4-core CPU,  8 GB RAM  (thin clients, USFF 1L PCs)
#                            mid  -- 4-8-core,   16 GB RAM
#                            high -- 8+-core,    32 GB RAM
#   --metrics              Add Prometheus/Grafana/cAdvisor/node-exporter to any bundle
#   --timezone TZ          System timezone (default: Europe/Kyiv)
#   --no-docker            Skip Docker installation (assume already installed)
#   --env ENV              Env template to use: prod|dev|test (default: prod)
#   --yes                  Non-interactive (assume yes to all prompts)
#
# Exit codes: 0 success, 1 error, 2 requirements not met

set -euo pipefail

# ── Locate bundle root ────────────────────────────────────────────────────────
_SELF="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || realpath "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
_SELF_DIR="$(cd "$(dirname "$_SELF")" && pwd)"

if [[ -f "$_SELF_DIR/release.manifest" ]]; then
  # Running as install.sh at bundle root
  BUNDLE_DIR="$_SELF_DIR"
else
  # Running as scripts/coop/coop-install.sh inside bundle or from the project repo
  BUNDLE_DIR="$(cd "$_SELF_DIR/../.." && pwd)"
fi

# ── Logging helpers ───────────────────────────────────────────────────────────
log()  { printf '[install] %s\n' "$*"; }
step() { printf '\n[install] === %s ===\n' "$*"; }
die()  { printf '[install] ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf '[install] WARN:  %s\n' "$*"; }

confirm() {
  # Returns 0 (yes) or 1 (no/skip when YES=true)
  if [[ "$YES" == true ]]; then return 0; fi
  printf '[install] %s [y/N] ' "$1"
  read -r _reply
  [[ "${_reply,,}" == "y" ]]
}

require_root() {
  [[ $EUID -eq 0 ]] || die "Run this script with sudo: sudo $0 $*"
}

# ── Defaults ──────────────────────────────────────────────────────────────────
INSTALL_DIR="/opt/coop"
DATA_DIR=""          # derived from INSTALL_DIR unless --data-dir is given
STORAGE_DEV=""
BUNDLE="standard"
HW_PROFILE="min"
TIMEZONE="Europe/Kyiv"
WITH_METRICS=false
SKIP_DOCKER=false
APP_ENV="prod"
YES=false

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-dir)  INSTALL_DIR="$2"; shift 2 ;;
    --data-dir)     DATA_DIR="$2";    shift 2 ;;
    --storage-dev)  STORAGE_DEV="$2"; shift 2 ;;
    --bundle)       BUNDLE="$2";      shift 2 ;;
    --hw-profile)   HW_PROFILE="$2";  shift 2 ;;
    --timezone)     TIMEZONE="$2";    shift 2 ;;
    --metrics)      WITH_METRICS=true; shift  ;;
    --no-docker)    SKIP_DOCKER=true;  shift  ;;
    --env)          APP_ENV="$2";      shift 2 ;;
    --yes|-y)       YES=true;          shift  ;;
    -h|--help)      sed -n '2,38p' "$0"; exit 0 ;;
    *) die "Unknown option: $1. Run with --help." ;;
  esac
done

[[ "$BUNDLE" == "min" || "$BUNDLE" == "standard" || "$BUNDLE" == "video" ]] \
  || die "Invalid --bundle: $BUNDLE (use min, standard, or video)"

[[ "$HW_PROFILE" == "min" || "$HW_PROFILE" == "mid" || "$HW_PROFILE" == "high" ]] \
  || die "Invalid --hw-profile: $HW_PROFILE (use min, mid, or high)"

[[ "$APP_ENV" == "prod" || "$APP_ENV" == "dev" || "$APP_ENV" == "test" ]] \
  || die "Invalid --env: $APP_ENV (use prod, dev, or test)"

[[ -z "$DATA_DIR" ]] && DATA_DIR="${INSTALL_DIR}/.data"

require_root "$@"

# ── System requirements check ─────────────────────────────────────────────────
step "System check"

# Architecture
case "$(uname -m)" in
  x86_64)  ARCH="amd64" ;;
  aarch64) ARCH="arm64" ;;
  *) die "Unsupported architecture: $(uname -m). Supported: x86_64 (amd64), aarch64 (arm64)." ;;
esac
log "Architecture:  $ARCH"

# OS
if [[ -f /etc/os-release ]]; then
  # shellcheck source=/dev/null
  source /etc/os-release
  OS_ID="${ID:-unknown}"
  OS_CODENAME="${VERSION_CODENAME:-${VERSION_ID:-unknown}}"
  log "OS:            ${PRETTY_NAME:-$OS_ID $OS_CODENAME}"
else
  OS_ID="unknown"; OS_CODENAME="unknown"
  warn "Cannot detect OS. Proceeding anyway."
fi

# RAM
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_RAM_GB=$(( TOTAL_RAM_KB / 1024 / 1024 ))
log "RAM:           ${TOTAL_RAM_GB} GB"
if [[ $TOTAL_RAM_GB -lt 6 ]]; then
  warn "Only ${TOTAL_RAM_GB} GB RAM detected (minimum recommended: 8 GB)."
  warn "Services may fail under load. Proceeding with profile=min limits."
  PROFILE="min"
fi

# Disk space at install root
_AVAIL_MB=$(df -m "$(dirname "$INSTALL_DIR")" 2>/dev/null | awk 'NR==2{print $4}' || echo 0)
log "Disk free:     ${_AVAIL_MB} MB at $(dirname "$INSTALL_DIR")"
if [[ $_AVAIL_MB -lt 4096 && -z "$STORAGE_DEV" ]]; then
  warn "Less than 4 GB free. Consider --storage-dev to mount additional storage for recordings."
fi

log "Install dir:   $INSTALL_DIR"
log "Data dir:      $DATA_DIR"
log "Bundle:        $BUNDLE"
log "HW profile:    $HW_PROFILE"
log "Metrics:       $WITH_METRICS"
log "Timezone:      $TIMEZONE"

# ── Docker helpers ────────────────────────────────────────────────────────────
_install_docker_static() {
  local pkgs_dir="$1"
  local docker_tgz
  docker_tgz="$(ls "$pkgs_dir"/docker-*.tgz 2>/dev/null | head -1 || true)"
  [[ -n "$docker_tgz" ]] || die "Docker static tarball not found in $pkgs_dir"

  log "Extracting Docker binaries from $(basename "$docker_tgz")..."
  local tmpdir; tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT
  tar -xzf "$docker_tgz" -C "$tmpdir"

  local bins=(docker dockerd docker-proxy docker-init containerd containerd-shim-runc-v2 runc)
  for bin in "${bins[@]}"; do
    [[ -f "$tmpdir/docker/$bin" ]] && install -m 0755 "$tmpdir/docker/$bin" /usr/local/bin/
  done
  rm -rf "$tmpdir"
  trap - EXIT

  log "Installing Docker Compose plugin..."
  mkdir -p /usr/local/lib/docker/cli-plugins
  install -m 0755 "$pkgs_dir/docker-compose" /usr/local/lib/docker/cli-plugins/docker-compose

  getent group docker >/dev/null 2>&1 || groupadd docker

  log "Installing systemd units..."
  for unit in containerd.service docker.service docker.socket; do
    [[ -f "$pkgs_dir/$unit" ]] && install -m 0644 "$pkgs_dir/$unit" /etc/systemd/system/
  done

  systemctl daemon-reload
  systemctl enable --now containerd docker
  log "Docker Engine installed and started."
}

_install_docker_online() {
  local os_id="$1" os_codename="$2"
  case "$os_id" in
    ubuntu|debian)
      log "Installing Docker CE from upstream apt repository ($os_id/$os_codename)..."
      apt-get update -qq
      apt-get install -y -qq ca-certificates curl gnupg
      install -m 0755 -d /etc/apt/keyrings
      curl -fsSL "https://download.docker.com/linux/${os_id}/gpg" \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
      chmod a+r /etc/apt/keyrings/docker.gpg
      printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/%s %s stable\n' \
        "$ARCH" "$os_id" "$os_codename" > /etc/apt/sources.list.d/docker.list
      apt-get update -qq
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
      ;;
    *)
      die "Online Docker install is only supported for Ubuntu/Debian. Install Docker manually and retry with --no-docker."
      ;;
  esac
}

# ── Docker installation ───────────────────────────────────────────────────────
step "Docker"

if [[ "$SKIP_DOCKER" == true ]]; then
  log "Skipping Docker installation (--no-docker)."
  command -v docker >/dev/null 2>&1         || die "docker not found. Remove --no-docker or install manually."
  docker compose version >/dev/null 2>&1    || die "docker compose plugin not found."
  log "Found: $(docker --version)"
  log "Found: $(docker compose version)"
elif command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  log "Docker already installed: $(docker --version)"
  log "Docker Compose: $(docker compose version)"
else
  PKGS_DIR="$BUNDLE_DIR/packages/${ARCH}"
  if [[ -d "$PKGS_DIR" && -f "$PKGS_DIR/docker-compose" ]]; then
    _install_docker_static "$PKGS_DIR"
  else
    warn "No bundled Docker packages found for arch=${ARCH}."
    if confirm "Attempt online Docker installation?"; then
      _install_docker_online "$OS_ID" "$OS_CODENAME"
    else
      die "Docker is required. Install manually and retry with --no-docker."
    fi
  fi
fi

# ── External storage setup ────────────────────────────────────────────────────
if [[ -n "$STORAGE_DEV" ]]; then
  step "Storage: $STORAGE_DEV"

  [[ -b "$STORAGE_DEV" ]] || die "Block device not found: $STORAGE_DEV"

  _CURRENT_MOUNT="$(lsblk -rno MOUNTPOINT "$STORAGE_DEV" 2>/dev/null | grep -v '^$' | head -1 || true)"
  if [[ -n "$_CURRENT_MOUNT" ]]; then
    warn "Device $STORAGE_DEV is currently mounted at: $_CURRENT_MOUNT"
    confirm "Unmount and reformat as ext4? ALL DATA ON $STORAGE_DEV WILL BE LOST." \
      || die "Aborted by user."
    umount "$STORAGE_DEV" 2>/dev/null || umount -l "$STORAGE_DEV" 2>/dev/null || true
  else
    confirm "Format $STORAGE_DEV as ext4 for coop data storage? ALL DATA WILL BE LOST." \
      || die "Aborted by user."
  fi

  log "Formatting $STORAGE_DEV as ext4..."
  mkfs.ext4 -F -L "coop-data" -m 1 "$STORAGE_DEV"

  _UUID="$(blkid -s UUID -o value "$STORAGE_DEV")"
  log "UUID: $_UUID"

  mkdir -p "$DATA_DIR"
  # noatime/nodiratime: reduce write cycles on flash storage (SD, eMMC, USB).
  # commit=120: flush journal every 2 min instead of 5 s (reduces eMMC/SD wear).
  _MOUNT_OPTS="defaults,noatime,nodiratime,commit=120"
  printf 'UUID=%s  %s  ext4  %s  0  2\n' "$_UUID" "$DATA_DIR" "$_MOUNT_OPTS" \
    >> /etc/fstab
  mount "$DATA_DIR"
  log "Mounted $STORAGE_DEV at $DATA_DIR (fstab entry added)."
fi

# ── Load Docker images ────────────────────────────────────────────────────────
step "Loading Docker images"

IMAGES_DIR="$BUNDLE_DIR/images"
if [[ -d "$IMAGES_DIR" ]] && compgen -G "$IMAGES_DIR/*.tar.gz" > /dev/null 2>&1; then
  for img in "$IMAGES_DIR"/*.tar.gz; do
    log "  load $(basename "$img") ..."
    docker load < "$img"
  done
else
  warn "No bundled images found in $IMAGES_DIR."
  warn "Docker will pull images from the registry on first start (requires internet)."
fi

# ── Install directory ─────────────────────────────────────────────────────────
step "Installing to $INSTALL_DIR"

mkdir -p "$INSTALL_DIR"/{docker/coop,config,env,logs}
mkdir -p "$DATA_DIR"

cp "$BUNDLE_DIR/docker/coop/docker-compose.coop.yml" "$INSTALL_DIR/docker/coop/"
cp -r "$BUNDLE_DIR/config/"*  "$INSTALL_DIR/config/"
cp -r "$BUNDLE_DIR/env/"*     "$INSTALL_DIR/env/"

while IFS= read -r -d '' script; do
  rel="${script#"$BUNDLE_DIR/scripts/"}"
  dest_dir="$INSTALL_DIR/scripts/$(dirname "$rel")"
  mkdir -p "$dest_dir"
  install -m 0755 "$script" "$dest_dir/"
done < <(find "$BUNDLE_DIR/scripts" -name "*.sh" -print0)

# System-wide command
ln -sf "$INSTALL_DIR/scripts/coop/coop-ctl.sh" /usr/local/bin/coop-ctl
log "Installed /usr/local/bin/coop-ctl -> $INSTALL_DIR/scripts/coop/coop-ctl.sh"

# ── Kernel parameters ─────────────────────────────────────────────────────────
step "Kernel parameters"

cat > /etc/sysctl.d/99-coop.conf << 'SYSCTL'
# Required for Redis background saves without OOM risk
vm.overcommit_memory = 1
# Allow Frigate and other watchers to track many inotify targets
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 512
SYSCTL
sysctl -p /etc/sysctl.d/99-coop.conf
log "Applied: vm.overcommit_memory=1, fs.inotify limits"

# ── Timezone ──────────────────────────────────────────────────────────────────
if command -v timedatectl >/dev/null 2>&1; then
  timedatectl set-timezone "$TIMEZONE"
  log "Timezone: $TIMEZONE"
fi

# ── Environment file ──────────────────────────────────────────────────────────
step "Generating environment"

ENV_FILE="$INSTALL_DIR/.data/.env"
TEMPLATE="$INSTALL_DIR/env/${APP_ENV}.env"
[[ -f "$TEMPLATE" ]] || die "Env template not found: $TEMPLATE"

_gen_password() { openssl rand -base64 18 | tr -d '\n/+=' | head -c 20; }

cp "$TEMPLATE" "$ENV_FILE"

# Absolute DATA_DIR (Docker Compose resolves relative paths from --project-directory)
sed -i "s|^DATA_DIR=.*|DATA_DIR=${DATA_DIR}|"                                              "$ENV_FILE"
sed -i "s|MOSQUITTO_HEALTH_PASSWORD=REPLACE_ME|MOSQUITTO_HEALTH_PASSWORD=$(_gen_password)|" "$ENV_FILE"
sed -i "s|CHIRPSTACK_PG_PASSWORD=REPLACE_ME|CHIRPSTACK_PG_PASSWORD=$(_gen_password)|"       "$ENV_FILE"
sed -i "s|CHIRPSTACK_API_SECRET=REPLACE_ME|CHIRPSTACK_API_SECRET=$(openssl rand -base64 32 | tr -d '\n')|" "$ENV_FILE"
sed -i "s|CHIRPSTACK_MQTT_PASSWORD=REPLACE_ME|CHIRPSTACK_MQTT_PASSWORD=$(_gen_password)|"   "$ENV_FILE"
sed -i "s|CHIRPSTACK_GWBRIDGE_MQTT_PASSWORD=REPLACE_ME|CHIRPSTACK_GWBRIDGE_MQTT_PASSWORD=$(_gen_password)|" "$ENV_FILE"
sed -i "s|FRIGATE_MQTT_PASSWORD=REPLACE_ME|FRIGATE_MQTT_PASSWORD=$(_gen_password)|"         "$ENV_FILE"
sed -i "s|OR_ADMIN_PASSWORD=REPLACE_ME|OR_ADMIN_PASSWORD=$(_gen_password)|"                 "$ENV_FILE"
sed -i "s|GRAFANA_ADMIN_PASSWORD=REPLACE_ME|GRAFANA_ADMIN_PASSWORD=$(_gen_password)|"       "$ENV_FILE"
sed -i "s|^TZ=.*|TZ=${TIMEZONE}|"                                                          "$ENV_FILE"

# ── Hardware resource profile ─────────────────────────────────────────────────
log "Applying hardware resource profile: $HW_PROFILE"
case "$HW_PROFILE" in
  min)
    # 4-core CPU, 8 GB RAM (thin client / USFF 1L-class minimum)
    sed -i 's|^COOP_FRIGATE_CPUS=.*|COOP_FRIGATE_CPUS=2.00|'               "$ENV_FILE"
    sed -i 's|^COOP_FRIGATE_MEMORY=.*|COOP_FRIGATE_MEMORY=2560M|'          "$ENV_FILE"
    sed -i 's|^COOP_FRIGATE_DETECTOR_THREADS=.*|COOP_FRIGATE_DETECTOR_THREADS=2|' "$ENV_FILE"
    sed -i 's|^COOP_CHIRPSTACK_CPUS=.*|COOP_CHIRPSTACK_CPUS=0.50|'         "$ENV_FILE"
    sed -i 's|^COOP_CHIRPSTACK_MEMORY=.*|COOP_CHIRPSTACK_MEMORY=512M|'     "$ENV_FILE"
    sed -i 's|^COOP_CS_POSTGRES_CPUS=.*|COOP_CS_POSTGRES_CPUS=0.50|'       "$ENV_FILE"
    sed -i 's|^COOP_CS_POSTGRES_MEMORY=.*|COOP_CS_POSTGRES_MEMORY=512M|'   "$ENV_FILE"
    ;;
  mid)
    # 4-8-core CPU, 16 GB RAM
    sed -i 's|^COOP_FRIGATE_CPUS=.*|COOP_FRIGATE_CPUS=3.00|'               "$ENV_FILE"
    sed -i 's|^COOP_FRIGATE_MEMORY=.*|COOP_FRIGATE_MEMORY=4096M|'          "$ENV_FILE"
    sed -i 's|^COOP_FRIGATE_DETECTOR_THREADS=.*|COOP_FRIGATE_DETECTOR_THREADS=3|' "$ENV_FILE"
    sed -i 's|^COOP_CHIRPSTACK_CPUS=.*|COOP_CHIRPSTACK_CPUS=0.75|'         "$ENV_FILE"
    sed -i 's|^COOP_CHIRPSTACK_MEMORY=.*|COOP_CHIRPSTACK_MEMORY=768M|'     "$ENV_FILE"
    sed -i 's|^COOP_CS_POSTGRES_CPUS=.*|COOP_CS_POSTGRES_CPUS=0.75|'       "$ENV_FILE"
    sed -i 's|^COOP_CS_POSTGRES_MEMORY=.*|COOP_CS_POSTGRES_MEMORY=768M|'   "$ENV_FILE"
    ;;
  high)
    # 8+-core CPU, 32 GB RAM
    sed -i 's|^COOP_FRIGATE_CPUS=.*|COOP_FRIGATE_CPUS=6.00|'               "$ENV_FILE"
    sed -i 's|^COOP_FRIGATE_MEMORY=.*|COOP_FRIGATE_MEMORY=8192M|'          "$ENV_FILE"
    sed -i 's|^COOP_FRIGATE_DETECTOR_THREADS=.*|COOP_FRIGATE_DETECTOR_THREADS=4|' "$ENV_FILE"
    sed -i 's|^COOP_CHIRPSTACK_CPUS=.*|COOP_CHIRPSTACK_CPUS=1.50|'         "$ENV_FILE"
    sed -i 's|^COOP_CHIRPSTACK_MEMORY=.*|COOP_CHIRPSTACK_MEMORY=1024M|'    "$ENV_FILE"
    sed -i 's|^COOP_CS_POSTGRES_CPUS=.*|COOP_CS_POSTGRES_CPUS=1.00|'       "$ENV_FILE"
    sed -i 's|^COOP_CS_POSTGRES_MEMORY=.*|COOP_CS_POSTGRES_MEMORY=1024M|'  "$ENV_FILE"
    ;;
esac

# ── Compose profiles ──────────────────────────────────────────────────────────
# Map bundle + metrics flag -> COOP_COMPOSE_PROFILES (read by coop-ctl on every start)
case "$BUNDLE" in
  min)      _PROFILES="lorawan" ;;
  standard) _PROFILES="lorawan,video" ;;
  video)    _PROFILES="video" ;;
esac
[[ "$WITH_METRICS" == true ]] && _PROFILES="${_PROFILES},metrics"

if grep -q '^COOP_COMPOSE_PROFILES=' "$ENV_FILE" 2>/dev/null; then
  sed -i "s|^COOP_COMPOSE_PROFILES=.*|COOP_COMPOSE_PROFILES=${_PROFILES}|" "$ENV_FILE"
else
  printf 'COOP_COMPOSE_PROFILES=%s\n' "$_PROFILES" >> "$ENV_FILE"
fi
# Remove legacy COOP_METRICS_ENABLED if present
sed -i '/^COOP_METRICS_ENABLED=/d' "$ENV_FILE" 2>/dev/null || true

chmod 0600 "$ENV_FILE"
log "Generated $ENV_FILE (profile=$PROFILE, metrics=$WITH_METRICS)"

# ── Log rotation ──────────────────────────────────────────────────────────────
step "Log rotation"

cat > /etc/logrotate.d/coop << LOGROTATE
${DATA_DIR}/mosquitto/log/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    postrotate
        docker exec coop-mosquitto mosquitto_pub -h localhost -t '\$SYS/broker/reload' -n 2>/dev/null || true
    endscript
}

${INSTALL_DIR}/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
}
LOGROTATE
log "Logrotate: /etc/logrotate.d/coop"

# Configure Docker daemon log limits globally (applies to all containers unless overridden)
DOCKER_DAEMON_JSON="/etc/docker/daemon.json"
if [[ ! -f "$DOCKER_DAEMON_JSON" ]]; then
  mkdir -p /etc/docker
  cat > "$DOCKER_DAEMON_JSON" << 'JSON'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "5"
  }
}
JSON
  log "Configured Docker global log limits (50 MB x 5 files per container)."
fi

# ── systemd service ───────────────────────────────────────────────────────────
step "systemd service"

cat > /etc/systemd/system/coop-stack.service << UNIT
[Unit]
Description=Coop IoT Edge Stack
Documentation=file://${INSTALL_DIR}/README
After=docker.service network-online.target
Requires=docker.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${INSTALL_DIR}
Environment=COOP_INSTALL_DIR=${INSTALL_DIR}
ExecStart=${INSTALL_DIR}/scripts/coop/coop-ctl.sh start
ExecStop=${INSTALL_DIR}/scripts/coop/coop-ctl.sh stop
ExecReload=${INSTALL_DIR}/scripts/coop/coop-ctl.sh restart
TimeoutStartSec=180
TimeoutStopSec=60
Restart=on-failure
RestartSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=coop-stack

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable coop-stack.service
log "Enabled coop-stack.service (auto-start on boot)"

# ── Bootstrap and start ───────────────────────────────────────────────────────
step "First start"

# Export compose profiles so bootstrap starts the right services.
_FIRST_PROFILES="$(grep -E '^COOP_COMPOSE_PROFILES=' "$ENV_FILE" | cut -d= -f2 | tr -d '[:space:]' || true)"
[[ -n "$_FIRST_PROFILES" ]] && export COMPOSE_PROFILES="$_FIRST_PROFILES"

# bootstrap: checks .env, generates TLS certs + MQTT users, starts the stack.
COOP_INSTALL_DIR="$INSTALL_DIR" \
  "$INSTALL_DIR/scripts/coop/coop-bootstrap.sh" up -d

# ── Summary ───────────────────────────────────────────────────────────────────
printf '\n'
log "================================================================"
log "Installation complete."
log ""
log "  Install dir : $INSTALL_DIR"
log "  Data dir    : $DATA_DIR"
log "  Bundle      : $BUNDLE"
log "  HW profile  : $HW_PROFILE"
log "  Profiles    : $_FIRST_PROFILES"
log "  Env file    : $INSTALL_DIR/.data/.env"
log ""
log "  Endpoints:"
log "    ChirpStack UI      http://localhost:8080"
log "    ChirpStack REST    http://localhost:8090"
log "    Frigate NVR        https://localhost:8971"
log "    MQTT plain         localhost:1883"
log "    MQTT TLS           localhost:8883"
log "    LoRaWAN UDP        localhost:1700"
if [[ "$WITH_METRICS" == true ]]; then
  log "    Prometheus         http://localhost:9090"
  log "    Grafana            http://localhost:3000"
fi
log ""
log "  Control:"
log "    coop-ctl start | stop | restart | status | logs | ps"
log "    systemctl status coop-stack"
log ""
log "  Credentials: sudo cat $INSTALL_DIR/.data/.env"
log "================================================================"
