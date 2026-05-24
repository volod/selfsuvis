# Coop Stack Distribution and Installation Guide

This document covers building offline installation bundles, deploying to edge computers, and configuring the coop-pilot IoT stack after installation.

---

## Overview

The coop-pilot IoT edge stack runs on commodity hardware (thin clients, USFF "1-liter class" mini PCs) at site without reliable internet access. The build machine creates a self-contained bundle that carries everything the target needs: Docker Engine binaries, Docker images, config files, and setup scripts. No internet is required on the target after the bundle is extracted.

The stack is organized around three functional layers:

1. **MQTT hub** (Mosquitto) -- always present; the central message bus for all services and external devices
2. **LoRaWAN network server** (ChirpStack) -- receives sensor telemetry from LoRa gateways (microphones, environmental sensors, door contacts) and routes decoded frames to MQTT
3. **Video NVR** (Frigate) -- IP camera recording and object detection with RTSP re-streaming

All three layers are optional at runtime via Docker Compose profiles. The installer activates the right set for the chosen bundle.

---

## Bundle Configurations

Three named bundles are supported. Choose based on site requirements.

### min -- Sensor and Control Hub

**Services:** Mosquitto + ChirpStack (postgres, redis, server, gateway-bridge, rest-api)

**Compose profiles activated:** `lorawan`

**Use when the site needs:**
- LoRaWAN microphone/audio sensor data collection (Dragino or similar LoRa sensors via EU868 gateway)
- Environmental monitoring (temperature, humidity, door contacts via LoRa)
- Mechanical system management (gate motors, relays, feeders via MQTT commands)
- Network-level LoRaWAN device management (register gateways, join devices, view frames in ChirpStack UI)
- Integration with the selfsuvis federated site map (sensor data flows to the selfsuvis API)

**Does not include:** video recording, Frigate NVR

**Typical RAM usage:** ~900 MB (all containers running)

```
+------------------+     +---------------------------------+
| Dragino LPS8N    | --> | chirpstack-gateway-bridge :1700 |
| EU868 gateway    |     +----------------+----------------+
+------------------+             |
                           mosquitto :1883/:8883
                                  |
                    +-------------+-------------+
                    |             |             |
             chirpstack    selfsuvis API   external MQTT
             UI :8080       (coop)    clients
```

---

### standard -- Full Site Stack (default)

**Services:** everything in min + Frigate NVR

**Compose profiles activated:** `lorawan,video`

**Use when the site needs:**
- All min capabilities AND
- IP/PoE camera recording with event clips
- Object detection (person, vehicle, animal) on camera feeds
- RTSP re-streams consumed by the selfsuvis SoundAnalyzer (port 8555)
- Per-section video coverage (up to 5 sections by default)

**Typical RAM usage:** ~3.5 GB (min + Frigate with 2 detector threads)

---

### video -- Standalone Video

**Services:** Mosquitto + Frigate NVR

**Compose profiles activated:** `video`

**Use when the site needs:**
- Camera surveillance without LoRaWAN infrastructure
- Sites that have audio/sensor data over standard IP network (no LoRa)
- Adding video to a site where ChirpStack runs elsewhere

**Typical RAM usage:** ~2.8 GB

---

### Adding Observability (+metrics)

Append `,metrics` to any bundle to add Prometheus, Grafana, cAdvisor, and node-exporter. This profile is recommended only when the edge system has headroom (16+ GB RAM) or the site has a dedicated monitoring operator.

| Bundle             | COOP_COMPOSE_PROFILES         |
|--------------------|-------------------------------|
| min                | `lorawan`                     |
| min + metrics      | `lorawan,metrics`             |
| standard           | `lorawan,video`               |
| standard + metrics | `lorawan,video,metrics`       |
| video              | `video`                       |
| video + metrics    | `video,metrics`               |

---

## System Requirements

| Component          | min bundle   | standard bundle | video bundle |
|--------------------|-------------|-----------------|--------------|
| CPU cores          | 2+ (4 rec.) | 4+              | 2+           |
| RAM                | 4 GB (8 rec.) | 8 GB           | 6 GB         |
| Storage (OS)       | 16 GB SSD    | 32 GB SSD       | 32 GB SSD    |
| Storage (data)     | 8 GB+        | 64 GB+ (video)  | 64 GB+       |
| OS                 | Linux x86_64 or aarch64     ||              |
| Docker             | 24+          | 24+             | 24+          |

Tested distributions: Ubuntu 22.04, Ubuntu 24.04, Debian 12. Other systemd-based Linux distributions should work with the static Docker binary path.

For video decode acceleration (Intel QSV, AMD VAAPI), `/dev/dri/renderD128` must be present. Without a GPU, set `FRIGATE_LIBVA_DRIVER_NAME=` (empty) and remove the `devices:` section from the compose file.

---

## Building a Release Bundle

Run on a machine that has Docker and internet access. The target (edge computer) does not need internet.

### Quick start

```bash
# Standard bundle, amd64, current git version
make coop-release

# With version tag
make coop-release VERSION=1.3.0

# Min bundle (no Frigate images, faster to build)
make coop-release-min VERSION=1.3.0

# Standard + metrics (Prometheus / Grafana)
make coop-release-metrics VERSION=1.3.0

# Video-only bundle
make coop-release-video VERSION=1.3.0
```

All bundles land in `./dist/`.

### Full script options

```
./scripts/coop/coop-release.sh [options]

  --version VERSION      Version tag (default: git describe --tags --always)
  --arch ARCH            amd64 or arm64 (default: amd64)
  --bundle BUNDLE        min | standard | video (default: standard)
  --with-metrics         Add Prometheus/Grafana/cAdvisor/node-exporter images
  --output-dir DIR       Output directory (default: ./dist)
  --no-images            Skip Docker image export (configs only, for testing)
  --no-docker-pkgs       Skip Docker Engine binary download
  --yes                  Non-interactive
```

Example: ARM64 standard bundle for a Raspberry Pi 4 site:

```bash
./scripts/coop/coop-release.sh --version 1.3.0 --arch arm64 --bundle standard --yes
```

### Bundle contents

```
coop-edge-1.3.0-amd64-standard/
  install.sh              # entry point (copy of scripts/coop/coop-install.sh)
  release.manifest        # version, bundle config, SHA256 checksums
  docker/
    docker-compose.coop.yml
  config/
    coop/
      mosquitto/          # mosquitto.conf, ACL, TLS cert placeholders
      chirpstack/         # chirpstack.toml
      redis/              # redis.conf
      frigate/            # config.yml with 5 section cameras (disabled by default)
      prometheus/         # prometheus.yml
      grafana/provisioning/
  env/
    prod.env              # production template (REPLACE_ME placeholders)
    dev.env
    test.env
  scripts/
    shared/
      common.sh
    coop/
      coop-ctl.sh
      coop-bootstrap.sh
      coop-compose.sh
      coop-credentials.sh
      coop-data-dirs.sh
      coop-env.sh
      coop-mosquitto-tls.sh
      coop-mqtt-users.sh
  images/
    eclipse-mosquitto_2.tar.gz
    chirpstack_chirpstack_4.tar.gz
    ...
  packages/amd64/
    docker-27.5.1.tgz
    docker-compose
    containerd.service
    docker.service
    docker.socket
```

---

## Installation

### Step 1 -- Copy and extract the bundle

```bash
scp ./dist/coop-edge-1.3.0-amd64-standard.tar.gz user@target:~/
ssh user@target
tar -xzf coop-edge-1.3.0-amd64-standard.tar.gz
cd coop-edge-1.3.0-amd64-standard
```

### Step 2 -- Run the installer

```bash
sudo ./install.sh --bundle standard --hw-profile min
```

The installer:
1. Checks architecture, OS, and RAM
2. Installs Docker Engine from the bundled static binary (falls back to apt for Ubuntu/Debian if the bundle lacks packages)
3. Optionally formats and mounts a dedicated storage device
4. Loads Docker images from the bundle
5. Creates `/opt/coop/` directory tree
6. Applies kernel parameters (`vm.overcommit_memory=1`, `fs.inotify` limits)
7. Sets the system timezone
8. Generates `.data/.env` with random secrets
9. Applies the chosen hardware resource profile to `.env`
10. Sets `COOP_COMPOSE_PROFILES` in `.env` to match the bundle
11. Configures log rotation and Docker daemon log limits
12. Creates and enables the `coop-stack` systemd service
13. Runs `coop-bootstrap.sh` for first start (generates TLS certs, MQTT users, starts containers)

### Installer options

```
sudo ./install.sh [options]

  --install-dir DIR      Where to install (default: /opt/coop)
  --data-dir DIR         Data storage root (default: /opt/coop/data)
  --storage-dev DEV      Block device for data (formats ext4 with flash-friendly options)
                         Examples: /dev/sdb  /dev/mmcblk0  /dev/nvme0n1
  --bundle BUNDLE        min | standard | video (default: standard)
  --hw-profile PROFILE   Hardware resource profile:
                           min  -- 4-core, 8 GB RAM  (default)
                           mid  -- 4-8-core, 16 GB RAM
                           high -- 8+-core, 32 GB RAM
  --metrics              Add metrics profile to COOP_COMPOSE_PROFILES
  --timezone TZ          System timezone (default: Europe/Kyiv)
  --no-docker            Skip Docker installation
  --env ENV              Env template: prod | dev | test (default: prod)
  --yes                  Non-interactive
```

### Storage device

If the edge computer has a separate disk or SD card for recordings, pass it with `--storage-dev`. The installer formats it as ext4 with write-wear optimizations (`noatime`, `nodiratime`, `commit=120`) and adds an fstab entry for automatic remount on boot.

```bash
# SD card as data storage
sudo ./install.sh --bundle standard --storage-dev /dev/mmcblk0

# Second SSD as data storage
sudo ./install.sh --bundle standard --storage-dev /dev/sdb --hw-profile mid
```

---

## Post-Installation Configuration

After installation, services start automatically. Several settings require manual configuration before the stack is fully operational.

### Credentials

All secrets are generated randomly during install. View them:

```bash
sudo cat /opt/coop/.data/.env
# or
sudo /opt/coop/scripts/coop-credentials.sh --list
```

### Mosquitto TLS

The installer generates a self-signed TLS certificate for MQTT over TLS (port 8883). For production, replace with a CA-signed certificate:

```bash
# Replace server cert and key
sudo cp your-server.crt /opt/coop/data/coop/mosquitto/certs/server.crt
sudo cp your-server.key /opt/coop/data/coop/mosquitto/certs/server.key
sudo coop-ctl restart mosquitto
```

### ChirpStack -- Register the LoRaWAN gateway

After first start (bundle: min or standard):

1. Open `http://<edge-ip>:8080` in a browser
2. Default credentials: `admin` / `admin` (change immediately)
3. Go to Gateways -- Add gateway
4. Enter the EUI printed on the gateway label (Dragino LPS8N shows it in the web UI at `Setup -> LoRa -> EUI`)
5. In the Dragino LPS8N web UI: `Setup -> LoRa -> Server Address` = edge computer IP, `Port Up/Down` = `1700`
6. Verify the gateway shows "Last seen: a few seconds ago" in ChirpStack

### ChirpStack -- Add LoRa devices (microphones / sensors)

1. Create a Device Profile: Applications -- Device profiles -- Add
   - MAC version: LoRaWAN 1.0.3
   - Regional parameters: EU868
   - Expected uplink interval: depends on sensor (e.g., 60 s for audio summary)
2. Create an Application: Applications -- Add
3. Add devices to the application using the DevEUI and AppKey printed on each sensor

### Frigate -- Configure cameras (bundle: standard, video)

Edit `/opt/coop/data/coop/frigate/config.yml`. Each section has a camera entry that is disabled by default:

```yaml
cameras:
  section_1:
    enabled: true          # change to true
    ffmpeg:
      inputs:
        - path: rtsp://admin:password@192.168.1.101:554/stream1
          roles:
            - detect
            - record
```

After saving, restart Frigate:

```bash
coop-ctl restart frigate
```

Frigate UI is at `https://<edge-ip>:8971` (self-signed cert; accept the browser warning).

### Hardware video decode

If the edge computer has an Intel GPU (Gen 9.5+):

```bash
# In /opt/coop/.data/.env
FRIGATE_LIBVA_DRIVER_NAME=iHD
```

For older Intel (before Gen 9.5): use `i965`. For AMD: `radeonsi`. For CPU-only, set to empty and remove the `devices:` block from the compose file.

Apply: `coop-ctl restart frigate`

### Grafana (metrics bundle only)

Access at `http://<edge-ip>:3000`. Default: `admin` / password from `GRAFANA_ADMIN_PASSWORD` in `.env`.

Prometheus datasource is auto-provisioned. Import dashboards from `grafana.com` or from `/opt/coop/config/coop/grafana/provisioning/`.

---

## Resource Tuning

Hardware resource limits are set in `/opt/coop/.data/.env` as `COOP_*` variables. All limits are Docker Compose `deploy.resources.limits` -- containers share unused capacity, these are ceilings not reservations.

```bash
# Edit limits
sudo nano /opt/coop/.data/.env

# Apply (no restart needed for limits; recreate containers to apply immediately)
coop-ctl restart
```

### Reference limits by profile

| Variable                      | min      | mid      | high     |
|-------------------------------|----------|----------|----------|
| COOP_MOSQUITTO_CPUS           | 0.15     | 0.15     | 0.20     |
| COOP_MOSQUITTO_MEMORY         | 128M     | 128M     | 256M     |
| COOP_CHIRPSTACK_CPUS          | 0.50     | 0.75     | 1.50     |
| COOP_CHIRPSTACK_MEMORY        | 512M     | 768M     | 1024M    |
| COOP_CS_POSTGRES_CPUS         | 0.50     | 0.75     | 1.00     |
| COOP_CS_POSTGRES_MEMORY       | 512M     | 768M     | 1024M    |
| COOP_FRIGATE_CPUS             | 2.00     | 3.00     | 6.00     |
| COOP_FRIGATE_MEMORY           | 2560M    | 4096M    | 8192M    |
| COOP_FRIGATE_SHM_SIZE         | 256mb    | 384mb    | 512mb    |
| COOP_FRIGATE_DETECTOR_THREADS | 2        | 3        | 4        |

---

## Service Management

After installation, use `coop-ctl` (available system-wide via `/usr/local/bin/coop-ctl`).

```bash
coop-ctl start             # start all configured services
coop-ctl stop              # graceful stop
coop-ctl restart           # stop then start
coop-ctl status            # container status + live resource snapshot
coop-ctl logs              # stream all logs
coop-ctl logs frigate      # stream logs for one service
coop-ctl ps                # list containers
coop-ctl shell chirpstack  # interactive shell inside a container
coop-ctl update            # pull latest images, recreate changed containers
coop-ctl config            # show resolved docker-compose configuration
coop-ctl env               # show .env (secrets masked)
```

The stack is also managed by systemd:

```bash
systemctl status coop-stack
systemctl restart coop-stack
journalctl -u coop-stack -f
```

### Changing the active bundle after install

To switch from standard to min (remove Frigate):

```bash
sudo sed -i 's/^COOP_COMPOSE_PROFILES=.*/COOP_COMPOSE_PROFILES=lorawan/' /opt/coop/.data/.env
coop-ctl restart
```

To add metrics:

```bash
sudo sed -i 's/^COOP_COMPOSE_PROFILES=lorawan,video/COOP_COMPOSE_PROFILES=lorawan,video,metrics/' /opt/coop/.data/.env
coop-ctl restart
```

---

## Port Reference

| Port      | Protocol | Service                 | Access            |
|-----------|----------|-------------------------|-------------------|
| 1700      | UDP      | LoRaWAN gateway (Semtech) | LAN (gateways)  |
| 1883      | TCP      | MQTT plain              | LAN only (firewall) |
| 8080      | TCP      | ChirpStack UI + gRPC    | localhost         |
| 8090      | TCP      | ChirpStack REST API     | localhost         |
| 8555      | TCP      | Frigate RTSP re-streams | localhost         |
| 8883      | TCP      | MQTT over TLS           | LAN + WAN sensors |
| 8971      | TCP/HTTPS| Frigate Web UI          | localhost         |
| 3000      | TCP      | Grafana                 | localhost         |
| 9090      | TCP      | Prometheus              | localhost         |

Bind addresses: `127.0.0.1:PORT` means only localhost. LAN access for ChirpStack and Frigate requires a reverse proxy (nginx, Caddy) or an SSH tunnel.

---

## Updating the Stack

### Update container images (online)

```bash
coop-ctl update
```

This pulls the latest tags for all running services and recreates changed containers with zero-downtime rolling updates.

### Deploy a new release bundle (offline)

```bash
# On build machine
make coop-release VERSION=1.4.0
scp dist/coop-edge-1.4.0-amd64-standard.tar.gz user@target:~/

# On target
tar -xzf coop-edge-1.4.0-amd64-standard.tar.gz
cd coop-edge-1.4.0-amd64-standard

# Update scripts and configs, preserve existing .env and data
sudo ./install.sh --bundle standard --no-docker --yes
```

Passing `--no-docker` skips Docker Engine reinstallation. The installer preserves `.data/.env` if it already exists (no secret regeneration).

---

## Troubleshooting

**Stack does not start after reboot**

```bash
systemctl status coop-stack
journalctl -u coop-stack --no-pager -n 50
coop-ctl status
```

**Frigate has no GPU decode**

Check that `/dev/dri/renderD128` exists and is mapped:

```bash
ls -la /dev/dri/
coop-ctl shell frigate -- vainfo
```

**Redis warning: Memory overcommit**

The installer applies `vm.overcommit_memory=1` via `/etc/sysctl.d/99-coop.conf`. Verify:

```bash
cat /proc/sys/vm/overcommit_memory   # should print 1
sudo sysctl -p /etc/sysctl.d/99-coop.conf
```

**MQTT clients cannot connect on 8883**

Verify the TLS certificate subject matches the hostname clients connect to:

```bash
openssl x509 -in /opt/coop/data/coop/mosquitto/certs/server.crt -noout -subject -dates
```

Regenerate if needed:

```bash
HOST=192.168.1.10 /opt/coop/scripts/coop-mosquitto-tls.sh "$HOST"
coop-ctl restart mosquitto
```

**ChirpStack gateway shows "Never seen"**

- Confirm the gateway EUI is registered in ChirpStack (Gateways list)
- Confirm `chirpstack-gateway-bridge` is running: `coop-ctl ps`
- Check UDP port 1700 is not blocked by firewall: `sudo ss -ulnp | grep 1700`
- On the Dragino LPS8N: verify Server Address = edge-computer-IP, UDP port = 1700

**Low disk space (recordings)**

```bash
df -h /opt/coop/data
# Reduce Frigate retention in config.yml:
#   record -> events -> retain -> days: 3  (default: 7)
coop-ctl restart frigate
```

For more detail see `docs/coop/troubleshooting.md`.
