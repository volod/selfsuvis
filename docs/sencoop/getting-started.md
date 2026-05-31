# Getting Started

This guide walks you through setting up the coop IoT monitoring stack and
integrating it with selfsuvis.

## Prerequisites

- Docker Engine 24.0+ with Compose V2
- Linux amd64 system (tested on Ubuntu 22.04+)
- At least 8GB RAM, 4 CPU cores
- Python 3.10+ (for testing and analytics)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd coop-stack-a-pilot
```

### 2. Configure Environment (First Run)

**Option A: Generate from template** (recommended):

```bash
# Production (default) — pinned images, OR_DEV_MODE=false
./scripts/sencoop/sencoop-env.sh

# Development — latest images, OR_DEV_MODE=true, databases exposed
./scripts/sencoop/sencoop-env.sh dev

# Test / CI — localhost bindings, isolated data dir, reduced limits
./scripts/sencoop/sencoop-env.sh test
```

This creates `.env` at the project root with randomly generated secrets. Save the printed credentials securely.

**Option B: Manual setup**:

```bash
cp env/prod.env .env
# Edit .env and replace any remaining REPLACE_ME values
```

To list credentials later:

```bash
./scripts/sencoop/sencoop-credentials.sh --list
```

Key variables to configure:

| Variable | Description |
|----------|-------------|
| `DATA_DIR` | Base path for bind-mounted data (default: `./data`). User-accessible. |
| `OR_HOSTNAME` | Your domain name (e.g., `coop.example.org`) |
| `OR_ADMIN_PASSWORD` | OpenRemote admin password |
| `MOSQUITTO_HEALTH_PASSWORD` | MQTT health check password |
| `CHIRPSTACK_PG_PASSWORD` | ChirpStack database password |
| `CHIRPSTACK_API_SECRET` | ChirpStack API secret (base64) |
| `CHIRPSTACK_MQTT_PASSWORD` | ChirpStack MQTT password |
| `FRIGATE_MQTT_PASSWORD` | Frigate MQTT password |

### 3. Add Cameras (Optional)

**Option A: Add via script** (RTSP or USB):

```bash
# RTSP camera
./scripts/sencoop/sencoop-camera.sh --name front_door --rtsp rtsp://user:pass@192.168.1.100:554/stream1 --restart

# USB camera (ensure device is passed to Frigate in docker/core/docker-compose.yml)
./scripts/sencoop/sencoop-camera.sh --name usb_cam --usb /dev/video0 --restart

# List configured cameras
./scripts/sencoop/sencoop-camera.sh --list
```

**Option B: Edit config manually** - see `data/coop/frigate/config.yml` (created by `coop-bootstrap.sh`) and [Sensor Integration](sensor-integration.md).

### 4. Start the Stack

Bootstrap creates `.env`, data directories, TLS certificates, and MQTT users when missing. Start the stack:

```bash
# Production (default)
./scripts/sencoop/sencoop-bootstrap.sh

# Development (applies docker-compose.dev.yml overlay)
APP_ENV=dev ./scripts/sencoop/sencoop-bootstrap.sh
```

PUID/PGID are set dynamically from your user. For compose-only (e.g. logs): `./scripts/sencoop/sencoop-compose.sh logs -f`.

**Optional:** To use a custom hostname for TLS, set `OR_HOSTNAME` in `.env` before bootstrap. Bootstrap generates certs with that hostname when missing.

Monitor startup:

```bash
docker compose logs -f
```

### 5. Verify Services

Check all containers are running:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

Expected output shows all containers as "Up" with healthy status.

## Accessing Services

| Service | URL | Notes |
|---------|-----|-------|
| OpenRemote | https://localhost | Main management UI |
| ChirpStack | http://localhost:8080 | LoRaWAN server |
| ChirpStack REST | http://localhost:8090 | REST API |
| Frigate | http://localhost:8971 | NVR interface |
| Prometheus | http://localhost:9090 | Monitoring (if enabled) |

## Integrating with selfsuvis

Once the coop IoT stack is running, connect it to the selfsuvis API:

### 1. Add coop env vars to selfsuvis `.env`

```env
# MQTT broker (matches the mosquitto container name on selfsuvis-net)
COOP_MQTT_HOST=mosquitto
COOP_MQTT_PORT=1883
COOP_MQTT_USER=selfsuvis
COOP_MQTT_PASSWORD=<your MQTT password>

# ChirpStack uplink topic
COOP_CHIRPSTACK_TOPIC=application/+/device/+/event/up

# Frigate event topics
COOP_FRIGATE_TOPIC_PREFIX=frigate
COOP_FRIGATE_API_URL=http://frigate:8971

# LLM backend for scene synthesis (optional but recommended)
REASONING_API_URL=http://gemma:8000
REASONING_MODEL=gemma3:12b
```

### 2. Start both stacks together

```bash
docker compose \
  -f docker/core/docker-compose.yml \
  -f docker/sencoop/docker-compose.sencoop.yml \
  up -d
```

### 3. Verify the integration

```bash
# Check the site state API
curl http://localhost:8080/site/state | python -m json.tool

# Check active Frigate camera sessions (should list registered cameras)
curl http://localhost:8080/site/cameras | python -m json.tool

# Request an LLM scene synthesis
curl http://localhost:8080/site/synthesis | python -m json.tool
```

### 4. Install coop extras (local dev only)

If running the API outside Docker:

```bash
pip install -e ".[sencoop]"
```

This installs `aiomqtt`, `docker`, `pandas`, `jinja2`, and `rich` — the optional
dependencies used by the MQTT subscriber, log analytics, and reporting CLI.

---

## Next Steps

- [Integration Guide](integration.md) — API endpoints, MQTT topics, scene synthesis, threat pipeline
- [Sensor Integration](sensor-integration.md) — adding LoRaWAN devices and IP cameras
- [Configuration Guide](../reference/configuration.md) — all env vars and tuning options
- [Analytics Guide](../reference/analytics.md) — log analytics CLI (`sencoop-analytics`)
- [Testing Guide](testing.md) — run tests to verify your setup
- [Troubleshooting](../operations/troubleshooting.md) — common issues and solutions
