# Quick Start

## Quick comparison

**Production (Docker):**
- Full containerized stack: postgres, qdrant, api, worker, ui, nginx, mediamtx
- Deploy with `make up`
- Best for: production deployments, consistent environments, no Python dependencies on host

**Local Development:**
- Services run individually or via Docker for backing services only
- Hot-reload on API changes
- Full local learning pipeline with fine-tuning and ONNX export
- Optional coop_pilot Steps 37-43 for live IoT site monitoring
- Best for: development, research, model experimentation, custom pipeline modifications

---

## Prerequisites (both paths)

- Git
- Docker Engine >= 24 with Compose v2
- NVIDIA Container Toolkit (optional, for GPU support) — run `sudo ./scripts/install/install_nvidia_docker.sh` if not installed

**Local dev additional requirements:**
- Python 3.10
- ffmpeg, libgl1 (`sudo ./scripts/install/install_system_deps.sh --with-python`)

---

Choose your path based on your use case:

| Path | Use case | Guide |
|---|---|---|
| **Production (Docker)** | Deploy the full stack with all services — recommended for production, no host Python needed | [Quick Start — Production](quickstart-production.md) |
| **Local Development** | Hot-reload development, working on pipeline code, or running the local learning pipeline (`selfsuvis --mode local`) | [Quick Start — Local](quickstart-local.md) |
| **IoT / coop monitoring** | Add LoRaWAN sensors + Frigate cameras for live multi-modal site awareness and LLM scene synthesis | [coop_pilot — Getting Started](coop/getting-started.md) |

---

## Run the full local learning path, including coop

The practical route is two stages:

1. Run the local video pipeline (`selfsuvis --mode local`) for the core learning path.
2. Start `coop_pilot` for Steps 37-43: MQTT, LoRaWAN, Frigate, rolling site state,
   scene synthesis, and realtime threat sectors.

```bash
# 1. Prepare local venv, models, sample data, and backing services.
bash scripts/ssv/ssv-setup.sh

# 2. Run the local video pipeline. Use the exact command printed by setup,
# or start with the minimal command below.
.venv/bin/selfsuvis --mode local \
  --videos-dir data/videos \
  --no-qdrant \
  --no-sfm \
  --no-gsplat

# 3. Install coop extras and start the IoT stack for learning-path Steps 37-43.
.venv/bin/pip install -e ".[coop_pilot]"
APP_ENV=test ./scripts/coop/coop-bootstrap.sh up -d

# 4. Start the local API so /site/* endpoints can subscribe to coop MQTT.
APP_ENV=dev COOP_MQTT_HOST=localhost COOP_MQTT_PORT=1883 COOP_MQTT_TLS=false \
  COOP_FRIGATE_API_URL=http://localhost:8971 \
  .venv/bin/uvicorn selfsuvis.app.main:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
curl -s http://localhost:8000/site/state | python -m json.tool
curl -s http://localhost:8000/site/cameras | python -m json.tool
curl -s http://localhost:8000/site/mesh | python -m json.tool
curl -s http://localhost:8000/site/synthesis | python -m json.tool
curl -s http://localhost:8000/site/threat | python -m json.tool
```

Stop the coop containers when done:

```bash
APP_ENV=test ./scripts/coop/coop-compose.sh down
```

For the detailed command sequence, see [Quick Start — Learning Path Pipeline](quickstart-pipeline.md#optional-step-7--run-coop_pilot-steps-36-42).

---

## Next steps

After completing your chosen quick start:

- [Configuration](configuration.md) — full env var reference and security settings
- [MediaMTX streaming](streaming-mediamtx.md) — live RTSP/RTMP ingress, path control, and realtime stream analysis
- [Data layout](data_layout.md) — where files are written, sensor sidecars, output artifacts
- [API reference](api.md) — HTTP endpoints including the robot pose API
- [Troubleshooting](troubleshooting.md) — common errors and fixes

### IoT edge monitoring (coop_pilot)

selfsuvis ships a built-in IoT edge layer (`coop_pilot`) for monitoring physical
sites with LoRaWAN sensors and IP cameras:

- **Site state API** (`/site/*`) — rolling-window aggregation of LoRaWAN sensor
  readings and Frigate camera detections; spatial sensor mesh with GPS neighbour links
- **Scene synthesis** (`/site/synthesis`) — LLM narrative fusing all sensor + camera
  modalities into a human-readable scene description
- **Threat pipeline** (`/site/threat`) — sector-level threat map fed by sensor anomalies
  and camera detections, compatible with the robot advisory API
- **Realtime audio analysis** — per-camera `SoundAnalyzer` running faster-whisper and
  FFT acoustic event classification (alarm, engine, impact, glass break)

See [coop_pilot — Integration Guide](coop/integration.md) for the full picture.
