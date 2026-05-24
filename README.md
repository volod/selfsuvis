# selfsuvis

Spatial memory engine for outdoor autonomy. Three interconnected playgrounds that
feed each other: a production server that answers queries, a local research pipeline
that builds world-model understanding, and an IoT sensor mesh that collects ground
truth from the physical world.

---

## Three Playgrounds

### 1. Production Server
`src/selfsuvis/app/` + `src/selfsuvis/worker/` + `src/selfsuvis/ui/`

FastAPI server + Streamlit UI + background worker. Ingest mission video, embed frames
with CLIP and DINOv3, caption with Florence-2, store in PostgreSQL + Qdrant, and answer
text and image search queries in real time. Optionally bridges to live RTSP streams via
MediaMTX and integrates coop sensor state into threat synthesis.

**When to use:** Deploy this to get a running search service over your mission archive.
Start here if you want to index videos and search them.

```bash
make up          # start api + worker + ui + qdrant
```

---

### 2. Local Research Pipeline
`src/selfsuvis/pipeline/`

36-step research and training workflow that processes a single mission video end to end.
Goes far beyond what the production server does: sensor fusion for physical SIGINT,
world-model video embeddings, 3D reconstruction, SSL pretraining (DAE + contrastive),
edge distillation, Qwen3 reasoning audit, and active-learning frame tagging.

This is where world-model investigation happens. Output feeds back into the production
server as fine-tuned embedders and annotated training data.

| Phase | Steps | What happens |
|---|---|---|
| **Perception core** | 1-8 | Frame extraction, CLIP+DINOv3 embedding, Gemma scene analysis, Florence-2 captioning, Whisper ASR, OCR, depth, object detection |
| **Physical SIGINT** | 9-20 | RF/SDR, thermal, multispectral, event camera, LiDAR, radar, GNSS-R, IMU, atmospheric, gas/radiation, acoustic sidecars fused into time-aligned context |
| **Tracking and 3D** | 21-27 | YOLO+SAM segmentation, RF-DETR tracking, world-model embeddings, Qwen+UniDriveVLA captioning, pycolmap SfM, nerfstudio Gaussian Splat |
| **Adaptation** | 28-36 | DAE + contrastive SSL, edge distillation (ViT-S/14 + EfficientViT-B1 ONNX), multi-model comparison, Qwen3 audit, active-learning tagging |

**When to use:** Run this locally to investigate a mission, adapt models to a new domain,
or build training data for the next production embedder.

```bash
selfsuvis --mode local --video /data/missions/my_mission.mp4
```

See [local learning path](docs/quickstart/local_path.md) for the step-by-step guide.

---

### 3. Coop Stack — IoT Sensor Mesh
`src/selfsuvis/coop_pilot/` | `docker/coop/` | `config/coop/`

Continuous site-awareness layer that ingests live sensor streams: LoRaWAN telemetry via
ChirpStack, RTSP camera feeds via Frigate NVR, MQTT acoustic and RF events, and
OpenRemote device state. Maintains a rolling-window site state (300 s sensors, 120 s
camera events) and fuses them into a unified scene synthesis.

The coop stack feeds the production server's threat synthesis (`app.state.coop_threat_aggregator`)
and the local pipeline's sensor-fusion phases (steps 9-20). Without real sensor data the
pipeline can still run with mock sidecars; with the coop stack running it ingests live feeds.

**When to use:** Run this on a gateway node alongside physical sensors to build continuous
coverage between discrete mission runs.

```bash
scripts/coop/coop-bootstrap.sh   # first-time setup
scripts/coop/coop-ctl.sh up      # start MQTT, ChirpStack, Frigate, Keycloak, etc.
```

See [coop docs](docs/coop/getting-started.md) for full setup.

---

## How the Three Connect

```
Coop stack (live sensors)
    |  MQTT / Frigate events
    v
Production server ----[REST / Qdrant]---- Client (robot, operator)
    ^
    | re-embed + fine-tune artifacts
    |
Local pipeline (per-mission analysis)
    ^
    | raw video + sensor logs
    |
Mission recordings (coop cameras / drone footage)
```

The coop stack collects. The pipeline understands. The server serves. A physical world
that can't be labelled by hand is progressively understood through the SSL loop.

---

## Quick Start

```bash
make up                      # production server (Docker)
selfsuvis --mode local ...   # local pipeline (Python venv)
scripts/coop/coop-ctl.sh up  # coop sensor mesh (Docker)
```

---

## Large Model Benchmarking

For large-model benchmarking and sidecar-based reasoning LLM comparisons, see the
[SSLM playground](src/sslm/README.md).

---

## Documents

| Section | Where |
|---|---|
| [Docs index](docs/README.md) | Full documentation index |
| [Quick start](docs/quickstart/quickstart.md) | Run any of the three stacks |
| [Local learning path](docs/quickstart/local_path.md) | 36-step essentials |
| [Architecture](docs/reference/architecture.md) | Component topology |
| [Pipeline reference](docs/reference/pipeline.md) | Pipeline data flow |
| [Configuration](docs/reference/configuration.md) | All env vars |
| [Secrets management](docs/reference/secrets-management.md) | Secrets separation and rotation |
| [Model catalog](docs/reference/model-catalog.md) | VRAM budgets, SSL models |
| [Coop stack](docs/coop/getting-started.md) | IoT sensor mesh setup |
| [Runbooks](docs/runbooks/README.md) | Per-component runbooks |
| [Architecture decisions](docs/adr/README.md) | ADR log |
