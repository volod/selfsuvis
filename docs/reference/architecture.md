# Architecture

## Three-playground model

Selfsuvis is organized around three loosely coupled playgrounds that share
storage and model artifacts but have independent lifecycles:

```text
Playground 1 — Production server
  FastAPI API + async worker + Streamlit UI
  Serves embedding queries; consumes jobs from PostgreSQL
  Writes frame vectors to Qdrant

Playground 2 — Local research pipeline
  LangGraph 36-step workflow (phase 0-7 + SSL + active learning)
  Runs per-mission analysis, trains self-supervised models
  Produces ONNX exports and fine-tune artifacts for Playground 1

Playground 3 — Coop IoT mesh
  MQTT subscriber, ChirpStack, Frigate, MediaMTX
  Ingests live sensor and camera streams
  Triggers indexing jobs in Playground 1 via the REST API

        Coop stack (live sensors)
              |
              | MQTT events / Frigate detections
              v
        Production server ----[REST / Qdrant]---- Client / Robot
              ^
              | re-embed + fine-tune artifacts
              |
        Local pipeline (per-mission analysis)
```

The three playgrounds share:
- A single PostgreSQL instance for job queuing and mission state
- A single Qdrant instance for frame and tile vectors
- The `selfsuvis.config` facade (`from selfsuvis.config import settings,
  coop_settings, realtime_settings`) for validated configuration

## Repository structure

```text
src/selfsuvis/
  app/          FastAPI routers, request dependencies, and API services
  models/       retrieval backbones and local model loaders
  pipeline/     indexing, mapping, media, storage, training, workflows
  worker/       PostgreSQL-backed async job worker
    handlers/   per-job-type handler modules (index, finetune, reembed, postflight)
    gpu.py      advisory GPU semaphore
    _run.py     persistent event-loop runner shared by all handlers
  coop/         IoT sensor mesh — MQTT, ChirpStack, Frigate integration
  realtime/     SLAM bridge runtimes — pose and occupancy adapters
  config/       unified config facade (Pure Fabrication; see GRASP note below)
  scripts/      packaged helper CLIs such as `selfsuvis-env`
docker/         compose files and container definitions
tests/          unit, integration, assets, and shared test helpers
docs/           operator, developer, and decision documentation
```

> **GRASP note** — `selfsuvis.config` is a Pure Fabrication package.  It does
> not represent a domain concept; it exists to reduce coupling between callers
> and three separate config subsystems and to provide a single validation
> entry point (`validate_all()`).  Existing deep imports such as
> `from selfsuvis.pipeline.core.config import settings` continue to work.

### Test structure

`tests/unit/` mirrors `src/selfsuvis/` where practical:

```text
tests/unit/
  app/
  models/
  pipeline/
  scripts/
  worker/
```

Reusable fake DB pools, factories, and non-fixture test helpers live in `tests/support/`.
The one intentional flat unit test is `tests/unit/test_multisite_enu.py`, which remains
at the root because it spans app, storage, and worker behavior together.

## Runtime architecture

```text
client/UI
   |
   v
FastAPI API  ----> PostgreSQL
   |                jobs, missions, frames, automation state
   |
   +-----------> Qdrant
   |              frame/tile vectors and payloads
   |
   +-----------> worker
                   consumes queued jobs
                   worker/handlers/index.py    -- INDEX (VideoIndexer)
                   worker/handlers/finetune.py -- SUPERVISED_FINETUNE
                   worker/handlers/reembed.py  -- REEMBED
                   worker/handlers/postflight.py -- POSTFLIGHT_MAPPING
                                                    POSTFLIGHT_SEMANTIC_GRAPH
```

Optional services:

- `nerfstudio` for splat generation
- `mapper` for map registration/fusion work
- `mediamtx` for stream ingestion and live RTSP/RTMP path management
- `cvat` for annotation workflows
- `coop` IoT edge stack: Mosquitto, ChirpStack, Frigate (see below)

### coop — IoT edge monitoring layer (Playground 3)

The `selfsuvis.coop` package adds a stationary site-monitoring layer on top of the
core selfsuvis API. It is fully optional: all components are lazy-imported and the
API starts normally when the MQTT broker is unreachable or `aiomqtt` is not installed.

```text
LoRaWAN sensors ──► ChirpStack ──► Mosquitto MQTT ──┐
Frigate cameras ─────────────────────────────────────┤
                                                      ▼
                                               MqttSubscriber
                                               (background task)
                                                      │
                       ┌──────────────────────────────┤
                       ▼                              ▼
               SiteStateAggregator           CoopRealtimeIngestor
               SensorMeshFusion                       │
               SceneSynthesizer              RealtimeThreatAggregator
                       │                              │
               GET /site/*                   GET /site/threat
               WS  /site/stream              robot advisory API
```

**CoopStreamService** (FastAPI lifespan):
  - Queries Frigate `/api/cameras` to discover enabled cameras
  - Registers each as `coop/{camera}` in MediaMTX (RTSP re-stream)
  - Starts an `RtspCaptioner` session per camera writing to `scene_timeline`
  - Optionally starts `SoundAnalyzer` per camera (faster-whisper + FFT)

**SceneSynthesizer**:
  - Fuses `SiteState` + `scene_timeline` captions into a prompt
  - Sends to `REASONING_API_URL` (OpenAI-compatible backend)
  - Returns `SceneSynthesis` (cached 10 s) at `GET /site/synthesis`

**pipeline/realtime/coop_ingest.py**:
  - `sensor_reading_to_event()` → `SensorEvent(sensor_type="lorawan")`
  - `camera_event_to_threat()` → `ThreatEvent(sensor_type="camera")`
  - Sector ID derived from GPS grid at ~110 m resolution

Docker compose: `docker/coop/docker-compose.coop.yml` joins `selfsuvis-net`.
Full reference: [coop — Integration Guide](../coop/integration.md).

### MediaMTX role

`mediamtx` is the live media edge for production deployments:

- accepts RTSP / RTMP publishers from drones, cameras, or test ffmpeg clients
- can proxy upstream RTSP / RTMP sources into a named path
- exposes an internal control API consumed by the FastAPI `/realtime/streams` endpoints
- provides the RTSP endpoint consumed by the background `RtspCaptioner` runtime

The SelfSuvis API service owns stream path lifecycle. Operators interact with `/realtime/streams`; the API then talks to MediaMTX over the internal compose network.

## Indexing flow

1. A request to `/index/video`, `/index/url`, or `/index/dir` creates a PostgreSQL job.
2. The worker claims the job and runs the indexing pipeline.
3. Video frames are sampled, quality-filtered, and embedded.
4. Core multimodal enrichments run: Florence captions, ASR, OCR, depth, and detection.
5. Optional higher-level analysis stages run when enabled:
   - YOLO + SAM semantic environment graph construction
   - Gemma-directed tracking with SAM prompts and RF-DETR sequence tracking
   - Qwen VLM detailed frame reasoning
   - UniDriveVLA expert understanding / perception / planning analysis
6. Metadata is written to PostgreSQL and vectors are written to Qdrant.
7. Optional downstream stages run: 3D mapping, reports, active learning, fine-tune triggers,
   model distillation, ONNX export, and multi-model comparison artifacts.

## Algorithmic additions

Recent pipeline additions that materially changed the system architecture:

- **Semantic environment graph**: production indexing and local runs can cluster mission
  detections into a mission-scoped semantic graph, persisted as JSON/markdown artifacts
  and referenced from downstream reports.
- **Gemma-directed tracking**: Gemma produces structured scene/object hints, SAM converts
  those hints into segmentation prompts, and RF-DETR tracks the selected object classes
  across the frame sequence.
- **UniDriveVLA expert pass**: an OpenAI-compatible vision backend produces normalized
  `understanding`, `perception`, `planning`, and `mixture_of_experts` outputs that are
  stored in `frame_facts_json["unidrive_vla"]` and summarized in local-run artifacts.
- **Resource-aware env generation**: `selfsuvis-env` generates a project-root `.env`
  from packaged presets and detected hardware, which is now the standard way to bootstrap
  local configuration.
- **Startup preflight and local analytics**: the CLI checks required model caches,
  Python packages, local sidecar model presence, and service reachability before a
  run starts; completed runs emit `analysis_summary.json` so a human can inspect
  modality coverage, degradation, tracking, mapping, training, and artifact health.
- **Realtime bridge runtimes**: pose and occupancy adapters can replay or bridge
  ROS/MAVLink-style traces into the realtime ingestion layer without making any
  single SLAM or mapping engine mandatory.
- **Security hardening**: production auth now fails closed when required secrets
  are missing, CVAT webhooks use HMAC-SHA256 body signatures, and API rate-limit
  buckets are bounded.

## Shared utility packages

Two packages centralise cross-cutting helpers that would otherwise be copy-pasted into every model file:

### `pipeline/core/gpu_utils.py`

GPU and device utilities used by every model loader and vision pipeline stage:

| Symbol | Purpose |
|--------|---------|
| `is_cuda_oom(exc)` | Returns `True` when `exc` is a CUDA out-of-memory error (works for both `torch.cuda.OutOfMemoryError` and older `RuntimeError` messages). |
| `resolve_device(device_cfg=None)` | Maps `settings.DEVICE` (or an explicit string) to `"cuda"`, `"mps"`, or `"cpu"` with proper availability checks including Apple MPS. |
| `pipeline_device_arg(device)` | Converts a device string to the integer HuggingFace `pipeline()` expects: `-1` for CPU, `0` for everything else. |

All three are re-exported from `pipeline.core` for convenience:

```python
from selfsuvis.pipeline.core import is_cuda_oom, resolve_device, pipeline_device_arg
```

### `pipeline/vision/registry.resolve_model_id`

Helper used by every vision model wrapper to avoid duplicating the four-line "read setting → auto-select → fallback" pattern:

```python
from selfsuvis.pipeline.vision.registry import resolve_model_id

def _resolve_model_id() -> str:
    return resolve_model_id(settings.DEPTH_MODEL, "depth", "depth-anything/Depth-Anything-V2-Base-hf")
```

When the setting is non-empty and not `"auto"`, the value is returned as-is. Otherwise `auto_select` applies the current policy for that model family, falling back to the explicit *fallback* ID if the catalog has no match.

## Query architecture

- `/query/text`: OpenCLIP text embedding against Qdrant vectors
- `/query/image`: image embedding with optional DINO vector space
- `/query/scene`: PostgreSQL filtering over `frame_facts_json` with optional CLIP reranking
- `/query/pose`: GPS or ENU spatial filtering plus vector ranking

## Main state stores

### PostgreSQL

Holds:

- `jobs`
- `missions`
- `frames`
- `processed_files`
- `change_detections`
- `global_map` and related mapping tables
- CVAT and automation state such as `cvat_tasks`, `system_state`, `gpu_jobs`, and model provenance tables

### Qdrant

Stores frame and tile points with named vectors and retrieval payloads such as:

- frame/tile type
- mission and robot IDs
- timestamps
- GPS and ENU coordinates
- model-version provenance

## Coordinate model

The spatial pipeline uses GPS payloads for broad filtering and ENU coordinates for local-map and robot-oriented queries when available.

---
[← Configuration](configuration.md) | [Pipeline →](pipeline.md)
