# Production Server (Playground 1)

FastAPI API + PostgreSQL-backed async worker + Streamlit UI. Ingests mission video,
embeds frames (CLIP + DINOv3), enriches them (Florence-2 captions, ASR, OCR, depth,
detection), stores metadata in PostgreSQL and vectors in Qdrant, and answers text,
image, scene, and pose queries in real time.

Start with `make up` (compose files under `docker/core/`).

## Module map

| Path | Role |
| --- | --- |
| `src/selfsuvis/app/main.py` | FastAPI app assembly, lifespan services, security middleware |
| `src/selfsuvis/app/routers/` | `admin`, `cvat`, `health`, `index`, `jobs`, `query`, `realtime`, `robot`, `scene` |
| `src/selfsuvis/app/routers/v1/` | Ops API: event ingest, incidents, fusion rules, zones, site snapshot, SSE stream |
| `src/selfsuvis/app/services/` | `search`, `live_streams`, `coop_streams`, `realtime`, `upload_utils`, `form_templates` |
| `src/selfsuvis/app/deps.py` | API-key auth (timing-safe compare), bounded rate limiting |
| `src/selfsuvis/worker/` | Job consumer; `gpu.py` advisory GPU semaphore; `_run.py` persistent event loop |
| `src/selfsuvis/worker/handlers/` | `index`, `finetune`, `reembed`, `postflight` job handlers |
| `src/selfsuvis/ui/` | Streamlit app (`app.py`, `pages/`, `components/`) |
| `src/selfsuvis/pipeline/` | Shared pipeline: core, vision, mapping, fusion, training, media, storage, realtime |
| `src/selfsuvis/realtime/` | SLAM/pose bridge runtime + adapters (`pose`, `occupancy`, `registry`) |
| `src/selfsuvis/models/` | Retrieval backbones and local model loaders |
| `src/selfsuvis/mapper/` | ICP fusion service (separate container, no GPU) |

## Job lifecycle

1. `/index/video`, `/index/url`, or `/index/dir` creates a PostgreSQL job row.
2. The worker claims the job (`worker/handlers/index.py` -> `VideoIndexer`).
3. Frames are sampled, quality-filtered, embedded (CLIP + DINO named vectors).
4. Core enrichments run: Florence captions, Whisper ASR, OCR, depth, detection.
5. Optional stages when enabled: YOLO+SAM semantic environment graph,
   Gemma-directed tracking (SAM prompts + RF-DETR), Qwen VLM frame reasoning,
   UniDriveVLA expert pass (stored in `frame_facts_json["unidrive_vla"]`).
6. Metadata -> PostgreSQL; vectors -> Qdrant; optional postflight jobs
   (`POSTFLIGHT_MAPPING`, `POSTFLIGHT_SEMANTIC_GRAPH`) run 3D mapping and graphs.

Job types: `INDEX`, `SUPERVISED_FINETUNE`, `REEMBED`, `POSTFLIGHT_MAPPING`,
`POSTFLIGHT_SEMANTIC_GRAPH` -- one handler module per type under `worker/handlers/`.

## Query surface

| Endpoint | Mechanism |
| --- | --- |
| `POST /query/text` | OpenCLIP text embedding vs Qdrant vectors |
| `POST /query/image` | Image embedding, optional DINO vector space |
| `POST /query/scene` | PostgreSQL filtering over `frame_facts_json`, optional CLIP rerank |
| `POST /query/pose` | GPS/ENU spatial filter + vector ranking; accepts robot advisory context |

## v1 ops API (`app/routers/v1/`)

Site-operations layer used by the sencoop mesh and operator tooling:

- `POST /api/v1/events/{modality}` -- normalized sensor event ingest (`EventEnvelope`).
- `GET /api/v1/site/state` -- DB-backed site snapshot (zones + incidents).
- `incidents.py` -- list/detail/ack/dismiss/notes/search/export.
- `rules.py` -- fusion rule CRUD (which event combinations escalate).
- `zones.py` -- zone CRUD + history.
- `GET /api/v1/events/stream` -- SSE push of incident notifications.

## Realtime layer

- **MediaMTX** is the media edge: accepts RTSP/RTMP publishers, proxies upstream
  sources, and is controlled by the API via `/realtime/streams` (ADR 0006).
- **RtspCaptioner** sessions write live captions to `scene_timeline`.
- **Bridge runtimes** (`ssv-realtime-bridge`, ADR 0011): pose and occupancy adapters
  replay or bridge ROS/MAVLink-style traces into realtime ingestion without making
  any single SLAM engine mandatory. Compose files under `docker/realtime/`.
- **Coop integration**: `app/services/coop_streams.py` discovers Frigate cameras,
  registers `coop/{camera}` paths in MediaMTX, and starts captioner sessions;
  `app.state.coop_threat_aggregator` receives sencoop events (see
  [sencoop-mesh.md](sencoop-mesh.md)).

## State stores

- **PostgreSQL** (ADR 0001, 0008): `jobs`, `missions`, `frames`, `processed_files`,
  `change_detections`, `global_map` + mapping tables, CVAT/automation state
  (`cvat_tasks`, `system_state`, `gpu_jobs`), model provenance. Migrations via
  `ssv-migrate` (`python -m selfsuvis.scripts.migrate_postgres`).
- **Qdrant** (ADR 0007): frame/tile points with named vectors (CLIP + DINO, ADR 0002)
  and payloads: type, mission/robot ids, timestamps, GPS + ENU, model provenance.

## Security posture

Implemented hardening (2026-02): fail-closed auth when secrets are missing,
`hmac.compare_digest` API-key check, fail-closed empty `ALLOWED_INDEX_PATHS`,
DNS-rebinding peer-IP validation, bounded rate-limit table, security headers
middleware, CVAT webhook HMAC-SHA256 signatures, SHA-256 `stable_point_id`.
Details: `docs/reference/configuration.md` (security section).

## Design decisions

ADRs 0001-0011 under `docs/adr/` cover the load-bearing choices: single SQL store,
dual embeddings, Florence-2, pycolmap+nerfstudio, PostgreSQL-based active tagging,
MediaMTX, Qdrant named vectors, FastAPI+worker queue, coop as optional lazy
integration, graceful degradation, realtime mapping as optional sidecars.
