# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

- **Never make git commits without an explicit user request.** Do not commit as part of any workflow, review, or copy operation unless the user explicitly says "commit" or "make a commit."

## What this project is

**Outdoor autonomy perception stack** — spatial memory engine for robotics. Ingest mission video from drones, rovers, or vehicles → extract frames → estimate camera poses (pycolmap SfM) → build dense 3D maps (nerfstudio splatfacto) → embed frames (OpenCLIP + DINOv3) → caption with Florence-2 → store in PostgreSQL + Qdrant → search by text or image query.

Self-improvement loop: each mission auto-tags uncertain/novel frames (`al_tag`) for annotation, building training data for future self-supervised model fine-tuning.

Multi-mission features: change detection across GPS-overlapping missions, robot advisory API (`POST /query/pose`), persistent GPS-based global map.

## Commands

### Development environment
```bash
make venv          # Create .venv and install all deps (requires uv on PATH)
make venv-pip      # Install pip into an existing .venv (if uv created it without pip)
make docker-check  # Verify Docker daemon is reachable (run if you get permission denied)
```

### Run the stack
```bash
make up            # Build and start all containers (API, worker, Qdrant, UI, PostgreSQL, nginx, mediamtx)
make down          # Stop all containers
make logs          # Stream logs (last 100 lines)
```

### PostgreSQL migration (first run or after wipe)
```bash
# After `make up postgres` starts, run once:
python scripts/migrate_postgres.py   # creates all tables
# Reset Qdrant if switching MODEL_NAME or re-indexing from scratch:
scripts/reset_qdrant.sh
```

### Tests
```bash
# Unit tests (no Docker required, use .venv)
make test-unit
.venv/bin/python -m pytest tests/unit/ -v

# Run a single unit test file
.venv/bin/python -m pytest tests/unit/test_utils.py -v

# Unit tests without cv2 (if numpy 2.x / opencv version mismatch)
make test-unit-no-cv2

# Integration tests (requires Docker; uses GPU by default)
make test
make test-no-gpu   # If NVIDIA Container Toolkit is not installed
INDEX_DIR_PATH=/your/path make test  # Override the directory used in dir-indexing tests
```

### Lint
```bash
make lint          # ruff check + ruff format --check
```

### Cursor IDE (Linux): agent terminal sandbox

If the Agent shows **Terminal sandbox could not start** (often mentioning AppArmor on kernel 6.2+), install Cursor's AppArmor package and **fully quit and restart Cursor**:

```bash
curl -fsSL https://downloads.cursor.com/lab/enterprise/cursor-sandbox-apparmor_0.6.0_all.deb -o /tmp/cursor-sandbox-apparmor.deb
sudo dpkg -i /tmp/cursor-sandbox-apparmor.deb
```

Check that profiles loaded: `sudo aa-status | grep cursor_sandbox`. The Linux sandbox expects Landlock (`CONFIG_SECURITY_LANDLOCK=y`) and unprivileged user namespaces (`kernel.unprivileged_userns_clone=1`, usual default).

**If it still fails:** open **Settings > Cursor Settings > Agents > Auto-Run** and choose **Ask Every Time** (or approve runs when prompted) so commands are not blocked waiting for the sandbox. See [Cursor Terminal / Sandbox](https://www.cursor.com/docs/agent/terminal).

## Architecture

### Services (each is a separate container / process)
- **`app/`** — FastAPI API. Handles HTTP, auth, rate limiting, job enqueueing, query serving, and robot pose API. Entrypoint: `app/main.py`.
- **`worker/main.py`** — Async-native background process (`asyncio.run`). Polls PostgreSQL job queue via `SELECT FOR UPDATE SKIP LOCKED`, runs the full indexing pipeline, marks jobs done.
- **`ui/app.py`** — Streamlit frontend. Forwards `API_KEY` header on every request to the API.
- **PostgreSQL 16** — Primary SQL store (replaces SQLite). Stores jobs, processed_files, missions, frames, embedding_clusters, change_detections, global_map, global_map_missions. Connection via `asyncpg`.
- **Qdrant** — Vector DB. Named vectors: `clip` (always present), `dino` (when `MODEL_NAME=dinov3`).
- **nginx** — Serves `DATA_DIR/maps/` as `/static/maps/` for SuperSplat iframe; adds CORS headers for `.ply` fetch.
- **MediaMTX** — RTSP/RTMP/WebRTC streaming server for video ingestion.
- **nerfstudio** — *(optional, GPU machines only — `docker-compose.override.yml`)* 3DGS reconstruction via `ns-train splatfacto`. Exposed as a thin FastAPI HTTP wrapper called by `pipeline/mapper.py`.

### Model wrappers (in `models/` and `pipeline/vision/`)
- **`models/openclip_model.py`** — `OpenCLIPEmbedder`: batched image + text embedding via OpenCLIP. Configured by `OPENCLIP_MODEL`, `OPENCLIP_PRETRAINED`, `DEVICE`, `USE_FP16`.
- **`models/dino_model.py`** — `DINOEmbedder`: DINOv3 image embedding. Loaded only when `MODEL_NAME=dinov3`.
- **`pipeline/florence_model.py`** — Florence-2 (`microsoft/Florence-2-large`) for image-to-text captioning. Batched inference; `FLORENCE_BATCH_SIZE` env var (default 16); OOM fallback to batch=1.
- **`pipeline/vision/rfdetr.py`** — `RFDETRTracker`: RF-DETR object detection + lightweight IoU-based multi-frame tracking. Wraps `rfdetr` package (`RFDETRBase` / `RFDETRLarge`). Configured by `RFDETR_ENABLED`, `RFDETR_MODEL` (`base`/`large`), `RFDETR_CONFIDENCE`. Track IDs assigned by greedy IoU matching (threshold 0.45) across frames; reset per video.

### Pipeline (in `pipeline/`)
The indexing pipeline is orchestrated by `VideoIndexer` in `pipeline/indexer.py`. Two separate frame extraction passes per mission:

**Pass A — Dense frames for SfM** (pycolmap needs multi-view overlap):
1. `pipeline/gps_extractor.py` — extract GPS from video (ffprobe atoms → SRT sidecar → GPMF → null fallback)
2. `pipeline/ffmpeg_utils.py` — dense extraction at `SFM_FPS` (default 2 fps)
3. `pipeline/sfm.py` — pycolmap Structure-from-Motion (CPU, ~5 min/1000 frames); writes `pose_json` per frame, sets `pose_status`
4. `pipeline/mapper.py` — nerfstudio splatfacto (GPU, separate container, ~10 min); runs **after** SfM completes (`pose_status=success`); writes `maps/{mission_id}/splat.ply`

**Pass B — Sparse keyframes for search** (existing adaptive logic):
5. `ffmpeg_utils.py` — adaptive frame extraction (histogram diff, SSIM, embed drift, `MAX_GAP_SEC`)
6. `pipeline/florence_model.py` — Florence-2 captioning per keyframe; stores `caption`, `caption_confidence`
7. `models/openclip_model.py` + `models/dino_model.py` — CLIP + DINOv3 embeddings → Qdrant upsert
8. Tile extraction + quality filters + dedup (unchanged from v0)
9. `pipeline/vision/yolo.py` + `pipeline/vision/sam.py` — YOLO11 detection + SAM2/3 mask refinement per frame
10. `pipeline/vision/rfdetr.py` + `pipeline/workflows/local/steps_gemma_tracking.py` — Gemma 4 directed tracking: Gemma analyses sampled frames → structured JSON with object categories + rough bboxes → SAM segments those objects (box-prompt or CLIP-filtered auto-mask) → RF-DETR tracks Gemma-priority classes across the sequence. Stores results in `frame_facts_json["gemma_tracking"]`. Requires `RFDETR_ENABLED=true` and `GEMMA_API_URL`.
11. `pipeline/active_learning.py` — compute `active_learning_score = 0.6×DINOv3_dist + 0.4×(1−caption_confidence)`; assign `al_tag` (`needs_annotation` | `novel` | `none`)
12. `pipeline/report_generator.py` — HTML mission summary (`reports/{mission_id}/summary.html`)
13. `pipeline/change_detection.py` — post-pipeline; GPS bbox Qdrant filter + embedding distance; writes `change_detections` table

### Agentic scene-understanding system (optional, separate from main indexing)
`pipeline/agentic_system.py` implements a multi-agent pipeline for structured scene analysis — not part of the default indexing flow. See `docs/pipeline.md` for details.

### Search (query path)
- Text → OpenCLIP text embedding → Qdrant `clip` vector search
- Image → OpenCLIP image embedding → Qdrant `clip` search; optionally reranked with DINOv3 score (70/30 blend)
- "Find more like this" → Qdrant `search()` on `dino` (or `clip` fallback) embedding of clicked frame
- `app/services/search.py` orchestrates the search; `app/state.py` holds shared model/store instances

### Robot pose API
- `app/routers/robot.py` — `POST /query/pose`
- Request: `{lat, lon, alt, heading_deg, radius_m (default 50), top_k (default 5)}`
- GPS coordinates required in v1 (tx/ty/tz-only path deferred to v2)
- Latency target: p99 < 200ms (advisory use only)
- Auth: `X-API-Key` header

### Job system
- `pipeline/job_db.py` — asyncpg-backed PostgreSQL job queue. API enqueues; worker polls and claims with `SELECT FOR UPDATE SKIP LOCKED`.
- `pipeline/processed_db.py` — asyncpg-backed PostgreSQL dedup registry (`processed_files` table). Tracks SHA-256 of processed files.

### Key data files (inside `./data/` by default)
- `frames/` — extracted keyframes keyed by `video_id`
- `tiles/` — extracted tiles keyed by `video_id/segment_id`
- `videos/` — stored video copies keyed by `video_id`
- `maps/{mission_id}/splat.ply` — 3DGS output from nerfstudio splatfacto
- `reports/{mission_id}/summary.html` — auto-generated mission summary
- `qdrant/` — Qdrant storage volume
- PostgreSQL tables: `jobs`, `processed_files`, `missions`, `frames`, `embedding_clusters`, `change_detections`, `global_map`, `global_map_missions`

## Configuration

All config lives in `pipeline/config.py` as a `Settings` class. Every field reads from an env var with a sensible default. Call `validate_settings()` at startup (already done in both `app/main.py` (via lifespan) and `worker/main.py`).

Critical env vars:
- `API_KEY` — empty = unauthenticated (startup warning logged)
- `ALLOWED_INDEX_PATHS` — comma-separated allowed base dirs for path-based indexing. **Empty = all path endpoints return 403** (fail-closed)
- `DATABASE_URL` — PostgreSQL connection string (e.g. `postgresql://user:pass@postgres:5432/selfsuvis`)
- `MODEL_NAME` — `openclip` (default) | `dinov2` | `dinov3`
- `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`
- `DATA_DIR` — root for frames/tiles/videos/maps/reports
- `DEVICE` — `auto` (default, prefers CUDA) | `cpu` | `cuda`; `USE_FP16` (default `true`) — FP16 on CUDA

**Pipeline tuning:**
- `SFM_FPS` — dense frame extraction rate for pycolmap (default `2`; separate from search keyframes)
- `PYCOLMAP_CAMERA_MODEL` — `SIMPLE_RADIAL` (default) | `PINHOLE` | `RADIAL`
- `FLORENCE_BATCH_SIZE` — Florence-2 batch size (default `16`; auto-falls back to `1` on OOM)
- `GPS_SIDECAR_PATH` — override auto-detection of GPS sidecar file
- `GPS_FILTER_2D` — `false` (default; uses lat-only Qdrant filter + Python lon post-filter) | `true` (requires validated Qdrant 2D payload indexes)

**Active learning:**
- `AL_TAG_K` — top-K frames tagged `needs_annotation` per mission (default `50`)
- `CHANGE_DETECTION_THRESHOLD` — cosine distance threshold for change detection (default `0.35` for CLIP, `0.25` for DINOv3)

**UI / services:**
- `STATIC_SERVER_URL` — nginx static base URL for splat.ply fetch (default `http://localhost:8080`)
- `SUPERSPLAT_SERVER_URL` — SuperSplat viewer URL for 3DGS iframe (default `http://localhost:8090`)
- `NERFSTUDIO_API_URL` — nerfstudio FastAPI wrapper URL (default `http://nerfstudio:8000`)
- `SAM_CHECKPOINT`, `SAM_MODEL_TYPE` — required to activate SAM segmentation in the agentic system
- `LABELS_FILE` — label vocab for zero-shot CLIP tagging
- `ALLOW_PRIVATE_URLS` — allow private/loopback IPs in URL downloads (default `false`)

## Security invariants

- API key check uses `hmac.compare_digest` (timing-safe) in `app/deps.py`
- Rate limiter is a per-client token bucket with LRU eviction (cap `_MAX_LIMITERS=50_000`)
- Path-based endpoints validate against `ALLOWED_INDEX_PATHS` via `resolve_allowed_path()` in `pipeline/utils.py`
- Qdrant point IDs use SHA-256 (`stable_point_id`). **Upgrading from SHA-1 requires wiping Qdrant and re-indexing** — run `scripts/reset_qdrant.sh`
- URL downloads validate peer IP post-connect to prevent DNS rebinding (`pipeline/net_utils.py`)
- Robot API (`POST /query/pose`) requires `X-API-Key`

## Testing notes

- Unit tests in `tests/unit/` require no running services. Some (`test_frame_extractor.py`, `test_heuristics.py`) depend on cv2; skip with `make test-unit-no-cv2` if there's an OpenCV/numpy version mismatch.
- cv2-free unit tests: `test_utils.py`, `test_utils_path.py`, `test_dedup.py`, `test_job_db.py`, `test_downloader.py`, `test_config.py`, `test_deps.py`, `test_net_utils.py`, `test_ffmpeg.py`.
- New unit tests to write (from eng review): `test_gps_utils.py`, `test_active_learning.py`, `test_job_db_postgres.py`, `test_change_detection.py`, `test_report_generator.py`.
- New integration tests to write: `test_robot_api.py`, `test_migration.py`.
- Integration tests use `data_test/` and `cache_test/` volumes (not `data/`) to avoid polluting dev data.
- Containers run as the current host `UID`/`GID` to avoid root-owned files in `data_test`.
