# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Local video semantic search server: index videos by content, then search by text or image query. Uses OpenCLIP for embeddings, Qdrant for vector storage, FastAPI as the API, Streamlit as the UI, and a background worker for video processing.

## Commands

### Development environment
```bash
make venv          # Create .venv and install all deps (requires uv on PATH)
```

### Run the stack
```bash
make up            # Build and start all containers (API, worker, Qdrant, UI)
make down          # Stop all containers
make logs          # Stream logs (last 100 lines)
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
```

### Lint
```bash
make lint          # ruff check + ruff format --check
```

## Architecture

### Services (each is a separate container / process)
- **`app/`** — FastAPI API. Handles HTTP, auth, rate limiting, job enqueueing, and query serving. Entrypoint: `app/main.py`.
- **`worker/main.py`** — Long-running background process. Polls SQLite job queue, downloads/copies video, runs the full indexing pipeline, marks jobs done.
- **`ui/app.py`** — Streamlit frontend. Forwards `API_KEY` header on every request to the API.
- **Qdrant** — External vector DB. Named vectors: `clip` (always present), `dino` (optional).

### Model wrappers (in `models/`)
- **`models/openclip_model.py`** — `OpenCLIPEmbedder`: batched image + text embedding via OpenCLIP. Configured by `OPENCLIP_MODEL`, `OPENCLIP_PRETRAINED`, `DEVICE`, `USE_FP16`.
- **`models/dino_model.py`** — `DINOEmbedder`: DINOv2/v3 image embedding. Loaded only when `MODEL_NAME` is `dinov2` or `dinov3`.

### Pipeline (in `pipeline/`)
The indexing pipeline lives entirely in `pipeline/` and is driven by `VideoIndexer` in `pipeline/indexer.py`:

1. **Frame extraction** (`ffmpeg_utils.py`) — ffmpeg decodes at `SAMPLE_FPS_MAX`
2. **Adaptive sampling + stabilisation** (`heuristics.py`) — histogram diff, phase correlation, embed drift, motion level
3. **Segment / keyframe selection** — per-segment keyframes chosen
4. **Full-frame CLIP embedding** — each keyframe embedded and upserted to Qdrant
5. **Tile extraction** (`TILE_SIZE`/`STRIDE`) — overlapping sliding window
6. **Tile quality filters** — blur, intensity, sky, edge density, std, entropy (`heuristics.py`)
7. **Dedup** — perceptual hash LRU (`dedup.py`) + cosine similarity via `RecentEmbeddingIndex` (`recent_index.py`)
8. **Qdrant upsert** (`qdrant_utils.py`) — stable point IDs derived with SHA-256

### Agentic scene-understanding system (optional, separate from main indexing)
`pipeline/agentic_system.py` implements a multi-agent pipeline for structured scene analysis:
- **`image_to_text_agent`** — generates scene descriptions; optionally uses `OpenCLIPTagger` (zero-shot label matching) and `SAMSegmenter` (SAM mask segmentation). Falls back to k-means colour segmentation if SAM is unavailable.
- **`ontology_agent`** / **`matching_agent`** — builds entity ontology and tracks segments across frames via IoU matching.
- **`process_frames`** — top-level entry point; writes `.jsonl` (one record per frame) + `.ontology.json` to an output directory.

`pipeline/elastic_indexer.py` can bulk-ingest the JSONL output into Elasticsearch (`ensure_index` + `bulk_index_jsonl`). Not wired into the default Docker stack.

`pipeline/vision_models.py` — `OpenCLIPTagger` (zero-shot classification using `pipeline/label_vocab.py`) and `SAMSegmenter` (SAM-based mask generation); used exclusively by the agentic system.

### Search (query path)
- Text → OpenCLIP text embedding → Qdrant `clip` vector search
- Image → OpenCLIP image embedding → Qdrant `clip` search; optionally reranked with DINO score (70/30 blend)
- `app/services/search.py` orchestrates the search; `app/state.py` holds shared model/store instances

### Job system
- `pipeline/job_db.py` — SQLite-backed job queue (`jobs.db`). API enqueues; worker polls and claims.
- `pipeline/processed_db.py` — SQLite dedup registry (`processed.db`). Tracks SHA-256 of processed files to skip re-indexing duplicates.

### Key data files (inside `./data/` by default)
- `frames/` — extracted frames keyed by `video_id`
- `tiles/` — extracted tiles keyed by `video_id/segment_id`
- `videos/` — stored video copies keyed by `video_id`
- `qdrant/` — Qdrant storage volume
- `jobs.db`, `processed.db`

## Configuration

All config lives in `pipeline/config.py` as a `Settings` class. Every field reads from an env var with a sensible default. Call `validate_settings()` at startup (already done in both `app/main.py` (via lifespan) and `worker/main.py`).

Critical env vars:
- `API_KEY` — empty = unauthenticated (startup warning logged)
- `ALLOWED_INDEX_PATHS` — comma-separated allowed base dirs for path-based indexing. **Empty = all path endpoints return 403** (fail-closed)
- `MODEL_NAME` — `openclip` (default) | `dinov2` | `dinov3`
- `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`
- `DATA_DIR` — root for frames/tiles/videos/DBs
- `DEVICE` — `auto` (default, prefers CUDA) | `cpu` | `cuda`; `USE_FP16` (default `true`) — FP16 on CUDA
- `SAM_CHECKPOINT`, `SAM_MODEL_TYPE` — required to activate SAM segmentation in the agentic system
- `LABELS_FILE` — path to newline-separated label vocab for zero-shot CLIP tagging (defaults to `pipeline/label_vocab.py` built-in list)
- `ALLOW_PRIVATE_URLS` — allow private/loopback IPs in URL downloads (default `false`)

## Security invariants

- API key check uses `hmac.compare_digest` (timing-safe) in `app/deps.py`
- Rate limiter is a per-client token bucket with LRU eviction (cap `_MAX_LIMITERS=50_000`)
- Path-based endpoints (`/index/video path=`, `/index/dir`, etc.) validate against `ALLOWED_INDEX_PATHS` via `resolve_allowed_path()` in `pipeline/utils.py`
- Qdrant point IDs use SHA-256 (`stable_point_id`). **Upgrading from SHA-1 requires wiping Qdrant and re-indexing** — run `scripts/reset_qdrant.sh`
- URL downloads validate peer IP post-connect to prevent DNS rebinding (`pipeline/net_utils.py`)

## Testing notes

- Unit tests in `tests/unit/` require no running services. Some (`test_frame_extractor.py`, `test_heuristics.py`) depend on cv2; skip with `make test-unit-no-cv2` if there's an OpenCV/numpy version mismatch.
- cv2-free unit tests: `test_utils.py`, `test_utils_path.py`, `test_dedup.py`, `test_job_db.py`, `test_downloader.py`, `test_config.py`, `test_deps.py`, `test_net_utils.py`, `test_ffmpeg.py`.
- Integration tests use `data_test/` and `cache_test/` volumes (not `data/`) to avoid polluting dev data.
- Containers run as the current host `UID`/`GID` to avoid root-owned files in `data_test`.
