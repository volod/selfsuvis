# Configuration

Defaults are in `env/dev.env`, `env/test.env`, and `env/prod.env`. Set `APP_ENV` (dev|test|prod) to select; env vars override file values. See `env/README.md`.

## Database

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `postgresql://selfsuvis:selfsuvis@postgres:5432/selfsuvis` | PostgreSQL connection string. SQLite is no longer used. |

Run `python scripts/migrate_postgres.py` once after first `docker compose up postgres` to create all tables.

## Models and embeddings

| Variable | Default | Notes |
|---|---|---|
| `MODEL_NAME` | `openclip` | `openclip` \| `dinov2` \| `dinov3`. Use `dinov3` for active learning scoring. |
| `OPENCLIP_MODEL` | `ViT-B-16` | OpenCLIP model name |
| `OPENCLIP_PRETRAINED` | `openai` | OpenCLIP pretrained weights |
| `DEVICE` | `auto` | `auto` (prefers CUDA) \| `cpu` \| `cuda` |
| `USE_FP16` | `true` | FP16 on CUDA (reduces VRAM ~2x) |
| `FLORENCE_BATCH_SIZE` | `16` | Florence-2 inference batch size. Auto-falls back to `1` on OOM. |
| `SAM_CHECKPOINT` | *(empty)* | Path to SAM checkpoint. Required to enable SAM segmentation in the agentic system. |
| `SAM_MODEL_TYPE` | `vit_h` | SAM model type |
| `LABELS_FILE` | *(built-in)* | Path to newline-separated label vocab for zero-shot CLIP tagging |

## Pipeline sampling

| Variable | Default | Notes |
|---|---|---|
| `SAMPLE_FPS_BASE` | `2` | Starting adaptive frame rate for search keyframes |
| `SAMPLE_FPS_MIN` | `0.5` | Minimum adaptive frame rate |
| `SAMPLE_FPS_MAX` | `5` | Maximum decode rate (ffmpeg) |
| `SFM_FPS` | `2` | Dense frame extraction rate for pycolmap SfM. Separate from search keyframes. |
| `HIST_THRESH` | `0.25` | Histogram / SSIM keep threshold |
| `EMBED_DRIFT_THRESH` | `0.15` | CLIP drift keep threshold |
| `MAX_GAP_SEC` | `10` | Force-keep interval (seconds) |

## SfM and 3DGS

| Variable | Default | Notes |
|---|---|---|
| `PYCOLMAP_CAMERA_MODEL` | `SIMPLE_RADIAL` | `SIMPLE_RADIAL` \| `PINHOLE` \| `RADIAL`. Provide the correct model for your drone/camera hardware (DJI, GoPro, etc.) for best pose accuracy. |
| `NERFSTUDIO_API_URL` | `http://nerfstudio:8000` | URL of the nerfstudio FastAPI wrapper. Only used when the nerfstudio container is running (`docker-compose.override.yml`). |

## GPS extraction

| Variable | Default | Notes |
|---|---|---|
| `GPS_SIDECAR_PATH` | *(auto)* | Override GPS sidecar file path. Auto-detection order: ffprobe atoms → SRT sidecar → GPMF → null. |
| `GPS_FILTER_2D` | `false` | When `false` (default), uses lat-only Qdrant filter + Python lon post-filter. Set `true` only after validating 2D Qdrant payload index performance (P1 blocker — see TODOS.md). |

## Active learning

| Variable | Default | Notes |
|---|---|---|
| `AL_TAG_K` | `50` | Top-K frames per mission tagged `needs_annotation`. |
| `CHANGE_DETECTION_THRESHOLD` | `0.35` | Cosine distance threshold for change detection. Default `0.35` for CLIP, `0.25` for DINOv3. |

## Tiles

| Variable | Default | Notes |
|---|---|---|
| `TILE_SIZE` | `384` | Tile crop size in pixels |
| `STRIDE` | `256` | Tile step in pixels |
| `MAX_TILES_PER_SEGMENT` | `200` | Hard cap per frame |
| `DEDUP_COS_SIM_THRESH` | `0.95` | Cosine similarity dedup threshold |
| `PHASH_HAMMING_MAX` | `6` | Perceptual hash Hamming distance threshold |

## UI and static serving

| Variable | Default | Notes |
|---|---|---|
| `STATIC_SERVER_URL` | `http://localhost:8080` | nginx base URL for `.ply` file fetch by SuperSplat iframe |
| `SUPERSPLAT_SERVER_URL` | `http://localhost:8090` | SuperSplat viewer URL for 3DGS Streamlit embed |

## Security and limits

| Variable | Default | Notes |
|---|---|---|
| `API_KEY` | *(empty)* | **Strongly recommended for production.** When unset the API is unauthenticated; a startup warning is logged. |
| `ALLOWED_INDEX_PATHS` | *(empty)* | Comma-separated base directories for path-based indexing. **When empty, all path-based endpoints are disabled** (`/index/video path=`, `/index/dir`, etc.). A startup warning is logged. |
| `MAX_UPLOAD_BYTES` | 2 GB | Maximum size of a single video upload |
| `MAX_DOWNLOAD_BYTES` | 2 GB | Maximum size of a URL download |
| `MAX_REDIRECTS` | 5 | Maximum HTTP redirects followed |
| `ALLOW_PRIVATE_URLS` | `false` | Allow private/loopback IPs in URL downloads (dev only) |
| `PRECHECK_URL_TIMEOUT` | 20 s | Timeout for HEAD request during URL precheck |
| `FFMPEG_TIMEOUT_SEC` | 3600 s | Hard timeout for ffmpeg per video |
| `WORKER_POLL_INTERVAL` | 2.0 s | How often the worker polls for new jobs |
| `TRUST_PROXY_HEADERS` | `false` | Trust `X-Forwarded-For` for rate-limit key (enable only behind a trusted reverse proxy) |
| `RATE_LIMIT_PER_MIN` | 120 | Max requests per client per minute (0 = disabled) |
| `RATE_LIMIT_BURST` | 60 | Initial token-bucket burst size |
| `MAX_IMAGE_PIXELS` | 80 000 000 | Pixel limit for image query uploads |
| `MAX_DIR_FILES` | 5 000 | Maximum files scanned per directory index |
| `MAX_DIR_BYTES` | 50 GB | Maximum total size scanned per directory index |
| `MAX_DIR_DEPTH` | 10 | Maximum directory recursion depth |

### Security headers

The API adds these headers to every response:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Cache-Control: no-store`

### UI API key

The Streamlit UI reads `API_KEY` from the environment and forwards it as `X-API-Key` on every request. Set the same value in both the API and UI containers.

## Notes

- `MODEL_NAME=dinov3` is required for active learning scoring (`active_learning_score`). If `MODEL_NAME=openclip`, score uses CLIP embedding distance only.
- Duplicate videos are tracked in PostgreSQL `processed_files` table (SHA-256).
- Qdrant point IDs use SHA-256 (`stable_point_id`). **Upgrading from SHA-1 requires wiping Qdrant and re-indexing** — run `scripts/reset_qdrant.sh`.
- nerfstudio 3DGS runs in a separate GPU container (`docker-compose.override.yml`). The main worker container does not need nerfstudio dependencies.

### PostgreSQL migration procedure

First-time setup or after wiping the database:

1. `docker compose up postgres` (or `make up`)
2. `python scripts/migrate_postgres.py`
3. Restart API + worker
4. Optionally: `scripts/reset_qdrant.sh` (required when switching `MODEL_NAME` or after SHA-1→SHA-256 upgrade)

---
[← Helpers](helpers.md) | [Architecture →](architecture.md)
