# Configuration

Defaults are in `env/dev.env`, `env/test.env`, and `env/prod.env`. Set `APP_ENV` (dev|test|prod) to select; env vars override file values. See `env/README.md`.

Key variables:
- `MODEL_NAME` = openclip | dinov2 | dinov3
- `OPENCLIP_MODEL`, `OPENCLIP_PRETRAINED`
- `SAMPLE_FPS_BASE`, `SAMPLE_FPS_MIN`, `SAMPLE_FPS_MAX`
- `HIST_THRESH`, `EMBED_DRIFT_THRESH`, `MAX_GAP_SEC`
- `TILE_SIZE`, `STRIDE`
- `DEDUP_COS_SIM_THRESH`, `MAX_TILES_PER_SEGMENT`
- `LOG_LEVEL` = DEBUG | INFO | WARNING | ERROR

## Security and limits

| Variable | Default | Notes |
|---|---|---|
| `API_KEY` | *(empty)* | **Strongly recommended for production.** When unset the API is unauthenticated; a startup warning is logged. |
| `ALLOWED_INDEX_PATHS` | *(empty)* | Comma-separated base directories for path-based indexing. **When empty, all path-based endpoints are disabled** (`/index/video path=`, `/index/dir`, `/index/precheck path=`, `/index/precheck_dir`). A startup warning is logged. |
| `MAX_UPLOAD_BYTES` | 2 GB | Maximum size of a single video upload. |
| `MAX_DOWNLOAD_BYTES` | 2 GB | Maximum size of a URL download. |
| `MAX_REDIRECTS` | 5 | Maximum HTTP redirects followed; each redirect is re-validated against private-IP rules. |
| `ALLOW_PRIVATE_URLS` | false | Set `true` to allow indexing from private/loopback URLs (development only). |
| `PRECHECK_URL_TIMEOUT` | 20 s | Timeout for the HEAD request during URL precheck. |
| `SQLITE_TIMEOUT` | 30 s | SQLite lock-wait timeout. |
| `FFMPEG_TIMEOUT_SEC` | 3600 s | Hard timeout for ffmpeg processing per video. |
| `WORKER_POLL_INTERVAL` | 2.0 s | How often the worker polls for new jobs. |
| `TRUST_PROXY_HEADERS` | false | When `true`, the `X-Forwarded-For` header is trusted for rate-limit key derivation. Only enable when behind a trusted reverse proxy that strips/overwrites this header. |
| `RATE_LIMIT_PER_MIN` | 120 | Max requests per client per minute (0 = disabled). |
| `RATE_LIMIT_BURST` | 60 | Initial token-bucket burst size. |
| `MAX_IMAGE_PIXELS` | 80 000 000 | Pixel limit for image query uploads. |
| `MAX_DIR_FILES` | 5 000 | Maximum files scanned per directory index. |
| `MAX_DIR_BYTES` | 50 GB | Maximum total size scanned per directory index. |
| `MAX_DIR_DEPTH` | 10 | Maximum directory recursion depth. |

### Security headers

The API automatically adds the following headers to every response:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Cache-Control: no-store`

### UI API key

The Streamlit UI reads `API_KEY` from the environment and forwards it as `X-API-Key` on every request. Set the same value in both the API and UI containers.

## Notes

- Frames are extracted at `SAMPLE_FPS_MAX` with ffmpeg, then adaptive skipping is applied.
- Named vectors in Qdrant: `clip` (OpenCLIP), optional `dino`.
- DINOv3 is optional and may have licensing ambiguity; use at your own risk.
- If you set `MODEL_NAME=dinov2` or `dinov3`, pre-download weights once while online (Torch Hub), then run offline.
- Duplicate videos are avoided using SHA256 hash tracking in `./data/processed.db`.

### Re-indexing after upgrade

Qdrant point IDs are derived from video/frame/tile metadata using SHA-256 (changed from SHA-1 in a previous release). If upgrading from an older version, run `scripts/reset_qdrant.sh` and re-index all videos to avoid ID collisions with stale data.

---
[← Helpers](helpers.md) | [Architecture →](architecture.md)
