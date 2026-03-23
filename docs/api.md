# API

All index, jobs, and query endpoints require `X-API-Key` when `API_KEY` is set. Rate limiting applies.

## GET /index/form
- Simple HTML form to upload a local video file or submit a URL for indexing. Renders a page with file input, URL input, enable_tiles checkbox, and optional API key. No auth required to view.

## GET /health
- Health check. Verifies Qdrant connectivity.
- Returns `{ status: "ok", qdrant: "connected" }` or 503 on failure.
- No auth required.

## POST /index/video
- Form: `file` (upload) or `path` (path within ALLOWED_INDEX_PATHS), `enable_tiles` (default true)
- Returns `{ video_id, job_id }`
- 403 if path not allowed; 413 if upload exceeds MAX_UPLOAD_BYTES

## POST /index/url
- Form: `url`, `enable_tiles` (default true)
- Returns `{ video_id, job_id }`
- 400 if URL invalid (scheme, hostname, private IP blocked unless ALLOW_PRIVATE_URLS)

## POST /index/dir
- Form: `path` (directory within ALLOWED_INDEX_PATHS), `enable_tiles` (default true)
- Returns `{ jobs: [{ video_id, job_id }, ...] }`
- 403 if path not allowed

## POST /index/precheck
- Form: `file` or `path` or `url` (one required)
- Returns `{ status, reason, ... }` — duplicate/new/maybe/unknown

## POST /index/precheck_dir
- Form: `path`, `enqueue` (default false), `enable_tiles` (default true)
- Returns `{ results: [...], jobs: [...] }` — per-file precheck; jobs if enqueue=true

## GET /jobs/{job_id}
- Returns `{ status, progress, started_at, finished_at, error }`
- 400 if job_id invalid (must be 1–64 hex chars)
- 404 if job not found

## POST /query/image
- Form: `file` (image upload), `top_k` (1–100, default 20), `search_type` (both|frame|tile), `vector_space` (clip|dino), `enable_rerank` (default true)
- Returns `{ results: [...] }`

## POST /query/text
- Body: `{ "text": "query string" }` (max 1000 chars)
- Query params: `top_k`, `search_type`, `enable_rerank`
- Returns `{ results: [...] }`

## GET /missions
- Returns list of missions with metadata (mission_id, video_id, pose_status, map_status, frame_count, al_tag distribution)

## GET /missions/{mission_id}
- Returns full mission metadata including report_path

## GET /missions/{mission_id}/changes
- Returns change detections for this mission vs. prior missions (frame pairs with embedding_distance, change_score)

## GET /missions/{mission_id}/export
- **Annotation queue export.** Auth: `X-API-Key` required.
- Query params: `al_tag` (needs_annotation|novel|all, default needs_annotation), `limit` (1–500, default 200)
- Returns: ZIP download (`Content-Type: application/zip`) containing:
  - `manifest.json` — export_version, exported_at, frames array per DESIGN.md export format spec
  - `{frame_id}.jpg` — JPEG for each frame in the export
- Uses `StreamingResponse` — does not buffer full ZIP in memory
- 404 if mission not found; 400 if `al_tag` value is invalid
- Export format is the v2 CVAT import contract; do not rename fields without a version bump (see DESIGN.md)

## POST /query/pose
- **Robot advisory API.** Auth: `X-API-Key` required.
- Body: `{ "lat": float, "lon": float, "alt": float|null, "heading_deg": float, "radius_m": 50, "top_k": 5 }`
- GPS coordinates required (tx/ty/tz-only path returns 400 in v1)
- Returns `{ frames: [...], query_ms: int }` — frames near the given GPS position with captions, al_tag, pose, gps
- Latency target: p99 < 200ms (advisory use only; hard real-time obstacle avoidance is out of scope)

---
[← Develop](develop.md) | [UI →](ui.md)
