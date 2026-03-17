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

---
[← Develop](develop.md) | [UI →](ui.md)
