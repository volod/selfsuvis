# API

## GET /health
- Health check for container orchestration. Verifies Qdrant connectivity.
- Returns `{ status: "ok", qdrant: "connected" }` or 503 on failure.

## POST /index/video
- Upload file or provide `path` (local path on the API container)
- Returns `{ video_id, job_id }`

## POST /index/url
- Form data: `url`
- Returns `{ video_id, job_id }`

## POST /index/dir
- Form data: `path` (directory inside the API container)
- Returns `{ jobs: [{video_id, job_id, video_path}, ...] }`

## POST /index/precheck
- Form data: `file` or `path` or `url`
- Returns duplicate/new/unknown with reason

## POST /index/precheck_dir
- Form data: `path` (directory), `enqueue` (true/false), `enable_tiles` (true/false)
- Returns per-file precheck results

## GET /jobs/{job_id}
- Returns job status and progress

## POST /query/image
Params:
- `top_k`
- `search_type` = tile | frame | both
- `vector_space` = clip | dino
- `enable_rerank` = true | false

## POST /query/text
Body: `{ "text": "query string" }`
Query params:
- `top_k` (1-100, default 20)
- `search_type` = tile | frame | both
- `enable_rerank` = true | false
