# Quick Start — Production (Docker)

Deploy SelfSuvis using Docker — the recommended path for production use. No host Python installation required.

---

## Prerequisites

**Required:**
- Git
- Docker Engine >= 24 with Compose v2

**GPU (optional but recommended):**
- NVIDIA Container Toolkit — run `sudo ./scripts/install_nvidia_docker.sh` if it is not installed

---

## 1. Clone and enter the repo

```bash
git clone <repo-url>
cd selfsuvis
```

---

## 2. Configure environment

The `env/prod.env` file is pre-populated with safe defaults. Set the two values that have no default:

```
API_KEY=<choose-a-secret>
ALLOWED_INDEX_PATHS=/app/data/videos
```

Edit `env/prod.env` directly:

```bash
$EDITOR env/prod.env
```

---

## 3. Start the stack

```bash
make up
```

This builds images and starts `postgres`, `qdrant`, `api`, `worker`, `ui`, `nginx`, and `mediamtx`. Wait until you see `api-1 | Application startup complete`.

---

## 4. Run the database migration (first run only)

```bash
python -m selfsuvis.scripts.migrate_postgres
```

If you do not have Python on the host, run it inside the api container:

```bash
docker exec -it docker-api-1 python -m selfsuvis.scripts.migrate_postgres
```

---

## 5. Open the UI

```
http://localhost:8501
```

Upload a video or provide a URL. Run text or image queries once indexing finishes.

---

## Useful make targets

| Command | Action |
|---|---|
| `make up` | Build and start all containers |
| `make down` | Stop all containers |
| `make logs` | Stream last 100 lines and follow |
| `make test-unit` | Run unit tests (no Docker required) |
| `make lint` | ruff check + format check |

---

## Running the pipeline

Once all services are up, the indexing pipeline runs automatically in the background worker whenever a job is enqueued. Here are the three ways to trigger it.

### Option A — UI (easiest)

Open `http://localhost:8501`. Use the upload widget to submit a video file or paste a URL. The worker picks it up immediately; progress is shown on the jobs page.

### Option B — API upload

```bash
curl -X POST http://localhost:8000/index/video \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/mission.mp4"
```

Response returns a `job_id`. Poll for status:

```bash
curl http://localhost:8000/jobs/<job_id> -H "X-API-Key: $API_KEY"
```

### Option C — index a local directory

Set `ALLOWED_INDEX_PATHS` in `env/prod.env` to the directory you want to expose, then:

```bash
curl -X POST http://localhost:8000/index/dir \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"path": "/abs/path/to/videos"}'
```

### What happens during indexing

1. **Frame extraction** — two passes: dense frames for SfM (pycolmap pose estimation), sparse keyframes for search
2. **Captioning** — Florence-2 generates a text caption per keyframe
3. **Embedding** — OpenCLIP (and optionally DINOv3) encodes each keyframe -> stored in Qdrant
4. **Active learning tagging** — frames with high uncertainty get `al_tag=needs_annotation` for future fine-tuning
5. **Change detection** — GPS-overlapping frames from earlier missions are compared; results saved to `change_detections`
6. **Report** — HTML mission summary written to `data/reports/<mission_id>/summary.html`

Full logs appear in the worker terminal. Job status transitions: `pending -> running -> finished` (or `error`).

### Querying after indexing

**Text search:**

```bash
curl -X POST http://localhost:8000/query/text \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "damaged road surface", "top_k": 10}'
```

**Image search:**

```bash
curl -X POST http://localhost:8000/query/image \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/reference.jpg" \
  -F "top_k=10"
```

Both endpoints return a ranked list of matching frames with paths, captions, and scores. The UI exposes both via the search bar and image-upload widget.

---

## Next steps

- [Configuration](configuration.md) — full env var reference and security settings
- [Data layout](data_layout.md) — where files are written, sensor sidecars, output artifacts
- [API reference](api.md) — HTTP endpoints including the robot pose API
- [Troubleshooting](troubleshooting.md) — common errors and fixes
