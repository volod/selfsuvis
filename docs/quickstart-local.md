# Quick Start — Local Service Setup

Set up the SelfSuvis API, worker, and UI for local development with hot-reload. This covers the Docker-backed service stack only.

> **Running the local learning pipeline (`selfsuvis --mode local`) instead?**
> See [Quick Start — Learning Path Pipeline](quickstart-pipeline.md).

---

## Prerequisites

**Required:**
- Git
- Docker Engine >= 24 with Compose v2
- Python 3.10
- ffmpeg, libgl1 (`sudo ./scripts/install_system_deps.sh --with-python`)

**GPU (optional but recommended):**
- NVIDIA Container Toolkit — run `sudo ./scripts/install_nvidia_docker.sh` if it is not installed

---

## 1. Install system dependencies and create the venv

```bash
sudo ./scripts/install_system_deps.sh --with-python
make venv
```

---

## 2. Configure environment

Generate `.env` at the project root. The generator auto-detects your GPU and RAM, picks appropriate models, and prints exactly which sidecar commands to run.

**Quick (non-interactive) — use detected hardware defaults:**

```bash
make env
```

**Interactive — choose sidecar backend, profile, models:**

```bash
make env-interactive
```

The interactive flow:
1. Shows detected hardware (GPU VRAM, RAM)
2. Asks for primary sidecar backend — `ollama` (default, easiest) or `vllm` (higher throughput, more setup)
3. Asks for profile — `minimal` / `balanced` (default) / `full`
4. Asks for environment name and output path

Both commands write to `repo_root/.env`. The config loader reads it after the packaged `src/selfsuvis/env/dev.env` defaults, so `DATABASE_URL`, `QDRANT_HOST=localhost`, `DEVICE`, and sidecar URLs are all set for you.

After generation, set the remaining values:

```bash
$EDITOR .env
```

```
API_KEY=<choose-a-secret>   # leave empty for unauthenticated dev use
HF_TOKEN=hf_xxx             # optional — gated HuggingFace models only
```

`ALLOWED_INDEX_PATHS` is pre-set to `./data/videos` by the generator. Drop mission videos there and the path-based indexing API will accept them. Change it if your videos live elsewhere.

---

## 3. Start sidecars

The generator prints the exact commands at the end of its output. Refer to those, or use the patterns below based on what you chose.

**Ollama** (default — models pulled automatically):

```bash
ollama serve                         # keep running in a terminal
ollama pull <GEMMA_API_MODEL>        # value from .env, e.g. gemma4:e4b
ollama pull <REASONING_MODEL>        # value from .env, e.g. deepseek-r1:14b
```

**vLLM** (if chosen for Qwen or Gemma — each in its own terminal):

```bash
# Qwen visual model (port 8010)
python -m vllm.entrypoints.openai.api_server \
  --model <QWEN_MODEL> --port 8010 --max-model-len 8192

# Gemma (port 8000) — only if GEMMA_API_BACKEND=vllm
python -m vllm.entrypoints.openai.api_server \
  --model <GEMMA_API_MODEL> --port 8000 --max-model-len 8192
```

Replace `<GEMMA_API_MODEL>` / `<QWEN_MODEL>` / `<REASONING_MODEL>` with the values written to `.env`.

---

## 4. Start backing services only

```bash
docker compose -f docker/docker-compose.yml up -d postgres qdrant
```

---

## 5. Run the database migration (first time only)

```bash
APP_ENV=dev .venv/bin/python -m selfsuvis.scripts.migrate_postgres
```

---

## 6. Start each service in a separate terminal

```bash
# Terminal 1 — API (hot-reload enabled)
APP_ENV=dev .venv/bin/uvicorn selfsuvis.app.main:app \
  --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Worker
APP_ENV=dev .venv/bin/python -m selfsuvis.worker

# Terminal 3 — UI
APP_ENV=dev .venv/bin/python -m selfsuvis.ui \
  --server.address 0.0.0.0 --server.port 8501
```

---

## Default service URLs

| Service | URL |
|---|---|
| UI | http://localhost:8501 |
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |
| Nginx static server | http://localhost:8080 |

---

## Running the API pipeline

Once services are up, the indexing pipeline runs automatically in the background worker. Trigger jobs via the UI at `http://localhost:8501` or via API:

```bash
curl -X POST http://localhost:8000/index/video \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/mission.mp4"
```

See the [Production Quick Start](quickstart-production.md) for full pipeline and querying details.

---

## Optional: run coop_pilot locally

For learning-path Steps 37-43, run the coop Docker stack and point the local API at
the localhost MQTT and Frigate ports:

```bash
.venv/bin/pip install -e ".[coop_pilot]"
APP_ENV=test ./scripts/coop-bootstrap.sh up -d

APP_ENV=dev \
COOP_MQTT_HOST=localhost \
COOP_MQTT_PORT=1883 \
COOP_MQTT_TLS=false \
COOP_FRIGATE_API_URL=http://localhost:8971 \
.venv/bin/uvicorn selfsuvis.app.main:app \
  --reload --host 0.0.0.0 --port 8000
```

Verify:

```bash
curl -s http://localhost:8000/site/state | python -m json.tool
curl -s http://localhost:8000/site/cameras | python -m json.tool
curl -s http://localhost:8000/site/threat | python -m json.tool
```

Stop coop containers:

```bash
APP_ENV=test ./scripts/coop-compose.sh down
```

For the full local learning sequence, see
[Quick Start — Learning Path Pipeline](quickstart-pipeline.md#optional-step-7--run-coop_pilot-steps-36-42).

---

## Next steps

- [Learning Path Pipeline](quickstart-pipeline.md) — run `selfsuvis --mode local` without Docker
- [Configuration](configuration.md) — full env var reference and security settings
- [Data layout](data_layout.md) — where files are written, sensor sidecars, output artifacts
- [API reference](api.md) — HTTP endpoints including the robot pose API
- [Troubleshooting](troubleshooting.md) — common errors and fixes
