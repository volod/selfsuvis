# Developer Guide

How to run the stack, tests, lint, and services locally.

## Run the stack (Docker)

```bash
make up
```

Starts qdrant, api, worker, and ui. Uses `data-dirs` first so `./data` and `./cache` are owned by you. URLs: API http://localhost:8000, UI http://localhost:8501, Qdrant http://localhost:6333.

## Run tests

### Unit tests (no Docker)

```bash
make test-unit
# or
pytest tests/unit/ -v
```

Uses `.venv` if present. If numpy/opencv mismatch: `make test-unit-no-cv2`.

### Integration tests (Docker)

```bash
make test          # with GPU
make test-no-gpu   # without GPU
```

Runs `test-dirs`, then api, worker, qdrant, and tests container. Directory-indexing: `make test-dir`.

## Run lint

```bash
make lint
```

Runs `ruff check` and `ruff format --check`. Install ruff first: `pip install ruff`.

## Run locally (no Docker)

Prerequisites: system deps (`sudo ./scripts/install/install_system_deps.sh --with-python`), venv (`make venv`), Qdrant running (e.g. `make up` for qdrant only, or `docker run -p 6333:6333 -v $(pwd)/data/qdrant:/qdrant/storage qdrant/qdrant:v1.7.4`).

Config: `APP_ENV=dev` (default) loads `src/selfsuvis/env/dev.env`. Override with env vars or a root `.env`.

### API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Worker

```bash
python -m selfsuvis.worker
```

### UI

```bash
streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
```

Set `API_URL=http://localhost:8000` so the UI talks to your local API.

## Adding a new vision model wrapper

Each model file (under `pipeline/vision/`) should import shared helpers from the central packages rather than copy-pasting them:

```python
# Device selection
from pipeline.core import is_cuda_oom, pipeline_device_arg, resolve_device

# Model ID resolution (settings value → "auto" → GPU-aware auto-select → fallback)
from pipeline.vision.registry import resolve_model_id

def _resolve_model_id() -> str:
    return resolve_model_id(settings.MY_MODEL, "my_task", "org/default-model")

def _get_device() -> str:
    return resolve_device()
```

Do **not** copy-paste `_is_cuda_oom`, `_resolve_device`, `_get_device`, or `_pipeline_device_arg` into a new file — the canonical implementations live in `pipeline/core/gpu_utils.py`.

---
[← Setup](../quickstart/setup.md) | [API →](../reference/api.md)
