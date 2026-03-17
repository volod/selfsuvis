# Setup

## Prerequisites
- Docker + Docker Compose
- NVIDIA Container Toolkit for GPU access (optional; use `make test-no-gpu` or run without GPU)

## Start services
```bash
make up
```

`make up` runs `data-dirs` first to create `./data` and `./cache` with correct ownership, then starts qdrant, api, worker, and ui. All containers run as the current host user so files in `data/` and `cache/` are owned by you.

For GPU: install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) or run `sudo ./scripts/install_nvidia_docker.sh`. Without GPU, use `make test-no-gpu` for tests.

## Local run (no Docker)

Install system libraries and tools on the Linux host (ffmpeg, OpenCV runtime deps):

```bash
sudo ./scripts/install_system_deps.sh
```

To also install Python 3 and venv/pip:

```bash
sudo ./scripts/install_system_deps.sh --with-python
```

Then create a virtualenv and install Python dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements/requirements_prod.txt -r requirements/requirements_test.txt
```

With uv: `make venv` (creates .venv, installs pip and deps). Ensure `uv` is on PATH (e.g. `export PATH="$HOME/.local/bin:$PATH"`).

To run the API locally: `uvicorn app.main:app --reload`. You still need Qdrant and the worker. For the full stack, use `make up`.

## URLs
- Streamlit UI: http://localhost:8501
- FastAPI: http://localhost:8000
- Qdrant: http://localhost:6333

---
[← Overview](overview.md) | [Develop →](develop.md)
