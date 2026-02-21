# Setup

## Prerequisites
- Docker + Docker Compose
- NVIDIA Container Toolkit for GPU access (optional but recommended)

## Start services
```bash
make up
```

If you need GPU inside containers:
```bash
docker compose up --build --remove-orphans
```

## Local run (no Docker)

Install system libraries and tools on the Linux host (ffmpeg, OpenCV runtime deps):

```bash
sudo ./scripts/install_system_deps.sh
```

To also install Python 3 and venv/pip:

```bash
sudo ./scripts/install_system_deps.sh --with-python
```

Then create a virtualenv, install Python dependencies, and run (API + Qdrant + worker need to be running; see docs for full stack):

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements/requirements_prod.txt -r requirements/requirements_test.txt
# Run API: uvicorn app.main:app --reload
# Or use make up for the full Docker stack.
```

With uv: `make venv` (creates .venv, installs pip and deps). Ensure `uv` is on PATH (e.g. `export PATH="$HOME/.local/bin:$PATH"`).

## Open UI
- Streamlit: http://localhost:8501
- FastAPI: http://localhost:8000
