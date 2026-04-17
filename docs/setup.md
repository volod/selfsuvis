# Setup

## Docker stack

Prerequisites:

- Docker with Compose support
- Optional GPU support via NVIDIA Container Toolkit

Start the main stack:

```bash
make up
python scripts/migrate_postgres.py
```

`make up` creates writable `data/` and `cache/` directories, then starts `postgres`, `qdrant`, `api`, `worker`, `ui`, `nginx`, `mediamtx`, and any default compose services. Run `scripts/migrate_postgres.py` once after PostgreSQL is available to create the schema.

Optional helpers:

- `make cvat-up` to start CVAT services
- `make logs` to follow stack logs
- `make down` to stop the stack

If GPU containers fail to start, install the toolkit with `sudo ./scripts/install_nvidia_docker.sh` or use CPU-only workflows where possible.

## Local development setup

Install host dependencies:

```bash
sudo ./scripts/install_system_deps.sh --with-python
make venv
```

`make venv` installs the project from `pyproject.toml` and uses its optional
dependency groups as the single source of truth for Python requirements.

Then start the services you need. Typical split:

```bash
docker compose -f docker/docker-compose.yml up -d postgres qdrant
python scripts/migrate_postgres.py
.venv/bin/uvicorn selfsuvis.app.main:app --reload --host 0.0.0.0 --port 8000
.venv/bin/python -m selfsuvis.worker
python -m selfsuvis.ui --server.address 0.0.0.0 --server.port 8501
```

## Default URLs

- UI: `http://localhost:8501`
- API: `http://localhost:8000`
- Qdrant: `http://localhost:6333`
- Nginx static server: `http://localhost:8080`
- SuperSplat viewer: `http://localhost:8090`
- CVAT when enabled: `http://localhost:8091`

---
[← Overview](overview.md) | [Develop →](develop.md)
