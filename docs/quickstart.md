# Quick Start

## Quick comparison

**Production (Docker):**
- Full containerized stack: postgres, qdrant, api, worker, ui, nginx, mediamtx
- Deploy with `make up`
- Best for: production deployments, consistent environments, no Python dependencies on host

**Local Development:**
- Services run individually or via Docker for backing services only
- Hot-reload on API changes
- Full 35-step learning pipeline with fine-tuning and ONNX export
- Best for: development, research, model experimentation, custom pipeline modifications

---

## Prerequisites (both paths)

- Git
- Docker Engine >= 24 with Compose v2
- NVIDIA Container Toolkit (optional, for GPU support) — run `sudo ./scripts/install_nvidia_docker.sh` if not installed

**Local dev additional requirements:**
- Python 3.10
- ffmpeg, libgl1 (`sudo ./scripts/install_system_deps.sh --with-python`)

---

Choose your path based on your use case:

| Path | Use case | Guide |
|---|---|---|
| **Production (Docker)** | Deploy the full stack with all services — recommended for production, no host Python needed | [Quick Start — Production](quickstart-production.md) |
| **Local Development** | Hot-reload development, working on pipeline code, or running the 35-step learning pipeline (`selfsuvis --mode local`) | [Quick Start — Local](quickstart-local.md) |

---

## Next steps

After completing your chosen quick start:

- [Configuration](configuration.md) — full env var reference and security settings
- [Data layout](data_layout.md) — where files are written, sensor sidecars, output artifacts
- [API reference](api.md) — HTTP endpoints including the robot pose API
- [Troubleshooting](troubleshooting.md) — common errors and fixes
