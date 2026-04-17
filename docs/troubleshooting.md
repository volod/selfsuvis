# Troubleshooting

## Docker permission errors

- Add your user to the `docker` group: `sudo usermod -aG docker $USER`
- Re-login or run `newgrp docker`

## PostgreSQL schema missing

If the API or worker starts but job/frame queries fail, run:

```bash
python scripts/migrate_postgres.py
```

## Qdrant unavailable

- Confirm the `qdrant` service is running
- Check `QDRANT_HOST`, `QDRANT_PORT`, and network reachability
- `GET /health` will return 503 when Qdrant is not usable

## Path indexing is rejected

If `/index/video path=...` or `/index/dir` returns path errors, set `ALLOWED_INDEX_PATHS` to a comma-separated allowlist. When it is empty, path-based indexing is disabled intentionally.

## GPU container start failures

- Install NVIDIA Container Toolkit with `sudo ./scripts/install_nvidia_docker.sh`
- Use CPU-only or reduced-model workflows if GPU access is not available

## Model download or load failures

- Pre-fetch required assets with `python scripts/prepare_models.py`
- Set `HF_TOKEN` for gated Hugging Face models
- Lower batch sizes or disable optional multimodal stages if VRAM is insufficient

## Root-owned runtime data

If services fail with permissions under `data/` or `cache/`:

```bash
make fix-data
```

For test directories:

```bash
sudo chown -R "$(id -u):$(id -g)" data cache_test
```

## UI shows no thumbnails or maps

- Ensure the UI container can see the same `data/` mount as the API/worker
- Check `STATIC_SERVER_URL` and `SUPERSPLAT_SERVER_URL`
- Confirm `/admin/missions` returns `splat_paths` for completed map outputs

## Slow indexing

- Lower sampling and tile counts
- Disable stages you do not need
- Use smaller sidecar models or disable remote caption/facts stages

---
[← Performance](performance.md) | [Licensing →](licensing.md)
