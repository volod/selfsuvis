# Helpers

## Install scripts (run with sudo when noted)
- `./scripts/install_system_deps.sh` — ffmpeg, OpenCV deps (Linux). Add `--with-python` for Python/venv.
- `./scripts/install_nvidia_docker.sh` — NVIDIA Container Toolkit for Docker GPU
- `./scripts/install_requirements.sh` — install Python deps into venv (called by `make venv`)
- `./scripts/ensure_venv_pip.sh` — ensure pip in venv (called by `make venv`)

## Pre-download weights for offline use
```bash
python scripts/prepare_models.py
DOWNLOAD_DINO=true DINO_MODEL=dinov2_vitb14 python scripts/prepare_models.py
```

## Sample API flow (index + query)

Scripts use `API_URL` (default `http://localhost:8000`). When `API_KEY` is set, add `-H "X-API-Key: $API_KEY"` to curl calls.
```bash
./scripts/sample_requests.sh /path/to/video.mp4 /path/to/image.jpg
```

## Batch index a directory
```bash
./scripts/index_dir.sh /path/to/video_dir true
```

## Index a URL
```bash
./scripts/index_url.sh https://example.com/video.mp4 true
```

## Watch a job
```bash
./scripts/job_watch.sh <job_id>
```

## Precheck (avoid double load)
```bash
./scripts/precheck.sh file /path/to/video.mp4
./scripts/precheck.sh path /path/to/video.mp4
./scripts/precheck.sh url https://example.com/video.mp4
```

## Precheck directory (optionally enqueue new)
```bash
./scripts/precheck_dir.sh /path/to/video_dir
./scripts/precheck_dir.sh /path/to/video_dir true true
```

## Clean frames/tiles cache
```bash
./scripts/clean_data.sh ./data
```

## Reset Qdrant collection
```bash
./scripts/reset_qdrant.sh
```

## List processed registry
```bash
python scripts/list_processed.py
```

## Hash a video
```bash
python scripts/hash_video.py /path/to/video.mp4
```

## Test CLI (end-to-end)
```bash
./scripts/test_cli.sh
```
Requires `make up` running. Indexes test assets and runs text/image queries.

---
[← UI](ui.md) | [Configuration →](configuration.md)
