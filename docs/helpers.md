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

## Reset Qdrant collection
```bash
./scripts/reset_qdrant.sh
```

---
[← UI](ui.md) | [Configuration →](configuration.md)
