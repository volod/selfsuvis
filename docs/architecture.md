# Architecture

## Repo Structure
```
app/          FastAPI service (API, auth, rate limiting, search)
worker/       background worker (polls jobs.db, runs VideoIndexer)
models/       embedding models (OpenCLIP, DINO)
pipeline/     ffmpeg, segmentation, tiling, heuristics, qdrant, agentic
ui/           Streamlit frontend
docker/       Dockerfiles and compose files
scripts/      shell/Python helpers (install, precheck, index, etc.)
tests/        unit tests (tests/unit/) and integration (tests/test_api.py)
docs/         documentation
```

## Services (Docker)
- **qdrant** — vector DB (port 6333), storage in `./data/qdrant`
- **api** — FastAPI (port 8000), GPU for embeddings
- **worker** — polls job queue, indexes videos, GPU
- **ui** — Streamlit (port 8501)

All services run as the current host user; `data/` and `cache/` are writable by you.

## Indexing Flow
1. Decode video to frames (ffmpeg)
2. Adaptive sampling and stabilization-aware change detection
3. Segment/keyframe selection
4. Full-frame embedding
5. Tile extraction + quality filters + dedup
6. Upsert to Qdrant with payloads

## Retrieval Flow
- Text → OpenCLIP text embedding → Qdrant search (clip)
- Image → OpenCLIP image embedding → Qdrant search (clip)
- Optional DINO image embedding for rerank/image-only search

---
[← Configuration](configuration.md) | [Examples →](examples.md)
