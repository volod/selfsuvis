# Configuration

Defaults live in `env/dev.env`, `env/test.env`, and `env/prod.env`. Set `APP_ENV=dev|test|prod` to pick one, then override individual values with environment variables. The authoritative source is [`pipeline/core/config.py`](/home/vola/src/selfsuvis/pipeline/core/config.py).

## Core storage and paths

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `./data` | Root for videos, frames, tiles, reports, maps, checkpoints, and edge assets |
| `VIDEOS_DIR` | `./data/videos` | Stored indexed video copies |
| `FRAMES_DIR` | `./data/frames` | Extracted frames |
| `TILES_DIR` | `./data/tiles` | Extracted tiles |
| `REPORTS_DIR` | `./data/reports` | Mission report output |
| `MAPS_DIR` | `./data/maps` | SfM, splat, and semantic graph outputs |
| `DATABASE_URL` | env-specific | PostgreSQL DSN |
| `QDRANT_HOST` / `QDRANT_PORT` | `qdrant` / `6333` | Qdrant connection |
| `QDRANT_COLLECTION` | `video_semantic` | Main vector collection |

Initialize PostgreSQL after first startup:

```bash
python scripts/migrate_postgres.py
```

## Retrieval and indexing

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `openclip` | `openclip`, `dinov2`, `dinov3`, or `gemma` |
| `OPENCLIP_MODEL` | `ViT-B-16` | Text/image retrieval backbone |
| `OPENCLIP_PRETRAINED` | `openai` | OpenCLIP weights tag |
| `DEVICE` | `auto` | Local inference device |
| `USE_FP16` | `true` | Half precision on CUDA |
| `SAMPLE_FPS_BASE` / `MIN` / `MAX` | `2 / 0.5 / 5` | Adaptive keyframe sampling |
| `HIST_THRESH` | `0.25` | Histogram/SSIM keep threshold |
| `EMBED_DRIFT_THRESH` | `0.15` | Embedding drift keep threshold |
| `MAX_GAP_SEC` | `10` | Force-keep interval |
| `TILE_SIZE` / `STRIDE` | `384 / 256` | Tile extraction geometry |
| `MAX_TILES_PER_SEGMENT` | `200` | Hard tile cap |
| `DEDUP_COS_SIM_THRESH` | `0.95` | Tile dedup cosine threshold |
| `PHASH_HAMMING_MAX` | `6` | Tile dedup perceptual-hash threshold |

## Captioning and multimodal enrichments

| Variable | Default | Purpose |
|---|---|---|
| `FLORENCE_BATCH_SIZE` | `16` | Local Florence batch size |
| `FLORENCE_API_URL` | empty | Use remote Florence endpoint instead of local model |
| `FLORENCE_MODEL` | `microsoft/Florence-2-large` | Remote Florence model ID |
| `QWEN_API_URL` | empty | Enables structured scene extraction sidecar |
| `QWEN_BACKEND` | `vllm` | `vllm` or `ollama` |
| `QWEN_MODEL` | backend-dependent | Qwen sidecar model |
| `GEMMA_MODEL_ID` | `google/gemma-3-4b-it` | Local Gemma embedder |
| `GEMMA_API_URL` | empty | Gemma sidecar endpoint |
| `GEMMA_API_MODEL` | backend-dependent | Gemma sidecar model |
| `REASONING_API_URL` | Gemma default | Final demo reasoning endpoint |
| `ASR_ENABLED` / `ASR_MODEL` | `false` / `auto` | Whisper transcription |
| `OCR_ENABLED` / `OCR_MODEL` | `false` / `auto` | OCR extraction |
| `DEPTH_ENABLED` / `DEPTH_MODEL` | `false` / `auto` | Depth estimation |
| `DETECTION_ENABLED` / `DETECTION_MODEL` | `false` / `auto` | HF detection stage |
| `YOLO_ENABLED` / `YOLO_MODEL` | `true` / `yolo11l` | YOLO detection path |
| `YOLO_SSG_ENABLED` | `true` | Build YOLO semantic scene graphs from 3D frame anchors |
| `YOLO_SSG_MIN_OBSERVATIONS` | `1` | Minimum observations before a semantic node is kept |
| `YOLO_SSG_CLUSTER_RADIUS_METERS` | `12` | Merge radius for production ENU anchors |
| `YOLO_SSG_NEAR_EDGE_RADIUS_METERS` | `20` | `near` edge radius for production ENU anchors |
| `YOLO_SSG_CLUSTER_RADIUS_PCA` | `0.85` | Merge radius for demo PCA/SfM anchors |
| `YOLO_SSG_NEAR_EDGE_RADIUS_PCA` | `1.5` | `near` edge radius for demo PCA/SfM anchors |
| `SAM_ENABLED` / `SAM_MODEL` | `true` / `auto` | SAM mask refinement |
| `WORLD_MODEL_ENABLED` / `WORLD_MODEL` | `false` / `nvidia/Cosmos-1.0-Autoregressive-4B` | Clip-level world-model embeddings |

## Mapping and spatial queries

| Variable | Default | Purpose |
|---|---|---|
| `SFM_FPS` | `2` | Dense frame extraction for SfM |
| `PYCOLMAP_CAMERA_MODEL` | `SIMPLE_RADIAL` | Camera model for pycolmap |
| `NERFSTUDIO_API_URL` | `http://nerfstudio:8000` | Nerfstudio wrapper |
| `MAPPER_API_URL` | `http://mapper:8000` | ICP/mapper service |
| `GPS_SIDECAR_PATH` | empty | Override GPS sidecar autodetection |
| `GPS_FILTER_2D` | `false` | Use 2D Qdrant GPS filter instead of 1D+Python fallback |
| `STATIC_SERVER_URL` | `http://localhost:8080` | Static map hosting |
| `SUPERSPLAT_SERVER_URL` | `http://localhost:8090` | Viewer base URL |

## Active learning and training

| Variable | Default | Purpose |
|---|---|---|
| `AL_TAG_K` | `50` | Top-K frames marked `needs_annotation` |
| `KMEANS_BATCH_THRESHOLD` | `25000` | Switch to MiniBatchKMeans above this size |
| `CHANGE_DETECTION_THRESHOLD_CLIP` | `0.35` | CLIP change threshold |
| `CHANGE_DETECTION_THRESHOLD_DINO` | `0.25` | DINO change threshold |
| `DINO_CHECKPOINT` | empty | Override pretrained DINO weights |
| `SSL_CHECKPOINT_DIR` | `./data/checkpoints` | Self-supervised checkpoint output |
| `SUP_CHECKPOINT_DIR` | `./data/checkpoints/supervised` | Supervised checkpoint output |
| `SUP_AUTO_TRIGGER` | `true` | Allow CVAT webhook to enqueue finetune jobs |
| `MIN_ANNOTATED_FRAMES` | `50` | Minimum annotations before training starts |
| `MIN_NEW_ANNOTATED_SINCE_RETRAIN` | `100` | New-label delta required for retraining |
| `MODEL_VERSION_ID` | `base` | Provenance tag stored with frame payloads |

## Security and request limits

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | empty | API auth key when set |
| `ALLOWED_INDEX_PATHS` | empty | Enables path-based indexing when non-empty |
| `MAX_UPLOAD_BYTES` | `2 GiB` | Upload limit |
| `MAX_DOWNLOAD_BYTES` | `2 GiB` | URL download limit |
| `PRECHECK_URL_TIMEOUT` | `20` | URL precheck timeout in seconds |
| `MAX_REDIRECTS` | `5` | URL download redirect cap |
| `ALLOW_PRIVATE_URLS` | `false` | Private-network URL access |
| `RATE_LIMIT_PER_MIN` | `120` | Request rate limit |
| `RATE_LIMIT_BURST` | `60` | Token-bucket burst |
| `MAX_DIR_FILES` | `5000` | Directory scan file limit |
| `MAX_DIR_BYTES` | `50 GiB` | Directory scan byte limit |
| `MAX_DIR_DEPTH` | `10` | Directory recursion limit |

## Notes

- If `ALLOWED_INDEX_PATHS` is empty, path-based indexing endpoints are disabled by design.
- The UI forwards `API_KEY` automatically when it is set in the UI container environment.
- Qdrant IDs and dedup hashes are SHA-256 based; changing embedding strategy may require a wipe and re-index.
- Production indexing writes mission-scoped semantic graph JSON to `MAPS_DIR/<mission_id>/semantic_environment_graph.json`.
- Demo runs write semantic graph artifacts under `3d_map/semantic_environment_graph.{json,md}`.
- For a full variable list, use [`pipeline/core/config.py`](/home/vola/src/selfsuvis/pipeline/core/config.py) as the source of truth.

---
[← Helpers](helpers.md) | [Architecture →](architecture.md)
