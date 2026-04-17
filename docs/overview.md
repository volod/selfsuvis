# Overview

`selfsuvis` is a video indexing and retrieval stack for outdoor robotics workflows. It ingests mission video, extracts searchable keyframes and tiles, stores embeddings in Qdrant, stores mission/frame state in PostgreSQL, and exposes search, admin, robot-query, and annotation endpoints through FastAPI.

## What is implemented

- Video ingestion from upload, allowed local paths, directories, and URLs
- Keyframe extraction, tile extraction, deduplication, and vector indexing
- OpenCLIP retrieval with optional DINO vectors
- Florence captioning during indexing
- Optional multimodal enrichments: Gemma, Qwen, ASR, OCR, depth, detection, YOLO, SAM, world-model steps
- Multi-modal sensor fusion: thermal/IR, multispectral, LiDAR, radar, GNSS-R, atmospheric, gas, radiation, acoustic, and event-camera sidecars aligned to video frame timestamps and written to `frame_facts_json["sensor_fusion"]`
- YOLO SSG semantic environment graphs built from detection output plus ENU/SfM/PCA frame anchors
- pycolmap pose estimation and optional nerfstudio/mapper 3D outputs
- Mission reports, change detection, robot pose queries, and scene queries
- Active-learning tagging plus CVAT webhook-driven supervised fine-tune triggers

## Main runtime pieces

- `api`: FastAPI routes for indexing, querying, admin, CVAT, and health
- `worker`: background job processor for indexing, re-embedding, and model operations
- `ui`: Streamlit UI with indexing, image/text search, admin stats, and 3DGS viewer
- `postgres`: primary SQL store for jobs, missions, frames, caption/facts metadata, and automation state
- `qdrant`: vector store for frame and tile retrieval

## Primary workflows

- Operator workflow: start the stack, run `src/selfsuvis/scripts/migrate_postgres.py`, index missions, then query by text, image, scene facts, or robot pose
- Model workflow: prefetch models with `src/selfsuvis/scripts/prepare_models.py`, fine-tune DINO with `src/selfsuvis/scripts/finetune_dino.py` or `src/selfsuvis/scripts/supervised_finetune_dino.py`, export ONNX with `src/selfsuvis/scripts/export_onnx.py`, build an edge gallery with `src/selfsuvis/scripts/build_gallery.py`
- Annotation workflow: pull candidate frames from `/admin/cvat/frames`, register task mappings, and let the CVAT webhook mark frames annotated and optionally enqueue fine-tuning

---
[← README](../README.md) | [Setup →](setup.md)
