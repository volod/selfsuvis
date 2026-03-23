# Overview

**Outdoor autonomy perception stack** — spatial memory engine for robotics. Ingest mission video from drones, rovers, or ground vehicles → extract frames → estimate camera poses (pycolmap SfM) → build dense 3D maps (nerfstudio 3DGS) → embed with OpenCLIP + DINOv3 → caption with Florence-2 → store in PostgreSQL + Qdrant → search by text or image.

Each new mission auto-registers against a persistent global 3D map. The system detects environmental change over time across missions. Robots can query the world model during flight via a REST API.

## Features

**Search and retrieval**
- Text-to-video and image-to-video retrieval (OpenCLIP shared embedding space)
- "Find more like this" — similarity search on DINOv3 embeddings (fallback: CLIP)
- Optional DINOv3 reranking for image queries (70/30 CLIP/DINO blend)

**3D mapping and pose**
- Camera pose estimation via pycolmap Structure-from-Motion (CPU)
- 3D Gaussian Splatting reconstruction via nerfstudio splatfacto (GPU, optional)
- Embedded 3DGS viewer in Streamlit (SuperSplat iframe)

**Multi-mission intelligence**
- Multi-mission change detection using GPS-bbox Qdrant filter + embedding distance
- Persistent global map: Phase 1 GPS point cloud in ENU frame (~50km radius)
- Robot advisory API (`POST /query/pose`) — p99 < 200ms, GPS-based, advisory use only

**Active learning loop**
- Per-frame uncertainty scoring: `0.6×DINOv3_dist + 0.4×(1−caption_confidence)`
- Auto-tags top-K frames as `needs_annotation`; novel frames (DINOv3 dist > 0.5) as `novel`
- Mission summary HTML report auto-generated at end of each mission

**Ingestion**
- Upload local files, index by path/directory, or submit URL
- Video streaming via MediaMTX (RTSP/RTMP/WebRTC/HLS)
- Adaptive sampling + stabilization-aware change detection
- Full-frame and tile embeddings with dedup heuristics

**Infrastructure**
- PostgreSQL 16 as primary SQL store (missions, frames, jobs, active learning, global map)
- Qdrant vector DB (named vectors: `clip`, optional `dino`)
- Async-native worker with `SELECT FOR UPDATE SKIP LOCKED` for safe concurrent job claiming
- Docker stack runs as current user; data and cache owned by you

---
[← README](../README.md) | [Setup →](setup.md)
