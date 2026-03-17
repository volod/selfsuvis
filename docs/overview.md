# Overview

On-prem, single-machine POC for video semantic search with **image-to-video** and **text-to-video** retrieval. Designed for an NVIDIA GPU (e.g. RTX 3060 12GB) and offline operation after dependencies are installed.

**Image-to-video**: Given an image (screenshot, photo), find video segments that look similar. Video frames and the query image are embedded with the same OpenCLIP encoder; nearest-neighbor search returns the most similar stored frames.

**Text-to-video**: Given a text query (e.g. "green field"), find matching segments. OpenCLIP maps text and images into a shared embedding space, so text queries search the same frame vectors.

## Features
- Video ingestion + segmentation using change detection (histogram + semantic drift + max gap)
- Adaptive sampling, stabilization-aware motion scoring
- Full-frame and tile embeddings with dedup heuristics
- OpenCLIP shared embedding space for text + image queries
- Optional DINOv2/DINOv3 embeddings for image-only rerank or search
- Vector search via Qdrant (named vectors: clip, optional dino)
- FastAPI + Streamlit UI
- Docker stack runs as current user; data and cache owned by you

---
[← README](../README.md) | [Setup →](setup.md)
