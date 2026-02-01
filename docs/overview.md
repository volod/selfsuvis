# Overview

On‑prem, single‑machine POC for video semantic search with **image‑to‑video** and **text‑to‑video** retrieval. Designed for an NVIDIA RTX 3060 (12GB) and offline operation after dependencies are installed.

## Features
- Video ingestion + segmentation using change detection (histogram + semantic drift + max gap)
- Adaptive sampling, stabilization‑aware motion scoring
- Full‑frame and tile (field‑square) embeddings with dedup heuristics
- OpenCLIP shared embedding space for text + image queries
- Optional DINOv2 embeddings for image‑only rerank or search
- Vector search via Qdrant (named vectors)
- FastAPI + Streamlit UI
