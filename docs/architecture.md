# Architecture

## Repo Structure
```
app/          FastAPI service
worker/       indexing worker process
models/       model loading/inference utils
pipeline/     ffmpeg, segmentation, tiling, heuristics, qdrant
ui/           Streamlit UI
scripts/      helper scripts
docs/         documentation
```

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
