# World Model Video Embeddings Runbook

> Covers: enabling temporal embeddings, model selection, clip chunking,
> and similarity-based temporal anomaly detection.

---

## 1. Architecture overview

```
VideoIndexer / step_world_model (step Q)
  └─ World model pass (clip windows)
       └─ WorldModel.encode_clips()    ← loaded in worker VRAM
            frame sequence → clip windows of WORLD_MODEL_CLIP_FRAMES frames
            → 768-dim embedding per clip
            → frame_facts_json["world_model"]["embedding_id"]
            → clip similarity scores vs. mission mean (anomaly score)
```

World model is **disabled by default** (`WORLD_MODEL_ENABLED=false`). Enable for
missions where temporal scene evolution matters: route surveillance, activity
recognition, change point detection.

---

## 2. Environment variables

| Variable | Default | Description |
|---|---|---|
| `WORLD_MODEL_ENABLED` | `false` | Enable world model temporal embedding pass |
| `WORLD_MODEL` | `nvidia/Cosmos-1.0-Autoregressive-4B` | HuggingFace model ID or `auto` |
| `WORLD_MODEL_CLIP_FRAMES` | `8` | Frames aggregated into one clip embedding |
| `WORLD_MODEL_STORE_EMBED` | `false` | Store raw embedding vector in DB (large; default off) |

---

## 3. Model selection

| Model ID | Params | VRAM | Notes |
|---|---|---|---|
| `facebook/timesformer-base-finetuned-k400` | 122 M | ~0.3 GB | Fast; K400 action recognition features |
| `MCG-NJU/videomae-base` | 122 M | ~0.3 GB | Masked autoencoder; strong temporal features |
| `MCG-NJU/videomae-large` | 307 M | ~0.6 GB | Better quality; good default for edge GPUs |
| `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` | 1 B | ~2.0 GB | Strong video-language retrieval |
| `nvidia/Cosmos-1.0-Autoregressive-4B` | 4 B | ~8.0 GB | **Default** — physical world model for robotics |

---

## 4. Quick start

```bash
# Enable with default model
WORLD_MODEL_ENABLED=true python main.py --mode local

# Lightweight model for edge GPU
WORLD_MODEL_ENABLED=true WORLD_MODEL=MCG-NJU/videomae-large python main.py --mode local

# Store embeddings in DB (increases storage significantly)
WORLD_MODEL_ENABLED=true WORLD_MODEL_STORE_EMBED=true python main.py --mode local

# Download weights
python scripts/prepare_models.py --world-model
```

---

## 5. Clip chunking

`WORLD_MODEL_CLIP_FRAMES` controls the temporal window:

| Value | Window | Best for |
|---|---|---|
| 4 | ~2s at 2 FPS | Short events, fast scenes |
| 8 | ~4s at 2 FPS | **Default** — balanced |
| 16 | ~8s at 2 FPS | Slow scene evolution, long-range patterns |
| 32 | ~16s at 2 FPS | Route-level similarity |

A clip with fewer frames than `WORLD_MODEL_CLIP_FRAMES` is padded or skipped
depending on the model's minimum input requirement.

---

## 6. Temporal anomaly scoring

Each clip embedding is compared to the mission mean embedding (cosine distance).
Clips with distance > 1.5× the mission standard deviation are flagged in the report
as potential anomaly moments.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `World model pass skipped` | `WORLD_MODEL_ENABLED=false` | Set `WORLD_MODEL_ENABLED=true` |
| `CUDA out of memory` | Cosmos 4B too large alongside other models | Switch to `MCG-NJU/videomae-large` |
| All clips have identical embeddings | Very static footage (hovering camera) | Expected; cosine similarity will be ~1.0 |
| Very slow: >10s per clip | CPU inference or large model | Set `DEVICE=cuda`; use smaller model |
| `ModuleNotFoundError` | transformers / timm not installed | `pip install transformers timm` |
