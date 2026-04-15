# Monocular Depth Estimation Runbook

> Covers: enabling depth estimation, model selection, output format,
> and GPU-aware auto-selection.

---

## 1. Architecture overview

```
VideoIndexer
  └─ Depth pass (per frame)
       └─ DepthModel.estimate_batch()    ← loaded in worker VRAM
            → 5-bucket depth percentiles [p10, p25, p50, p75, p90]
            → frame_facts_json["depth"]
```

Depth is **disabled by default** (`DEPTH_ENABLED=false`). Enable for missions
where relative scene geometry matters (obstacle proximity, terrain slope,
structural distances). The output is compact enough for DB storage — only
5 percentile values per frame, not the full depth map.

---

## 2. Environment variables

| Variable | Default | Description |
|---|---|---|
| `DEPTH_ENABLED` | `false` | Enable depth estimation pass |
| `DEPTH_MODEL` | `auto` | Model ID or `auto` for GPU-aware selection |
| `DEVICE` | `auto` | Device for inference |

---

## 3. Model selection

| Model ID | Params | VRAM | Notes |
|---|---|---|---|
| `depth-anything/Depth-Anything-V2-Small-hf` | 25 M | ~0.05 GB | **Auto default** for low VRAM; fast |
| `depth-anything/Depth-Anything-V2-Base-hf` | 97 M | ~0.2 GB | Good indoor+outdoor balance |
| `depth-anything/Depth-Anything-V2-Large-hf` | 335 M | ~0.7 GB | Best DepthAnything quality |
| `apple/DepthPro-hf` | 1.1 B | ~2.2 GB | Metric depth + focal length; best for precise distances |

**Auto-selection** (`DEPTH_MODEL=auto`): picks the largest model that fits within
available VRAM with a 2 GB safety margin.

---

## 4. Quick start

```bash
# Enable with auto model selection
DEPTH_ENABLED=true python main.py --mode local

# Explicit model
DEPTH_ENABLED=true DEPTH_MODEL=depth-anything/Depth-Anything-V2-Large-hf python main.py --mode local

# Download weights
python scripts/prepare_models.py --depth
```

---

## 5. Interpreting output

```json
{"depth": [0.12, 0.28, 0.51, 0.72, 0.89]}
```

Values are normalised to [0, 1] within each frame (relative depth).
- `p10`: near objects (foreground) depth percentile
- `p50`: median scene depth
- `p90`: far objects (background)

High p10 = something close to the camera. Low p90 = short-range scene (indoor, close-up).

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Depth pass skipped` | `DEPTH_ENABLED=false` | Set `DEPTH_ENABLED=true` |
| All values near 0.5 | Model convergence failure on blank/uniform frame | Expected for featureless frames (sky, water) |
| `CUDA out of memory` | DepthPro at 1.1B too large alongside other models | Use `auto` or select smaller model |
| Very slow: >2s per frame | Running on CPU | Set `DEVICE=cuda` |
| `ModuleNotFoundError: transformers` | transformers not installed | `pip install transformers` |
