# Performance Tuning

## GPU memory
- Use `USE_FP16=true` (default)
- Reduce OpenCLIP model size (e.g., `ViT-B-16`)
- Reduce batch sizes if you see OOM

## Throughput knobs
- `SAMPLE_FPS_BASE`, `SAMPLE_FPS_MAX` lower for fewer frames
- `MAX_TILES_PER_SEGMENT` lower for fewer tiles
- `TILE_SIZE` bigger for fewer tiles
- `STRIDE` bigger to reduce overlap

## Quality vs speed
- Lower `EMBED_DRIFT_THRESH` to keep more segments
- Increase `HIST_THRESH` to keep fewer segments

## Tuning recipes

### Fast, lightweight indexing (large datasets)
Goal: maximize throughput, lower storage.
```
SAMPLE_FPS_BASE=1
SAMPLE_FPS_MAX=2
MAX_GAP_SEC=15
HIST_THRESH=0.35
EMBED_DRIFT_THRESH=0.25
MAX_TILES_PER_SEGMENT=80
STRIDE=320
TILE_SIZE=384
DEDUP_COS_SIM_THRESH=0.97
```

### Balanced (default-ish)
Goal: good coverage without exploding tiles.
```
SAMPLE_FPS_BASE=2
SAMPLE_FPS_MAX=5
MAX_GAP_SEC=10
HIST_THRESH=0.25
EMBED_DRIFT_THRESH=0.15
MAX_TILES_PER_SEGMENT=200
STRIDE=256
TILE_SIZE=384
DEDUP_COS_SIM_THRESH=0.95
```

### High recall (small dataset, best match quality)
Goal: more segments and tiles for better recall.
```
SAMPLE_FPS_BASE=3
SAMPLE_FPS_MAX=6
MAX_GAP_SEC=6
HIST_THRESH=0.18
EMBED_DRIFT_THRESH=0.10
MAX_TILES_PER_SEGMENT=300
STRIDE=192
TILE_SIZE=320
DEDUP_COS_SIM_THRESH=0.93
```

### Strong dedup for repetitive flight video
Goal: reduce near-duplicate tiles in long smooth flights.
```
DEDUP_COS_SIM_THRESH=0.97
PHASH_HAMMING_MAX=4
CELL_WINDOW_SEC=8
```
