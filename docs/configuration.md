# Configuration

Defaults live in `pipeline/config.py` and can be overridden with env vars.

Key variables:
- `MODEL_NAME` = openclip | dinov2 | dinov3
- `OPENCLIP_MODEL`, `OPENCLIP_PRETRAINED`
- `SAMPLE_FPS_BASE`, `SAMPLE_FPS_MIN`, `SAMPLE_FPS_MAX`
- `HIST_THRESH`, `EMBED_DRIFT_THRESH`, `MAX_GAP_SEC`
- `TILE_SIZE`, `STRIDE`
- `DEDUP_COS_SIM_THRESH`, `MAX_TILES_PER_SEGMENT`
- `LOG_LEVEL` = DEBUG | INFO | WARNING | ERROR

Notes:
- Frames are extracted at `SAMPLE_FPS_MAX` with ffmpeg, then adaptive skipping is applied.
- Named vectors in Qdrant: `clip` (OpenCLIP), optional `dino`.
- DINOv3 is optional and may have licensing ambiguity; use at your own risk.
- If you set `MODEL_NAME=dinov2` or `dinov3`, pre-download weights once while online (Torch Hub), then run offline.
- Duplicate videos are avoided using SHA256 hash tracking in `./data/processed.db`.
