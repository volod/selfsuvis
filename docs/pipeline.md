# Pipeline: Step-by-Step Guide

How the video indexing pipeline works, how to trigger it, and how to run each stage in isolation.

## Overview

Two separate frame extraction passes per mission; SfM must complete before 3DGS starts.

```
Video file / URL / RTSP (MediaMTX)
      │
      ├─► GPS extraction          pipeline/gps_extractor.py
      │   (ffprobe → SRT → GPMF → null fallback)
      │
      ├─► Pass A: Dense frames (SFM_FPS=2fps)
      │      │
      │      ▼
      │   SfM (pycolmap)          pipeline/sfm.py   CPU ~5min/1000 frames
      │      │ pose_status=success
      │      ▼
      │   3DGS (nerfstudio)       pipeline/mapper.py  GPU ~10min (separate container)
      │   → maps/{mission_id}/splat.ply
      │
      └─► Pass B: Sparse keyframes (adaptive)
             │
             ▼
          Florence-2 caption      pipeline/florence_model.py
             │
             ▼
          OpenCLIP + DINOv3 embed models/openclip_model.py, models/dino_model.py
             │
             ▼
          Tile extract + filter + dedup  (unchanged from v0)
             │
             ▼
          Qdrant upsert           pipeline/qdrant_utils.py
          (gps_json, mission_id, pose_json payloads included)
             │
             ▼
          Active learning         pipeline/active_learning.py
          al_score = 0.6×DINOv3_dist + 0.4×(1−caption_confidence)
          al_tag: needs_annotation > novel > none
             │
             ▼
          Mission report          pipeline/report_generator.py → reports/{id}/summary.html
             │
             ▼
          Change detection        pipeline/change_detection.py → change_detections table
```

### Legacy search-only overview (steps 1–8, unchanged)

```
Video file / URL
      │
      ▼
1. Frame extraction      ffmpeg decodes at SAMPLE_FPS_MAX fps → JPEG files
      │
      ▼
2. Adaptive sampling     motion/quality/SSIM check → keep or skip
      │
      ▼
3. Full-frame embedding  OpenCLIP → CLIP vector (+ optional DINOv3)
      │
      ▼
4. Segment upsert        Qdrant "frame" points
      │
      ▼
5. Tile extraction       sliding window crops (TILE_SIZE × TILE_SIZE, step STRIDE)
      │
      ▼
6. Tile quality filters  blur / sky / entropy / edge density
      │
      ▼
7. Tile deduplication    perceptual hash LRU + cell-edge score + cosine similarity
      │
      ▼
8. Tile embedding        CLIP embed → Qdrant "tile" points
```

---

## Running the full pipeline

### 1. Start the stack

```bash
make up
```

Starts Qdrant, API (port 8000), worker, and UI (port 8501). The worker polls `jobs.db` and processes jobs automatically.

### 2. Index a video

**Upload a local file:**
```bash
curl -s \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/video.mp4" \
  -F "enable_tiles=true" \
  http://localhost:8000/index/video | python -m json.tool
```

Response:
```json
{
  "video_id": "a3f8c1d2e4b56789...",
  "job_id":   "b1c2d3e4f5a67890..."
}
```

**Index from URL:**
```bash
./scripts/index_url.sh https://example.com/clip.mp4
# or manually:
curl -s \
  -H "X-API-Key: $API_KEY" \
  -F "url=https://example.com/clip.mp4" \
  -F "enable_tiles=true" \
  http://localhost:8000/index/url | python -m json.tool
```

**Index a local path** (requires `ALLOWED_INDEX_PATHS` to include the directory):
```bash
curl -s \
  -H "X-API-Key: $API_KEY" \
  -F "path=/data/videos/clip.mp4" \
  http://localhost:8000/index/video | python -m json.tool
```

**Index an entire directory:**
```bash
./scripts/index_dir.sh /data/videos true   # args: path enable_tiles
# or manually:
curl -s \
  -H "X-API-Key: $API_KEY" \
  -F "path=/data/videos" \
  -F "enable_tiles=true" \
  http://localhost:8000/index/dir | python -m json.tool
```

### 3. Watch job progress

```bash
./scripts/job_watch.sh <job_id>
# or manually:
watch -n2 "curl -s -H 'X-API-Key: $API_KEY' http://localhost:8000/jobs/<job_id> | python -m json.tool"
```

Job response while running:
```json
{
  "id": "b1c2d3e4f5a67890",
  "status": "running",
  "progress": {
    "frames_processed": 42,
    "segments_found": 18,
    "frames_indexed": 18,
    "tiles_indexed": 312,
    "frame_fps": 1.4,
    "embed_fps": 1.2
  }
}
```

Job response when done:
```json
{
  "id": "b1c2d3e4f5a67890",
  "status": "finished",
  "progress": {
    "segments": 18,
    "frames": 18,
    "tiles": 312
  }
}
```

### 4. Search

**Text query:**
```bash
curl -s \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "person walking on street"}' \
  "http://localhost:8000/query/text?top_k=5&search_type=both" \
  | python -m json.tool
```

**Image query:**
```bash
curl -s \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/query.jpg" \
  -F "top_k=5" \
  -F "search_type=both" \
  -F "enable_rerank=true" \
  http://localhost:8000/query/image | python -m json.tool
```

`search_type`: `frame` | `tile` | `both`
`enable_rerank`: blends CLIP score (70%) with DINOv2 score (30%) when `MODEL_NAME=dinov2`.

Result:
```json
{
  "results": [
    {
      "video_id": "a3f8c1d2...",
      "segment_id": 12,
      "t_sec": 24.6,
      "score": 0.87,
      "type": "tile",
      "frame_path": "data/frames/a3f8c1d2.../frame_0000000049.jpg",
      "tile_path":  "data/tiles/a3f8c1d2.../12/tile_24600_256_128.jpg",
      "bbox": [256, 128, 384, 384]
    }
  ]
}
```

### 5. Pre-check for duplicates before indexing

```bash
# Single file
./scripts/precheck.sh /path/to/video.mp4
# or:
curl -s \
  -H "X-API-Key: $API_KEY" \
  -F "path=/path/to/video.mp4" \
  http://localhost:8000/index/precheck | python -m json.tool

# Directory scan (dry run)
./scripts/precheck_dir.sh /data/videos false true   # args: path enqueue enable_tiles
# Directory scan + enqueue new videos immediately
./scripts/precheck_dir.sh /data/videos true true
```

### 6. Inspect processed videos

```bash
# Show the last 200 indexed files from the dedup registry
PROCESSED_DB=./data/processed.db .venv/bin/python scripts/list_processed.py
```

---

## End-to-end CLI test

The `test_cli.sh` script indexes two bundled test videos and runs text + image queries:

```bash
make up
./scripts/test_cli.sh
# with a real video directory:
INDEX_DIR_PATH=/data/videos ./scripts/test_cli.sh
```

---

## Running the pipeline directly in Python

No Docker needed — useful for development and debugging. Requires Qdrant reachable at `localhost:6333`.

```bash
# Start Qdrant only
docker run -p 6333:6333 -v $(pwd)/data/qdrant:/qdrant/storage qdrant/qdrant:v1.7.4
```

```python
from pipeline.indexer import VideoIndexer

indexer = VideoIndexer(enable_tiles=True)

result = indexer.index_video(
    video_path="/path/to/video.mp4",
    video_id="my_video_01",
    progress_cb=lambda p: print(p),
)
print(result)
# {'segments': 24, 'frames': 24, 'tiles': 418}
```

Environment is loaded from `env/dev.env` by default (`QDRANT_HOST=localhost`, `SAMPLE_FPS_MAX=5`, etc.). Override any setting with environment variables:

```bash
SAMPLE_FPS_MAX=1 TILE_SIZE=224 STRIDE=112 .venv/bin/python - <<'PY'
from pipeline.indexer import VideoIndexer
VideoIndexer().index_video("video.mp4", "dev_test")
PY
```

---

## Pipeline stage details

### Step 1 — Frame extraction (`pipeline/ffmpeg_utils.py`)

ffmpeg decodes the video at `SAMPLE_FPS_MAX` (default **5 fps**) and writes JPEGs to `data/frames/<video_id>/frame_NNNNNNNNNN.jpg`.

```bash
# Equivalent manual ffmpeg command
ffmpeg -i video.mp4 -vf fps=5 -q:v 2 data/frames/my_id/frame_%010d.jpg
```

Key config:
| Variable | Default | Effect |
|---|---|---|
| `SAMPLE_FPS_MAX` | `5` | Decode rate; more = finer granularity, slower |
| `FFMPEG_TIMEOUT_SEC` | `3600` | Hard timeout per video |

---

### Step 2 — Adaptive sampling (`pipeline/indexer.py → _process_frame`)

Each extracted frame is tested against the last kept frame. A frame is kept if **any** condition holds:

| Signal | Threshold | Config |
|---|---|---|
| Histogram difference | > 0.25 | `HIST_THRESH` |
| SSIM difference (1−SSIM) | > 0.25 | `HIST_THRESH` |
| CLIP embedding drift (1−cosine) | > 0.15 | `EMBED_DRIFT_THRESH` |
| Time since last kept frame | > 10 s | `MAX_GAP_SEC` |

The inter-frame step adapts to motion:
- motion < `MOTION_LOW` (0.02) → step doubles (slow scene, sample less)
- motion > `MOTION_HIGH` (0.08) → step halves (fast scene, sample more)

Frames that fail `frame_quality_ok()` are rejected before the keep decision:
- Laplacian blur variance < `BLUR_LAPL_VAR_MIN_FRAME` (80)
- Mean intensity outside [`MEAN_INTENSITY_MIN`, `MEAN_INTENSITY_MAX`] = [20, 235]

Optional stabilisation: phase-correlation shift alignment before diffing, controlled by `STAB_ENABLE=true` (default on).

---

### Step 3 — Full-frame embedding (`models/openclip_model.py`)

Each kept frame is embedded with OpenCLIP (`ViT-B-16 / openai` by default).

Key config:
| Variable | Default |
|---|---|
| `OPENCLIP_MODEL` | `ViT-B-16` |
| `OPENCLIP_PRETRAINED` | `openai` |
| `DEVICE` | `auto` (prefers CUDA) |
| `USE_FP16` | `true` |

Use DINOv2 in addition to CLIP:
```bash
MODEL_NAME=dinov2 make up
```

---

### Step 4 — Segment upsert (`pipeline/qdrant_utils.py`)

Each kept frame becomes a Qdrant point with:
- **stable ID** = SHA-256(`video_id` + `segment_id` + `timestamp_ms` + `"frame"`)
- **vectors**: `clip` (always), `dino` (if `MODEL_NAME=dinov2/3`)
- **payload**: `type="frame"`, `video_id`, `segment_id`, `t_sec`, `frame_path`

Points are upserted in batches of 128.

---

### Step 5 — Tile extraction (`pipeline/indexer.py → _index_tiles`)

A sliding window of `TILE_SIZE × TILE_SIZE` pixels steps across each kept frame.

| Variable | Default | Effect |
|---|---|---|
| `TILE_SIZE` | `384` | Crop size in pixels |
| `STRIDE` | `256` | Step between crop origins |
| `MAX_TILES_PER_SEGMENT` | `200` | Hard cap per frame |

A 1920×1080 frame at defaults produces up to ~42 candidate tiles before filtering.

Tiles are only extracted when `enable_tiles=true` **or** when CLIP embedding drift > `TILE_INDEX_IF_EMBED_DRIFT_GT` (0.10).

---

### Step 6 — Tile quality filters (`pipeline/heuristics.py → tile_quality_ok`)

Each candidate tile is rejected if it fails any of:

| Filter | Threshold | Config | Rejects |
|---|---|---|---|
| Laplacian blur | < 60 | `BLUR_LAPL_VAR_MIN_TILE` | Blurry tiles |
| Mean intensity | < 20 or > 235 | `MEAN_INTENSITY_MIN/MAX` | Black / overexposed |
| Pixel std deviation | < 12 | `TILE_STD_MIN` | Flat tiles |
| Shannon entropy | < 3.5 | `TILE_ENTROPY_MIN` | Low-information tiles |
| Sky/haze | blue_ratio > 0.35 and edge_density < 0.02 | `SKY_BLUE_RATIO_MAX`, `EDGE_DENSITY_MIN` | Sky, haze |

---

### Step 7 — Tile deduplication

Three independent dedup passes; a tile is **skipped** if any fires:

**Pass 1 — Perceptual hash** (`pipeline/dedup.py`)
Difference hash (dhash) checked against an LRU of recent hashes.
Threshold: Hamming distance ≤ `PHASH_HAMMING_MAX` (6). LRU size: `PHASH_LRU_SIZE` (50 000).

**Pass 2 — Cell edge score** (`pipeline/indexer.py`)
The frame is divided into `CELL_SIZE × CELL_SIZE` (256 px) cells. A tile is skipped if the same cell already has a higher-scoring tile within the last `CELL_WINDOW_SEC` (5 s). Score = `edge_density × 1000 + pixel_std`.

**Pass 3 — Cosine similarity** (`pipeline/recent_index.py`)
CLIP vector compared against a rolling window of recent tile embeddings.
Skipped if similarity > `DEDUP_COS_SIM_THRESH` (0.95). Window: `DEDUP_RECENT_TILES` (200 000) tiles, TTL `DEDUP_TTL_SEC` (120 s).

Tiles that survive all three passes are written to `data/tiles/<video_id>/<segment_id>/tile_<ms>_<x>_<y>.jpg` and their embeddings are committed to the dedup indices.

---

### Step 8 — Tile upsert

Same as Step 4 for tiles. Stable ID = SHA-256(`video_id` + `segment_id` + `timestamp_ms` + `"tile"` + `x` + `y`).

Payload fields: `type="tile"`, `video_id`, `segment_id`, `t_sec`, `frame_path`, `tile_path`, `x`, `y`, `w`, `h`.

---

## Optional: Agentic scene-understanding pass

`pipeline/agentic_system.py` runs a separate multi-agent pipeline for structured scene descriptions. It is **not** part of the default indexing flow.

```python
from pipeline.frame_extractor import extract_frames_adaptive
from pipeline.agentic_system import process_frames

# Extract frames first
frame_records = extract_frames_adaptive(
    "/path/to/video.mp4",
    out_dir="/tmp/frames",
    min_interval_sec=1.0,
    max_gap_sec=10.0,
)

# Run scene understanding
result = process_frames(
    video_name="my_video",
    frame_records=frame_records,
    output_dir="/tmp/output",
    model_type="openclip_only",   # or "openclip_sam" if SAM checkpoint available
    verbose=True,
)
print(result)
# {'jsonl_path': '/tmp/output/my_video.jsonl', 'ontology_path': '/tmp/output/my_video.ontology.json'}
```

To enable SAM segmentation, download a SAM checkpoint and set:
```bash
SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth SAM_MODEL_TYPE=vit_h .venv/bin/python ...
```

Output files:
- `<video_name>.jsonl` — one JSON record per frame with `description`, `segments`, `entities`, `tracks`
- `<video_name>.ontology.json` — entity summary across all frames

---

## New pipeline stages (v1)

### GPS extraction (`pipeline/gps_extractor.py`)

Auto-detects GPS telemetry from drone video in priority order:
1. ffprobe GPS atoms (MP4 location tag, some DJI formats)
2. SRT sidecar file (same filename, `.srt` extension — standard DJI format)
3. GPMF binary format (GoPro Max, GoPro Hero)
4. Null fallback with warning

Output: `List[Optional[dict]]` with `{lat, lon, alt, timestamp_ms}` per frame, synced to frame timestamps from ffprobe. Written to `frames.gps_json` per keyframe.

Override auto-detection: `GPS_SIDECAR_PATH=/path/to/file.srt`

---

### Structure-from-Motion (`pipeline/sfm.py`)

Runs pycolmap on the **dense frame set** (extracted at `SFM_FPS=2fps`). Outputs per-frame `pose_json` (`{rotation_matrix, translation, intrinsics}`). Sets `missions.pose_status` to `success` or `failed`.

Camera model: `PYCOLMAP_CAMERA_MODEL` env var (default `SIMPLE_RADIAL`). Set `PINHOLE` or `RADIAL` for known drone hardware (DJI, GoPro) to significantly improve pose accuracy.

Estimated runtime: ~5 minutes per 1000 frames on CPU.

---

### 3DGS reconstruction (`pipeline/mapper.py`)

Runs nerfstudio `splatfacto` in a separate Docker container via HTTP. Triggered only after `pose_status=success`. Sequential: SfM must complete first (v1 design).

Output: `maps/{mission_id}/splat.ply` — viewable via the SuperSplat iframe in Streamlit.

Estimated runtime: ~10 minutes on GPU. Set `NERFSTUDIO_API_URL` to point at the nerfstudio wrapper service.

---

### Florence-2 captioning (`pipeline/florence_model.py`)

Runs `microsoft/Florence-2-large` per keyframe. Outputs `caption` (text) and `caption_confidence` (logit-derived float). Both stored in PostgreSQL `frames` table.

Batch size: `FLORENCE_BATCH_SIZE` (default 16). Auto-fallback to batch=1 on OOM.

---

### Active learning scoring (`pipeline/active_learning.py`)

Computed after embeddings are available:

```
active_learning_score = 0.6 × DINOv3_distance_from_nearest_centroid
                      + 0.4 × (1 − caption_confidence)
```

Cluster centroids: k-means (k=20) over all mission DINOv3 embeddings, updated after each mission, stored in `embedding_clusters` table.

**al_tag assignment** (precedence):
1. `needs_annotation` — top-`AL_TAG_K` (default 50) frames by score
2. `novel` — DINOv3 distance > 0.5 from any centroid
3. `none` — all others

If `MODEL_NAME=openclip` (no DINOv3), score uses CLIP embedding distance only.

---

### Mission summary report (`pipeline/report_generator.py`)

Auto-generated HTML report at end of each mission: `reports/{mission_id}/summary.html`

Contents: mission metadata, top-5 frames by `active_learning_score` (thumbnails + captions), 3D camera path screenshot, uncertainty histogram, `al_tag` distribution.

---

### Multi-mission change detection (`pipeline/change_detection.py`)

Triggered post-pipeline after `pose_status=success`. Finds prior-mission frames in the same GPS area via Qdrant filter, computes embedding distance, flags pairs above `CHANGE_DETECTION_THRESHOLD`.

Default filter strategy (`GPS_FILTER_2D=false`): lat-only Qdrant filter → Python post-filter on lon. Enable `GPS_FILTER_2D=true` only after validating 2D Qdrant payload index performance (see TODOS.md P1).

Results stored in `change_detections` table. Accessible via `GET /missions/{id}/changes`.

---

## Key configuration reference

All settings live in `pipeline/config.py` and are read from environment variables, with defaults from `env/dev.env`.

| Variable | Default | Description |
|---|---|---|
| `SAMPLE_FPS_MAX` | `5` | ffmpeg decode rate (max) |
| `SAMPLE_FPS_BASE` | `2` | Starting adaptive rate for search keyframes |
| `SAMPLE_FPS_MIN` | `0.5` | Minimum adaptive rate |
| `SFM_FPS` | `2` | Dense frame rate for pycolmap SfM |
| `PYCOLMAP_CAMERA_MODEL` | `SIMPLE_RADIAL` | pycolmap camera model |
| `FLORENCE_BATCH_SIZE` | `16` | Florence-2 batch size |
| `GPS_FILTER_2D` | `false` | Enable 2D Qdrant GPS bbox filter |
| `AL_TAG_K` | `50` | Top-K frames for needs_annotation |
| `CHANGE_DETECTION_THRESHOLD` | `0.35` | Cosine threshold for change detection |
| `HIST_THRESH` | `0.25` | Histogram / SSIM keep threshold |
| `EMBED_DRIFT_THRESH` | `0.15` | CLIP drift keep threshold |
| `MAX_GAP_SEC` | `10` | Force-keep interval |
| `TILE_SIZE` | `384` | Tile crop size (px) |
| `STRIDE` | `256` | Tile step (px) |
| `MAX_TILES_PER_SEGMENT` | `200` | Tile cap per frame |
| `DEDUP_COS_SIM_THRESH` | `0.95` | Cosine similarity dedup |
| `PHASH_HAMMING_MAX` | `6` | Perceptual hash dedup |
| `MODEL_NAME` | `openclip` | `openclip` \| `dinov2` \| `dinov3` |
| `OPENCLIP_MODEL` | `ViT-B-16` | OpenCLIP model name |
| `OPENCLIP_PRETRAINED` | `openai` | OpenCLIP pretrained weights |
| `DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` |
| `ALLOWED_INDEX_PATHS` | _(empty)_ | Comma-separated allowed dirs for path-based indexing |

See `docs/configuration.md` for the full list.

---

[← Developer Guide](develop.md) | [Configuration →](configuration.md)
