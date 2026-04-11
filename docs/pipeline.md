# Pipeline

This page describes the current indexing flow and the local full-analysis flow.

## Production indexing flow

1. A client calls one of the indexing endpoints.
2. The API validates input and writes a job to PostgreSQL.
3. The worker claims the job and runs `pipeline.workflows.indexer.VideoIndexer`.
4. The pipeline:
   - extracts frames
   - performs adaptive keep/skip decisions
   - embeds kept frames
   - optionally extracts and indexes tiles
   - captions frames with Florence and optional sidecar-backed enrichments
   - optionally runs UniDriveVLA expert analysis when `UNIDRIVE_ENABLED=true` and
     `UNIDRIVE_API_URL` is set; stores normalized understanding/perception/planning
     output in `frame_facts_json["unidrive_vla"]`
   - runs YOLO/SAM when enabled and writes a mission-scoped semantic environment graph
   - runs Gemma directed tracking when `RFDETR_ENABLED=true` and `GEMMA_API_URL` is set:
     Gemma analyses sampled frames → SAM segments Gemma-identified objects → RF-DETR
     tracks those objects across the full frame sequence; results stored in
     `frame_facts_json["gemma_tracking"]` per frame
   - writes metadata to PostgreSQL and vectors to Qdrant
5. Optional spatial and reporting stages run after indexing:
   - pycolmap pose estimation
   - nerfstudio/mapper outputs
   - change detection
   - mission reports
   - active-learning tagging

## Useful command-line helpers

```bash
./scripts/precheck.sh path /path/to/video.mp4
./scripts/precheck_dir.sh /path/to/video_dir true true
./scripts/index_url.sh https://example.com/video.mp4 true
./scripts/index_dir.sh /path/to/video_dir true
./scripts/job_watch.sh <job_id>
```

## Typical API flow

Start the stack and initialize PostgreSQL:

```bash
make up
python scripts/migrate_postgres.py
```

Index a file:

```bash
curl -s -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/video.mp4" \
  -F "enable_tiles=true" \
  http://localhost:8000/index/video | python -m json.tool
```

Search by text:

```bash
curl -s -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"green field"}' \
  "http://localhost:8000/query/text?top_k=5&search_type=both" | python -m json.tool
```

Search by image:

```bash
curl -s -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/query.jpg" \
  -F "top_k=5" \
  -F "search_type=both" \
  -F "vector_space=clip" \
  http://localhost:8000/query/image | python -m json.tool
```

## Running the indexer directly in Python

```python
from pipeline.workflows.indexer import VideoIndexer

indexer = VideoIndexer(enable_tiles=True)
result = indexer.index_video("/path/to/video.mp4", "dev_test")
print(result)
```

This path is useful for local debugging when PostgreSQL/Qdrant are already reachable. The returned dict includes a `semantic_graph` summary when YOLO SSG is enabled and a `unidrive_summary` when UniDrive enrichment is enabled.

## Local Full-Analysis Mode

`main.py` now defaults to the local full-analysis and training pipeline (`--mode local`),
implemented by the current CLI in [`pipeline/workflows/cli_parser.py`](/home/vola/src/selfsuvis/pipeline/workflows/cli_parser.py) and the runner modules under [`pipeline/workflows/local`](/home/vola/src/selfsuvis/pipeline/workflows/local).

Common options:

```bash
python main.py
python main.py --mode local --input /path/to/video.mp4
python main.py --mode local --dir /path/to/video_dir --no-qdrant --no-sfm --no-gsplat
python main.py --mode local --qwen-api-url http://localhost:8010/v1
python main.py --mode local --gemma-api-url http://localhost:11434/v1
python main.py --mode local --no-yolo --no-sam
python main.py --mode local --gemma-api-url http://localhost:11434/v1 --no-rfdetr
python main.py --mode local --gemma-api-url http://localhost:11434/v1 --rfdetr-model large
python main.py --mode local --unidrive-api-url http://localhost:8030/v1 --unidrive-model owl10/UniDriveVLA_Nusc_Base_Stage3
```

The local full-analysis flow combines local models and sidecar-backed models for Gemma, Qwen, Florence, and final reasoning.

### Local Step Order (23 steps)

**Perception and analysis (Steps 1–20)**

| Step | ID | Description |
|------|----|-------------|
| 1  | A   | Frame extraction |
| 2  | B   | Vector store indexing (CLIP + DINOv3) |
| 3  | J   | Gemma 4 open-weight multimodal analysis |
| 4  | L   | Florence-2 scene captioning |
| 5  | M   | ASR transcription (Whisper) |
| 6  | N   | OCR text extraction |
| 7  | O   | Monocular depth estimation |
| 8  | P   | Object detection (HF RT-DETR / Grounding DINO) |
| 9  | RF  | RF / SDR electromagnetic passive sensing (TorchSig) |
| 10 | TH  | Thermal / infrared imaging (LWIR radiometric) |
| 11 | MS  | Multispectral / hyperspectral imaging |
| 12 | EV  | Event camera (neuromorphic sensing) |
| 13 | LD  | LiDAR / active ranging (ToF, FMCW) |
| 14 | RD  | Radar (FMCW, Doppler, SAR) |
| 15 | GS  | GNSS-R + satellite signal reception (ADS-B, AIS, NOAA APT) |
| 16 | IM  | Inertial + barometric sensing (IMU, barometer, anemometer) |
| 17 | AT  | Atmospheric / environmental sensing |
| 18 | CH  | Chemical / gas / radiation sensing |
| 19 | AC  | Acoustic sensing (mic arrays, ultrasonic, hydrophone) |
| 20 | SF  | Sensor fusion analysis — temporal alignment, cross-modal detections, `frame_facts_json["sensor_fusion"]` |

**Detection, tracking, and 3D reconstruction (Steps 21–27)**

| Step | ID | Description |
|------|----|-------------|
| 21 | P2  | YOLO11 + SAM2/3 detection and segmentation |
| 22 | P3  | Gemma 4 directed tracking |
| 23 | Q   | World model video embeddings |
| 24 | R   | Qwen VLM detailed captioning |
| 25 | S   | UniDriveVLA expert analysis |
| 26 | C   | Base model transformation test |
| 27 | I   | 3D map + Gaussian Splat |

**Self-supervised learning and model adaptation (Steps 28–35)**

| Step | ID | Description |
|------|----|-------------|
| 28 | D   | SSL DINOv3 fine-tuning |
| 29 | E   | Knowledge distillation — maximum hydration chain |
| 30 | F   | ONNX export + gallery |
| 31 | G   | Fine-tuned model search test |
| 32 | H   | Model comparison + video description |
| 33 | T   | Multi-model comparison |
| 34 | Z   | Video synthesis |
| 35 | AA  | Agentic flow audit |

### Step SF — Sensor fusion analysis

Runs after YOLO/SAM (P2) and before Gemma directed tracking (P3). Consumes sidecar files and `sensor_packets` from the current session, temporally aligns all modalities to video frame timestamps, and writes `frame_facts_json["sensor_fusion"]` for each frame.

Key outputs per frame:

- `modalities_present` / `modalities_missing` — which sensor streams had valid data within `REALTIME_MAX_SENSOR_LAG_MS` of the frame timestamp
- `fusion_confidence` — geometric mean of per-modality confidence scores, penalised by `weather_factor`
- `cross_modal_detections` — YOLO RGB detections merged with thermal detections by IoU ≥ 0.4; `cross_modal_agreement` flags objects confirmed by multiple sensors
- `degradation_flags` — e.g. `["high_humidity", "wind_blur", "rf_shadow"]` derived from atmospheric sensor readings and RF SNR
- `pose_source` — which navigation filter produced the frame's pose (`ekf_imu_gps`, `vins`, `orbslam3`, `gps_fallback`)
- `pose_covariance_trace` — scalar summary of pose uncertainty; used to weight this frame's contribution to the global map

Active learning integration: frames where `cross_modal_agreement = false` for any high-priority detection are escalated to `al_tag = "novel"`. Frames with `plume_proximity_m < 50` or `dose_rate_usv_h > 1.0` are hard-flagged `al_tag = "needs_annotation"` regardless of visual novelty score.

Sidecar files consumed (all optional; step degrades gracefully when absent):

```
<video>.thermal.mp4       FLIR radiometric video
<video>.env.jsonl         atmospheric sensor log (t_sec, temp_c, humidity_pct, pressure_hpa, wind_speed_ms, wind_dir_deg)
<video>.adsb.jsonl        dump1090 aircraft list (one JSON per second)
<video>.gnssr.bin         GNSS-R delay-Doppler maps
<video>.lidar.bin         LiDAR point cloud (PCD or MCAP)
<video>.gas.jsonl         gas sensor log (t_sec, co2_ppm, voc_ppm, pm25_ug_m3, dose_rate_usv_h)
```

Config env vars: `SENSOR_FUSION_ENABLED` (default `false`), `SENSOR_FUSION_MAX_LAG_MS` (default `100`), `THERMAL_MODEL` (auto-resolves to YOLO-nano fine-tuned on FLIR ADAS).

See `docs/learning_path.md` Step 20 for the full fusion architecture, per-modality methods, library references, and public datasets. Steps 10–19 cover individual sensor families by physical principle.

### Step S — UniDriveVLA expert analysis

Runs after Qwen in the local pipeline and as an optional sparse enrichment pass in
production indexing. Requires `UNIDRIVE_API_URL` or `--unidrive-api-url`.

Normalized output schema:

- `understanding`: scene summary, traffic context, risk level, key agents
- `perception`: object list, drivable-area estimate, lane structure
- `planning`: recommended action, trajectory hint, hazards
- `mixture_of_experts`: consensus summary, expert agreement, disagreement points

Artifacts and outputs:

- Local: `unidrive_analysis.md`
- Local: `multi_model_comparison.md` when both Qwen and UniDrive are enabled
- Production: `frame_facts_json["unidrive_vla"]` and `index_video(...).unidrive_summary`

### Step P3 — Gemma 4 directed tracking

Runs after P2 (YOLO+SAM). Requires `--gemma-api-url` (or `GEMMA_API_URL` env var) to be
configured. Disabled with `--no-rfdetr` or `RFDETR_ENABLED=false`.

**Gemma structured scene analysis**: Up to 12 sampled frames are sent to the Gemma 4 sidecar
with a structured JSON prompt. Gemma returns:
- `scene_type` (e.g. `urban_street`, `rural_terrain`, `aerial`)
- `dominant_objects` with rough fractional bounding boxes (`[x1, y1, x2, y2]`)
- `tracking_priority` — ordered list of category labels to focus on

Responses are aggregated across sampled frames: most-common `scene_type` wins;
objects are merged by category; `tracking_priority` labels ranked by cross-frame frequency.

**SAM directed segmentation**:

- *Path A* (preferred): Gemma's `rough_bbox` values are fed directly as box prompts to
  `SAMPredictor.predict_boxes`. Efficient when Gemma can localise objects (~±20% tolerance).
- *Path B* (fallback): When Gemma cannot localise (uses the whole-frame fallback bbox),
  `SAM2AutomaticMaskGenerator` generates candidate masks at low density
  (`points_per_side=16`). Each mask crop is embedded by CLIP and scored against Gemma's
  object categories via cosine similarity (threshold 0.18). Masks above threshold are kept.

**RF-DETR tracking**: `RFDETRBase` or `RFDETRLarge` (`pip install rfdetr`) runs on up to 90
sampled frames, filtered to Gemma's `tracking_priority` labels. Persistent track IDs are
assigned by greedy IoU matching (threshold 0.45) across consecutive frames. IDs reset per
video/mission.

**Artifacts** (under `<output_dir>/<video>/`):

- `gemma_tracking_results.json` — scene summary, per-frame detections with track IDs, and per-frame SAM metadata
- `gemma_tracking/frame_*_tracked.jpg` — annotated frames with RF-DETR tracking boxes and IDs
- `gemma_tracking_summary.md` — Gemma scene interpretation, tracking statistics, and SAM-path summary

Current implementation detail: SAM outputs are persisted as metadata in
`gemma_tracking_results.json` and summarized in `gemma_tracking_summary.md`. The rendered
`frame_*_tracked.jpg` images currently show tracking boxes only; they do not re-render SAM
mask overlays.

**Config env vars**: `RFDETR_ENABLED` (default `true`), `RFDETR_MODEL` (`base`/`large`,
default `base`), `RFDETR_CONFIDENCE` (default `0.35`).

**Production**: `VideoIndexer._run_gemma_directed_tracking_pass` stores tracking results in
`frame_facts_json["gemma_tracking"]` for each frame record when both `RFDETR_ENABLED=true`
and `GEMMA_API_URL` are set.

## Pipeline outputs

Expect artifacts under `data/` such as:

- `data/videos/`
- `data/frames/`
- `data/tiles/`
- `data/reports/`
- `data/maps/`
- `data/checkpoints/`
- `data/models/`
- `data/gallery/`

Relevant semantic-graph artifacts:

- Production: `data/maps/<mission_id>/semantic_environment_graph.json`
- Local run: `<output_dir>/<video>/3d_map/semantic_environment_graph.json`
- Local summary: `<output_dir>/<video>/3d_map/semantic_environment_graph.md`

Relevant Gemma directed tracking artifacts (local runs):

- `<output_dir>/<video>/gemma_tracking_results.json`
- `<output_dir>/<video>/gemma_tracking/frame_*_tracked.jpg`
- `<output_dir>/<video>/gemma_tracking_summary.md`

For exact directories and defaults, see [`configuration.md`](./configuration.md).

---
[← Developer Guide](develop.md) | [Configuration →](configuration.md)
