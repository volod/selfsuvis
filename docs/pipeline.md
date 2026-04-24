# Pipeline

This page describes the current indexing flow and the local full-analysis flow.

## Production indexing flow

1. A client calls one of the indexing endpoints.
2. The API validates input and writes a job to PostgreSQL.
3. The worker claims the job and runs `selfsuvis.pipeline.workflows.indexer.VideoIndexer`.
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

The current production indexing path also includes an initial probabilistic
platform-state fusion slice when GPS is available:

- GPS extracted from video metadata is converted into typed position measurements
- optional `.imu.jsonl` and `.baro.jsonl` sidecars next to the source video are
  used as acceleration and altitude inputs
- a constant-velocity Kalman filter produces posterior summaries on indexed frame
  timestamps
- results are stored in `frame_facts_json["state_fusion"]`

## Useful command-line examples

```bash
curl -s -F "path=/path/to/video.mp4" \
  http://localhost:8000/index/precheck | python -m json.tool

curl -s -F "path=/path/to/video_dir" -F "enqueue=true" -F "enable_tiles=true" \
  http://localhost:8000/index/precheck_dir | python -m json.tool

curl -s -F "url=https://example.com/video.mp4" -F "enable_tiles=true" \
  http://localhost:8000/index/url | python -m json.tool

curl -s -F "path=/path/to/video_dir" -F "enable_tiles=true" \
  http://localhost:8000/index/dir | python -m json.tool

JOB_ID=<job_id>
while true; do
  STATUS="$(curl -s http://localhost:8000/jobs/${JOB_ID})"
  echo "$STATUS" | python -m json.tool
  STATE="$(printf '%s' "$STATUS" | python -c 'import json,sys; print(json.load(sys.stdin).get("status",""))')"
  [[ "$STATE" == "finished" || "$STATE" == "error" ]] && break
  sleep 2
done
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
from selfsuvis.pipeline.workflows.indexer import VideoIndexer

indexer = VideoIndexer(enable_tiles=True)
result = indexer.index_video("/path/to/video.mp4", "dev_test")
print(result)
```

This path is useful for local debugging when PostgreSQL/Qdrant are already reachable. The returned dict includes a `semantic_graph` summary when YOLO SSG is enabled and a `unidrive_summary` when UniDrive enrichment is enabled.

## Local Full-Analysis Mode

`main.py` now defaults to the local full-analysis and training pipeline (`--mode local`),
implemented by the current CLI in `src/selfsuvis/pipeline/workflows/cli_parser.py` and
the runner modules under `src/selfsuvis/pipeline/workflows/local`.

Common options:

```bash
selfsuvis
selfsuvis --mode local --input /path/to/video.mp4
selfsuvis --mode local --dir /path/to/video_dir --no-qdrant --no-sfm --no-gsplat
selfsuvis --mode local --qwen-api-url http://localhost:8010/v1
selfsuvis --mode local --gemma-api-url http://localhost:11434/v1
selfsuvis --mode local --no-yolo --no-sam
selfsuvis --mode local --gemma-api-url http://localhost:11434/v1 --no-rfdetr
selfsuvis --mode local --gemma-api-url http://localhost:11434/v1 --rfdetr-model large
selfsuvis --mode local --unidrive-api-url http://localhost:8030/v1 --unidrive-model owl10/UniDriveVLA_Nusc_Base_Stage3
```

The local full-analysis flow combines local models and sidecar-backed models for Gemma,
Qwen, Florence, UniDrive, and final reasoning.

### Local Step Order (23 top-level steps)

The current local runner executes 23 top-level steps. This list matches
`src/selfsuvis/pipeline/workflows/local/runner.py`.

| Step | Phase | Description |
|------|-------|-------------|
| 01 | Ingest | Frame extraction |
| 02 | Ingest | Vector store indexing (CLIP + DINOv3) |
| 03 | Analyze | Gemma multimodal analysis |
| 04 | Analyze | Florence-2 scene captioning |
| 05 | Analyze | ASR transcription |
| 06 | Analyze | OCR text extraction |
| 07 | Analyze | Depth estimation |
| 08 | Analyze | Object detection |
| 09 | Analyze | YOLO11 + SAM2/3 detection and semantic graph construction |
| 10 | Analyze | Gemma 4 directed tracking |
| 11 | Analyze | World model video embeddings |
| 12 | Analyze | Qwen VLM detailed captioning |
| 13 | Analyze | UniDriveVLA expert analysis |
| 14 | Eval | Base model transformation test |
| 15 | Map | 3D map + Gaussian Splat |
| 16 | Adapt | SSL DINOv3 fine-tuning |
| 17 | Adapt | Knowledge distillation |
| 18 | Export | ONNX export + gallery build |
| 19 | Eval | Fine-tuned model transformation test |
| 20 | Eval | Model comparison + video description |
| 21 | Audit | Multi-model comparison |
| 22 | Synthesize | Video synthesis |
| 23 | Audit | Agentic flow audit |

Not every step runs on every machine or configuration. Steps may be skipped when a
feature flag is disabled, a sidecar URL is not configured, a resource gate blocks the
stage, or an earlier fine-tune quality gate does not pass.

Current local-run optimizations also make a few steps adaptive instead of fully exhaustive:

- Step 06 (OCR) prescreens frames from Florence caption confidence before sending them to the OCR model or sidecar.
- Step 12 (Qwen) uses bounded sampled-frame selection instead of captioning every frame.
- Step 07 (Depth) uses a fast auto profile by default unless an explicit model or quality profile is requested.
- Step 23 (agentic flow audit) uses a simple first-pass prompt and accepts that answer when it satisfies the required output structure; a compact fallback prompt is only used when the first response is empty or incomplete.
- The local pipeline now also runs a probabilistic platform-state fusion example and writes `state_fusion.md` / `state_fusion.json` when GPS telemetry is available.

### Step 13 — UniDriveVLA expert analysis

Runs after Qwen in the local pipeline and as an optional sparse enrichment pass in
production indexing. Requires `UNIDRIVE_API_URL` or `--unidrive-api-url`.

**Adapter design:** `pipeline/vision/unidrive.py` is a thin HTTP adapter that works with
any OpenAI-compatible vision endpoint.  The structured driving-domain schema is prompted
from the backend model; no direct model loading occurs in the worker process.
For non-road missions (aerial, off-road, maritime), use a Qwen2.5-VL-7B sidecar as the
backend rather than the driving-specific `owl10/UniDriveVLA_Nusc_*` checkpoint.

See [`docs/runbooks/unidrive-api.md`](runbooks/unidrive-api.md) for setup and
backend selection guidance.

Normalized output schema:

- `understanding`: scene summary, traffic context, risk level, key agents
- `perception`: object list, drivable-area estimate, lane structure
- `planning`: recommended action, trajectory hint, hazards
- `mixture_of_experts`: consensus summary, expert agreement, disagreement points

Artifacts and outputs:

- Local: `unidrive_analysis.md`
- Local: `multi_model_comparison.md` when both Qwen and UniDrive are enabled
- Production: `frame_facts_json["unidrive_vla"]` and `index_video(...).unidrive_summary`

### Step 10 — Gemma 4 directed tracking

Runs after step 09 (YOLO+SAM). Requires `--gemma-api-url` (or `GEMMA_API_URL` env var) to be
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

Post-run analytics:

- main CLI: `selfsuvis --mode analyse --run-dir <output_dir>/<video>`
- module form: `python -m selfsuvis --mode analyse --run-dir <output_dir>/<video>`
- guide: [`analytics.md`](./analytics.md)

For exact directories and defaults, see [`configuration.md`](./configuration.md).

---
[← Developer Guide](develop.md) | [Configuration →](configuration.md)
