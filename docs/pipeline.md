# Pipeline

This page describes the current indexing and demo flows, not historical design stages.

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

This path is useful for local debugging when PostgreSQL/Qdrant are already reachable. The returned dict includes a `semantic_graph` summary when YOLO SSG is enabled.

## Demo mode

`main.py --mode demo` runs the standalone demo pipeline defined by the current CLI in [`pipeline/workflows/cli_parser.py`](/home/vola/src/selfsuvis/pipeline/workflows/cli_parser.py) and the demo runner modules under [`pipeline/workflows/demo`](/home/vola/src/selfsuvis/pipeline/workflows/demo).

Common options:

```bash
python main.py --mode demo
python main.py --mode demo --no-qdrant --no-sfm --no-gsplat
python main.py --mode demo --asr --ocr --depth --detection
python main.py --mode demo --qwen --qwen-api-url http://localhost:8010/v1
python main.py --mode demo --gemma-api-url http://localhost:11434/v1
python main.py --mode demo --no-yolo --no-sam
python main.py --mode demo --gemma-api-url http://localhost:11434/v1 --no-rfdetr
python main.py --mode demo --gemma-api-url http://localhost:11434/v1 --rfdetr-model large
```

The demo combines local models and sidecar-backed models for Gemma, Qwen, Florence, and final reasoning.

### Demo step order (21 steps)

| Step | ID | Description |
|------|----|-------------|
| 1  | A   | Frame extraction |
| 2  | B   | Vector store indexing |
| 3  | J   | Gemma multimodal analysis |
| 4  | L   | Florence-2 scene captioning |
| 5  | M   | ASR transcription |
| 6  | N   | OCR text extraction |
| 7  | O   | Depth estimation |
| 8  | P   | Object detection (HF) |
| 9  | P2  | YOLO11 + SAM2/3 detection |
| 10 | P3  | Gemma 4 directed tracking |
| 11 | Q   | World model video embeddings |
| 12 | R   | Qwen VLM detailed captioning |
| 13 | C   | Base model transformation test |
| 14 | I   | 3D map + Gaussian Splat |
| 15 | D   | SSL DINOv3 fine-tuning |
| 16 | E   | Knowledge distillation |
| 17 | F   | ONNX export + gallery |
| 18 | G   | Fine-tuned model search test |
| 19 | H   | Model comparison |
| 20 | Z   | Video synthesis |
| 21 | AA  | Agentic flow audit |

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

- `gemma_tracking_results.json` — per-frame detections with track IDs + SAM mask metadata
- `gemma_tracking/frame_*_tracked.jpg` — annotated frames (tracking boxes + IDs)
- `gemma_tracking_summary.md` — Gemma scene interpretation, tracking statistics, SAM counts

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
- Demo: `<output_dir>/<video>/3d_map/semantic_environment_graph.json`
- Demo summary: `<output_dir>/<video>/3d_map/semantic_environment_graph.md`

Relevant Gemma directed tracking artifacts (demo):

- `<output_dir>/<video>/gemma_tracking_results.json`
- `<output_dir>/<video>/gemma_tracking/frame_*_tracked.jpg`
- `<output_dir>/<video>/gemma_tracking_summary.md`

For exact directories and defaults, see [`configuration.md`](./configuration.md).

---
[← Developer Guide](develop.md) | [Configuration →](configuration.md)
