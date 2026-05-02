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

### LangGraph orchestration (opt-in)

The pipeline has a LangGraph-based orchestrator as an opt-in replacement for the monolithic
`run_video_pipeline()` in `runner.py`. Activate it with an env var — no CLI change needed:

```bash
SELFSUVIS_USE_GRAPH=1 APP_ENV=dev selfsuvis --mode local --videos-dir data/videos
```

Both paths produce identical artifacts. The graph path adds:

- **Parallel fan-out** — steps 4–8 (Florence, ASR, OCR, depth, detection) are dispatched
  concurrently; they serialise only on GPU via the existing `_prep_vram_for_step` guard.
- **Resumable checkpoints** — each node's output is persisted; a failed run resumes from
  the last completed node rather than starting over.
- **LangSmith tracing** — full execution traces visible in the LangSmith UI when
  `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set.
- **Agentic quality improvements** on the six LLM nodes (see below).

**Key files:**

| File | Purpose |
|------|---------|
| `pipeline/workflows/local/graph_state.py` | `PipelineState` TypedDict — single state schema |
| `pipeline/workflows/local/runner_graph.py` | `build_graph()` + `run_graph_pipeline()` entry point |
| `pipeline/workflows/local/graph_nodes/phase1.py` | Nodes: init, extract, index |
| `pipeline/workflows/local/graph_nodes/phase2_parallel.py` | Nodes: Florence, ASR, OCR, depth, detection |
| `pipeline/workflows/local/graph_nodes/phase2_serial.py` | Nodes: Gemma, merge, platform fusion, world model, Qwen, UniDrive, SceneTok, base search, full fusion |
| `pipeline/workflows/local/graph_nodes/phase2_tracking.py` | Nodes: YOLO+SAM, Gemma tracking |
| `pipeline/workflows/local/graph_nodes/phase2_map.py` | Nodes: 3D map submit/join |
| `pipeline/workflows/local/graph_nodes/phase3_ssl.py` | Nodes: SSL finetune, distill, ONNX export, FT search, compare |
| `pipeline/workflows/local/graph_nodes/phase4.py` | Nodes: multi-model compare, synthesis, audit, analytics |
| `pipeline/workflows/local/graph_nodes/agentic_helpers.py` | Shared helpers: `json_guard`, `llm_call_with_retry`, `critique_pass`, `moe_consensus_score` |

**Environment variables for the graph path:**

| Variable | Default | Effect |
|----------|---------|--------|
| `SELFSUVIS_USE_GRAPH` | `` (off) | Set to `1` to activate the LangGraph orchestrator |
| `SELFSUVIS_CHECKPOINT_PATH` | `` (in-memory) | Path to a SQLite file for persistent checkpoints across process restarts |
| `SELFSUVIS_RESUME_THREAD_ID` | `` | Thread ID of a prior run to resume from its last checkpoint |
| `LANGCHAIN_TRACING_V2` | `` (off) | Set to `true` to emit execution traces to LangSmith |
| `LANGCHAIN_API_KEY` | `` | LangSmith API key (required when tracing is on) |

**Resuming a failed run** (persistent checkpoints only):

```bash
# First run — note the thread_id printed in the logs
SELFSUVIS_USE_GRAPH=1 SELFSUVIS_CHECKPOINT_PATH=data/checkpoints.db \
  selfsuvis --mode local --videos-dir data/videos
# => "Starting graph pipeline for drone_mission (thread_id=drone_mission_1714123456)"

# Resume after failure — nodes already completed are skipped
SELFSUVIS_USE_GRAPH=1 SELFSUVIS_CHECKPOINT_PATH=data/checkpoints.db \
  SELFSUVIS_RESUME_THREAD_ID=drone_mission_1714123456 \
  selfsuvis --mode local --videos-dir data/videos
```

### Agentic improvements in the LangGraph path

The six LLM nodes have targeted quality enhancements that are only active when
`SELFSUVIS_USE_GRAPH=1`:

| Step | Improvement |
|------|-------------|
| 03 Gemma analysis | Post-hoc CLIP cosine similarity check on every `fact_verification` claim; claims below threshold flagged as `clip_verified=false`; `unverified_claims` count surfaced in trace |
| 10 Gemma tracking | JSON-guard fallback: when Gemma JSON parse fails and returns empty `target_labels`, RF-DETR falls back to `["person","vehicle","sign"]` instead of silently tracking nothing |
| 12 Qwen captioning | One retry pass for frames with `parse_error=True` using a simplified prompt (no extra context); prior-state chain skips confirmed-bad frames rather than anchoring on them |
| 13 UniDriveVLA | Per-frame Jaccard MoE consensus score computed across expert outputs; frames below threshold 0.5 flagged as `low_moe_agreement=true` and logged as warnings |
| 29 Video synthesis | Draft → CLIP-evidence-grounded critique pass → conditional regeneration when verdict is `MAJOR_CONTRADICTION` |
| 30 Agentic flow audit | Reflection sub-loop after generation: checks whether all pipeline step IDs are covered; appends a `## Reflection Gaps` section to `agentic_flow.md` when gaps are found |

### Local Step Order

The local runner reports 32 runtime/post-run steps. The LangGraph path covers the
main graph nodes through per-video analytics (steps 1–16, 20–21, 23–26, 29–30);
steps 17–19, 22, 27–28, 31–32 run only in the monolithic runner.

| Step | Phase | Description | LangGraph node |
|------|-------|-------------|----------------|
| 01 | Ingest | Frame extraction | `p1_extract_frames` |
| 02 | Ingest | Vector store indexing (CLIP + DINOv3) | `p1_index_vectors` |
| 03 | Analyze | Gemma multimodal analysis *(agentic)* | `p2_gemma_analysis` |
| 04 | Analyze | Florence-2 scene captioning | `p2_florence_caption` *(parallel)* |
| 05 | Analyze | ASR transcription | `p2_asr` *(parallel)* |
| 06 | Analyze | OCR text extraction | `p2_ocr` *(parallel)* |
| 07 | Analyze | Depth estimation | `p2_depth` *(parallel)* |
| 08 | Analyze | Object detection | `p2_detection` *(parallel)* |
| 09 | Analyze | YOLO11 + SAM2/3 detection | `p2_yolo_sam` |
| 10 | Analyze | Gemma 4 directed tracking *(agentic)* | `p2_gemma_tracking` |
| 11 | Analyze | World model video embeddings | `p2_world_model` |
| 12 | Analyze | Qwen VLM detailed captioning *(agentic)* | `p2_qwen_caption` |
| 13 | Analyze | UniDriveVLA expert analysis *(agentic)* | `p2_unidrive` |
| 14 | Analyze | SceneTok streaming encoder + segmentation decoder | `p2_scenetok` |
| 15 | Eval | Base model transformation test | `p2_base_search` |
| 16 | Map | 3D map + Gaussian Splat *(background thread)* | `p2_map_3d_submit` / `p2_map_3d_join` |
| 17 | State | Physical scene state summary → `physical_state_summary.json` | — |
| 18 | State | Environmental field state → `field_state_summary.json` | — |
| 19 | State | Threat primitives → `threat_primitives.json` | — |
| 20 | Adapt | SSL DINOv3 fine-tuning | `p3_ssl_finetune` |
| 21 | Adapt | Knowledge distillation Stage 1: teacher → ViT-S/14 | `p3_distill` *(SSL gate required)* |
| 22 | Adapt | Knowledge distillation Stage 2: ViT-S/14 → EfficientViT-B1 + ONNX | — *(Stage 1 required)* |
| 23 | Export | ONNX export + gallery build → `edge_models/` | `p3_onnx_export` *(SSL gate required)* |
| 24 | Eval | Fine-tuned model transformation test | `p3_ft_search` *(SSL gate required)* |
| 25 | Eval | Model comparison + video description | `p3_compare` *(SSL gate required)* |
| 26 | Audit | Multi-model comparison | `p4_multi_model_compare` |
| 27 | Threat | Local threat inference → `local_threat_assessment.json` | — |
| 28 | Threat | Action policy → `policy_decision.json` | — |
| 29 | Synthesize | Video synthesis *(agentic)* | `p4_synthesis` |
| 30 | Audit | Agentic flow audit *(agentic)* | `p4_audit` |
| 31 | Adapt | Drone detection edge training *(opt-in)* | — |
| 32 | Optimize | Model/run advisor → `model_run_advisor.md` | run-level postprocessing |

Not every step runs on every machine or configuration. Steps may be skipped when a
feature flag is disabled, a sidecar URL is not configured, a resource gate blocks the
stage, or an earlier fine-tune quality gate does not pass.

### coop_pilot learning extension

`coop_pilot` is a continuous site-awareness extension, not another stage inside a
single `selfsuvis --mode local` video run. In the learning path it follows the
36-step conceptual local curriculum as Steps 37-43:

| Step | Focus | Runtime surface |
|------|-------|-----------------|
| 37 | Coop stack bootstrap and health | `scripts/coop-bootstrap.sh`, `scripts/coop-compose.sh`, `tests/coop/test_stack_health.py` |
| 38 | MQTT and LoRaWAN ingestion | ChirpStack MQTT uplinks, `LoRaWANDecoder`, `SensorReading` |
| 39 | Frigate event ingestion | Frigate MQTT events, `FrigateEventDecoder`, `CameraEvent` |
| 40 | Rolling site state | `SiteStateAggregator`, `/site/state`, `/site/sensors`, `/site/cameras` |
| 41 | RTSP bridge and acoustic evidence | MediaMTX bridge sessions, live-stream analysis, synthetic acoustic events |
| 42 | Site mesh and scene synthesis | GPS proximity graph, `/site/mesh`, `/site/synthesis` |
| 43 | Realtime threat bridge and analytics | `coop_ingest`, `/site/threat`, `coop-analytics` |

Use [Local Learning Path](local_path.md#coop_pilot-extension-steps) for the short
study sequence and [coop_pilot IoT edge monitoring](learning_path/16_coop_pilot_iot_edge_monitoring.md)
for the deep dive.

Current local-run optimizations also make a few steps adaptive instead of fully exhaustive:

- Step 06 (OCR) prescreens frames from Florence caption confidence before sending them to the OCR model or sidecar.
- Step 12 (Qwen) uses bounded sampled-frame selection instead of captioning every frame.
- Step 07 (Depth) uses a fast auto profile by default unless an explicit model or quality profile is requested.
- Step 30 (agentic flow audit) in the monolith uses a simple first-pass prompt and accepts that answer when it satisfies the required output structure; in the LangGraph path a reflection sub-loop also runs.
- The local pipeline runs a probabilistic platform-state fusion pass and writes `state_fusion.md` / `state_fusion.json` when GPS telemetry is available.

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

## Perspective Directions

This page describes the current runtime.
The forward-looking roadmap now lives in the learning-path deep dive:

- [Advanced directions: global threats, sensor meshes, and cross-modal world models](future_implementation_directions.md)

Use that document for:

- the recommended next technical directions after the current runner
- the main papers behind temporal SSL, world models, RL, and physical-state estimation
- the proposed expansion from context fusion toward local/global threat analysis over a realtime sensor mesh
- the post-run model advisor that inspects `analysis_summary.json`, warnings, hardware resources, and `.env`

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
- `data/local_runs/model_run_advisor.md`
- `data/local_runs/model_run_advisor.json`

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
