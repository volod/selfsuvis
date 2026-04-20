# Architecture

## Repository structure

```text
src/selfsuvis/
  app/        FastAPI routers, request dependencies, and API services
  models/     retrieval backbones and local model loaders
  pipeline/   indexing, mapping, media, storage, realtime, training, workflows
  scripts/    packaged helper CLIs such as `selfsuvis-env`
  worker/     PostgreSQL-backed async job worker
docker/       compose files and container definitions
tests/        unit, integration, assets, and shared test helpers
docs/         operator, developer, and decision documentation
```

### Test structure

`tests/unit/` mirrors `src/selfsuvis/` where practical:

```text
tests/unit/
  app/
  models/
  pipeline/
  scripts/
  worker/
```

Reusable fake DB pools, factories, and non-fixture test helpers live in `tests/support/`.
The one intentional flat unit test is `tests/unit/test_multisite_enu.py`, which remains
at the root because it spans app, storage, and worker behavior together.

## Runtime architecture

```text
client/UI
   |
   v
FastAPI API  ----> PostgreSQL
   |                jobs, missions, frames, automation state
   |
   +-----------> Qdrant
   |              frame/tile vectors and payloads
   |
   +-----------> worker
                   consumes queued jobs and runs
                   selfsuvis.pipeline.workflows.indexer.VideoIndexer
```

Optional services:

- `nerfstudio` for splat generation
- `mapper` for map registration/fusion work
- `mediamtx` for stream ingestion
- `cvat` for annotation workflows

## Indexing flow

1. A request to `/index/video`, `/index/url`, or `/index/dir` creates a PostgreSQL job.
2. The worker claims the job and runs the indexing pipeline.
3. Video frames are sampled, quality-filtered, and embedded.
4. Core multimodal enrichments run: Florence captions, ASR, OCR, depth, and detection.
5. Optional higher-level analysis stages run when enabled:
   - YOLO + SAM semantic environment graph construction
   - Gemma-directed tracking with SAM prompts and RF-DETR sequence tracking
   - Qwen VLM detailed frame reasoning
   - UniDriveVLA expert understanding / perception / planning analysis
6. Metadata is written to PostgreSQL and vectors are written to Qdrant.
7. Optional downstream stages run: 3D mapping, reports, active learning, fine-tune triggers,
   model distillation, ONNX export, and multi-model comparison artifacts.

## Algorithmic additions

Recent pipeline additions that materially changed the system architecture:

- **Semantic environment graph**: production indexing and local runs can cluster mission
  detections into a mission-scoped semantic graph, persisted as JSON/markdown artifacts
  and referenced from downstream reports.
- **Gemma-directed tracking**: Gemma produces structured scene/object hints, SAM converts
  those hints into segmentation prompts, and RF-DETR tracks the selected object classes
  across the frame sequence.
- **UniDriveVLA expert pass**: an OpenAI-compatible vision backend produces normalized
  `understanding`, `perception`, `planning`, and `mixture_of_experts` outputs that are
  stored in `frame_facts_json["unidrive_vla"]` and summarized in local-run artifacts.
- **Resource-aware env generation**: `selfsuvis-env` generates a project-root `.env`
  from packaged presets and detected hardware, which is now the standard way to bootstrap
  local configuration.

## Shared utility packages

Two packages centralise cross-cutting helpers that would otherwise be copy-pasted into every model file:

### `pipeline/core/gpu_utils.py`

GPU and device utilities used by every model loader and vision pipeline stage:

| Symbol | Purpose |
|--------|---------|
| `is_cuda_oom(exc)` | Returns `True` when `exc` is a CUDA out-of-memory error (works for both `torch.cuda.OutOfMemoryError` and older `RuntimeError` messages). |
| `resolve_device(device_cfg=None)` | Maps `settings.DEVICE` (or an explicit string) to `"cuda"`, `"mps"`, or `"cpu"` with proper availability checks including Apple MPS. |
| `pipeline_device_arg(device)` | Converts a device string to the integer HuggingFace `pipeline()` expects: `-1` for CPU, `0` for everything else. |

All three are re-exported from `pipeline.core` for convenience:

```python
from selfsuvis.pipeline.core import is_cuda_oom, resolve_device, pipeline_device_arg
```

### `pipeline/vision/registry.resolve_model_id`

Helper used by every vision model wrapper to avoid duplicating the four-line "read setting → auto-select → fallback" pattern:

```python
from selfsuvis.pipeline.vision.registry import resolve_model_id

def _resolve_model_id() -> str:
    return resolve_model_id(settings.DEPTH_MODEL, "depth", "depth-anything/Depth-Anything-V2-Base-hf")
```

When the setting is non-empty and not `"auto"`, the value is returned as-is. Otherwise `auto_select` applies the current policy for that model family, falling back to the explicit *fallback* ID if the catalog has no match.

## Query architecture

- `/query/text`: OpenCLIP text embedding against Qdrant vectors
- `/query/image`: image embedding with optional DINO vector space
- `/query/scene`: PostgreSQL filtering over `frame_facts_json` with optional CLIP reranking
- `/query/pose`: GPS or ENU spatial filtering plus vector ranking

## Main state stores

### PostgreSQL

Holds:

- `jobs`
- `missions`
- `frames`
- `processed_files`
- `change_detections`
- `global_map` and related mapping tables
- CVAT and automation state such as `cvat_tasks`, `system_state`, `gpu_jobs`, and model provenance tables

### Qdrant

Stores frame and tile points with named vectors and retrieval payloads such as:

- frame/tile type
- mission and robot IDs
- timestamps
- GPS and ENU coordinates
- model-version provenance

## Coordinate model

The spatial pipeline uses GPS payloads for broad filtering and ENU coordinates for local-map and robot-oriented queries when available.

---
[← Configuration](configuration.md) | [Pipeline →](pipeline.md)
