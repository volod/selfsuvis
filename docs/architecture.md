# Architecture

## Repository structure

```text
app/        FastAPI routers, dependencies, schemas, and request services
pipeline/   indexing, mapping, captioning, spatial logic, model integration
models/     retrieval backbones and local model loaders
worker/     PostgreSQL-backed async job worker
ui/         Streamlit client
scripts/    setup, model prep/export, and pipeline helper CLIs
docker/     compose files and container definitions
tests/      unit and integration coverage
docs/       operator, developer, and decision documentation
```

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
                   consumes queued jobs and runs VideoIndexer
```

Optional services:

- `nerfstudio` for splat generation
- `mapper` for map registration/fusion work
- `mediamtx` for stream ingestion
- `cvat` for annotation workflows

## Indexing flow

1. A request to `/index/video`, `/index/url`, or `/index/dir` creates a PostgreSQL job.
2. The worker claims the job and runs the indexing pipeline.
3. Video frames are sampled and filtered.
4. Frames are captioned and embedded.
5. Tiles are optionally extracted, filtered, deduplicated, and embedded.
6. Metadata is written to PostgreSQL and vectors are written to Qdrant.
7. Optional downstream steps run: change detection, reports, mapping, multimodal facts, fine-tune triggers.

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
from pipeline.core import is_cuda_oom, resolve_device, pipeline_device_arg
```

### `pipeline/vision/registry.resolve_model_id`

Helper used by every vision model wrapper to avoid duplicating the four-line "read setting → auto-select → fallback" pattern:

```python
from pipeline.vision.registry import resolve_model_id

def _resolve_model_id() -> str:
    return resolve_model_id(settings.DEPTH_MODEL, "depth", "depth-anything/Depth-Anything-V2-Small-hf")
```

When the setting is non-empty and not `"auto"`, the value is returned as-is.  Otherwise `auto_select` picks the largest model that fits in available VRAM, falling back to the explicit *fallback* ID if the catalog has no match.

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
