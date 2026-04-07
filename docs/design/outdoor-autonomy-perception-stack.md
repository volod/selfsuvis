# Design: Outdoor Autonomy Perception Stack — Spatial Memory Engine with Active Tagging (Training Loop in v2)

---

## Problem Statement

Build an outdoor autonomy perception stack: a system that ingests mission video from drones, rovers, or ground vehicles, produces a queryable 3D semantic world model (spatial memory), and continuously improves its own perception models using accumulated mission data (self-improvement loop). Self-use first (own outdoor autonomy system), then productize to other outdoor robotics teams.

The existing `selfsuvis` POC (adaptive frame extraction, OpenCLIP embeddings, Qdrant vector search, FastAPI, Docker Compose) is the foundation to build on.

---

## Demand Evidence

- Solving own problem: builder is developing an outdoor autonomy system (drone/rover/vehicle) and needs a perception layer that understands the environment, not just records it.
- No current paying users; pre-product stage. Self-use is the demand signal.
- Market: outdoor robotics teams (surveying, inspection, autonomous navigation) face the same problem — massive video output, no queryable world model.

---

## Status Quo

Today, a field robotics team collects mission video and either:
1. Manually scrubs footage to find events/locations of interest
2. Runs separate offline tools for SLAM (LIO-SAM, VINS-Fusion), object detection (YOLOX), and captioning — with no unified interface
3. Stores raw video without spatial indexing, losing the ability to query "where did we see X"

The workaround costs hours per mission review and produces no reusable training data.

---

## Target User & Narrowest Wedge

**Target user:** Outdoor robotics engineer building a drone/rover/vehicle autonomy system. Has mission video. Needs to query it spatially and semantically. Will pay with engineering time now, money later.

**Narrowest wedge:** Upload a mission MP4 → get a searchable, spatially-anchored archive: "show me all frames where we saw a path with dense vegetation" → results with GPS-less 3D pose + timestamp + frame + image caption.

---

## Constraints

- Language: Python 3, PyTorch, CUDA
- Single-machine deployment: Docker Compose (v1); k8s cluster (future)
- Must run on developer hardware (ideally GPU optional for search, GPU required for map reconstruction)
- No external cloud dependencies for core pipeline
- Existing selfsuvis architecture: FastAPI + Qdrant + Streamlit + Docker Compose — preserved and extended. SQLite (`jobs.db`, `processed.db`) migrated to PostgreSQL; all SQL-based storage unified in one PostgreSQL instance.

---

## Premises

1. **Core product = 3D semantic map + image-to-text description.** The 3D map gives spatial anchoring (where was each frame captured, what does the space look like). The text description gives natural language searchability. Both are essential — neither alone is the product.

2. **Self-improvement loop = the competitive moat.** Data → training → better models → better maps. This compounds over time. No competitor has your accumulated mission data once you're deployed.

3. **Dense 3D map (3DGS) is core from v1, not a later upgrade.** The 3D map is the differentiator for navigation and spatial understanding. A sparse pose graph is not sufficient. (User held this premise against Codex's challenge — strong signal.)

4. **File processing is v1; MediaMTX streaming is v1.5.** File ingest is the fastest path to a working demo and test loop. Existing video files can be re-broadcast via MediaMTX for streaming testing — so live stream support uses the same pipeline once MediaMTX is integrated.

5. **PostgreSQL for dataset metadata; no FiftyOne in v1.** FiftyOne is MongoDB-only and cannot use PostgreSQL. PostgreSQL fits the existing SQLite-based architecture pattern, CVAT (v2 annotation) uses PostgreSQL natively (shared DB), and the active learning tagging logic is simple enough to own directly. lakeFS, Datumaro, and Elasticsearch remain deferred.

6. **DINOv3 or R3M over generic CLIP for egocentric video.** Models pre-trained on web images fail on robot camera views in non-obvious ways. R3M (time-contrastive on human video) and DINOv3 are better priors for robot POV perception.

---

## Cross-Model Perspective

**Codex cold read (independent, ran without seeing this conversation):**

> The strongest version is a spatial memory layer for outdoor autonomy — it turns raw mission video into a queryable world model so engineers (and later robots) can ask "where did we see X, from what pose, in what spatial context?" and reuse the same corpus to improve perception over time. If it works, it is not a video analytics tool; it is the data backbone of field robotics.
>
> The key quote is "I'm solving my own problem." This should not start as generic robotics infra. It should start as the shortest path from a field run to a useful internal artifact: searchable observations, spatial grounding, and better training data for your own autonomy stack. If it does that well, productization is plausible; if it does not, the platform story is fiction.
>
> Challenged premise 3: V1 should prove searchable spatial memory, not "usable 3D map" as an end in itself. A sparse pose graph + semantically indexed keyframes answers most real internal questions; dense 3D adds demo value but little operational value.
>
> 48-hour prototype: MP4 → adaptive keyframe extraction → pycolmap camera poses → DINOv3/R3M embeddings → Florence-2 captions → store frame/timestamp/pose/caption/embedding in Qdrant + FiftyOne metadata → Streamlit with timeline, sparse 3D camera path, text search, image search, and click-through from result to map location.

**Cross-model synthesis:**
- Both agree: "solving own problem" anchor + self-improvement loop = moat
- Both agree: DINOv3, FiftyOne, existing Qdrant/FastAPI stack
- **Disagreement on premise 3:** Codex argues sparse pose graph is sufficient for v1. User kept dense 3DGS as core. User's reasoning: 3D map IS the differentiator for navigation; sparse pose graph serves ML but not robot planning. This is defensible — if the robot needs to navigate, it needs a dense map, not just indexed keyframes.
- Recommendation: consider pycolmap as a fast v0.5 prototype (matches Codex's 48h spec), then upgrade to nerfstudio/splatfacto dense 3DGS as v1. This sequences risk without changing architecture.

---

## Approaches Considered

### Approach A: "Searchable Spatial Memory" (minimal viable)
- Extend selfsuvis: replace CLIP with DINOv3, add pycolmap poses + nerfstudio/splatfacto 3DGS, add Florence-2 captioning, add FiftyOne metadata layer
- File ingest only
- No streaming, no training loop, no distillation
- **Effort:** M — human ~6 weeks / CC ~3-4 days
- **Risk:** Low
- **Rejected because:** doesn't architect for the self-improvement moat from day 1; retrofitting active learning later is expensive

### Approach B: "Full Perception Platform v1" (complete vision)
- Full spec: streaming, 3DGS, captioning, CVAT annotation, self-supervised training pipeline, knowledge distillation, edge export, k8s
- **Effort:** XL — human ~6 months / CC ~3-4 weeks
- **Risk:** High (integration complexity before validation)
- **Rejected because:** ocean, not a lake; minimal perception principle; no validation before full build

### Approach C: "Active Tagging from Day 1 — Learning Activated in v2" (chosen)
- Same core as A, but data pipeline is architected around active tagging from first mission
- Every inference run auto-flags uncertain/novel frames for annotation via FiftyOne (tagging only; training is v2)
- The moat data architecture is correct from day 1 — no retrofit when training is added in v2
- **Effort:** M-L — human ~8 weeks / CC ~5-6 days
- **Risk:** Medium

---

## Recommended Approach: C — Active Tagging from Day 1 (Training Activates in v2)

### Architecture

**Ingestion**
- File upload (existing FastAPI + SQLite job queue, preserved)
- HTTPS/HTTP link ingestion (existing downloader, preserved)
- MediaMTX streaming server (new container, **v1 — file re-streaming only**) — use case in v1 is re-streaming local files for testing the streaming code path; live RTSP/RTMP ingest from real cameras is v1.5
- S3/FTP link support: **deferred to v2** (require boto3 + credential management; not needed before self-use validation)

> **Note on Qdrant migration:** The existing index uses OpenCLIP embeddings. Switching the primary image embedding to DINOv3 invalidates all existing vectors. Before v1 goes live, run `scripts/reset_qdrant.sh` to wipe + re-index. The `dino` named vector already exists in the codebase (see `qdrant_utils.py`), so this is a config change + re-index, not a schema redesign.

**Frame Extraction** *(existing, extend)*
- Keep existing adaptive sampling pipeline (`frame_extractor.py`, `heuristics.py`)
- Add GPS/IMU metadata extraction from video containers (if available)
- Add frame dedup hash (SHA-256, existing `dedup.py`)

**Embedding + Captioning** *(new)*
- **Keep OpenCLIP** for the `clip` named vector — it is the cross-modal text↔image search vector. OpenCLIP text encoder and image encoder are in the same embedding space; replacing image encoding alone while keeping the text encoder breaks search.
- **Add DINOv3** as the `dino` named vector (existing `dino_model.py` supports dinov3 — upgrade pretrained weights). Used for visual similarity reranking, active learning uncertainty scoring, and future self-supervised training. NOT used for text queries.
- Search path: text query → OpenCLIP text encoder → `clip` vector → results; optionally reranked by DINOv3 `dino` similarity (70/30 blend, same pattern as existing code).
- Image query path: OpenCLIP image encoder → `clip` search + DINOv3 → `dino` reranking.
- Add Florence-2 (`models/florence_model.py`) for image-to-text description (English captions with confidence). Captions stored as metadata in Qdrant payload and FiftyOne samples.

**Camera Pose Estimation** *(new module: `pipeline/sfm.py`)*
- pycolmap (Structure-from-Motion) — fast, battle-tested, Python API. v1 deliverable.
- Input: keyframes (JPEG files on disk, output of frame extractor)
- Output: `colmap_sparse/` directory (images.bin, cameras.bin, points3D.bin) + `poses.json` (per frame_id: rotation matrix, translation, camera intrinsics, confidence)
- `poses.json` is the interface between `sfm.py` and `mapper.py`; also written to Qdrant payload per frame
- Failure handling: if pycolmap reconstruction fails (insufficient overlap, textureless surfaces), log mission as `pose_status=failed`, store frames with `pose=null` in Qdrant, skip 3DGS step. Frames are still searchable via CLIP+DINO; only spatial query is degraded.
- Fallback: if GPS/IMU available in video container (ffprobe metadata), store lat/lon/alt as `gps` payload field in Qdrant alongside or instead of colmap pose.

**3D Map Reconstruction** *(new module: `pipeline/mapper.py`)*
- nerfstudio `splatfacto` (3D Gaussian Splatting) — v1 deliverable, runs asynchronously after sfm.py completes
- Input: keyframes + `colmap_sparse/` directory (output of sfm.py)
- Output: `maps/{mission_id}/splat.ply` — viewable in web-based 3DGS viewers
- Qdrant payload: add `map_id` and `splat_point_nearest` (nearest 3DGS splat center to frame pose) for spatial search
- Trigger: background worker job, fires after `pose_status=success`
- **Docker note:** nerfstudio requires tinycudann and custom CUDA extensions — not pip-installable in a standard PyTorch container. A custom `Dockerfile.nerfstudio` is required with the nerfstudio base image (`dromni/nerfstudio`). Add as optional `docker-compose.override.yml` service so users without GPU can skip it.

**Data Storage**
- **Qdrant:** per-frame vectors — named vector `clip` (OpenCLIP image embedding, for text+image search) + named vector `dino` (DINOv3 embedding, for reranking + uncertainty scoring). Payload: `video_id`, `frame_id`, `timestamp_ms`, `pose` (rotation + translation or null), `gps` (lat/lon/alt or null), `caption` (Florence-2 text), `caption_confidence`, `active_learning_score`, `map_id`.
- **PostgreSQL `frames` table** (new Docker Compose service, `postgres:16`): per-frame dataset metadata — `frame_id`, `video_id`, `mission_id`, `timestamp_ms`, `filepath`, `caption`, `caption_confidence`, `active_learning_score`, `al_tag` (enum: `needs_annotation` / `uncertain` / `novel` / `annotated` / `used_for_training` / `none`), `pose_json`, `gps_json`, `created_at`. Indexed on `mission_id`, `al_tag`, `active_learning_score`. CVAT (v2 annotation) will share this PostgreSQL instance natively — no extra database needed when annotation is added.
- **PostgreSQL — single database, all SQL storage.** SQLite (`jobs.db`, `processed.db`) removed entirely. Tables:
  - `missions(mission_id TEXT PK, video_id TEXT, source_path TEXT, ingest_mode TEXT, started_at TIMESTAMPTZ, completed_at TIMESTAMPTZ, frame_count INT, pose_status TEXT CHECK(IN('pending','success','failed')), map_status TEXT CHECK(IN('pending','success','failed','skipped')))`
  - `jobs` — replaces `job_db.py` / `jobs.db`: same schema as existing SQLite jobs table, migrated to PostgreSQL. `pipeline/job_db.py` updated to use `asyncpg` / `psycopg2`.
  - `processed_files` — replaces `processed_db.py` / `processed.db`: `(sha256 TEXT PK, video_id TEXT, indexed_at TIMESTAMPTZ)`. Dedup check becomes a `SELECT 1 FROM processed_files WHERE sha256=$1`.
- **Filesystem:** existing `DATA_DIR` structure extended — add `maps/{mission_id}/splat.ply`, `sfm/{mission_id}/colmap_sparse/`, `sfm/{mission_id}/poses.json`
- **SQLite removed.** `pipeline/job_db.py` and `pipeline/processed_db.py` rewritten to use PostgreSQL. No `jobs.db` or `processed.db` files on disk.

**Active Tagging** *(new: `pipeline/active_learning.py`)*
- After each mission's inference run, compute per-frame uncertainty score:
  - DINOv3 embedding distance from nearest cluster centroid (k-means on accumulated embeddings, k=20)
  - Florence-2 caption confidence score (from model output logits)
  - Diversity filter: if DINOv3 cosine similarity > 0.97 to an already-tagged frame, skip (no new information)
- Combined score: `0.6 * embedding_distance + 0.4 * (1 - caption_confidence)`
- Write `al_tag` + `active_learning_score` to PostgreSQL `frames` table: top-50 frames per mission → `needs_annotation`; frames with embedding distance > 0.5 from any cluster centroid → `novel`; rest → `none`
- Top-50 threshold configurable via env var `AL_TAG_K` (default 50)
- Mirror `active_learning_score` to Qdrant payload for query-time filtering (retrieve "show uncertain frames near this location")
- **v1 scope:** tagging only. No model retraining in v1.
- AWML optional integration in v2; CVAT annotation in v2 writes back to the same PostgreSQL `frames` table (`al_tag=annotated`)

**Search** *(extend existing)*
- Text query → OpenCLIP text encoder → `clip` Qdrant search → top results; optionally reranked by `dino` score (70/30 blend, existing pattern in `app/services/search.py`)
- Image query → OpenCLIP image encoder → `clip` search + DINOv3 → `dino` reranking
- Results payload includes: pose (rotation+translation), gps, timestamp_ms, caption, map_id, video_id
- New Streamlit page: mission timeline (x=timestamp, y=active_learning_score, query from PostgreSQL) + sparse 3D camera path scatter (x/y/z from pycolmap translation vectors, plotly 3D scatter). Click result → show frame + caption inline.

**MediaMTX Streaming** *(new container)*
- Separate `docker-compose` service
- RTSP/RTMP/WebRTC input → frame capture → feed into existing FastAPI ingest pipeline
- Control panel: stream subscription management
- Can re-stream local video files for testing

### Component-to-Framework Mapping

| Function | Framework | Why |
|---|---|---|
| Frame extraction | existing pipeline | Already adaptive + tuned |
| Embeddings | DINOv3 (existing `dino_model.py`) | Better for robot POV vs. generic CLIP |
| Image captioning | Florence-2 (Microsoft) | Strong open-source VLM, Python, CUDA |
| SfM/poses | pycolmap | Battle-tested, Python API, fast |
| Dense 3D map | nerfstudio splatfacto | Best maintained 3DGS, CUDA, exportable |
| Vector search | Qdrant (existing) | Named vectors, production-grade |
| Dataset metadata | PostgreSQL `frames` table | Reuses existing infra pattern; CVAT (v2) shares same DB natively |
| Annotation | CVAT (v2, shares PostgreSQL) | CVAT deferred — training pipeline not needed until v2 |
| Streaming | MediaMTX | Battle-tested RTSP/RTMP/WebRTC, Go, Docker-friendly. v1: file re-stream only; v1.5: live RTSP/RTMP |
| API | FastAPI (existing) | Preserve existing architecture |
| UI | Streamlit (existing) | Extend with mission timeline + 3D path view |

**Deferred (v2+):** lakeFS, Datumaro, Elasticsearch, AWML, self-supervised training pipeline, knowledge distillation, edge model export, k8s, CVAT full integration, S3/FTP ingestion, live camera streaming (full RTSP ingest vs. file re-streaming)

### Self-Improvement Loop Architecture (designed in v1, activated in v2)

```
Mission video
    ↓
Adaptive frame extraction
    ↓
DINOv3 embeddings + Florence-2 captions
    ↓
Uncertainty scoring → al_tag (needs_annotation / novel) in PostgreSQL frames table
    ↓
[v1] Export frame batch as ZIP (manifest.json + JPEGs) for external annotation
[v2] Human annotation via CVAT (direct integration)
    ↓
[v2] Fine-tune DINOv3 on annotated mission data
    ↓
[v2] Distill to edge model (TorchDistil / knowledge distillation)
    ↓
[v2] Deploy edge model → better uncertainty estimation → better tags
    ↓
Loop
```

The v1 architecture doesn't implement the training or distillation, but it writes the data in the format those steps require. No retrofit needed.

---

## Open Questions

1. **GPS availability:** Does the drone/rover have GPS? If yes, pycolmap can be seeded with GPS priors for faster convergence. If no, pure visual SfM must handle GPS-denied environments — which adds complexity (loop closure, drift correction).
2. **3DGS reconstruction compute:** nerfstudio splatfacto requires ~10-30 min GPU time per mission scene (depends on scene size and keyframe count). Is this acceptable latency for the self-use case?
3. **Florence-2 vs. LLaVA vs. InternVL:** Florence-2 is fast and strong, but for detailed English descriptions of outdoor scenes, InternVL2 may be better. Worth a 1-day comparison on mission footage before committing.
4. **AWML readiness:** Released May 2025, documentation is thin. Decide in v2 whether AWML or a custom PyTorch Lightning training loop is cleaner.

---

## Success Criteria

- [ ] Upload a 5-minute drone mission video → receive searchable results with spatial context within 10 minutes (on GPU hardware)
- [ ] Text query "path with vegetation" returns relevant frames with pose + timestamp
- [ ] Image query returns visually similar frames from the same or different missions
- [ ] Each mission auto-generates a FiftyOne dataset with uncertainty-tagged frames for annotation
- [ ] 3D map (nerfstudio splatfacto) exported and viewable for a completed mission
- [ ] MediaMTX container accepts a re-streamed local file as a test input

---

## Dependencies

- nerfstudio (splatfacto) — requires CUDA GPU; **custom Docker image** (`dromni/nerfstudio` base) needed due to tinycudann + custom CUDA extensions; add as `docker-compose.override.yml` optional service
- Florence-2 — Hugging Face `transformers >= 4.41`, model: `microsoft/Florence-2-large`
- pycolmap — `pip install pycolmap` (no CUDA needed for SfM)
- PostgreSQL 16 — `postgres:16` Docker Compose service; `psycopg2-binary` or `asyncpg` in Python; `DATABASE_URL` env var
- MediaMTX — `bluenviron/mediamtx` Docker image (Go binary); new `docker-compose.yml` service
- DINOv3 — existing `dino_model.py` supports dinov3 via torch.hub; upgrade pretrained weights to dinov3 checkpoint
- OpenCLIP — existing, preserved for cross-modal text↔image search (`clip` named vector)

---

## The Assignment

**Before writing a single line of new code:** Run a real mission. Take your drone/rover out, record 5 minutes of outdoor footage, and load it into the existing selfsuvis pipeline. Note what breaks, what's missing, and what the output is missing that you needed.

That field run is your design validation. It tells you whether pycolmap gives you useful poses on real footage, whether DINOv3 embeddings actually cluster meaningfully for your terrain type, and whether Florence-2 captions are accurate enough to be searchable. You can't know this from the spec.

## What I noticed about how you think

- You came in with a 12-feature spec but immediately identified the core when pushed: "I'm building infra for a robotics/autonomous system." That compression is rare in founders — most defend every feature on the list.
- You pushed back on Codex's challenge ("sparse pose graph is enough") with a specific reason: "3D map is the differentiator for navigation, not just ML." That's domain expertise speaking, not attachment to scope.
- The phrase "picture to text description is essential too" — you added this without being asked. You know what you need; you're not guessing.
- You cited specific research papers (DINOv3, 3DVision conference, V-JEPA 2) and evaluated them, not just listed them. That's a researcher's instinct applied to product design — unusual and valuable here.
