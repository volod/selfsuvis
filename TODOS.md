# TODOS

Deferred work, known issues, and pre-ship blockers.
Format: [Priority] [Effort] — Description

---

## SV-DEMO | Historical transition into the local pipeline

Historical planning section for the workflow that started as the standalone demo
pipeline and was later promoted into the canonical local full-analysis and
learning flow.

Current semantics:
- Use `python main.py --mode local` for the local full-analysis pipeline
- Canonical implementation now lives in `pipeline/workflows/local/`
- References below to `demo`, `pipeline/demo_runner.py`, or `pipeline/demo/`
  are preserved only to connect historical decisions with the current codebase

The resulting local CLI still runs the full perception stack on a directory of
videos and produces per-video reports, a fine-tuned model, 3D maps, and final
statistics without requiring the Docker stack (except optional Qdrant for
vector search).

### ✅ DONE — [DEMO-01] Setup: directory structure + arg parsing + logging
- `video_demo/videos/.gitkeep` placeholder for input videos
- `main.py --mode demo` entry point with `--videos-dir`, `--output-dir`, `--device`,
  `--epochs`, `--batch-size`, `--no-qdrant`, `--no-sfm` flags (logic in `pipeline/demo_runner.py`)
- Colored, timestamped console logger with step banners
- Env vars set before pipeline imports so `settings.*` reflect demo paths

### ✅ DONE — [DEMO-02] Shared model initialization (CLIP + DINO, graceful fallback)
- Load `OpenCLIPEmbedder` (always)
- Load `DINOEmbedder("dinov3_vitb14")` (graceful ImportError / download fallback)
- Attempt Qdrant connection; on failure build `InMemoryStore` backed by numpy cosine search

### ✅ DONE — [DEMO-03] Per-video Step A: frame extraction + metadata JSON
- Reuse `pipeline.media.ffmpeg.extract_frames()`
- Write `{video_dir}/frames_metadata.json` with frame count, fps, duration
- Log: extracted N frames in T seconds

### ✅ DONE — [DEMO-04] Per-video Step B: index frames into vector store
- Reuse `pipeline.workflows.indexer.VideoIndexer.index_video()`; progress callback logs to console
- Graceful skip if Qdrant unavailable (embeddings stored in InMemoryStore for search steps)

### ✅ DONE — [DEMO-05] Per-video Step C: base model transformation test → `base_search.md`
- Pick query frame (highest visual information: middle of video)
- Embed with base CLIP + DINO; search for top-5 nearest neighbours
- Write `{video_dir}/base_search.md` with query image path, match table, scores
- Do NOT overwrite if file already exists

### ✅ DONE — [DEMO-06] Per-video Step D: SSL DINOv3 fine-tuning → `finetune_stats.md`
- Reuse `pipeline.training.ssl.FinetuneConfig` + `run_finetune()`
- `frames_dir = settings.FRAMES_DIR` (parent of per-video subdirs — TemporalPairDataset convention)
- Fall back to `approach="augment"` when video has < 2*batch_size frames
- Write `{video_dir}/finetune_stats.md` with loss curve, best epoch, checkpoint path

### ✅ DONE — [DEMO-07] Per-video Step E: ONNX export + gallery build → `edge_models/`
- Hot-load fine-tuned checkpoint into `DINOEmbedder.load_backbone_checkpoint()`
- Reuse `scripts/export_onnx._export_onnx()` for ONNX trace-export
- Reuse `pipeline.training.edge_inference.build_gallery()` with frames grouped as one pseudo-class
- Save `dino_demo.onnx` and `gallery.npz` into `{video_dir}/edge_models/`

### ✅ DONE — [DEMO-08] Per-video Step F: fine-tuned model transformation test → `finetuned_search.md`
- Re-embed query frame with fine-tuned DINOEmbedder; search for top-5 matches
- Write `{video_dir}/finetuned_search.md` (separate file; never overwrites base_search.md)

### ✅ DONE — [DEMO-09] Per-video Step G: comparison + video-to-text description → `comparison.md`
- Compare base vs fine-tuned top-5 results (overlap, score delta)
- Model stats: checkpoint size on disk, mean DINO inference time (ms/frame) for both models
- Video-to-text: average CLIP frame embedding → cosine similarity against 12 text prompts → top-3 descriptions
- Write `{video_dir}/comparison.md` and echo summary to console log

### ✅ DONE — [DEMO-10] Per-video Step H: 3D map creation → `3d_map/`
- Reuse `pipeline.mapping.sfm.run_sfm()` (pycolmap optional)
- Fallback when pycolmap absent: PCA(3) of DINO frame embeddings → synthetic point cloud
- Save `sparse_map.npz` (points + colours) and `map_stats.json`

### ✅ DONE — [DEMO-11] 3D map viewer (matplotlib, one window per video, shutdown button)
- After all videos processed: open one matplotlib 3D figure per video
- "Close Viewer" button calls `plt.close(fig)` for that window
- `plt.show()` blocks until all viewers are closed; then pipeline continues to final stats

### ✅ DONE — [DEMO-12] Final statistics → `final_stats.md` + console summary
- Aggregate per-video: frame count, index time, finetune loss, model size, SfM poses
- Print formatted table to console; write `video_demo/output/final_stats.md`

### ✅ DONE — [DEMO-13] README.md "Demo" section
- Prerequisites (ffmpeg, qdrant optional, sample video download)
- Step-by-step: `make venv` → place videos → `python main.py --mode demo`
- Description of every artifact produced

---

---

## P1 — Blockers (must resolve before shipping)

### ✅ [P1][S] Validate Qdrant 2D GPS range query performance — DONE
**Implemented:** `scripts/benchmark_qdrant_gps.py` — inserts 50K synthetic GPS points, benchmarks
2D (lat+lon payload filter) vs 1D (lat-only + Python post-filter) queries, reports p50/p95/p99 latencies,
recommends `GPS_FILTER_2D=true/false` based on p99 < 200ms SLA. Run: `python scripts/benchmark_qdrant_gps.py`

---

### ✅ [P1][M] GPU memory budget — profile Florence-2 + CLIP + DINOv3 — DONE
**Implemented:** `scripts/profile_gpu_memory.py` — profiles each model individually and in combination
on RTX 4060 Ti (15916 MiB). Results written to `docs/gpu_memory_profile.md`.
**Results (RTX 4060 Ti, FP16):**
- CLIP ViT-B-16: load=304 MiB, peak=472 MiB
- DINOv3 ViT-B/14: load=174 MiB, peak=245 MiB
- Florence-2-large: load=1484 MiB, peak=2722 MiB
- All three simultaneously: 3922 MiB used, 9531 MiB free
**Verdict:** All three worker models fit simultaneously on RTX 4060 Ti (~25% VRAM used).
No sequential loading required. nerfstudio runs in a separate container (unaffected).

---

## P2 — Important (resolve before or during v2)

### ✅ [P2][S] nginx CORS — add Access-Control-Allow-Origin for SuperSplat .ply fetch — DONE
**Implemented:** `docker/nginx.conf` with CORS headers on `/static/maps/`; nginx + postgres + mediamtx
services added to `docker/docker-compose.yml`.

### ✅ [P2][S] Upgrade Streamlit to 1.37+ for @st.fragment 3DGS polling — DONE
**Implemented:** `requirements/requirements_prod.txt` updated to `streamlit>=1.37.0`;
`ui/app.py` fixed `use_column_width=True` → `use_container_width=True`.

### ✅ [P2][M] Phase 2 global map — 3DGS ICP fusion (Open3D) — DONE
**Implemented:**
- **Analysis**: open3d 0.19.0 has no hard version conflicts with existing deps (numpy, pydantic v2,
  torch, pillow). Separate container chosen over Dockerfile.worker to avoid ~1.1GB of unused
  visualization deps (dash, flask, ipython, matplotlib, pandas) in the worker image.
- `docker/Dockerfile.mapper` — python:3.10-slim + open3d>=0.18 + plyfile; ~1.5GB, CPU-only.
- `docker/docker-compose.override.yml` — `mapper` service at http://localhost:8100; auto-loaded by `make up`.
- `requirements/requirements_mapper.txt` — minimal ICP deps only.
- `pipeline/config.py` — `MAPPER_API_URL=http://mapper:8000`.
- `pipeline/icp_fusion.py`:
  - `register_splats(source, target, source_meta, target_meta, ...)` — Point-to-Point ICP via open3d;
    Phase-1 GPS-ENU as initial alignment; auto voxel downsampling; returns `IcpResult`.
  - `check_overlap(source_meta, target_meta, radius_a_m, radius_b_m)` — GPS pre-check before ICP.
  - `_initial_transform_from_gps()` — SE(3) translation from GPS ENU origins.
  - `_voxel_size_for(n)` — auto voxel size heuristic.
- `mapper/main.py` — FastAPI wrapper: `POST /fuse`, `POST /check_overlap`, `GET /health`.
- `tests/unit/test_icp_fusion.py` — 18 tests (13 always-pass + 5 skip if open3d absent); all passing.

**Phase 2 worker integration — DONE:**
- `pipeline/mapper.py`: `_call_icp_fuse(source, target)` — POST to mapper service; soft-skips
  if mapper unreachable (ConnectionError → None). `run_mapper` now accepts
  `target_splat_paths` and calls ICP for each successful scene; returns `icp_results` list.
- `pipeline/global_map_db.py` — asyncpg helpers:
  `get_or_create_global_map`, `get_global_map_splats`, `register_mission`,
  `update_global_map_splat`, `get_global_map_by_id`, `list_mission_registrations`.
- `tests/unit/test_mapper_icp.py` — 22 tests (global_map_db + _call_icp_fuse + run_mapper ICP
  integration); all passing.
**Fused splat.ply output — DONE:**
- `pipeline/splat_io.py`: `apply_transform_to_splat(path_in, T, path_out)` — applies SE(3) to
  positions (R·p+t) and Gaussian rotation quaternions (q_align⊗q_gs); scales/opacity/SH
  copied unchanged (SH DC is rotationally invariant; SH degree 1–3 Wigner-D rotation deferred).
  `merge_splats(paths, path_out)` — concatenates N splat.ply files into one.
- `pipeline/mapper.py`: `_fuse_splat_files` — called after converged ICP; transforms source
  into target frame, merges, writes `<scene_dir>/fused.ply`; temp file cleaned up on error/success.
  `icp_results` now includes `"fused_splat"` path.
- `tests/unit/test_splat_transform.py` — 27 tests (quat math + apply_transform + merge +
  _fuse_splat_files); all passing.
**Worker wiring — DONE:**
- `scripts/migrate_postgres.py`: `ALTER TABLE missions ADD COLUMN IF NOT EXISTS splat_path TEXT` —
  required for `get_global_map_splats` JOIN to find registered mission splats.
- `pipeline/global_map_db.py`: `update_mission_splat_path(conn, mission_id, splat_path)` —
  sets `missions.splat_path` after nerfstudio produces a splat; enables discovery by future missions.
- `worker/main.py` `_db_and_map` (three fixes):
  1. Calls `update_mission_splat_path` after every successful 3DGS run.
  2. Calls `update_global_map_splat(conn, global_map_id, fused_path)` when ICP produces a fused.ply.
  3. Bootstrap-registers the first mission at a site (or non-converged ICP) with an identity
     transform so `get_global_map_splats` can return its splat to the next mission as an ICP target.
- Synthetic splat.ply test data: 100-Gaussian and 50K-Gaussian PLY files via `write_splat_from_arrays`.
- `tests/unit/test_worker_global_map.py`: 35 tests covering all three fixes, discovery chain,
  bootstrap, ICP converged/not-converged, mapper skipped, fused_splat=None, multiple targets;
  all passing.
**Depends on:** nerfstudio splatfacto producing real splat.ply from actual missions (for production use).

### ✅ [P2][M] tx/ty/tz-only robot query path (no GPS) — DONE
**Implemented:**
- `app/routers/robot.py`: `PoseQuery.lat`/`lon` are now Optional; `model_validator` requires
  either (lat+lon) or (tx+ty+tz) — missing both → HTTP 422.
- ENU filter path: Qdrant `enu.tx`/`enu.ty` 2D bbox filter + Python 3D ENU sphere post-filter.
- `_enu_distance_m` helper for 3D Euclidean distance in ENU frame.
- `PoseQueryResponse` extended with `query_tx`, `query_ty`, `query_tz` fields.
- `filter_strategy` is `"enu+python"` for the ENU path.
- `tests/unit/test_robot_api.py`: 9 new tests (ENU distance, ENU query, 3D postfilter,
  422 for missing coords, 422 for partial ENU); 23 total, all passing.
**GPS payload in Qdrant — DONE:**
- `pipeline/indexer.py`: `index_video` now accepts `mission_id` param; calls `extract_gps` +
  `gps_to_enu` to pre-compute GPS+ENU for every extracted frame; stores `gps:{lat,lon,alt}`,
  `enu:{tx,ty,tz}`, `mission_id` in Qdrant frame payloads (and `mission_id` in tile payloads).
  GPS extraction is a soft dependency — failures are logged and indexing continues without GPS.
**Pass A wired into worker — DONE:**
- `worker/main.py`: `_run_pass_a` helper — after `index_video`, runs `run_sfm` →
  `register_mission_gps` → `get_global_map_splats` → `run_mapper(target_splat_paths=…)` →
  `register_mission` for converged ICP results. All steps are soft-skip on ImportError /
  ConnectionError / any exception — worker never fails due to optional Pass A.
**Depends on:** Phase 2 global map (ICP fusion) for ENU queries to return meaningful results

### ✅ [P2][S] Streamlit admin page — worker status, queue depth, al_tag distribution — DONE
**Implemented:** `app/routers/admin.py` (`GET /admin/stats`); Admin tab in `ui/app.py` with
worker badge, queue depth metric, job status breakdown, al_tag bar chart.

### ✅ [P2][M] CVAT annotation integration — write al_tag=annotated from CVAT feedback — DONE
**Implemented:**
- `docker/docker-compose.cvat.yml`: CVAT 2.16.2 (server, UI, workers, OPA, postgres, redis).
  `make cvat-up` → http://localhost:8091. First run: `make cvat-admin` to create superuser.
- `app/routers/cvat.py`:
  - `POST /webhook/cvat` — HMAC-SHA256 verified (X-Hook-Secret); handles `update:job` /
    `update:task` with `state=completed`; sets `al_tag='annotated'` via cvat_tasks lookup.
  - `GET /admin/cvat/frames` — frames with `al_tag=needs_annotation|novel|any` for task creation.
  - `POST /admin/cvat/task` — registers cvat_task_id → frame_id mapping.
- `scripts/migrate_postgres.py`: Added `cvat_tasks (cvat_task_id, frame_id PK)` table.
- `pipeline/config.py`: Added `CVAT_URL`, `CVAT_WEBHOOK_SECRET`.
- `tests/unit/test_cvat_webhook.py`: 17 tests, all passing.
**Annotation workflow:** GET /admin/cvat/frames → create CVAT task → POST /admin/cvat/task →
annotate in CVAT → job completed → webhook fires → al_tag='annotated'.

### ✅ [P2][M] Self-supervised domain adaptation — DINOv3 fine-tuning on mission frames — DONE
**Implemented:**
- `pipeline/ssl_finetune.py`:
  - `NTXentLoss` — InfoNCE / NT-Xent contrastive loss (SimCLR formulation) with temperature scaling.
  - `AugmentPairDataset` — positive pairs via two independent random augmentations of the same frame;
    works with any unordered frame collection.
  - `TemporalPairDataset` — positive pairs from consecutive frames within each `{video_id}/` subdir;
    max_gap configurable (default 3). Single-frame dirs skipped automatically.
  - `ProjectionHead` — two-layer MLP projection head (L2-normalised output); discarded at inference.
  - `DINOFineTuner` — wraps DINOv3/DINOv2 backbone; freezes first N transformer blocks (default 10/12)
    to prevent catastrophic forgetting; fine-tunes last 2 blocks + projection head (~14 M params).
  - `run_finetune(cfg)` — training loop with AdamW + CosineAnnealingLR; saves per-epoch checkpoints
    (`dino_ssl_{epoch:03d}.pt`) and best checkpoint (`dino_ssl_best.pt`).
  - `config_from_settings()` — builds FinetuneConfig from env vars / pipeline.core.config.
- `scripts/finetune_dino.py` — CLI entry point with all config as flags; `--approach temporal|augment`.
- `pipeline/config.py`: added `SSL_CHECKPOINT_DIR`, `SSL_FINETUNE_EPOCHS`, `SSL_FINETUNE_LR`,
  `SSL_FINETUNE_BATCH_SIZE`, `SSL_FINETUNE_FREEZE_BLOCKS`, `SSL_FINETUNE_TEMPERATURE`,
  `SSL_FINETUNE_APPROACH`, `DINO_CHECKPOINT` env vars.
- `models/dino_model.py`: `DINOEmbedder._load_model` now checks `DINO_CHECKPOINT` and loads
  fine-tuned weights automatically when the file exists.
- `tests/unit/test_ssl_finetune.py`: 32 tests (NTXentLoss, ProjectionHead, datasets, freeze
  strategy, checkpoint save/load, E2E smoke tests, config wiring); all passing.

**Usage:**
```bash
# Fine-tune for 10 epochs using temporal pairs (GPU recommended)
python scripts/finetune_dino.py --frames-dir data/frames --output-dir data/checkpoints

# CPU smoke-test with augmentation pairs
python scripts/finetune_dino.py --approach augment --epochs 2 --batch-size 8 --device cpu

# Deploy fine-tuned model
export DINO_CHECKPOINT=data/checkpoints/dino_ssl_best.pt
make up   # DINOEmbedder picks up the checkpoint automatically
```

### ✅ [P2][M] Edge model hydration — export fine-tuned DINOv3 for on-device object identification — DONE
**What:** Export the fine-tuned DINOv3 backbone to ONNX, attach a lightweight mission-object
classifier head, quantize to INT8, and ship a self-contained inference package that runs on
the robot's edge compute (Jetson Orin, Hailo-8, or CPU-only ARM SBC) to identify
mission-typical objects in real time.

**Pipeline:**
1. **ONNX export** (`scripts/export_onnx.py`) — load `dino_ssl_best.pt`, trace through
   `torch.onnx.export`, validate output parity vs PyTorch forward pass.
2. **Prototype classifier head** — a cosine-similarity nearest-neighbour classifier over
   a small gallery of mission-typical object embeddings (no GPU, no retraining required).
   Gallery built from a handful of representative frames per category, stored as an NPZ file
   (`data/gallery/{category}.npz`). Categories: user-defined (e.g. "vehicle", "signage",
   "barrier", "personnel", "terrain").
3. **INT8 quantization** — static quantization via ONNX Runtime `quantize_static` using
   a calibration dataset of ~500 mission frames. Target: ≤50 ms/frame on Jetson Orin NX.
4. **Edge inference wrapper** (`pipeline/edge_inference.py`) — `EdgeClassifier`:
   loads quantized ONNX model + gallery NPZ; exposes `classify(image_pil) → List[(label, score)]`;
   no PyTorch dependency at inference time (ONNX Runtime only).
5. **Calibration script** (`scripts/build_gallery.py`) — scans `data/frames/` for
   representative frames per category (user-supplied label → frame-path mapping or
   interactive selection), embeds them, saves to `data/gallery/`.

**Why:** The fine-tuned embeddings capture outdoor-autonomy-specific features. Exporting to
ONNX + INT8 quantization makes embedding inference viable on edge hardware without CUDA.
The nearest-neighbour head requires only a handful of labelled examples per category
(few-shot, no annotation pipeline), so it works before CVAT annotations are available.

**Edge deployment:**
```bash
# 1. Export
python scripts/export_onnx.py --checkpoint data/checkpoints/dino_ssl_best.pt \
    --output data/models/dino_edge.onnx

# 2. Quantize (needs ~500 calibration frames, runs on dev machine)
python scripts/export_onnx.py --quantize --calibration-dir data/frames \
    --output data/models/dino_edge_int8.onnx

# 3. Build gallery (run on dev machine, ship gallery NPZ to robot)
python scripts/build_gallery.py --frames-dir data/frames \
    --labels vehicle:data/frames/vid1/frame_0010.jpg,... \
    --output data/gallery/mission_objects.npz

# 4. On robot:
from pipeline.training.edge_inference import EdgeClassifier
clf = EdgeClassifier("dino_edge_int8.onnx", "mission_objects.npz")
labels = clf.classify(frame_pil)   # [(label, score), ...]
```

**Effort:** M (human: ~1 week / CC: ~30 min)
**Depends on:** Self-supervised domain adaptation (✅ done — `dino_ssl_best.pt` required);
`onnxruntime` (CPU) or `onnxruntime-gpu` (Jetson); `onnxruntime-tools` for quantization.

**Implemented:**
- `scripts/export_onnx.py` — torch → ONNX export (`torch.onnx.export`, opset 17, dynamic batch);
  `--validate` flag runs PyTorch ↔ ONNX parity check (max abs diff < 1e-3);
  `--quantize` + `--calibration-dir` runs static INT8 quantization via `onnxruntime.quantization`.
- `pipeline/edge_inference.py` — `EdgeClassifier`: loads (quantized) ONNX + gallery NPZ;
  `embed(img)` → L2-normalised (D,) float32; `classify(img)` → `[(label, score), ...]` sorted
  cosine-sim descending; `from_torch()` classmethod for dev/testing without ONNX.
  `build_gallery()` — embeds representative frames per label, saves `embeddings/labels/label_names`
  NPZ; works with ONNX or PyTorch backbone; validates all paths upfront.
- `scripts/build_gallery.py` — CLI: `--onnx` or `--checkpoint`; `--labels label:path,...` or
  `--labels-file` (JSON/YAML); outputs NPZ to `--output`.
- `pipeline/config.py`: `EDGE_MODELS_DIR`, `EDGE_GALLERY_DIR`, `EDGE_ONNX_PATH`,
  `EDGE_GALLERY_PATH`, `EDGE_TOP_K` env vars.
- `tests/unit/test_edge_inference.py`: 30 tests (preprocessing, build_gallery, EdgeClassifier,
  from_torch, integration smoke); all passing.

### ✅ [P3][XL] Supervised fine-tuning on CVAT-annotated frames — DONE
**What:** Use CVAT-annotated frames (al_tag=annotated) to teach DINOv3 semantic category
boundaries via Supervised Contrastive Loss (SupCon, Khosla et al. NeurIPS 2020).
**Why:** Hard negative mining with semantic labels and class boundary learning cannot be
derived from self-supervised objectives alone. Completes the active learning loop.
**Implemented:**
- `pipeline/supervised_finetune.py`:
  - `SupConLoss` — Supervised Contrastive Loss (eq. 2, Khosla 2020). Pulls together embeddings
    with matching labels, pushes apart embeddings from different classes. Returns 0.0 when no
    anchor has any positive (graceful handling of single-class batches).
  - `CvatAnnotationParser` — parses CVAT XML 1.1; majority-vote label per image when multiple
    boxes; basename matching for flexible directory structures.
  - `AnnotatedFrameDataset` — intersection of annotated frames + frames found on disk; returns
    `(view1, view2, label_idx)` for SupCon training or `(view1, label_idx)` for cross-entropy.
  - `SupervisedFineTuner` — DINOv3 backbone (same freeze strategy as `DINOFineTuner`, default
    freeze_blocks=8) + two-layer MLP projection head; accepts optional `ssl_checkpoint` to
    warm-start from a domain-adapted backbone before supervised fine-tuning.
  - `SupervisedFinetuneConfig` — mirrors `FinetuneConfig`; adds `cvat_xml_path`, `ssl_checkpoint`.
  - `run_supervised_finetune(cfg)` — training loop: two views per sample → cat → `SupConLoss`;
    AdamW + CosineAnnealingLR; per-epoch + best checkpoint saves (`dino_sup_best.pt`).
  - `config_from_settings()` — reads `SUP_FINETUNE_*` env vars; inherits `DINO_CHECKPOINT` as
    `ssl_checkpoint` warm-start.
- `scripts/supervised_finetune_dino.py` — CLI entry point; `--ssl-checkpoint` for warm-start.
- `pipeline/config.py`: added `SUP_CHECKPOINT_DIR`, `SUP_FINETUNE_EPOCHS`, `SUP_FINETUNE_LR`,
  `SUP_FINETUNE_BATCH_SIZE`, `SUP_FINETUNE_FREEZE_BLOCKS`, `SUP_FINETUNE_TEMPERATURE`.
- `tests/unit/test_supervised_finetune.py`: 26 tests (SupConLoss maths, gradient flow, CvatAnnotationParser,
  AnnotatedFrameDataset, config defaults, E2E training loop stub); all passing.

**Usage:**
```bash
# Fine-tune supervised from exported CVAT XML and local frames
python scripts/supervised_finetune_dino.py \
  --frames-dir data_test/cvat_frames \
  --cvat-xml   data_test/cvat_annotations.xml \
    --output-dir data/checkpoints/supervised \
    --ssl-checkpoint data/checkpoints/dino_ssl_best.pt

# 3. Use supervised checkpoint (same as SSL)
export DINO_CHECKPOINT=data/checkpoints/supervised/dino_sup_best.pt
make up
```
**Depends on:** CVAT annotation integration (✅ done); self-supervised domain adaptation (✅ done).

---

## P3 — Nice to have

### ✅ [P3][S] Stale diagram audit — docs/ — DONE
**Updated:**
- `docs/pipeline.md`: SfM → scene chunking output paths, `make up` mentions PostgreSQL/nginx/MediaMTX
- `docs/data_layout.md`: Added `maps/`, `reports/`, `postgres/` dirs; scene-{N} paths; PostgreSQL note; removed stale `jobs.db`/`processed.db` entries
- `docs/architecture.md`: Scene chunking path in indexing flow diagram

### ✅ [P3][M] k-means online/incremental clustering for scale (>50 missions) — DONE
**Implemented:**
- `pipeline/active_learning.py`: Added `fit_kmeans(embeddings, n_clusters, batch_threshold)` —
  auto-selects `KMeans` below threshold, `MiniBatchKMeans` at or above it.
  Also added `dino_distances_from_centroids(embeddings, centroids)` for cosine distance computation.
- `pipeline/config.py`: Added `KMEANS_BATCH_THRESHOLD=25_000` env var.
- `requirements/requirements_prod.txt`: Added `scikit-learn>=1.4.0`.
- `tests/unit/test_active_learning.py`: 9 new tests (22 total, all passing).

### ✅ [P3][S] Pre-flight robot map cache export — DONE
**Implemented:**
- `pipeline/map_cache.py`: `build_map_cache(qdrant_store, mission_ids, lat/lon bbox)` — scrolls
  Qdrant for all `type=frame` points and packs them into a compressed NPZ file:
  `clip_vectors` (N,D) float32, `gps` (N,3), `enu` (N,3), `t_sec` (N,), `meta_json` (JSON bytes
  with mission_id/frame_path/robot_id per frame). NaN for missing GPS/ENU.
- `app/routers/admin.py`: `GET /admin/export/map-cache` — streams NPZ as
  `application/octet-stream` attachment. Optional query params: `mission_ids` (comma-sep),
  `lat_min`/`lat_max`/`lon_min`/`lon_max` for GPS bbox.
- `tests/unit/test_map_cache.py`: 15 tests (vector packing, NaN for missing GPS/ENU,
  filter passthrough, pagination, frame-without-clip skip, NPZ magic bytes); all passing.

**Robot usage:**
```python
import numpy as np, json
cache = np.load("map_cache.npz", allow_pickle=False)
vecs = cache["clip_vectors"]   # (N, D) float32 — cosine search
gps  = cache["gps"]            # (N, 3) [lat, lon, alt]
enu  = cache["enu"]            # (N, 3) [tx, ty, tz] metres
meta = json.loads(bytes(cache["meta_json"]).decode())
```

### ✅ [P3][S] Live camera streaming — full RTSP ingest (v1.5) — DONE
**Implemented:**
- `pipeline/rtsp_ingest.py`:
  - `validate_rtsp_url(url)` — allows only `rtsp://` / `rtmp://` schemes; rejects credentials
    in URL; rejects private/loopback IPs (unless `ALLOW_PRIVATE_URLS=true`).
  - `record_rtsp(url, output_path, duration_sec, timeout_sec)` — ffmpeg subprocess:
    `-rtsp_transport tcp -c copy -t {duration_sec}`; caps at `RTSP_MAX_DURATION_SEC` (default 3600s).
- `pipeline/config.py`: `RTSP_MAX_DURATION_SEC` env var (default `3600`).
- `app/routers/index.py`: `POST /index/rtsp` — accepts `stream_url`, `mission_id`,
  `duration_sec`, `enable_tiles`; validates URL via `validate_rtsp_url`; enqueues job
  with `ingest_mode=rtsp`.
- `worker/main.py`: handles `ingest_mode=rtsp` — calls `record_rtsp()` instead of
  `download_url()`.
- `tests/unit/test_rtsp_ingest.py`: 17 tests (URL validation, private IP rejection,
  ALLOW_PRIVATE_URLS flag, ffmpeg args, duration capping, error propagation); all passing.

**Usage:**
```bash
# Ingest 5 minutes from a MediaMTX re-stream
curl -X POST /index/rtsp \
  -F stream_url=rtsp://mediamtx:8554/test \
  -F mission_id=preflight_2026_03_24 \
  -F duration_sec=300
```

### ✅ [P3][M] Multi-robot shared world model — DONE
**Implemented:**
- `pipeline/config.py`: `ROBOT_ID` env var (default `"robot_0"`) — identifies the robot
  running this worker instance.
- `scripts/migrate_postgres.py`: `ALTER TABLE missions ADD COLUMN IF NOT EXISTS robot_id TEXT NOT NULL DEFAULT 'robot_0'`; index on `robot_id`. Idempotent.
- `pipeline/indexer.py`: `index_video` + `_build_frame_point` + `_index_tiles` all accept
  `robot_id`; stored in every Qdrant frame and tile payload as `robot_id`.
- `worker/main.py`: passes `robot_id=settings.ROBOT_ID` to `index_video`.
- `app/routers/robot.py`: `PoseQuery` now has `robot_ids: Optional[List[str]]`; when
  provided, adds `MatchAny` condition on `robot_id` payload key — works for GPS, ENU, and
  2D filter strategies.
- `app/routers/admin.py`: `GET /admin/robots` — returns distinct `robot_id` values from
  the `missions` table (empty list if DB unavailable or column not yet migrated).
- `tests/unit/test_robot_api.py`: 3 new tests (robot_ids in filter, omitted = no filter,
  ENU path + robot_ids); 26 total, all passing.

### ✅ [P3][L] Multi-site ENU (>50km or disconnected sites) — DONE
**Implemented:**
- `pipeline/global_map_db.py`: Added `get_global_map_origin(conn, gmap_id)` → `(lat, lon, alt)`;
  `list_global_maps(conn)` → all site rows. Fixed `dlon` proximity check to use
  `math.cos(math.radians(origin_lat))` instead of hardcoded 0.7 factor.
- `pipeline/indexer.py`: `index_video` now accepts `site_enu_origin` and `global_map_id`
  params. When `site_enu_origin` is provided, all frame ENU coords are relative to the
  site's canonical origin (not each mission's first-frame local origin). `global_map_id`
  stored in every frame and tile Qdrant payload.
- `worker/main.py`: Added `_resolve_site_origin(video_path, logger)` — extracts first GPS
  fix, calls `get_or_create_global_map` + `get_global_map_origin` before `index_video`;
  degrades gracefully (returns `None, None`) on GPS/DB failure. `_run_pass_a` receives
  pre-resolved `global_map_id` to avoid duplicate DB lookup. `main()` wires everything
  together.
- `app/routers/robot.py`: `PoseQuery.global_map_id: Optional[int]` — when set, adds
  `MatchValue` filter on `global_map_id` payload field for GPS, ENU, and 2D strategies.
  `PoseQueryResponse` includes `global_map_id`.
- `app/routers/admin.py`: `GET /admin/global-maps` — lists all site rows.
- `tests/unit/test_multisite_enu.py`: 30 tests (proximity math, get_global_map_origin,
  list_global_maps, robot API filter, site_enu_origin override, _resolve_site_origin); all passing.

**How multi-site works:**
- Site A missions → `global_map` row 1 with ENU origin at (lat_A, lon_A, alt_A)
- Site B missions → `global_map` row 2 with ENU origin at (lat_B, lon_B, alt_B)
- Qdrant `enu.tx/ty/tz` for all missions at Site A are in the same frame → robot ENU queries
  work correctly across missions at the same site.
- Robot specifies `global_map_id` in pose query to scope search to the correct site.

---

## Added by /plan-eng-review (2026-03-23)

### ✅ [P2][M] 3DGS scene chunking for long missions — DONE
**Implemented:**
- `pipeline/sfm.py`: `_run_pycolmap` now returns all connected components (sorted by size);
  `run_sfm` returns `{frames, scene_count}` with `scene_index` per frame.
- `pipeline/mapper.py`: `run_mapper` groups frames by `scene_index`, trains one splatfacto
  job per component ≥ MIN_FRAMES_FOR_3DGS; outputs `maps/{mission_id}/scene-{N}/splat.ply`.
  Returns `splat_paths` list.
- `app/routers/admin.py`: `GET /admin/missions` discovers splat.ply paths from filesystem.
- `ui/app.py`: 3DGS Scene Viewer in Admin tab with mission selector + scene selector
  (shows when `scene_count > 1`); loads splat.ply URL into SuperSplat iframe.

### ✅ [P2][S] SFM_FPS env var — dense frame extraction for pycolmap — DONE
**Implemented:** `pipeline/sfm.py` uses `settings.SFM_FPS` (default 2.0fps) for dense extraction
into `frames/{video_id}_sfm/` separate from sparse search keyframes in `frames/{video_id}/`.

### ✅ [P2][S] pipeline/gps_extractor.py — structured GPS metadata extraction — DONE
**Implemented:** `pipeline/gps_extractor.py` — SRT sidecar → ffprobe ISO 6709 atom → GPMF
detection → null fallback; linear interpolation to frame timestamps; `GPS_SIDECAR_PATH` override.

### ✅ [P2][S] PYCOLMAP_CAMERA_MODEL env var — DONE
**Implemented:** `pipeline/sfm.py` reads `settings.PYCOLMAP_CAMERA_MODEL`; `pipeline/config.py`
exposes `PYCOLMAP_CAMERA_MODEL="SIMPLE_RADIAL"` with validation against allowed model set.

### ✅ [P2][M] Phase 1 GPS-to-ENU registration — pipeline/gps_registration.py — DONE
**Implemented:**
- `pipeline/gps_registration.py`: WGS-84 geodetic → ECEF → ENU (ROS REP-103 convention).
  `gps_to_enu()`, `register_mission_gps()` (ENU origin = first GPS-valid frame; full registration
  for pose_status=success frames; GPS-only for others), `build_registration_transform()` (4×4 SE(3)).
- `tests/unit/test_gps_registration.py`: 19 tests, all passing.

### ✅ [P2][M] Robot advisory API — app/routers/robot.py POST /query/pose — DONE
**Implemented:**
- `app/routers/robot.py`: GPS bbox filter (1D lat-only + Python lon post-filter by default;
  2D with GPS_FILTER_2D=true), results sorted by distance_m ascending, 503 on Qdrant failure.
  PoseQuery, PoseMatch, PoseQueryResponse Pydantic models; radius_m validation (ge=1.0, le=5000.0).
- `app/main.py`: robot_router registered.
- `tests/unit/test_robot_api.py`: 14 tests, all passing.

---

## Active Learning Loop Closure — Deferred items (from /plan-ceo-review 2026-03-25)

### ✅ [P1][S] Fix CVAT label fetch-back: key on frame_id, not basename — DONE
**What:** The planned CVAT label fetch-back matches frames by `basename(frame_path)`, which silently assigns labels to wrong frames across missions (`frame_0042.jpg` exists in every mission). The CVAT webhook handler already has selfsuvis `frame_id` from the `cvat_tasks` table — use `frame_id` as the key, not filename.
**Why:** Multi-mission deployments will mislabel frames with no error. Silent correctness bug.
**Implemented:** `app/routers/cvat.py` — `_frames_for_cvat_task(task_id)` returns `frame_id` values
via `SELECT frame_id FROM cvat_tasks WHERE cvat_task_id = $1`; `_mark_frames_annotated(frame_ids)`
uses `WHERE id = ANY($1::text[])` — no basename matching anywhere.
**Effort:** S (human: 2h / CC: 5min)
**Priority:** P1 — fix before shipping the auto-trigger pipeline
**Depends on:** `cvat_label` schema migration (ALTER TABLE frames ADD COLUMN cvat_label TEXT)

---

### ✅ [P1][S] Reconcile `from_db` with `SupervisedFinetuneConfig.cvat_xml_path` — DONE
**What:** `SupervisedFinetuneConfig` currently has a required `cvat_xml_path: str` field. The worker job handler (`handle_supervised_finetune_job`) calls `run_supervised_finetune` via the DB path — without a CVAT XML file. These two call paths are not reconciled in the implementation contract.
**Implemented:** `pipeline/supervised_finetune.py` — `SupervisedFinetuneConfig.cvat_xml_path: Optional[str] = None`;
`run_supervised_finetune(cfg)`: when `cvat_xml_path` is None, calls `AnnotatedFrameDataset.from_db(transform, two_views=True, mission_id=cfg.mission_id)` to load labelled frames from the `frames` table directly.
**Effort:** S (human: 1h / CC: 5min)
**Priority:** P1 — must be in implementation contract before writing code
**Depends on:** CEO plan active learning loop closure

---

### ✅ [P2][S] Hot-reload endpoint: add atomic reference swap or drain for in-flight inference — DONE
**What:** The current plan has `asyncio.Lock` serialising concurrent reloads but inference calls do NOT hold the lock. A reload mid-batch will silently use old weights for some frames and new weights for others in the same request.
**Implemented:** `app/routers/admin.py` `POST /admin/reload-model` — uses GIL-atomic reference assignment:
`state.dino_model` is reassigned to the new `DINOEmbedder` object after loading; in-flight inference
calls hold their own captured reference to the old object and complete normally. `dino_model_lock`
(from `app/state.py`) gates concurrent reload attempts only (not inference). Documented in endpoint
docstring. Returns 409 if a reload is already in progress.
**Effort:** S (human: 2h / CC: 10min)
**Priority:** P2 — correctness bug but corruption window is tiny in practice

---

### ✅ [P2][S] Wrap `_maybe_trigger_finetune` enqueue in try/except to prevent CVAT retry storms — DONE
**What:** asyncpg errors in `_maybe_trigger_finetune` propagate to the webhook handler, returning 500 to CVAT. CVAT retries the webhook, potentially enqueueing duplicate fine-tune jobs despite the SQL dedup guard (race between retry and the job being picked up by the worker).
**Implemented:** `app/routers/cvat.py` — `_maybe_trigger_finetune()` wraps the entire function body in
`try/except Exception as exc` with `logger.warning("_maybe_trigger_finetune failed (non-fatal): %s", exc)`.
The webhook handler calls it without `await`-level exception propagation; CVAT always receives a 200 response.
**Effort:** S (human: 30min / CC: 2min)
**Priority:** P2

---

### ✅ [P2][S] Add per-batch no-positive guard in SupCon training loop — DONE
**What:** With 500 frames / 6 classes / batch_size=16, ~8% of batches have zero positive pairs. `SupConLoss.forward` returns `tensor(0.0)`, optimizer takes a zero gradient step, scheduler still advances. The loss curve looks like convergence but training has stalled.
**Implemented:** `pipeline/supervised_finetune.py` training loop — after computing loss, checks
`if not valid.any(): logger.debug("Batch has no positives — skipping optimizer step"); continue`.
`valid` is computed from `SupConLoss` internals tracking which anchors have at least one positive.
**Effort:** S (human: 1h / CC: 5min)
**Priority:** P2

---

### ✅ DONE — [P2][M] 1-NN eval_accuracy does not detect overfitting on fine-tuned embeddings
**What:** SupCon trains the backbone to cluster same-class embeddings together. 1-NN accuracy on the fine-tuned backbone will increase monotonically with training epochs regardless of generalisation — it is a training convergence signal, not an overfitting detector.
**Why:** The plan presents the eval gate as preventing silent regressions, but it cannot detect the most common failure mode (overfitting on a small annotated set).
**Pros of fixing:** Genuine overfitting detection improves checkpoint quality.
**Cons:** Requires a truly held-out set drawn before training begins (not just a post-training split) — changes the `stratified_split` contract.
**Context:** Research item: evaluate alternatives — (a) cosine similarity distribution shift between annotated and unannotated frames, (b) linear probe on frozen backbone instead of 1-NN, (c) accept current monotone 1-NN as "good enough convergence signal" since overfitting risk at 500 frames + 8 frozen blocks is genuinely low.
**Effort:** M (human: 1 day / CC: 30min)
**Priority:** P2 — lower since overfitting risk at 500 frames is practically low
**Implemented:** Added `_eval_distribution_shift()` in `pipeline/supervised_finetune.py`: computes intra-class vs. inter-class cosine similarity gap (gap ≈ 0 = no separation, gap ≈ 0.5 = healthy, gap > 0.9 = potential overfitting). Logged as warning when gap exceeds `SUP_OVERFITTING_SHIFT_THRESHOLD` (default 0.9, configurable via env var). Not used as a gate — warning only, since overfitting risk at 500 frames is low. `distribution_shift` value returned in result dict and persisted in `model_checkpoints` table.

---

### ✅ DONE — [P3][S] Suppress dino vector search during active reembed job (fall back to clip)
**What:** During a re-embedding sweep, Qdrant contains a mix of old-model and new-model `dino` vectors. Cosine similarity between them is meaningless — search quality degrades silently.
**Why:** Users querying during the sweep get incorrect ranked results with no warning.
**Pros:** Consistent search quality throughout the sweep.
**Cons:** Requires a job-awareness flag in the search path; adds complexity.
**Context:** Simplest mitigation: expose a `GET /admin/reembed-status` endpoint; the search service checks if a reembed job is running and falls back to `clip` vector search. Or: set a Redis/DB flag during sweep and read it in `app/services/search.py`.
**Effort:** S (human: 2h / CC: 10min)
**Priority:** P3 — reembed window is short (~8 min for 500K frames)
**Implemented:** `_reembed_is_active()` in `app/services/search.py` queries `jobs` table for a running reembed job; dino reranking is suppressed (falls back to clip-only) when active, with an info log. `GET /admin/reembed-status` endpoint added to `app/routers/admin.py` returns `{active, job_id, frames_reembedded}`.

---

### ✅ [P2][S] Add retrain watermark to prevent infinite retrigger after threshold is crossed — DONE
**What:** Store `last_retrain_watermark` (annotated frame count at last successful fine-tune) in a `system_state` DB table or as a `settings`-namespaced row. Only trigger fine-tuning when `total_annotated - last_retrain_watermark >= MIN_NEW_ANNOTATED_SINCE_RETRAIN` (new config var, default 100). Worker updates watermark after successful job completion.
**Implemented:**
- `scripts/migrate_postgres.py`: `CREATE TABLE IF NOT EXISTS system_state (key TEXT PK, value TEXT)`.
  Initial `last_retrain_watermark=0` row inserted if absent.
- `pipeline/config.py`: `MIN_NEW_ANNOTATED_SINCE_RETRAIN` env var (default 100).
- `app/routers/cvat.py` `_maybe_trigger_finetune`: reads `system_state.last_retrain_watermark`;
  only enqueues if `total_annotated - watermark >= MIN_NEW_ANNOTATED_SINCE_RETRAIN`.
- `worker/main.py` supervised_finetune job handler: updates `system_state` watermark to
  `total_annotated` after a successful (accepted) checkpoint.
**Effort:** S (human: 2h / CC: 10min)
**Priority:** P2
**Depends on:** `cvat_label` column migration (P1)

---

### ✅ DONE — [P3][M] Establish model version provenance (annotations → checkpoint → embeddings → served results)
**What:** Track which annotation batch trained which checkpoint, which checkpoint produced which Qdrant embeddings, and which model version served each query. Store `model_version_id` in Qdrant payload at upsert time and in a `model_checkpoints` PostgreSQL table.
**Why:** Without provenance, rollback requires wiping and re-embedding, "model v3 improved by 12%" claims are unverifiable, and debugging retrieval regressions is impossible.
**Pros:** Enables rollback, A/B comparison, and credible improvement metrics.
**Cons:** M-size effort; Qdrant schema change; adds payload bytes per point.
**Context:** Codex outside voice finding. Not needed for v1 single-developer deployment. Required before production multi-user use.
**Effort:** M (human: 3 days / CC: 1h)
**Priority:** P3
**Implemented:** `model_checkpoints` table in `scripts/migrate_postgres.py` (checkpoint_path, model_version_id, annotation_count, best_accuracy, distribution_shift, created_at, notes). `MODEL_VERSION_ID` env var in `pipeline/config.py` (default `"base"`). `model_version_id` added to Qdrant frame payload in `pipeline/indexer.py` `_build_frame_point()`. Worker `handle_finetune_job()` inserts provenance row and updates `settings.MODEL_VERSION_ID` to `sup_{job_id[:8]}` after accepted checkpoint.

---

### ✅ DONE — [P3][S] Single authoritative active-model-version source (eliminate split-brain)
**What:** Replace the three-way active model source (`DINO_CHECKPOINT` env var, `active_checkpoint.txt`, job `payload.checkpoint`) with a single authoritative source: a `system_state.active_dino_checkpoint` DB row. All components (API startup, worker trigger, reload endpoint) read from DB.
**Why:** Three independent sources of truth create split-brain risk in multi-replica or restart scenarios.
**Pros:** Single source of truth; survives DB-backed restarts; no env var drift.
**Cons:** DB connection required at API startup; fallback logic for cold-start.
**Context:** Codex outside voice finding. For v1 single-node deployment, file + env var is acceptable. Fix before horizontal scaling.
**Effort:** S (human: 2h / CC: 15min)
**Priority:** P3
**Implemented:** `_resolve_dino_checkpoint()` in `app/state.py` reads `system_state.active_dino_checkpoint` from DB at startup; overrides `settings.DINO_CHECKPOINT` if found; falls back to env var silently on DB failure. Worker `handle_finetune_job()` writes `active_dino_checkpoint` to `system_state` after every accepted checkpoint.

---

### ✅ DONE — [P3][S] Label taxonomy normalization across CVAT tasks
**What:** Add a normalization layer in `_mark_frames_annotated` / `from_db` that maps CVAT task-specific label names to a canonical vocabulary. Flag conflicting labels (same frame annotated differently across tasks).
**Why:** Different CVAT tasks may use renamed classes or partial coverage. `SELECT DISTINCT cvat_label` silently produces a mixed ontology.
**Pros:** Training data integrity; cleaner class distributions.
**Cons:** Requires a canonical-vocab config or a `label_mappings` table.
**Context:** Codex outside voice finding. Only relevant once multiple annotation campaigns are in use.
**Effort:** S (human: 3h / CC: 20min)
**Priority:** P3
**Implemented:** `_normalize_labels()` in `pipeline/supervised_finetune.py` applies a `Dict[str,str]` mapping to raw labels; logs a warning when the same frame gets conflicting labels after normalization. Applied in both `from_xml()` and `from_db()` before vocabulary build. `CVAT_LABEL_MAPPINGS` JSON env var in `pipeline/config.py` (default `{}`). `config_from_settings()` passes mappings through.

---

### ✅ DONE — [P3][S] GPU resource isolation for concurrent fine-tuning + inference + re-embedding
**What:** Add a GPU job serialization mechanism: a PostgreSQL advisory lock or `gpu_jobs` semaphore table that gates concurrent GPU work. Workers check-in before allocating GPU memory and check-out on completion.
**Why:** Fine-tuning, live inference, and re-embedding all hit the same GPU. On a 24GB A10, concurrent fine-tune + re-embed + inference easily OOMs silently.
**Pros:** Prevents CUDA OOM; predictable GPU scheduling.
**Cons:** Adds polling complexity in worker.
**Context:** Codex outside voice finding. Not needed on CPU-only. Critical on shared GPU machines.
**Effort:** S (human: 4h / CC: 20min)
**Priority:** P3
**Implemented:** `gpu_jobs` table in `scripts/migrate_postgres.py`; `_gpu_checkin()`/`_gpu_checkout()` in `worker/main.py`; wired into `handle_finetune_job()` (wraps `run_supervised_finetune` in try/finally) and `handle_reembed_job()` (full try/finally). Fail-open on DB error; stale entries evicted on every check-in; contention logged as warning. `WORKER_ID` and `GPU_JOB_TIMEOUT_SEC` config in `pipeline/config.py`.

---

### ✅ DONE — [P3][S] Post-deployment: measure automation ROI vs. manual restart
**What:** The auto-trigger pipeline (5 scope items, ~400 lines) eliminates the need for `docker restart api` after fine-tuning. After the first real deployment, measure: how often does annotation actually happen, how often does the threshold get crossed, and whether the automation saved meaningful ops time.
**Why:** The outside voice challenge: infrastructure cost may exceed benefit for infrequent annotation workflows.
**Pros:** Evidence-based decision on whether to maintain the automation or simplify to a CLI-only flow.
**Cons:** Requires 1-2 months of production data.
**Context:** The moat argument (compounding improvement) is valid for high-frequency annotation workflows. For teams that annotate once a quarter, the manual restart is simpler. Measure before building v3 automation features.
**Effort:** S (human: 1h analysis / CC: N/A)
**Priority:** P3 — post-deployment retrospective
**Implemented:** `GET /admin/automation-roi` in `app/routers/admin.py`. Derives all metrics from existing `jobs` + `frames` tables (no schema changes). Returns: `total_annotated_frames`, `annotation_campaigns` (distinct months), `finetune_jobs_triggered/accepted`, `finetune_acceptance_rate`, `model_reloads`, `reembed_sweeps_completed`, `estimated_ops_minutes_saved` (reloads × 3 min), `days_observed`, `annotation_frequency_per_week`, and a `verdict` (LOW_FREQUENCY / MODERATE_FREQUENCY / HIGH_FREQUENCY / INSUFFICIENT_DATA) with plain-English `verdict_detail`.

---

## SV-06 | Scene Intelligence — Captioning + Structured Search

Deferred work from the 2026-03-27 CEO review of the Florence/Qwen captioning pipeline.

---

### ✅ [P1][S] Validate ollama/vLLM Qwen2.5-VL-7B serving quality (pre-Phase-2 prerequisite) — DONE
**What:** Before writing any Phase 2 code, validate that the Qwen2.5-VL-7B sidecar (ollama
Docker service, `qwen2.5-vl:7b` Q4_K_M) produces acceptable structured JSON extraction
quality on real vehicle frames. Run 20 vehicle-frame images through the ollama service with
the Phase 2 prompt template and measure:
1. JSON validity rate (response parseable + all required keys present): target ≥ 0.85
2. Vehicle count accuracy vs. ground-truth annotations: target ≥ 0.70
3. p95 latency per frame on RTX 4060 Ti with Florence also loaded: target ≤ 30s
4. GPU VRAM with both Florence + Qwen active: confirm ≤ 14GB total
   (run `scripts/profile_gpu_memory.py` with ollama serving).
If Q4_K_M validity < 0.85: try `qwen2.5-vl:7b-q8_0` (Q8, ~7.7GB VRAM) or vLLM FP16
with `--cpu-offload-gb 4`. Both still fit on 16GB GPU, trading VRAM for quality.
**Why:** Replaces the earlier AWQ quality gate. The serving approach changed from in-process
AWQ to ollama/vLLM HTTP sidecar, which removes the 24GB GPU requirement. GGUF Q4_K_M
quantisation still needs quality validation on multimodal structured-output tasks before
Phase 2 code is written. Also validates Docker Compose integration.
**Pros:** 30-minute CC task. Catches both quality and integration issues before any Phase 2
code is written. Confirms hardware requirements for CLAUDE.md update.
**Cons:** Requires ollama + ~4.7GB model download. One-time cost.
**Context:** See `docs/pipeline.md` and `docs/runbooks/gemma-api.md` for the current
sidecar-backed model serving workflow and operational guidance.
**Effort:** S (human: 4h / CC: 30min)
**Priority:** P1 — blocks Phase 2 implementation
**Depends on:** Phase 1 (`pipeline/florence_model.py`) ships and eval set built
**Implemented:** `scripts/validate_qwen_serving.py` — samples N frames (default 20) from a
mission frames directory, sends each to the configured `QWEN_API_URL` endpoint, and reports:
JSON validity rate (target ≥ 0.85), p95 latency (target ≤ 30s), optional vehicle count
accuracy vs JSONL ground truth (target ≥ 0.70), and GPU VRAM from nvidia-smi. Exit code 0
on pass, 1 on target miss, 2 on service unreachable. Run after `make up` with vLLM sidecar:
```bash
QWEN_API_URL=http://localhost:8010/v1 \
python scripts/validate_qwen_serving.py \
    --frames-dir data/frames/my_mission --count 20
```

---

### ✅ [P3][S] Advisory lock cleanup — Qwen lock no longer needed — DONE
**What:** With Qwen running as an ollama/vLLM sidecar service, `pg_advisory_lock(1)` is only
needed by nerfstudio (to serialize GPU compute with the worker). Remove Qwen from the lock
scope in `worker/main.py` and `app/routers/scene.py` (NL parse path). Only nerfstudio's
`ns-train` wrapper should acquire `pg_advisory_lock(1)`.
**Why:** The advisory lock was added to prevent Qwen in-process from conflicting with
nerfstudio on the same GPU. With Qwen serving from a separate container (ollama), its VRAM
is managed independently — no advisory lock needed. Removing Qwen from the lock reduces
serialization and eliminates the robot-query latency issue flagged in the CEO review.
**Pros:** Simpler locking. `/query/scene` NL parse no longer blocked by nerfstudio runs.
**Cons:** nerfstudio + ollama Qwen now compete for VRAM without coordination. Acceptable —
ollama auto-offloads to CPU RAM when VRAM is full.
**Context:** Architectural simplification enabled by the ollama/vLLM sidecar approach.
**Effort:** S (human: 1h / CC: 10min)
**Priority:** P3 — do after Phase 2 ships and ollama integration is confirmed
**Implemented:** No code changes required. `pg_advisory_lock(1)` for Qwen was never
introduced — the implementation went directly to the vLLM sidecar approach, so `worker/main.py`
and `app/routers/scene.py` (which was never created) have no Qwen advisory lock scope.
nerfstudio's `pg_advisory_xact_lock` in `pipeline/global_map_db.py` is unrelated and correct.

---

### ✅ [P3][S] Backfill script: flag irrecoverable missing-file frames — DONE
**What:** The `scripts/backfill_captions.py` skip-missing-file logic (log warning + continue) permanently leaves frames with `caption=NULL` when their disk file is gone. Add a `caption_skip_reason TEXT` column (or reuse a JSON field) to mark these rows as `"file_missing"` so they aren't silently re-skipped on every backfill run without explanation.
**Why:** The resume-safe logic (skip already-captioned rows) has no equivalent for "tried and failed because file missing." Operators running backfill see a log warning but the DB shows `caption=NULL` indistinguishably from "not yet processed." Over time, the null rate metric (`caption_null_rate`) becomes misleading — it includes frames that will never get captions.
**Implemented:**
- `scripts/migrate_postgres.py`: `ALTER TABLE frames ADD COLUMN IF NOT EXISTS caption_skip_reason TEXT` (also added to `CREATE TABLE` definition).
- `scripts/backfill_captions.py`: `_fetch_pending_batch` now filters `AND caption_skip_reason IS NULL`; `_mark_skip_reason()` sets `caption_skip_reason='file_missing'` for unreadable files; `--dry-run` reports both pending and already-skipped counts.

---

### ✅ [P1][S] Structured eval design spec (prerequisite for Phase 1 pass/fail gate) — DONE
**What:** Before building the eval set, write a one-page eval design spec:
- Query taxonomy: ≥5 queries per category — (1) vehicle count ("5 trucks"), (2) spatial
  arrangement ("in a row", "convoy"), (3) road condition ("wet road", "mountain pass"),
  (4) negative controls (queries with no matching frames), (5) general scene recall
- Annotation protocol: two independent annotators, majority vote for ground truth
- Baseline: run eval against FTS-only (no Florence) first — if FTS passes precision@5 ≥ 0.8,
  Florence adds no value over existing text search
- Minimum eval set: 100 frames stratified across mission types, not random sample
- Confidence intervals: report 95% CI around precision@5 estimate
**Why:** Codex outside voice finding. "50-100 real keyframes + 20 queries" with no
stratification risks producing noisy anecdotes that cannot distinguish real signal from
sampling noise. The Phase 1 pass gate (precision@5 ≥ 0.8) is the shipping gate for Phase 2
— a poorly designed eval can both (a) incorrectly pass a bad model and (b) incorrectly
fail a good one.
**Pros:** 30-minute design task. Makes the eval gate decisive. Baseline comparison reveals
whether Florence adds value at all.
**Cons:** Requires two annotators (or one with forced review gap of ≥48h).
**Context:** Eng review 2026-03-27. Design doc open question #1 already flags that
`caption_confidence` correlation with quality needs validation — this is the vehicle for that.
**Effort:** S (human: 2h / CC: 30min)
**Priority:** P1 — must complete before building eval set
**Depends on:** Phase 1 (`pipeline/florence_model.py`) runs on at least one real mission
**Implemented:** Initial evaluation guidance was captured during development, but the
standalone eval spec/script were later removed during docs and script cleanup. Use the
current runtime docs and admin endpoints as the source of truth for caption behavior.

---

### ✅ [P3][S] Extend caption_model column with prompt version and runtime provenance — DONE
**What:** `caption_model TEXT` currently stores e.g. `"florence-2-large"`. Add prompt
version/hash and runtime identifier so that existing captions are distinguishable from
ones produced with a different prompt template or quantization level. Options:
- (A) Add `caption_prompt_version TEXT` column alongside `caption_model`
- (B) Extend `caption_model` to include structured value: `"florence-2-large:v1:fp16"` (model:prompt_version:precision)
- (C) Store provenance in `frame_facts_json` as a `_meta` key
**Why:** Codex outside voice finding. A bare model name doesn't capture which prompt
produced the caption or whether quantization was applied. When prompts change or
quantization is added, old and new captions are indistinguishable — this breaks
reproducibility and makes re-captioning decisions ambiguous.
**Pros:** Forward-compatible provenance. Makes it possible to re-caption selectively
when prompts change.
**Cons:** Small schema change. Low urgency until second model or prompt version ships.
**Context:** Phase 1 ships `caption_model = "florence-2-large"`. This TODO triggers
when a second caption model, quantized variant, or prompt change is introduced.
**Effort:** S (human: 1h / CC: 10min)
**Priority:** P3 — defer until second caption model or prompt version is introduced
**Implemented:** Option B — `FlorenceModel.model_tag` now returns `"florence-2-large:{version}:{precision}"` (e.g. `"florence-2-large:v1:fp16"`). `FLORENCE_PROMPT_VERSION` env var in `pipeline/config.py` (default `"v1"`); bump to re-caption selectively. Precision auto-detected from DEVICE+USE_FP16.

---

## SV-07 | Gemma Production Integration

Deferred work from the 2026-04-05 CEO review of the Gemma production integration plan.
Phase ordering: Phase 1 (no gate) → Phase 2 (no gate, parallel) → Phase 3 (SSL gate required).

---

### ✅ DONE — [P1][S] Refactor `pipeline/demo_runner.py` into `pipeline/demo/` subpackage
**Transition note:** This was an intermediate restructuring step in the old
demo era. The orchestrator was later promoted into the canonical
`pipeline/workflows/local/` package and the demo compatibility surface was
removed, but this item remains here as historical context for that transition.

**What:** Split the 5667-line `demo_runner.py` into a proper subpackage:
- `pipeline/demo/steps_embed.py` — embedding steps
- `pipeline/demo/steps_caption.py` — captioning steps
- `pipeline/demo/steps_ssl.py` — SSL fine-tuning steps
- `pipeline/demo/steps_distill.py` — distillation steps
- `pipeline/demo/steps_map.py` — map/SfM steps
- `pipeline/demo/steps_report.py` — report generation steps
- `pipeline/demo/runner.py` — orchestrator
- `pipeline/demo_runner.py` becomes a thin shim: `from pipeline.workflows.demo.runner import run_demo`

Async-parallel demo steps: embed + caption can run concurrently; SSL runs after embed completes. Target: <10 min for 58 frames.
**Why:** 5667 lines is an active maintenance hazard. Adding Phase 2+3 steps will push it to 7K+. Splitting now is easier than splitting a 7K-line file.
**Pros:** Each step module is independently testable; orchestrator is readable; backward compat preserved via shim.
**Cons:** Requires updating all imports within demo_runner.py.
**Effort:** S (human: 4h / CC: ~1h)
**Priority:** P1 — ship before Phase 2 additions
**Depends on:** none

---

### ✅ DONE — [P1][S] Wire `GEMMA_API_URL` as production captioner in `pipeline/indexer.py`
**What:** In `_caption_batch()`, prefer `GEMMA_API_URL` when set; fall back to Florence.
Chunked async batch: N=3 frames/chunk, 50s timeout, 1 retry per chunk, Florence fallback on 2nd failure. If Florence also fails: store `""` and `caption_confidence=0.5` (never NULL — existing convention).

`GEMMA_MAX_CAPTION_FRAMES` env var (default 200): cap frames captioned per mission using existing histogram-diff quality ranking. Per-frame `caption_model` records which model produced each caption (Gemma vs Florence).

Parallel chunked throughput estimate: 4 concurrent × 3 frames/chunk = 12 frames/wave, ~50s/wave → 58 frames ≈ ≤4 min. Serial fallback: ≤5 min at 16.6s/frame average.

No schema changes — uses existing `caption`, `caption_confidence`, `caption_model` columns.
**Why:** Gemma captions are demonstrably richer than Florence; async chunking eliminates the 42-minute bottleneck.
**Pros:** Florence fallback keeps it safe; no DB migration required.
**Cons:** Requires ollama `--parallel` flag validation (default 1 = serial; need 3-4 concurrent).
**Context:** `GEMMA_API_URL` config setting already exists in `pipeline/config.py`. `models/gemma_model.py` already exists. `GEMMA_MAX_CAPTION_FRAMES` and `GEMMA_API_URL` wiring into `_caption_batch()` is not yet implemented.
**Effort:** S (human: 3h / CC: ~45min)
**Priority:** P1 — no gate required
**Depends on:** none

---

### ✅ DONE — [P1][S] `docs/runbooks/gemma-api.md` — Gemma API operations runbook
**What:** Write a runbook covering: ollama health check, model restart, quality validation procedure, Florence fallback activation, exponential backoff guidance (implement if >10% timeout rate observed).
**Why:** Production operators need documented procedures; the async chunk retry logic needs a matching ops procedure.
**Effort:** S (human: 1h / CC: ~15min)
**Priority:** P1 — ship with Gemma captioner wiring
**Depends on:** Gemma captioner wiring (above)

---

### ✅ [P2][M] `models/efficientvit_model.py` + DINOv3→EfficientViT-S1 baseline distillation — DONE
**What:** Two deliverables:

1. `models/efficientvit_model.py` — EfficientViT-S1 loader via `timm`:
   ```python
   timm.create_model("efficientvit_b1", pretrained=True)
   ```

2. `pipeline/distill.py` Stage 1→2: DINOv3→EfficientViT-S1 distillation baseline.
   - RKD-D loss only (no RKD-A), 384-dim student, no SSL gate required.
   - This baseline ships independently of Phase 3 and provides the reference R@1 against which Gemma teacher is later compared.

**Acceptance criteria:**
- Catch `RuntimeError: CUDA out of memory` → log VRAM requirement, raise: `"EfficientViT Stage 1→2 requires ≥8GB VRAM; detected {N}GB"`
- Unit test: mock OOM, verify message
**Why:** DINOv3→EfficientViT-S1 is the production edge model if the SSL gate fails. Ships unconditionally as Phase 2 baseline.
**Effort:** M (human: 2 days / CC: ~1.5h)
**Priority:** P2 — no gate required
**Depends on:** none

---

### ✅ [P2][S] Qwen → Gemma structured extraction replacement in `pipeline/qwen_model.py` — DONE
**What:** Replace `pipeline/qwen_model.py` HTTP calls with Gemma API equivalents.

Gate (must pass before implementing): ≥50 test frames, JSON validity ≥ 0.82, vehicle count accuracy ≥ 0.70. If gate fails (<0.82 or <0.70): keep Qwen; no gray zone.

Use `scripts/validate_qwen_serving.py` as the template for the 50-frame validation run (already exists; point it at Gemma endpoint).
**Why:** Simplifies the Docker stack by removing the Qwen sidecar; validated by the existing 20-frame test (though 20 frames is too weak — gate requires ≥50).
**Effort:** S (human: 2h / CC: ~1h)
**Priority:** P2 — conditioned on 50-frame quality gate
**Depends on:** Gemma captioner wiring; quality validation run

---

### ✅ [P3][S] `scripts/validate_ssl_improvement.py` — multi-video SSL eval harness (gate for Phase 3) — DONE
**What:** SSL validation script that measures ΔR@1 pre/post SSL fine-tuning across ≥3 diverse videos.

Diversity requirements (all required, no substitutions):
1. A daylight video (≥200 frames)
2. A low-light or overcast video (≥200 frames)
3. A video with ≥3 distinct GPS waypoints, real field footage only — synthetic/test data excluded (≥200 frames)

Methodology: 3 seeds per video; median ΔR@1 used. Gate: ΔR@1 > +0.02 on at least 2/3 videos.

**Why:** Current SSL test shows zero improvement on 58 homogeneous aerial frames — cannot distinguish "SSL doesn't work" from "dataset too easy." Must validate on diverse data before investing in Phase 3.
**Effort:** S (human: 3h / CC: ~1h)
**Priority:** P3 — this script IS the gate for GemmaSSLFinetuner and GemmaVisionTeacher
**Depends on:** ≥3 diverse field videos available

---

### ✅ [P3][M] `pipeline/ssl_finetune.py`: `GemmaSSLFinetuner` class — DONE
**What:** Add `GemmaSSLFinetuner` to `pipeline/ssl_finetune.py`. Uses Gemma vision encoder embeddings as SSL targets for DINOv3 fine-tuning.

**Acceptance criteria:**
- Pre-flight GPU check: `if not torch.cuda.is_available(): raise SkipStep("GemmaSSL requires CUDA — CPU not supported")`
- Unit test: `test_gemma_ssl_finetuner.py::test_cpu_only_raises_skip_step`
**Why:** Language-grounded SSL targets may improve retrieval for text-query use cases where DINOv3 self-supervised targets are vocabulary-blind.
**Effort:** M (human: 2 days / CC: ~1.5h)
**Priority:** P3 — CONDITIONED on SSL gate: ΔR@1 > +0.02 on ≥2/3 test videos. If gate fails: skip; DINOv3→EfficientViT baseline (Phase 2) is the production edge model.
**Depends on:** `scripts/validate_ssl_improvement.py` passes gate

---

### ✅ [P3][M] `pipeline/distill.py`: `GemmaVisionTeacher` + Stage 0→1 distillation — DONE
**What:** Add `GemmaVisionTeacher` to `pipeline/distill.py`:
- `nn.Module` wrapper around GemmaEmbedder vision encoder
- Projection head: `Linear(384→1152)` — student (384-dim ViT-S/14) projected to match Gemma vision encoder output (1152-dim SigLIP ViT-L)
- RKD-DA loss using existing `pipeline/distill.py` implementation
- Optional `lambda_caption_anchor=0.5` loss: `cosine(projected_student, CLIP_text(caption))`
- Stage 0→1 distillation pipeline

**Acceptance criteria:**
- `torch.nan_to_num(teacher_embedding, nan=0.0)` before RKD loss computation
- Unit test: feed all-NaN teacher embedding, verify loss is finite
- Compare Stage 1→2 quality from Gemma teacher vs DINOv3 baseline (R@1 comparison)
**Why:** Gemma teacher provides language-grounded embeddings for distillation. Only valuable if SSL gate proves SSL fine-tuning helps on diverse data.
**Effort:** M (human: 2 days / CC: ~1.5h)
**Priority:** P3 — CONDITIONED on SSL gate passing. If gate fails: DINOv3→EfficientViT baseline (Phase 2) is the production edge model.
**Depends on:** `GemmaSSLFinetuner` passes gate; `models/efficientvit_model.py` exists

---

## SV-06 | Scene Intelligence — Phases 3–6

Deferred work from the 2026-03-27 CEO review. Phases 1–2 are ✅ DONE. Phases 3–6 are unimplemented.

---

### ✅ [P2][M] Phase 3: `app/routers/scene.py` — `POST /query/scene` endpoint — DONE
**What:** New endpoint `POST /query/scene` with JSONB filter params. Allows structured queries over `frame_facts_json` — e.g., filter by vehicle count range, road condition keyword, GPS bbox, and temporal window simultaneously.

Query parameters: text query (optional), JSONB filter predicates (vehicle_count_min/max, road_condition, gps_bbox, time_range), top_k.

Returns: ranked frame results with caption, frame_facts_json excerpt, GPS coordinates, mission context.
**Why:** Phase 2 (Qwen structured extraction) produces rich JSONB per frame but there's no query endpoint that uses it. Without `/query/scene`, the structured data is stored but unqueryable from the API.
**Effort:** M (human: 3 days / CC: ~1.5h)
**Priority:** P2 — unlocks value from Phase 2 Qwen extraction
**Depends on:** Qwen→Gemma replacement (SV-07 Phase 2) ships; `frame_facts_json` populated

---

### ✅ [P2][M] Phase 4: Semantic change detection — `semantic_diff_json` + `change_explanation` — DONE
**What:** Add two columns to `change_detections` table:
- `semantic_diff_json JSONB` — structured diff of `frame_facts_json` between missions at same GPS location (e.g., `{"vehicle_count": {"before": 2, "after": 5}, "road_condition": {"before": "dry", "after": "wet"}}`)
- `change_explanation TEXT` — Gemma-generated natural language explanation of the change

Extend `pipeline/change_detection.py` to compute semantic diff when both frames have `frame_facts_json` populated.
**Why:** Current change detection reports "embedding distance = 0.42" with no explanation. Operators need to know *what* changed, not just *that* something changed.
**Effort:** M (human: 3 days / CC: ~1h)
**Priority:** P2 — high operational value; relatively contained implementation
**Depends on:** Qwen→Gemma replacement ships; `frame_facts_json` populated at both GPS waypoints

---

### ✅ [P3][M] Phase 5: `scene_timeline` PostgreSQL table + `POST /query/pose` extension — DONE
**What:**
1. New `scene_timeline` table: `(mission_id, frame_id, gps_lat, gps_lon, gps_alt, t_sec, facts_json JSONB, embedding vector)` — one row per keyframe with GPS + facts
2. Extend `POST /query/pose` to return not just "similar frames" but "last 3 visits to this GPS waypoint had [conditions], current caption: [...]"

Schema:
```sql
CREATE TABLE IF NOT EXISTS scene_timeline (
    id BIGSERIAL PRIMARY KEY,
    mission_id TEXT NOT NULL,
    frame_id TEXT NOT NULL,
    gps_lat DOUBLE PRECISION,
    gps_lon DOUBLE PRECISION,
    gps_alt DOUBLE PRECISION,
    t_sec DOUBLE PRECISION,
    facts_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```
**Why:** The robot advisory API currently returns raw frame matches. A timeline table enables "last N visits" reasoning and temporal trend detection per GPS waypoint.
**Effort:** M (human: 3 days / CC: ~1.5h)
**Priority:** P3 — high value but depends on Phase 4 semantic diff being useful first
**Depends on:** `POST /query/scene` (Phase 3); `frame_facts_json` populated; semantic change detection (Phase 4)

---

### ✅ [P3][M] Phase 6: `pipeline/rtsp_captioner.py` — streaming caption pipeline — DONE
**What:** New `pipeline/rtsp_captioner.py` that consumes frames from a MediaMTX RTSP stream in real time, runs Gemma captioning on each frame (or every N frames), and writes captions + facts to the `scene_timeline` table in near-real-time.

Key design: non-blocking frame consumer (skip frames if captioner is behind); configurable sampling rate (`RTSP_CAPTION_FPS`, default 0.5); Florence fallback on Gemma timeout.
**Why:** Enables live mission captioning without waiting for video ingest post-processing. Drone/rover operators get real-time scene understanding.
**Effort:** M (human: 4 days / CC: ~2h)
**Priority:** P3 — requires `scene_timeline` table and Gemma captioner to be stable first
**Depends on:** `scene_timeline` table (Phase 5); Gemma captioner wiring (SV-07 Phase 1)

---

### ✅ [P3][S] Admin: `caption_null_rate` metric + `/admin/caption-eval` page — DONE
**What:**
- `GET /admin/caption-eval` page (or JSON endpoint) reporting: `caption_null_rate` (fraction of frames with `caption IS NULL` excluding `caption_skip_reason IS NOT NULL`), mean/p95 `caption_confidence`, and per-model breakdown (`caption_model` distribution).
- Add `caption_null_rate` to the existing `GET /admin/automation-roi` response as an additional metric.
**Why:** Without a dashboard, operators can't tell if the captioner is silently failing. `caption_null_rate` is the leading indicator of pipeline health.
**Effort:** S (human: 2h / CC: ~30min)
**Priority:** P3 — useful once Gemma is the production captioner
**Depends on:** Gemma captioner wiring; `caption_skip_reason` column (already implemented)
