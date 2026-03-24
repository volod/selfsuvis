# TODOS

Deferred work, known issues, and pre-ship blockers.
Format: [Priority] [Effort] — Description

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
**Remaining (blocked on real mission data):**
  - Worker wires `get_global_map_splats` → `run_mapper(target_splat_paths=...)` → `register_mission`.
**Depends on:** nerfstudio splatfacto producing real splat.ply from actual missions.

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
  - `config_from_settings()` — builds FinetuneConfig from env vars / pipeline.config.
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

### [P2][M] Edge model hydration — export fine-tuned DINOv3 for on-device object identification
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
from pipeline.edge_inference import EdgeClassifier
clf = EdgeClassifier("dino_edge_int8.onnx", "mission_objects.npz")
labels = clf.classify(frame_pil)   # [(label, score), ...]
```

**Effort:** M (human: ~1 week / CC: ~30 min)
**Depends on:** Self-supervised domain adaptation (✅ done — `dino_ssl_best.pt` required);
`onnxruntime` (CPU) or `onnxruntime-gpu` (Jetson); `onnxruntime-tools` for quantization.

### [P3][XL] Supervised fine-tuning on CVAT-annotated frames
**What:** Use CVAT-annotated frames (al_tag=annotated) to teach DINOv3 semantic category
boundaries via supervised contrastive loss or a lightweight classification head.
**Why:** Hard negative mining with semantic labels and class boundary learning cannot be
derived from self-supervised objectives alone. Completes the active learning loop.
**Effort:** XL (human: ~1 month / CC: ~1 week)
**Depends on:** CVAT annotation integration (✅ done); >1000 annotated frames (data dependency —
annotators must process missions through CVAT first); self-supervised domain adaptation above
(train supervised on top of the domain-adapted checkpoint, not the generic pretrained one)

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
