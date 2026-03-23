# TODOS

Deferred work, known issues, and pre-ship blockers.
Format: [Priority] [Effort] — Description

---

## P1 — Blockers (must resolve before shipping)

### [P1][S] Validate Qdrant 2D GPS range query performance
**What:** Benchmark simultaneous lat AND lon range filter on a synthetic 50K-point Qdrant collection.
**Why:** Change detection and robot API both rely on GPS bounding box queries. If lat+lon simultaneous
filters are slow at 50K points, the change detection SLA is broken before a line of production code ships.
The plan notes a Python fallback (lat-only filter + post-filter lon in Python) but this must be validated
before the architecture is locked — not discovered after integration.
**Pros:** Catches the bottleneck early; validates whether the Qdrant payload index config is correct.
**Cons:** Requires writing a benchmark script; ~1 hour of work.
**Context:** `pipeline/change_detection.py` and `app/routers/robot.py` both use GPS bbox Qdrant filters.
Qdrant payload indexes must be created at collection init: `CREATE PAYLOAD INDEX ON gps.lat (float range)`
and `gps.lon (float range)`. If 2D filter is slow, fall back to: Qdrant lat-only filter → Python post-filter
on lon. This fallback should be the default until performance is validated.
**Effort:** S (human: ~4h / CC: ~15 min)
**Depends on:** Qdrant collection with payload indexes configured

---

### [P1][M] GPU memory budget — profile Florence-2 + CLIP + DINOv3 + nerfstudio
**What:** Profile peak GPU memory for all models running on the same worker GPU: Florence-2 (~12GB),
CLIP (~4GB), DINOv3 (~4GB), nerfstudio splatfacto (varies by scene, ~8-16GB). Document loading order
and whether they can coexist or must be loaded/unloaded sequentially.
**Why:** An OOM crash mid-mission loses all in-flight work and leaves PostgreSQL in a partial state
(frame rows without captions). If the GPU can't hold all models simultaneously, an explicit model
lifecycle (load → inference → unload → load next) must be designed into `pipeline/indexer.py`.
**Pros:** Prevents OOM in production; informs Dockerfile.nerfstudio hardware requirements.
**Cons:** Requires target hardware to profile on.
**Context:** nerfstudio runs in a separate container (docker-compose.override.yml, GPU machines only).
Florence-2, CLIP, DINOv3 all run in the same GPU worker container. The nerfstudio container runs
sequentially AFTER pycolmap SfM (v1 design decision). Profile at minimum: Florence-2 + CLIP +
DINOv3 simultaneously on the worker GPU. Baseline: RTX 4090 (24GB VRAM) or A100 (40/80GB).
**Effort:** M (human: ~1 day / CC: ~30 min)
**Depends on:** Target GPU hardware availability

---

## P2 — Important (resolve before or during v2)

### [P2][S] nginx CORS — add Access-Control-Allow-Origin for SuperSplat .ply fetch
**Resolved: Build it now in this PR.**
(Tracked here for implementation reference: nginx config for /static/maps/ path needs
`add_header Access-Control-Allow-Origin "*";` to allow SuperSplat iframe to fetch splat.ply.
Also: nginx service must be added to docker-compose.yml — it is in the architecture plan but
missing from the compose file. Required for splat.ply serving and SuperSplat iframe to work.)

### [P2][S] Upgrade Streamlit to 1.37+ for @st.fragment 3DGS polling
**What:** Upgrade Streamlit from 1.31.1 to ≥1.37 in `docker/Dockerfile.ui`.
**Why:** DESIGN.md specifies `@st.fragment` for isolated 30s 3DGS poll. Streamlit 1.31.1
does not have `st.fragment`. Without it, the full mission detail page reruns every 30s
while map is pending — reloading all frame thumbnails and charts on every poll cycle.
**Context:** Streamlit 1.37 introduced `@st.fragment`. Test for breaking changes in
column layout (`st.columns()` responsive behavior changed in 1.35) and `use_column_width`
deprecation (replaced by `use_container_width` in 1.34).
**How to apply:** In DESIGN.md, the 3DGS viewer section already specifies `@st.fragment`.
Update `docker/Dockerfile.ui` pip install line and run `make test` to catch regressions.
**Effort:** S (human: ~2h / CC: ~15 min)
**Depends on:** Nothing

### [P2][M] Phase 2 global map — 3DGS ICP fusion (Open3D)
**What:** ICP-based registration of per-mission 3DGS splat.ply files into a single persistent global map.
**Why:** Phase 1 (GPS-to-ENU) gives rough metric alignment but no photometric fusion. Phase 2 produces a
dense, unified 3D model that enables precise cross-mission visual comparison and global localization.
**Pros:** Enables the true "living world model" vision. Robot API can use metric pose (tx/ty/tz) without GPS.
**Cons:** XL effort; requires Open3D; research-grade ICP on 3DGS is not production-ready as of 2026-03.
**Context:** `frames.global_pose_json` and `global_map_missions.registration_transform_json` schema already
supports Phase 2. `registration_error` is NULL in Phase 1 (GPS); Phase 2 populates it from ICP residual.
Open3D must NOT be added to Dockerfile.worker in v1.
**Effort:** XL (human: ~2 months / CC: ~1 week)
**Depends on:** Phase 1 GPS registration shipped; nerfstudio splatfacto reconstruction working

### [P2][M] tx/ty/tz-only robot query path (no GPS)
**What:** Allow `POST /query/pose` with only local metric coordinates (tx/ty/tz) — no GPS required.
**Why:** Robots operating in GPS-denied environments (underground, indoor-outdoor transition zones)
cannot provide lat/lon. The Phase 2 global map (ICP-registered) provides metric poses that make this query meaningful.
**Pros:** Unlocks robot use in GPS-denied environments.
**Cons:** Requires Phase 2 ICP global map to be useful (Phase 1 GPS-to-ENU doesn't help without GPS).
**Context:** v1 returns HTTP 400 with message "GPS coordinates required for v1 spatial query;
tx/ty/tz-only path deferred to v2." Schema already accepts tx/ty/tz fields.
**Effort:** M (human: ~1 week / CC: ~30 min)
**Depends on:** Phase 2 global map (ICP fusion)

### [P2][S] Streamlit admin page — worker status, queue depth, al_tag distribution
**What:** Add an admin tab to the Streamlit UI showing: live worker status, job queue depth,
total missions processed, al_tag distribution chart (needs_annotation / novel / none counts).
**Why:** Observable systems are easier to debug. Currently there is no at-a-glance health view.
**Pros:** Immediate operational value; makes active learning results visible.
**Cons:** Additional Streamlit page; requires PostgreSQL queries.
**Context:** Query jobs table for queue depth; query frames table for al_tag counts per mission.
**Effort:** S (human: ~4h / CC: ~15 min)

### [P2][M] CVAT annotation integration — write al_tag=annotated from CVAT feedback
**What:** CVAT shares the same PostgreSQL instance in v2 and writes `al_tag='annotated'` back
to frames when a frame is labeled. No schema change required — the CHECK constraint already
includes 'annotated' (reserved for v2).
**Why:** Closes the annotation half of the active learning loop.
**Context:** See ADR-0005. CVAT connects to PostgreSQL natively.
**Effort:** M (human: ~1 week / CC: ~30 min)
**Depends on:** CVAT v2 integration

### [P2][XL] Self-supervised fine-tuning pipeline (v2)
**What:** Use annotated frames (al_tag=annotated) to fine-tune DINOv3 on mission-domain data.
**Why:** Closes the active learning loop. Custom-tuned DINOv3 produces better embeddings for
outdoor autonomy scenes than the generic pretrained model.
**Effort:** XL (human: ~1 month / CC: ~1 week)
**Depends on:** CVAT annotation integration; sufficient annotated frames (>1000)

---

## P3 — Nice to have

### [P3][S] Stale diagram audit — docs/
**What:** Review all ASCII diagrams in docs/ to verify they reflect the v1 architecture
(PostgreSQL, new pipeline modules, new Docker services).
**Context:** Implementation will add docs/architecture.md; verify other docs stay in sync.
**Effort:** S (human: ~2h / CC: ~10 min)

### [P3][M] k-means online/incremental clustering for scale (>50 missions)
**What:** Replace batch k-means (sklearn) with mini-batch k-means or online clustering
when the total frame count across missions exceeds ~25K frames (50 missions × 500 frames).
**Why:** Full k-means refit over all historical embeddings becomes expensive at scale.
**Context:** Current design: k-means (k=20) run over all mission DINOv3 embeddings after each mission,
updated incrementally. At 50+ missions, switch to mini-batch k-means or IncrementalPCA + k-means.
**Effort:** M (human: ~3 days / CC: ~20 min)

### [P3][S] Pre-flight robot map cache export
**What:** Export the global map as a local cache for the robot to carry onboard,
enabling faster query without network round-trip.
**Why:** Reduces robot dependency on network connectivity during flight.
**Context:** Defer until robot integration is confirmed and the advisory API pattern is validated.
**Effort:** S-M depending on cache format (human: ~3 days / CC: ~20 min)

### [P3][S] Live camera streaming — full RTSP ingest (v1.5)
**What:** Accept live RTSP/RTMP from real drone FPV, IP cameras, rovers via MediaMTX.
**Why:** Enables real-time mission indexing without post-flight upload.
**Context:** v1 MediaMTX validates streaming code path via file re-streaming only
(`ffmpeg -re -i mission.mp4 -f rtsp rtsp://localhost:8554/test`).
**Effort:** M (human: ~1 week / CC: ~30 min)
**Depends on:** Camera hardware and network config

### [P3][M] Multi-robot shared world model
**What:** Multiple robots contributing to and reading from the same global map.
**Why:** Enables coordinated multi-agent outdoor autonomy.
**Context:** v1 is single-robot. PostgreSQL schema supports multi-mission; no FK coupling to robot identity.
**Effort:** M-L (human: ~2 weeks / CC: ~1 hour)

### [P3][L] Multi-site ENU (>50km or disconnected sites)
**What:** Support multiple ENU origins for missions that exceed the ~50km valid radius
or are at geographically disconnected sites.
**Why:** Current ENU origin is fixed at first GPS-valid mission; valid ~50km radius only.
**Context:** v2. Implement as multiple `global_map` rows (one per site).
**Effort:** L (human: ~2 weeks / CC: ~1 hour)

---

## Added by /plan-eng-review (2026-03-23)

### [P2][M] 3DGS scene chunking for long missions
**What:** Split long missions into sub-scenes at pycolmap disconnected component boundaries.
Each connected scene gets its own `maps/{mission_id}/scene-{N}/splat.ply`.
**Why:** nerfstudio splatfacto trained on disconnected scenes (takeoff → transit → inspection)
produces poor-quality Gaussians. SfM will naturally identify connected components.
**Context:** pycolmap reconstruction returns multiple models when scenes are disconnected.
Implement scene detection first; pass each connected component separately to ns-train.
The `maps/{mission_id}/` directory can contain multiple splat.ply files.
Streamlit viewer shows a scene selector if >1 scenes.
**Effort:** M (human: ~1 week / CC: ~30 min)
**Depends on:** pipeline/sfm.py + pipeline/mapper.py shipped first

### [P2][S] SFM_FPS env var — dense frame extraction for pycolmap
**What:** Extract a denser frame set for SfM (default: 2fps via SFM_FPS env var) separate from
the sparse search keyframes (1 keyframe/3s). pycolmap needs multi-view overlap.
**Why:** Decided in eng review: separate frame sets prevent the sampling strategy conflict.
**Context:** pipeline/sfm.py receives the dense frame set; pipeline/indexer.py continues
using sparse keyframes for embedding. Two ffmpeg extraction passes per mission.
**Effort:** S (human: ~4h / CC: ~10 min)

### [P2][S] pipeline/gps_extractor.py — structured GPS metadata extraction
**What:** Extract GPS telemetry from drone video in priority order:
1. ffprobe GPS atoms (MP4 location tag, some DJI formats)
2. SRT sidecar file (same filename, .srt extension, standard DJI format)
3. GPMF binary format (GoPro Max, GoPro Hero)
4. Null fallback with warning
**Why:** Decided in eng review: "GPS from video container metadata" is underspecified.
`GPS_SIDECAR_PATH` env var overrides the auto-detection path.
**Context:** Output: List[Optional[dict]] with `{lat, lon, alt, timestamp_ms}` per frame.
Synchronized to frame timestamps from ffprobe. Write to `frames.gps_json` per keyframe.
**Effort:** S-M (human: ~2 days / CC: ~15 min)

### [P2][S] PYCOLMAP_CAMERA_MODEL env var
**What:** Allow specifying pycolmap camera model: `SIMPLE_RADIAL` (default) | `PINHOLE` | `RADIAL`.
**Why:** DJI and GoPro cameras have known intrinsics. Providing the correct model significantly
improves pose accuracy vs. estimating from EXIF.
**Context:** Set in pipeline/sfm.py before calling pycolmap reconstruction.
Advanced users with known drone hardware should specify their camera model.
**Effort:** S (human: ~1h / CC: ~5 min)
