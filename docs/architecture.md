# Architecture

## Repo Structure
```
app/          FastAPI service (API, auth, rate limiting, search, robot pose API)
worker/       async-native background worker (polls PostgreSQL jobs, runs VideoIndexer)
models/       embedding models (OpenCLIP, DINOv3, Florence-2)
pipeline/     frame extraction, SfM, 3DGS mapper, GPS, captioning, active learning,
              change detection, report generator, Qdrant utils, job/processed DB
ui/           Streamlit frontend (search, mission viewer, 3DGS viewer, change detection)
ui/components/ splat_viewer.py — SuperSplat iframe component
app/routers/  index.py, query.py, jobs.py, robot.py (POST /query/pose)
docker/       Dockerfiles and compose files (+ docker-compose.override.yml for nerfstudio)
scripts/      migrate_postgres.py, reset_qdrant.sh, index helpers, precheck, etc.
tests/        unit tests (tests/unit/) and integration (tests/test_api.py, test_robot_api.py)
docs/         documentation and ADRs
```

## Services (Docker)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Docker Compose                          │
├──────────┬──────────┬──────────┬──────────┬────────┬──────────┤
│  qdrant  │   api    │  worker  │    ui    │ nginx  │mediamtx  │
│  :6333   │  :8000   │          │  :8501   │ :8080  │  :8554   │
│  vector  │ FastAPI  │  async   │Streamlit │ static │ RTSP/    │
│   DB     │  + GPU   │  worker  │  + GPU   │ /maps  │ RTMP     │
└──────────┴──────────┴──────────┴──────────┴────────┴──────────┘
                │                        │
         ┌──────┴──────┐          ┌──────┴──────┐
         │ postgres:16 │          │ nerfstudio  │ ← optional (GPU only)
         │    :5432    │          │  :8001      │   docker-compose.override.yml
         │  jobs       │          │  ns-train   │
         │  missions   │          │  splatfacto │
         │  frames     │          └─────────────┘
         │  ...        │
         └─────────────┘
```

All services run as the current host user; `data/` and `cache/` are writable by you.

## Indexing Flow

Two parallel frame-extraction passes per mission, then sequential SfM → 3DGS:

```
Video file / URL / RTSP (MediaMTX)
        │
        ├── GPS extraction ──────────────────────────────────────────────────┐
        │   pipeline/gps_extractor.py                                        │
        │   (ffprobe atoms → SRT sidecar → GPMF → null fallback)            │
        │                                                                    │
        ├── Pass A: Dense frames (SFM_FPS=2fps) ─────────────────────────┐  │
        │   pipeline/sfm.py (pycolmap, CPU ~5min/1000 frames)            │  │
        │   → pose_json per frame, pose_status=success/failed            │  │
        │   → pipeline/mapper.py (nerfstudio, GPU ~10min)                │  │
        │     [only if pose_status=success]                              │  │
        │   → maps/{mission_id}/splat.ply  (or scene-{N}/splat.ply)     │  │
        │                                                                │  │
        └── Pass B: Sparse keyframes (adaptive) ─────────────────────────┘  │
            pipeline/florence_model.py → caption, caption_confidence        │
            models/openclip_model.py  → clip vector                         │
            models/dino_model.py      → dino vector (MODEL_NAME=dinov3)     │
            Tile extraction + quality filters + dedup                        │
            pipeline/qdrant_utils.py  → upsert to Qdrant                    │
            (gps_json from GPS extractor ──────────────────────────────────┘)
            pipeline/active_learning.py → active_learning_score, al_tag
            pipeline/report_generator.py → reports/{mission_id}/summary.html
            pipeline/change_detection.py → change_detections table
```

**al_tag precedence:** `needs_annotation` (top-K by score) > `novel` (DINOv3 dist > 0.5) > `none`

## Retrieval Flow

```
Text query  ──► OpenCLIP text embed  ──► Qdrant clip search
Image query ──► OpenCLIP image embed ──► Qdrant clip search ──► optional DINOv3 rerank (70/30)
"Find more" ──► DINOv3 embed of frame ──► Qdrant dino search (fallback: clip)
Robot pose  ──► GPS bbox Qdrant filter ──► frames near lat/lon ──► ranked response
```

## PostgreSQL Schema (key tables)

```sql
missions   (mission_id PK, video_id, pose_status, map_status, frame_count, ...)
frames     (frame_id PK, mission_id FK, caption, caption_confidence,
            active_learning_score, al_tag, pose_json, gps_json, global_pose_json, ...)
embedding_clusters  (cluster_id, centroid_json JSONB)
change_detections   (mission_id_a FK, mission_id_b FK, frame_id_a FK, frame_id_b FK,
                     embedding_distance, change_score, ...)
global_map          (global_map_id PK, origin_lat, origin_lon, origin_alt, ...)
global_map_missions (global_map_id FK, mission_id FK, registration_transform_json JSONB, ...)
```

See CEO plan (`docs/reviews/2026-03-23-ceo-review.md`) for full DDL.

## Coordinate System

ENU (East-North-Up), ROS REP-103. Quaternion: `(qx,qy,qz,qw)` body→ENU.
Global ENU origin fixed at first GPS-valid mission. Valid ~50km radius; multi-site deferred to v2.

---

## UI Design Decisions

### Sidebar Navigation Order
```
├─ Search          (text/image query — primary, existing page)
├─ Missions        (mission list + detail)
├─ Annotation Queue (frames flagged needs_annotation or novel)
└─ Settings
```

### Mission Detail Page Layout (single-page scroll)
```
MISSION DETAIL
┌─────────────────────────────────────────────────────────────────┐
│ ← Missions  (breadcrumb)                                        │
├─────────────────────────────────────────────────────────────────┤
│ METADATA CARD                                                   │
│ mission_id | frame_count | pose_status badge | map_status badge │
│ al_tag distribution bar | [View Report] [Index Form]            │
├─────────────────────────────────────────────────────────────────┤
│ TIMELINE (x=timestamp, y=active_learning_score)                 │
│ Hover=frame metadata, Click=expand frame inline                 │
├─────────────────────────────────────────────────────────────────┤
│ 3D CAMERA PATH (pycolmap poses, Plotly 3D scatter)              │
│ Color=al_tag (green/yellow/red), Size=al_score                  │
│ Hover=frame thumbnail tooltip, Click=expand inline              │
│ If pose_status=failed: "SfM unavailable" message               │
├─────────────────────────────────────────────────────────────────┤
│ FRAME GRID (4 cols, paginated 50/page)                          │
│ Sort: al_score DESC default | by time | by al_tag               │
│ Each card: thumbnail, caption, al_tag badge, [Find more]        │
├─────────────────────────────────────────────────────────────────┤
│ CHANGE DETECTION (if changes exist)                             │
│ Side-by-side frame pairs, sorted by change_score DESC           │
│ Hidden section if change_detections count = 0                   │
├─────────────────────────────────────────────────────────────────┤
│ 3DGS VIEWER (SuperSplat iframe, full width, 600px min height)   │
│ States: pending spinner | failed banner | success=iframe        │
├─────────────────────────────────────────────────────────────────┤
│ REPORT LINK → /static/reports/{mission_id}/summary.html         │
└─────────────────────────────────────────────────────────────────┘
```

### Result Card (reusable across Search, Frame Grid, Annotation Queue)
Every result card includes: thumbnail, caption, mission_id badge (clickable),
al_tag badge (red=needs_annotation, yellow=novel, green=none), score, timestamp,
[Find more like this] button.

### Interaction State Specification

| Feature | Loading | Empty | Error | Partial |
|---------|---------|-------|-------|---------|
| Mission list | Spinner | "No missions indexed yet. [Upload video]" | "Failed to load" + retry | — |
| Timeline chart | Spinner "Loading frames..." | "No frames in this mission" | "Chart failed" + reload | — |
| 3D scatter | Spinner | n/a (hidden if pose_status≠success) | Show if exception | pose_status=failed → "SfM unavailable — no camera poses" |
| Frame grid | st.spinner per page | "No frames found" | "Load error" + retry | Some frames without captions: show "[caption unavailable]" |
| Change detection | Spinner | "No changes detected vs prior missions" | "Load error" | — |
| 3DGS viewer | "3D map generating... (~10 min)" spinner, poll map_status every 30s | n/a | map_status=failed → "Reconstruction failed: [reason]. Download sparse point cloud: [link]" | map_status=skipped → "3D viewer unavailable (SfM failed)" |
| "Find more" | st.spinner in expander | "No similar frames found. [Try text search]" | "Search failed" + retry | DINO unavailable → CLIP fallback (show "[CLIP fallback]" label) |
| Annotation queue | Spinner | "No frames flagged for annotation. Complete more missions." | "Load error" | — |
| Mission report | n/a (HTML static) | Report shows "Missing data: [reason]" banner | Link 404 → "Report not generated. Re-index to regenerate." | — |

---
[← Configuration](configuration.md) | [Pipeline →](pipeline.md)

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | issues_open | 7 proposals, 7 accepted, 0 deferred |
| Codex Review | `/codex review` | Independent 2nd opinion | 1 | issues_found | 3 gaps resolved |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 2 | **CLEAN** | Run 1: 7 issues, 2 critical gaps. Run 2: 4 new issues resolved — 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 1 | **CLEAN** | score: 2/10 → 9/10, 8 decisions |

**VERDICT:** CEO + ENG + DESIGN CLEARED — ready to implement.
