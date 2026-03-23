# ADR-0004: pycolmap + nerfstudio splatfacto for Camera Poses and 3D Map

Date: 2026-03-23
Status: Accepted
Deciders: @vola

---

## Context

The core product is a spatially-anchored semantic world model. Each video frame must be
associated with a camera pose (where the camera was, what direction it faced) and the
system must produce a dense 3D map of the scene.

Two distinct steps are required:
1. **Camera pose estimation** (Structure-from-Motion) — recover camera positions and
   orientations from video frames
2. **Dense 3D reconstruction** — build a photorealistic, queryable 3D map from poses + frames

Alternatives considered:

**Pose estimation:**
- pycolmap (COLMAP Python bindings) — battle-tested, Python API, no CUDA needed for SfM
- VINS-Fusion / LIO-SAM — production SLAM but requires ROS2, too heavy for v1
- GPS + IMU from video metadata — available as a fallback when present

**Dense 3D reconstruction:**
- nerfstudio `splatfacto` (3D Gaussian Splatting) — best maintained 3DGS implementation,
  CUDA, Python API, exportable `.ply`
- NeRF-based methods — slower, worse editability, superseded by 3DGS for this use case
- Sparse pose graph only (Codex recommendation) — faster to build but insufficient for
  navigation use case; rejected by decision-maker

## Decision

**Camera pose estimation:** `pipeline/sfm.py` using **pycolmap**.
- Input: keyframes (JPEG) output by the frame extractor
- Output: `sfm/{mission_id}/colmap_sparse/` + `sfm/{mission_id}/poses.json`
  (per frame_id: rotation matrix, translation, camera intrinsics, confidence)
- Fallback: if reconstruction fails (insufficient overlap, textureless surfaces),
  `pose_status=failed` written to PostgreSQL `missions`; frames stored with
  `pose=null`; 3DGS step skipped. Frames remain searchable via CLIP+DINO.
- GPS/IMU from video container metadata stored as `gps_json` payload alongside or
  instead of colmap pose when available.

**Dense 3D map:** `pipeline/mapper.py` using **nerfstudio splatfacto**.
- Input: keyframes + `colmap_sparse/` directory (output of sfm.py)
- Output: `maps/{mission_id}/splat.ply`
- Trigger: background worker job, fires only when `pose_status=success`
- Requires a custom Docker image (`Dockerfile.nerfstudio` based on `dromni/nerfstudio`)
  due to tinycudann and custom CUDA extension build requirements. Exposed as an optional
  `docker-compose.override.yml` service so users without GPU can skip reconstruction.

The pipeline is sequential: frame extraction → pycolmap → nerfstudio. Frames are indexed
into Qdrant after pycolmap (with pose metadata); the 3DGS job runs asynchronously and
adds `map_id` / `splat_point_nearest` to Qdrant payload on completion.

## Consequences

**Good:**
- Dense 3D map (not just sparse pose graph) supports robot navigation use case
- pycolmap is CUDA-free — pose estimation runs on CPU-only machines
- Failure handling is defined: degraded mode (no pose) is better than pipeline failure
- nerfstudio `.ply` output is viewable in standard 3DGS web viewers

**Bad / Tradeoffs:**
- nerfstudio requires a custom Docker image (tinycudann build is non-trivial)
- nerfstudio splatfacto reconstruction: ~10-30 min GPU time per scene
- pycolmap can fail on GPS-denied rapid-motion video or textureless outdoor scenes
  (open fields, water, uniform sky) — this is a known limitation for drones
- The 10-minute end-to-end success criterion applies to frame extraction + embedding +
  tagging only; 3DGS reconstruction is a separate async job with no latency target in v1
