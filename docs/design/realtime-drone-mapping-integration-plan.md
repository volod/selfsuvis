# Design: Near-Real-Time 3D Map and Sensor Fusion Integration Plan for Autonomous Drones

Date: 2026-04-08
Status: Proposed

## Goal

Extend `selfsuvis` from an offline mission-video spatial memory system into a hybrid stack with:

- real-time drone state estimation during flight
- near-real-time local 3D occupancy / mesh generation
- semantic fusion from camera and optional LiDAR into the map
- retention of the existing offline `pycolmap -> splatfacto -> ICP` path for higher-quality post-flight reconstruction

The intent is not to replace the current mapping pipeline. The intent is to add an online map path for autonomy and keep the current offline path for refinement, indexing, and search.

## Current Baseline In This Repo

Today the repo already has:

- offline frame/video indexing via `pipeline.workflows.indexer.VideoIndexer`
- mission storage in PostgreSQL and retrieval vectors in Qdrant
- offline SfM via [`pipeline/mapping/sfm.py`](/home/vola/src/selfsuvis/pipeline/mapping/sfm.py)
- GPS-to-ENU registration via [`pipeline/mapping/gps_registration.py`](/home/vola/src/selfsuvis/pipeline/mapping/gps_registration.py)
- offline 3DGS via [`pipeline/mapping/mapper.py`](/home/vola/src/selfsuvis/pipeline/mapping/mapper.py)
- global-map registration tables and helpers via [`pipeline/storage/global_maps.py`](/home/vola/src/selfsuvis/pipeline/storage/global_maps.py)
- advisory spatial query via [`app/routers/robot.py`](/home/vola/src/selfsuvis/app/routers/robot.py)

This is a strong base for mission review, but it is not a flight-time autonomy stack because the pose and map are generated after ingest, not continuously from synchronized sensors.

## Recommended Technical Direction

Use two mapping tracks:

### 1. Online autonomy track

Used during flight and optimized for latency.

- `RGB + IMU` drones:
  - primary: `VINS-Fusion`
  - fallback/lightweight option: `ORB-SLAM3`
- `RGB + IMU + LiDAR` drones:
  - primary: `LIO-SAM`
- dense local mapping:
  - primary: `nvblox` when NVIDIA GPU is available
  - fallback: `voxblox` on CPU
- semantic fusion:
  - existing YOLO / SAM stack in this repo for class and mask generation
  - promote depth from summary features to full map products for fusion

### 2. Offline refinement track

Used after or between missions and optimized for quality.

- keep `pycolmap` for batch camera-pose recovery
- keep `splatfacto` / 3DGS for dense scene reconstruction
- keep current ICP / GPS registration flow for multi-mission fusion

The online track should produce occupancy and semantic maps. The offline track should produce better splats and searchable scene memory.

## Architecture Changes

Add a real-time ingestion and mapping path without breaking the existing indexing flow.

### New runtime components

1. `telemetry` service
- accepts synchronized sensor packets from drone bridge or ROS2 bridge
- writes low-latency sensor data to PostgreSQL and optional ring-buffer cache

2. `realtime_mapper` service
- consumes camera, IMU, GPS, and optional LiDAR streams
- runs VIO/LIO + occupancy fusion
- publishes poses, local map tiles, and semantic observations

3. `bridge` service
- optional adapter for ROS2, MAVLink/MAVSDK, RTSP, or proprietary drone SDKs
- normalizes incoming messages into repo-native payloads

4. existing `worker`
- remains responsible for async, heavier jobs
- now triggers offline reconstruction from recorded sensor sessions and keyframes

### Revised high-level flow

```text
drone sensors
camera + imu + gps + optional lidar
    |
    v
bridge / telemetry ingest
    |
    +--> realtime pose estimator (VINS-Fusion / ORB-SLAM3 / LIO-SAM)
    |
    +--> realtime dense map (nvblox / voxblox)
    |
    +--> semantic fusion (YOLO / SAM / depth projection)
    |
    +--> PostgreSQL + Qdrant + map tile store
    |
    +--> /query/pose and future autonomy APIs

recorded mission data
    |
    v
worker async jobs
    |
    +--> pycolmap
    +--> splatfacto
    +--> ICP / global-map fusion
```

## Repository Integration Plan

### Phase 0: Storage and schema groundwork

Add new PostgreSQL tables for real-time data. Keep them separate from existing `missions` and `frames` tables so the online path can run at higher frequency without overloading the current indexing schema.

Recommended new tables:

- `robot_sessions`
  - `id`
  - `robot_id`
  - `mission_id`
  - `started_at`
  - `ended_at`
  - `sensor_profile_json`
  - `status`

- `sensor_packets`
  - `id`
  - `session_id`
  - `sensor_type` (`camera`, `imu`, `gps`, `lidar`, `barometer`)
  - `t_device`
  - `t_server`
  - `seq`
  - `payload_json`

- `realtime_poses`
  - `id`
  - `session_id`
  - `source` (`vins`, `orbslam3`, `liosam`, `gps_fallback`)
  - `t_sec`
  - `position_enu_json`
  - `orientation_quat_json`
  - `velocity_enu_json`
  - `covariance_json`
  - `tracking_status`
  - `global_map_id`

- `map_tiles`
  - `id`
  - `session_id`
  - `global_map_id`
  - `tile_key`
  - `map_type` (`tsdf`, `esdf`, `occupancy`, `mesh`, `semantic`)
  - `storage_path`
  - `resolution_m`
  - `updated_at`
  - `bounds_json`
  - `stats_json`

- `semantic_observations`
  - `id`
  - `session_id`
  - `frame_id`
  - `class_name`
  - `confidence`
  - `position_enu_json`
  - `bbox_json`
  - `mask_ref`
  - `track_id`
  - `facts_json`
  - `created_at`

- `map_fusion_jobs`
  - `id`
  - `session_id`
  - `job_type` (`offline_reconstruct`, `tile_merge`, `global_register`)
  - `status`
  - `input_json`
  - `result_json`
  - `created_at`
  - `updated_at`

Why:

- `frames` remains mission-review oriented
- `realtime_poses` and `sensor_packets` need much higher write rates
- `map_tiles` lets the system publish partial map updates instead of monolithic scene files

### Phase 1: Add real-time package boundaries

Create a new package:

```text
pipeline/realtime/
    __init__.py
    ingest.py
    sync.py
    session.py
    pose.py
    occupancy.py
    semantics.py
    tile_store.py
    quality.py
```

Module responsibilities:

- `ingest.py`
  - normalize incoming packets
  - validate timestamps and sensor ids
  - batch DB writes

- `sync.py`
  - approximate-time synchronization for camera / IMU / GPS / LiDAR
  - enforce drift bounds and dropped-packet metrics

- `session.py`
  - create, rotate, and finalize flight sessions
  - link session to existing `missions` rows when recordings are saved

- `pose.py`
  - abstraction over real-time pose backends
  - `VinsFusionBackend`
  - `OrbSlam3Backend`
  - `LioSamBackend`

- `occupancy.py`
  - abstraction over `nvblox` or `voxblox`
  - build TSDF, ESDF, occupancy, and mesh products

- `semantics.py`
  - project detections and masks into the map
  - maintain voxel / tile class counts and confidence

- `tile_store.py`
  - persist tile artifacts under `data/maps/realtime/<session_id>/`
  - manage incremental updates, compaction, and snapshots

- `quality.py`
  - track drift, relocalization, sensor lag, and map-confidence signals

### Phase 2: Add external-service adapters

Do not embed `VINS-Fusion`, `ORB-SLAM3`, `LIO-SAM`, or `nvblox` directly into the Python process first. Wrap them as sidecar services, the same way this repo already treats `nerfstudio` and the mapper.

Recommended new services:

- `docker/Dockerfile.realtime_mapper`
- `docker/Dockerfile.telemetry`
- optional `docker/docker-compose.realtime.yml`

Recommended service APIs:

- `POST /sessions`
- `POST /sessions/{id}/packets`
- `GET /sessions/{id}/pose/latest`
- `GET /sessions/{id}/map/latest`
- `GET /sessions/{id}/health`
- `POST /sessions/{id}/finalize`

For the mapper sidecar:

- `POST /estimate_pose`
- `POST /integrate_frame`
- `POST /integrate_lidar`
- `GET /map_tile/{tile_key}`
- `GET /stats`

This keeps the Python app stable while allowing C++ / ROS-heavy components to evolve separately.

### Phase 3: Bridge layer for drones and ROS2

Add:

```text
pipeline/media/drone_bridge.py
pipeline/media/ros_bridge.py
pipeline/media/mavlink.py
```

Responsibilities:

- convert MAVLink or MAVSDK telemetry into repo-native packets
- subscribe to ROS2 topics when the drone stack already runs in ROS2
- ingest RTSP video frames with timestamps aligned to telemetry

Expected source integrations:

- MAVSDK / MAVLink for PX4 or ArduPilot telemetry
- RTSP for camera stream
- ROS2 topics for teams already using ROS2

The bridge should output one internal packet schema, for example:

```json
{
  "session_id": "uuid",
  "sensor_type": "imu",
  "t_device": 1712345678.123,
  "seq": 88421,
  "payload": {
    "ax": 0.1,
    "ay": -0.2,
    "az": 9.7,
    "gx": 0.01,
    "gy": 0.00,
    "gz": -0.03
  }
}
```

### Phase 4: Promote depth from summary feature to map input

Current depth in [`pipeline/vision/depth.py`](/home/vola/src/selfsuvis/pipeline/vision/depth.py) stores percentile summaries for mission understanding. That is useful for captioning and retrieval, but not enough for geometry fusion.

Add a second output mode:

- `DEPTH_OUTPUT_MODE=summary|dense`
- `dense` mode writes depth maps or compressed depth tiles for selected frames

Recommended implementation:

- keep current summary path unchanged
- add `estimate_dense()` returning float depth map plus confidence map
- only run dense depth for frames selected by:
  - motion threshold
  - semantic importance
  - occupancy update cadence

Store dense depth under:

- `data/maps/realtime/<session_id>/depth/frame_<ts>.npz`

This allows `RGB + IMU` drones to contribute geometry to `nvblox` even without LiDAR.

### Phase 5: Semantic map fusion

Reuse the current object understanding stack but change the output target from frame metadata only to frame metadata plus world-anchored semantic map updates.

Implementation path:

1. run detection and optional segmentation on selected frames
2. use current pose estimate plus depth to back-project detections into ENU
3. update semantic voxels / tiles:
   - class histogram
   - confidence
   - last_seen_at
   - observation count
4. write compact observation records to PostgreSQL
5. optionally mirror searchable semantic landmarks into Qdrant

Suggested class families for drone autonomy:

- obstacle: `tree`, `pole`, `wire`, `fence`, `building_edge`
- terrain: `ground`, `road`, `grass`, `water`, `roof`
- dynamic: `person`, `vehicle`, `animal`, `other_drone`
- mission-specific: `landing_zone`, `marker`, `tower`, `panel`, `structure_defect`

For this repo, the pragmatic first step is not open-vocabulary everything. The pragmatic first step is a small autonomy-focused ontology.

### Phase 6: Query and API expansion

Extend the API with a new router:

```text
app/routers/realtime.py
```

Endpoints:

- `POST /realtime/session/start`
- `POST /realtime/session/{id}/packet`
- `GET /realtime/session/{id}/state`
- `GET /realtime/session/{id}/map`
- `GET /realtime/session/{id}/semantic-nearby`
- `POST /realtime/session/{id}/stop`

Extend [`app/routers/robot.py`](/home/vola/src/selfsuvis/app/routers/robot.py):

- when a live session exists, query `realtime_poses` and `map_tiles` before falling back to indexed mission frames
- add optional response fields:
  - `local_obstacles`
  - `semantic_alerts`
  - `map_freshness_ms`
  - `pose_source`

Important separation:

- `/query/pose` remains advisory and search-focused
- `/realtime/*` becomes flight-state and map-state focused

### Phase 7: Worker integration for post-flight refinement

When a real-time session is finalized:

1. persist sensor logs and selected keyframes
2. create or link a `mission_id`
3. enqueue async jobs:
   - offline `pycolmap`
   - offline `splatfacto`
   - mission/global-map registration
   - semantic graph consolidation
4. backfill `frames`, `missions`, and Qdrant payloads from the recorded session

This keeps one consistent product story:

- during flight: occupancy + semantic awareness
- after flight: better searchable splats and scene memory

## Concrete File-Level Plan

### New files

- `docs/design/realtime-drone-mapping-integration-plan.md`
- `app/routers/realtime.py`
- `app/services/realtime.py`
- `pipeline/realtime/__init__.py`
- `pipeline/realtime/ingest.py`
- `pipeline/realtime/sync.py`
- `pipeline/realtime/session.py`
- `pipeline/realtime/pose.py`
- `pipeline/realtime/occupancy.py`
- `pipeline/realtime/semantics.py`
- `pipeline/realtime/tile_store.py`
- `pipeline/realtime/quality.py`
- `pipeline/storage/realtime.py`
- `tests/unit/test_realtime_ingest.py`
- `tests/unit/test_realtime_sync.py`
- `tests/unit/test_realtime_pose_router.py`
- `tests/unit/test_semantic_tile_fusion.py`

### Existing files to extend

- [`pipeline/core/config.py`](/home/vola/src/selfsuvis/pipeline/core/config.py)
  - add realtime env vars
- [`app/main.py`](/home/vola/src/selfsuvis/app/main.py)
  - register realtime router
- [`worker/main.py`](/home/vola/src/selfsuvis/worker/main.py)
  - finalize session -> enqueue refinement jobs
- [`pipeline/storage/global_maps.py`](/home/vola/src/selfsuvis/pipeline/storage/global_maps.py)
  - add tile-aware global-map helpers
- [`pipeline/vision/depth.py`](/home/vola/src/selfsuvis/pipeline/vision/depth.py)
  - add dense output mode
- [`docs/architecture.md`](/home/vola/src/selfsuvis/docs/architecture.md)
  - add realtime services to runtime diagram
- [`docs/pipeline.md`](/home/vola/src/selfsuvis/docs/pipeline.md)
  - document the live pipeline alongside the existing indexing flow

## Configuration Additions

Add env vars in [`pipeline/core/config.py`](/home/vola/src/selfsuvis/pipeline/core/config.py):

- `REALTIME_ENABLED=false`
- `REALTIME_BACKEND=auto`
- `REALTIME_POSE_BACKEND=vins`
- `REALTIME_MAP_BACKEND=nvblox`
- `REALTIME_PACKET_BATCH_SIZE=128`
- `REALTIME_MAX_SENSOR_LAG_MS=120`
- `REALTIME_SESSION_TIMEOUT_SEC=30`
- `REALTIME_DEPTH_ENABLED=true`
- `REALTIME_DEPTH_MODEL=depth-anything/Depth-Anything-V2-Small-hf`
- `REALTIME_SEMANTICS_ENABLED=true`
- `REALTIME_TILE_RESOLUTION_M=0.2`
- `REALTIME_TILE_SIZE_M=20`
- `REALTIME_MESH_EXPORT_INTERVAL_SEC=5`
- `REALTIME_POSE_MIN_CONFIDENCE=0.5`
- `REALTIME_USE_LIDAR_IF_AVAILABLE=true`

## Milestone Plan

### Milestone 1: Minimal live pose

Ship first:

- ingest session start/stop
- sensor packet storage
- latest pose API
- `VINS-Fusion` or `ORB-SLAM3` sidecar integration
- unit tests around packet validation and session lifecycle

Exit criterion:

- repo can accept synchronized camera/IMU/GPS and return a live ENU pose stream

### Milestone 2: Live occupancy map

Ship next:

- `nvblox` or `voxblox` sidecar integration
- tile store
- local occupancy API
- map freshness and health metrics

Exit criterion:

- repo can publish updated occupancy / mesh tiles during flight

### Milestone 3: Semantic map

Ship next:

- selected-frame detection + segmentation
- depth back-projection into ENU
- semantic tile updates
- obstacle / terrain / landing-zone classes

Exit criterion:

- repo can answer “what is near the drone right now?” with geometry and classes

### Milestone 4: Post-flight refinement bridge

Ship next:

- session finalization into `missions`
- worker-enqueued offline reconstruction
- unified live + offline map references in `global_map`

Exit criterion:

- live sessions automatically become refined searchable missions after landing

## Risks and Tradeoffs

### Do not use 3DGS for the flight-time safety loop

3DGS is useful for post-flight reconstruction and visualization. It is not the right first representation for obstacle-aware control because:

- occupancy queries are awkward
- dynamic obstacle handling is poor
- latency and update cost are worse than TSDF / occupancy approaches

### Sensor synchronization will decide whether this works

The highest technical risk is not model selection. It is timestamp quality across:

- camera frames
- IMU packets
- GPS fixes
- LiDAR scans

This needs explicit drift monitoring and packet-quality metrics from the first implementation.

### Start narrow on semantics

Do not start with a broad VLM-generated taxonomy. Start with a hard-coded small class set tied to drone navigation and mission goals.

### Sidecars are operationally heavier but architecturally safer

ROS/C++ SLAM stacks are easier to isolate than to embed into the current Python app process. The repo already uses that pattern successfully for mapping.

## Recommended Order For This Repo

If only one implementation path is funded now, use this order:

1. real-time session ingestion and packet schema
2. `VINS-Fusion` sidecar for `RGB + IMU + GPS`
3. `nvblox` occupancy tiles
4. dense depth output in `pipeline/vision/depth.py`
5. semantic tile fusion using existing YOLO / SAM
6. worker finalization into current offline reconstruction path

If LiDAR is already available on the drone, swap step 2 for `LIO-SAM`.

## Success Criteria

The integration is successful when:

- a live drone session can be started from this stack
- the stack returns a usable ENU pose stream within sub-second latency
- occupancy tiles update continuously during motion
- nearby semantic obstacles can be queried from the API
- finalizing a session automatically produces a standard mission for offline refinement and search

## External References

- ORB-SLAM3: <https://github.com/UZ-SLAMLab/ORB_SLAM3>
- VINS-Fusion: <https://github.com/HKUST-Aerial-Robotics/VINS-Fusion>
- LIO-SAM: <https://github.com/TixiaoShan/LIO-SAM>
- RTAB-Map ROS2: <https://index.ros.org/r/rtabmap_ros/>
- nvblox: <https://nvidia-isaac.github.io/nvblox/v0.0.9/index.html>
- SAM 2: <https://github.com/facebookresearch/sam2>
- Grounding DINO: <https://github.com/IDEA-Research/GroundingDINO>
- Depth Anything V2: <https://github.com/DepthAnything/Depth-Anything-V2>
