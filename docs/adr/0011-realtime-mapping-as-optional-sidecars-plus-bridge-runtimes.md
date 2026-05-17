# ADR-0011: Realtime Mapping Uses Optional Sidecars, Owned Bridge Runtimes, and Chained Post-Flight Jobs

Date: 2026-05-02  
Status: Accepted

## Context

Realtime drone mapping now spans three distinct concerns:
- live telemetry ingest from MAVSDK or ROS transports
- online pose / occupancy estimation during the mission
- heavier post-flight mapping and semantic consolidation after recording

Those concerns have different dependency profiles and operational constraints.
The base API should still run without ROS, MAVSDK, SLAM engines, or GPU mapping
services installed. At the same time, operators need a concrete deployment shape
for realtime mapping rather than one large implicit “realtime mode”.

## Decision

Keep realtime mapping modular and optional:
- a project-owned reference realtime service provides the HTTP contract and
  local fallback behavior
- open-source pose / occupancy engines are integrated as optional sidecars
  selected through named adapters
- MAVSDK and ROS ingestion are owned by project bridge runtimes rather than
  being left as normalization helpers only
- heavy post-flight work is executed as explicit worker jobs after indexing,
  not hidden inside the live API or a single opaque index stage

Current implementation:
- reference service: `docker/realtime/docker-compose.realtime.yml`,
  `src/selfsuvis/mapper/realtime_main.py`
- OSS sidecar module: `docker/realtime/docker-compose.realtime-engines.yml`
- sidecar adapter catalog: `src/selfsuvis/realtime_pilot/adapters/`
- telemetry bridge runtimes: `src/selfsuvis/realtime_pilot/bridge_runtime.py`,
  `docker/realtime/docker-compose.realtime-bridge.yml`
- post-flight jobs: `src/selfsuvis/worker/main.py`

## Consequences

Positive:
- The repo has a clear separation between API, bridge transport ownership,
  online estimators, and post-flight mapping stages
- Operators can deploy only the realtime pieces that fit the hardware and
  mission profile
- Realtime mapping is observable through explicit backends, sidecars, bridge
  runtimes, and worker job types

Trade-offs:
- The architecture has more moving parts than a single embedded mapper process
- Some integrations are only fully active when external engine images or ROS /
  MAVSDK runtimes are provided by the deployment
- End-to-end correctness depends on coordination across API, database, bridge
  runtimes, sidecars, and worker chaining
