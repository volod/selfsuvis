# Sencoop IoT Sensor Mesh (Playground 3)

Standalone package `src/sencoop/` plus a containerized edge stack
(`docker/sencoop/docker-compose.sencoop.yml`, config under `config/sencoop/`,
ops scripts under `scripts/sencoop/`). It provides continuous site awareness
between discrete missions: LoRaWAN telemetry, NVR camera events, acoustic
analysis, and LLM scene synthesis, all fused into rolling site state and threat
events consumed by the production server.

Everything is optional and lazy-imported: the API starts normally when the MQTT
broker is unreachable or `aiomqtt` is absent (ADR 0009, 0010).

## Package modules (verified surface)

| Module | Key classes | Role |
| --- | --- | --- |
| `sencoop/config.py` | `CoopPilotSettings` | `COOP_*` env: MQTT host/auth/TLS, ChirpStack + Frigate topics, rolling windows (300 s sensors / 120 s camera events) |
| `sensors/mqtt_subscriber.py` | `MqttSubscriber` | Background task subscribing ChirpStack (`application/+/device/+/event/up`) and Frigate (`frigate/...`) topics |
| `sensors/lorawan_decoder.py` | `SensorReading`, `decode_chirpstack_uplink()` | Uplink JSON -> typed reading; codec-object alias mapping (temperature, humidity, co2, pressure, battery, motion, gps) + raw `data` passthrough |
| `sensors/frigate_events.py` | `CameraEvent`, `FrigateEventConsumer` | Frigate detection events -> typed camera events |
| `sensors/rtsp_bridge.py` | `FrigateRtspBridge` | Frigate camera discovery -> MediaMTX `coop/{camera}` re-streams |
| `sensors/sound_analyzer.py` | `SoundAnalyzer`, `AcousticObservation` | Per-camera faster-whisper + FFT acoustic analysis |
| `mesh/site_state.py` | `SiteStateAggregator`, `SiteState` | Rolling-window snapshot of sensors + camera detections |
| `mesh/fusion.py` | `SensorMeshFusion`, `MeshNode`, `SiteMesh` | GPS-proximity neighbour links between sensor nodes |
| `mesh/scene_synthesis.py` | `SceneSynthesizer`, `SceneSynthesis` | Site state + `scene_timeline` captions -> `REASONING_API_URL` LLM -> cached narrative (10 s) |
| `analytics/` | collector, per-service parsers (mosquitto, chirpstack, frigate, openremote), reporter | `sencoop-analytics` CLI: log stats to console/JSON/HTML/Markdown |
| `camera_cli.py` | -- | Backing for `scripts/sencoop/sencoop-camera.sh` (add RTSP/USB cameras to Frigate config) |

## Edge stack containers

Profiles: `lorawan`, `video`, `metrics` (Make targets `sencoop-up`,
`sencoop-up-min`, `sencoop-up-video`, `sencoop-metrics-up`).

| Component | Container | Role |
| --- | --- | --- |
| Mosquitto | `coop-mosquitto` | Central MQTT bus; TLS on 8883; per-user topic ACLs |
| ChirpStack v4 | `coop-chirpstack` (+ `coop-cs-gwbridge`, `coop-cs-rest`, `coop-cs-postgres`, `coop-cs-redis`) | LoRaWAN network server (EU868); gateway UDP 1700 -> MQTT `eu868/#`; decoded uplinks -> `application/#`; REST facade on 8090 |
| Frigate | `coop-frigate` | NVR with detection; RTSP/USB cameras; events -> `frigate/#` |
| OpenRemote | Manager + Keycloak + nginx proxy + PostgreSQL | IoT asset management UI on 443; OAuth2/OIDC |
| Prometheus / Grafana / cAdvisor | `mon-*` | Optional `metrics` profile |

Data flow:

```
LoRa device -> gateway -> UDP:1700 -> gw-bridge -> MQTT eu868/# -> ChirpStack
           -> MQTT application/{app}/device/{devEUI}/event/up -> MqttSubscriber
RTSP/USB camera -> Frigate -> MQTT frigate/# -> MqttSubscriber
MqttSubscriber -> SiteStateAggregator + SensorMeshFusion + SceneSynthesizer
              -> CoopRealtimeIngestor -> RealtimeThreatAggregator
```

## Integration into the production server

- `pipeline/realtime/coop_ingest.py`: `sensor_reading_to_event()` ->
  `SensorEvent(sensor_type="lorawan")`; `camera_event_to_threat()` ->
  `ThreatEvent(sensor_type="camera")`; sector id from GPS grid at ~110 m.
- Endpoints: `GET /site/state`, `/site/mesh`, `/site/sensors`, `/site/cameras`,
  `/site/synthesis[?force=true]`, `/site/threat`, `WS /site/stream`.
- `CoopStreamService` (FastAPI lifespan) re-discovers Frigate cameras every 60 s,
  registers MediaMTX paths, starts `RtspCaptioner` (and optionally `SoundAnalyzer`)
  per camera.
- `selfsuvis.config.coop_settings` proxies `sencoop.config.settings`.

## Operations

- `scripts/sencoop/sencoop-bootstrap.sh` -- data dirs, TLS certs, MQTT users, stack up.
- `sencoop-ctl.sh`, `sencoop-compose.sh` (injects PUID/PGID), `sencoop-status`.
- `sencoop-release.sh` -- offline air-gapped bundles: `standard`, `min` (no video),
  `video`, `--with-metrics`; Make wrappers `sencoop-release*`.
- Sizing (pilot): 3-5 sites, 5-10 LoRaWAN devices, 2-6 cameras on one amd64 nettop
  (8 GB / 4 cores budget in `docs/sencoop/architecture.md`).

Full docs: `docs/sencoop/` (getting-started, architecture, sensor-integration,
integration, analytics, distribution, testing, troubleshooting).

## Deliberate gaps (drive the forward plan)

The mesh today is **uplink-only and consume-only**:

- No downlink/command path to devices (ChirpStack downlinks unused).
- No device provisioning automation (devices are registered by hand in the UI).
- No device inventory/registry, health ledger, or firmware version tracking.
- No first-party firmware: nodes are third-party devices with vendor codecs.
- No LoRa P2P/mesh (Meshtastic-style) transport, only LoRaWAN star topology.
- No HAB (high-altitude balloon) payload/tracking support.
- OpenRemote runs but is not synchronized with ChirpStack device state.
- No Node-RED (or similar) operator-editable automation layer.
- RF/SIGINT sensing is limited to what LoRaWAN devices report; no presence
  scanning, no spectrum awareness, no FPGA signal front-end.

These gaps are the subject of the field-device-layer scope in
[`../plan.md`](../plan.md).
