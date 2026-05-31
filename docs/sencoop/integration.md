# coop — selfsuvis Integration Guide

This document covers how `coop` integrates with the selfsuvis API, realtime
pipeline, and threat aggregation layer.

## Overview

```
LoRaWAN field sensors ──► ChirpStack ──► Mosquitto MQTT ──┐
Frigate camera streams ──────────────────────────────────┤
                                                          ▼
                                                   MqttSubscriber
                                                          │
                           ┌──────────────────────────────┤
                           ▼                              ▼
                   SiteStateAggregator           CoopRealtimeIngestor
                   SensorMeshFusion                       │
                   SceneSynthesizer              RealtimeThreatAggregator
                           │                              │
                           ▼                              ▼
                   GET /site/state              GET /site/threat
                   GET /site/mesh               robot advisory API
                   GET /site/synthesis          /query/pose enrichment
                   WS  /site/stream
```

**Additionally:** `CoopStreamService` registers each Frigate camera in MediaMTX and
starts an `RtspCaptioner` session per camera, writing live captions to `scene_timeline`.
`SceneSynthesizer` pulls those captions when building the LLM narrative.

---

## Site state API endpoints

| Endpoint | Description |
|---|---|
| `GET /site/state` | Rolling-window snapshot: all sensor readings + camera detections |
| `GET /site/mesh` | Spatial sensor mesh with GPS-proximity neighbour links |
| `GET /site/sensors` | All active LoRaWAN device summaries |
| `GET /site/cameras` | All active Frigate camera summaries |
| `GET /site/synthesis` | LLM scene narrative fusing all modalities (~10 s cache) |
| `GET /site/synthesis?force=true` | Force a fresh LLM call bypassing cache |
| `GET /site/threat` | Realtime sector-level threat map from coop sensors |
| `WS /site/stream` | WebSocket pushing `SiteState` JSON every N seconds |

---

## MQTT topics

The `MqttSubscriber` listens on two Mosquitto topic patterns (configured via env):

```
COOP_CHIRPSTACK_TOPIC=application/+/device/+/event/up   # LoRaWAN uplinks
COOP_FRIGATE_TOPIC_PREFIX=frigate                        # → frigate/events, frigate/+/events
```

ChirpStack publishes device uplinks to `application/{app_id}/device/{dev_eui}/event/up`.
Frigate publishes detection events to `frigate/events` and `frigate/{camera}/events`.

---

## Frigate RTSP bridge

`CoopStreamService` (started in the FastAPI lifespan) calls the Frigate HTTP API to
discover enabled cameras, then for each camera:

1. Registers an RTSP re-stream path in MediaMTX as `coop/{camera}`.
2. Starts an `RtspCaptioner` session that writes scene captions to `scene_timeline`.

Camera discovery repeats every 60 seconds (configurable via `CoopStreamService(refresh_sec=…)`),
so cameras added or disabled in Frigate are picked up without a restart.

The MediaMTX path `coop/{camera}` can be viewed by any RTSP client:

```
rtsp://localhost:8554/coop/entrance
```

---

## Scene synthesis

`SceneSynthesizer.synthesize()` is called by `GET /site/synthesis`. It:

1. Calls `SiteStateAggregator.get_state()` for current sensor + camera state.
2. Queries `scene_timeline` for the 20 most recent captions (last 5 minutes).
3. Sends a structured prompt to `REASONING_API_URL` (OpenAI-compatible endpoint).
4. Parses the JSON response into a `SceneSynthesis` object (cached 10 s).

Required env vars:

```env
REASONING_API_URL=http://gemma:8000     # or any OpenAI-compatible endpoint
REASONING_MODEL=gemma3:12b
REASONING_TIMEOUT_SEC=30
```

---

## Threat pipeline integration

`CoopRealtimeIngestor` converts coop observations to `SensorEvent` / `ThreatEvent`
envelopes and feeds them into `RealtimeThreatAggregator`:

- **LoRaWAN readings** → `SensorEvent(sensor_type="lorawan")`, sector derived from
  GPS grid (`grid:{lat_cell}:{lon_cell}` at ~110 m resolution).
- **Frigate detections** → `ThreatEvent(sensor_type="camera")`, threat score from
  detection confidence.

The aggregated snapshot is available at `GET /site/threat` and is compatible with
the robot advisory schema used by `POST /query/pose`.

---

## Docker compose

Start the IoT infrastructure alongside the main selfsuvis stack:

```bash
docker compose \
  -f docker/core/docker-compose.yml \
  -f docker/sencoop/docker-compose.sencoop.yml \
  up -d
```

Required env vars (add to `.env`):

```env
COOP_MQTT_HOST=mosquitto
COOP_MQTT_PORT=1883
COOP_MQTT_USER=selfsuvis
COOP_MQTT_PASSWORD=change_me
COOP_FRIGATE_API_URL=http://frigate:8971
COOP_CHIRPSTACK_TOPIC=application/+/device/+/event/up
COOP_FRIGATE_TOPIC_PREFIX=frigate
```

See [getting-started.md](getting-started.md) for the full setup procedure and
[configuration.md](../reference/configuration.md) for all available env vars.

---

## Learning pipeline use cases

### Sensor-annotated mission indexing

When a robot mission runs over an area monitored by LoRaWAN sensors, the sensor
data in `SiteStateAggregator` can enrich mission metadata:

- CO₂ spikes during a video segment → flag frames for fire/hazard annotation
- Motion events correlated with camera detections → reduce active-learning uncertainty
  score for frames that already have environmental confirmation

This enrichment is currently a manual integration point: query `GET /site/state`
or `GET /site/synthesis` from your mission workflow and inject the response into
`frame_facts_json` during `pipeline/indexer.py` processing.

### Change detection with sensor correlation

`change_detections` table entries can be cross-referenced with sensor history to
distinguish environmental changes (weather) from structural changes (construction):

- Temperature/humidity shift during a GPS-overlapping mission → likely weather
- No environmental sensor change but high visual delta → likely structural

### Continuous monitoring mode

Deploy the full stack without a mission (no video ingestion) for stationary site
monitoring. The site state API, WebSocket stream, and scene synthesis provide a
live human-readable dashboard without requiring any robot or drone.

---

## Further reading

- [getting-started.md](getting-started.md) — first-time setup
- [sensor-integration.md](sensor-integration.md) — adding new sensor types
- [analytics.md](../reference/analytics.md) — log analytics CLI (`coop-analytics`)
- [architecture.md](../reference/architecture.md) — coop component diagram
- [testing.md](testing.md) — MQTT integration tests and load testing
