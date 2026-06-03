# coop — IoT Edge Monitoring Deep Dive

This document explains the `coop` subpackage and how it extends selfsuvis
from a mission-indexing system into a continuous stationary site-awareness platform.

It covers:
1. Why IoT edge monitoring differs from mission indexing
2. MQTT, LoRaWAN, and Frigate — what each contributes
3. The rolling window model and `SiteStateAggregator`
4. Sensor mesh fusion and GPS-proximity neighbour linking
5. Acoustic analysis — FFT classification and Whisper transcription
6. RTSP bridge — Frigate cameras into MediaMTX + RtspCaptioner
7. Scene synthesis — multi-modal LLM narrative generation
8. Threat pipeline integration — from sensor readings to sector risk maps
9. What to inspect and common failure modes

Related code:
- `src/sencoop/` — full IoT edge subpackage
- `src/selfsuvis/pipeline/realtime/coop_ingest.py` — sensor→event bridge
- `src/selfsuvis/app/services/coop_streams.py` — RTSP bridge lifecycle
- `src/selfsuvis/app/routers/site_state.py` — API endpoints
- `src/selfsuvis/app/main.py` — lifespan wiring

Related docs:
- [Sensor fusion fundamentals](03_sensor_fusion_fundamentals.md)
- [Threat primitives and local inference](15_threat_primitives_local_inference.md)
- [coop integration guide](../sencoop/integration.md)
- [Architecture](../reference/architecture.md)

---

## 1. Why IoT Edge Monitoring Differs From Mission Indexing

The core selfsuvis pipeline is **retrospective**: a video file arrives, the pipeline
indexes it, and results land in PostgreSQL and Qdrant. Queries answer "what did the
robot see on mission X?"

The coop layer is **prospective**: physical sensors transmit continuously and
the system must answer "what is happening right now?" with sub-minute latency.

The consequences of this distinction are significant:

| Dimension | Mission indexing | IoT edge monitoring |
|---|---|---|
| Input | Video file | Live MQTT messages + RTSP streams |
| Latency | Minutes to hours | Sub-minute (rolling window) |
| State | Immutable per mission | Mutable, continuously evicted |
| Uncertainty | Per-frame, bounded | Per-sensor, continuously updated |
| LLM calls | Per-mission, unlimited time | Cached (10 s TTL), timeout-guarded |
| Failure mode | Pipeline step failure | Sensor dropout, broker disconnect |

This means the coop layer needs different abstractions:

- **Rolling deque windows** instead of append-only tables
- **Eviction by timestamp** instead of persistence
- **Asyncio task fan-out** instead of sequential pipeline steps
- **Graceful degradation** (missing broker → empty state) instead of job failure

---

## 2. MQTT, LoRaWAN, and Frigate

### MQTT

MQTT is the message bus connecting all IoT edge devices to selfsuvis. A Mosquitto
broker runs as a Docker container on `selfsuvis-net`. All three sensor sources —
ChirpStack, Frigate, and any direct MQTT devices — publish to this broker.

The `MqttSubscriber` class (`coop/sensors/mqtt_subscriber.py`) subscribes to
two topic patterns:

```
application/+/device/+/event/up     # ChirpStack LoRaWAN uplinks
frigate/events                       # Frigate NVR detection events
frigate/+/events                     # Per-camera variant
```

`+` is the MQTT single-level wildcard. Both subscriptions are active simultaneously
via a single aiomqtt connection. On message arrival, `_dispatch()` routes by topic
prefix to the appropriate decoder.

The subscriber reconnects automatically on broker disconnect (5 s default interval).
This matters: network partitions between the API container and Mosquitto are a
normal operational condition, not an exception.

### LoRaWAN / ChirpStack

LoRaWAN (Long Range Wide Area Network) is the wireless protocol used by field
sensors — temperature/humidity nodes, CO₂ monitors, motion detectors, soil probes.

The protocol stack:

```
Field sensor → LoRa radio → Gateway → ChirpStack → MQTT → selfsuvis
                 915 MHz                 LNS
```

- **Gateway**: bridges LoRa radio frames to UDP packets (Semtech packet forwarder).
- **ChirpStack**: the LoRaWAN Network Server. Manages device activation (ABP/OTAA),
  deduplication (multiple gateways can receive the same frame), and uplink routing
  to MQTT as JSON payloads.
- **Uplink payload**: a JSON envelope containing `devEui`, `time`, `rxInfo` (RSSI,
  SNR), `fCnt` (frame counter), and a `data.object` block decoded by the device
  codec.

`decode_chirpstack_uplink()` in `lorawan_decoder.py` parses this into a `SensorReading`
dataclass with typed optional fields. The codec `object` is walked via `_FIELD_ALIASES`
to normalize diverse field naming conventions across sensor manufacturers.

**Key insight:** The `rssi` and `snr` fields are not measurements of the physical
environment — they are measurements of the RF link quality between the sensor and the
gateway. A low RSSI indicates the sensor is at range limit or obstructed, not that
the environment is hostile.

### Frigate NVR

Frigate is an open-source Network Video Recorder designed for object detection. It
runs as a Docker container, accepts camera RTSP streams, runs an object detection
model (YOLO-based) on each frame, and publishes detection events to MQTT.

A Frigate detection event payload includes:
- `id` — unique event ID
- `camera` — camera name
- `label` — detected class (`person`, `car`, `dog`, …)
- `score` — detection confidence 0-1
- `top_score` — peak confidence across the event's lifetime
- `start_time` — Unix timestamp of first detection
- `end_time` — Unix timestamp when object disappeared (null if ongoing)
- `box` — bounding box `{xmin, ymin, xmax, ymax}` in pixel coordinates, plus
  normalized `region` fractions

`FrigateEventConsumer.decode()` converts this into a `CameraEvent` dataclass. All
timestamps use `datetime.fromtimestamp(ts, tz=timezone.utc)` — Frigate emits Unix
seconds, not ISO strings.

---

## 3. Rolling Window Model and SiteStateAggregator

`SiteStateAggregator` (`mesh/site_state.py`) is the central in-memory store. It
holds one `deque` per sensor device and one `deque` per camera:

```python
_sensors: dict[str, deque[SensorReading]]   # keyed by dev_eui
_cameras: dict[str, deque[CameraEvent]]     # keyed by camera name
```

Both deques are unbounded in `collections.deque` but evicted by age on every insert.

### Eviction logic

```python
cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._sensor_window_sec)
while q and q[0].received_at < cutoff:
    q.popleft()   # oldest is at left (front)
```

New readings are appended to the right (`q.append(reading)`). The deque is therefore
always sorted oldest-left, newest-right. Eviction pops from the left until the front
element is within the window.

Default window sizes (configurable):
- `COOP_SENSOR_WINDOW_SEC=300` — 5 minutes for LoRaWAN sensors
- `COOP_CAMERA_EVENT_WINDOW_SEC=120` — 2 minutes for Frigate events

**Why different windows?**
Sensors transmit every 1-15 minutes depending on battery life. A 5-minute window
captures 1-5 readings per device — enough for trend detection without stale data.
Camera detections fire at much higher rates (multiple per second on busy scenes); the
shorter window prevents the deque growing too large while keeping recency.

### Thread safety

All mutations go through `asyncio.Lock`. `get_state()` also holds the lock while
building the snapshot. This is safe because all callers are on the same event loop
— there is no thread-level parallelism in the coop stack.

### SiteState snapshot

`get_state()` returns a `SiteState` Pydantic model. Building the snapshot involves:

1. Walking each sensor deque and building a `SensorSummary` (latest reading + stats)
2. Walking each camera deque and building a `CameraEventSummary` (recent detections)
3. Computing `active_motion` (any sensor with `motion=True` in the window)
4. Counting sensors and cameras

The snapshot is computed on demand and never cached — callers at `GET /site/state`
get a fresh view every time they ask.

---

## 4. Sensor Mesh Fusion and GPS-Proximity Linking

`SensorMeshFusion` (`mesh/fusion.py`) takes a `SiteStateAggregator` and produces
a `SiteMesh` — a spatial graph of sensor nodes with proximity-derived edges.

### MeshNode

Each node corresponds to one sensor or camera:
- Sensor nodes carry GPS coordinates from the `SensorReading` (if the device has GPS)
- Camera nodes have no GPS — they appear as isolated nodes with no edges unless you
  manually assign coordinates

### Haversine distance

Neighbour linking uses the Haversine formula to compute great-circle distance:

```python
def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0   # Earth radius in metres
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return 2 * R * math.asin(math.sqrt(a))
```

Two nodes become neighbours if their distance is below `proximity_radius_m` (default
100 m). Edges are bidirectional and carry the distance as a weight.

### Why GPS proximity matters

A CO₂ spike in sensor A is more significant if sensor B (10 m away) also shows
elevated CO₂. Without proximity information you cannot distinguish "whole-area event"
from "single-sensor anomaly". The mesh links make cross-sensor correlation possible
without a database query.

The `GET /site/mesh` endpoint exposes the full graph so a dashboard can render sensor
nodes on a satellite image with proximity edges overlaid.

---

## 5. Acoustic Analysis

`SoundAnalyzer` (`sensors/sound_analyzer.py`) extracts audio from a Frigate camera's
RTSP stream and runs two independent analyses per 4-second chunk.

### Audio capture

A subprocess call to `ffmpeg` captures raw PCM:

```bash
ffmpeg -nostdin -loglevel error \
  -rtsp_transport tcp -i rtsp://... \
  -t 4 -vn \
  -acodec pcm_s16le -ar 16000 -ac 1 \
  -f s16le pipe:1
```

Output is piped directly to stdout as raw int16 samples at 16 kHz mono.
`subprocess.run()` with `capture_output=True` reads the entire chunk in one call.
The timeout is `chunk_sec + 10` seconds to handle slow-starting RTSP connections.

**Why not use an asyncio subprocess?** The chunk duration is 4 seconds — a blocking
`subprocess.run()` run in `asyncio.to_thread()` is simpler and safer than managing
an async subprocess pipe that may block on RTSP reconnections.

### FFT-based acoustic event classification

```python
float_audio = audio.astype(np.float64) / 32768.0
spectrum = np.abs(np.fft.rfft(float_audio))
freqs = np.fft.rfftfreq(len(float_audio), d=1.0 / _SAMPLE_RATE)
total_energy = float(np.sum(spectrum ** 2)) or 1.0
```

`rfft` returns the one-sided spectrum of a real signal (N/2 + 1 bins for N samples).
`rfftfreq` maps each bin index to its frequency in Hz.

For each acoustic signature, the ratio of band energy to total energy is computed:

```python
mask = (freqs >= f_lo) & (freqs <= f_hi)
band_energy = float(np.sum(spectrum[mask] ** 2))
ratio = band_energy / total_energy
```

The four built-in signatures with their frequency bands:

| Event | Band | Threshold | Physical basis |
|---|---|---|---|
| `alarm` | 2-4 kHz | 0.40 | Tonal electronic alarms concentrate energy in a narrow mid-frequency band |
| `engine` | 80-400 Hz | 0.35 | Motor/engine fundamental harmonics fall in this range |
| `impact` | 200 Hz-2 kHz | 0.50 | Broadband transient (e.g. metal impact, door slam) elevates energy across this wide band |
| `glass` | 3-8 kHz | 0.30 | High-frequency crinkle of breaking glass |

The threshold is the minimum energy ratio for the event to be reported. All four
checks run on every non-silent chunk — multiple events can co-occur (e.g. an engine
impact produces both `engine` and `impact`).

**Limitations of the spectral classifier:**
- No temporal modelling — a single chunk with elevated energy in 2-4 kHz is reported
  as an alarm regardless of duration, repetition, or modulation pattern.
- Overlapping signatures — a loud voice conversation concentrates energy in 200 Hz-4 kHz
  and can trigger `impact` or `alarm` thresholds.
- No noise floor adaptation — a noisy site (truck traffic) will have elevated low-frequency
  energy that artificially suppresses the `engine` ratio.

These limitations are acceptable at the prototype stage. A real deployment would add
temporal smoothing, per-camera noise floor tracking, and possibly a learned
mel-spectrogram classifier.

### Faster-Whisper transcription

For non-silent chunks, `_transcribe()` runs `faster-whisper` with the `tiny` model
in `int8` mode on CPU:

```python
self._whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
segments, _ = self._whisper.transcribe(float_audio, language=None, beam_size=1)
```

The model is loaded lazily on first call and cached on the `SoundAnalyzer` instance.
`language=None` enables automatic language detection. `beam_size=1` is greedy decoding
— faster but lower accuracy than the default beam_size=5.

The `tiny` model is 75 MB and runs in ~0.5-1 s per 4-second chunk on a modern CPU.
For production with many cameras, either upgrade to a GPU or deploy one analyzer
process per camera with process-level concurrency.

### AcousticObservation

Each chunk produces an `AcousticObservation` dataclass:

```python
@dataclass
class AcousticObservation:
    camera: str
    recorded_at: datetime
    chunk_duration_sec: float
    speech_transcript: str | None
    acoustic_events: list[dict]   # [{"event": "alarm", "energy_ratio": 0.43}, ...]
    rms_db: float                 # roughly dBFS
    silence: bool                 # True if rms_db < -45
```

In `CoopStreamService`, acoustic events are injected as synthetic `CameraEvent` objects
back into `SiteStateAggregator`. This means acoustic alarms appear in the site state
snapshot, in `GET /site/cameras`, and in the `SceneSynthesizer` prompt — acoustic and
visual evidence are merged into a single state model.

---

## 6. RTSP Bridge — Frigate Cameras into MediaMTX

`FrigateRtspBridge` (`sensors/rtsp_bridge.py`) bridges Frigate's RTSP re-streams
into the selfsuvis MediaMTX instance and starts live captioning for each camera.

### Discovery loop

Every `refresh_interval_sec` seconds (default 60), `_sync_cameras()`:

1. Calls `GET {COOP_FRIGATE_API_URL}/api/cameras` to list enabled cameras
2. Computes `live - current` (new cameras to start)
3. Computes `current - live` (removed/disabled cameras to stop)
4. Starts or stops each delta camera

This means cameras added or disabled in Frigate are picked up automatically without
restarting the API.

### Per-camera startup

For each new camera:

1. **MediaMTX path registration**: `MediaMtxClient.ensure_path(path_name="coop/{camera}", source_url="rtsp://frigate:8554/{camera}")`. MediaMTX will pull the RTSP stream from Frigate and re-publish it at `rtsp://mediamtx:8554/coop/{camera}`.

2. **RtspCaptioner session**: `RealtimeStreamManager.start(session_id="coop-{camera}", mission_id="coop-live-{camera}", path_name="coop/{camera}")`. This starts a background task that reads frames from the MediaMTX path, runs Florence-2 (or Gemma) captioning, and writes captions + structured facts to `scene_timeline` in PostgreSQL.

### Why MediaMTX in the middle?

Frigate already provides an RTSP re-stream. Why add MediaMTX?

- **Decoupling**: the RtspCaptioner connects to MediaMTX, not Frigate. If Frigate
  restarts, MediaMTX buffers the reconnect — the captioner doesn't need to handle it.
- **Multi-consumer**: MediaMTX allows multiple readers of the same path. The captioner,
  a live dashboard viewer, and a recording process can all connect to `coop/{camera}`
  simultaneously without Frigate needing to support multiple RTSP clients.
- **Lifecycle control**: the selfsuvis API owns path lifecycle via the MediaMTX control
  API. Paths are created and deleted programmatically, not by editing a config file.

---

## 7. Scene Synthesis — Multi-Modal LLM Narrative

`SceneSynthesizer` (`mesh/scene_synthesis.py`) fuses all available evidence into a
single LLM-generated narrative at `GET /site/synthesis`.

### Input assembly

The synthesizer pulls from two sources:

**SiteState** (from `SiteStateAggregator`):
- All LoRaWAN sensor summaries (temperature, humidity, CO₂, motion, battery)
- All camera event summaries (recent detected objects, total event count)
- `active_motion` aggregate flag

**scene_timeline** (PostgreSQL):
```sql
SELECT mission_id, ts, caption, facts_json
FROM scene_timeline
WHERE ts > now() - interval '5 minutes'
ORDER BY ts DESC
LIMIT 20
```
These are the captions written by `RtspCaptioner` from the MediaMTX bridge, so they
describe what the camera actually sees in near-real-time prose.

### Prompt structure

`_build_prompt()` assembles a Markdown-formatted prompt:

```markdown
## Current Site Sensor State

### Environmental Sensors
  device=aabbccdd, seen=2026-05-01T10:00:00+00:00, temp=22.5°C, humidity=65%, motion=True

### Camera Detections (last 2 min)
  camera=entrance objects=[person(0.91),car(0.73)] motion_events=4

### Live Scene Captions (most recent)
  [coop-live-entrance@2026-05-01T10:01:30+00:00] A person walks toward the entrance gate...

Motion detected by LoRaWAN: True

Respond in JSON matching this schema:
{"narrative":"...","threat_summary":"...","dominant_activities":[...],...}
```

### LLM call

The prompt is sent to `REASONING_API_URL/v1/chat/completions` (OpenAI-compatible) with:
- `max_tokens = settings.REASONING_MAX_TOKENS_COMPACT` (typically ~400)
- `temperature = 0.3` — low enough for factual consistency, high enough to avoid
  degenerate repetition
- Timeout from `settings.REASONING_TIMEOUT_SEC`

The system prompt instructs the model to be factual and admit when data is absent
rather than inventing details.

### Response parsing

`_parse_llm_response()` extracts the first JSON object from the response text
(handling markdown fences the model may add). It maps directly to `SceneSynthesis`
fields. If JSON extraction fails, the raw text is stored in `narrative` and all
structured fields default to empty.

### Caching

The synthesizer holds a 10-second cache (configurable `cache_sec`). The `asyncio.Lock`
prevents duplicate LLM calls when concurrent requests arrive during a cache miss.

**Why 10 seconds?** LoRaWAN sensors update every 1-15 minutes. Camera events update
faster but the deque window averages to a stable picture. A 10-second cache hits the
sweet spot between LLM cost and freshness. Operators who need a real-time view should
use the WebSocket `/site/stream` endpoint (pushes raw `SiteState` without LLM) and
trigger synthesis only when they want a human-readable summary.

---

## 8. Threat Pipeline Integration

`CoopRealtimeIngestor` (`pipeline/realtime/coop_ingest.py`) bridges coop
observations into the selfsuvis `RealtimeThreatAggregator`.

### GPS grid sectors

LoRaWAN sensors with GPS produce a sector ID:

```python
_GRID_DEG = 0.001   # ~110 m per cell at mid-latitudes

def _sector_from_gps(lat, lon) -> str:
    lat_cell = math.floor(lat / _GRID_DEG)
    lon_cell = math.floor(lon / _GRID_DEG)
    return f"grid:{lat_cell}:{lon_cell}"
```

Sensors within 110 m of each other map to the same sector. Multiple sensors in the
same sector produce multiple `SensorEvent` objects with the same `sector_id`, which
the aggregator treats as cross-sensor support — increasing threat score confidence.

### SensorEvent from LoRaWAN

```python
SensorEvent(
    event_time=reading.received_at,
    ingest_time=now,
    node_id=reading.dev_eui,
    sensor_type="lorawan",
    sector_id=grid_sector,
    payload={
        "temperature_c": 22.5,
        "motion": True,
        "rssi": -70.0,
        "snr": 8.0,
    }
)
```

`SensorEvent` objects are consumed by `RealtimeThreatAggregator.consume()` but
**do not directly affect threat scores**. They inform the health and degraded-mode
logic (enough sensors reporting = high automation confidence).

### ThreatEvent from Frigate

```python
ThreatEvent(
    event_time=event.started_at,
    ingest_time=now,
    node_id="frigate:entrance",
    sensor_type="camera",
    sector_id="unknown",   # cameras have no GPS; override via camera_sector_map
    payload={
        "threat_type": "camera_detection",
        "score": 0.87,
        "label": "person",
        "camera": "entrance",
    }
)
```

`ThreatEvent` objects **do affect threat scores**. The aggregator accumulates events
per sector and applies a probabilistic fusion formula:

```python
remaining = 1.0
for score in score_terms:
    remaining *= (1.0 - max(0.0, min(1.0, score)))
aggregated = 1.0 - remaining   # independent-probability combination
```

A single `person` detection at 0.87 confidence → threat score ≈ 0.87. Two independent
detections at 0.87 each → threat score ≈ 0.98. This makes cross-sensor agreement
produce a higher risk level even when individual scores are moderate.

### Camera sector override

Frigate cameras have no GPS, so their sector defaults to `"unknown"`. Use the
`camera_sector_map` constructor argument to assign sectors:

```python
CoopRealtimeIngestor(
    threat_agg,
    camera_sector_map={
        "entrance": "grid:48856:2352",
        "parking": "grid:48857:2351",
    }
)
```

Alternatively, configure sectors by querying a site map at startup. The current code
does not do this automatically — it is a deliberate gap to avoid hardcoding sensor
coordinates.

### GET /site/threat

The threat snapshot from `GET /site/threat` is compatible with the robot advisory
schema at `POST /query/pose`. A robot entering a sector with high threat score
will receive an `abort` or `reroute` advisory from the standard route advisory logic.

---

## 9. What To Inspect and Common Failure Modes

### Artifacts and endpoints to inspect

| What to check | Where |
|---|---|
| MQTT messages arriving | `docker logs coop-mosquitto` — watch for `CONNECT` and `PUBLISH` lines |
| LoRaWAN device uplinks | ChirpStack UI at `http://localhost:8080` → Applications → Devices |
| Frigate detections | Frigate UI at `http://localhost:8971` → Events |
| Site state snapshot | `GET /site/state` — all sensors and cameras should appear |
| Active RTSP sessions | `GET /site/cameras` — each Frigate camera should have a `session_id` |
| Live captions in DB | `SELECT * FROM scene_timeline ORDER BY ts DESC LIMIT 10;` |
| Scene synthesis | `GET /site/synthesis` — narrative should mention active sensors/cameras |
| Threat snapshot | `GET /site/threat` — sectors with detections should have non-zero `threat_score` |

### Common failure modes

**Symptom: `GET /site/state` returns empty sensors and cameras**

Likely causes:
1. `COOP_MQTT_HOST` points at the wrong broker — the subscriber keeps reconnecting
2. Mosquitto not reachable from the API container — check `selfsuvis-net` membership
3. ChirpStack/Frigate MQTT credentials wrong — check subscriber logs for `AUTH` errors
4. Rolling window too short — all readings are older than `COOP_SENSOR_WINDOW_SEC`

**Symptom: `GET /site/cameras` shows cameras but no `session_id`**

The `FrigateRtspBridge` discovered cameras but `RealtimeStreamManager.start()` failed.
Check API logs for `captioner start failed for {camera}`. Usually means MediaMTX is
unreachable or the `MEDIAMTX_API_URL` is wrong.

**Symptom: `GET /site/synthesis` returns `"synthesis timeout"` or `"synthesis error"`**

1. `REASONING_API_URL` is wrong or the backend is down
2. `REASONING_TIMEOUT_SEC` is too short for the model to respond
3. The LLM returned malformed JSON — check logs for the raw response

**Symptom: SoundAnalyzer not producing events despite sound**

1. `ffmpeg` not on PATH inside the API container — check `which ffmpeg`
2. RTSP stream from Frigate camera has no audio track — not all cameras have audio
3. `faster-whisper` not installed — install with `pip install faster-whisper`
4. Energy ratios below thresholds — lower `_ACOUSTIC_SIGNATURES` thresholds for your environment

**Symptom: All camera events map to `sector_id="unknown"` in threat snapshot**

Normal until `camera_sector_map` is configured. Pass a dict mapping camera names to
sector IDs when constructing `CoopRealtimeIngestor`. Sector IDs for your site can be
computed from known camera GPS coordinates using the same `_sector_from_gps()` formula.

---

## 10. Study Exercises

**Exercise 1 — MQTT inspection**

Start the coop stack, then:
```bash
docker exec -it coop-mosquitto mosquitto_sub -t '#' -u health -P health
```
Watch the raw MQTT messages. Identify ChirpStack uplink payloads, Frigate event
payloads, and any other topics. Decode one ChirpStack payload by hand using the
JSON structure in `lorawan_decoder.py`.

**Exercise 2 — Rolling window eviction**

Write a small test that inserts 10 `SensorReading` objects with timestamps spanning
8 minutes into a `SiteStateAggregator` with a 5-minute window. Call `get_state()`
and verify that only the 5-minute window of readings is returned. Then advance the
timestamps and verify eviction works correctly.

**Exercise 3 — FFT acoustic analysis**

Generate a synthetic sine wave at 3 kHz using numpy (`np.sin(2 * np.pi * 3000 * t)`)
and pass it to `_classify_acoustic_events()`. Verify that the `alarm` signature
fires (3 kHz is in the 2-4 kHz alarm band). Then try a mix of two sine waves
at 200 Hz and 3 kHz and check which signatures fire.

**Exercise 4 — Scene synthesis prompt**

Call `_build_prompt()` with a constructed `SiteState` that has two sensors (one with
motion, one without) and one camera with two recent detections. Print the prompt.
Verify that the prompt would give an LLM enough information to write a useful narrative.
Then identify what information is missing that you would want in a real deployment.

**Exercise 5 — Threat sector design**

Your site has 4 LoRaWAN sensors at these GPS coordinates:
- Sensor A: 48.8566, 2.3522
- Sensor B: 48.8567, 2.3523
- Sensor C: 48.8600, 2.3600
- Sensor D: 48.8601, 2.3601

Use `_sector_from_gps()` to compute sector IDs for each. Which sensors share a sector?
What does that mean for threat aggregation? How would you tune `_GRID_DEG` to change
the sector granularity?

**Exercise 6 — End-to-end integration**

Start the full stack with `docker compose -f docker/core/docker-compose.yml -f docker/sencoop/docker-compose.sencoop.yml up -d`.
Configure at least one test camera in Frigate (a test RTSP source or file loop will work).
Watch `GET /site/cameras` until the camera's `session_id` appears.
Run `SELECT caption FROM scene_timeline ORDER BY ts DESC LIMIT 5;` to verify captions.
Then call `GET /site/synthesis` and inspect the narrative.

---

## 11. Further Reading

| Resource | Why |
|---|---|
| MQTT specification v5.0 — [mqtt.org](https://mqtt.org/mqtt-specification/) | Topic wildcards, QoS levels, retained messages |
| ChirpStack documentation — [chirpstack.io/docs](https://www.chirpstack.io/docs/) | Device activation, codec framework, uplink payload structure |
| Frigate documentation — [docs.frigate.video](https://docs.frigate.video) | Detection configuration, RTSP re-stream options, MQTT event schema |
| LoRaWAN specification — [lora-alliance.org](https://lora-alliance.org/resource_hub/lorawan-104-specification-package/) | PHY layer, frame counters, ADR, spreading factors |
| Whisper paper — [2212.04356](https://arxiv.org/abs/2212.04356) | Weak supervision at scale for speech recognition |
| Cohen et al., "Environmental Sound Classification on Microcontrollers" (2022) | Why simple FFT classifiers work for constrained edge deployments |
| [Sensor fusion fundamentals](03_sensor_fusion_fundamentals.md) | Clocks, calibration, and uncertainty — applies directly to LoRaWAN timestamp alignment |
| [Threat primitives and local inference](15_threat_primitives_local_inference.md) | The two-source gate and evidence-gated scoring used by the threat aggregator |
| [coop integration guide](../sencoop/integration.md) | Operator reference: MQTT topics, docker-compose, env vars, API endpoints |
