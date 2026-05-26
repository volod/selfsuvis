# Quick Start — Learning Path Pipeline

Run the full local learning pipeline (`selfsuvis --mode local`) directly on your
machine. It processes videos, runs every enabled perception and captioning stage,
fine-tunes a DINOv3 model on the mission frames, and exports it to ONNX.

The local runner has 33 reported runtime/post-run steps and maps to the 36-step
conceptual learning path. The optional `coop` extension adds Steps 37-43
for live IoT site monitoring after the video pipeline run.

> **Setting up the API/worker/UI service stack instead?**
> See [Quick Start — Local Service Setup](quickstart-local.md).

---

## Two paths to get started

### Path A — One-shot bootstrap (recommended)

`scripts/ssv/ssv-setup.sh` automates **every manual step below** in a single command: it installs the venv, downloads model weights, starts Ollama, downloads a test video, generates sensor sidecars, starts Docker services, and runs the DB migration. It prints the exact `selfsuvis --mode local` run command at the end.

```bash
bash scripts/ssv/ssv-setup.sh
```

Common variants:

```bash
# CPU-only machine — skip Docker and Ollama:
bash scripts/ssv/ssv-setup.sh --no-docker --no-ollama

# Already have models; only want fresh sensor sample data:
bash scripts/ssv/ssv-setup.sh --sensor-data-only

# With HuggingFace token for gated Gemma weights:
HF_TOKEN=hf_xxxx bash scripts/ssv/ssv-setup.sh
```

Already ran setup once? Skip directly to [Step 6 — Run the pipeline](#step-6--run-the-pipeline).
Use `--no-utilyze` if you want to skip the optional Utilyze step.
If you already completed the video pipeline and only want the coop extension, skip
to [Optional Step 7 — Run coop Steps 37-43](#optional-step-7--run-coop-steps-37-43).

---

### Path B — Manual setup

The manual sequence is: **install venv → generate env → download models → start sidecars → get test data → run pipeline**.

Follow Steps 1–5 below, then proceed to [Step 6 — Run the pipeline](#step-6--run-the-pipeline).

---

### Step 1 — Install the venv

```bash
sudo ./scripts/install/install_system_deps.sh --with-python
make venv
```

---

### Step 2 — Generate the local env

```bash
make env               # non-interactive: auto-detects GPU and RAM
# or
make env-interactive   # prompts for sidecar backend, profile, models
```

Then open `.data/.env` and set `API_KEY` (leave blank for unauthenticated local use). `ALLOWED_INDEX_PATHS` is pre-filled to `./.data/videos`.

---

### Step 3 — Download model weights

Run this once before the first pipeline run. All weights are cached locally; subsequent runs skip already-cached models.

**Core models only** (OpenCLIP + DINOv2/v3 — always required):

```bash
.venv/bin/python -m selfsuvis.scripts.prepare_models --clip --dino
```

**Balanced set** (adds Florence-2 captioning, YOLO11, SAM):

```bash
.venv/bin/python -m selfsuvis.scripts.prepare_models \
  --clip --dino --florence --yolo --sam
```

**Full set** (everything including ASR, OCR, depth, detection, world model):

```bash
.venv/bin/python -m selfsuvis.scripts.prepare_models --all
```

**flash-attn** (optional — CUDA only; speeds up Florence-2 and Gemma attention; uses a prebuilt wheel when available, otherwise compiles from source which takes several minutes):

```bash
.venv/bin/python -m selfsuvis.scripts.prepare_models --flash-attn
```

**Reasoning model** (optional — Step 30, agentic flow audit; pulled via Ollama):

```bash
.venv/bin/python -m selfsuvis.scripts.prepare_models --reasoning
# pull a different tag:
.venv/bin/python -m selfsuvis.scripts.prepare_models --reasoning --reasoning-model deepseek-r1:14b
```

Default tag is `qwen3:14b` (~8 GB). `deepseek-r1:14b` (~9 GB) is a strong alternative. The pipeline auto-selects a reasoning model when this flag is omitted, but pulling it in advance avoids a cold-start delay at step 30.

**SceneTok** (optional — Step 14, streaming scene encoder + segmentation decoder; requires **~24 GB VRAM**, RTX 4090 minimum):

```bash
.venv/bin/python -m selfsuvis.scripts.prepare_models --scenetok
```

This installs the scenetok package from GitHub, downloads `va-videodc_re10k.ckpt` from MPI Nextcloud (public, no login required), and fetches its HuggingFace dependencies (`hustvl/vavae-imagenet256-f16d32-dinov2`, `hpcai-tech/Open-Sora-v2-Video-DC-AE`). Checkpoint variants available: `va-videodc_re10k` (default, RealEstate10K), `va-videodc_dl3dv`, `va-wan_dl3dv`.

```bash
# Non-default checkpoint variant:
.venv/bin/python -m selfsuvis.scripts.prepare_models \
  --scenetok --scenetok-checkpoint va-videodc_dl3dv
```

> **Segmentation decoder note:** The base SceneTok checkpoint produces novel-view RGB renders via a rectified flow decoder. The segmentation decoder — which replaces the RGB head with a mask output head to produce per-frame 3D-stable segmentation masks — must be fine-tuned separately. Pass `--scenetok-checkpoint` to point the pipeline at a trained segmentation checkpoint.

**Gemma open-weight** (optional — loads locally instead of via Ollama sidecar; requires `HF_TOKEN` in `.data/.env` and license accepted at `huggingface.co/google/gemma-3-4b-it`):

```bash
.venv/bin/python -m selfsuvis.scripts.prepare_models --gemma
# or a smaller variant:
.venv/bin/python -m selfsuvis.scripts.prepare_models \
  --gemma --gemma-model google/gemma-3-1b-it
```

**Check what is already cached** (no downloads):

```bash
.venv/bin/python -m selfsuvis.scripts.prepare_models --verify
.venv/bin/python -m selfsuvis.scripts.prepare_models --verify --all
```

The local CLI now runs its own startup preflight before `--mode local` begins.
It checks Python dependencies, local model caches, Ollama/vLLM-local model presence,
and common degraded-runtime conditions such as unreachable Qdrant or mapper services.
If a required local artifact is missing, the run stops before frame extraction.

The API and worker run the same production preflight at startup in logging-only mode.
To make production startup fail hard on preflight errors, set:

```bash
export STARTUP_PREFLIGHT_STRICT=true
```

**Per-step model reference:**

| Flag | Pipeline step | Default model |
|---|---|---|
| `--clip` | Step 2 — embedding | `ViT-B-16 / openai` |
| `--dino` | Step 2 — embedding | `dinov2_vitb14`, `dinov3_vitb14` |
| `--flash-attn` | — (attention kernel) | prebuilt wheel or source build (CUDA only) |
| `--florence` | Step 4 — captioning | `microsoft/Florence-2-large` |
| `--whisper` | Step 5 — ASR | `openai/whisper-large-v3-turbo` |
| `--ocr` | Step 6 — OCR | auto-selected by VRAM |
| `--depth` | Step 7 — depth estimation | auto-selected by VRAM |
| `--detection` | Step 8 — object detection | auto-selected by VRAM |
| `--yolo` | Step 9 — YOLO11 detection | `yolo11l` (~48 MB) |
| `--sam` | Step 9 — SAM segmentation | `facebook/sam3` (falls back to `sam2-hiera-large`) |
| `--world-model` | Step 11 — world model video embeddings | auto-selected by VRAM |
| `--gemma` | Step 3 — scene analysis | `google/gemma-3-4b-it` |
| `--unidrive` | Step 13 — UniDriveVLA | `owl10/UniDriveVLA_Nusc_Base_Stage3` |
| `--scenetok` | Step 14 — SceneTok streaming encoder + segmentation decoder | `va-videodc_re10k.ckpt` (github.com/mohammadasim98/scenetok) |
| `--reasoning` | Step 30 — agentic flow audit | `qwen3:14b` (Ollama); alt: `deepseek-r1:14b` |

---

### Step 3b — Prepare drone audio dataset (Step 32)

Step 32 of the local pipeline trains a small `DroneAudioCNN` on `geronimobasso/drone-audio-detection-samples`.
The dataset is cached in `.data/drone-audio-data/`; download and split it once before the first run.
`selfsuvis-setup.sh` does this automatically (Step 4d). For manual setup:

```bash
# Download and split into train/val/test (≈200 MB, no login required):
.venv/bin/python -m selfsuvis.scripts.prepare_audio_data

# Custom cache directory:
.venv/bin/python -m selfsuvis.scripts.prepare_audio_data \
  --data-dir /mnt/.data/drone-audio-data

# Verify an existing split (no network, no writes):
.venv/bin/python -m selfsuvis.scripts.prepare_audio_data --verify

# Limit to 100 samples per class for a quick smoke-test:
.venv/bin/python -m selfsuvis.scripts.prepare_audio_data --max-per-class 100
```

Or via the installed entry point:

```bash
ssv-prepare-audio --verify
ssv-prepare-audio --data-dir .data/drone-audio-data
```

Step 32 runs automatically on each local pipeline run. To skip it:

```bash
.venv/bin/selfsuvis --mode local --videos-dir .data/videos --no-drone-audio
```

To control training epochs:

```bash
# Override epochs (default: DRONE_AUDIO_EPOCHS env var, or 10):
.venv/bin/selfsuvis --mode local --videos-dir .data/videos --drone-audio-epochs 20
```

**Simulate drone sound** (useful for testing the trained ONNX model):

```bash
# Drone flyover from 200 m at 10 m/s — classic counter-UAS test case:
./scripts/audio/play_drone_sound.sh --scenario flyover --distance 200 --speed 10

# Hovering drone at 30 m altitude:
./scripts/audio/play_drone_sound.sh --scenario hover --distance 30 --duration 15

# Override speaker/output calibration when auto-detected volume is not enough:
./scripts/audio/play_drone_sound.sh --scenario flyover --distance 80 \
  --speaker-ref-db 78 --system-volume 0.6

# Ask for microphone/speaker placement guidance:
./scripts/audio/play_drone_sound.sh --placement-help \
  --mic-type headset --player-type laptop --probe-distance-m 0.3

# Save to WAV instead of playing:
./scripts/audio/play_drone_sound.sh --scenario approach --distance 500 --speed 20 \
  --output .data/reports/sim_approach_500m.wav
```

---

### Step 4 — Start sidecars

The Ollama sidecars serve Gemma (scene analysis) and the reasoning model. They must be running before `selfsuvis --mode local` starts.

```bash
ollama serve                              # keep running in a terminal
ollama pull <GEMMA_API_MODEL>             # value from .data/.env, e.g. gemma4:e4b
ollama pull qwen3:14b                     # Step 30 reasoning model (or deepseek-r1:14b)
```

If using vLLM instead (set during `make env-interactive`):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <GEMMA_API_MODEL> --port 8000 --max-model-len 8192
```

Skip this step entirely if you downloaded Gemma open-weight locally in Step 3 — the pipeline will load it directly.

**SceneTok sidecar** (optional — preferred runtime for Step 14 when served from another GPU or host):

SceneTok uses a Hydra-based CLI (`python -m src.main`), not an OpenAI-compatible API, so it needs a thin FastAPI wrapper (same pattern as the nerfstudio sidecar). This repo now ships that wrapper as the packaged module `selfsuvis.scripts.scenetok_server`.

Start it before the pipeline if you want sidecar mode:

```bash
SCENETOK_CHECKPOINT=va-videodc_re10k \
  python -m selfsuvis.scripts.scenetok_server
```

Then set:

```bash
export SCENETOK_API_URL=http://localhost:8040
```

If `SCENETOK_API_URL` is absent or unreachable, the pipeline falls back to loading SceneTok locally, which still requires a roughly 24 GB class GPU.

---

### Step 5 — Test video and sensor data

The pipeline needs at least one video in `.data/videos/` and, for sensor fusion, matching sidecar files.

> **Note on step labels in this section:** the "Step N" labels in the sensor tables below are *sensor-type identifiers* used by `ssv-prepare-sensor-data.sh` to name its output directories (`step09_rf/`, `step10_thermal/`, ...). They are a data-organization convention and are separate from the pipeline runner's execution steps (1-32).

#### Option A — Use your own footage

```bash
cp /path/to/mission.mp4 .data/videos/
```

Then generate sensor sidecars keyed to that video's basename (the script auto-detects the file):

```bash
bash scripts/ssv/ssv-prepare-sensor-data.sh .data/sensors
```

#### Option B — Download a public-domain test video

`ssv-setup.sh` handles download, trim, and sensor sidecar generation automatically:

```bash
bash scripts/ssv/ssv-setup.sh --sensor-data-only
```

This downloads the US Highway 60 drone flyover (~27 MB, no login), trims it to 10 s, and generates all sensor sidecars in one step. If the CDN is unreachable it falls back to a second archive.org clip, then generates a synthetic video with ffmpeg.

#### What `ssv-prepare-sensor-data.sh` does

The script creates one directory per sensor step under `.data/sensors/` and generates synthetic sidecar files named after the video currently in `.data/videos/` — so they are immediately usable by the pipeline without any renaming.

| Directory | Step | What is generated / downloaded |
|---|---|---|
| `step09_rf/` | Step 9 | SigMF meta file (synthetic); RadioML dataset requires manual download |
| `step10_thermal/` | Step 10 | README + download notes; FLIR ADAS requires registration |
| `step11_multispectral/` | Step 11 | Indian Pines + Salinas `.mat` files (auto-downloaded) |
| `step12_event_camera/` | Step 12 | README + download notes; N-Caltech101 requires manual download |
| `step13_lidar/` | Step 13 | README + download notes; KITTI velodyne requires registration |
| `step14_radar/` | Step 14 | README + download notes; RADIATE requires manual download |
| `step15_gnss_satellite/` | Step 15 | OpenSky ADS-B JSON (live API) + synthetic JSONL sidecar |
| `step16_imu/` | Step 16 | Synthetic IMU, barometer, and wind JSONL sidecars (200 Hz / 5 Hz / 1 Hz) |
| `step17_atmospheric/` | Step 17 | Synthetic atmospheric JSONL sidecar (temp, humidity, pressure, wind) |
| `step18_gas_radiation/` | Step 18 | Open-Meteo AQI JSON (live, no key) + synthetic gas JSONL sidecar |
| `step19_acoustic/` | Step 19 | ESC-50 metadata CSV (auto-downloaded) + synthetic acoustic JSONL sidecar |

Steps that require registration (9, 10, 12, 13, 14, 15) print manual download instructions and create placeholder directories; the pipeline degrades gracefully when their data is absent.

#### Copy sidecars next to the video

Generated sidecars must sit in the same directory as the video and share its basename:

```bash
VIDEO_BASE="drone_mission"   # change to match your file

cp .data/sensors/step16_imu/${VIDEO_BASE}.imu.jsonl    .data/videos/
cp .data/sensors/step16_imu/${VIDEO_BASE}.baro.jsonl   .data/videos/
cp .data/sensors/step16_imu/${VIDEO_BASE}.wind.jsonl   .data/videos/
cp .data/sensors/step17_atmospheric/${VIDEO_BASE}.env.jsonl  .data/videos/
cp .data/sensors/step18_gas_radiation/${VIDEO_BASE}.gas.jsonl .data/videos/
cp .data/sensors/step19_acoustic/${VIDEO_BASE}.audio.wav      .data/videos/
# ... repeat for any other steps you have data for
```

Full sidecar naming reference:

| Filename pattern | Sensor step |
|---|---|
| `<video>.iq` or `<video>.sigmf-data` | Step 9 — RF/SDR (float32 I/Q) |
| `<video>.thermal.mp4` or `<video>.thermal/` | Step 10 — thermal (GREY16 video or TIFF sequence) |
| `<video>.multispectral/` | Step 11 — per-band GeoTIFF directory |
| `<video>.events.raw` or `<video>.events.h5` | Step 12 — event camera stream |
| `<video>.lidar.pcd` or `<video>.lidar.mcap` | Step 13 — LiDAR point cloud |
| `<video>.radar.bin` or `<video>.radar.csv` | Step 14 — radar ADC IQ or detections |
| `<video>.adsb.jsonl` or `<video>.gnssr.bin` | Step 15 — ADS-B aircraft log or GNSS-R IQ |
| `<video>.imu.jsonl` | Step 16 — IMU (200 Hz) |
| `<video>.baro.jsonl` | Step 16 — barometer (5 Hz) |
| `<video>.wind.jsonl` | Step 16 — anemometer (1 Hz) |
| `<video>.env.jsonl` | Step 17 — atmospheric (temp, humidity, pressure) |
| `<video>.gas.jsonl` | Step 18 — gas / radiation (CO2, VOC, PM2.5, dose rate) |
| `<video>.audio.wav` or `<video>.audio_array.h5` | Step 19 — acoustic (48 kHz WAV or mic array) |

---

## Step 6 — Run the pipeline

**Minimal run** — fewest dependencies, in-memory store, no 3D reconstruction:

```bash
.venv/bin/selfsuvis --mode local \
  --videos-dir .data/videos \
  --no-qdrant \
  --no-sfm \
  --no-gsplat
```

For the highest-quality local run on this profile, keep `SELFSUVIS_USE_GRAPH=1`
in `.data/.env` or export it before the command. The graph path retries Qwen structured
caption parse failures and usually produces better `detailed_captions.md`.

**Standard run** — with Qdrant for base-model search (Step 15) and fine-tuned search (Step 24):

```bash
# Start Qdrant if not already running:
docker compose -f docker/core/docker-compose.yml up -d qdrant

.venv/bin/selfsuvis --mode local \
  --videos-dir .data/videos
```

The local pipeline uses `.data/videos` only. `data/videos` is not supported.

**Fast iteration** — skip slow optional steps (ASR, OCR, depth, captioning):

```bash
.venv/bin/selfsuvis --mode local \
  --videos-dir .data/videos \
  --no-qdrant \
  --no-sfm \
  --no-gsplat \
  --no-caption \
  --no-asr \
  --no-ocr \
  --no-depth
```

## Learning-path outputs to inspect after a full run

If you are using this quickstart as the practical entrypoint into the learning path, inspect these
artifacts first after the run finishes:

1. `final_stats.md`
   Step timings, skipped stages, and top-level warnings.
2. `analysis_summary.json`
   Compact machine-readable truth about coverage, diagnostics, degraded stages, and warnings.
3. `detailed_captions.md`
   Confirms whether Qwen produced valid structured output or only parse errors.
4. `unidrive_analysis.md`
   Domain-structured VLM analysis for the same mission frames.
5. `3d_map/map_stats.json`
   What the mapper actually recovered: SfM poses, anchors, points, and fallback path.
6. `3d_map/map_quality_advisor.md`
   Why the map quality is good or poor from a capture-quality perspective.
7. `<video_name>/local_threat_assessment.json`
   Clip-level threat score, automation confidence, top threats, and contradiction metrics. Check
   `local_threat_score` and `recommended_action` first.
8. `<video_name>/threat_primitives.json`
   The raw evidence behind the threat score: each primitive includes `type`, `score`,
   `uncertainty`, `persistence_sec`, and `evidence_sources`.
9. `global_threat_summary.json` *(root of output dir — appears after all videos complete)*
   Cross-video aggregated threat: sector risk levels, persistent anomalies, and route advisories.
10. `model_run_advisor.md` *(root of output dir — appears after all videos complete)*
   Post-run findings plus recommended `.data/.env` updates, model pulls, and rerun commands for the current machine.

For a full artifact-by-artifact walkthrough, read:

- [Local run artifact analysis](../learning_path/08_local_run_artifact_analysis.md)
- [Tracking, world models, and 3D mapping](../learning_path/05_tracking_mapping_steps_21_27.md)
- [Threat primitives and local inference](../learning_path/15_threat_primitives_local_inference.md)
- [Local run analytics](../reference/analytics.md)

## Optional Step 7 — Run coop Steps 37-43

`coop` is not part of a single `selfsuvis --mode local` video run. It is the
continuous site-awareness extension after the local learning pipeline. Use this
when you want to practice Steps 37-43 locally: MQTT/LoRaWAN ingestion, Frigate
events, rolling site state, RTSP/acoustic evidence, site mesh, scene synthesis,
and realtime threat sectors.

### 7.1 Install coop extras

If you used `make venv`, install the optional coop dependencies into the same venv:

```bash
.venv/bin/pip install -e ".[coop]"
```

### 7.2 Start the coop stack

Use `APP_ENV=test` for a localhost-bound learning stack. Bootstrap creates
`.data/.env`, writable data directories, Mosquitto TLS certs, and MQTT users when
they are missing.

```bash
APP_ENV=test ./scripts/coop/coop-bootstrap.sh up -d
```

Check container health:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

Expected coop containers include:

- `coop-mosquitto`
- `coop-chirpstack`
- `coop-cs-gwbridge`
- `coop-cs-rest`
- `coop-cs-postgres`
- `coop-cs-redis`
- `coop-frigate`

### 7.3 Start the local API with coop enabled

Run this after Step 4/5 from the local service setup, so Postgres is up and migrated.
The API starts the coop MQTT subscriber during FastAPI lifespan startup.

```bash
APP_ENV=dev \
COOP_MQTT_HOST=localhost \
COOP_MQTT_PORT=1883 \
COOP_MQTT_TLS=false \
COOP_FRIGATE_API_URL=http://localhost:8971 \
COOP_CHIRPSTACK_TOPIC='application/+/device/+/event/up' \
COOP_FRIGATE_TOPIC_PREFIX=frigate \
.venv/bin/uvicorn selfsuvis.app.main:app --host 0.0.0.0 --port 8000
```

Set `REASONING_API_URL` as well if you want `/site/synthesis` to call an
OpenAI-compatible reasoning backend:

```bash
REASONING_API_URL=http://localhost:11434/v1
```

### 7.4 Inspect the coop endpoints

In another terminal:

```bash
curl -s http://localhost:8000/site/state | python -m json.tool
curl -s http://localhost:8000/site/sensors | python -m json.tool
curl -s http://localhost:8000/site/cameras | python -m json.tool
curl -s http://localhost:8000/site/mesh | python -m json.tool
curl -s http://localhost:8000/site/synthesis | python -m json.tool
curl -s http://localhost:8000/site/threat | python -m json.tool
```

At first these may be empty. They populate when ChirpStack publishes uplinks on
`application/+/device/+/event/up` or Frigate publishes detection events under
`frigate/#`.

### 7.5 Run the coop checks

```bash
.venv/bin/python -m pytest -q tests/coop tests/unit/coop/test_sensor_decoders.py
```

For a broader check of the site-state API and realtime threat bridge:

```bash
.venv/bin/python -m pytest -q \
  tests/coop \
  tests/unit/coop/test_sensor_decoders.py \
  tests/unit/app/routers/test_site_state.py \
  tests/unit/pipeline/realtime
```

### 7.6 Stop the coop stack

```bash
APP_ENV=test ./scripts/coop/coop-compose.sh down
```

The short study sequence is in [Local Learning Path](local_path.md#coop-extension-steps).
The deep dive is [coop IoT edge monitoring](../learning_path/16_coop_iot_edge_monitoring.md).

**Full run — realistic single GPU (12 GB minimum, sequential Ollama sidecars)**:

This is the highest-step-count recipe that is still realistic on a single 12-16 GB consumer GPU.
It keeps only one Ollama model resident at a time, avoids SceneTok entirely, and routes the
"UniDrive" step through the same Qwen vision model instead of a separate UniDriveVLA vLLM server.

Use this when you want the most pipeline coverage on one GPU:

- Keep: ASR, OCR, depth, detection, world model, Gemma scene analysis, Qwen detailed captioning, reasoning
- Keep: `--unidrive`, but point it at Qwen2.5-VL via Ollama
- Omit: `--scenetok`
- Omit: separate UniDriveVLA vLLM process on 12-16 GB cards

```bash
# —— Terminal 1: Ollama daemon —————————————————————————————————————————————————
# Only one sidecar model stays resident at a time.
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=0 \
  ollama serve

# Pull once before the first run:
# 12 GB GPU with 48+ GB RAM: prefer qwen2.5vl:7b + qwen3:14b for quality.
# If you hit OOM or swapping, fall back to qwen2.5vl:3b + qwen3:8b.
ollama pull gemma4:e4b
ollama pull qwen2.5vl:7b
ollama pull qwen3:14b

# —— Terminal 2: pipeline ————————————————————————————————————————————————————————
# Single-GPU realistic maximum:
# - Gemma via Ollama for scene analysis / synthesis
# - Qwen via Ollama for detailed captioning
# - "UniDrive" step also uses the same Qwen sidecar endpoint/model
# - SceneTok disabled
.venv/bin/selfsuvis --mode local \
  --videos-dir        .data/videos \
  --asr               \
  --ocr               \
  --depth             \
  --detection         \
  --qwen              \
  --world-model       \
  --unidrive          \
  --rfdetr-model      base \
  --gemma-api-url     http://localhost:11434/v1 \
  --gemma-api-backend ollama \
  --gemma-api-model   gemma4:e4b \
  --qwen-api-url      http://localhost:11434/v1 \
  --qwen-backend      ollama \
  --qwen-model        qwen2.5vl:7b \
  --unidrive-api-url  http://localhost:11434/v1 \
  --unidrive-backend  ollama \
  --unidrive-model    qwen2.5vl:7b \
  --reasoning-api-url http://localhost:11434/v1 \
  --reasoning-backend ollama \
  --reasoning-model   qwen3:14b
```

> **Why this is the realistic single-GPU path:** Ollama can evict Gemma, Qwen, and reasoning
> models between steps (`OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_KEEP_ALIVE=0`), so the worker
> only has to coexist with one sidecar model at a time. The `UniDriveVLA` step is still useful
> when pointed at Qwen2.5-VL for aerial / off-road missions, and this repo's runbook recommends
> that backend over the driving-specific `owl10/*` checkpoint for arbitrary single-camera video.

### 3D mapping expectations for the learning path

The local runner can now produce a richer degraded map than older versions even when SfM is only
partially successful:

- short clips may auto-switch to exhaustive pycolmap matching
- sparse SfM can be enriched with interpolated anchors, detections, tracks, and coarse depth cues
- every run now writes a map-quality advisor explaining whether the video itself was suitable for a
  high-quality map

This does not remove the underlying capture requirements.
If you want a very high quality 3D map, aim for:

- `25-40 s` over the same area
- `1280x720` minimum, `1920x1080+` preferred
- real lateral parallax, not only forward drift
- at least one oblique pass around `25-40°` off nadir
- lower altitude or tighter FOV so roads, poles, and vehicles occupy meaningful pixels

If `analysis_summary.json` warns that the 3D map is degraded, open:

- `3d_map/map_stats.json`
- `3d_map/map_quality_advisor.md`

before assuming the mapper is misconfigured. In many aerial runs, the real problem is short,
high-altitude, nadir-heavy footage with weak triangulation geometry.

**Full run — maximum steps (Ollama sidecars + local PyTorch where supported)**:

Use Ollama for the LLM / VLM sidecars that are sidecar-only in this repo (Gemma,
Qwen, reasoning), and use in-process PyTorch for optional local-only steps when
there is enough VRAM headroom. `OLLAMA_MAX_LOADED_MODELS=1` evicts each Ollama
model before the next loads; the pipeline sends `keep_alive=0` after each step
so local torch models reclaim VRAM between stages.

What "maximum steps" means in practice:

- Qwen Step 12 still uses Ollama — there is no in-process PyTorch Qwen path in this repo
- UniDrive Step 13 can run locally from cached HF weights
- SceneTok Step 14 can run locally only on roughly 20-24 GB+ VRAM

> On a single 12-16 GB GPU, this usually means: local UniDrive may be possible,
> but local SceneTok is still unrealistic. For that class of card, use the
> realistic single-GPU recipe above and omit `--scenetok`.

```bash
# —— Terminal 1: Ollama daemon —————————————————————————————————————————————————
# OLLAMA_MAX_LOADED_MODELS=1  — only one model resident in VRAM at a time
# OLLAMA_NUM_PARALLEL=1       — single concurrent request (no extra KV caches)
# OLLAMA_KEEP_ALIVE=0         — evict immediately after each request
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=0 \
  ollama serve

# Pull models once (in a separate terminal or before starting the daemon):
ollama pull gemma4:e4b        # Steps 3, 23   — scene analysis, video synthesis      (~5 GB)
ollama pull qwen2.5vl:7b      # Step 12       — detailed captioning                  (~5 GB)
ollama pull qwen3:14b         # Step 30       — agentic flow audit / reasoning        (~8 GB)

# Cache local-only weights once before the run:
# Also warm the YOLOv8n training weights once if you plan to enable Step 31.
# .venv/bin/python -m selfsuvis.scripts.prepare_models --yolo
# .venv/bin/python -m selfsuvis.scripts.prepare_models --unidrive
# .venv/bin/python -m selfsuvis.scripts.prepare_models --scenetok

# —— Terminal 2: pipeline ————————————————————————————————————————————————————————
# All available steps enabled. UniDrive runs locally if HF weights are cached.
# SceneTok runs locally only if the machine has enough VRAM; otherwise omit --scenetok.
.venv/bin/selfsuvis --mode local \
  --videos-dir        .data/videos \
  --asr               \
  --ocr               \
  --depth             \
  --detection         \
  --qwen              \
  --world-model       \
  --unidrive          \
  --scenetok          \
  --rfdetr-model      base \
  --gemma-api-url     http://localhost:11434/v1 \
  --gemma-api-backend ollama \
  --qwen-api-url      http://localhost:11434/v1 \
  --qwen-backend      ollama \
  --unidrive-model    owl10/UniDriveVLA_Nusc_Base_Stage3 \
  --reasoning-api-url http://localhost:11434/v1 \
  --reasoning-backend ollama \
  --reasoning-model   qwen3:14b
```

> **Local UniDrive note:** in-process UniDrive works only when HF weights are
> already cached locally. If `UNIDRIVE_API_URL` is unset and the weights are not
> cached, Step 13 is skipped.
>
> **Local SceneTok note:** if `SCENETOK_API_URL` is unset, SceneTok falls back to
> local torch and is enabled only when enough VRAM is detected. On 12-16 GB cards,
> omit `--scenetok`; on 24 GB+ cards, local sequential execution is realistic.
>
> **SceneTok env note:** `SCENETOK_ENABLED=true` in `.data/.env` is now respected when the
> CLI omits both `--scenetok` and `--no-scenetok`. Only the explicit flags override it.

**Full run — vLLM only** (all LLM/VLM steps via vLLM, including UniDriveVLA):

Each vLLM server reserves a fixed GPU memory fraction (`--gpu-memory-utilization`).
Run servers on separate GPUs when possible; on a single GPU start them one at a time
and let the pipeline's inter-step eviction (`keep_alive=0` equivalent) clear KV caches.

```bash
# —— Terminal 1: Gemma 12B — Steps 3, 23, 24 (scene analysis, video synthesis, audit) —
# Gemma 12B is reused for both scene analysis and the agentic-audit reasoning step.
# ~10 GB VRAM at 0.45 utilisation; use a 24 GB card or reduce utilisation on 16 GB.
python -m vllm.entrypoints.openai.api_server \
  --model              google/gemma-3-12b-it \
  --port               11434 \
  --gpu-memory-utilization 0.45 \
  --max-model-len      8192 \
  --enforce-eager \
  --swap-space         4

# —— Terminal 2: Qwen2.5-VL — Step 12 (detailed captioning) ———————————————————
# ~5 GB VRAM at 0.45 utilisation; start AFTER Gemma is ready if sharing one GPU
python -m vllm.entrypoints.openai.api_server \
  --model              Qwen/Qwen2.5-VL-7B-Instruct \
  --port               8010 \
  --gpu-memory-utilization 0.45 \
  --max-model-len      8192 \
  --enforce-eager \
  --swap-space         4

# —— Terminal 3: UniDriveVLA — Step 13 (expert driving analysis) ————————————————
# ~4 GB VRAM; requires HF weights downloaded via prepare_models --unidrive
python -m vllm.entrypoints.openai.api_server \
  --model              owl10/UniDriveVLA_Nusc_Base_Stage3 \
  --port               8030 \
  --gpu-memory-utilization 0.40 \
  --max-model-len      4096 \
  --enforce-eager \
  --swap-space         4

# —— Terminal 4: SceneTok — Step 14 (streaming scene encoder + segmentation decoder) ——
# ~24 GB VRAM minimum (RTX 4090); NOT served via vLLM — uses a thin FastAPI wrapper.
# SceneTok is a vision encoder-decoder, not a language model, so vLLM does not apply.
# If SCENETOK_API_URL is unset, the pipeline falls back to loading it as a local torch model.
SCENETOK_CHECKPOINT=va-videodc_re10k.ckpt \
  PORT=8040 python -m selfsuvis.scripts.scenetok_server

# —— Terminal 5: pipeline ——————————————————————————————————————————————————————
# Steps enabled: all including unidrive (Step 13) and scenetok (Step 14)
.venv/bin/selfsuvis --mode local \
  --videos-dir        .data/videos \
  --asr               \
  --ocr               \
  --depth             \
  --detection         \
  --qwen              \
  --world-model       \
  --unidrive          \
  --scenetok          \
  --rfdetr-model      base \
  --gemma-api-url     http://localhost:11434/v1 \
  --gemma-api-backend vllm \
  --qwen-api-url      http://localhost:8010/v1 \
  --qwen-backend      vllm \
  --unidrive-api-url  http://localhost:8030/v1 \
  --unidrive-backend  vllm \
  --scenetok-api-url  http://localhost:8040 \
  --reasoning-api-url http://localhost:11434/v1 \
  --reasoning-backend vllm \
  --reasoning-model   google/gemma-3-12b-it
```

> **Memory budget (single 16 GB GPU):**
> Ollama variant — peak is Florence-2 (~4 GB) + CLIP/DINO (~2 GB) = ~6 GB; sidecars evicted before each local step.
> vLLM variant — each server reserves its fraction at startup; pipeline models load into the remaining headroom. Reduce `--gpu-memory-utilization` if you see OOM during YOLO+SAM or Depth steps.
>
> **SceneTok (Step 14) requires ~24 GB VRAM minimum** (tested on RTX 4090 24 GB, L40S 45 GB, Quadro RTX 8000 48 GB). On a 16 GB card, omit `--scenetok` or offload it to a second GPU or host. A sidecar only separates the process boundary; if SceneTok runs on the **same GPU**, it still competes for the same VRAM. SceneTok training requires A100 (40/80 GB) or H100; the segmentation decoder fine-tuning has the same requirement.

---

## Key flags reference

| Flag | Default | What it controls |
|---|---|---|
| `--videos-dir` | `.data/videos` | Input video directory |
| `--output-dir` | `.data/local_runs` | Where results are written |
| `--fps` | `2.0` | Frame extraction rate |
| `--epochs` | `3` | SSL DINOv3 fine-tuning epochs per video |
| `--batch-size` | `4` | Fine-tuning batch size |
| `--top-k` | `5` | Neighbours shown in search tests |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` |
| `--no-qdrant` | off | Use in-memory cosine search instead of Qdrant |
| `--no-sfm` | off | Skip pycolmap SfM; use PCA point-cloud fallback |
| `--no-gsplat` | off | Skip 3D Gaussian Splatting |
| `--no-caption` | off | Skip Florence-2 captioning |
| `--no-asr` | off | Skip Whisper ASR |
| `--no-ocr` | off | Skip OCR text extraction |
| `--no-depth` | off | Skip depth estimation |
| `--no-yolo` | off | Skip YOLO11 detection |
| `--no-sam` | off | Skip SAM segmentation |
| `--no-rfdetr` | off | Skip RF-DETR tracking |
| `--no-distill` | off | Skip knowledge distillation; export teacher to ONNX |
| `--no-onnx` | off | Skip ONNX export |
| `--drone-audio` | on | Enable Step 32 — DroneAudioCNN training and ONNX export |
| `--no-drone-audio` | — | Skip Step 32 drone audio training |
| `--drone-audio-epochs` | `10` | Training epochs for DroneAudioCNN |
| `--rfdetr-model` | `base` | RF-DETR tier: `base` or `large` |
| `--gemma-api-url` | from `.data/.env` | Gemma sidecar endpoint |
| `--qwen-api-url` | from `.data/.env` | Qwen sidecar endpoint |
| `--reasoning-api-url` | from `.data/.env` | Reasoning sidecar endpoint |
| `--scenetok` | off | Enable SceneTok Step 14 — streaming scene encoder + segmentation decoder (~24 GB VRAM) |
| `--scenetok-api-url` | from `.data/.env` | SceneTok FastAPI sidecar endpoint; falls back to local torch if unset |
| `--scenetok-checkpoint` | `va-videodc_re10k` | Checkpoint variant: `va-videodc_re10k`, `va-videodc_dl3dv`, `va-wan_dl3dv` |

---

## Outputs

After a successful run, `.data/local_runs/<video_name>/` contains:

| Path | Contents |
|---|---|
| `frames/` | Extracted keyframes |
| `sparse_map.npz` | Pycolmap sparse point-cloud (or PCA fallback) |
| `captions.json` | Florence-2 caption per frame |
| `knowledge.json` | Gemma scene analysis |
| `physical_state_summary.json` | Clip-level physical state: pose confidence, near-field occupancy, tracked object velocities, free-space estimate |
| `field_state_summary.json` | Environmental field estimates: visibility hazard, RF interference intensity, thermal anomaly strength |
| `threat_primitives.json` | Structured evidence-gated threat primitives with type, score, uncertainty, persistence, and evidence sources |
| `local_threat_assessment.json` | Clip-level threat score, top threats, automation confidence, source-pair contradictions, and trust penalty |
| `policy_decision.json` | Recommended operator action (`continue` / `reduce_speed` / `reroute` / `abort` / `inspect_sensor`) with reason |
| `checkpoints/` | SSL fine-tuned DINOv3 checkpoint |
| `student_distilled.pth` | Knowledge-distilled student weights |
| `model.onnx` | ONNX-exported model ready for deployment |
| `summary_report.html` | Human-readable mission summary |
| `scenetok_tokens.npz` | Compressed SceneTok scene tokens — permutation-invariant latent representation of the full video |
| `scenetok_views/` | Novel view renders from the rectified flow decoder (one image per sampled viewpoint) |
| `scenetok_masks/` | Per-frame segmentation masks from the fine-tuned segmentation decoder *(experimental — requires a trained segmentation checkpoint)* |

After all videos complete, `.data/local_runs/` also contains run-level artifacts:

| Path | Contents |
|---|---|
| `final_stats.md` | Step timings, skipped stages, and top-level warnings |
| `global_threat_summary.json` | Cross-video threat aggregation: sector risk levels, persistent anomalies, route advisories |
| `threat_memory/` | Cross-mission persistent threat records keyed by sector |
| `threat_calibration.json` | Reliability diagram, threat score histogram, disagreement-rate trends, persistence threshold sweeps |
| `threat_eval_summary.json` | False positive / false negative analysis when ground-truth labels are present |
| `model_run_advisor.md` | Post-run warnings analysis and recommended model/env changes for this hardware |
| `model_run_advisor.json` | Machine-readable version of the model/run advisor |

---

## Threat and physical state pipeline

The pipeline now runs five threat-aware steps automatically — no additional flags are required.

### Per-video steps (always run)

**Step 17 — Physical state summary**
Aggregates depth, tracking, Kalman-filtered pose, and object detections into a compact
`physical_state_summary.json`. Key fields to read:

```bash
cat .data/local_runs/drone_mission/physical_state_summary.json | python3 -m json.tool | \
  grep -E "platform_pose_confidence|near_field_occupancy|free_space_estimate|confirmed_tracks"
```

| Field | Meaning |
|---|---|
| `platform_pose_confidence` | 0–1; 1 = tightly constrained SfM pose |
| `near_field_occupancy_density` | fraction of central image area occupied by tracked bboxes |
| `free_space_estimate` | conservative lower bound on clear space ahead |
| `confirmed_tracks` | number of tracks with at least two observations |

**Step 18 — Environmental field state**
Estimates coarse hazard fields from RGB captions, depth, and sensor sidecars.

```bash
cat .data/local_runs/drone_mission/field_state_summary.json | python3 -m json.tool
```

Each entry in `field_cells` has `field_type` (`visibility`, `rf_interference`, `thermal`),
`intensity_mean` (0–1), and `temporal_gradient` (positive = worsening over clip duration).

**Step 19 — Threat primitives**
Combines physical state, field state, and multi-modal evidence into structured primitives.

```bash
# List primitive types and scores
python3 -c "
import json, pathlib
p = json.loads(pathlib.Path('.data/local_runs/drone_mission/threat_primitives.json').read_text())
for prim in p.get('primitives', []):
    print(f\"{prim['type']:30s} score={prim['score']:.3f}  persist={prim.get('persistence_sec', 0):.1f}s\")
"
```

**Step 26 — Local threat inference**
Aggregates persisted primitives into a clip-level threat estimate.

```bash
python3 -c "
import json, pathlib
t = json.loads(pathlib.Path('.data/local_runs/drone_mission/local_threat_assessment.json').read_text())
print('threat score        :', t['local_threat_score'])
print('automation confidence:', t['automation_confidence'])
print('top threats:')
for th in t.get('top_threats', [])[:3]:
    print(' ', th['type'], th['score'])
"
```

**Step 27 — Action policy**
Maps the threat estimate and sensor-health context into a fixed operator action vocabulary.

```bash
python3 -c "
import json, pathlib
p = json.loads(pathlib.Path('.data/local_runs/drone_mission/policy_decision.json').read_text())
print('action:', p['recommended_action'])
print('reason:', p['policy_reason'])
"
```

The fixed action vocabulary is: `continue` | `reduce_speed` | `reroute` | `abort` | `inspect_sensor`.

### Run-level: global threat, memory, and calibration

After all videos in the batch finish, the runner aggregates across videos and writes run-level
artifacts. These always appear in the `--output-dir` root (default: `.data/local_runs/`).

**Global threat summary**

```bash
python3 -c "
import json, pathlib
g = json.loads(pathlib.Path('.data/local_runs/global_threat_summary.json').read_text())
print('global threat score:', g.get('global_threat_score', 0.0))
print('sector risk levels:')
for k, v in g.get('sector_risk_levels', {}).items():
    print(f'  {k}: {v}')
print('persistent anomalies:', len(g.get('persistent_anomalies', [])))
print('route advisories:', len(g.get('route_advisories', [])))
"
```

**Threat memory** — persists threat evidence across runs. Sector records accumulate under
`threat_memory/<sector_id>.json`. Each record has `threat_type`, `mean_score`, `occurrence_count`,
`last_seen`, and `persistence_trend`.

**Threat calibration** — read the reliability diagram and score histogram to diagnose whether
the threat scorer is well-calibrated for your mission domain:

```bash
python3 -c "
import json, pathlib
c = json.loads(pathlib.Path('.data/local_runs/threat_calibration.json').read_text())
print('records:', c['record_count'])
print('histogram bins:', c['threat_score_histogram'])
"
```

To enable false-positive / false-negative analysis, create `.data/local_runs/threat_labels.json`
with video-level ground-truth (`{"drone_mission": true, "safe_flight": false}`). The runner reads
it automatically and writes results to `threat_eval_summary.json`.

---

## Sensor mesh replay (Python API)

The `selfsuvis.pipeline.realtime` module lets you replay a completed local-run output directory
as ordered sensor-mesh event envelopes. This is how you simulate multi-node operation from
single-GPU run artifacts.

```python
from pathlib import Path
from selfsuvis.pipeline.realtime import replay_local_run, write_replay_jsonl
from selfsuvis.pipeline.realtime import RealtimeThreatAggregator

# Replay all video sub-directories in the output dir as timestamped events
output_dir = Path(".data/local_runs")
events = replay_local_run(
    output_dir,
    node_id_prefix="uav",                          # node IDs become uav_1, uav_2, …
    ingest_delay_sec_by_kind={"threat": 1.5},       # simulate 1.5 s latency for threat events
)

# Write the event stream to a JSONL file for inspection
write_replay_jsonl(events, output_dir / "replay.jsonl")
print(f"Replayed {len(events)} events")

# Feed events into the aggregator to get operator snapshots
agg = RealtimeThreatAggregator()
agg.consume_all(events)
snapshot = agg.snapshot()

print("sector threat rows:", len(snapshot["sector_rows"]))
print("degraded mode:", snapshot.get("degraded_mode", {}).get("active", False))
for row in snapshot["sector_rows"][:3]:
    print(f"  sector={row['sector_id']}  score={row['aggregated_score']:.3f}  freshness={row['freshness_weight']:.2f}")
```

Each event envelope contains `event_kind` (`sensor` | `threat` | `node_health`), `event_time`,
`ingest_time`, `sector_id`, `node_id`, `sensor_type`, and a `freshness_weight` in [0, 1] that
decays as the event ages. Events with `freshness_weight < 0.1` are treated as expired by the
aggregator.

To inspect the raw replay stream without Python:

```bash
# Write JSONL, then view
python3 -c "
from pathlib import Path
from selfsuvis.pipeline.realtime import replay_local_run, write_replay_jsonl
events = replay_local_run(Path('.data/local_runs'))
write_replay_jsonl(events, Path('.data/local_runs/replay.jsonl'))
print(len(events), 'events written')
"
head -5 .data/local_runs/replay.jsonl | python3 -m json.tool
```

---

## LangGraph orchestration (opt-in)

The pipeline ships a LangGraph-based orchestrator alongside the default monolithic runner.
Activate it with a single env var — no CLI change needed:

```bash
SELFSUVIS_USE_GRAPH=1 APP_ENV=dev .venv/bin/selfsuvis --mode local \
  --videos-dir .data/videos
```

Both paths produce byte-for-byte identical artifacts. The graph path additionally provides:

- **Parallel execution of steps 4–8** (Florence, ASR, OCR, depth, detection run concurrently; they serialise on GPU automatically via the existing VRAM guard)
- **Resumable checkpoints** — set `SELFSUVIS_CHECKPOINT_PATH` to persist node state to SQLite; set `SELFSUVIS_RESUME_THREAD_ID` to skip already-completed nodes after a crash
- **Optional LangSmith tracing** — set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` to see full execution graphs in the LangSmith UI
- **Agentic quality improvements** on the six LLM steps (steps 3, 10, 12, 13, 29, 30)

### Agentic improvements (LangGraph path only)

| Step | What changed |
|------|-------------|
| 03 Gemma analysis | Every `fact_verification` claim is scored against CLIP frame embeddings; claims below cosine similarity 0.25 are flagged `clip_verified=false` |
| 10 Gemma tracking | JSON parse failure now falls back to `["person","vehicle","sign"]` instead of silently tracking nothing |
| 12 Qwen captioning | Frames with `parse_error=True` get one retry with a simplified prompt; the prior-state chain skips confirmed-bad frames |
| 13 UniDriveVLA | Per-frame Jaccard MoE consensus score computed; frames below 0.5 flagged and logged as warnings |
| 29 Video synthesis | Draft → critique (evidence-grounded) → conditional regeneration on `MAJOR_CONTRADICTION` verdict |
| 30 Agentic flow audit | Reflection loop checks that all 32 step IDs are covered; appends a `## Reflection Gaps` section to `agentic_flow.md` when gaps are found |

### Checkpointing and resume

```bash
# Run with persistent checkpoints
SELFSUVIS_USE_GRAPH=1 SELFSUVIS_CHECKPOINT_PATH=.data/checkpoints.db \
  APP_ENV=dev .venv/bin/selfsuvis --mode local --videos-dir .data/videos
# Logs: "Starting graph pipeline for drone_mission (thread_id=drone_mission_1714123456)"

# Resume after failure — completed nodes are skipped
SELFSUVIS_USE_GRAPH=1 SELFSUVIS_CHECKPOINT_PATH=.data/checkpoints.db \
  SELFSUVIS_RESUME_THREAD_ID=drone_mission_1714123456 \
  APP_ENV=dev .venv/bin/selfsuvis --mode local --videos-dir .data/videos
```

Without `SELFSUVIS_CHECKPOINT_PATH` the graph uses an in-memory `MemorySaver` — state
survives exceptions within the same process but not across restarts.

### Graph env vars reference

| Variable | Default | Effect |
|----------|---------|--------|
| `SELFSUVIS_USE_GRAPH` | `` (off) | `1` to activate the LangGraph path |
| `SELFSUVIS_CHECKPOINT_PATH` | `` | SQLite file path for persistent checkpoints |
| `SELFSUVIS_RESUME_THREAD_ID` | `` | Thread ID to resume from; printed in run logs |
| `LANGCHAIN_TRACING_V2` | `` | `true` to emit traces to LangSmith |
| `LANGCHAIN_API_KEY` | `` | LangSmith API key |

---

## Runtime notes

- OCR is prescreened from Florence caption confidence in local full-analysis runs, so it may process only a subset of frames.
- Qwen detailed captioning uses bounded sampled-frame selection rather than captioning every extracted frame.
- Depth `auto` now prefers a faster local profile by default; use `DEPTH_AUTO_PROFILE=quality` if you want the heavier path.
- In the monolith path the agentic audit uses a simple first-pass prompt and only retries when the answer is empty or structurally incomplete. In the LangGraph path a reflection sub-loop also runs after a successful generation.
- SceneTok (Step 14) is off by default. Pass `--scenetok` to enable it; it is skipped silently if available VRAM is detected as below 20 GB unless the sidecar URL is set. The base checkpoint outputs novel-view RGB renders; the segmentation-decoder variant (`scenetok_masks/`) is experimental and requires a separately fine-tuned checkpoint where the rectified flow decoder head has been replaced with a mask prediction head.

---

## Next steps

- [Local Service Setup](quickstart-local.md) — API, worker, and UI on Docker
- [Configuration](../reference/configuration.md) — full env var reference and security settings
- [Data layout](../reference/data_layout.md) — where files are written, sensor sidecars, output artifacts
- [Troubleshooting](../operations/troubleshooting.md) — common errors and fixes
