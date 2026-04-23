# Quick Start — Local Development

Set up SelfSuvis for local development with hot-reload on the API or when working on the pipeline code directly.

---

## Prerequisites

**Required:**
- Git
- Docker Engine >= 24 with Compose v2
- Python 3.10
- ffmpeg, libgl1 (`sudo ./scripts/install_system_deps.sh --with-python`)

**GPU (optional but recommended):**
- NVIDIA Container Toolkit — run `sudo ./scripts/install_nvidia_docker.sh` if it is not installed

---

## 1. Install system dependencies and create the venv

```bash
sudo ./scripts/install_system_deps.sh --with-python
make venv
```

---

## 2. Configure environment

Generate `.env` at the project root. The generator auto-detects your GPU and RAM, picks appropriate models, and prints exactly which sidecar commands to run.

**Quick (non-interactive) — use detected hardware defaults:**

```bash
make env
```

**Interactive — choose sidecar backend, profile, models:**

```bash
make env-interactive
```

The interactive flow:
1. Shows detected hardware (GPU VRAM, RAM)
2. Asks for primary sidecar backend — `ollama` (default, easiest) or `vllm` (higher throughput, more setup)
3. Asks for profile — `minimal` / `balanced` (default) / `full`
4. Asks for environment name and output path

Both commands write to `repo_root/.env`. The config loader reads it after the packaged `env/dev.env` defaults, so `DATABASE_URL`, `QDRANT_HOST=localhost`, `DEVICE`, and sidecar URLs are all set for you.

After generation, set the remaining values:

```bash
$EDITOR .env
```

```
API_KEY=<choose-a-secret>   # leave empty for unauthenticated dev use
HF_TOKEN=hf_xxx             # optional — gated HuggingFace models only
```

`ALLOWED_INDEX_PATHS` is pre-set to `./data/videos` by the generator. Drop mission videos there and the path-based indexing API will accept them. Change it if your videos live elsewhere.

---

## 3. Start sidecars

The generator prints the exact commands at the end of its output. Refer to those, or use the patterns below based on what you chose.

**Ollama** (default — models pulled automatically):

```bash
ollama serve                         # keep running in a terminal
ollama pull <GEMMA_API_MODEL>        # value from .env, e.g. gemma4:e4b
ollama pull <REASONING_MODEL>        # value from .env, e.g. deepseek-r1:14b
```

**vLLM** (if chosen for Qwen or Gemma — each in its own terminal):

```bash
# Qwen visual model (port 8010)
python -m vllm.entrypoints.openai.api_server \
  --model <QWEN_MODEL> --port 8010 --max-model-len 8192

# Gemma (port 8000) — only if GEMMA_API_BACKEND=vllm
python -m vllm.entrypoints.openai.api_server \
  --model <GEMMA_API_MODEL> --port 8000 --max-model-len 8192
```

Replace `<GEMMA_API_MODEL>` / `<QWEN_MODEL>` / `<REASONING_MODEL>` with the values written to `.env`.

---

## 4. Start backing services only

```bash
docker compose -f docker/docker-compose.yml up -d postgres qdrant
```

---

## 5. Run the database migration (first time only)

```bash
APP_ENV=dev .venv/bin/python -m selfsuvis.scripts.migrate_postgres
```

---

## 6. Start each service in a separate terminal

```bash
# Terminal 1 — API (hot-reload enabled)
APP_ENV=dev .venv/bin/uvicorn selfsuvis.app.main:app \
  --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Worker
APP_ENV=dev .venv/bin/python -m selfsuvis.worker

# Terminal 3 — UI
APP_ENV=dev .venv/bin/python -m selfsuvis.ui \
  --server.address 0.0.0.0 --server.port 8501
```

---

## Default service URLs

| Service | URL |
|---|---|
| UI | http://localhost:8501 |
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |
| Nginx static server | http://localhost:8080 |

---

## Running the API pipeline

Once services are up, the indexing pipeline runs automatically in the background worker. Trigger jobs via the UI at `http://localhost:8501` or via API:

```bash
curl -X POST http://localhost:8000/index/video \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/mission.mp4"
```

See the [Production Quick Start](quickstart-production.md) for full pipeline and querying details.

---

## Learning path pipeline (`selfsuvis --mode local`)

The local mode runs the full 35-step learning pipeline directly on your machine — no API server or Docker stack needed. It processes videos, runs every perception and captioning step, fine-tunes a DINOv3 model on the mission frames, and exports it to ONNX.

### One-shot bootstrap (recommended for first-timers)

`setup_local_full.sh` does everything in one command — venv, model weights, Ollama, test video download, sensor sidecars, Docker services, DB migration — and prints the exact run command at the end:

```bash
bash scripts/setup_local_full.sh
```

Common variants:

```bash
# CPU-only machine — skip Docker and Ollama:
bash scripts/setup_local_full.sh --no-docker --no-ollama

# Already have models; only want fresh sensor sample data:
bash scripts/setup_local_full.sh --sensor-data-only

# With HuggingFace token for gated Gemma weights:
HF_TOKEN=hf_xxxx bash scripts/setup_local_full.sh
```

If the one-shot script works for you, skip to [Step 6 — Run the pipeline](#step-6--run-the-pipeline) below. The manual steps that follow explain each phase individually.

---

The manual setup sequence is: **install venv -> generate env -> download models -> start sidecars -> get test data -> run pipeline**.

---

### Step 1 — Install the venv

If you have not done this yet:

```bash
sudo ./scripts/install_system_deps.sh --with-python
make venv
```

---

### Step 2 — Generate the local env

```bash
make env               # non-interactive: auto-detects GPU and RAM
# or
make env-interactive   # prompts for sidecar backend, profile, models
```

Then open `.env` and set `API_KEY` (leave blank for unauthenticated local use). `ALLOWED_INDEX_PATHS` is pre-filled to `./data/videos`.

---

### Step 3 — Download model weights

Run this once before the first pipeline run. All weights are cached locally; subsequent runs skip already-cached models.

**Core models only** (OpenCLIP + DINOv2/v3 — always required):

```bash
APP_ENV=dev .venv/bin/python -m selfsuvis.scripts.prepare_models --clip --dino
```

**Balanced set** (adds Florence-2 captioning, YOLO11, SAM):

```bash
APP_ENV=dev .venv/bin/python -m selfsuvis.scripts.prepare_models \
  --clip --dino --florence --yolo --sam
```

**Full set** (everything including ASR, OCR, depth, detection, world model):

```bash
APP_ENV=dev .venv/bin/python -m selfsuvis.scripts.prepare_models --all
```

**Gemma open-weight** (optional — loads locally instead of via Ollama sidecar; requires `HF_TOKEN` in `.env` and license accepted at `huggingface.co/google/gemma-3-4b-it`):

```bash
APP_ENV=dev .venv/bin/python -m selfsuvis.scripts.prepare_models --gemma
# or a smaller variant:
APP_ENV=dev .venv/bin/python -m selfsuvis.scripts.prepare_models \
  --gemma --gemma-model google/gemma-3-1b-it
```

**Check what is already cached** (no downloads):

```bash
APP_ENV=dev .venv/bin/python -m selfsuvis.scripts.prepare_models --verify
APP_ENV=dev .venv/bin/python -m selfsuvis.scripts.prepare_models --verify --all
```

**Per-step model reference:**

| Flag | Pipeline step | Default model |
|---|---|---|
| `--clip` | Step 2 — embedding | `ViT-B-16 / openai` |
| `--dino` | Step 2 — embedding | `dinov2_vitb14`, `dinov3_vitb14` |
| `--florence` | Step 4 — captioning | `microsoft/Florence-2-large` |
| `--whisper` | Step 5 — ASR | `openai/whisper-large-v3-turbo` |
| `--ocr` | Step 6 — OCR | auto-selected by VRAM |
| `--depth` | Step 7 — depth estimation | auto-selected by VRAM |
| `--detection` | Step 8 — object detection | auto-selected by VRAM |
| `--yolo` | Step 21 — YOLO11 detection | `yolo11l` (~48 MB) |
| `--sam` | Step 21 — SAM segmentation | `facebook/sam3` (falls back to `sam2-hiera-large`) |
| `--world-model` | Step 23 — video embeddings | auto-selected by VRAM |
| `--gemma` | Step 3 — scene analysis | `google/gemma-3-4b-it` |
| `--unidrive` | Step 25 — UniDriveVLA | `owl10/UniDriveVLA_Nusc_Base_Stage3` |

---

### Step 4 — Start sidecars

The Ollama sidecars serve Gemma (scene analysis) and the reasoning model. They must be running before `selfsuvis --mode local` starts.

```bash
ollama serve                              # keep running in a terminal
ollama pull <GEMMA_API_MODEL>             # value from .env, e.g. gemma4:e4b
ollama pull <REASONING_MODEL>             # value from .env, e.g. deepseek-r1:14b
```

If using vLLM instead (set during `make env-interactive`):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <GEMMA_API_MODEL> --port 8000 --max-model-len 8192
```

Skip this step entirely if you downloaded Gemma open-weight locally in Step 3 — the pipeline will load it directly.

---

### Step 5 — Test video and sensor data

The pipeline needs at least one video in `data/videos/` and, for sensor fusion steps 9–19, matching sidecar files.

#### Option A — Use your own footage

```bash
cp /path/to/mission.mp4 data/videos/
```

Then generate sensor sidecars keyed to that video's basename (the script auto-detects the file):

```bash
bash scripts/prepare_sensor_data.sh data/sensors
```

#### Option B — Download a public-domain test video

`setup_local_full.sh` handles download, trim, and sensor sidecar generation automatically:

```bash
bash scripts/setup_local_full.sh --sensor-data-only
```

This downloads the US Highway 60 drone flyover (~27 MB, no login), trims it to 10 s, and generates all sensor sidecars in one step. If the CDN is unreachable it falls back to a second archive.org clip, then generates a synthetic video with ffmpeg.

#### What `prepare_sensor_data.sh` does

The script creates one directory per sensor step under `data/sensors/` and generates synthetic sidecar files named after the video currently in `data/videos/` — so they are immediately usable by the pipeline without any renaming.

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

cp data/sensors/step16_imu/${VIDEO_BASE}.imu.jsonl    data/videos/
cp data/sensors/step16_imu/${VIDEO_BASE}.baro.jsonl   data/videos/
cp data/sensors/step16_imu/${VIDEO_BASE}.wind.jsonl   data/videos/
cp data/sensors/step17_atmospheric/${VIDEO_BASE}.env.jsonl  data/videos/
cp data/sensors/step18_gas_radiation/${VIDEO_BASE}.gas.jsonl data/videos/
cp data/sensors/step19_acoustic/${VIDEO_BASE}.audio.wav      data/videos/
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

### Step 6 — Run the pipeline

**Minimal run** — fewest dependencies, in-memory store, no 3D reconstruction:

```bash
APP_ENV=dev .venv/bin/selfsuvis --mode local \
  --videos-dir data/videos \
  --no-qdrant \
  --no-sfm \
  --no-gsplat
```

**Standard run** — with Qdrant for retrieval tests at steps 26 and 31:

```bash
# Start Qdrant if not already running:
docker compose -f docker/docker-compose.yml up -d qdrant

APP_ENV=dev .venv/bin/selfsuvis --mode local \
  --videos-dir data/videos
```

**Fast iteration** — skip slow optional steps (ASR, OCR, depth, captioning):

```bash
APP_ENV=dev .venv/bin/selfsuvis --mode local \
  --videos-dir data/videos \
  --no-qdrant \
  --no-sfm \
  --no-gsplat \
  --no-caption \
  --no-asr \
  --no-ocr \
  --no-depth
```

**Full run — Ollama only** (all LLM/VLM steps via Ollama, single GPU):

Ollama serves one model at a time. `OLLAMA_MAX_LOADED_MODELS=1` ensures each model is
evicted before the next loads; the pipeline also sends `keep_alive=0` after each step
so local torch models (CLIP, DINO, Florence, YOLO, SAM, Depth) reclaim VRAM cleanly.

```bash
# —— Terminal 1: Ollama daemon —————————————————————————————————————————————————
# OLLAMA_MAX_LOADED_MODELS=1  — only one model resident in VRAM at a time
# OLLAMA_NUM_PARALLEL=1       — single concurrent request (no extra KV caches)
# OLLAMA_KEEP_ALIVE=0         — evict immediately after each request
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=0 \
  ollama serve

# Pull models once (in a separate terminal or before starting the daemon):
ollama pull gemma4:e4b        # Steps 3, 22, 35 — scene analysis + audit (~5 GB)
ollama pull qwen2.5vl:7b      # Step 24        — detailed captioning    (~5 GB)

# —— Terminal 2: pipeline ————————————————————————————————————————————————————————
# PYTORCH_CUDA_ALLOC_CONF     — expandable segments avoid fragmentation OOM
# Steps enabled: embed(1-2) caption(4) asr(5) ocr(6) depth(7) detection(8)
#                yolo+sam(21) rfdetr(22) gemma-tracking(22) qwen(24)
#                world-model(23) unidrive skipped (vLLM only) sfm(P1) gsplat(P2)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
APP_ENV=dev .venv/bin/selfsuvis --mode local \
  --videos-dir        data/videos \
  --asr               \
  --ocr               \
  --depth             \
  --detection         \
  --qwen              \
  --world-model       \
  --no-unidrive       \
  --rfdetr-model      base \
  --gemma-api-url     http://localhost:11434/v1 \
  --gemma-api-backend ollama \
  --qwen-api-url      http://localhost:11434/v1 \
  --qwen-backend      ollama \
  --reasoning-api-url http://localhost:11434/v1 \
  --reasoning-backend ollama
```

**Full run — vLLM only** (all LLM/VLM steps via vLLM, including UniDriveVLA):

Each vLLM server reserves a fixed GPU memory fraction (`--gpu-memory-utilization`).
Run servers on separate GPUs when possible; on a single GPU start them one at a time
and let the pipeline's inter-step eviction (`keep_alive=0` equivalent) clear KV caches.

```bash
# —— Terminal 1: Gemma 4 — Steps 3, 22, 35 (scene analysis, tracking, audit) —
# ~5 GB VRAM at 0.45 utilisation on a 16 GB card
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m vllm.entrypoints.openai.api_server \
  --model              google/gemma-3-4b-it \
  --port               11434 \
  --gpu-memory-utilization 0.45 \
  --max-model-len      8192 \
  --enforce-eager \
  --swap-space         4

# —— Terminal 2: Qwen2.5-VL — Step 24 (detailed captioning) ———————————————————
# ~5 GB VRAM at 0.45 utilisation; start AFTER Gemma is ready if sharing one GPU
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m vllm.entrypoints.openai.api_server \
  --model              Qwen/Qwen2.5-VL-7B-Instruct \
  --port               8010 \
  --gpu-memory-utilization 0.45 \
  --max-model-len      8192 \
  --enforce-eager \
  --swap-space         4

# —— Terminal 3: UniDriveVLA — Step 25 (expert driving analysis) ————————————————
# ~4 GB VRAM; requires HF weights downloaded via prepare_models --unidrive
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m vllm.entrypoints.openai.api_server \
  --model              owl10/UniDriveVLA_Nusc_Base_Stage3 \
  --port               8030 \
  --gpu-memory-utilization 0.40 \
  --max-model-len      4096 \
  --enforce-eager \
  --swap-space         4

# —— Terminal 4: pipeline ——————————————————————————————————————————————————————
# Steps enabled: all including unidrive (Step 25)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
APP_ENV=dev .venv/bin/selfsuvis --mode local \
  --videos-dir        data/videos \
  --asr               \
  --ocr               \
  --depth             \
  --detection         \
  --qwen              \
  --world-model       \
  --unidrive          \
  --rfdetr-model      base \
  --gemma-api-url     http://localhost:11434/v1 \
  --gemma-api-backend vllm \
  --qwen-api-url      http://localhost:8010/v1 \
  --qwen-backend      vllm \
  --unidrive-api-url  http://localhost:8030/v1 \
  --unidrive-backend  vllm \
  --reasoning-api-url http://localhost:11434/v1 \
  --reasoning-backend vllm
```

> **Memory budget (single 16 GB GPU):**
> Ollama variant — peak is Florence-2 (~4 GB) + CLIP/DINO (~2 GB) = ~6 GB; sidecars evicted before each local step.
> vLLM variant — each server reserves its fraction at startup; pipeline models load into the remaining headroom. Reduce `--gpu-memory-utilization` if you see OOM during YOLO+SAM or Depth steps.

---

### Key flags reference

| Flag | Default | What it controls |
|---|---|---|
| `--videos-dir` | `data/videos` | Input video directory |
| `--output-dir` | `data/local_runs` | Where results are written |
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
| `--rfdetr-model` | `base` | RF-DETR tier: `base` or `large` |
| `--gemma-api-url` | from `.env` | Gemma sidecar endpoint |
| `--qwen-api-url` | from `.env` | Qwen sidecar endpoint |
| `--reasoning-api-url` | from `.env` | Reasoning sidecar endpoint |

---

### Outputs

After a successful run, `data/local_runs/<video_name>/` contains:

| Path | Contents |
|---|---|
| `frames/` | Extracted keyframes |
| `sparse_map.npz` | Pycolmap sparse point-cloud (or PCA fallback) |
| `captions.json` | Florence-2 caption per frame |
| `knowledge.json` | Gemma scene analysis |
| `checkpoints/` | SSL fine-tuned DINOv3 checkpoint |
| `student_distilled.pth` | Knowledge-distilled student weights |
| `model.onnx` | ONNX-exported model ready for deployment |
| `summary_report.html` | Human-readable mission summary |

---

## Local runtime notes

- OCR is prescreened from Florence caption confidence in local full-analysis runs, so it may process only a subset of frames.
- Qwen detailed captioning uses bounded sampled-frame selection rather than captioning every extracted frame.
- Depth `auto` now prefers a faster local profile by default; use `DEPTH_AUTO_PROFILE=quality` if you want the heavier path.
- The final agentic audit uses a simple first-pass prompt and only retries when that answer is empty or structurally incomplete.

---

## Next steps

- [Configuration](configuration.md) — full env var reference and security settings
- [Data layout](data_layout.md) — where files are written, sensor sidecars, output artifacts
- [API reference](api.md) — HTTP endpoints including the robot pose API
- [Troubleshooting](troubleshooting.md) — common errors and fixes
