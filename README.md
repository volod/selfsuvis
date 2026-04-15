# selfsuvis — Outdoor Autonomy Perception Stack

Spatial memory engine for outdoor robotics: ingest mission video from drones, rovers,
or vehicles → extract frames → estimate camera poses (pycolmap) → build dense 3D maps
(nerfstudio splatfacto) → embed frames (OpenCLIP + DINOv3) → caption with Florence-2 →
store in Qdrant + PostgreSQL → search by text or image query.

Self-improvement loop: each mission auto-tags uncertain/novel frames for annotation,
building training data for future self-supervised model fine-tuning and edge distillation.

## Quick start (Docker)

```bash
make up
```

Open the Streamlit UI at http://localhost:8501, upload a video or provide a URL,
and run text or image queries.

---

## Local full run (all 35 pipeline steps)

Follow these steps in order — each one is a prerequisite for the next.

### Step 1 — Environment variables (.env)

Copy `.env.sample` to `.env` and fill in your secrets **before running anything else**.
The setup script sources `.env` automatically.

```bash
cp .env.sample .env
```

Then edit `.env` and set at minimum:

```
HF_TOKEN=hf_xxxx          # HuggingFace token — required for gated models (Gemma, Phi variants)
```

Obtain a token at https://huggingface.co/settings/tokens (Read scope), then accept
the model licence on each model's HuggingFace page before running the setup.

Without `HF_TOKEN`: Gemma runs via Ollama (pulled automatically in Step 3 of setup)
and all non-gated models download without a token.

Optional overrides in `.env` (see `.env.sample` for full reference):

```
GEMMA_API_URL=http://localhost:11434/v1   # Ollama endpoint for Gemma generative steps
GEMMA_API_MODEL=gemma4:e4b               # model tag to use
```

### Step 2 — GPU model selection

The pipeline auto-selects models based on detected VRAM. Verify your GPU, then
decide which variant to run before starting setup.

```bash
nvidia-smi   # check VRAM
```

| VRAM | Analysis model (Steps 3, 22) | Reasoning model (Step 35) |
|---|---|---|
| 8 GB | `gemma4:e4b` | `qwen3:8b` |
| 16 GB | `gemma4:e4b` | `deepseek-r1:14b` |
| 24 GB | `gemma4:4b` | `deepseek-r1:14b` |
| 48 GB | `gemma4:12b` | `qwen3:30b` |
| 80 GB | `gemma4:26b` | `deepseek-r1:32b` |
| CPU 64 GB RAM | `gemma4:4b` | `deepseek-r1:14b` |

To override auto-detection (e.g. when the GPU driver is not reachable from the process):

```bash
export GPU_TOTAL_GB_HINT=16
export GPU_FREE_GB_HINT=12
```

You can also pin specific models via `.env`:

```
GEMMA_API_MODEL=gemma4:12b
```

### Step 3 — Run setup

One-shot bootstrap — creates the test data layout, downloads a test video, installs
the Python environment, downloads model weights, pulls Ollama models, and starts Docker:

```bash
bash scripts/setup_local_full.sh
```

Common flags:

```bash
bash scripts/setup_local_full.sh --no-docker        # skip Qdrant / PostgreSQL
bash scripts/setup_local_full.sh --no-ollama        # skip Ollama; use HF weights
bash scripts/setup_local_full.sh --sensor-data-only # regenerate sensor sidecars only
```

After setup completes, the script prints the exact run command for your configuration.

See [local_path.md](docs/local_path.md) for the short 35-step path and
[docs/learning_path/README.md](docs/learning_path/README.md) for the deep-dive study set,
sidecar naming, and per-sensor guidance.

### Step 4 — Run the pipeline

**Minimal run** (Steps 1–9, no sidecar servers needed):

```bash
.venv/bin/python main.py --mode local \
  --input data_test/videos/drone_mission.mp4 \
  --no-qdrant
```

**Full run — Ollama sidecars + all sensor steps** (sensor steps on by default):

```bash
python main.py --mode local \
  --input data_test/videos/drone_mission.mp4 \
  --qwen \
  --unidrive \
  --gemma-api-url    http://localhost:11434/v1 \
  --qwen-api-url     http://localhost:11434/v1 \
  --unidrive-model   owl10/UniDriveVLA_Nusc_Base_Stage3 \
  --rfdetr-model   base
```

This keeps Gemma and Qwen on Ollama, but runs UniDrive locally from cached Hugging Face
weights when available. `--unidrive-api-url http://localhost:11434/v1` is intentionally
not shown here because there is no published Ollama UniDriveVLA model; use local HF
weights or a vLLM sidecar instead.

**Full run — vLLM sidecars** (Qwen2.5-VL + UniDriveVLA, GPU-only):

```bash
# Terminal 1 — Qwen2.5-VL (Step 24)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8010 --max-model-len 8192

# Terminal 2 — UniDriveVLA (Step 25)
python -m vllm.entrypoints.openai.api_server \
  --model owl10/UniDriveVLA_Nusc_Base_Stage3 \
  --port 8030 --max-model-len 4096

# Terminal 3 — pipeline
.venv/bin/python main.py --mode local \
  --input data_test/videos/drone_mission.mp4 \
  --qwen \
  --unidrive \
  --gemma-api-url    http://localhost:11434/v1 \
  --qwen-api-url     http://localhost:8010/v1 \
  --unidrive-api-url http://localhost:8030/v1
```

> **vLLM note:** Florence-2 (`Florence2ForConditionalGeneration`) was removed in
> vLLM 0.11+. Load Florence-2 locally; if Ollama is also running, the pipeline
> sends `keep_alive=0` before loading Florence to free VRAM automatically.

### Sensor sidecar naming

Each sidecar file sits beside the video with the same basename.
`scripts/setup_local_full.sh` generates synthetic test sidecars automatically,
named after the test video it downloads.

```
data/videos/mission.mp4
data/videos/mission.iq              # Step  9 — RF/SDR IQ (float32 interleaved)
data/videos/mission.thermal.mp4     # Step 10 — FLIR LWIR radiometric video
data/videos/mission.multispectral/  # Step 11 — per-band GeoTIFF directory
data/videos/mission.events.raw      # Step 12 — Prophesee event stream
data/videos/mission.lidar.pcd       # Step 13 — LiDAR point cloud (PCD/MCAP)
data/videos/mission.radar.bin       # Step 14 — radar ADC IQ (TI DCA1000)
data/videos/mission.adsb.jsonl      # Step 15 — ADS-B aircraft log (dump1090)
data/videos/mission.gnssr.bin       # Step 15 — GNSS-R IQ capture
data/videos/mission.imu.jsonl       # Step 16 — IMU samples (200 Hz)
data/videos/mission.baro.jsonl      # Step 16 — barometer (5 Hz)
data/videos/mission.wind.jsonl      # Step 16 — anemometer (1 Hz)
data/videos/mission.env.jsonl       # Step 17 — atmospheric (temp/humidity/wind)
data/videos/mission.gas.jsonl       # Step 18 — gas/radiation (CO2, VOC, dose rate)
data/videos/mission.audio.wav       # Step 19 — acoustic (48 kHz WAV)
```

To regenerate sidecars for a different video:

```bash
cp /path/to/my_mission.mp4 data_test/videos/
bash scripts/setup_local_full.sh --sensor-data-only
```

---

## Output artifacts

For each video `<name>.mp4` the pipeline writes to `<output-dir>/<name>/`:

| File / Dir | Step | Contents |
|---|---|---|
| `gemma_analysis.md` | 3 | Gemma scene change detection, clustering, CLIP+DINOv3 comparison |
| `gemma_captions.md` | 3 | Per-frame natural-language descriptions (requires `--gemma-api-url`) |
| `scene_captions.md` | 4 | Florence-2 captions per keyframe |
| `asr_subtitles.md` | 5 | Whisper ASR segments + per-frame subtitle coverage |
| `multimodal_features.md` | 6–8 | OCR text, depth percentiles, detections, world model |
| `detailed_captions.md` | 24 | Qwen VLM structured per-frame analysis (requires `--qwen-api-url`) |
| `unidrive_analysis.md` | 25 | UniDriveVLA understanding / perception / planning / MoE |
| `multi_model_comparison.md` | 32 | Gemma vs Qwen vs UniDriveVLA expert-agreement summary |
| `finetune_stats.md` | 28 | SSL fine-tuning loss curve + config |
| `finetuned_search.md` | 31 | Queries re-run with fine-tuned model |
| `comparison.md` | 32 | Side-by-side model comparison + video-to-text description |
| `edge_models/` | 30 | ONNX model + frame gallery for edge deployment |
| `checkpoints/` | 28 | Fine-tuned `.pt` checkpoints |
| `3d_map/sparse_map.ply` | 27 | Sparse SfM or PCA point cloud |
| `3d_map/gaussian_splat.ply` | 27 | 3D Gaussian Splat — see [docs/gaussian_splat.md](docs/gaussian_splat.md) |
| `3d_map/semantic_environment_graph.json` | 27 | YOLO SSG scene graph |
| `gemma_tracking_results.json` | 22 | Gemma-directed tracking per frame |
| `gemma_tracking/frame_*_tracked.jpg` | 22 | Annotated frames with RF-DETR track boxes |
| `final_stats.md` | 35 | Per-video and aggregate statistics |

---

## Docs

| Document | Contents |
|---|---|
| [Local learning path](docs/local_path.md) | Short 35-step essentials plus a realistic day-by-day syllabus |
| [Learning path deep dives](docs/learning_path/README.md) | Detailed human-oriented explanations for every phase and step group |
| [Pipeline](docs/pipeline.md) | Agentic pipeline architecture and data flow |
| [Architecture](docs/architecture.md) | System components and service topology |
| [Configuration](docs/configuration.md) | All env vars with defaults and security notes |
| [3D Gaussian Splat map](docs/gaussian_splat.md) | Step 27 — gsplat modes, outputs, viewing |
| [Setup](docs/setup.md) | Manual setup without the bootstrap script |
| [API](docs/api.md) | HTTP API reference |
| [UI](docs/ui.md) | Streamlit UI usage |
| [Data layout](docs/data_layout.md) | Directory structure and file naming |
| [Performance](docs/performance.md) | Latency targets and tuning |
| [Troubleshooting](docs/troubleshooting.md) | Common errors and fixes |
| [Tests](docs/tests.md) | Unit and integration test guide |
| [Development](docs/development.md) | Contributing, code style, project conventions, workflow skills |
| [Architecture decisions](docs/adr/README.md) | ADR log |
| [Design doc](docs/design/outdoor-autonomy-perception-stack.md) | Original design document |
