# selfsuvis — Outdoor Autonomy Perception Stack

Spatial memory engine for outdoor robotics: ingest mission video from drones, rovers,
or vehicles → extract frames → estimate camera poses (pycolmap) → build dense 3D maps
(nerfstudio splatfacto) → embed frames (OpenCLIP + DINOv3) → caption with Florence-2 →
store in Qdrant + PostgreSQL → search by text or image query.

Self-improvement loop: each mission auto-tags uncertain/novel frames for annotation,
building training data for future self-supervised model fine-tuning and edge distillation.

## Quick start

```bash
make up
```

Then open the Streamlit UI (default: http://localhost:8501), 
upload a video or provide a URL (file or stream), start understanding, 
and run text or image queries.

## Demo

Run the end-to-end demonstration pipeline — no Docker, no GPU required (CPU fallback available):

### Prerequisites

- Python 3.10+ with virtualenv set up (`make venv`)
- `ffmpeg` on PATH (`brew install ffmpeg` / `sudo apt install ffmpeg`)
- At least one `.mp4` or `.mov` video file in `data_test/videos/` (two test clips are already committed there)
- *(Optional)* Qdrant running locally on `localhost:6333` for vector search; falls back to in-memory search automatically

### Start Qdrant locally (optional)

Use the project's existing Compose configuration — data is persisted in `data/qdrant/`:

```bash
# Start only the Qdrant service (no GPU, no API, no worker needed for the demo)
env UID=$(id -u) GID=$(id -g) docker compose -f docker/docker-compose.yml up -d qdrant

# Verify it is up:
curl -s http://localhost:6333/healthz   # → {"title":"qdrant","version":"..."}

# Stop:
env UID=$(id -u) GID=$(id -g) docker compose -f docker/docker-compose.yml stop qdrant
```

If Qdrant is not running the demo falls back to an in-memory cosine-similarity store automatically — no action needed.

### Sample videos

Any outdoor footage works. Two test clips are already in `data_test/videos/`. Additional free 4K samples:

- Mixkit: https://mixkit.co/free-stock-video/nature/ → download as `.mp4`, place in `data_test/videos/`
- Pexels: https://www.pexels.com/search/videos/outdoor/ → download, place in `data_test/videos/`

### Download required models (optional — cached on first run)

```bash
# Default: OpenCLIP + DINOv2/v3
python scripts/prepare_models.py

# DINO weights only (auto → hub → HF fallback)
python scripts/prepare_models.py --dino

# Force HF only — useful when GitHub is blocked
python scripts/prepare_models.py --dino --source hf

# Force torch.hub only — no HF fallback
python scripts/prepare_models.py --dino --source hub

# Gemma open-weight (step J local embedder — gated, requires HF_TOKEN)
python scripts/prepare_models.py --gemma
python scripts/prepare_models.py --gemma --gemma-model google/gemma-4-31b-it   # 31B

# Pre-cache Whisper ASR + Florence-2 captioning models
python scripts/prepare_models.py --whisper --florence

# Everything at once
python scripts/prepare_models.py --all
```

### Florence-2 scene captioning (step L)

Step L captions every keyframe using Florence-2-large (`<MORE_DETAILED_CAPTION>`).
By default the model is loaded locally into the same GPU process.
If another process (e.g. Ollama with a 7B VLM) already occupies most VRAM, the
pipeline automatically tries two strategies before falling back to Qwen API:

1. **Ollama VRAM eviction (automatic)** — when `--qwen-api-url` points to Ollama
   (port 11434), the pipeline sends `keep_alive=0` to unload the running VLM before
   loading Florence-2 (~11–12 GiB freed). Ollama reloads its model automatically
   when step R (Qwen) sends the first request. No extra flags needed.

   > **Note on vLLM:** Florence-2 (`Florence2ForConditionalGeneration`) was
   > supported in vLLM up to v0.10.2 and was **removed in v0.11+**. The current
   > `vllm/vllm-openai:latest` image does not support Florence-2. If you have a
   > custom OpenAI-compatible server that does serve Florence-2, pass its URL via
   > `--florence-api-url`. For all standard setups, use the Ollama auto-eviction
   > approach above.

### Qwen VLM sidecar (optional — for detailed scene captioning, step R)

Step R (`--qwen`) calls an OpenAI-compatible vision endpoint for structured per-frame analysis.
It uses ASR subtitles (from step M) and OCR text (from step N) as context in the prompt.

> **Important:** Do **not** `pip install vllm` into the project virtualenv —
> vLLM replaces pydantic, protobuf, fastapi, and transformers with incompatible
> versions and will break the project. Run vLLM in Docker instead.
>
> **Florence-2 is not supported in vLLM 0.11+.** Only Qwen can be served via
> vLLM Docker. Florence-2 is loaded locally with automatic Ollama eviction (see
> Option 2) or skipped with `--no-caption`.

**Option 1 — vLLM Docker: Qwen only**

```bash
# Qwen2.5-VL-7B for step R (detailed captioning), port 8010:
docker run --gpus all --rm -p 8010:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --max-model-len 8192 --limit-mm-per-prompt image=1

# Run — Qwen via vLLM Docker, Florence loaded locally:
python main.py --mode demo --asr --ocr --qwen \
  --qwen-api-url http://localhost:8010/v1
```

**Option 2 — Ollama: Qwen only (Florence loaded locally with auto-eviction)**

Ollama runs Qwen; Florence-2 is loaded locally. The pipeline automatically sends
`keep_alive=0` to Ollama before loading Florence (~11–12 GiB freed), then Ollama
reloads when step R runs. No extra flags needed.

```bash
# Install ollama (https://ollama.com), then pull the model:
ollama pull qwen2.5vl:7b

# Run — Florence local (Ollama auto-evicted before step L), Qwen via Ollama:
python main.py --mode demo --asr --ocr --qwen \
  --qwen-api-url http://localhost:11434/v1 --qwen-model qwen2.5vl:7b

# Or set env vars permanently:
export QWEN_API_URL=http://localhost:11434/v1
export QWEN_BACKEND=ollama
python main.py --mode demo --asr --ocr --qwen
```

**Option 3 — Remote / Docker Compose:**

```bash
# In docker-compose.yml, add a qwen service, then:
QWEN_API_URL=http://qwen:8010/v1 \
  python main.py --mode demo --asr --ocr --qwen
```

> **Note:** If `QWEN_API_URL` is empty (the default), step R is skipped automatically.
> The demo still runs to completion without it.

### Gemma 4 sidecar (optional — for open-weight video analysis, step J)

Step J uses Gemma 4 to analyse sampled video frames.
It runs in two modes that complement each other:

| Mode | What it does | When active |
|---|---|---|
| **Embedding mode** (local) | Loads `GemmaEmbedder` for scene change detection, clustering, zero-shot classification, cross-modal retrieval, and comparison with CLIP and DINOv3 | `MODEL_NAME=gemma` (default in demo) |
| **Generative mode** (sidecar) | Calls Ollama/vLLM to produce a natural-language description for each sampled frame | `GEMMA_API_URL` is set |

The default sidecar model is **`gemma4:31b`** — Gemma 4's largest open-weight variant.
Ollama automatically offloads transformer layers to RAM when VRAM is insufficient,
so it runs on mixed GPU+CPU hardware (e.g. 12 GB VRAM + 32 GB RAM handles 31B INT4).

**Option 1 — Ollama (recommended, mixed GPU+CPU):**

```bash
# Install Ollama: https://ollama.com
ollama pull gemma4:31b

# Run with Gemma sidecar (step J generative descriptions):
python main.py --mode demo --gemma-api-url http://localhost:11434/v1

# Or export once:
export GEMMA_API_URL=http://localhost:11434/v1
python main.py --mode demo
```

**Option 2 — vLLM Docker (GPU-only, higher throughput):**

```bash
docker run --gpus all --rm -p 8020:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model google/gemma-4-31b-it --trust-remote-code \
  --max-model-len 4096

python main.py --mode demo \
  --gemma-api-url http://localhost:8020/v1 \
  --gemma-api-model google/gemma-4-31b-it
```

**Combined — Gemma + Qwen + full multimodal:**

```bash
# Ollama serves both Gemma 4 31B and Qwen2.5-VL 7B:
ollama pull gemma4:31b
ollama pull qwen2.5vl:7b

python main.py --mode demo --asr --ocr --depth --detection \
  --gemma-api-url http://localhost:11434/v1 \
  --qwen --qwen-api-url http://localhost:11434/v1 --qwen-model qwen2.5vl:7b
```

> **Note:** If `GEMMA_API_URL` is empty, the generative descriptions sub-step is
> skipped. Embedding-based analysis (scene change, clustering, CLIP/DINOv3 comparison)
> still runs whenever `MODEL_NAME=gemma` (the demo default).

### Split analysis and reasoning models

The demo now uses two separate LLM roles:

- **Video analysis model**: used during step J for repeated sampled-frame analysis. Keep this relatively light.
- **Reasoning model**: used only in the final `agentic_flow.md` audit step. This can be larger and slower.
- The local `GemmaEmbedder` is reused across the run, and repeated image embeddings are cached in-process to avoid unnecessary recomputation.

New demo flags:

```bash
--gemma-api-url http://localhost:11434/v1
--gemma-api-model gemma4:4b
--reasoning-api-url http://localhost:11434/v1
--reasoning-model deepseek-r1:14b
```

If you do **not** set `--gemma-api-model` or `--reasoning-model`, the demo auto-selects them from detected hardware:

| Hardware profile | Step J analysis default | Final reasoning default |
|---|---|---|
| 8 GB VRAM + 64 GB RAM | `gemma4:e4b` | `qwen3:8b` |
| 16 GB VRAM + 64 GB RAM | `gemma4:e4b` | `deepseek-r1:14b` |
| 24 GB VRAM + 64 GB RAM | `gemma4:4b` | `deepseek-r1:14b` |
| 48 GB VRAM + 64 GB RAM | `gemma4:12b` | `qwen3:30b` |
| 80 GB VRAM | `gemma4:26b` | `deepseek-r1:32b` |
| CPU / RAM-only, 64+ GB RAM | `gemma4:4b` | `deepseek-r1:14b` |
| CPU / RAM-only, 96+ GB RAM | `gemma4:12b` | `deepseek-r1:32b` |

For a machine with **16 GB GPU VRAM** and **64+ GB RAM**, the practical default is:

- **Step J analysis**: `gemma4:e4b`
- **Final reasoning audit**: `deepseek-r1:14b`

If Ollama or another process keeps the GPU occupied from a previous run, the demo now unloads known sidecar models before sizing the run. If VRAM detection still fails because the NVIDIA driver is not reachable from the current process, set:

```bash
export GPU_TOTAL_GB_HINT=16
export GPU_FREE_GB_HINT=12
```

These hints affect only model planning.

Recommended use:

- Use the smaller analysis model when the step runs many times over video samples.
- Use a reasoning-first model for the final audit/report step where long-form synthesis matters more than latency.
- If both models are served by Ollama, pull both in advance:

```bash
ollama pull gemma4:e4b
ollama pull deepseek-r1:14b
```

Example:

```bash
python main.py --mode demo --asr --ocr --depth --detection --qwen \
  --gemma-api-url http://localhost:11434/v1 \
  --reasoning-api-url http://localhost:11434/v1
```

The demo also logs VRAM snapshots before and after local model loads, model offload/restore events, and sidecar-backed Gemma/Qwen/reasoning calls so you can see whether memory was actually freed between steps. The final reasoning timeout is longer than the default API timeout; override with `REASONING_TIMEOUT_SEC` if needed.

For the local Gemma processor path, the repo now sets `use_fast=False` explicitly. That avoids a future `transformers` default flip changing behavior silently during demo runs.

### 3D Gaussian Splat map (step I)

Step I builds a 3D Gaussian Splat of each video scene using
[gsplat](https://github.com/nerfstudio-project/gsplat) — the reference
implementation from the nerfstudio project.

**Two initialization modes (auto-selected):**

| Mode | When | Quality |
|---|---|---|
| `gsplat_sfm` | pycolmap installed + ≥3 poses recovered | Best — real camera poses + 3D scene points |
| `gsplat_free` | pycolmap unavailable or SfM failed | Good — forward-facing pose estimate from frame timestamps |

**Prerequisites:**

```bash
# gsplat is in requirements_prod.txt — install with the rest of deps:
make venv

# Verify CUDA kernels compile (first call JIT-compiles, ~60s):
python -c "from gsplat.rendering import rasterization; print('gsplat OK')"
```

**Output** per video in `3d_map/`:
- `gaussian_splat.ply` — standard 3DGS PLY (viewable in SuperSplat, Luma AI, etc.)
- `view_splat.html` — standalone browser viewer (uses GaussianSplats3D CDN)
- `sparse_map.ply` — classic sparse point cloud (SfM or PCA)

**Viewing the generated 3D Gaussian Splat:**

**Option A — Drag-and-drop (easiest, no local server):**

1. Open https://playcanvas.com/supersplat/editor in your browser
2. Drag `3d_map/gaussian_splat.ply` onto the page
3. Use mouse/trackpad to orbit, pan, and zoom

**Option B — Built-in HTML viewer (local server required for CORS):**

```bash
# Serve the output directory over HTTP:
cd data_test/videos_test/<video-name>/3d_map/
python -m http.server 8765

# Open in browser:
# http://localhost:8765/view_splat.html
```

Controls: left-drag to orbit · right-drag to pan · scroll to zoom

**Option C — View NPZ point cloud** (no gsplat needed, matplotlib):

```bash
python main.py --mode demo --view-npz data_test/videos_test/<name>/3d_map/sparse_map.npz
```

**Skip gsplat** (faster runs, point-cloud only):

```bash
python main.py --mode demo --no-gsplat
```

### Run

```bash
# Basic — uses data_test/videos/, writes to data_test/videos_test/
python main.py --mode demo

# Custom directories
python main.py --mode demo --videos-dir /path/to/videos --output-dir /path/to/output

# CPU only (no CUDA required)
python main.py --mode demo --device cpu

# Skip optional steps
python main.py --mode demo --no-qdrant --no-sfm --no-onnx

# Enable multimodal steps (each loads its model lazily on first frame):
python main.py --mode demo --asr                   # Whisper speech-to-text
python main.py --mode demo --ocr                   # OCR text extraction per frame
python main.py --mode demo --depth                 # Depth estimation per frame
python main.py --mode demo --detection             # Object detection per frame
python main.py --mode demo --world-model           # World model video embeddings
python main.py --mode demo --qwen --qwen-api-url http://localhost:8010/v1  # Qwen VLM (step R)
python main.py --mode demo --gemma-api-url http://localhost:11434/v1      # Gemma 4 31B (step J)

# Full multimodal — Florence local (Ollama auto-evicted before step L), Qwen via vLLM Docker:
python main.py --mode demo --asr --depth --detection --world-model --ocr --qwen --qwen-api-url http://localhost:8010/v1

# Full multimodal — Gemma + Qwen + DeepSeek-R1 reasoning via Ollama:
export WORLD_MODEL=nvidia/Cosmos-1.0-Autoregressive-4B  # do not force gpu <16GB
python main.py --mode demo --asr --depth --detection --world-model --ocr \
  --gemma-api-url http://localhost:11434/v1 --gemma-api-model gemma4:e4b \
  --qwen --qwen-api-url http://localhost:11434/v1 --qwen-model qwen2.5vl:7b \
  --reasoning-api-url http://localhost:11434/v1 --reasoning-model deepseek-r1:14b

# Select specific models (default: GPU-aware auto-selection):
python main.py --mode demo --asr --asr-model openai/whisper-large-v3

python main.py --mode demo --qwen --qwen-model Qwen/Qwen2.5-VL-72B-Instruct --qwen-api-url http://host:8010/v1

# Full options
python main.py --mode demo --help
```

### Output artifacts

For each video `<name>.mp4` the demo writes `<output-dir>/<name>/`:

| File / Dir | Contents |
|---|---|
| `gemma_analysis.md` | Step J: Gemma scene change detection, clustering, classification, CLIP+DINOv3 comparison |
| `gemma_captions.md` | Step J: Per-frame natural-language descriptions from Gemma 4 sidecar (`--gemma-api-url`) |
| `base_search.md` | Nearest-neighbour results with the base DINOv3 model |
| `scene_captions.md` | Per-frame Florence-2 captions (step L) |
| `asr_subtitles.md` | Whisper ASR segments + per-frame subtitle coverage (step M, `--asr`) |
| `multimodal_features.md` | OCR text, depth percentiles, detections, world model (steps N–Q) |
| `detailed_captions.md` | Qwen VLM structured per-frame scene analysis with ASR context (step R, `--qwen`) |
| `finetune_stats.md` | SSL fine-tuning loss curve + config |
| `finetuned_search.md` | Same queries re-run with the fine-tuned model |
| `comparison.md` | Side-by-side comparison + video-to-text description |
| `edge_models/` | ONNX model + frame gallery for edge deployment |
| `checkpoints/` | Fine-tuned `.pt` checkpoints |
| `3d_map/sparse_map.ply` | Sparse point cloud (SfM camera centres or PCA fallback) |
| `3d_map/gaussian_splat.ply` | 3D Gaussian Splat — open in SuperSplat or `view_splat.html` |
| `3d_map/view_splat.html` | Standalone browser viewer (serve with `python -m http.server 8765`) |
| `final_stats.md` | Per-video and aggregate statistics |

After all videos are processed, an interactive 3D scatter viewer opens for each video (close with the on-screen button or the window's close control). To view the Gaussian Splat, use the HTML viewer or SuperSplat (see §3D Gaussian Splat map above).

## Docs
- [Architecture decisions](docs/adr/README.md)
- [Design doc](docs/design/outdoor-autonomy-perception-stack.md)
- [Overview](docs/overview.md)
- [Setup](docs/setup.md)
- [Develop](docs/develop.md)
- [API](docs/api.md)
- [UI](docs/ui.md)
- [Helpers](docs/helpers.md)
- [Configuration](docs/configuration.md)
- [Pipeline](docs/pipeline.md)
- [Architecture](docs/architecture.md)
- [Examples](docs/examples.md)
- [Data layout](docs/data_layout.md)
- [Performance](docs/performance.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Licensing](docs/licensing.md)
- [Tests](docs/tests.md)
- [Learning path & sources](docs/learning_path.md)

# Skils
[kasetto](https://github.com/pivoshenko/kasetto)

[YK](https://github.com/garrytan/gstack)
/office-hours → /plan-ceo-review → /plan-eng-review → [build] → /review → /qa → /ship

cloc $(git ls-files)

[G](https://github.com/sickn33/antigravity-awesome-skills)
[ACC](https://github.com/hesreallyhim/awesome-claude-code)
[ECC](https://github.com/affaan-m/everything-claude-code)

[Build](https://github.com/codecrafters-io/build-your-own-x)
