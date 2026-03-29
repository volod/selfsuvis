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

# Full multimodal — Florence local (Ollama auto-evicted before step L), Qwen via vLLM Docker:
python main.py --mode demo --asr --ocr --qwen \
  --qwen-api-url http://localhost:8010/v1

# Full multimodal — Florence local (Ollama auto-evicted), Qwen via Ollama:
python main.py --mode demo --asr --ocr --qwen \
  --qwen-api-url http://localhost:11434/v1 --qwen-model qwen2.5vl:7b

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

## Sources

### Self-Supervised Video Learning

- [V-JEPA 2 — Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985) — Assran et al., 2025. Combines large-scale internet video pretraining with limited robot interaction data to produce self-supervised video models capable of world-understanding and physical planning.
- [V-JEPA 2.1 — Unlocking Dense Features in Video Self-Supervised Learning](https://arxiv.org/pdf/2603.14482) — Mur-Labadia et al., 2026. Extends V-JEPA 2 with a dense predictive loss that forces both visible and masked tokens to contribute, producing high-quality dense visual representations for images and video.
- [VideoMAE — Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602) — Tong et al., 2022. Applies masked autoencoding to video with an extremely high tube-masking ratio, producing strong spatiotemporal representations from far less labeled data than prior video models.
- [Intuitive physics understanding emerges from self-supervised pretraining on natural videos](https://arxiv.org/html/2502.11831v1) — Garrido et al., 2025. Shows that video prediction models trained via masked prediction in representation space spontaneously acquire intuitive physics concepts such as object permanence and shape consistency.
- [IntPhys 2 — Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments](https://arxiv.org/abs/2506.09849) — Bordes et al., 2025. Introduces a video benchmark testing deep learning models on four core intuitive physics principles in complex synthetic scenes.

### Vision Foundation Models

- [DINOv3](https://ai.meta.com/dinov3/) · [GitHub](https://github.com/facebookresearch/dinov3) · [Paper](https://ai.meta.com/research/publications/dinov3/) — Meta AI. Self-supervised vision transformer. Primary image embedding model for robot camera views (named vector `dino`).
- [DINOv2 — Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — Oquab et al., 2023. Trains all-purpose visual features on a curated large-scale dataset, achieving strong zero-shot performance across diverse vision tasks without fine-tuning.
- [DINO — Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) — Caron et al., 2021. Demonstrates that self-supervised ViTs trained with DINO develop semantic segmentation structure and scene layout understanding not seen in supervised counterparts.
- [Florence-2 — Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242) — Xiao et al., 2023. Single prompt-based vision-language model handling detection, segmentation, captioning, and grounding with a unified sequence-to-sequence architecture. Used for image-to-text captioning in this system.
- [CLIP — Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — Radford et al., 2021. Learns visual representations by contrastively aligning images and natural language captions at scale. Retained as cross-modal text↔image search vector (`clip`).
- [R3M — A Universal Visual Representation for Robot Manipulation](https://arxiv.org/abs/2203.12601) — Nair et al., 2022. Pretrains compact visual representations on diverse human ego-video using time-contrastive learning for data-efficient robot learning. Strong prior for robot camera views.
- [SAM — Segment Anything](https://arxiv.org/abs/2304.02643) — Kirillov et al., 2023. Promptable segmentation foundation model trained on 1B+ masks that generalizes to arbitrary segmentation tasks zero-shot.
- [Depth Anything V2](https://arxiv.org/abs/2406.09414) — Yang et al., 2024. Monocular depth estimation; enables metric-scale depth from a single camera without LiDAR or stereo rig.
- [GaussianFusion — Gaussian-Based Multi-Sensor Fusion for End-to-End Autonomous Driving](https://arxiv.org/abs/2506.00034) — Liu et al., 2025. Uses 3D Gaussians as a unified scene representation to fuse multi-sensor inputs for interpretable, end-to-end autonomous driving.

### 3D Scene Reconstruction

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — Kerbl et al., 2023. Represents scenes as explicit 3D Gaussians to achieve real-time, high-quality novel-view synthesis. Core 3D map reconstruction method (`nerfstudio splatfacto`).
- [NeRF — Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) — Mildenhall et al., 2020. Foundational neural radiance field method for novel view synthesis from posed images.
- [Instant NGP — Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://arxiv.org/abs/2201.05989) — Müller et al., 2022. Reduces NeRF training and rendering time by orders of magnitude via a learned multiresolution hash table encoding.
- [Awesome NeRF and 3DGS SLAM](https://github.com/3D-Vision-World/awesome-NeRF-and-3DGS-SLAM) — Curated list of SLAM methods combining neural radiance fields and Gaussian splatting.
- [Awesome 3D Vision 2026 Conference](https://github.com/harpreetsahota204/awesome_3DVision_2026_conference) — Papers from 3D Vision 2026.

### Knowledge Transfer & Edge Deployment

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton et al., 2015. Introduces knowledge distillation: a small student network matches a large teacher by training on soft probability outputs. Foundation of the knowledge transfer methodology used in this system.
- [Knowledge Transfer in Model-Based Reinforcement Learning Agents for Efficient Multi-Task Learning](https://arxiv.org/abs/2501.05329) — Kuzmenko et al., 2025. Distills a 317M-parameter multi-task world model into a 1M-parameter student that achieves state-of-the-art benchmark performance. Reference for world-model knowledge distillation.
- [Transformers in Reinforcement Learning: A Survey](https://arxiv.org/abs/2307.05979) — Agarwal et al., 2023. Surveys how transformer architectures address key RL challenges. Reference for edge agent architectures.

### Multimodal & Scene Understanding

- [A Comprehensive Review of Multimodal Large Language Models](https://arxiv.org/abs/2408.01319) — Wang et al., 2024. Reviews capabilities and limitations of MLLMs integrating text, images, video, and audio across diverse tasks.

### Tools & Frameworks

- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) — Modular framework for neural radiance fields and 3D Gaussian Splatting (`splatfacto`). Used for dense 3D map reconstruction.
- [MediaMTX](https://github.com/bluenviron/mediamtx) — Zero-dependency streaming server (RTSP, RTMP, WebRTC, HLS, SRT). Used for video stream ingestion.
- [COLMAP / pycolmap](https://github.com/colmap/pycolmap) — Structure-from-Motion and multi-view stereo. Used for camera pose estimation.
- [Qdrant](https://github.com/qdrant/qdrant) — Vector database with named vector support. Stores `clip` and `dino` embeddings with spatial metadata.
- [ROS / Agentic ROS](https://agenticros.com/) — Robot Operating System reference for future ROS2 integration.
- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) — OCR foundation model for text extraction from scene imagery.
- [NVIDIA PhysicsNemo](https://developer.nvidia.com/physicsnemo) — Physics-informed ML framework for simulation and physical world modeling.
- [rl-tools / Raptor](https://github.com/rl-tools/raptor) — High-performance RL inference on edge/embedded devices.
- [OpenGauss](https://github.com/math-inc/OpenGauss) — Open-source Gaussian process framework.
- [build-your-own-x](https://github.com/codecrafters-io/build-your-own-x) — Reference implementations for learning core systems from scratch.

# Skils
[kasetto](https://github.com/pivoshenko/kasetto)

[YK](https://github.com/garrytan/gstack)
/office-hours → /plan-ceo-review → /plan-eng-review → [build] → /review → /qa → /ship

cloc $(git ls-files)

[G](https://github.com/sickn33/antigravity-awesome-skills)
[ACC](https://github.com/hesreallyhim/awesome-claude-code)
[ECC](https://github.com/affaan-m/everything-claude-code)

[Build](https://github.com/codecrafters-io/build-your-own-x)
