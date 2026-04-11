# Learning Path

This document follows the real `python main.py --mode local` pipeline in this repo.
It explains, step by step:

- what each step does
- which model or tool is used
- what `auto` resolves to in practice in this codebase
- which paper to read first
- how a human should study the topic behind that step

Purpose note:

- This document is intentionally written as a detailed description of the approaches used at each pipeline step.
- It is meant for a human who wants to deep dive into the underlying technology, not just for someone who wants to run the pipeline once.
- Use it as a study guide for the representation-learning, multimodal reasoning, mapping, training, deployment, and audit choices made in this codebase.

The current local full-analysis pipeline has 35 ordered steps per video:

**Perception and analysis (Steps 1–20)**

1. Frame extraction
2. Vector store indexing (CLIP + DINOv3)
3. Gemma open-weight multimodal analysis
4. Florence scene captioning
5. ASR transcription
6. OCR text extraction
7. Depth estimation (monocular)
8. Object detection (HuggingFace RT-DETR / Grounding DINO)
9. RF / SDR electromagnetic passive sensing (TorchSig)
10. Thermal / infrared imaging (LWIR)
11. Multispectral / hyperspectral imaging
12. Event camera (neuromorphic sensing)
13. LiDAR / active ranging (ToF, FMCW)
14. Radar (FMCW, Doppler, SAR)
15. GNSS-R + satellite signal reception (ADS-B, AIS, NOAA APT, GOES)
16. Inertial + barometric sensing (IMU, barometer, anemometer)
17. Atmospheric / environmental sensing (temperature, humidity, wind)
18. Chemical / gas / radiation sensing
19. Acoustic sensing (mic arrays, ultrasonic, hydrophone, infrasound)
20. Sensor fusion analysis

**Detection, tracking, and 3D reconstruction (Steps 21–27)**

21. YOLO11 + SAM2/3 detection and segmentation
22. Gemma 4 directed tracking
23. World model video embeddings
24. Qwen detailed captioning
25. UniDriveVLA expert analysis
26. Base model search test
27. 3D map + Gaussian Splat

**Self-supervised learning and model adaptation (Steps 28–35)**

28. SSL DINOv3 fine-tuning
29. Knowledge distillation — maximum hydration chain
30. ONNX export + gallery build
31. Fine-tuned search test
32. Model comparison + video description
33. Multi-model comparison
34. Video synthesis
35. Agentic flow audit

## Before You Start

Minimum practical setup:

1. Create the venv: `make venv`
2. Install `ffmpeg`
3. Put `.mp4` or `.mov` files in `data_test/videos/`
4. Optionally run Qdrant on `localhost:6333`
5. Optionally prefetch local-model assets with `python scripts/prepare_models.py --all`

Useful mental model:

- Steps 1–2 build the raw visual memory (frames + vectors).
- Steps 3–8 attach language, text, geometry: captioning, ASR, OCR, depth, object detection.
- Steps 9–19 extend perception into the physical world: each step covers one sensing modality grouped by its physical principle and SIGINT algorithm (RF/SDR, thermal, multispectral, event camera, LiDAR, radar, GNSS-R/satellite, IMU/inertial, atmospheric, gas/radiation, acoustic).
- Step 20 fuses all modalities into `frame_facts_json["sensor_fusion"]` with temporal alignment, cross-modal detections, and active-learning escalation.
- Steps 21–22 run YOLO11 + SAM and Gemma-directed tracking — consuming sensor-fusion context.
- Steps 23–27 add world model embeddings, Qwen captioning, UniDriveVLA, search tests, and 3D Gaussian Splat.
- Steps 28–35 adapt, compress, evaluate, and synthesise the full representation.

Study note:

- The numbered list above is the canonical runtime order.
- Some deep-dive sections later in this document are grouped pedagogically rather than strictly in runtime order.
- When in doubt, use the canonical 35-step list and [`pipeline/workflows/local/runner.py`](/home/vola/src/selfsuvis/pipeline/workflows/local/runner.py) as the execution source of truth.

The agentic flow (Steps 3 → 4 → 5–9 → 20 → 21 → 22): see the
**Agentic Knowledge Flow** section below for a full data-flow diagram.

## Local Full Run Setup

This section covers everything needed to run all 35 pipeline steps locally — including sidecar VLM/LLM servers, sensor sample data, and model weights. Run `scripts/setup_local_full.sh` for a one-shot bootstrap, or follow the steps below manually.

---

### 1. Base environment

```bash
# Python environment
make venv                             # creates .venv with all pip deps

# System dependencies (Ubuntu/Debian)
sudo apt-get install -y ffmpeg curl wget git python3-dev

# Optional: verify GPU
nvidia-smi                            # must show your GPU for CUDA steps
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

### 2. Core model weights

```bash
# OpenCLIP + DINOv3 (always required)
.venv/bin/python scripts/prepare_models.py

# Step 4 — Florence-2 captioning
.venv/bin/python scripts/prepare_models.py --florence

# Step 5 — Whisper ASR
.venv/bin/python scripts/prepare_models.py --whisper

# Step 6 — OCR (auto-selects by VRAM)
.venv/bin/python scripts/prepare_models.py --ocr

# Step 7 — Depth estimation (Apple DepthPro default)
.venv/bin/python scripts/prepare_models.py --depth

# Step 8 — HuggingFace RT-DETR / Grounding DINO
.venv/bin/python scripts/prepare_models.py --detection

# Step 21 — YOLO11l detection
.venv/bin/python scripts/prepare_models.py --yolo

# Step 21 — SAM3 / SAM2 segmentation
.venv/bin/python scripts/prepare_models.py --sam

# Step 23 — World model video embeddings
.venv/bin/python scripts/prepare_models.py --world-model

# Step 25 — UniDriveVLA expert analysis
.venv/bin/python scripts/prepare_models.py --unidrive

# Step 3/22 — Gemma 4 open-weight (requires HF_TOKEN; ~8 GiB)
export HF_TOKEN=<your_huggingface_token>
.venv/bin/python scripts/prepare_models.py --gemma

# Download everything at once
HF_TOKEN=<your_token> .venv/bin/python scripts/prepare_models.py --all
```

---

### 3. Sidecar VLM / LLM servers (Ollama)

Steps 3 and 22 (Gemma multimodal), Step 24 (Qwen2.5-VL), and Step 25 (UniDriveVLA) can run against local Ollama or vLLM endpoints instead of loading weights directly.

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull models for each step
# Step 3 / Step 22 — Gemma 4 multimodal (vision-capable)
ollama pull gemma4:e4b               # ~5 GiB, default
ollama pull gemma4:12b               # ~13 GiB, higher quality

# Step 24 — Qwen2.5-VL detailed captioning
ollama pull qwen2.5vl:7b             # ~5 GiB
ollama pull qwen2.5vl:72b            # ~45 GiB, research grade

# Step 25 — UniDriveVLA (if available as Ollama model)
# UniDriveVLA is typically served via vLLM; see vLLM section below

# Verify Ollama is running
ollama serve &                        # starts on http://localhost:11434
curl http://localhost:11434/api/tags  # should list pulled models

# Run the pipeline pointing at Ollama
.venv/bin/python main.py --mode local   --gemma-api-url http://localhost:11434/v1   --qwen-api-url  http://localhost:11434/v1   --input data_test/videos/mission.mp4
```

---

### 4. Sidecar VLM server (vLLM)

vLLM is preferred for Qwen2.5-VL and UniDriveVLA when serving multiple workers or when Ollama does not support the model.

```bash
# Install vLLM (CUDA required)
pip install vllm

# Step 24 — Qwen2.5-VL via vLLM
python -m vllm.entrypoints.openai.api_server   --model Qwen/Qwen2.5-VL-7B-Instruct   --port 8010   --max-model-len 8192 &

# Step 25 — UniDriveVLA via vLLM
python -m vllm.entrypoints.openai.api_server   --model owl10/UniDriveVLA_Nusc_Base_Stage3   --port 8030   --max-model-len 4096 &

# Run the pipeline pointing at vLLM endpoints
.venv/bin/python main.py --mode local   --gemma-api-url   http://localhost:11434/v1   --qwen-api-url    http://localhost:8010/v1   --unidrive-api-url http://localhost:8030/v1   --input data_test/videos/mission.mp4
```

---

### 5. Sensor sample data

Download public sample data for each physical sensor modality. Each sidecar file is placed beside its matching video in `data_test/videos/`. The `scripts/prepare_sensor_data.sh` script automates all downloads below.

```bash
# One-shot download of all sensor sample data
bash scripts/prepare_sensor_data.sh data_test/videos/
```

**What the script downloads (per sensor step):**

| Step | Modality | Sample | Source |
|---|---|---|---|
| 9 | RF / SDR | RadioML 2018.01a (1 shard, ~300 MB) | https://www.deepsig.ai/datasets |
| 10 | Thermal | FLIR ADAS sample images (RGB+thermal pairs) | https://www.flir.com/oem/adas/adas-dataset-form/ |
| 11 | Multispectral | Indian Pines hyperspectral (salinas_corrected.mat) | http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes |
| 12 | Event camera | N-Caltech101 sample (100 events files, ~50 MB) | https://www.garrickorchard.com/datasets/n-caltech101 |
| 13 | LiDAR | KITTI odometry sequence 00 velodyne (2 scans, ~20 MB) | https://www.cvlibs.net/datasets/kitti/ |
| 14 | Radar | RADIATE Oxford sequence sample (5 frames, ~30 MB) | https://pro.hw.ac.uk/radiate/ |
| 15 | GNSS-R / ADS-B | CYGNSS DDM sample (1 orbit, ~50 MB) + OpenSky 1-hour ADS-B JSON | https://podaac.jpl.nasa.gov/dataset/CYGNSS_L1_V3.1 |
| 16 | IMU | EuRoC MAV MH_01_easy (IMU + ground truth, ~20 MB) | https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.pdf |
| 17 | Atmospheric | ERA5 single-level sample (1 day, 0.25° grid, ~5 MB) | https://cds.climate.copernicus.eu/ |
| 18 | Gas / radiation | OpenAQ CSV (24 h, one station) + Safecast GeoJSON sample | https://openaq.org/ |
| 19 | Acoustic | ESC-50 sample (10 clips, ~5 MB) + xeno-canto 5 bird recordings | https://github.com/karolpiczak/ESC-50 |

---

### 6. End-to-end local run command

After setup, run all 35 steps on a test video with all sidecars present:

```bash
# Minimal run (Steps 1–9, no sidecar servers needed)
.venv/bin/python main.py --mode local   --input data_test/videos/mission.mp4   --no-qdrant

# Full run with Ollama sidecars + all sensors enabled
.venv/bin/python main.py --mode local   --input data_test/videos/mission.mp4   --gemma-api-url    http://localhost:11434/v1   --qwen-api-url     http://localhost:11434/v1   --unidrive-api-url http://localhost:8030/v1   --rfdetr-model     base   SENSOR_FUSION_ENABLED=true

# Full run with Docker stack (PostgreSQL + Qdrant + worker)
make up
python scripts/migrate_postgres.py
curl -s -H "X-API-Key: $API_KEY"   -F "file=@data_test/videos/mission.mp4"   http://localhost:8000/index/video | python -m json.tool
```

---

### 7. Environment variable reference for sensor steps

```bash
# Step 9 — RF/SDR
RF_ENABLED=true
RF_SAMPLE_RATE=1000000
RF_WINDOW_SEC=0.5
RF_CLASSIFIER_CHECKPOINT=data/models/rf_classifier.pt

# Steps 10–19 — Physical sensors (all default false; enable per mission)
THERMAL_ENABLED=true
THERMAL_MODEL=auto                     # auto-resolves to YOLO-nano (FLIR ADAS fine-tune)

MULTISPECTRAL_ENABLED=true
MULTISPECTRAL_BANDS=R,G,RE,NIR         # band names in the sidecar GeoTIFF directory

EVENT_CAMERA_ENABLED=true
EVENT_WINDOW_MS=10                     # accumulation window for event-to-frame conversion

LIDAR_ENABLED=true
LIDAR_VOXEL_SIZE=0.1                   # voxel grid leaf size (metres)

RADAR_ENABLED=true
RADAR_CFAR_THRESHOLD=15                # CFAR threshold in dB above noise floor

GNSS_R_ENABLED=true
ADSB_ENABLED=true
ADSB_CONFLICT_RADIUS_M=3000
ADSB_CONFLICT_ALT_M=300

IMU_ENABLED=true
IMU_FREQ_HZ=200

WEATHER_ENABLED=true

GAS_ENABLED=true
GAS_VOC_ALARM_PPB=1000
GAS_DOSE_ALARM_USV_H=1.0              # hard-flags al_tag="needs_annotation"

ACOUSTIC_ENABLED=true
ACOUSTIC_SAMPLE_RATE=48000

# Step 20 — Sensor fusion
SENSOR_FUSION_ENABLED=true
SENSOR_FUSION_MAX_LAG_MS=100           # max timestamp offset to align a sensor reading to a frame
```

---


## Step-By-Step Learning Path

### Step 1. Frame extraction

**What the local pipeline does**

The pipeline uses `ffmpeg` to decode the source video and save JPEG frames at the requested FPS. This is not an ML step, but every downstream model depends on its output quality and sampling rate.

**Tool / model used**

- `ffmpeg`
- Output: `data/local_runs/<video>/frames/`

**Why it matters**

This step decides temporal resolution. If you sample too sparsely, you lose motion and speech alignment. If you sample too densely, every later step gets slower and more expensive.

**Essential reading**

- FFmpeg documentation: https://ffmpeg.org/documentation.html

**How a human should learn this topic**

Learn video basics first: FPS, GOP/keyframes, H.264/H.265 compression, color spaces, and how frame rate changes affect motion analysis. Then practice extracting the same clip at `1`, `2`, `4`, and `8` FPS and compare what information survives.

---

### Step 2. Vector store indexing

**What the local pipeline does**

Each extracted frame is embedded into two visual spaces and then inserted into a vector store. This is the memory layer used later for search, comparison, and retrieval-based reasoning.

**Models used in this repo**

- `OpenCLIP` image encoder from [models/openclip_model.py](../models/openclip_model.py)
- `DINO` image encoder from [models/dino_model.py](../models/dino_model.py)
- Current default OpenCLIP config: `ViT-B-16` with the `openai` weights
- Current DINO label in the local pipeline: `dinov3_vitb14`
- Important repo detail: in this codebase, `dinov3_*` is an alias for `DINOv2` register-token checkpoints served from `facebookresearch/dinov2`
- Store backend: Qdrant if available, otherwise in-memory cosine search

**Why it matters**

This is the core representation-learning stage. CLIP gives image-text alignment; DINO gives strong self-supervised visual similarity. The system uses both, but DINO is the main retrieval backbone for frame-to-frame search.

**Essential reading**

- CLIP: https://arxiv.org/abs/2103.00020
- DINOv2: https://arxiv.org/abs/2304.07193

**How a human should learn this topic**

Start with embeddings, cosine similarity, and nearest-neighbour retrieval. Then learn the difference between contrastive vision-language models like CLIP and self-supervised pure-vision models like DINO. Finally, study approximate nearest-neighbour indexing and why Qdrant/HNSW matters for scale.

---

### Step 3. Gemma 4 open-weight multimodal analysis

**What the local pipeline does**

The pipeline loads `GemmaEmbedder` (backed by `google/gemma-4-it-2b` or the configured
`GEMMA_MODEL_ID`) and runs six embedding-based analyses on a sample of up to 30 frames:

1. **Scene change detection** — consecutive-frame cosine distance flags visual transitions
2. **Scene clustering** — greedy cosine grouping into visually distinct scene states
3. **Zero-shot scene classification** — frame embeddings matched against a label vocabulary
4. **Cross-modal text → frame retrieval** — text probe embeddings retrieved against frame embeddings
5. **Temporal video embedding** — mean-pool of all frame embeddings → one video-level vector
6. **Generative frame descriptions** — when `GEMMA_API_URL` is set (Ollama/vLLM sidecar),
   each frame is described in natural language via the Gemma 4 chat endpoint

When a Gemma sidecar is configured, each frame also receives a **generative description**
written to `gemma_captions.md`.

After all analyses, the step loads a temporary DINOv3 ViT-B/14 and OpenCLIP ViT-B/16 and
computes two cross-model comparison metrics on the same frames:
- **Mean pairwise cosine similarity** (lower = more discriminative embedding space)
- **Mutual nearest-neighbour overlap @ k** (fraction of frames whose top-k neighbours agree
  between Gemma and DINOv3/CLIP — measures structural alignment between embedding spaces)

Results and interpretation are written to `gemma_analysis.md`.

**Models used**

- `GemmaEmbedder` in [models/gemma_model.py](../models/gemma_model.py)
- Current local default: `google/gemma-3-4b-it` via HuggingFace `transformers`
- The processor is now loaded with `use_fast=False` explicitly so runtime behavior stays stable across future `transformers` releases
- Vision encoder: Gemma multimodal vision path pooled into the language hidden space
- Embedding dim: 2560 (Gemma 4 2B hidden size), L2-normalised
- Optional sidecar: Ollama `gemma4:e4b` at `GEMMA_API_URL` for generative descriptions on 16 GB-class GPUs
- Requires `HF_TOKEN` for gated HuggingFace model download
- Important runtime detail: the embedder instance is reused across the local pipeline, and image embeddings are cached in-process to avoid recomputing the same frame repeatedly across indexing, Gemma analysis, and Gemma-teacher distillation

**Why it matters**

Gemma 4 is a natively multimodal model: its image embeddings live in the same vector space as
its text embeddings, so a text query and a frame can be compared directly without a separate
cross-modal alignment step (unlike CLIP, which requires paired training).  This matters for
outdoor autonomy because mission queries are often textual ("find frames with a T-intersection")
while the indexed content is visual.

The DINOv3 and CLIP comparisons answer the practical question: *how much of the visual structure
that specialised vision models capture is also present in Gemma's embedding space?*  The MNN
agreement metric directly predicts whether Gemma can substitute DINOv3 for retrieval tasks.

**Key findings from this codebase**

See full analysis in `docs/design/gemma4-video-analysis-ssl-distillation.md`.  Summary:

- On 30 fps outdoor video, Gemma embeddings of near-duplicate frames have **near-zero pairwise
  cosine similarity** (0.0000 in recorded runs) — the language backbone forces even similar
  frames apart in embedding space.  DINOv3 is far more stable (0.94+ on the same frames).
- **MNN agreement with DINOv3 can reach 100 %** on small samples — Gemma and DINOv3 agree
  on which frames are nearest neighbours even though Gemma's global cosine distances are larger.
  This means Gemma is a valid retrieval backbone when the query is text-based.
- **Gemma uniquely enables multi-frame reasoning**: unlike Florence-2, Qwen, or DINOv3, it can
  accept multiple images in one call and reason about what changed between them — directly solving
  the 30 fps redundancy problem in `scene_captions.md`.
- Gemma 4 can **replace the Qwen sidecar** for structured scene extraction (vehicle groups, road
  surface, road condition) without requiring an Ollama process at 10–12 GB VRAM.

**Essential reading**

- Gemma 4 technical report: https://arxiv.org/abs/2503.19786
- SigLIP (Gemma's vision encoder): https://arxiv.org/abs/2303.15343
- DINOv2 (for comparison context): https://arxiv.org/abs/2304.07193
- CLIP (for comparison context): https://arxiv.org/abs/2103.00020

**How a human should learn this topic**

Start with the distinction between contrastive image-text models (CLIP) and natively multimodal
generative models (Gemma 4): CLIP learns a shared vision-language space via paired training;
Gemma 4 produces cross-modal embeddings as a byproduct of its generative objective.  Then study
mean-pooled hidden-state embeddings versus dedicated embedding heads.  Finally, learn to interpret
MNN overlap as a model-agreement metric: high MNN means two models have learned the same notion
of visual similarity; low MNN means they've specialised for different aspects of the content.

A good practical exercise: run this step on a video, then open `gemma_analysis.md` and ask
whether the MNN agreement is above or below 0.7.  That number tells you directly whether you can
swap DINOv3 for Gemma embeddings in Qdrant without losing retrieval quality.

---

### Step 4. Florence scene captioning

**What the local pipeline does**

The local pipeline captions every keyframe with a detailed textual scene description. This gives you a readable semantic summary before later multimodal steps add speech, OCR, or object structure.

**Agentic enrichment (new)**

If Step 3 (Gemma) completed successfully and identified a dominant scene type, the pipeline forwards a `domain_hint` string into the Florence prompt:

```
[Context: Dominant scene: military convoy | Known objects: truck, soldier]
<MORE_DETAILED_CAPTION>
```

This steers Florence toward domain-specific vocabulary before the caption is even generated.  The hint is built by `VideoKnowledge.domain_hint()` and is empty if Gemma was skipped or found no scene type.

**Model used**

- `microsoft/Florence-2-large`
- Wrapper: [pipeline/florence_model.py](../pipeline/florence_model.py)
- Prompt used by the repo: `<MORE_DETAILED_CAPTION>` (with optional domain hint prefix)

**Why it matters**

This step turns raw pixels into natural-language scene summaries. Those summaries help both debugging and downstream human inspection.  The captions are also stored in `VideoKnowledge` as the canonical per-frame scene description used by all later steps.

**Essential reading**

- Florence-2: https://arxiv.org/abs/2311.06242

**How a human should learn this topic**

Learn image captioning as a sequence-generation problem. Then study prompt-conditioned vision models and how one model can do captioning, detection, grounding, and segmentation with task prompts. A good exercise is to compare Florence captions against hand-written captions for 20 frames and note what details the model systematically misses.

---

### Step 5. ASR transcription

**What the local pipeline does**

The pipeline extracts audio from the video, runs speech recognition, and aligns subtitle segments to video frames.

**Model used**

- Wrapper: [pipeline/vision/asr.py](../pipeline/vision/asr.py)
- Practical default in this repo: `openai/whisper-large-v3-turbo`
- Important repo behavior: if `ASR_MODEL=auto` selects a non-Whisper model that cannot provide native timestamps in this pipeline, the wrapper falls back to Whisper

**Why it matters**

Speech often contains mission context, place names, instructions, or narration that is not visible in the image. Step 9 later injects this ASR text into the VLM prompt.

**Essential reading**

- Whisper: https://openai.com/research/whisper/

**How a human should learn this topic**

Study audio preprocessing, spectrograms, encoder-decoder speech models, and timestamp alignment. Then learn practical ASR failure modes: overlapping speakers, clipped words, noise, and code-switching. A good exercise is to compare Whisper output against ground truth for one short clip and mark insertions, deletions, and timing drift.

---

### Step 6. OCR text extraction

**What the local pipeline does**

The pipeline looks for visible text inside each frame. That text can come from road signs, dashboards, labels, UI overlays, subtitles burned into the video, equipment markings, or documents in view.

**Models used**

- Wrapper: [pipeline/vision/ocr.py](../pipeline/vision/ocr.py)
- `OCR_MODEL=auto` is GPU-aware and can choose TrOCR, GOT-OCR2, Florence, Qwen, Phi-3.5 Vision, or DeepSeek OCR depending on setup
- In recent local runs, `auto` selected `microsoft/Phi-3.5-vision-instruct`
- When a Qwen/Ollama sidecar is already active, the repo can route OCR through that sidecar instead of loading another heavy local VLM

**Why it matters**

OCR is often the difference between generic understanding and operational understanding. Text tells you what object you are looking at, not just what it resembles.

**Essential reading**

- TrOCR: https://arxiv.org/abs/2109.10282
- Qwen2.5-VL technical report: https://arxiv.org/abs/2502.13923

**How a human should learn this topic**

Start with the difference between document OCR and scene-text OCR. Then learn layout, perspective distortion, low-resolution text, multilingual text, and text-plus-graphics reasoning. A practical exercise is to collect 50 failure cases and group them into small text, blur, low contrast, non-Latin script, curved text, and occlusion.

---

### Step 7. Depth estimation

**What the local pipeline does**

The pipeline predicts monocular depth and stores a compact five-number summary per frame instead of a full dense depth map.

**Model used**

- Wrapper: [pipeline/depth_model.py](../pipeline/depth_model.py)
- `DEPTH_MODEL=auto` is registry-driven
- In recent local runs, `auto` selected `apple/DepthPro-hf`
- The wrapper will retry on CPU if CUDA runs out of memory

**Why it matters**

Depth gives a lightweight geometric prior: near/far structure, scene openness, clutter, and relative scale. That becomes useful for motion interpretation, 3D reconstruction, and future robotics extensions.

**Essential reading**

- Depth Pro: https://arxiv.org/abs/2410.02073

**How a human should learn this topic**

Study the difference between metric depth, relative depth, and inverse depth. Then learn why monocular depth is fundamentally ambiguous and how modern models still recover useful structure from large-scale training. A good exercise is to inspect depth predictions on indoor, outdoor, aerial, and low-light frames and see where relative ordering breaks.

---

### Step 8. Object detection

**What the local pipeline does**

The pipeline predicts object instances and normalized bounding boxes for each frame.

**Model used**

- Wrapper: [pipeline/vision/detection.py](../pipeline/vision/detection.py)
- `DETECTION_MODEL=auto` is registry-driven
- In recent local runs, `auto` selected `SenseTime/deformable-detr`
- Open-vocabulary alternatives are also supported via `DETECTION_LABELS`

**Why it matters**

Detection converts a scene from global semantics into object-level structure. It is the first step toward counting, tracking, event reasoning, and symbolic world state.

**Essential reading**

- Deformable DETR: https://arxiv.org/abs/2010.04159

**How a human should learn this topic**

Learn the difference between classification, detection, and segmentation. Then study IoU, confidence calibration, small-object failure modes, and open-vocabulary detection. A good exercise is to compare detector outputs on crowded frames versus sparse frames and see how confidence behaves.

---

### Step 9. RF signal analysis (TorchSig) *(new)*

**What the local pipeline does**

When an IQ capture file is present alongside the mission video (or when the audio track is used as
a proxy), `RFSignalAnalyzer` slices a 0.5-second IQ window around each kept frame and computes
four spectral metrics stored in `frame_facts_json["rf_signal"]`:

| Field | What it measures |
|---|---|
| `snr_db` | Estimated signal-to-noise ratio: top-10 % vs bottom-10 % bin energy (dB) |
| `spectral_flatness` | Wiener entropy — 0 = pure tone (narrow carrier), 1 = flat noise (wideband / jamming) |
| `peak_freq_ratio` | Normalised position of the dominant frequency bin (0–1 of bandwidth) |
| `occupied_bw_ratio` | Fraction of spectrum above noise floor + 3 dB (occupied channel estimate) |
| `modulation_class` | Optional — only present when `RF_CLASSIFIER_CHECKPOINT` is set |
| `source` | `iq_file` \| `sigmf` \| `audio_proxy` |

The pass runs after the HuggingFace detection step and before YOLO. It is CPU-only — no GPU
required. Implementation: [`pipeline/vision/rf_analyzer.py`](../pipeline/vision/rf_analyzer.py).

**Enable the pass:**

```bash
# With a raw IQ sidecar (interleaved float32 I/Q at 1 MHz):
RF_ENABLED=true RF_SAMPLE_RATE=1000000 python worker/main.py

# With SigMF (sample rate auto-read from .sigmf-meta):
RF_ENABLED=true python worker/main.py

# Fallback to audio proxy (no SDR hardware needed):
RF_ENABLED=true python worker/main.py   # falls back automatically if no .iq found
```

Place the IQ file next to the video with the same basename:
```
data/videos/mission_042.mp4
data/videos/mission_042.iq        ← raw interleaved float32
data/videos/mission_042.sigmf-data ← SigMF binary (+ mission_042.sigmf-meta)
```

**Recording your own IQ captures with SDR hardware**

You need a software-defined radio (SDR) that captures the drone's operating frequency band.
Common choices for drone/rover missions:

| Hardware | Price | Frequency | Bandwidth | Notes |
|---|---|---|---|---|
| **RTL-SDR v4** | ~$35 | 500 kHz – 1.75 GHz | 3.2 MHz | Best entry-level, USB |
| **HackRF One** | ~$340 | 1 MHz – 6 GHz | 20 MHz | Half-duplex, wide coverage |
| **LimeSDR Mini** | ~$200 | 10 MHz – 3.5 GHz | 30.72 MHz | Full-duplex |
| **USRP B205mini** | ~$800 | 70 MHz – 6 GHz | 56 MHz | Research grade, low noise |
| **ADALM-Pluto+** | ~$230 | 325 MHz – 3.8 GHz | 56 MHz | Full-duplex, AD9363; software-unlock extends to ~70 MHz – 6 GHz |
| **bladeRF 2.0 micro** | ~$480 | 47 MHz – 6 GHz | 56 MHz | Full-duplex, USB 3.0, FPGA-programmable; good mid-range option |
| **Per Vices Crimson TNG** | ~$7,000+ | 100 MHz – 18 GHz | up to 1 GHz | Full-duplex, multi-channel; covers X-band (8–12 GHz) and Ku-band; research/defense grade |

Common drone RF bands to tune to:
- **2.400–2.483 GHz** — Wi-Fi / DJI OcuSync / FPV control links
- **5.725–5.850 GHz** — 5.8 GHz FPV video downlink
- **900 MHz** — LoRa/FHSS long-range control links (ArduPilot, ExpressLRS)
- **1.575 GHz** — GPS L1 (monitor for jamming/spoofing)

**Step 1 — Install capture software**

```bash
# GNU Radio (Linux) — most flexible, scriptable:
sudo apt install gnuradio

# SDR++ (cross-platform GUI + audio/IQ recording):
# https://www.sdrpp.org — download AppImage / Windows installer

# SoapySDR + Python (direct API, useful for automated capture):
pip install SoapySDR
```

**Step 2 — Record an IQ capture file**

Using GNU Radio Companion (GRC):

1. Launch `gnuradio-companion`
2. Add: **RTL-SDR Source** (or HackRF / UHD USRP) → **File Sink**
3. Set **File Sink** output type to `Complex float32` — this is the native `.iq` format selfsuvis reads
4. Set center frequency to match the drone band (e.g. `2437000000` for 2.4 GHz Ch 6)
5. Set sample rate to match `RF_SAMPLE_RATE` (default `1000000` = 1 MHz)
6. Run the flowgraph during the mission; stop and save the `.iq` file

Using SoapySDR Python (scripted capture synced to video recording):

```python
import SoapySDR
import numpy as np

# Open RTL-SDR
sdr = SoapySDR.Device({"driver": "rtlsdr"})
sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, 1e6)
sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, 2.437e9)
sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 30)

rxStream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
sdr.activateStream(rxStream)

num_samples = int(1e6 * 30)  # 30 seconds at 1 MHz
buf = np.zeros(1024, dtype=np.complex64)

with open("mission.iq", "wb") as f:
    collected = 0
    while collected < num_samples:
        sr = sdr.readStream(rxStream, [buf], len(buf))
        if sr.ret > 0:
            # Write interleaved float32 I/Q (selfsuvis native format)
            out = np.empty(sr.ret * 2, dtype=np.float32)
            out[0::2] = buf[:sr.ret].real
            out[1::2] = buf[:sr.ret].imag
            f.write(out.tobytes())
            collected += sr.ret

sdr.deactivateStream(rxStream)
sdr.closeStream(rxStream)
```

**Step 3 — Record in SigMF format** (recommended — preserves sample rate and center frequency)

```bash
pip install sigmf

# With gr-sigmf GNU Radio block, or use the Python API:
python - <<'EOF'
import sigmf, sigmf.archive
import numpy as np
from sigmf import SigMFFile

# Create metadata file
meta = SigMFFile(
    data_file="mission.sigmf-data",
    global_info={
        SigMFFile.DATATYPE_KEY: "cf32_le",
        SigMFFile.SAMPLE_RATE_KEY: 1_000_000,
        SigMFFile.AUTHOR_KEY: "selfsuvis",
        SigMFFile.DESCRIPTION_KEY: "Drone 2.4 GHz control link capture",
    }
)
meta.add_capture(0, metadata={
    SigMFFile.FREQUENCY_KEY: 2_437_000_000,
    SigMFFile.DATETIME_KEY: "2025-06-01T12:00:00Z",
})
meta.tofile("mission.sigmf-meta")
EOF
# Then write raw CF32 I/Q samples to mission.sigmf-data (same float32 interleaved format)
```

**Step 4 — Place files and run**

```bash
# Name the IQ file to match the video basename:
cp mission.iq data/videos/mission_042.iq

# Run indexing with RF enabled:
RF_ENABLED=true RF_SAMPLE_RATE=1000000 python worker/main.py
```

**Training a modulation classifier**

TorchSig includes a synthetic dataset generator for 24 modulation classes. Train a classifier
and export it as TorchScript, then point the pipeline at it:

```bash
# Install TorchSig:
pip install torchsig

# Generate a training dataset (runs purely on CPU, no real SDR needed):
python - <<'EOF'
from torchsig.datasets import TorchSigNarrowband
from torchsig.datasets.dataset_metadata import NarrowbandMetadata

# 10 k samples per class, SNR sweep −20 dB to +30 dB
dataset = TorchSigNarrowband(
    root="data/torchsig_narrowband",
    train=True,
    num_samples=240_000,  # 10k × 24 classes
)
EOF

# Train a simple EfficientNet-B0 classifier on the generated dataset
# (see docs/rf_training.md for the full training script)

# Export to TorchScript:
python - <<'EOF'
import torch
model = torch.load("checkpoints/rf_classifier.pt")
model.eval()
scripted = torch.jit.script(model)
scripted.save("checkpoints/rf_classifier_jit.pt")
EOF

# Enable modulation classification in selfsuvis:
RF_ENABLED=true RF_CLASSIFIER_CHECKPOINT=checkpoints/rf_classifier_jit.pt python worker/main.py
```

See [`docs/rf_training.md`](rf_training.md) for the complete training pipeline.

**Why it matters**

Vision-only pipelines are blind to the radio environment during a mission. RF analysis adds a
complementary sensing axis: even without visual change, a sudden drop in `snr_db` or a spike in
`spectral_flatness` can indicate interference, jamming, or loss of the control link. Correlating
these events with specific frames creates a richer mission record and can flag safety-critical
moments that produce no visual artifact.

**Essential reading**

- TorchSig repository and documentation: https://github.com/torchdsp/torchsig
- DeepSig RadioML dataset paper: https://arxiv.org/abs/1602.04105 (O'Shea & Corgan 2016)
- SigMF specification: https://github.com/sigmf/SigMF
- GNU Radio tutorials: https://wiki.gnuradio.org/index.php/Tutorials
- RTL-SDR quick-start guide: https://www.rtl-sdr.com/rtl-sdr-quick-start-guide/

**Public IQ datasets for testing without real SDR hardware**

| Dataset | Modulations | Size | Link |
|---|---|---|---|
| DeepSig RadioML 2018.01a | 24 | 25 GB | https://www.deepsig.ai/datasets |
| TorchSig NarrowBand (synthetic) | 53 | Generate on demand | `torchsig.datasets.TorchSigNarrowband` |
| TorchSig WideBand (synthetic) | 53 | Generate on demand | `torchsig.datasets.TorchSigWideband` |
| SigMF reference captures | Various | Small | https://github.com/sigmf/SigMF/tree/main/examples |
| Signal Identification Wiki DB | Real-world | Various | https://www.sigidwiki.com/wiki/Database |

**How a human should learn this topic**

1. **Start with IQ fundamentals**: understand what in-phase and quadrature components represent, why complex-valued samples are used, and how sampling rate relates to observable bandwidth. The GNU Radio tutorials cover this well before you touch any hardware.
2. **Run the audio proxy first**: set `RF_ENABLED=true` on an existing video without any `.iq` file. The pass falls back to the audio track. Inspect the `rf_signal` dict in a frame record — you will see non-trivial `spectral_flatness` and `snr_db` values even from audio, which builds intuition for what the metrics mean.
3. **Download RadioML 2018**: load one `.hdf5` shard, slice out 1 024 complex samples of a known modulation (e.g. QAM16), save as a `.iq` file, and run it through `_extract_features()` in `pipeline/vision/rf_analyzer.py`. Observe how `spectral_flatness` near 0 for narrow carriers and near 1 for FM/noise.
4. **Capture with RTL-SDR**: tune to 100 MHz FM radio (strong, always-on signal), record 30 seconds, and run with `RF_ENABLED=true`. Verify `occupied_bw_ratio` is high (FM is wideband) and `spectral_flatness` is moderate.
5. **Train a toy classifier**: use `torchsig.datasets.TorchSigNarrowband` to generate a balanced 4-class dataset (BPSK / QPSK / QAM16 / FM), train a small CNN, export to TorchScript, and test the `modulation_class` field end-to-end in the pipeline.
6. **Compare against a mission video**: run a drone flight with RF capture at 2.4 GHz, then plot `snr_db` vs time. Look for correlation with GPS altitude (higher altitude → cleaner line-of-sight → higher SNR) or mission phases (take-off, hover, return).

---

### Step 10. Thermal / infrared imaging

**Physical principle**

All objects above absolute zero emit electromagnetic radiation proportional to their temperature (Planck's law). Long-wave infrared (LWIR, 8–14 µm) cameras detect this passive thermal emission without any active illuminator. Mid-wave infrared (MWIR, 3–5 µm) is used when targets are hot enough to contrast strongly against a cold sky (engines, missiles, exhaust plumes).

**SIGINT / signal processing algorithms**

| Algorithm | Purpose |
|---|---|
| Non-uniformity correction (NUC) | Two-point calibration (shutter flat-field + scene) to remove fixed-pattern noise from uncooled LWIR detectors |
| Radiometric calibration | Convert raw 14-bit ADU → apparent temperature (°C) using sensor datasheet look-up table + emissivity model |
| Thermal contrast normalisation | Histogram equalisation or CLAHE to make cold-sky / warm-ground contrast perceptually visible |
| Thermal detection (YOLO-nano on FLIR ADAS) | Fine-tuned nano-scale object detector; class set: person / vehicle / bike |
| Cross-modal IoU merge | Match thermal detections with RGB YOLO detections by projected IoU; disagreement → novelty signal |

**Hardware**

| Sensor | Resolution | NETD | Price | Notes |
|---|---|---|---|---|
| FLIR Lepton 3.5 | 160×120 | 50 mK | ~$200 | USB module; 8.7 Hz; radiometric via `pylepton` |
| FLIR Boson 640 | 640×512 | 50 mK | ~$3,000 | Drone-grade; 14-bit radiometric TIFF; CameraLink / USB3 |
| DJI Zenmuse H20T | 640×512 + 4K RGB | 50 mK | ~$8,500 | Integrated gimbal; GPS-synced dual stream |
| SEEK Thermal Compact Pro | 320×240 | 70 mK | ~$400 | USB-C; good for ground vehicles |
| InfiRay C200 | 256×192 | 40 mK | ~$300 | Lightweight; UART+USB; suitable for small UAVs |

**Key libraries**

- `pylepton` — FLIR Lepton 3/3.5 Python driver (USB SPI)
- `flirpy` — parse FLIR radiometric TIFF + extract temperature arrays
- `opencv-python` — CLAHE, histogram equalisation, homography alignment to RGB
- `ultralytics` — YOLO-nano fine-tuned on FLIR ADAS thermal dataset

**Public datasets**

| Dataset | Content | Link |
|---|---|---|
| FLIR ADAS Thermal | 14 k RGB+thermal pairs (pedestrian/vehicle/bike) | https://www.flir.com/oem/adas/adas-dataset-form/ |
| KAIST Multispectral Pedestrian | 95 k RGB+LWIR frames, day/night | https://soonminhwang.github.io/rgbt-ped-detection/ |
| DroneVehicle | 56 k drone RGB+IR vehicle detection pairs | https://github.com/VisDrone/DroneVehicle |
| M3ED | Stereo + event + IMU + LiDAR + thermal | https://m3ed.io/ |

**How a human should learn this topic**

1. Install `pylepton`, connect a FLIR Lepton 3.5 via USB, capture 100 frames indoors. Apply CLAHE, identify yourself by heat signature in the scene.
2. Download FLIR ADAS, train `yolo11n` for 20 epochs, verify mAP@0.5 on thermal validation split. Compare against the RGB baseline — night frames should be much better in thermal.
3. Film a dual RGB+thermal sequence of a car parked outside (engine hot vs engine cold). Verify that the `mean_temp_c` in `frame_facts_json["thermal"]` drops over time as the engine cools. This teaches radiometric interpretation.

Sidecar: `data/videos/mission_042.thermal.mp4` — FLIR radiometric video (14-bit TIFF sequence or encoded GREY16).

---

### Step 11. Multispectral / hyperspectral imaging

**Physical principle**

Multispectral cameras capture 4–10 discrete spectral bands; hyperspectral cameras capture continuous spectra across hundreds of narrow bands (typically 400–1000 nm). Reflectance spectra are the fingerprint of surface materials: healthy vegetation reflects strongly in NIR, stressed vegetation shows reduced NIR and elevated red-edge reflectance, water absorbs NIR almost completely.

**SIGINT / signal processing algorithms**

| Algorithm | Purpose |
|---|---|
| Reflectance calibration | Divide scene radiance by calibration panel radiance → unitless reflectance (0–1) |
| Vegetation indices | NDVI = (NIR−R)/(NIR+R); NDRE = (RE−R)/(RE+R); CIR false-colour composite |
| Spectral unmixing | Linear unmixing of endmembers (ENVI/pysptools) to identify pure material fractions per pixel |
| Anomaly detection (RX detector) | Mahalanobis distance from global mean spectrum → flag spectrally anomalous pixels |
| Classification | SVM or CNN on spectral feature vectors → crop type, mineral, material label per pixel |

**Hardware**

| Sensor | Bands | Wavelength | Price | Notes |
|---|---|---|---|---|
| Parrot Sequoia+ | 4+RGB | 530/550/735/790 nm | ~$2,500 | Agriculture multispectral; in-camera NDVI |
| MicaSense RedEdge-MX | 5 | G/R/RE/NIR + RGB | ~$4,900 | GPS-tagged; calibrated reflectance panel included |
| Senop HSC-2 | up to 380 | 400–1000 nm | ~$25,000+ | Pushbroom hyperspectral; drone mount; mineral mapping |
| Sony IMX487 UV | 1 (UV) | 200–400 nm | ~$1,500+ | Detect oil sheens, explosives residue, biological markers |
| Teledyne DALSA | Configurable | NIR/SWIR/MWIR | ~$10,000+ | Industrial / precision agriculture |

**Key libraries**

- `rasterio` — read/write GeoTIFF multi-band rasters
- `pysptools` — hyperspectral unmixing, endmember extraction (ATGP, N-FINDR)
- `spectral` — hyperspectral Python library (SPy); ENVI file format I/O
- `sklearn` — SVM spectral classification
- `numpy` — band arithmetic for NDVI, NDRE, RX anomaly detector

**Public datasets**

| Dataset | Content | Link |
|---|---|---|
| Agriculture-Vision | Aerial multispectral crop disease | https://www.agriculture-vision.com/ |
| Indian Pines / Pavia / Salinas | Hyperspectral remote sensing benchmarks | http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes |
| NASA EO-1 Hyperion | Satellite hyperspectral archive, global | https://earthexplorer.usgs.gov/ |
| DESIS (ISS) | 235-band VNIR hyperspectral, 30 m GSD | https://www.dlr.de/content/en/articles/missions-projects/desis/desis-mission.html |

Sidecar: `data/videos/mission_042.multispectral/` — directory of per-band GeoTIFF files named `band_R.tif`, `band_G.tif`, `band_RE.tif`, `band_NIR.tif`.

---

### Step 12. Event camera (neuromorphic sensing)

**Physical principle**

An event camera (Dynamic Vision Sensor, DVS) does not capture frames. Each pixel independently fires an asynchronous event `(x, y, t, polarity)` whenever the log-luminance change at that pixel exceeds a threshold (~15–20% contrast change). Result: microsecond temporal resolution with no motion blur, very low latency (~1 µs), and high dynamic range (>120 dB vs 60 dB for standard cameras).

**SIGINT / signal processing algorithms**

| Algorithm | Purpose |
|---|---|
| Event-to-frame accumulation | Bin events into fixed time windows (e.g. 10 ms) → polarity histogram image for CNN ingestion |
| Surface of Active Events (SAE) | Per-pixel timestamp of last event → depth-like surface representation |
| Event-based optical flow | Lucas-Kanade or contrast maximisation on event streams → instant dense flow without frame latency |
| Event-based corner detection | eHarris or Arc* detector on event stream → fast feature tracking |
| LSTM/SNN on raw event stream | Spiking neural network for gesture or action recognition directly on event tuples |

**Hardware**

| Sensor | Resolution | Latency | Price | Notes |
|---|---|---|---|---|
| Prophesee EVK4 (IMX636) | 1280×720 | ~1 µs | ~$1,200 | USB3; MetavisionSDK; Linux/Windows |
| iniVation DAVIS346 | 346×260 | ~1 µs | ~$1,500 | Combined event + frame + IMU; USB3 |
| Inivation DVXplorer Lite | 640×480 | ~1 µs | ~$700 | Entry-level; USB3 |
| Samsung DVS Gen4 | 1280×960 | ~10 µs | Research | Highest resolution; integrated in phones |

**Key libraries**

- `metavision_sdk` — Prophesee SDK; event recording, filtering, visualisation (Linux/Windows)
- `tonic` — PyTorch Dataset wrappers for event camera datasets (N-MNIST, DAVIS346, DSEC)
- `dv-processing` — iniVation DV framework; Python/C++ bindings
- `spikingjelly` — SNN training framework for event data in PyTorch

**Public datasets**

| Dataset | Content | Link |
|---|---|---|
| DSEC | Event + LiDAR + stereo + GPS, driving | https://dsec.ifi.uzh.ch/ |
| N-ImageNet | Event-camera classification benchmark (1000 classes) | https://github.com/82magnolia/n_imagenet |
| N-Caltech101 | Event-camera object recognition | https://www.garrickorchard.com/datasets/n-caltech101 |
| MVSEC | Monocular event + frame + IMU, driving + drone | https://daniilidis-group.github.io/mvsec/ |

Sidecar: `data/videos/mission_042.events.raw` — raw event stream in Prophesee RAW format or `mission_042.events.h5` in iniVation DV format.

---

### Step 13. LiDAR / active ranging

**Physical principle**

LiDAR (Light Detection And Ranging) emits short laser pulses (905 nm or 1550 nm) and measures the round-trip time-of-flight (ToF) to compute range. Spinning LiDARs sweep a laser across azimuth angles; solid-state LiDARs use MEMS mirrors or OPA beamsteering. FMCW LiDAR continuously modulates frequency to measure both range and Doppler velocity simultaneously from a single return.

**SIGINT / signal processing algorithms**

| Algorithm | Purpose |
|---|---|
| Point cloud filtering | Statistical outlier removal, radius outlier removal (open3d) to clean raw returns |
| Ground plane removal | RANSAC plane fitting → separate ground from obstacles |
| Euclidean clustering | DBSCAN or voxel-based clustering → individual obstacle candidates |
| ICP registration | Iterative Closest Point → align successive frames → LiDAR odometry |
| PointPillars / PointNet++ | 3D object detection from raw point clouds (BEV voxel pillars) |
| LOAM / LIO-SAM | LiDAR inertial odometry and mapping; tightly-coupled with IMU |

**Hardware**

| Sensor | Lines / points | Range | Price | Notes |
|---|---|---|---|---|
| Garmin LiDAR-Lite v4 | 1D | 0–10 m | ~$130 | Lightweight 1D ToF; I2C; terrain-following |
| Livox Mid-360 | Solid-state, 200k pt/s | 0.1–40 m | ~$500 | 360° FOV; ROS2 driver; good drone mount |
| Ouster OS0-128 | 128 lines, 2.6M pt/s | 0–50 m | ~$4,000 | High-density; integrates with LIO-SAM |
| Velodyne VLP-16 (Puck) | 16 lines | 0–100 m | ~$4,000 | Classic ground-vehicle unit; wide community support |
| Hesai AT128 | 128 lines | 0–200 m | ~$2,500 | Long-range automotive grade |

**Key libraries**

- `open3d` — point cloud I/O (PCD, PLY), filtering, visualisation, ICP
- `pyransac3d` — fast RANSAC plane/sphere/cylinder fitting on point clouds
- `mmdetection3d` — PointPillars, BEVFusion, SECOND — 3D detection from LiDAR
- `ROS2 + ros-humble-sensor-msgs` — PointCloud2 message type; Livox ROS2 driver

**Public datasets**

| Dataset | Content | Link |
|---|---|---|
| KITTI | LiDAR + stereo + GPS + IMU | https://www.cvlibs.net/datasets/kitti/ |
| SemanticKITTI | Annotated 3D point clouds (KITTI) | http://www.semantic-kitti.org/ |
| PandaSet | LiDAR + camera, urban scenes | https://pandaset.org/ |
| nuScenes | 32-beam LiDAR + 6 cameras + radar | https://www.nuscenes.org/ |
| SubT-MRS Challenge | Underground LiDAR + IMU + RGB + thermal | https://superodometry.com/ |

Sidecar: `data/videos/mission_042.lidar.mcap` — MCAP container with `sensor_msgs/PointCloud2` topics, or `mission_042.lidar.pcd` for a single merged scan.

---

### Step 14. Radar (FMCW / Doppler / SAR)

**Physical principle**

Radar (Radio Detection And Ranging) emits modulated microwave signals and receives reflections. FMCW (Frequency Modulated Continuous Wave) linearly chirps the frequency; the beat frequency between transmitted and received signals gives range. Doppler shift gives radial velocity. 4D imaging radar adds elevation angle. SAR (Synthetic Aperture Radar) flies a platform along a path to synthesise a large aperture, achieving sub-meter resolution from a km altitude.

**SIGINT / signal processing algorithms**

| Algorithm | Purpose |
|---|---|
| Range-FFT | Fast Fourier transform on IQ chirp beat → range profile |
| Doppler-FFT (2D FFT) | Second FFT across chirps within a frame → range-Doppler map |
| CFAR detection | Constant False Alarm Rate thresholding → extract target peaks from clutter |
| Angle-FFT / MUSIC / ESPRIT | Direction-of-arrival estimation from multi-element array → azimuth/elevation of target |
| DBSCAN clustering on detections | Cluster point-cloud detections into objects |
| SAR image formation (back-projection or ω-k) | Coherent integration across aperture → 2D high-resolution reflectivity image |

**Hardware**

| Sensor | Type | Range | Price | Notes |
|---|---|---|---|---|
| TI AWR1843 | 77 GHz FMCW | 0–100 m | ~$150 | 3D point cloud; USB; good entry-level |
| Ainstein US-D1 | 24 GHz FMCW | 0.5–50 m | ~$300 | Terrain-following; works through dust/fog/smoke |
| Acconeer XM125 | 60 GHz pulse | 0.03–3 m | ~$30 | Ultra-short range; presence/gesture/vitals |
| Imec 122 GHz radar | D-band FMCW | Research | — | Experimental; sub-cm resolution |
| Capella Space SAR | Spaceborne SAR | Global | Pay-per-image | 0.5 m resolution, all-weather, night-capable |

**Key libraries**

- `OpenRadar` — Python FMCW radar signal processing (range-Doppler, CFAR, MUSIC): https://github.com/PreSenseRadar/OpenRadar
- `mmwave-studio` — TI mmWave SDK companion (Windows DCA1000 capture)
- `pysar` — SAR image formation (back-projection, ω-k algorithm)
- `scipy.signal` — window functions, CFAR, spectral estimation

**Public datasets**

| Dataset | Content | Link |
|---|---|---|
| RADIATE | Radar + LiDAR + stereo + GPS, all weather | https://pro.hw.ac.uk/radiate/ |
| View-of-Delft | 4D radar + LiDAR + stereo | https://github.com/tudelft-iv/view-of-delft-dataset |
| ColoRadar | Radar + LiDAR + IMU, indoor/outdoor | https://arpg.github.io/coloradar/ |
| nuScenes radar | 5-radar point clouds + camera + LiDAR | https://www.nuscenes.org/ |

Sidecar: `data/videos/mission_042.radar.bin` — raw IQ ADC samples in TI DCA1000 format, or `mission_042.radar.csv` for pre-processed range-Doppler detections.

---

### Step 15. GNSS-R and satellite signal reception

**Physical principle**

Standard GNSS gives position. **GNSS-R** (reflectometry) uses reflected GNSS signals (direct path vs. ground reflection) to measure surface properties: soil moisture, sea surface roughness, snow depth, vegetation water content. The delay-Doppler map (DDM) captures the time-delay and Doppler-shift of the reflected signal relative to the direct signal. Separately, satellite *emission* reception (ADS-B at 1090 MHz, AIS at 162 MHz, NOAA APT at 137 MHz, GOES at 1694 MHz) extracts operational intelligence using passive SDR.

**SIGINT / signal processing algorithms**

| Algorithm | Purpose |
|---|---|
| DDM generation | Cross-correlate reflected L1 signal with clean replica → delay-Doppler map (2D function) |
| Reflectivity (Γ) estimation | `Γ = P_reflected / P_direct` after orbit and antenna pattern correction |
| Soil moisture inversion | Empirical polynomial fit from Γ to volumetric water content (VWC) |
| ADS-B Mode-S decoding | Manchester-coded 1090 ES squitter → 24-bit ICAO address, lat/lon, altitude, velocity |
| AIS NRZI/HDLC decoding | GMSK-modulated AIS VDM sentences → vessel MMSI, position, heading, speed |
| APT image decoding | AM synchronous detection, line sync → NOAA weather satellite image (2 visible/IR channels) |

**Hardware** (passive reception — all SDR-based)

| Modality | Frequency | Hardware | Software |
|---|---|---|---|
| GNSS-R direct path | 1.575 GHz L1 RHCP | RHCP patch antenna + RTL-SDR or HackRF | `pyGNSSR`, `gnssr` |
| GNSS-R reflected | 1.575 GHz L1 LHCP | LHCP down-pointing patch + RTL-SDR | Simultaneous dual-receiver setup |
| ADS-B | 1090 MHz | RTL-SDR + 1090 MHz antenna | `dump1090-mutability`, `dump1090-fa` |
| AIS | 161.975 / 162.025 MHz | RTL-SDR + marine antenna | `rtl-ais`, `AIS-catcher` |
| NOAA APT weather | 137.5 / 137.62 MHz | RTL-SDR + V-dipole antenna | `noaa-apt`, `WXtoImg` |
| GOES LRIT/HRIT | 1694.1 MHz | RTL-SDR + LNA + dish | `goestools` |
| Iridium L-band | 1616–1626 MHz | HackRF + patch antenna | `gr-iridium` |

**Key libraries**

- `pyGNSSR` — GNSS-R DDM generation from raw IQ: https://github.com/piyushrpt/PyGNSSR
- `dump1090-mutability` — ADS-B decoder with JSON output: https://github.com/mutability/dump1090
- `rtl-ais` — AIS decoder from RTL-SDR: https://github.com/dgiardini/rtl-ais
- `noaa-apt` — NOAA APT weather satellite image decoder: https://noaa-apt.mbernardi.com.ar/
- `goestools` — GOES-R LRIT/HRIT receiver and decoder: https://github.com/pietern/goestools

**Public datasets**

| Dataset | Content | Link |
|---|---|---|
| CYGNSS L1 v3.1 | NASA GNSS-R DDMs, ocean + land | https://podaac.jpl.nasa.gov/dataset/CYGNSS_L1_V3.1 |
| ESA SMOS Level 2 | Soil moisture via passive microwave | https://earth.esa.int/eogateway/missions/smos |
| TechDemoSat-1 | First spaceborne GNSS-R DDMs | https://www.sstl.co.uk/TDS-1-data |
| OpenSky Network | Historical ADS-B, 10+ years, global | https://opensky-network.org/data/datasets |
| MarineCadastre AIS | US coastal AIS vessel tracks | https://marinecadastre.gov/ais/ |

Sidecar: `data/videos/mission_042.gnssr.bin` — raw GNSS-R IQ capture, or `mission_042.adsb.jsonl` (one aircraft JSON per second from `dump1090`).

---

### Step 16. Inertial and barometric sensing (IMU + barometer + anemometer)

**Physical principle**

- **Accelerometer**: MEMS capacitive proof mass detects specific force (acceleration + gravity). Integrates to velocity, double-integrates to position — but errors accumulate (bias, drift, random walk).
- **Gyroscope**: MEMS vibratory (Coriolis effect) measures angular rate. Integrates to orientation. Allan variance characterises noise at each averaging interval.
- **Barometer**: measures air pressure. Used as altitude estimate: ΔP ≈ −ρg·Δh (−12 hPa / 100 m at sea level). MS5611/BMP390 achieve ±10 cm altitude resolution.
- **Anemometer**: mechanical (cup) or ultrasonic (transit-time difference) wind speed + direction.

**SIGINT / signal processing algorithms**

| Algorithm | Purpose |
|---|---|
| IMU pre-integration | Accumulate Δv, Δθ between keyframes without full numerical integration — used in VINS-Fusion |
| Strapdown navigation equations | Convert body-frame specific force + angular rate → world-frame position, velocity, orientation |
| Allan variance analysis | Characterise ARW, bias instability, RRW of an IMU — determines which EKF noise parameters to set |
| Complementary filter | Simple fused attitude: gyro for fast dynamics, accelerometer for low-freq correction |
| EKF/UKF IMU+GPS fusion | 15-state Kalman filter; GPS provides absolute correction at fix rate; IMU fills between |
| Wind vector decomposition | Ultrasonic: `v_wind = (t_downstream − t_upstream) · c² / (2 · d)` per axis |

**Hardware**

| Sensor | Type | DoF | Price | Notes |
|---|---|---|---|---|
| ICM-42688-P | MEMS IMU | 6 | ~$5 | Used in most drone FCs; SPI; 32 kHz |
| BMI088 | MEMS IMU | 6 | ~$8 | Vibration-robust; used in racing drones |
| VectorNav VN-100 | Navigation IMU | 9 | ~$800 | 0.05° static accuracy; UART/SPI; temperature-compensated |
| ADIS16505-3 | Tactical IMU | 6 | ~$400 | Shock-rated; precision bias stability |
| MS5611 | Barometer | — | ~$8 | ±10 cm altitude resolution; I2C/SPI |
| RM Young 81000 | Ultrasonic anemometer | 3D wind | ~$2,500 | No moving parts; ±0.1 m/s accuracy |

**Key libraries**

- `filterpy` — EKF/UKF/particle filter in Python: https://github.com/rlabbe/filterpy
- `robot_localization` — ROS2 production EKF node with GPS/UTM support
- `imusim` — IMU simulation and error modelling (for Allan variance analysis)
- `pyproj` — geodetic ↔ ENU coordinate transforms (WGS-84 ↔ UTM)

**Public datasets**

| Dataset | Platform | Link |
|---|---|---|
| EuRoC MAV | Drone, stereo + IMU + ground truth | https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.pdf |
| TUM-VI | Handheld, monocular + IMU | https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset |
| ADVIO | Pedestrian, phone IMU + GPS + BLE | https://github.com/AaltoVision/ADVIO |
| SubT-MRS | Underground, IMU + LiDAR + thermal | https://superodometry.com/ |

Sidecar: `data/videos/mission_042.imu.jsonl` — one JSON per IMU sample `{t, ax, ay, az, gx, gy, gz}` at ≥200 Hz; `mission_042.baro.jsonl` — `{t, pressure_hpa, temp_c}` at 1–10 Hz; `mission_042.wind.jsonl` — `{t, speed_ms, dir_deg, gust_ms}` at 1 Hz.

---

### Step 17. Atmospheric / environmental sensing

**Physical principle**

- **Temperature / humidity**: capacitive polymer (SHT4x) or thermistor + hygristor. RH is partial pressure of water vapour / saturation pressure at that temperature — must be cross-compensated (temperature changes saturation pressure).
- **Barometric pressure**: already covered in Step 16; also used for short-range weather nowcasting.
- **Wind**: ultrasonic transit-time (sonic anemometer) or mechanical cup/vane — wind speed and direction as time-varying 3D vector.
- **Solar irradiance**: photodiode pyranometer — incident shortwave radiation (W/m²).

**SIGINT / signal processing algorithms**

| Algorithm | Purpose |
|---|---|
| Temperature-humidity cross-compensation | Apply `RH_corrected = RH_raw / (1 − 0.02 × (T − 25))` for hygristor drift |
| Dew-point calculation | Magnus formula: `Td = 243.04 × (ln(RH/100) + 17.625T/(243.04+T)) / (17.625 − ln(RH/100) − 17.625T/(243.04+T))` |
| Visibility / fog model | `visibility_m = −log(0.05) / (extinction_coeff)` where extinction is fitted from RH above 95% |
| Wind chill / heat index | Compute apparent temperature from temp + wind + humidity → mission safety envelope |
| Pressure altitude | `altitude_m = 44330 × (1 − (P/P0)^0.1903)` — compared against GPS altitude for GPS health check |
| Turbulence intensity (TI) | `TI = std(wind_speed) / mean(wind_speed)` over a 60-second window → flight risk score |

**Hardware**

| Sensor | Measurements | Interface | Price | Notes |
|---|---|---|---|---|
| BME280 | Temperature, humidity, pressure | I2C/SPI | ~$5 | Most popular; ±0.5°C, ±3% RH |
| SHT45 (Sensirion) | Temperature, humidity | I2C | ~$15 | High-accuracy: ±0.1°C, ±1% RH |
| MS5611 | Pressure, temperature | SPI/I2C | ~$8 | Precision barometer; ±10 cm altitude |
| Davis 7911 | Wind speed + direction | Reed switch | ~$100 | Mechanical; simple pulse counting |
| RM Young 81000V | 3D wind vector | RS-422 | ~$2,500 | Ultrasonic; no moving parts; aviation-grade |
| Apogee SP-110 | Solar irradiance | Analog | ~$300 | Calibrated pyranometer; 0–2000 W/m² |

**Key libraries**

- `pynmea2` — parse NMEA sentences (some weather stations output NMEA wind/pressure)
- `metpy` — meteorological calculations (dew point, CAPE, visibility, turbulence indices)
- `siphon` — access NOAA/NWS/ECMWF APIs programmatically
- `pyserial` — serial communication with weather station sensors

**Public datasets**

| Dataset | Content | Link |
|---|---|---|
| ERA5 Reanalysis (ECMWF) | Global atmosphere, 1940–present, hourly | https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels |
| NOAA ISD (ASOS/AWOS) | Global surface station observations | https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database |
| NOAA Radiosonde archive | Upper-air profiles, temp/wind/humidity | https://weather.uwyo.edu/upperair/sounding.html |
| Copernicus Climate Data Store | ERA5, ERA5-Land, CERRA | https://cds.climate.copernicus.eu/ |

Sidecar: `data/videos/mission_042.env.jsonl` — one JSON per second: `{t, temp_c, humidity_pct, pressure_hpa, wind_speed_ms, wind_dir_deg, solar_w_m2}`.

---

### Step 18. Chemical / gas / radiation sensing

**Physical principle**

- **Electrochemical (EC) sensors** (NO2, CO, SO2, H2S): target gas oxidised at a working electrode; current proportional to gas concentration. Cross-sensitive to temperature, humidity, and interfering gases.
- **Non-dispersive infrared (NDIR)** (CO2, CH4): IR beam passes through sample gas; absorption at a specific wavelength correlates with concentration (Beer-Lambert law).
- **Photoionisation detector (PID)** (VOC): UV lamp (10.6 eV) ionises volatile organics; ion current → ppb-level VOC concentration.
- **Optical particle counter (OPC)** (PM1/PM2.5/PM10): laser scatters off particles; forward/side-scatter + Mie theory → particle size and count.
- **Scintillation detector** (gamma, beta): ionising radiation excites a scintillator crystal (NaI, CsI:Tl, BGO) → photon flash counted by PMT or SiPM; energy spectrum → isotope identification.
- **Geiger-Müller (GM) tube** (gamma, beta, X-ray): ionising radiation creates charge avalanche in gas-filled tube → counts per second → dose rate (µSv/h).

**SIGINT / signal processing algorithms**

| Algorithm | Purpose |
|---|---|
| Cross-sensitivity matrix inversion | Correct EC sensor for temperature/humidity/interferer using factory calibration matrix |
| Baseline drift correction | Rolling 24-hour minimum subtraction (WMA) for EC sensors with slow drift |
| Gaussian dispersion inversion | Fit downwind concentration profile to back-project plume source location |
| Gamma spectrum analysis | Peak finding (photopeak identification) in ADC histogram → Cs-137 (661 keV), Co-60, Ra-226 |
| Kriging / Gaussian Process | Spatial interpolation of point measurements to continuous 2D contamination map |
| PMT pile-up correction | Correct dead-time losses at high count rates in scintillation detectors |

**Hardware**

| Sensor | Analyte | Principle | Price | Notes |
|---|---|---|---|---|
| SCD41 (Sensirion) | CO2 | NDIR | ~$40 | Self-calibration; I2C; 400–5000 ppm; ±40 ppm |
| Alphasense OPC-N3 | PM1/2.5/10 | OPC laser | ~$500 | Calibrated number + mass concentration |
| SGP41 (Sensirion) | VOC, NOx index | MOX | ~$15 | Relative index only; good for anomaly detection |
| miniPID 2 (Ion Science) | VOC (total PID) | Photoionisation | ~$400 | 1 ppb resolution; 0–10,000 ppm range |
| Alphasense NO2-A43F | NO2 | Electrochemical | ~$80 | 0.1 ppb resolution; 3-electrode design |
| Radiacode 103 | Gamma (+ beta) | CsI:Tl scintillation | ~$300 | Full spectrum 18 keV–3 MeV; USB/BLE; identifies isotopes |
| Ludlum 44-9 | Beta + gamma | GM tube | ~$800 | 0.02–100 mR/hr; robust field instrument |

**Key libraries**

- `smbus2` — I2C sensor communication (SCD41, SGP41, SHT45)
- `pyserial` — serial communication (OPC-N3, miniPID, Radiacode via USB)
- `scipy.signal` — peak detection for gamma spectrum analysis
- `scikit-gstat` — variogram fitting and kriging for spatial contamination mapping
- `openatmos` — atmospheric dispersion modelling (Gaussian plume, AERMOD)

**Public datasets**

| Dataset | Content | Link |
|---|---|---|
| OpenAQ | Global open air quality (PM2.5, NO2, O3, CO) | https://openaq.org/ |
| EPA AirNow | US real-time + historical AQI | https://www.airnow.gov/ |
| CAMS (Copernicus) | Global atmospheric composition reanalysis | https://atmosphere.copernicus.eu/ |
| Safecast Global Map | Crowdsourced gamma dose rate (post-Fukushima) | https://safecast.org/tilemap/ |
| IAEA IRIX data format | Nuclear emergency data standard + examples | https://www.iaea.org/resources/tools-and-services/nuclear-and-radiological-emergency-response |

**Active learning escalation**: any frame where `dose_rate_usv_h > 1.0` or `voc_ppm > 10.0` is hard-tagged `al_tag = "needs_annotation"` regardless of visual novelty — the spatial memory must flag contaminated zones for mandatory human review.

Sidecar: `data/videos/mission_042.gas.jsonl` — `{t, co2_ppm, voc_ppb, no2_ppb, pm25_ug_m3, pm10_ug_m3, dose_rate_usv_h}` at 1 Hz.

---

### Step 19. Acoustic sensing

**Physical principle**

Microphones transduce pressure oscillations to electrical signals via MEMS (capacitive diaphragm) or electret elements. Ultrasonic transducers operate above 20 kHz for ranging (pulse-echo ToF). Hydrophones use piezoelectric ceramics sensitive to underwater pressure variations. Infrasound sensors (0.001–20 Hz) detect very-long-wavelength pressure waves from distant events (explosions, volcanoes, large machinery).

**SIGINT / signal processing algorithms**

| Algorithm | Purpose |
|---|---|
| Beamforming (delay-and-sum) | Steer a microphone array to a direction by delaying channels by the propagation time difference → spatial filtering |
| MUSIC / ESPRIT | Subspace methods for acoustic direction-of-arrival (DoA) from ULA/UCA arrays |
| TDOA localisation (GCC-PHAT) | Generalised Cross-Correlation with Phase Transform between microphone pairs → time difference of arrival → triangulate source |
| MFCC + CNN classification | Mel-frequency cepstral coefficients feature extraction → CNN/RNN → drone motor noise, gunshot, vehicle engine, wildlife calls |
| Acoustic anomaly detection | Autoencoder reconstruction error on spectrogram → flag unknown sound events |
| Matched filter | Cross-correlate received pulse with known transmitted pulse → ultrasonic range (MaxBotix, LV-EZ series) |
| Hydrophone spectral analysis | FFT of hydrophone signal → identify vessel propeller blade rate (BPF = n_blades × RPM / 60) |

**Hardware**

| Sensor | Type | Frequency | Price | Notes |
|---|---|---|---|---|
| TDK ICS-43434 | MEMS mic | 20 Hz–20 kHz | ~$3 | Omnidirectional; I2S; used in arrays for beamforming |
| Wildlife Acoustics SM4 | Bioacoustic recorder | 20 Hz–96 kHz | ~$900 | 384 kHz option for bats; GPS-tagged; SD card |
| MaxBotix HRLV-EZ4 | Ultrasonic ToF | 42 kHz | ~$30 | 0.03–5 m terrain-following; UART/Analog |
| Aquarian AS-1 | Hydrophone | 10 Hz–100 kHz | ~$200 | Underwater acoustic; detect vessel propellers |
| Infiltec INFRA20 | Infrasound mic | 0.02–50 Hz | ~$150 | Detect distant explosions, volcano eruptions |
| Hosiden HSM3A | MEMS array element | 20 Hz–20 kHz | ~$2 | Small form factor; used in phone-size arrays |

**Key libraries**

- `sounddevice` — real-time audio capture and playback (cross-platform, PortAudio)
- `librosa` — audio feature extraction (MFCC, spectral centroid, chroma, onset detection)
- `pyroomacoustics` — microphone array beamforming (delay-and-sum, MVDR, MUSIC, GCC-PHAT)
- `torchaudio` — audio deep learning: spectrogram transforms, pre-trained models (wav2vec2, HuBERT)
- `bioacoustics` — wildlife sound classification models and datasets

**Public datasets**

| Dataset | Content | Link |
|---|---|---|
| DCASE 2024 Challenge | Acoustic scene + event classification | https://dcase.community/challenge2024/ |
| xeno-canto | 800k+ GPS-tagged bird recordings | https://xeno-canto.org/ |
| FSD50K | 51 k clips, 200 sound classes (FreeSound) | https://zenodo.org/record/4060432 |
| Macaulay Library (Cornell) | Wildlife audio + video, research | https://www.macaulaylibrary.org/ |
| MVSEC | Stereo + event + IMU + audio, driving + drone | https://daniilidis-group.github.io/mvsec/ |
| ESC-50 | 2000 clips, 50 environmental sound classes | https://github.com/karolpiczak/ESC-50 |

Sidecar: `data/videos/mission_042.audio.wav` — 48 kHz WAV; for mic arrays `mission_042.audio_array.h5` (channels × samples, float32).

---

### Step 20. Sensor fusion analysis

**Physical principle**

Sensor fusion combines readings from multiple modalities — each with its own coordinate frame, clock, noise model, and sampling rate — into a unified per-frame state estimate. The theoretical basis is Bayesian estimation: the posterior over vehicle state given all sensor observations is the product of independent likelihood terms (assuming conditional independence given state). The extended / unscented Kalman filter (EKF/UKF) implements this online for Gaussian noise; particle filters handle non-Gaussian distributions.

**SIGINT / signal processing algorithms**

| Algorithm | Modality pair | Purpose |
|---|---|---|
| EKF 15-state navigation | IMU + GPS | Dead-reckoning pose between GPS fixes; maintains 3D position, velocity, orientation, biases |
| Cross-modal IoU merge | RGB + Thermal | Match detections from two optical paths; disagreement → novelty |
| Frustum PointNet / BEVFusion | RGB + LiDAR | Project camera frustum onto LiDAR points; fused 3D detection |
| GNSS-R soil moisture | GNSS-R + GPS | Georeference reflectivity measurements to GPS waypoints → soil moisture map |
| Interference heat map | RF + GPS | Grid `snr_db` by position → RF shadow / jamming zone map |
| Gaussian dispersion inversion | Gas + GPS + wind | Back-project plume source from downwind concentration profile |
| Weather confidence factor | Weather + all modalities | Scale detection confidence down under fog, icing, or high-wind conditions |
| Temporal alignment | All | Nearest-neighbour interpolation of sensor readings to video frame timestamps |

**Fusion architectures**

| Architecture | When to use | Key tool |
|---|---|---|
| EKF / UKF | Pose and navigation, Gaussian noise | `filterpy` |
| Late fusion (score averaging) | Independent models per sensor, simple merge | per-model confidence scores |
| Cross-attention transformer | Rich multi-modal feature fusion (RGB+depth+thermal) | `timm` + custom cross-attention heads |
| BEVFusion | LiDAR + camera 3D detection | `mmdetection3d` |
| Deep state-space model | Irregular multi-rate time series (weather + gas + RF) | Mamba / GRU-D |

**Spatial calibration toolchain**

| Pair | Tool |
|---|---|
| Camera ↔ IMU | [kalibr](https://github.com/ethz-asl/kalibr) — continuous-time calibration |
| Camera ↔ LiDAR | [lidar_camera_calibration](https://github.com/ankitdhall/lidar_camera_calibration) |
| Camera ↔ Thermal | Manual homography — co-located calibration target |
| Camera ↔ Radar | [OpenRadar](https://github.com/PreSenseRadar/OpenRadar) — retroreflector markers |

**`frame_facts_json["sensor_fusion"]` output schema**

```json
{
  "modalities_present": ["rgb", "thermal", "gps", "imu", "rf", "weather"],
  "modalities_missing": ["lidar", "gas"],
  "fusion_confidence": 0.87,
  "weather_factor": 0.92,
  "cross_modal_detections": [
    {
      "label": "person",
      "rgb_conf": 0.78,
      "thermal_conf": 0.91,
      "fused_conf": 0.94,
      "depth_m": 12.4,
      "lidar_range_m": null,
      "cross_modal_agreement": true
    }
  ],
  "degradation_flags": ["high_humidity", "wind_blur"],
  "rf_shadow": false,
  "plume_proximity_m": null,
  "pose_source": "ekf_imu_gps",
  "pose_covariance_trace": 0.003
}
```

Frames where any detection has `cross_modal_agreement = false` are escalated to `al_tag = "novel"`. Frames with `plume_proximity_m < 50` or `dose_rate_usv_h > 1.0` are hard-tagged `al_tag = "needs_annotation"`.

**Key libraries**

| Task | Library |
|---|---|
| EKF / UKF / particle filter | https://github.com/rlabbe/filterpy |
| Production ROS2 EKF | https://github.com/cra-ros-pkg/robot_localization |
| Camera-IMU calibration | https://github.com/ethz-asl/kalibr |
| LiDAR-camera fusion | https://github.com/open-mmlab/mmdetection3d |
| Point cloud processing | https://www.open3d.org/ |
| Geometric camera transforms | https://kornia.readthedocs.io/ |
| Radar signal processing | https://github.com/PreSenseRadar/OpenRadar |
| Irregular time series | https://github.com/state-spaces/mamba |

**Essential reading**

- Kalman and Bayesian Filters in Python (free): https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
- BEVFusion — unified LiDAR-camera 3D perception: https://arxiv.org/abs/2205.13542
- TokenFusion — transformer multi-modal fusion: https://arxiv.org/abs/2204.08721
- VINS-Mono — visual-inertial odometry: https://arxiv.org/abs/1708.03852
- LIO-SAM — LiDAR-inertial odometry: https://arxiv.org/abs/2007.00258

**Public multi-modal fusion datasets**

| Dataset | Modalities | Link |
|---|---|---|
| nuScenes | 6× RGB + LiDAR + radar + GPS + IMU | https://www.nuscenes.org/ |
| RADIATE | Radar + LiDAR + stereo + GPS, all weather | https://pro.hw.ac.uk/radiate/ |
| M3ED | Event + stereo + IMU + LiDAR + thermal | https://m3ed.io/ |
| KITTI-360 | 360° stereo + LiDAR + GPS + IMU | https://www.cvlibs.net/datasets/kitti-360/ |
| SubT-MRS Challenge | RGB + thermal + LiDAR + IMU + gas | https://superodometry.com/ |

**How a human should learn this topic**

1. **Temporal alignment first**: take any two sensor logs from the same mission and plot them together against wall-clock time. GPS and barometer both measure altitude — their curves should match after alignment. Any offset reveals a clock skew bug.
2. **EKF on EuRoC MAV**: implement a 15-state IMU+GPS EKF in `filterpy` using the EuRoC dataset (ground truth available). Compare dead-reckoning (IMU-only) vs EKF-fused vs ground truth. The gap between IMU-only and EKF is the motivation for fusion.
3. **Project LiDAR into camera**: open a KITTI sequence in `open3d`, apply the calibration matrix, visualise the point cloud coloured by the camera image. Verify road is grey/black, lane markings are white. This is the sanity check for extrinsic calibration.
4. **Build a two-stream RGB+thermal detector**: train separate YOLO-nano models on FLIR ADAS RGB and thermal splits, then implement late fusion by weighting logits by validation mAP. Verify night-time mAP is higher with fusion than RGB alone.
5. **Run BEVFusion on nuScenes**: use `mmdetection3d` BEVFusion config, run inference on 3 scenes, inspect the bird's-eye-view feature map. Camera contributes texture; LiDAR contributes geometry; fused BEV is richer than either alone.

---

### Step 21. YOLO11 + SAM2/3 detection and segmentation

**What the local pipeline does**

The pipeline runs YOLO11 on each sampled frame to detect object instances, assigns every detection a priority label (human → vehicle → artificial → other), and optionally refines each detection with a SAM2/3 segmentation mask. Those detections are then reused by the 3D map stage to build a lightweight semantic scene graph (YOLO SSG). The step writes:

- `yolo_sam/frame_{t:.3f}_annotated.jpg` — color-coded bounding boxes + SAM mask overlays per frame
- `yolo_sam_results.json` — full per-frame detection JSON with label, confidence, normalised bbox, priority, and mask area fraction
- `detection_comparison.md` — side-by-side table comparing YOLO11 vs the HF detector (step 8) by object count, priority bucket, and per-frame speed
- `3d_map/semantic_environment_graph.json` — graph nodes and edges once the map step has anchor positions
- `3d_map/semantic_environment_graph.md` — compact human-readable SSG summary

**CLI flags**

```bash
python main.py --mode local                               # YOLO + SSG on by default
python main.py --mode local --no-sam                      # YOLO detection only
python main.py --mode local --yolo-model yolo11m          # larger model
python main.py --mode local --sam-model sam2              # force SAM2
python main.py --mode local --no-yolo                     # disable YOLO + SSG entirely
```

**Model used**

| Component | Default (`auto`) | Notes |
|-----------|-----------------|-------|
| YOLO11 detector | `yolo11n.pt` (~6 MB) | Downloaded automatically by ultralytics on first run |
| Segmentation | SAM3 → SAM2 → SAM1 fallback | `sam3` and `sam2` are included in the default requirements; SAM3 is preferred automatically |

YOLO tiers (set with `--yolo-model` or `YOLO_MODEL=`):

| Model | Size | COCO mAP50-95 | Best for |
|-------|------|--------------|---------|
| `yolo11n` | 6 MB | 39.5 | edge / fast local runs |
| `yolo11s` | 18 MB | 47.0 | balanced |
| `yolo11m` | 38 MB | 51.5 | higher accuracy |
| `yolo11l` | 48 MB | 53.4 | server |
| `yolo11x` | 109 MB | 54.7 | max quality |

**Priority taxonomy**

Detections are sorted and color-coded by safety-relevant priority:

| Priority | Color | Trigger labels |
|----------|-------|----------------|
| 1 — **Human** | 🔴 Red | `person`, any label containing "pedestrian" / "rider" |
| 2 — **Vehicle** | 🔵 Blue | `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `boat`, `train`, `airplane`, … |
| 3 — **Artificial** | 🟢 Green | `traffic light`, `stop sign`, `pole`, `fence`, `building`, `barrier`, … |
| 4 — **Other** | ⚫ Grey | natural objects, uncategorized |

The priority ordering exists because outdoor autonomous systems must react to humans first, then dynamic vehicles, then static infrastructure.  All downstream steps (VideoKnowledge, Qwen prompt context) see detections in this sorted order.

**Why it matters**

Step 8 (HF detector) uses a transformer-based model suited to open-vocabulary queries. YOLO11 trades vocabulary flexibility for raw inference speed (YOLO11n runs at ~100 FPS on a T4) and a model architecture with explicit instance segmentation via SAM.  Together they give the pipeline:

1. A fast coverage pass at all frames (YOLO)
2. Pixel-level object masks for spatial reasoning (SAM)
3. A detector comparison artifact that exposes where the two models agree or diverge

**Essential reading**

- YOLOv8/YOLO11 architecture overview: https://docs.ultralytics.com/models/yolo11/
- Ultralytics paper: https://arxiv.org/abs/2304.00501
- Segment Anything (SAM): https://arxiv.org/abs/2304.02643
- SAM 2 — real-time video segmentation: https://arxiv.org/abs/2408.00714
- SAM 3 (repository): https://github.com/facebookresearch/sam3

**How a human should learn this topic**

Start with the anchor detectors:

1. Understand how YOLO frames detection as a single-pass regression (grid cells, anchor boxes, class probability × objectness). Run `yolo11n predict` on a test image and inspect the raw output tensors.
2. Learn the transformer-DETR family (step 8) to appreciate the accuracy–speed tradeoff: DETR uses attention to reason about the whole image; YOLO uses local regression and non-maximum suppression.
3. Study Segment Anything: prompting a foundation model with a bounding box to get a pixel mask. Compare the result to a threshold on the depth map (step 7) — they capture different aspects of "where is the object".
4. Run the local pipeline with YOLO and detection enabled together and look at `detection_comparison.md`. Note which categories each detector misses and why.
5. Build a simple scene graph from detections: merge recurring `truck` observations across nearby frames, add `near` edges to co-visible `person` and `truck` nodes, and inspect where this approximation is helpful versus geometrically wrong.
6. Design your own priority function: consider what `priority=1` should mean for indoor versus outdoor versus underwater scenes.

---

### Step 22. Gemma 4 directed tracking *(new)*

**What the local pipeline does**

This step uses a Gemma sidecar to inspect up to 12 sampled frames and return structured JSON:
`scene_type`, `dominant_objects` with rough fractional boxes, `areas_of_interest`, and a
priority-ordered `tracking_priority` list. That scene summary then drives two downstream passes:

1. **SAM-directed segmentation**: when Gemma gives a specific `rough_bbox`, SAM uses it as a
   direct box prompt. When Gemma falls back to an almost-whole-frame box, the code switches to
   automatic mask generation and keeps only masks whose CLIP embedding matches one of Gemma's
   named object categories.
2. **RF-DETR tracking**: RF-DETR runs on up to 90 frames and filters detections by
   `tracking_priority` when Gemma provided one. Persistent track IDs are then assigned by greedy
   IoU matching across adjacent frames.

The step writes:

- `gemma_tracking_results.json` — scene summary plus per-frame RF-DETR detections and SAM metadata
- `gemma_tracking_summary.md` — human-readable explanation of Gemma's scene interpretation and tracking totals
- `gemma_tracking/frame_{t:.3f}_tracked.jpg` — annotated tracking frames

**Why it matters**

This is the first place where the pipeline uses a language model to *steer* classic perception
rather than just describe its outputs. Gemma narrows the search space to likely categories and
regions; SAM and RF-DETR then do the spatial work. That makes the stage cheaper than open-ended
tracking and more interpretable than running a detector across every class all the time.

It also exposes a practical integration issue: label vocabularies matter. If Gemma emits
`pickup truck` and RF-DETR emits `truck`, the substring-based label filter still works. If Gemma
chooses a label with no overlap with the detector vocabulary, tracking can come back empty even
when the object is visible.

**How a human should learn this topic**

Run the local pipeline once with `--gemma-api-url` set and inspect the three P3 artifacts together:

1. Read `gemma_tracking_summary.md` first. Verify that `scene_type`, `tracking_priority`, and
   dominant objects match the video at a high level.
2. Open `gemma_tracking_results.json` and compare `dominant_objects[*].rough_bbox` with
   `frames[*].sam_masks[*].source` to see whether the run used direct Gemma box prompts
   (`gemma_bbox`) or the CLIP-filtered fallback (`clip_filtered_automask`).
3. Inspect a few `gemma_tracking/frame_*_tracked.jpg` frames and confirm that the same `track_id`
   persists across adjacent frames for the same object.
4. If tracking is unexpectedly empty, check Gemma's `tracking_priority` labels before blaming
   RF-DETR. Vocabulary mismatch is a more common failure mode than total detector failure.

One implementation detail to keep in mind while debugging: the saved JPEGs currently render
tracking boxes and IDs only. SAM outputs are stored as metadata in JSON and summarized in
markdown, but are not re-painted onto the tracking frames.

**Essential reading**

- RF-DETR repository and docs: https://github.com/roboflow/rf-detr
- Segment Anything (SAM): https://arxiv.org/abs/2304.02643
- CLIP: https://arxiv.org/abs/2103.00020

---

### Step 23. World model video embeddings

**What the local pipeline does**

The pipeline groups consecutive frames into clips and computes one temporal embedding per clip. This is the video-native representation step, as opposed to the frame-by-frame image encoders used earlier.

**Model used**

- Wrapper: [pipeline/world_model.py](../pipeline/world_model.py)
- Registry `auto` may nominate very large models like Cosmos or V-JEPA-family checkpoints
- Important repo behavior: runtime-incompatible models are skipped and the wrapper falls back to `MCG-NJU/videomae-base`
- Current practical runtime model: `MCG-NJU/videomae-base`

**Why it matters**

This is where the system starts to reason over temporal context instead of isolated frames. Even if the current output is only an embedding, the step introduces the right abstraction for future dynamics and prediction models.

**Essential reading**

- VideoMAE: https://arxiv.org/abs/2203.12602
- V-JEPA 2, for where this part of the stack is heading conceptually: https://arxiv.org/abs/2506.09985

**How a human should learn this topic**

First learn why image encoders are not enough for temporal understanding. Then study clip sampling, masked video modeling, and the difference between recognition, anticipation, and world modeling. A useful exercise is to compare embeddings from single frames versus 8- or 16-frame clips and ask what motion information is preserved.

---

### Step 24. Qwen detailed captioning

**What the local pipeline does**

The local pipeline sends each frame, plus all accumulated context from previous steps, to a vision-language model for a richer structured description than the Florence caption. By Step 12 every earlier analysis is complete, making this the most information-dense step in the pipeline.

**Agentic enrichment (new)**

Qwen receives six types of prior knowledge per frame, assembled by `VideoKnowledge.context_for_frame(t_sec)`:

| Context block | Source step | Example injected text |
|---|---|---|
| `[Prior scene description]` | Florence caption (Step 4) | `convoy of trucks moving through dust` |
| `[Scene segment N, Xs–Ys]` | Segment analysis on Florence output | `Scene segment 2, 4.0s–12.5s: vehicles on gravel road` |
| `[Audio context]` | ASR subtitles (Step 5) | `convoy moving north, checkpoint ahead` |
| `[Visible text]` | OCR (Step 6) | `B-12  EXIT ONLY` |
| `[Depth profile]` | Depth estimation (Step 7) | `near_ratio=0.18  mean=22.40` |
| `[Detected objects]` | Object detection (Step 8) | `truck, person, barrier` |
| `[Prior frame state]` | Previous Qwen output | `vehicles=2×truck  road=gravel  condition=clear` |

In addition, `domain_hint` from Step 3 (Gemma) is prepended to the Qwen system prompt:

```
[Scene domain: Dominant scene: military convoy | Known objects: truck, soldier]
You are a precise outdoor-scene analyst …
```

Each Qwen result is fed back into `VideoKnowledge` via `update_qwen_state()` so that the *next* frame's `[Prior frame state]` block reflects the just-extracted vehicle and road data.  This gives Qwen a rolling memory of what it has seen so far without reprocessing previous frames.

**Model used**

- Wrapper: [pipeline/vision/qwen.py](../pipeline/vision/qwen.py)
- Typical local sidecar in this repo: `qwen2.5vl:7b` via Ollama
- Alternative: OpenAI-compatible vLLM endpoint

**Why it matters**

By Step 12 the pipeline has already extracted language, geometry, sound, and object structure independently.  Rather than treating those as parallel outputs, Qwen can now reason *across* them: "the depth profile shows an obstacle approaching (Step 7), detection found a barrier (Step 8), and ASR says 'checkpoint ahead' (Step 5) — is the scene consistent?"  This cross-modal reasoning was not possible when each step ran in isolation.

**Essential reading**

- Qwen2.5-VL technical report: https://arxiv.org/abs/2502.13923

**How a human should learn this topic**

Study multimodal prompting, context packing, and grounding failures. Then learn how OCR and ASR context can improve visual interpretation but can also bias it. A good exercise is to run the same frame with and without the full `context_for_frame()` block injected and compare how the structured output (vehicle groups, road surface, scene summary) changes.

---

### Step 25. UniDriveVLA expert analysis

**What the local pipeline does**

The local pipeline sends a sparse set of frames to a UniDriveVLA bridge and
normalizes the output into four blocks:

- `understanding`
- `perception`
- `planning`
- `mixture_of_experts`

This step is deliberately bridge-based rather than in-process. UniDriveVLA is a
full autonomous-driving VLA stack, not a lightweight captioning model, so
selfsuvis integrates it through an OpenAI-compatible sidecar contract.

**Models used**

- External bridge target: `owl10/UniDriveVLA_Nusc_Base_Stage3` by default
- Runtime wrapper: [pipeline/vision/unidrive.py](../pipeline/vision/unidrive.py)
- Local artifact: `unidrive_analysis.md`

**Why it matters**

This is the first step in the local pipeline that explicitly separates:

- semantic understanding
- scene perception
- action/planning guidance
- expert-consensus synthesis

That separation is useful even outside pure driving tasks because it exposes
where a model is describing, where it is localizing, and where it is making an
action recommendation.

**Essential reading**

- UniDriveVLA repository: https://github.com/xiaomi-research/unidrivevla

**How a human should learn this topic**

Study the difference between a generic VLM and a domain VLA. A VLM mostly tells
you what is visible. A VLA tries to turn that into perception plus action
structure. In practice, compare `detailed_captions.md` and `unidrive_analysis.md`
for the same timestamps and ask:

1. which facts overlap?
2. where does UniDrive add planning-specific language?
3. where does the mixture-of-experts summary preserve disagreement instead of collapsing it?

---

### Step 26. Base model search test

**What the local pipeline does**

The system uses the middle frame as a query and retrieves visually similar neighbours from the frame store using the base representation.

**Models used**

- Primary search space: DINO embeddings
- Fallback: CLIP embeddings
- Store backend: Qdrant or in-memory cosine index

**Why it matters**

This is the first quantitative sanity check on the representation. If retrieval is weak here, later adaptation steps have a clear target.

**Essential reading**

- DINOv2: https://arxiv.org/abs/2304.07193
- CLIP: https://arxiv.org/abs/2103.00020

**How a human should learn this topic**

Learn embedding evaluation through nearest-neighbour retrieval, not just classification accuracy. Build intuition for failure cases such as background bias, viewpoint changes, and repeated textures. The practical question is simple: does the model retrieve semantically similar frames for the right reason.

---

### Step 28. SSL DINO fine-tuning

**What the local pipeline does**

The local pipeline fine-tunes the DINOv3 ViT-B/14 backbone on this mission's extracted frames using
NT-Xent contrastive loss.  Two strategies are chosen automatically:

- **Temporal pairs** (default when frames ≥ 2 × batch\_size): frame[i] and frame[i+k] are
  positive pairs.  Exploits the fact that consecutive outdoor frames show the same scene.
- **Augmentation pairs** (fallback for short clips): two independently augmented views of the
  same frame are the positive pair.

The top 2 transformer blocks and the projection head are trained; the first 10 blocks are frozen
to prevent catastrophic forgetting.  Produces `finetune_stats.md` with loss curve, convergence
analysis, and per-epoch interpretation.

**Models used**

- Backbone: `dinov3_vitb14` (DINOv2 ViT-B/14 register tokens from `facebookresearch/dinov2`)
- Loss: NT-Xent (InfoNCE, SimCLR formulation) — `pipeline/training/ssl.py`
- Optimiser: AdamW + cosine annealing LR schedule

**Why it matters**

Pre-trained DINOv3 was trained on internet images.  Mission video from drones or rovers contains
specific terrain, vehicle classes, and lighting conditions not well-represented online.  SSL on
mission frames teaches the backbone that consecutive frames of *this specific scene* should have
similar representations, tightening the embedding manifold for this domain without any labels.

**Gemma 4 SSL finding** (from `docs/design/gemma4-video-analysis-ssl-distillation.md`):

The same SSL approach applies to Gemma 4's SigLIP vision encoder.  Fine-tuning only the top 4
of 24 ViT-L blocks (~60 M trainable params) with `TemporalPairDataset` is feasible on 16 GB
VRAM in ~8 min per mission.  The LM backbone is fully frozen — the SSL loop never touches the
2 B language model.  This is Strategy A (vision encoder only SSL) in the design document.

**Essential reading**

- DINOv2: https://arxiv.org/abs/2304.07193
- SimCLR (NT-Xent loss): https://arxiv.org/abs/2002.05709
- SigLIP (Gemma 4 vision encoder): https://arxiv.org/abs/2303.15343

**How a human should learn this topic**

Study self-supervised contrastive learning, positive/negative pair construction, and representation
collapse (all-zero embeddings).  Then focus on temporal positives for video: which frame pairs
should be positive in fast motion, cuts, zooms, and camera shake?  A good exercise is to visualise
the embedding space with UMAP before and after SSL and observe whether same-scene frames cluster
more tightly.

---

### Step 29. Knowledge distillation — maximum hydration chain

**What the local pipeline does**

The fine-tuned teacher is distilled into a smaller ViT-S/14 student using **Relational Knowledge
Distillation with Distance + Angle losses (RKD-DA)** plus two regularisers:

| Loss | Purpose |
|---|---|
| `L_RKD_dist` (λ=25) | Preserve pairwise distance structure: if A and B are far apart for the teacher, keep them far for the student |
| `L_RKD_angle` (λ=50) | Preserve triplet angle structure: neighbourhood geometry, not just distances |
| `L_cosine` (λ=1) | Direct cosine anchor: student projection ≈ teacher embedding |
| `L_KoLeo` (λ=0.1) | Spread regulariser: prevent student embeddings from collapsing to a single cluster |
| `L_caption` (λ=0.5, **new**) | Caption anchor: pull student toward CLIP text embeddings of Florence/Gemma captions — transfers language semantics into the small model |

**Maximum-hydration distillation** (automatic when both Gemma and captions are available):

1. **Teacher upgrade**: when `MODEL_NAME=gemma`, the pipeline uses `GemmaVisionTeacher` as the
   distillation teacher instead of DINOv3.  Gemma’s SigLIP-grounded embeddings carry richer
   semantic structure than pure-vision DINOv3 — the student inherits language-grounded visual
   similarity.
2. **Caption anchor loss**: Florence-2 captions from step 4 are re-embedded with the CLIP text
   encoder and used as anchor targets for the student.  This teaches the student that a frame’s
   embedding should be close to the text description of what it contains — zero-shot generalisation
   for free.
3. Both additions are backward-compatible: they activate only when the relevant inputs are
   available and default to off (λ=0) otherwise.

**Models used**

- Teacher: fine-tuned DINOv3 ViT-B/14 (or `GemmaVisionTeacher` when `MODEL_NAME=gemma`)
- Student: `dinov2_vits14` (22 M params, 384-dim, ~4× compression from ViT-B/14)
- Implementation: [pipeline/distill.py](../pipeline/distill.py)
- Caption anchor: CLIP text encoder from `models/openclip_model.py`

**Why it matters**

RKD-DA preserves pairwise neighbourhood topology directly, which optimises Recall@K for
retrieval — the metric that matters for this pipeline.  The caption anchor layer adds a second
signal: the student learns not just "these frames are visually similar to each other" but also
"this frame looks like what this sentence describes", which generalises to unseen text queries.

**Gemma teacher chain finding** (from `docs/design/gemma4-video-analysis-ssl-distillation.md`):

With a two-stage chain — Gemma 4 vision encoder (300 M, dim=1152) → ViT-S/14 (22 M, dim=384)
— estimated Recall@1 ≈ 0.92, versus 0.78 for a directly pretrained ViT-S/14 without
distillation.  Adding Stage 2 (ViT-S/14 → EfficientViT-S1, 6.6 M params) gives R@1 ≈ 0.85
with an 8 ms/frame CPU latency — suitable for edge/robot deployment.

Total compression from Gemma 4 teacher to final edge model: ~60×.

**Essential reading**

- Distilling the Knowledge in a Neural Network: https://arxiv.org/abs/1503.02531
- Relational Knowledge Distillation (RKD-DA): https://arxiv.org/abs/1904.05068
- KoLeo regulariser: https://arxiv.org/abs/2305.12320

**How a human should learn this topic**

Learn the difference between task distillation (match logits), feature distillation (match
intermediate activations), and relational distillation (match pairwise structure).  Then study
what information a student must preserve for retrieval versus classification: retrieval needs
Recall@K, not accuracy.  The practical exercise is to compare retrieval quality (Recall@1, @5)
before and after distillation and observe how each loss term contributes.

---

### Step 30. ONNX export + gallery build

**What the local pipeline does**

The adapted backbone is exported to ONNX and then used to build an embedding gallery for lightweight edge inference.

**Models used**

- Export target: fine-tuned teacher by default, or distilled student if that path succeeds
- Runtime artifact: `edge_models/dino_local.onnx`
- Gallery artifact: `edge_models/gallery.npz`

**Why it matters**

This step turns a research model into a deployable artifact. It is the operational packaging stage.

**Essential reading**

- ONNX documentation: https://onnx.ai/

**How a human should learn this topic**

Learn model export, operator compatibility, dynamic versus static shapes, and runtime providers. The key practical skill is not “convert a model once,” but “trace what broke after export and prove the exported model still produces a compatible embedding space.”

---

### Step 31. Fine-tuned search test

**What the local pipeline does**

The local pipeline reruns the same retrieval test from Step 14, now with the adapted model.

**Models used**

- Fine-tuned DINO backbone or exported ONNX runtime equivalent

**Why it matters**

This is the direct “did adaptation help?” check. It closes the loop between training and task utility.

**Essential reading**

- DINOv2: https://arxiv.org/abs/2304.07193

**How a human should learn this topic**

Learn to evaluate representation changes with controlled before/after comparisons. Review top-k retrieval changes, not just the top hit. The question to ask is whether the model became more domain-sensitive without becoming overly narrow.

---

### Step 32. Model comparison + video description

**What the local pipeline does**

The pipeline compares baseline and adapted retrieval results and derives a
coarse CLIP-based video-level description from the prompt bank in the local
runner.

This is still the “single-model family” comparison stage: base vs fine-tuned
retrieval plus CLIP text-description scoring.

**What the local pipeline does**

The pipeline ranks a curated set of text prompts against the average CLIP embedding of the video and writes out a short video-level description.

**Models used**

- OpenCLIP text and image encoders
- Prompt bank defined in the local runner

**Why it matters**

This step is a simple but strong cross-modal summary: it asks which textual hypotheses best fit the visual evidence across the whole clip.

**Essential reading**

- CLIP: https://arxiv.org/abs/2103.00020

**How a human should learn this topic**

Study prompt-set design, prompt leakage, and dataset bias in text-image similarity. Try rewriting the prompt bank for a narrower domain such as traffic, ISR, agriculture, or industrial inspection and see how the top descriptions change.

---

### Step 33. Multi-model comparison

**What the local pipeline does**

When both Qwen and UniDrive are enabled, the pipeline writes
`multi_model_comparison.md`. This compares:

- Gemma’s video-level scene classification output
- Qwen’s structured per-frame scene summaries
- UniDrive’s understanding + Mixture-of-Experts consensus output

The comparison includes:

- matched timestamps between Qwen and UniDrive
- token-overlap agreement between summaries
- UniDrive risk distribution
- UniDrive expert-agreement distribution
- concrete matched examples for inspection

**Why it matters**

This is the first explicit *cross-model* evaluation step in the local pipeline.
It does not ask only “did training improve retrieval?” It asks “do our major
multimodal analyzers agree on what is happening, and where do they diverge?”

**Essential reading**

- CLIP: https://arxiv.org/abs/2103.00020
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- UniDriveVLA repository: https://github.com/xiaomi-research/unidrivevla

**How a human should learn this topic**

Read `comparison.md` and `multi_model_comparison.md` together. The first tells
you whether the representation improved. The second tells you whether the major
reasoning models agree about the scene. Those are different questions and both matter.

---

### Step 27. 3D map + Gaussian Splat

**What the local pipeline does**

The pipeline first tries classical Structure-from-Motion with pycolmap to recover poses and sparse geometry. If SfM fails or is partial, it falls back to a PCA point cloud. It then reuses the frame anchors from that map to attach YOLO detections into a semantic scene graph, and optionally builds a Gaussian Splat representation for interactive viewing.

**Models and tools used**

- SfM toolchain: `pycolmap` / COLMAP-style incremental mapping
- Sparse-map fallback: PCA over frame embeddings
- Semantic scene graph builder: YOLO SSG over ENU/SfM/PCA frame anchors
- Gaussian Splat builder: repo `gsplat` integration
- Output directory: `3d_map/`

**Why it matters**

This step converts representation learning into explicit scene structure. It is the bridge from semantic understanding to geometry and rendering, and it is where the pipeline upgrades 2D detections into a reusable 3D semantic environment graph.

**Essential reading**

- Structure-from-Motion Revisited (COLMAP): https://openaccess.thecvf.com/content_cvpr_2016/html/Schoenberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html
- 3D Gaussian Splatting for Real-Time Radiance Field Rendering: https://arxiv.org/abs/2308.04079

**How a human should learn this topic**

Learn camera geometry, epipolar constraints, feature matching, bundle adjustment, and failure modes like low parallax or repeated texture. Then study why Gaussian splatting is useful after pose recovery. After that, inspect how the YOLO SSG attaches detections to those anchors: it is an observation graph, not perfect object localization. The key conceptual shift is from “which object is in the frame” to “where is the camera in a consistent 3D world, and where are semantic observations concentrated inside that world.”

---

### Step 34. Video synthesis

**What the local pipeline does**

The pipeline synthesizes a final ontology-style summary and a narrative report from all earlier artifacts. This is a report-writing step driven by multimodal context rather than a single raw model output.

**Models used**

- OpenAI-compatible chat endpoint
- In practice this is usually the configured Qwen sidecar
- Input context includes captions, OCR, ASR, detections, descriptions, map outputs, and other accumulated local-run artifacts

**Why it matters**

This step turns the pipeline from a collection of independent analyses into a human-readable final product.

**Essential reading**

- Qwen2.5-VL technical report: https://arxiv.org/abs/2502.13923

**How a human should learn this topic**

Study ontology design, schema-first prompting, and evidence-backed summarization. The main skill here is not generic prompt writing; it is learning how to convert noisy multimodal evidence into a consistent structured report without hallucinating unsupported conclusions.

---

### Step 35. Agentic flow audit

**What the local pipeline does**

The final step generates `agentic_flow.md`, a reasoning-heavy audit artifact that explains how context moved from one step to the next, what each step added, and where misidentification or stale/wrong context could propagate.

Unlike Step 22, which writes a user-facing summary of the video, Step 23 is a system-facing audit of the pipeline itself.

**Models used**

- OpenAI-compatible reasoning endpoint
- Practical default on a 16 GB GPU + large RAM machine: `deepseek-r1:14b`
- Larger reasoning models can be pinned explicitly with `--reasoning-model`
- The prompt path uses a compact-first strategy and a longer timeout budget because reasoning models can be substantially slower than captioning models

**Why it matters**

This step makes the local pipeline inspectable. It turns a long multimodal run into a traceable reasoning document, which is critical when the pipeline is used for robotics, surveillance, or operational reporting.

**Essential reading**

- DeepSeek-R1 report: https://arxiv.org/abs/2501.12948
- Practical prompt engineering for reasoning models: study model cards and deployment docs for your actual Ollama-served model

**How a human should learn this topic**

Learn to distinguish three different outputs:

- task inference: "what is in the frame?"
- report synthesis: "what happened in the video?"
- system audit: "why did the pipeline conclude this, and where could it be wrong?"

The practical exercise is to compare `video_synthesis.md` and `agentic_flow.md` from the same run. The first should read like a mission summary; the second should read like a reasoning provenance document.

---

## Agentic Knowledge Flow

### What problem it solves

Steps 3–9 each run a specialised model independently and write results to disk. Without an
accumulator, Step 12 (Qwen) knows only what it can see in the raw image, a subtitle string, and
an OCR string. It cannot ask "was a barrier detected 2 seconds ago?" or "does this scene belong
to the same segment as the last 8 frames?"

The `VideoKnowledge` accumulator (in `pipeline/workflows/local/runner.py`) solves this: each step
*deposits* structured results into a shared object and later steps *query* it per frame.  The
accumulator is never serialised to disk; it lives for the lifetime of one video pass.

### Data flow diagram

```
Step 3  Gemma analysis ──────────► VideoKnowledge.add_gemma()
                                        │  scene_type, n_transitions, n_clusters, mnn_dino
                                        ▼
Step 4  Florence captioning ◄─── domain_hint()           VideoKnowledge.add_captions()
                                        │  _captions, _segments (Jaccard-based)
                                        ▼
Step 5  ASR ────────────────────────────────────────────► VideoKnowledge.add_asr()
                                        │  _asr {t_sec: text}
                                        ▼
Step 6  OCR ────────────────────────────────────────────► VideoKnowledge.add_ocr()
                                        │  _ocr {t_sec: text}
                                        ▼
Step 7  Depth ──────────────────────────────────────────► VideoKnowledge.add_depth()
                                        │  _depth {t_sec: {near_ratio, mean_depth, …}}
                                        ▼
Step 8  Detection ──────────────────────────────────────► VideoKnowledge.add_detections()
                                        │  _detections {t_sec: [labels]}, known_entities
                                        ▼
Step 12 Qwen ◄────────── domain_hint() + context_for_frame(t_sec) per frame
             └──────────────────────────────────────────► update_qwen_state(result)
                                                               (rolling memory for next frame)
```

### What `context_for_frame(t_sec)` returns

`VideoKnowledge.context_for_frame(t)` assembles up to seven text blocks from deposited data and
returns a single string injected into the Qwen user prompt.  Each block is optional — if the
relevant step was skipped or found no data for this timestamp, the block is omitted.

```
[Prior scene description]: convoy of trucks moving through dust
[Scene segment 2, 4.0s–12.5s]: vehicles on gravel road
[Audio context]: convoy moving north, checkpoint ahead
[Visible text]: B-12  EXIT ONLY
[Depth profile]: near_ratio=0.18  mean=22.40
[Detected objects]: truck, person, barrier
[Prior frame state]: vehicles=2×truck  road=gravel  condition=clear
```

Nearest-frame lookup uses a sorted timestamp index with a configurable `max_gap` (default 2–5 s)
so frames without an exact match still get context from the nearest available entry.

### What `domain_hint()` returns

`VideoKnowledge.domain_hint()` builds a short one-line summary from Gemma's scene classification
and the top detected entity classes:

```
Dominant scene: military convoy | Known objects: truck, soldier, barrier | Visual transitions: 3
```

This is passed to Step 4 (Florence) as a prompt prefix and to Step 12 (Qwen) as a system-prompt
prefix.  Both calls default to no prefix when Gemma was skipped.

### Temporal rolling memory in Step 12

After Qwen processes each frame it calls `VideoKnowledge.update_qwen_state(result)`.  The next
frame's `context_for_frame()` includes a `[Prior frame state]` block derived from that result.
This gives Qwen a one-frame look-back without loading previous images:

```
Frame N context:
  [Prior frame state]: vehicles=2×truck  road=gravel  condition=clear

Frame N+1 sees this in its prompt; Qwen can now reason:
  "road_condition has changed from 'clear' to 'wet' — likely rain or crossing a puddle"
```

### Where to find the implementation

| Component | File | Function / class |
|---|---|---|
| Accumulator | [pipeline/workflows/local/runner.py](../pipeline/workflows/local/runner.py) | `VideoKnowledge` |
| Per-frame context | [pipeline/workflows/local/runner.py](../pipeline/workflows/local/runner.py) | `VideoKnowledge.context_for_frame()` |
| Domain hint | [pipeline/workflows/local/runner.py](../pipeline/workflows/local/runner.py) | `VideoKnowledge.domain_hint()` |
| Qwen batch with context | [pipeline/vision/qwen.py](../pipeline/vision/qwen.py) | `QwenModel.extract_batch()` |
| Pipeline wiring | [pipeline/workflows/local/runner.py](../pipeline/workflows/local/runner.py) | `_run_video_pipeline()` |

### How a human should study this pattern

The agentic accumulator is an instance of the broader *chain-of-thought with external tools*
pattern from language model research, applied to a multimodal inference pipeline.  To study it:

1. Read the `VideoKnowledge` class from top to bottom in `local/runner.py` — notice how each
   `add_*` method stores data in timestamp-indexed dicts and how `_nearest()` handles sparse
   lookups across time.
2. Print `context_for_frame()` output for 10 frames from a real video run and check whether the
   seven blocks contain plausible information.
3. Run Qwen twice on the same video — once with `knowledge=None` (disable wiring) and once with
   the full accumulator — and compare `detailed_captions.md`.  The structured output
   (vehicle groups, road condition) should be more stable and specific with the accumulator.
4. Add a new step between detection and Qwen (e.g. a simple heuristic that flags "high crowd
   density") and practice depositing its output into `VideoKnowledge` so Qwen sees it.

---

## Advanced Implementation Guide

This section is for a human who wants to deeply understand how to replace, tune, or re-implement
the model used at each pipeline step. Read it as an engineering guide, not just a learning checklist.

### Step-by-step implementation recommendations

| Step | Current role | What to understand before replacing it | What breaks if you change it carelessly |
|---|---|---|---|
| 1. Frame extraction | Establishes the timeline and sampling rate | ffmpeg decode, FPS sampling, timestamp stability | ASR/OCR/detection alignment drifts; retrieval and segmentation become inconsistent |
| 2. Vector indexing | Creates reusable frame memory | embedding normalization, vector dimensions, store schema | search, comparison, and retrieval tests become incomparable |
| 3. Gemma analysis | Adds multimodal semantic priors | image-text embedding space, multimodal prompt formatting, batch reuse | wrong scene priors contaminate Florence, Qwen, and the final audit |
| 4. Florence captioning | Produces canonical per-frame scene text | task prompting, caption segmentation, domain hints | later steps lose the baseline textual scene description |
| 5. ASR | Adds aligned audio evidence | timestamped ASR, subtitle-window alignment | Qwen receives wrong audio context and hallucinates causal explanations |
| 6. OCR | Adds visible-text evidence | scene-text OCR, prompt formatting for VLM OCR, prescreen logic | named entities, signs, and UI text disappear or become wrong context |
| 7. Depth | Adds lightweight geometry | relative vs metric depth, aggregation, failure under low texture | later prompts use misleading near/far reasoning |
| 8. Detection | Adds explicit object structure | detector calibration, open-vocabulary labels, small-object recall | entity lists become noisy and poison downstream prompts |
| 9. RF analysis | Adds radio-environment signal metrics | IQ sampling theory, spectrogram computation, SNR estimation | RF link events are invisible; channel degradation frames go unflagged |
| 10. YOLO + SAM | Adds fast prioritized instance structure | detector-speed tradeoffs, mask prompting, taxonomy stability | safety-relevant entities are sorted or segmented incorrectly |
| 11. Gemma directed tracking | Uses Gemma to steer SAM + RF-DETR | vocabulary alignment, bbox prompting, track persistence | tracked-object context becomes sparse or wrong |
| 12. World model | Adds clip-level temporal features | clip sampling, temporal embeddings, runtime fallback behavior | temporal context becomes uninterpretable or inconsistent |
| 13. Qwen reasoning | Fuses all prior evidence per frame | prompt packing, evidence precedence, state carry-over | one wrong upstream cue propagates across multiple frames |
| 14. UniDriveVLA expert analysis | Adds understanding/perception/planning decomposition | bridge schema stability, expert-consensus interpretation | planning or risk signals drift away from visual evidence |
| 15/20. Retrieval tests | Quantify representation quality | controlled query selection, top-k comparison | adaptation results become untrustworthy |
| 16. 3D mapping | Produces spatial world structure | SfM assumptions, pose quality, splat generation | geometry looks plausible but is physically wrong |
| 17. SSL fine-tuning | Tightens domain-specific embeddings | positive-pair design, freeze schedule, collapse avoidance | adapted model overfits or improves only numerically |
| 18. Distillation | Compresses teacher structure | teacher quality, relational loss, anchor losses | student inherits teacher bugs or loses retrieval geometry |
| 19. ONNX export | Creates deployable runtime | export correctness, operator coverage, embedding parity | edge model diverges silently from training model |
| 21. Video description | Produces coarse text hypothesis | prompt-bank design, CLIP text/image alignment | top-level description becomes prompt-biased |
| 22. Multi-model comparison | Exposes disagreement across major analyzers | timestamp matching, agreement metrics, schema normalization | divergence remains hidden behind one preferred model |
| 23. Video synthesis | Produces user-facing summary | schema-first generation, evidence selection, contradiction handling | report sounds confident but hides disagreement |
| 24. Agentic audit | Produces system-facing reasoning trace | compact provenance prompts, timeout budgeting, risk framing | audit step falls back too often or repeats unsupported claims |

### How to think about model selection per step

Use these questions before swapping any model:

1. Is this step producing a **representation**, a **fact**, or a **report**?
2. Does the next step consume raw output directly, or only a compact summary?
3. Is latency dominated by repeated per-frame calls or a single final reasoning call?
4. Is the model used for **visual similarity**, **semantic grounding**, or **long-form reasoning**?

That leads to the practical pattern used in this repo:

- use lighter repeated models for high-frequency frame work
- use heavier reasoning models only for low-frequency synthesis/audit steps
- preserve stable dimensions and schemas where later steps depend on them
- prefer explicit prompt templates for multimodal models instead of implicit defaults

### Per-step implementation advice

#### Steps 1-2: data and memory first

Before changing any model, prove the data path is stable. If timestamps, frame order, or vector dimensions shift, the rest of the pipeline may still run but its outputs stop being comparable.

#### Step 3: Gemma analysis

Treat Gemma as two systems:

- a local embedder for reusable image/text space
- a sidecar generative model for descriptive analysis

If you replace the local embedder, preserve:

- embedding dimension contract via `image_dim()` and `text_dim()`
- L2 normalization
- multimodal prompt formatting with explicit image placeholders
- caching when the same frames are embedded repeatedly

If you replace the sidecar model, preserve:

- OpenAI-compatible endpoint behavior
- concise frame-level prompting
- predictable timeout and unload behavior under Ollama

#### Steps 4-10: evidence layering

These steps are not independent once `VideoKnowledge` is in play. Any replacement should be judged by how well it improves the accumulated context, not just its isolated output quality.

For example:

- a better OCR model is only useful if Qwen can consume its output without increasing false context
- a better detector is only useful if its labels remain stable enough to become prompt context
- a better captioner is only useful if segment analysis remains meaningful

#### Steps 12-14: adaptation and deployment

Never treat a lower loss as sufficient. For this stack, adaptation quality means:

- retrieval neighborhoods improve
- export preserves embedding behavior
- the smaller model remains useful for the actual query types the pipeline supports

That is why the repo keeps:

- before/after retrieval tests
- distillation metrics
- ONNX export plus gallery build

#### Steps 18-19: synthesis versus audit

Use different models or at least different prompts for these two steps:

- Step 22 should optimize for coherent user-facing summarization
- Step 23 should optimize for provenance, risk analysis, and context-flow inspection

Do not collapse them into one generic “LLM summary” step. That removes the distinction between
"what the video means" and "why the pipeline thinks so."

### Best way to study implementation deeply

For each step:

1. Read the wrapper in `pipeline/` or `models/`.
2. Identify input contract, output contract, and runtime fallback path.
3. Run the local pipeline on a short clip and inspect the artifact produced by that step.
4. Replace just one model or prompt variable and rerun.
5. Compare not only the local artifact, but also the later steps that consumed it.

That last part is the main advanced lesson of this repo: each model matters partly because of
its own output quality, and partly because of how its output conditions later reasoning.

---

## Gemma 4 Findings Across the Pipeline

These are the experimentally observed and analytically derived findings from running this
pipeline with `google/gemma-4-it-2b` and `gemma4:e4b` (Ollama).  Full analysis:
`docs/design/gemma4-video-analysis-ssl-distillation.md`.

### Embedding space behaviour (Step 3)

| Observation | Implication |
|---|---|
| Gemma mean pairwise cosine ≈ 0.00 on 30 fps video | Language backbone forces even near-duplicate frames apart — highly discriminative but unstable for dense optical-flow-style similarity |
| DINOv3 mean pairwise cosine ≈ 0.94 on same frames | DINOv3 clusters near-duplicate frames tightly — stable for frame-to-frame similarity, less useful for semantic diversity |
| MNN@3 Gemma vs DINOv3 = 100 % on small samples | Despite different cosine magnitudes, both models agree on which frames are neighbours — Gemma can substitute DINOv3 for Qdrant retrieval |
| MNN@3 Gemma vs CLIP = 100 % on small samples | Gemma and CLIP agree on visual structure — Gemma can serve both image-to-image and text-to-image queries from a single index |

### Caption quality (Steps 3, 4, 10)

- Florence-2 (Step 4) and Qwen (Step 12) produce **identical or near-identical captions for
  consecutive 30 fps frames** because they process each frame independently.
- Gemma 4 **multi-image reasoning** (multiple frames in one prompt) is the correct fix: send
  `[frame_A][frame_B]` with prompt "what changed?" to get transition descriptions instead of
  repeated static descriptions.
- The `_analyze_caption_sequence()` function in `local/runner.py` uses token-Jaccard similarity
  to group frames into segments and only shows a caption change when Jaccard < 0.45.

### What Gemma 4 can do that no other model in the pipeline can

1. **Multi-frame temporal reasoning**: "Is the vehicle slowing down between these 4 frames?"
2. **Native audio + vision grounding**: processes Whisper audio tokens natively, not as injected text
3. **Structured extraction without sidecar**: replaces Qwen (10–12 GB Ollama process) with a
   local 4 GB call
4. **Anomaly explanation**: "Is this frame consistent with the mission baseline? If not, why?"
5. **Language-grounded embeddings**: text query and image query directly comparable in one space

### SSL fine-tuning feasibility (Step 12)

- **Strategy A (recommended)**: freeze LM backbone, fine-tune top 4 of 24 SigLIP ViT-L blocks.
  ~60 M trainable params.  Reuses existing `TemporalPairDataset` and NT-Xent loss unchanged.
  Needs 16 GB VRAM, ~8 min per mission on RTX 3090.
- **Strategy B**: LoRA adapters on full model (2–8 M trainable params), 24 GB VRAM with
  gradient checkpointing.  Best fidelity but higher compute.
- Both strategies are fully unsupervised — no labels required.

### Maximum-hydration distillation chain (Step 13)

```
Gemma 4 SigLIP ViT-L/16   300 M params  dim=1152   (teacher)
        ↓  RKD-DA + KoLeo + caption anchor
DINOv2 ViT-S/14             22 M params  dim=384    (stage 1 student)
        ↓  RKD-D + KoLeo
EfficientViT-S1              6.6 M params  dim=384  (stage 2 student, edge)
        ↓  INT8 ONNX export
Edge model: 7 MB, 8 ms/frame on CPU
```

- Caption anchor loss (λ=0.5): CLIP text embeddings of Florence captions pull the student toward
  language-grounded targets → zero-shot text-query generalisation for free.
- Estimated Recall@1: 0.92 after stage 1, 0.85 after stage 2 (vs 0.78 direct pretrain).
- Total compression from teacher to edge model: ~60×.

---

## What `auto` Means In Practice

These are the model choices that matter most in the current codebase:

| Step | `auto` behavior in practice |
|---|---|
| ASR | Tries registry choice, but falls back to `openai/whisper-large-v3-turbo` when timestamp support is required |
| OCR | GPU-aware chooser; recent runs picked `microsoft/Phi-3.5-vision-instruct` |
| Depth | Recent runs picked `apple/DepthPro-hf` |
| Detection | Recent runs picked `SenseTime/deformable-detr` |
| World model | Registry may propose Cosmos/V-JEPA, but the runtime wrapper currently falls back to `MCG-NJU/videomae-base` |

If you want stable docs-to-runtime correspondence during experiments, explicitly pin:

```bash
ASR_MODEL=openai/whisper-large-v3-turbo
OCR_MODEL=microsoft/Phi-3.5-vision-instruct
DEPTH_MODEL=apple/DepthPro-hf
DETECTION_MODEL=SenseTime/deformable-detr
WORLD_MODEL=MCG-NJU/videomae-base
```

## How To Study The Whole Stack

Recommended order for a human learner:

1. Learn representation learning first: CLIP, DINO, and Gemma open-weight.
2. Add language grounding: Florence, Whisper, OCR, Qwen.
3. Add geometry: depth, detection, SfM, Gaussian splats.
4. Add adaptation: SSL fine-tuning and distillation.
5. Add deployment: ONNX export and gallery search.
6. Finish with synthesis: ontology and narrative generation.

If you only have one weekend:

1. Read CLIP, DINOv2, Florence-2, Whisper, and the Gemma open-weight blog.
2. Run the local pipeline on one short video with `HF_TOKEN` set (for Gemma) or `GEMMA_API_URL` set (for sidecar).
3. Inspect `gemma_analysis.md`, `gemma_captions.md`, `base_search.md`, `scene_captions.md`, `asr_subtitles.md`,
   `comparison.md`, and `video_synthesis.md`.
4. Then study depth/detection/3D only after you understand the retrieval-and-caption core.
