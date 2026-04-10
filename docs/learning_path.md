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

The current local full-analysis pipeline has 23 ordered steps per video:

1. Frame extraction
2. Vector store indexing
3. Gemma open-weight analysis
4. Florence scene captioning
5. ASR transcription
6. OCR text extraction
7. Depth estimation
8. Object detection (HuggingFace RT-DETR / Grounding DINO)
9. **YOLO11 + SAM2/3 detection and segmentation** *(new)*
10. Gemma 4 directed tracking
11. World model video embeddings
12. Qwen detailed captioning
13. UniDriveVLA expert analysis
14. Base model search test
15. 3D map + Gaussian Splat
16. SSL DINO fine-tuning
17. Knowledge distillation
18. ONNX export + gallery build
19. Fine-tuned search test
20. Model comparison + video description
21. Multi-model comparison
22. Video synthesis
23. Agentic flow audit

## Before You Start

Minimum practical setup:

1. Create the venv: `make venv`
2. Install `ffmpeg`
3. Put `.mp4` or `.mov` files in `data_test/videos/`
4. Optionally run Qdrant on `localhost:6333`
5. Optionally prefetch local-model assets with `python scripts/prepare_models.py --all`

Useful mental model:

- Steps 1-2 build the raw visual memory.
- Step 3 analyses that memory with Gemma open-weight across all multimodal use-cases.
- Steps 4-13 attach language, text, geometry, temporal structure, and expert driving analysis — and feed results forward
  into a shared `VideoKnowledge` accumulator so each step benefits from all earlier steps.
- Steps 9-10 add priority-aware detection/segmentation and Gemma-directed tracking.
- Steps 14-20 evaluate and adapt the representation.
- Steps 21-23 build cross-model synthesis, a narrative summary, and a final reasoning audit.
- The 3D map (step 15) runs concurrently with steps 5-13 in a background thread.

Study note:

- The numbered list above is the canonical runtime order.
- Some deep-dive sections later in this document are grouped pedagogically rather than strictly in runtime order.
- When in doubt, use the canonical 23-step list and [`pipeline/workflows/local/runner.py`](/home/vola/src/selfsuvis/pipeline/workflows/local/runner.py) as the execution source of truth.

The agentic flow (Steps 3 → 4 → 5–10 → 12 → 20 → 21): see the
**Agentic Knowledge Flow** section below for a full data-flow diagram.

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

### Step 9. YOLO11 + SAM2/3 detection and segmentation *(new)*

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

### Step 10. Gemma 4 directed tracking *(new)*

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

### Step 11. World model video embeddings

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

### Step 12. Qwen detailed captioning

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

### Step 13. UniDriveVLA expert analysis

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

### Step 14. Base model search test

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

### Step 16. SSL DINO fine-tuning

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

### Step 17. Knowledge distillation — maximum hydration chain

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

### Step 18. ONNX export + gallery build

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

### Step 19. Fine-tuned search test

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

### Step 20. Model comparison + video description

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

### Step 21. Multi-model comparison

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

### Step 15. 3D map + Gaussian Splat

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

### Step 22. Video synthesis

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

### Step 23. Agentic flow audit

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
| 9. YOLO + SAM | Adds fast prioritized instance structure | detector-speed tradeoffs, mask prompting, taxonomy stability | safety-relevant entities are sorted or segmented incorrectly |
| 10. Gemma directed tracking | Uses Gemma to steer SAM + RF-DETR | vocabulary alignment, bbox prompting, track persistence | tracked-object context becomes sparse or wrong |
| 11. World model | Adds clip-level temporal features | clip sampling, temporal embeddings, runtime fallback behavior | temporal context becomes uninterpretable or inconsistent |
| 12. Qwen reasoning | Fuses all prior evidence per frame | prompt packing, evidence precedence, state carry-over | one wrong upstream cue propagates across multiple frames |
| 13. UniDriveVLA expert analysis | Adds understanding/perception/planning decomposition | bridge schema stability, expert-consensus interpretation | planning or risk signals drift away from visual evidence |
| 14/19. Retrieval tests | Quantify representation quality | controlled query selection, top-k comparison | adaptation results become untrustworthy |
| 15. 3D mapping | Produces spatial world structure | SfM assumptions, pose quality, splat generation | geometry looks plausible but is physically wrong |
| 16. SSL fine-tuning | Tightens domain-specific embeddings | positive-pair design, freeze schedule, collapse avoidance | adapted model overfits or improves only numerically |
| 17. Distillation | Compresses teacher structure | teacher quality, relational loss, anchor losses | student inherits teacher bugs or loses retrieval geometry |
| 18. ONNX export | Creates deployable runtime | export correctness, operator coverage, embedding parity | edge model diverges silently from training model |
| 20. Video description | Produces coarse text hypothesis | prompt-bank design, CLIP text/image alignment | top-level description becomes prompt-biased |
| 21. Multi-model comparison | Exposes disagreement across major analyzers | timestamp matching, agreement metrics, schema normalization | divergence remains hidden behind one preferred model |
| 22. Video synthesis | Produces user-facing summary | schema-first generation, evidence selection, contradiction handling | report sounds confident but hides disagreement |
| 23. Agentic audit | Produces system-facing reasoning trace | compact provenance prompts, timeout budgeting, risk framing | audit step falls back too often or repeats unsupported claims |

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
