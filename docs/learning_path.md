# Learning Path

This document follows the real `python main.py --mode demo` pipeline in this repo.
It explains, step by step:

- what each step does
- which model or tool is used
- what `auto` resolves to in practice in this codebase
- which paper to read first
- how a human should study the topic behind that step

The current demo has 17 ordered steps per video:

1. Frame extraction
2. Vector store indexing
3. Florence scene captioning
4. ASR transcription
5. OCR text extraction
6. Depth estimation
7. Object detection
8. World model video embeddings
9. Qwen detailed captioning
10. Base model search test
11. SSL DINO fine-tuning
12. Knowledge distillation
13. ONNX export + gallery build
14. Fine-tuned search test
15. Model comparison + video description
16. 3D map + Gaussian Splat
17. Video synthesis

## Before You Start

Minimum practical setup:

1. Create the venv: `make venv`
2. Install `ffmpeg`
3. Put `.mp4` or `.mov` files in `data_test/videos/`
4. Optionally run Qdrant on `localhost:6333`
5. Optionally prefetch large models with `python scripts/prepare_models.py --all`

Useful mental model:

- Steps 1-2 build the raw visual memory.
- Steps 3-9 attach language, text, geometry, and temporal structure.
- Steps 10-15 evaluate and adapt the representation.
- Steps 16-17 build 3D scene structure and a narrative summary.

## Step-By-Step Learning Path

### Step 1. Frame extraction

**What the demo does**

The pipeline uses `ffmpeg` to decode the source video and save JPEG frames at the requested FPS. This is not an ML step, but every downstream model depends on its output quality and sampling rate.

**Tool / model used**

- `ffmpeg`
- Output: `data_test/videos_test/<video>/frames/`

**Why it matters**

This step decides temporal resolution. If you sample too sparsely, you lose motion and speech alignment. If you sample too densely, every later step gets slower and more expensive.

**Essential reading**

- FFmpeg documentation: https://ffmpeg.org/documentation.html

**How a human should learn this topic**

Learn video basics first: FPS, GOP/keyframes, H.264/H.265 compression, color spaces, and how frame rate changes affect motion analysis. Then practice extracting the same clip at `1`, `2`, `4`, and `8` FPS and compare what information survives.

---

### Step 2. Vector store indexing

**What the demo does**

Each extracted frame is embedded into two visual spaces and then inserted into a vector store. This is the memory layer used later for search, comparison, and retrieval-based reasoning.

**Models used in this repo**

- `OpenCLIP` image encoder from [models/openclip_model.py](/home/vola/src/selfsuvis/models/openclip_model.py)
- `DINO` image encoder from [models/dino_model.py](/home/vola/src/selfsuvis/models/dino_model.py)
- Current default OpenCLIP config: `ViT-B-16` with the `openai` weights
- Current DINO label in the demo: `dinov3_vitb14`
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

### Step 3. Florence scene captioning

**What the demo does**

The demo captions every keyframe with a detailed textual scene description. This gives you a readable semantic summary before later multimodal steps add speech, OCR, or object structure.

**Model used**

- `microsoft/Florence-2-large`
- Wrapper: [pipeline/florence_model.py](/home/vola/src/selfsuvis/pipeline/florence_model.py)
- Prompt used by the repo: `<MORE_DETAILED_CAPTION>`

**Why it matters**

This step turns raw pixels into natural-language scene summaries. Those summaries help both debugging and downstream human inspection.

**Essential reading**

- Florence-2: https://arxiv.org/abs/2311.06242

**How a human should learn this topic**

Learn image captioning as a sequence-generation problem. Then study prompt-conditioned vision models and how one model can do captioning, detection, grounding, and segmentation with task prompts. A good exercise is to compare Florence captions against hand-written captions for 20 frames and note what details the model systematically misses.

---

### Step 4. ASR transcription

**What the demo does**

The pipeline extracts audio from the video, runs speech recognition, and aligns subtitle segments to video frames.

**Model used**

- Wrapper: [pipeline/asr_model.py](/home/vola/src/selfsuvis/pipeline/asr_model.py)
- Practical default in this repo: `openai/whisper-large-v3-turbo`
- Important repo behavior: if `ASR_MODEL=auto` selects a non-Whisper model that cannot provide native timestamps in this pipeline, the wrapper falls back to Whisper

**Why it matters**

Speech often contains mission context, place names, instructions, or narration that is not visible in the image. Step 9 later injects this ASR text into the VLM prompt.

**Essential reading**

- Whisper: https://openai.com/research/whisper/

**How a human should learn this topic**

Study audio preprocessing, spectrograms, encoder-decoder speech models, and timestamp alignment. Then learn practical ASR failure modes: overlapping speakers, clipped words, noise, and code-switching. A good exercise is to compare Whisper output against ground truth for one short clip and mark insertions, deletions, and timing drift.

---

### Step 5. OCR text extraction

**What the demo does**

The pipeline looks for visible text inside each frame. That text can come from road signs, dashboards, labels, UI overlays, subtitles burned into the video, equipment markings, or documents in view.

**Models used**

- Wrapper: [pipeline/ocr_model.py](/home/vola/src/selfsuvis/pipeline/ocr_model.py)
- `OCR_MODEL=auto` is GPU-aware and can choose TrOCR, GOT-OCR2, Florence, Qwen, Phi-3.5 Vision, or DeepSeek OCR depending on setup
- In recent demo runs, `auto` selected `microsoft/Phi-3.5-vision-instruct`
- When a Qwen/Ollama sidecar is already active, the repo can route OCR through that sidecar instead of loading another heavy local VLM

**Why it matters**

OCR is often the difference between generic understanding and operational understanding. Text tells you what object you are looking at, not just what it resembles.

**Essential reading**

- TrOCR: https://arxiv.org/abs/2109.10282
- Qwen2.5-VL technical report: https://arxiv.org/abs/2502.13923

**How a human should learn this topic**

Start with the difference between document OCR and scene-text OCR. Then learn layout, perspective distortion, low-resolution text, multilingual text, and text-plus-graphics reasoning. A practical exercise is to collect 50 failure cases and group them into small text, blur, low contrast, non-Latin script, curved text, and occlusion.

---

### Step 6. Depth estimation

**What the demo does**

The pipeline predicts monocular depth and stores a compact five-number summary per frame instead of a full dense depth map.

**Model used**

- Wrapper: [pipeline/depth_model.py](/home/vola/src/selfsuvis/pipeline/depth_model.py)
- `DEPTH_MODEL=auto` is registry-driven
- In recent demo runs, `auto` selected `apple/DepthPro-hf`
- The wrapper will retry on CPU if CUDA runs out of memory

**Why it matters**

Depth gives a lightweight geometric prior: near/far structure, scene openness, clutter, and relative scale. That becomes useful for motion interpretation, 3D reconstruction, and future robotics extensions.

**Essential reading**

- Depth Pro: https://arxiv.org/abs/2410.02073

**How a human should learn this topic**

Study the difference between metric depth, relative depth, and inverse depth. Then learn why monocular depth is fundamentally ambiguous and how modern models still recover useful structure from large-scale training. A good exercise is to inspect depth predictions on indoor, outdoor, aerial, and low-light frames and see where relative ordering breaks.

---

### Step 7. Object detection

**What the demo does**

The pipeline predicts object instances and normalized bounding boxes for each frame.

**Model used**

- Wrapper: [pipeline/detection_model.py](/home/vola/src/selfsuvis/pipeline/detection_model.py)
- `DETECTION_MODEL=auto` is registry-driven
- In recent demo runs, `auto` selected `SenseTime/deformable-detr`
- Open-vocabulary alternatives are also supported via `DETECTION_LABELS`

**Why it matters**

Detection converts a scene from global semantics into object-level structure. It is the first step toward counting, tracking, event reasoning, and symbolic world state.

**Essential reading**

- Deformable DETR: https://arxiv.org/abs/2010.04159

**How a human should learn this topic**

Learn the difference between classification, detection, and segmentation. Then study IoU, confidence calibration, small-object failure modes, and open-vocabulary detection. A good exercise is to compare detector outputs on crowded frames versus sparse frames and see how confidence behaves.

---

### Step 8. World model video embeddings

**What the demo does**

The pipeline groups consecutive frames into clips and computes one temporal embedding per clip. This is the video-native representation step, as opposed to the frame-by-frame image encoders used earlier.

**Model used**

- Wrapper: [pipeline/world_model.py](/home/vola/src/selfsuvis/pipeline/world_model.py)
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

### Step 9. Qwen detailed captioning

**What the demo does**

The demo sends each frame, plus optional ASR and OCR context, to a vision-language model for a richer structured description than the Florence caption.

**Model used**

- Wrapper: [pipeline/qwen_model.py](/home/vola/src/selfsuvis/pipeline/qwen_model.py)
- Typical local sidecar in this repo: `qwen2.5vl:7b` via Ollama
- Alternative: OpenAI-compatible vLLM endpoint

**Why it matters**

This step is the highest-level semantic interpreter in the pipeline. It can integrate image content with speech and OCR, which earlier specialized models cannot do jointly.

**Essential reading**

- Qwen2.5-VL technical report: https://arxiv.org/abs/2502.13923

**How a human should learn this topic**

Study multimodal prompting, context packing, and grounding failures. Then learn how OCR and ASR context can improve visual interpretation but can also bias it. A good exercise is to run the same frame with and without OCR/ASR context and compare how the explanation changes.

---

### Step 10. Base model search test

**What the demo does**

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

### Step 11. SSL DINO fine-tuning

**What the demo does**

The demo fine-tunes the DINO backbone on the extracted frames using temporal positives. Nearby frames are treated as semantically related views of the same local scene state.

**Models used**

- Teacher backbone being adapted: `dinov3_vitb14` alias in this repo
- Fine-tuning code path: [pipeline/demo_runner.py](/home/vola/src/selfsuvis/pipeline/demo_runner.py#L1471) and the SSL modules it calls
- Loss family: temporal contrastive self-supervision

**Why it matters**

This is domain adaptation. The base foundation model is strong, but this step bends it toward the mission distribution represented by the current video collection.

**Essential reading**

- DINOv2: https://arxiv.org/abs/2304.07193
- SimCLR, for NT-Xent style contrastive training: https://arxiv.org/abs/2002.05709

**How a human should learn this topic**

Study self-supervised contrastive learning, positive/negative pair construction, and representation collapse. Then focus on temporal positives for video. A good exercise is to ask which frame pairs should be positive in fast motion, cuts, zooms, and camera shake.

---

### Step 12. Knowledge distillation

**What the demo does**

The fine-tuned large teacher is used to train a smaller student backbone. The code uses relational knowledge distillation, not just plain logits matching.

**Models used**

- Teacher: fine-tuned DINO ViT-B/14
- Student: `dinov2_vits14`
- Implementation: [pipeline/distill.py](/home/vola/src/selfsuvis/pipeline/distill.py)
- Losses: RKD distance, RKD angle, cosine anchor, KoLeo regularization

**Why it matters**

This is the edge-deployment bridge. You keep most of the teacher’s structure while moving to a cheaper student suitable for faster inference.

**Essential reading**

- Distilling the Knowledge in a Neural Network: https://arxiv.org/abs/1503.02531
- Relational Knowledge Distillation: https://arxiv.org/abs/1904.05068

**How a human should learn this topic**

Learn the difference between task distillation, feature distillation, and relational distillation. Then study what information a student should preserve for retrieval, not just classification. A good exercise is to compare retrieval quality before and after distillation rather than looking only at training loss.

---

### Step 13. ONNX export + gallery build

**What the demo does**

The adapted backbone is exported to ONNX and then used to build an embedding gallery for lightweight edge inference.

**Models used**

- Export target: fine-tuned teacher by default, or distilled student if that path succeeds
- Runtime artifact: `edge_models/dino_demo.onnx`
- Gallery artifact: `edge_models/gallery.npz`

**Why it matters**

This step turns a research model into a deployable artifact. It is the operational packaging stage.

**Essential reading**

- ONNX documentation: https://onnx.ai/

**How a human should learn this topic**

Learn model export, operator compatibility, dynamic versus static shapes, and runtime providers. The key practical skill is not “convert a model once,” but “trace what broke after export and prove the exported model still produces a compatible embedding space.”

---

### Step 14. Fine-tuned search test

**What the demo does**

The demo reruns the same retrieval test from Step 10, now with the adapted model.

**Models used**

- Fine-tuned DINO backbone or exported ONNX runtime equivalent

**Why it matters**

This is the direct “did adaptation help?” check. It closes the loop between training and task utility.

**Essential reading**

- DINOv2: https://arxiv.org/abs/2304.07193

**How a human should learn this topic**

Learn to evaluate representation changes with controlled before/after comparisons. Review top-k retrieval changes, not just the top hit. The question to ask is whether the model became more domain-sensitive without becoming overly narrow.

---

### Step 15. Model comparison + video description

**What the demo does**

The pipeline ranks a curated set of text prompts against the average CLIP embedding of the video and writes out a short video-level description.

**Models used**

- OpenCLIP text and image encoders
- Prompt bank defined in the demo runner

**Why it matters**

This step is a simple but strong cross-modal summary: it asks which textual hypotheses best fit the visual evidence across the whole clip.

**Essential reading**

- CLIP: https://arxiv.org/abs/2103.00020

**How a human should learn this topic**

Study prompt-set design, prompt leakage, and dataset bias in text-image similarity. Try rewriting the prompt bank for a narrower domain such as traffic, ISR, agriculture, or industrial inspection and see how the top descriptions change.

---

### Step 16. 3D map + Gaussian Splat

**What the demo does**

The pipeline first tries classical Structure-from-Motion with pycolmap to recover poses and sparse geometry. If SfM fails or is partial, it falls back to a PCA point cloud. It then builds a Gaussian Splat representation for interactive viewing.

**Models and tools used**

- SfM toolchain: `pycolmap` / COLMAP-style incremental mapping
- Sparse-map fallback: PCA over frame embeddings
- Gaussian Splat builder: repo `gsplat` integration
- Output directory: `3d_map/`

**Why it matters**

This step converts representation learning into explicit scene structure. It is the bridge from semantic understanding to geometry and rendering.

**Essential reading**

- Structure-from-Motion Revisited (COLMAP): https://openaccess.thecvf.com/content_cvpr_2016/html/Schoenberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html
- 3D Gaussian Splatting for Real-Time Radiance Field Rendering: https://arxiv.org/abs/2308.04079

**How a human should learn this topic**

Learn camera geometry, epipolar constraints, feature matching, bundle adjustment, and failure modes like low parallax or repeated texture. Then study why Gaussian splatting is useful after pose recovery. The key conceptual shift is from “which object is in the frame” to “where is the camera in a consistent 3D world.”

---

### Step 17. Video synthesis

**What the demo does**

The pipeline synthesizes a final ontology-style summary and a narrative report from all earlier artifacts. This is a report-writing step driven by multimodal context rather than a single raw model output.

**Models used**

- Qwen sidecar / OpenAI-compatible chat endpoint
- Input context includes captions, OCR, ASR, detections, descriptions, and 3D outputs

**Why it matters**

This step turns the pipeline from a collection of independent analyses into a human-readable final product.

**Essential reading**

- Qwen2.5-VL technical report: https://arxiv.org/abs/2502.13923

**How a human should learn this topic**

Study ontology design, schema-first prompting, and evidence-backed summarization. The main skill here is not generic prompt writing; it is learning how to convert noisy multimodal evidence into a consistent structured report without hallucinating unsupported conclusions.

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

1. Learn representation learning first: CLIP and DINO.
2. Add language grounding: Florence, Whisper, OCR, Qwen.
3. Add geometry: depth, detection, SfM, Gaussian splats.
4. Add adaptation: SSL fine-tuning and distillation.
5. Add deployment: ONNX export and gallery search.
6. Finish with synthesis: ontology and narrative generation.

If you only have one weekend:

1. Read CLIP, DINOv2, Florence-2, Whisper.
2. Run the demo on one short video.
3. Inspect `base_search.md`, `scene_captions.md`, `asr_subtitles.md`, `comparison.md`, and `video_synthesis.md`.
4. Then study depth/detection/3D only after you understand the retrieval-and-caption core.
