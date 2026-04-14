# Perception Core: Steps 1-8

This phase builds the first useful representation of the mission.
By the end of Step 8, the system has raw frames, embeddings, language summaries, speech, visible text, geometry hints, and object structure.

<a id="step-1-frame-extraction"></a>
## Step 1. Frame extraction

What it does:
Turn a compressed video into sampled image frames.

Why it matters:
Every later model inherits this sampling choice.
If you undersample, you miss events.
If you oversample, cost rises everywhere.

Implementation:
- [`pipeline/workflows/local/steps_embed.py`](../../pipeline/workflows/local/steps_embed.py)
- [`pipeline/media/frames.py`](../../pipeline/media/frames.py)

What a human should focus on:
- FPS
- keyframes vs sampled frames
- motion loss
- alignment with audio and sidecar data

<a id="step-2-vector-store-indexing"></a>
## Step 2. Vector store indexing

What it does:
Encode each frame into embedding spaces and insert them into search memory.

Why it matters:
This is the retrieval backbone for search, comparison, and later evaluation.

Implementation:
- [`models/openclip_model.py`](../../models/openclip_model.py)
- [`models/dino_model.py`](../../models/dino_model.py)
- [`pipeline/workflows/local/steps_embed.py`](../../pipeline/workflows/local/steps_embed.py)
- [`pipeline/storage/qdrant.py`](../../pipeline/storage/qdrant.py)

What a human should focus on:
- CLIP vs DINO roles
- cosine similarity
- nearest-neighbor retrieval
- Qdrant vs in-memory fallback

<a id="step-3-gemma-multimodal-analysis"></a>
## Step 3. Gemma multimodal analysis

What it does:
Create scene-level semantic context, sample-level descriptions, and cross-modal retrieval probes.

Why it matters:
It upgrades the pipeline from “frame store” to “scene-aware memory”.

Implementation:
- [`models/gemma_model.py`](../../models/gemma_model.py)
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)

What a human should focus on:
- scene change detection
- scene clustering
- text-image retrieval
- what Gemma adds that CLIP and DINO do not

<a id="step-4-florence-scene-captioning"></a>
## Step 4. Florence scene captioning

What it does:
Give each key frame a readable language description.

Why it matters:
It is the first human-friendly semantic layer.
Later steps reuse these captions as context.

Implementation:
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
- [`pipeline/vision/factory.py`](../../pipeline/vision/factory.py)

What a human should focus on:
- prompt-conditioned captioning
- domain hints
- caption drift across similar frames

<a id="step-5-asr-transcription"></a>
## Step 5. ASR transcription

What it does:
Extract and timestamp speech from the audio track.

Why it matters:
Speech often contains mission context that never appears visually.

Implementation:
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
- [`pipeline/vision/asr.py`](../../pipeline/vision/asr.py)
- [`pipeline/media/audio.py`](../../pipeline/media/audio.py)

What a human should focus on:
- timestamps
- noise and overlap failures
- when speech improves later reasoning

<a id="step-6-ocr-text-extraction"></a>
## Step 6. OCR text extraction

What it does:
Read visible text from the scene.

Why it matters:
OCR often changes a vague interpretation into a precise one.

Implementation:
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
- [`pipeline/vision/ocr.py`](../../pipeline/vision/ocr.py)

What a human should focus on:
- scene text vs document text
- blur and small-font failures
- multilingual or stylized text

<a id="step-7-depth-estimation"></a>
## Step 7. Depth estimation

What it does:
Predict a compact geometric summary from monocular imagery.

Why it matters:
This gives the pipeline an approximate 3D prior before full mapping exists.

Implementation:
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
- [`pipeline/vision/depth.py`](../../pipeline/vision/depth.py)

What a human should focus on:
- relative vs metric depth
- scene openness
- where monocular depth is trustworthy vs weak

<a id="step-8-object-detection"></a>
## Step 8. Object detection

What it does:
Find labeled object instances in each frame.

Why it matters:
This is the first explicit object-structured representation.

Implementation:
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
- [`pipeline/vision/detection.py`](../../pipeline/vision/detection.py)

What a human should focus on:
- box quality
- vocabulary limits
- open-vocabulary vs fixed-class tradeoffs

## End Of Phase: What You Should Understand

After Steps 1-8, a human should be able to answer:

- What frames exist?
- How are they indexed?
- What does the system think is happening in language?
- What speech and visible text were present?
- What is near or far?
- Which objects were detected?

If you cannot answer those questions, do not move to the sensor and fusion phase yet.
