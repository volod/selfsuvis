# Tracking, World Models, And 3D Mapping: Steps 21-27

This phase turns evidence into structure.
The pipeline shifts from “what is in this frame?” to “what persists, how does it evolve, and where does it exist in space?”

<a id="step-21-yolo--sam-detection-and-segmentation"></a>
## Step 21. YOLO + SAM detection and segmentation

What it does:
Refine object understanding with faster detector passes and segmentation masks.

Why it matters:
Masks and stronger localized detections make later tracking and graph building more useful.

Implementation:
- [`pipeline/workflows/local/steps_yolo_sam.py`](../../pipeline/workflows/local/steps_yolo_sam.py)
- [`pipeline/vision/yolo.py`](../../pipeline/vision/yolo.py)
- [`pipeline/vision/sam.py`](../../pipeline/vision/sam.py)

<a id="step-22-gemma-directed-tracking"></a>
## Step 22. Gemma directed tracking

What it does:
Use language-guided scene understanding to focus tracking on likely important objects.

Why it matters:
This is where reasoning starts steering perception, not just describing it.

Implementation:
- [`pipeline/workflows/local/steps_gemma_tracking.py`](../../pipeline/workflows/local/steps_gemma_tracking.py)

<a id="step-23-world-model-video-embeddings"></a>
## Step 23. World model video embeddings

What it does:
Encode clips instead of isolated frames.

Why it matters:
Temporal context is required for motion, continuity, and clip-level similarity.

Implementation:
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)

<a id="step-24-qwen-detailed-captioning"></a>
## Step 24. Qwen detailed captioning

What it does:
Combine accumulated evidence into richer per-frame reasoning.

Why it matters:
This is one of the densest reasoning steps because it receives cross-step context.

Implementation:
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
- [`pipeline/vision/qwen.py`](../../pipeline/vision/qwen.py)
- [`pipeline/workflows/local/_common.py`](../../pipeline/workflows/local/_common.py)

Human focus:
- multimodal prompt packing
- context contamination
- rolling state across frames

<a id="step-25-unidrivevla-expert-analysis"></a>
## Step 25. UniDriveVLA expert analysis

What it does:
Add a domain-specific expert layer for understanding, perception, and planning.

Why it matters:
It separates description from action-oriented interpretation.

Implementation:
- [`pipeline/vision/unidrive.py`](../../pipeline/vision/unidrive.py)
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)

<a id="step-26-base-model-search-test"></a>
## Step 26. Base model search test

What it does:
Run a retrieval sanity check on the baseline representation.

Why it matters:
If retrieval is weak here, later adaptation has a clear target.

Implementation:
- [`pipeline/workflows/local/steps_embed.py`](../../pipeline/workflows/local/steps_embed.py)

<a id="step-27-3d-map-and-gaussian-splat"></a>
## Step 27. 3D map and Gaussian Splat

What it does:
Build geometric structure, camera poses, and optional splat-based scene rendering.

Why it matters:
This is the shift from frame-wise evidence to persistent spatial structure.

Implementation:
- [`pipeline/workflows/local/steps_map.py`](../../pipeline/workflows/local/steps_map.py)
- [`pipeline/mapping`](../../pipeline/mapping)
- [`docs/gaussian_splat.md`](../gaussian_splat.md)

Human focus:
- pose recovery
- sparse vs dense structure
- what a semantic graph attached to a map really means

## End Of Phase: What You Should Understand

After Steps 21-27, a human should be able to answer:

- Which objects persist across frames?
- What context is fed into dense reasoning?
- Do the temporal models capture more than still images?
- Can the scene be placed into a usable 3D structure?
