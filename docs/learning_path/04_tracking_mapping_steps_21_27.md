# Tracking, World Models, And 3D Mapping: Steps 21-27

This phase turns evidence into structure.
The pipeline shifts from "what is in this frame?" to "what persists across frames, how does it evolve, and where does it exist in space?"

The core move is from frame-wise snapshots to temporal continuity and spatial geometry.

---

<a id="step-21-yolo--sam-detection-and-segmentation"></a>
## Step 21. YOLO + SAM detection and segmentation

**What it does:**
Run YOLO (or RF-DETR) over each keyframe to produce fast bounding box detections.
Then run SAM (Segment Anything Model) to refine those detections into pixel-accurate masks.
Combine box + mask into a structured per-frame object list.

**Why it matters:**
Step 8 provided approximate detections as context labels.
This step provides pixel-level masks that enable:
- Spatial reasoning: where exactly is the object, how large is it, is it occluded?
- Instance separation: which pixels belong to vehicle A vs vehicle B?
- Foundation for tracking: a mask is a far better starting region for tracking than a bounding box.
- Semantic graph building: objects with masks can be connected spatially ("vehicle A is to the left of building B").

**Implementation:**
- [`pipeline/workflows/local/steps_yolo_sam.py`](../../pipeline/workflows/local/steps_yolo_sam.py)
- [`pipeline/vision/yolo.py`](../../pipeline/vision/yolo.py)
- [`pipeline/vision/sam.py`](../../pipeline/vision/sam.py)

**Key concepts:**

*YOLO vs RF-DETR:*
YOLO11 is a single-stage anchor-based detector optimized for speed.
RF-DETR is a transformer-based detector with better accuracy on small or overlapping objects but slower inference.
The pipeline supports both; `RFDETR_ENABLED` controls which is active.
For aerial footage with many small vehicles, RF-DETR typically outperforms YOLO.

*SAM prompt modes:*
SAM can be prompted in multiple ways:
- Box prompt: give a bounding box from YOLO → SAM refines it to a mask.
- Point prompt: click one or more points inside an object → SAM grows a mask.
- Auto-mask (no prompt): SAM generates all possible masks in the image without guidance.
Box-prompted SAM is used here because YOLO boxes provide good spatial priors.

*Mask quality vs computational cost:*
SAM produces high-quality masks but is expensive.
The pipeline only runs SAM on frames where YOLO produced high-confidence detections.
Low-confidence frames get no masks to save compute.

**Output artifact:**
Per-frame object list with bounding box, class label, confidence, and RLE-encoded mask.
Optionally: overlay images showing masks on frames.

**Human focus:**
- Understand why bounding boxes are insufficient for spatial reasoning: a box around a vehicle includes road, sky, and adjacent vehicles.
- Learn what SAM's "everything" mode produces vs prompted mode: everything mode is much slower and noisier.
- Know the NMS (Non-Maximum Suppression) step: YOLO produces many overlapping boxes; NMS keeps only the highest-confidence one per object.
- Understand mask IoU (Intersection over Union) as the standard quality metric for segmentation.

**Common failure modes:**
- YOLO misses small objects at altitude → SAM never runs; those objects are invisible to later tracking.
- YOLO box is inaccurate → SAM mask extends into wrong region.
- SAM out of memory on high-resolution frames → reduce input resolution or disable SAM on long videos.
- Motion blur → YOLO produces wide uncertain boxes; masks are coarse.

---

<a id="step-22-gemma-directed-tracking"></a>
## Step 22. Gemma directed tracking

**What it does:**
Sample frames from the video, send them to Gemma with a structured question: "What object categories should be tracked in this mission?".
Parse the JSON response to extract a list of object categories and approximate bounding box regions.
Use SAM to segment those Gemma-specified objects.
Use RF-DETR to track Gemma-priority classes across the frame sequence using IoU-based multi-frame matching.

**Why it matters:**
Standard tracking (Step 21) tracks everything the detector finds.
That includes irrelevant background objects that add noise without mission value.
Gemma-directed tracking inverts the priority: language understanding steers perception.
If Gemma determines "this is a vehicle convoy", tracking focuses on vehicles and ignores trees and buildings.
This is the step where reasoning starts directing perception, not just describing it.

**Implementation:**
- [`pipeline/workflows/local/steps_gemma_tracking.py`](../../pipeline/workflows/local/steps_gemma_tracking.py)
- [`pipeline/vision/rfdetr.py`](../../pipeline/vision/rfdetr.py) — `RFDETRTracker`

**Key concepts:**

*Language-guided perception:*
Traditional computer vision: detect everything, then filter by label.
Language-guided perception: ask what to look for first, then run targeted detection.
The distinction matters at scale: running full detection on 1000 frames is expensive; running targeted detection on 1000 frames for 2 classes is much cheaper.

*IoU-based tracking (greedy matching):*
The tracker maintains a set of active tracks.
For each new frame, it computes the IoU (Intersection over Union) between each new detection and each existing track.
A new detection is matched to the track with highest IoU if that IoU exceeds 0.45.
If no track matches, a new track is started.
Track IDs are reset per video (not persistent across missions).

*Context accumulation into VideoKnowledge:*
Tracking results are stored in `frame_facts_json["gemma_tracking"]`.
This allows later steps (Qwen, report generator) to query what was being tracked and what the tracking found.

**Output artifact:**
`gemma_tracking_summary.md` in the video output directory: which categories Gemma prioritized, which tracks were maintained, and how track continuity compares across frames.

**Human focus:**
- Understand the IoU matching criterion: two boxes with IoU < 0.45 are considered different objects; tracks break at that threshold.
- Learn when tracking breaks: fast motion, occlusion, re-entry of an object after it left the frame.
- Know what happens when Gemma is unavailable: the system falls back to tracking all high-confidence YOLO detections.
- Understand why tracking priority matters for a mission report: a tracked vehicle that disappears is more significant than a detected tree in frame 23.

**Common failure modes:**
- Gemma unavailable → directed tracking skips; only default YOLO tracking runs.
- Gemma returns ambiguous categories ("object") → tracker targets too broad a class; everything matches.
- Fast-moving objects → IoU between consecutive frames is below 0.45 even for the same object; track breaks and restarts.
- Object leaves frame temporarily → new track ID assigned on re-entry; track history breaks.

---

<a id="step-23-world-model-video-embeddings"></a>
## Step 23. World model video embeddings

**What it does:**
Encode short video clips (groups of consecutive frames) into temporal embeddings that capture motion and appearance jointly.
Compute clip-level similarity to find recurring scene patterns across the mission.

**Why it matters:**
Frame embeddings (Step 2) lose temporal information: a frame at rest and the same frame mid-motion look identical to CLIP.
Video embeddings encode the temporal evolution — what changed, how fast, what direction.
This allows:
- Clip-level retrieval ("find another moment like this 10-second sequence")
- Temporal anomaly detection ("this clip is unusual compared to the rest of the mission")
- Scene segmentation at clip granularity rather than frame granularity

**Implementation:**
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
- [`pipeline/vision/world.py`](../../pipeline/vision/world.py)

**Key concepts:**

*Temporal vs spatial features:*
CLIP and DINO process one frame at a time.
Video models (Video-CLIP, VideoMAE, InternVideo) process a stack of frames simultaneously.
The temporal axis adds information about: optical flow direction, object motion trajectories, scene transitions.

*Clip chunking:*
The pipeline divides the frame sequence into clips of N frames (configurable).
Each clip gets one embedding.
Short clips (N=4) are more temporally local; long clips (N=32) capture longer-range motion patterns.

*Clip similarity:*
Two clips are similar if their video embeddings are close in cosine distance.
This enables finding repetitive patterns (e.g., repeated passes over the same area) or finding the most anomalous clip in the mission.

**Output artifact:**
Per-clip embedding array (saved as `.npy`), clip metadata with `{start_frame, end_frame, start_t, end_t, embedding_id}`.

**Human focus:**
- Understand the fundamental difference between frame embeddings and video embeddings: what information is added by the temporal axis.
- Learn when temporal embeddings help retrieval vs when frame embeddings are sufficient (static scenes vs dynamic scenes).
- Know the clip length vs temporal resolution tradeoff.

**Common failure modes:**
- Video model not available → step skipped; only frame-level embeddings exist.
- Very short video → not enough frames to form meaningful clips.
- Highly repetitive mission footage (e.g., camera hovering in place) → all clips are nearly identical; no useful temporal variation.

---

<a id="step-24-qwen-detailed-captioning"></a>
## Step 24. Qwen detailed captioning

**What it does:**
For each keyframe, build a full multi-source context string from `VideoKnowledge.context_for_frame(t_sec)` and send it — along with the frame image — to Qwen-VL.
Request a structured JSON response containing: vehicle groups, road surface, traffic conditions, infrastructure, and safety observations.

**Why it matters:**
This is the densest reasoning step in the pipeline.
Qwen receives not just the image but accumulated context from every earlier step:
- Florence caption (what this frame looks like)
- Scene segment (what phase of the mission this is)
- ASR text (what was said near this frame)
- OCR text (what text was visible)
- Depth profile (near/far geometry)
- Detected objects (what was found by detection)
- Previous Qwen state (what the last frame contained)

The output is not a caption but a structured observation: a reasoning result that downstream steps can consume directly.

**Implementation:**
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
- [`pipeline/vision/qwen.py`](../../pipeline/vision/qwen.py)
- [`pipeline/workflows/local/_common.py`](../../pipeline/workflows/local/_common.py) — `VideoKnowledge.context_for_frame()` and `update_qwen_state()`

**Key concepts:**

*Multimodal prompt packing:*
Qwen receives a prompt that combines: a system message with domain context, a user message with the full context string, and the image.
The prompt is carefully ordered: system → context → image → question.
This ordering matters: models attend differently to information presented early vs late in context.

*Rolling state:*
After processing each frame, `update_qwen_state()` stores the result in `VideoKnowledge._last_qwen`.
The next frame's context includes a `[Prior frame state]` line showing what Qwen found in the previous frame.
This lets Qwen track continuity: "the three vehicles present in the prior frame are now four" is a valid observation.

*Structured JSON output:*
Qwen is prompted to return JSON, not free text.
This makes the output machine-readable for later report generation.
If the output is not valid JSON, `parse_error` is flagged and `update_qwen_state()` skips storing it (to avoid propagating bad state).

**Output artifact:**
`detailed_captions.md` in the video output directory: per-frame structured observations in Markdown table format.
Raw JSON responses saved alongside for debugging.

**Human focus:**
- Read a full `context_for_frame()` string and understand what information each line comes from.
- Compare the output for the same frame with and without prior context: how does the rolling state change the response?
- Find a frame where the wrong prior state from the previous frame propagated into an incorrect Qwen observation.
- Understand why JSON structured output is better than free text for downstream reasoning.

**Common failure modes:**
- Qwen API unavailable → step skipped for that frame; output has gaps.
- Context string too long → prompt exceeds model context length; earlier evidence is truncated.
- Prior state is wrong (from a bad previous frame) → error propagates forward until a clear scene change resets it.
- JSON parse failure → Qwen returned free text or malformed JSON; that frame's state is not stored; chain resets.

---

<a id="step-25-unidrivevla-expert-analysis"></a>
## Step 25. UniDriveVLA expert analysis

**What it does:**
Send selected keyframes (or the full sequence) to UniDriveVLA, a vision-language-action model pretrained on driving and outdoor autonomy scenarios.
Receive domain-specific structured analysis: scene understanding, object relationships, trajectory predictions, and recommended actions.

**Why it matters:**
Qwen is a general-purpose multimodal reasoner.
UniDriveVLA is a domain-specific expert trained on driving scenarios.
For missions involving road networks, vehicle behavior, and outdoor navigation, UniDriveVLA provides:
- Higher-quality scene understanding for domain-specific events (turn signals, lane changes, right-of-way conflicts)
- Action-oriented interpretation ("the vehicle should slow down") rather than just description ("a vehicle is near the intersection")
- Standardized driving-domain taxonomy for cross-mission comparison

**Implementation:**
- [`pipeline/vision/unidrive.py`](../../pipeline/vision/unidrive.py) — thin OpenAI-compatible HTTP adapter
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py) — `step_unidrive_analysis()`
- [`pipeline/core/config.py`](../../pipeline/core/config.py) — `UNIDRIVE_*` settings
- Model prep: `python scripts/prepare_models.py --unidrive`

**Key concepts:**

*VLA (Vision-Language-Action) models:*
A VLA extends a VLM (Vision-Language Model) with an action head.
Where a VLM produces text descriptions, a VLA produces both descriptions and action recommendations.
UniDriveVLA is trained on large-scale driving datasets; its action head is calibrated for road navigation scenarios.

*Adapter design — why not run UniDriveVLA directly:*
The upstream UniDriveVLA checkpoint (`owl10/UniDriveVLA_Nusc_Base_Stage3`) is trained on
multi-camera nuScenes format data.
It expects a specific input format that does not match arbitrary single-camera mission video.
The selfsuvis implementation uses a thin HTTP adapter (`UniDriveVLAModel`) that talks to
any OpenAI-compatible vision endpoint and requests the UniDriveVLA structured output schema.
This means **any capable VLM** can be the backend — the driving structure comes from the
prompt, not from the model's training data.
For non-road missions (aerial, maritime, off-road), point `--unidrive-api-url` at
a Qwen2.5-VL-7B sidecar, which handles the schema equally well without domain mismatch.

*Domain-specific vs general-purpose reasoning:*
General VLMs (Qwen, GPT-4V) can reason about any scene but lack calibrated priors for specific domains.
Domain-specific VLAs have strong priors for their training domain but perform poorly outside it.
For aerial drone footage of rural terrain, a driving-specific model may be less useful than a general model.
The adapter pattern lets you swap backends depending on mission type without changing the pipeline.

*Output structure:*
The adapter normalises any backend response into a four-key schema:
- `understanding`: scene summary, traffic context, risk level, key agents
- `perception`: object list with salience, drivable-area estimate, lane structure
- `planning`: recommended action, trajectory hint, hazards
- `mixture_of_experts`: consensus summary, expert agreement, disagreement points
Recommended actions are advisory interpretations ("reduce speed"), not robot commands.

*Backend selection:*
- Road / urban missions: use `owl10/UniDriveVLA_Nusc_Large_Stage3` if vLLM bridge is available
- Aerial / off-road / maritime: use `Qwen/Qwen2.5-VL-7B-Instruct` as backend (avoids domain mismatch)
- Low VRAM (< 8 GB): use `Qwen/Qwen2.5-VL-3B-Instruct`

**Output artifact:**
`unidrive_analysis.md` in the video output directory: per-frame UniDriveVLA structured analysis.
`multi_model_comparison.md` when both Qwen and UniDrive are enabled (side-by-side comparison).

**Human focus:**
- Compare `detailed_captions.md` (Qwen) vs `unidrive_analysis.md` (UniDriveVLA) for the same frame: what does the domain expert add?
- Read `pipeline/vision/unidrive.py` and understand how `_build_user_content()` packs image + context into the OpenAI message format.
- Find a frame where the domain mismatch (driving model on non-driving footage) causes UniDriveVLA to produce wrong or irrelevant output.
- Try replacing the backend with a Qwen2.5-VL-7B sidecar and compare output quality for aerial footage.
- Understand that VLA "actions" are advisory interpretations, not robot commands.

**Common failure modes:**
- `UNIDRIVE_API_URL` not set → step skipped; output uses Qwen analysis only.  Enable with `--unidrive-api-url`.
- Backend model is not vision-capable → HTTP 400 or empty response; adapter returns `service_unavailable`.
- Mission domain does not match backend training domain → off-domain output; switch to a general VLM backend.
- High-altitude aerial footage with driving-domain model → vehicles interpreted as abstract dots; quality drops.
- Backend JSON parse failure → adapter records `parse_error: true` and logs the raw response at DEBUG level.

---

<a id="step-26-base-model-search-test"></a>
## Step 26. Base model search test

**What it does:**
Run a set of fixed test queries against the baseline embedding store (Step 2, before any fine-tuning).
For each query, retrieve the top-K most similar frames and record the results.
Compare results to expected matches if ground truth is available, or record for human review.

**Why it matters:**
This is the diagnostic step before adaptation.
If the baseline retrieval is already excellent, fine-tuning (Steps 28-31) may not be needed.
If the baseline retrieval is weak, the search test identifies which queries fail and what kinds of scenes are confused.
The test result is the baseline that Step 31 (post-fine-tuning search test) will compare against.
Without this step, you cannot know whether fine-tuning improved anything.

**Implementation:**
- [`pipeline/workflows/local/steps_embed.py`](../../pipeline/workflows/local/steps_embed.py)

**Key concepts:**

*What makes a good search test:*
Use queries that span different failure modes:
- Easy positive: "red truck on highway" when one clearly exists.
- Hard positive: "truck viewed from above at night" — tests whether CLIP handles aerial perspective.
- Hard negative: "empty road" — should not retrieve frames with vehicles.
- Out-of-distribution: "submarine" — should retrieve nothing or distant neighbors.

*Precision at K (P@K):*
For each query, count the fraction of top-K results that are actually relevant.
P@5 = 3/5 means 3 of the top 5 retrieved frames are relevant.
This is the standard metric for retrieval evaluation.

*Recall at K (R@K):*
For each query, what fraction of all relevant frames appear in the top K results?
Useful when you care about finding everything, not just getting a few good results.

**Output artifact:**
Search test report: per-query top-K results with frame paths, similarity scores, and (if available) relevance labels.
Summary table: P@K and R@K for each query.

**Human focus:**
- Run three or four queries on your own mission data and inspect the top-5 results manually.
- Identify which query types fail and why (wrong CLIP vocabulary, aerial perspective confusion, domain mismatch).
- Record your findings: you will compare this against Step 31 results after fine-tuning.

**Common failure modes:**
- No relevant frames in the corpus for a query → P@K is undefined; the metric does not tell you the query was out-of-distribution.
- Duplicate or near-duplicate frames dominate results → retrieval looks good numerically but all results show the same moment.
- Query text uses domain vocabulary that CLIP was not trained on → retrieval is random.

---

<a id="step-27-3d-map-and-gaussian-splat"></a>
## Step 27. 3D map and Gaussian Splat

**What it does:**
Run SfM (Structure-from-Motion) using pycolmap on the dense frame set extracted at `SFM_FPS`.
Recover camera poses and a sparse 3D point cloud.
Optionally run nerfstudio splatfacto to produce a dense 3D Gaussian Splat (3DGS) from the SfM poses.
Store both the sparse structure and the dense rendering model.

**Why it matters:**
This is the shift from frame-wise evidence to persistent spatial structure.
A 3D map enables:
- GPS-registered spatial search: "what did the system see at this location at 47.12°N, 8.45°E?"
- Cross-mission change detection: compare the 3D structure from mission A to mission B at the same location.
- View synthesis: the Gaussian Splat can render novel viewpoints not in the original video.
- Robot pose advisory: `POST /query/pose` uses the map to answer "what should I expect to see here?"

**Implementation:**
- [`pipeline/workflows/local/steps_map.py`](../../pipeline/workflows/local/steps_map.py)
- [`pipeline/mapping/`](../../pipeline/mapping)
- [`docs/gaussian_splat.md`](../gaussian_splat.md)

**Key concepts:**

*Structure-from-Motion (SfM):*
SfM recovers 3D structure and camera poses from a set of 2D images.
It works by finding matching feature points (SIFT, ORB, SuperPoint) across overlapping images, then solving for the 3D positions that are consistent with all observations.
pycolmap is the production-quality open-source implementation used here.
Requires: multiple overlapping views of each scene point (why `SFM_FPS=2` instead of 1).

*Sparse vs dense reconstruction:*
SfM output is sparse: typically 50,000-500,000 3D points for a 1000-frame sequence.
These points are at feature matches, not at every surface.
Dense reconstruction (MVS: Multi-View Stereo) fills in the surface; nerfstudio splatfacto does this implicitly with Gaussian primitives.

*3D Gaussian Splatting (3DGS):*
3DGS represents the scene as millions of 3D Gaussian ellipsoids, each with position, shape, opacity, and color.
Rendering is fast (real-time on GPU) because the Gaussians are splatted (projected) onto the camera plane.
The resulting model supports: view synthesis from arbitrary angles, real-time flythrough, and density queries.

*Pose status gate:*
The pipeline only runs splatfacto after SfM succeeds (`pose_status = success`).
If SfM fails (insufficient overlap, poor feature matches, featureless surfaces), the Gaussian Splat step is skipped.

**Output artifact:**
`maps/{mission_id}/splat.ply` — the 3DGS scene model (large file: 100 MB-2 GB depending on duration).
Sparse point cloud: `sparse_reconstruction/` directory with COLMAP format files.
Per-frame `pose_json`: camera pose (rotation + translation) for each frame that was successfully localized.

**Human focus:**
- Understand why SfM needs overlap: a single frame cannot provide any 3D information alone.
- Learn the SfM failure modes: texture-less surfaces (white walls, water), pure rotation without translation, too-fast camera motion.
- Understand what a Gaussian Splat represents: not a mesh, not a point cloud, but a probabilistic volumetric representation.
- Know the scale ambiguity problem: SfM without GPS produces a map in arbitrary units; GPS registration resolves this.

**Common failure modes:**
- Too few overlapping views (low `SFM_FPS`) → SfM produces a degenerate map or fails entirely.
- Featureless terrain (uniform grass, water, sand) → insufficient feature matches; SfM fails.
- Pure rotation (drone pivoting in place without translation) → SfM has degenerate geometry; reconstruction is wrong.
- Long video with scene changes → SfM may split into disconnected sub-models.
- nerfstudio container not running → splatfacto step skipped; only sparse SfM output exists.

---

## End Of Phase: What You Should Understand

After Steps 21-27, a human should be able to answer:

- Which objects persist across frames and with what track IDs?
- What language context is fed into the densest reasoning step (Qwen) and where does each piece come from?
- Do the temporal clip models capture genuinely different information than frame embeddings?
- Can the scene be placed into a usable 3D spatial structure?
- What does the baseline retrieval performance look like before adaptation?

If you cannot answer those questions, especially the last two, do not proceed to adaptation.
Adaptation is only valuable if you know what the baseline is and where it fails.

## Related Docs

- [Sensors and fusion: Steps 9-20](03_sensor_steps_09_20.md)
- [Adaptation and audit: Steps 28-35](05_adaptation_eval_steps_28_35.md)
- [Agentic knowledge flow](06_agentic_knowledge_flow.md)
- [3D Gaussian Splat](../gaussian_splat.md)
