# Perception Core: Steps 1-8

This phase builds the first useful representation of the mission.
By the end of Step 8 the system has raw frames, embeddings, language summaries, speech, visible text, geometry hints, and object structure.
Everything in later phases is built on top of what these eight steps produce.

---

<a id="step-1-frame-extraction"></a>
## Step 1. Frame extraction

**What it does:**
Turn a compressed video into a sequence of JPEG frames saved to disk, each tagged with a timestamp in seconds.

**Why it matters:**
Every later model inherits the sampling decision made here.
If you undersample you miss events.
If you oversample, cost rises across all downstream models.
The FPS choice is also the implicit temporal resolution of the entire pipeline.

**Implementation:**
- [`pipeline/workflows/local/steps_embed.py`](../../pipeline/workflows/local/steps_embed.py) — `step_extract_frames()`
- [`pipeline/media/frames.py`](../../pipeline/media/frames.py)

**Key concept: why FFmpeg, not raw decoding?**
FFmpeg handles codec quirks, variable frame rates, B-frames, and container metadata.
Trying to decode video with OpenCV or PIL directly often produces wrong timestamps or drops frames silently.

**Output artifact:**
- `frames_metadata.json` in the video output directory: a JSON list of `{path, t_sec}` for every extracted frame.
- One JPEG file per frame, named by index.

**What a human should focus on:**
- How FPS interacts with scene change rate: a slow drone flyover needs 1-2 fps; a car chase needs 5-10.
- Keyframe vs regular-frame sampling: some extractors only decode I-frames (fast but sparse), others decode all frames (slow but complete).
- Timestamp drift: the `t_sec` written to metadata must align with ASR and sidecar GPS timestamps or fusion will be off.
- Motion blur: at high speeds, a low shutter or compressed video frame may be useless even if extracted correctly.

**Common failure modes:**
- Video container has no timestamps → all frames get wrong `t_sec` values → fusion is broken.
- H.265/HEVC codec not installed → FFmpeg silently falls back or errors.
- Corrupt keyframe at start of file → frame count is wrong, metadata has off-by-one errors.
- Very high FPS video + low target FPS → FFmpeg skips with rounding, missing exact scene change moments.

---

<a id="step-2-vector-store-indexing"></a>
## Step 2. Vector store indexing

**What it does:**
Encode each frame into one or two embedding spaces (CLIP, DINOv3) and insert the vectors into a searchable store.

**Why it matters:**
This is the retrieval backbone.
Every search query, similarity comparison, and later evaluation step operates on these vectors.
Getting them wrong poisons the retrieval signal for the rest of the pipeline.

**Implementation:**
- [`models/openclip_model.py`](../../models/openclip_model.py) — `OpenCLIPEmbedder`
- [`models/dino_model.py`](../../models/dino_model.py) — `DINOEmbedder`
- [`pipeline/workflows/local/steps_embed.py`](../../pipeline/workflows/local/steps_embed.py)
- [`pipeline/storage/qdrant.py`](../../pipeline/storage/qdrant.py)

**Key concepts:**

*CLIP vs DINOv3:*
- CLIP (Contrastive Language-Image Pretraining) is trained on image-text pairs; its embeddings align visual content with language.
  Use CLIP when you want text queries like "military vehicle" to retrieve relevant frames.
- DINOv3 is trained with self-supervised vision-only signals; its embeddings capture fine-grained visual texture, structure, and part similarity better than CLIP.
  Use DINO when you want "find more visually similar frames" without a text description.
- The pipeline supports both simultaneously: Qdrant named vectors (`clip`, `dino`) hold each separately.

*Cosine similarity:*
Embeddings are L2-normalized before storage. Retrieval uses cosine distance (1 - dot product).
A distance near 0 means nearly identical. Near 1 means very different. Near 2 means polar opposites (rare in practice).

*In-memory fallback:*
When Qdrant is not running, an `InMemoryStore` is used. It behaves identically for search but does not persist between runs.

**Output artifact:**
- Qdrant collection with one point per frame. Each point has a payload with `frame_path`, `t_sec`, `video_id`, `caption`, etc.
- Or: in-memory numpy array of shape `(n_frames, embedding_dim)`.

**What a human should focus on:**
- CLIP vs DINO roles and when each one retrieves better results.
- What happens to search quality if you change the OpenCLIP model variant (model size vs retrieval quality tradeoff).
- Nearest-neighbor retrieval — why cosine similarity is preferred over Euclidean distance for high-dimensional embeddings.
- The difference between Qdrant and in-memory fallback for a dev run.

**Common failure modes:**
- GPU OOM during batch embedding → embedder silently switches to CPU or crashes; check VRAM.
- Wrong `OPENCLIP_PRETRAINED` value → model loads but embeddings are from a different pretrained weight; retrieval looks random.
- Qdrant collection schema mismatch after switching `MODEL_NAME` → old and new embeddings are mixed; run `scripts/reset_qdrant.sh` first.

---

<a id="step-3-gemma-multimodal-analysis"></a>
## Step 3. Gemma multimodal analysis

**What it does:**
Sample up to 30 frames from the video, run them through Gemma to produce:
- Zero-shot scene classification (what kind of environment is this?)
- Scene change detection (when does the visual content shift significantly?)
- Scene clustering (how many visually distinct segments exist?)
- Text-image retrieval probes (how well does the embedding space match scene descriptions?)

The results are deposited into `VideoKnowledge` so later steps can use them as domain context.

**Why it matters:**
This upgrades the pipeline from "a bag of frames" to "a scene-aware memory".
Without Gemma analysis, Qwen and Florence see no scene context and must reason entirely from the current frame.
With Gemma analysis, they receive a domain hint like `"Dominant scene: road | Known objects: truck, car | Visual transitions: 3"`.

**Implementation:**
- [`models/gemma_model.py`](../../models/gemma_model.py) — `GemmaEmbedder`
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
- [`pipeline/workflows/local/_common.py`](../../pipeline/workflows/local/_common.py) — `VideoKnowledge.add_gemma()`

**Key concepts:**

*Multimodal prompting:*
Gemma receives images and text in a single prompt. The model sees sampled frames alongside a structured question.
This is different from Florence (which produces a fixed caption) or CLIP (which only computes a similarity score).

*Scene change threshold (`_SCENE_CHANGE_THRESH = 0.25`):*
Scene changes are detected by comparing DINO embeddings of successive sampled frames.
If cosine distance exceeds 0.25, a transition is logged.
Tune this if your mission has rapid camera cuts (lower threshold) or very slow pan (higher threshold).

*Zero-shot classification:*
The pipeline uses a fixed set of text probes (e.g., "road or highway from above", "military vehicle or equipment") and finds which probe has highest CLIP similarity to the sampled frames.
The winner becomes `scene_type` in `VideoKnowledge`.

**Output artifact:**
- `gemma_analysis.md` in the video output directory: a structured Markdown report of scene classifications, transition timestamps, cluster summaries, and retrieval probe scores.

**What a human should focus on:**
- What scene changes Gemma detected vs what you can see visually when stepping through frames.
- Whether the dominant scene type makes sense for the mission.
- How the domain hint changes later Qwen outputs — compare Qwen captions from a run with Gemma vs without.
- What Gemma adds that CLIP and DINO alone do not: language-grounded reasoning, not just vector similarity.

**Common failure modes:**
- Gemma API is unreachable → `VideoKnowledge.scene_type` stays empty → domain hints are blank → Qwen quality degrades.
- Sampled 30 frames are all from the same scene (e.g., a static camera) → all transitions missed, cluster count is 1.
- Text probes do not match your mission domain → zero-shot classification gives wrong `scene_type` → domain hints mislead later steps.

---

<a id="step-4-florence-scene-captioning"></a>
## Step 4. Florence scene captioning

**What it does:**
Run Microsoft Florence-2 on each keyframe and produce a short natural-language description of the visual content.

**Why it matters:**
This is the first human-readable semantic layer.
Every later reasoning step that needs "what is happening visually" can use these captions as context rather than re-running a heavy model on raw pixels.
The captions are also how the pipeline builds scene segments (groups of temporally adjacent frames with similar content).

**Implementation:**
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
- [`pipeline/vision/florence.py`](../../pipeline/vision/florence.py)
- [`pipeline/vision/factory.py`](../../pipeline/vision/factory.py)

**Key concepts:**

*Florence-2 architecture:*
Florence-2 is a vision-language model trained on a unified task formulation.
You give it a task token like `<MORE_DETAILED_CAPTION>` and it produces captions conditioned on that token.
It can also do detection, grounding, and segmentation tasks with different tokens.

*Domain hints:*
The pipeline prefixes Florence prompts with a domain hint from `VideoKnowledge.domain_hint()`.
If Gemma already identified "urban environment", Florence receives that context and produces more relevant captions than it would from a blank prompt.

*Caption drift:*
Similar frames often get slightly different captions because small visual changes shift Florence's attention.
This is expected but means you cannot rely on exact string matching to compare frames. Use embedding similarity instead.

*Jaccard segment analysis:*
After captioning, `_analyze_caption_sequence()` computes Jaccard token overlap between adjacent captions.
When overlap drops below 0.45, a new scene segment starts.
This is the `segment_id` and `is_new_segment` fields attached to each caption.

**Output artifact:**
- `scene_captions.md` in the video output directory: one caption per frame with timestamp, segment ID, and scene transition markers.

**What a human should focus on:**
- Read ten captions and compare them to what you actually see in the corresponding frames.
- Find a frame where the caption is clearly wrong and explain why Florence failed there.
- Look at segment boundaries and check whether they correspond to real scene changes.
- Compare captions from a run with a domain hint vs a run without to see the difference.

**Common failure modes:**
- Florence OOM on a long video → `FLORENCE_BATCH_SIZE` auto-falls back to 1 (slow but correct).
- Domain hint is wrong (bad Gemma classification) → captions are plausible but off-domain.
- Very dark, blurry, or motion-streaked frames → captions are generic ("outdoors", "a scene").
- Segment boundary fires too often on a slowly panning camera → too many micro-segments.

---

<a id="step-5-asr-transcription"></a>
## Step 5. ASR transcription

**What it does:**
Extract the audio track from the video, run it through a speech recognition model (Whisper), and produce a timestamped subtitle map.

**Why it matters:**
Speech often contains mission context that never appears visually:
pilot callouts, operator radio communications, GPS coordinates announced aloud, warning alerts, mission objectives.
Without ASR, the pipeline is blind to everything spoken.

**Implementation:**
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
- [`pipeline/vision/asr.py`](../../pipeline/vision/asr.py)
- [`pipeline/media/audio.py`](../../pipeline/media/audio.py)

**Key concepts:**

*Whisper and the VAD (Voice Activity Detection) tradeoff:*
Whisper transcribes the full audio stream.
For mission videos with long silent passages, VAD preprocessing removes silence first, which speeds up transcription significantly.
The trade-off: VAD may clip the start or end of short speech segments, causing timestamp drift.

*Timestamp alignment:*
ASR produces word-level or segment-level timestamps.
These are aligned to `t_sec` frame timestamps so `VideoKnowledge.add_asr()` can look up what was said near any frame.
The lookup window is ±2 seconds by default.

*Language and accent:*
Whisper handles multilingual audio but switches language mode based on the first 30 seconds.
Mission videos with language-switching (e.g., bilingual radio traffic) may get wrong transcript segments.

**Output artifact:**
- `asr_subtitles.md` in the video output directory: a timestamped list of speech segments.

**What a human should focus on:**
- Find a frame where speech clarifies something ambiguous in the image.
- Find a case where background noise or propeller sound causes ASR hallucinations.
- Check whether timestamps in the subtitle map align with what you hear when playing the video.
- Look at what Qwen does with audio context vs without: read two `detailed_captions.md` entries for the same frame.

**Common failure modes:**
- No audio track in the video → `asr_subtitles.md` is empty; pipeline continues normally.
- Video codec has audio in an unsupported format → FFmpeg audio extraction fails silently.
- High background noise (rotor wash, wind, engine) → Whisper produces hallucinated words or repetitive filler.
- Very long silence → Whisper sometimes inserts placeholder text ("Thank you for watching.") for no-speech regions.

---

<a id="step-6-ocr-text-extraction"></a>
## Step 6. OCR text extraction

**What it does:**
Run an OCR model over each keyframe to extract machine-readable text from the image.

**Why it matters:**
OCR often changes a vague visual interpretation into a precise one.
A frame showing "Sector 7 Checkpoint" resolves location ambiguity that no caption model can infer from pixels.
HUD overlays, road signs, building names, vehicle markings, and instrument panels are all invisible to embedding models but readable by OCR.

**Implementation:**
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
- [`pipeline/vision/ocr.py`](../../pipeline/vision/ocr.py)

**Key concepts:**

*Scene text vs document text:*
OCR trained on document scans (aligned, clean, high-contrast) performs poorly on scene text (perspective-distorted, low-contrast, partial, stylized).
The pipeline uses a scene-text OCR model. Expect better results on road signs and overlays than on distant billboard text.

*Text confidence filtering:*
Most OCR outputs include a per-character or per-word confidence score.
Low-confidence reads (garbled words, partial matches) are filtered before being added to `VideoKnowledge`.

**Output artifact:**
- `ocr_results.md` in the video output directory: one entry per frame with extracted text and timestamps.

**What a human should focus on:**
- Find a frame where OCR text changes the meaning of the scene.
- Find a failure case: blurred text, small text, unusual font, or angled sign.
- Check whether OCR output is injected into the Qwen context for that frame (`context_for_frame()` includes `[Visible text]` lines).
- Compare frames with and without text and see if Qwen reasoning differs.

**Common failure modes:**
- Text too small for resolution → OCR returns empty or garbled characters.
- Perspective distortion on signs viewed at angle → character recognition breaks down.
- HUD overlay with non-standard font → font recognition fails.
- OCR model detects noise as characters → produces garbage strings that pollute Qwen context.

---

<a id="step-7-depth-estimation"></a>
## Step 7. Depth estimation

**What it does:**
Run a monocular depth estimation model (e.g., Depth-Anything, MiDaS) on each keyframe and produce a compact geometric summary:
`near_ratio` (fraction of pixels classified as close), `mean_depth`, and a coarse zone label.

**Why it matters:**
This gives the pipeline an approximate 3D prior before any explicit geometry exists.
A deep scene (large mean depth, low near ratio) implies open terrain or long-range observation.
A shallow scene (high near ratio) implies close-range inspection or a cluttered space.
This prior guides Qwen's spatial reasoning and flags frames where real SfM mapping may be critical.

**Implementation:**
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
- [`pipeline/vision/depth.py`](../../pipeline/vision/depth.py)

**Key concepts:**

*Relative vs metric depth:*
Monocular depth models produce relative depth maps: they can tell you "this pixel is closer than that pixel" but not the actual distance in meters.
The pipeline uses these maps only for summary statistics (near/far ratio), not for absolute measurements.
For metric depth, a LiDAR point cloud or stereo camera is required (see Step 13).

*Depth zone summary:*
Rather than storing a full depth map (expensive), the pipeline computes:
- `near_ratio`: fraction of pixels below a depth threshold
- `mean_depth`: average normalized depth value
- A coarse label: `open`, `cluttered`, or `close-range`

These three values are small enough to include in the Qwen context string for every frame.

**Output artifact:**
- `depth_summary.md` in the video output directory (or equivalent JSON): per-frame depth statistics.

**What a human should focus on:**
- Look at frames labeled `open` vs `cluttered` and confirm the labels match your visual impression.
- Find a frame where monocular depth is clearly wrong (e.g., a flat texture that looks far but is close, or glass surface).
- Check where depth context appears in a `detailed_captions.md` entry: `[Depth profile]: near_ratio=0.12 mean=0.78`.
- Understand why monocular depth is a prior, not a measurement: it can guide but not replace geometry.

**Common failure modes:**
- Fisheye or ultra-wide lens → depth model distortion artifacts at image edges.
- Uniform texture (flat ground, still water, concrete) → depth model hallucinates structure.
- High-altitude drone footage → everything looks equidistant; near/far ratio is uniformly low.
- Reflected surface (glass, water) → depth model sees the reflection as real geometry.

---

<a id="step-8-object-detection"></a>
## Step 8. Object detection

**What it does:**
Run a detection model over each keyframe to find labeled bounding boxes around objects.
Detected labels are collected into an entity inventory in `VideoKnowledge`.

**Why it matters:**
This is the first explicit object-structured representation.
Where CLIP captures "what does this scene look like?", detection captures "what distinct objects are here and approximately where?"
The entity inventory from this step enriches the domain hint passed to Qwen.

**Implementation:**
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
- [`pipeline/vision/detection.py`](../../pipeline/vision/detection.py)

**Key concepts:**

*Fixed vocabulary vs open vocabulary:*
Standard YOLO-family detectors use a fixed class list (COCO 80 classes, or domain-specific variants).
Open-vocabulary detectors (DETIC, GroundingDINO) can detect anything described in text.
The pipeline uses a configurable detector: check `detection.py` for which model is active.

*Entity inventory:*
`VideoKnowledge.add_detections()` accumulates all labels seen across all frames and sorts them by frequency.
The top 15 entities become `known_entities` and appear in the domain hint as "Known objects: truck, car, person…".
Later steps (Qwen, UniDriveVLA) receive this as world context.

*Confidence threshold:*
Low-confidence detections are filtered before storage.
Too high a threshold misses real objects. Too low a threshold floods the entity list with noise.

**Output artifact:**
- `detection_results.md` or per-frame JSON with bounding boxes, labels, and confidence scores.

**What a human should focus on:**
- Check whether the entity inventory matches what you see in the video.
- Find a missed detection (false negative) and explain why the detector failed.
- Find a false detection and note whether it corrupts the domain hint.
- Compare the fixed-vocabulary detector results to what an open-vocabulary detector would find for your domain.

**Common failure modes:**
- COCO-trained detector on aerial footage → misclassifies vehicles or misses small objects.
- Crowd of similar objects → NMS suppresses valid detections as duplicates.
- Partial occlusion or truncated frame edge → detection box is clipped or missed.
- Very small objects in high-altitude frames → detector operates at too low a resolution.

---

## End Of Phase: What You Should Understand

After Steps 1-8, a human should be able to answer:

- What frames exist and at what temporal resolution?
- How are they indexed and how does retrieval work?
- What does the system think is happening in language?
- What speech and visible text were present?
- What is near vs far in each scene?
- Which objects were detected and how does that feed into context?

If you cannot answer those questions, do not move to the sensor and fusion phase yet.

The clearest test: take one frame from your mission output and reconstruct the full context string that Qwen would receive for that frame.
It should contain a Florence caption, a scene segment, any ASR text, any OCR text, depth statistics, and detected objects.
If any of those lines is missing or clearly wrong, trace it back to the step that produced it.

## Related Docs

- [Runtime and study guide](01_runtime_and_study_guide.md)
- [Sensors and fusion: Steps 9-20](03_sensor_steps_09_20.md)
- [Agentic knowledge flow](06_agentic_knowledge_flow.md)
