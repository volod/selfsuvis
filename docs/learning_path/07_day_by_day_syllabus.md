# Day-By-Day Syllabus

A realistic 28-day study plan for a human who wants to understand the local pipeline deeply.
The first 21 days build understanding from foundation to advanced.
Days 22-28 are a practical application week: re-run, write, and verify.

**How to use this:**
- Do not skip days within a week; each day builds on the previous.
- Do skip weeks if you already have strong background in that domain (e.g., skip Week 2 if you have worked with sensors before).
- Every exercise is tied to a real artifact or code file. If you cannot find the artifact, the exercise is telling you something ran or was configured incorrectly.

---

## Week 1: Build The Base Mental Model

**Prerequisites:** Python familiarity, basic ML vocabulary (embedding, loss, model).
No specialized knowledge required.

---

### Day 1 — Orientation

**Topics:**
- Read `README.md` end-to-end.
- Read [`local_path.md`](../local_path.md): the 35-step table.
- Inspect one output directory from a previous run (or ask someone to share one).

**Exercise:**
List every artifact you find in the output directory and guess which step produced it.
Check against the pipeline architecture in `CLAUDE.md`.

**Concept checkpoint:**
Can you describe the pipeline in two sentences — what goes in and what comes out?

---

### Day 2 — Frame Extraction (Step 1)

**Topics:**
- Step 1: how FFmpeg extracts frames from video.
- FPS, keyframes, and video container timestamps.
- Read [`pipeline/media/frames.py`](../../pipeline/media/frames.py).

**Pre-reading:**
- FFmpeg documentation on `-r` flag and `-vf select` filter.
- What is a GOP (Group of Pictures) and why do I-frames matter?

**Exercise:**
Open `frames_metadata.json` from an output directory.
Count the frames. Compute the actual sampling rate (frame_count / duration_sec).
Does it match the expected FPS? If not, find out why.

**Concept checkpoint:**
What is the difference between decoding every frame and decoding only I-frames?
What information do you lose by decoding only I-frames?

---

### Day 3 — Embeddings And Vector Search (Step 2)

**Topics:**
- Step 2: CLIP and DINOv3 embeddings.
- Cosine similarity: what it measures and why it is preferred over Euclidean distance for embeddings.
- Qdrant vs in-memory fallback.
- Read [`models/openclip_model.py`](../../models/openclip_model.py).

**Pre-reading:**
- "CLIP: Learning Transferable Visual Models From Natural Language Supervision" — read the Abstract and Introduction.
- What is a contrastive objective? What does "image-text alignment" mean geometrically?

**Exercise:**
Inspect retrieval neighbors for five frames from an output directory.
For each, note: do the top-3 neighbors make sense? Find one case where the retrieval is wrong and explain why.

**Concept checkpoint:**
Why does CLIP align images and text in the same space?
Why does DINOv3 not use text? What does this difference mean for retrieval?

---

### Day 4 — Gemma Multimodal Analysis (Step 3)

**Topics:**
- Step 3: Gemma scene classification, change detection, and scene clustering.
- `VideoKnowledge.add_gemma()` and `domain_hint()`.
- Read [`pipeline/workflows/local/_common.py`](../../pipeline/workflows/local/_common.py) — the `VideoKnowledge` class.

**Pre-reading:**
- What is zero-shot classification? How does CLIP-based zero-shot classification work?
- What is the difference between a text probe and a learned classifier?

**Exercise:**
Open `gemma_analysis.md` from an output directory.
What scene type did Gemma classify?
How many transitions were detected?
Do the transition timestamps correspond to visible scene changes when you step through the frames?

**Concept checkpoint:**
What is the `domain_hint()` string and which later step uses it?
What happens if Gemma is unavailable — what exactly goes blank?

---

### Day 5 — Florence Captioning (Step 4)

**Topics:**
- Step 4: Florence-2 architecture, task tokens, and prompt-conditioned captioning.
- Caption drift and scene segmentation via Jaccard overlap.
- Read [`pipeline/vision/florence.py`](../../pipeline/vision/florence.py).

**Pre-reading:**
- What is a task token in a unified VLM formulation?
- What is Jaccard similarity? How is it used for token overlap?

**Exercise:**
Choose ten frames from `scene_captions.md`.
Write your own captions for each frame (one sentence each).
Compare your captions to Florence captions. Note: where does Florence succeed? Where does it fail?

**Concept checkpoint:**
What is a scene segment boundary in this pipeline?
How is it computed? What threshold governs it?

---

### Day 6 — ASR And OCR As Non-Visual Evidence (Steps 5-6)

**Topics:**
- Step 5: Whisper speech recognition, timestamps, and the VAD tradeoff.
- Step 6: OCR for scene text, confidence filtering, and failure modes.
- Read [`pipeline/vision/asr.py`](../../pipeline/vision/asr.py).

**Pre-reading:**
- What is Voice Activity Detection (VAD)? Why is it used before ASR?
- What is the difference between scene text OCR and document OCR?

**Exercise:**
Find one frame in your output where speech or visible text is present.
Trace how that text appears in the `context_for_frame()` string for that frame's timestamp.
Look at the corresponding Qwen output — did it use the text?

**Concept checkpoint:**
What is the ASR lookup window (±N seconds)?
Why does a too-wide window cause "context contamination"?

---

### Day 7 — Depth And Detection As Geometric Structure (Steps 7-8)

**Topics:**
- Step 7: monocular depth estimation — relative vs metric depth, depth zones, failure modes.
- Step 8: object detection — fixed vs open vocabulary, NMS, entity inventory.
- Read [`pipeline/vision/depth.py`](../../pipeline/vision/depth.py) and [`pipeline/vision/detection.py`](../../pipeline/vision/detection.py).

**Pre-reading:**
- What is monocular depth estimation? Why is it inherently ambiguous (scale ambiguity)?
- What is Non-Maximum Suppression (NMS)? When does it remove the wrong box?

**Exercise:**
Pick five frames with interesting scene content.
Find the depth summary for each in the output.
Classify each as: depth estimate is reasonable / depth estimate is misleading. Explain your reasoning.
Then look at the entity inventory in `VideoKnowledge` and check whether it matches what you see in the video.

**Concept checkpoint:**
Why is the depth estimate called a "prior" not a "measurement"?
What distinguishes it from a LiDAR range measurement?

---

## Week 2: Learn The Sensor Expansion

**Prerequisites:** Week 1 complete. Basic physics helpful but not required.

---

### Day 8 — RF / SDR Sensing (Step 9)

**Topics:**
- Step 9: IQ data, spectrograms, SNR, spectral flatness, occupied bandwidth.
- Read [`pipeline/vision/rf_analyzer.py`](../../pipeline/vision/rf_analyzer.py).

**Pre-reading:**
- What is a complex signal? What are the I and Q components?
- How does a short-time Fourier transform produce a spectrogram?

**Exercise:**
Look at the RF-related fields in an output JSON (if available).
Explain each field in one plain-language sentence: `center_freq`, `bandwidth`, `snr_db`, `flatness`.
If no RF sidecar was present in your run: describe what you would expect to see in a drone control-link interference scenario.

**Concept checkpoint:**
What does "spectral flatness near 1" indicate about the signal type?
What is the physical interpretation of a peak in the spectrogram?

---

### Day 9 — Thermal, Multispectral, Event Cameras (Steps 10-12)

**Topics:**
- Step 10: LWIR thermal — emissivity, radiometric vs non-radiometric, thermal contrast.
- Step 11: multispectral — spectral indices (NDVI, NDWI), band registration.
- Step 12: event cameras — asynchronous events, time surface, no-motion blind spot.

**Pre-reading:**
- What is emissivity? Why does a metal surface appear cold in LWIR even when hot?
- What is NDVI and what does a value above 0.4 indicate?

**Exercise:**
For each of Steps 10, 11, 12: write one paragraph answering:
- What does this sensor measure that RGB cannot?
- What is its primary failure mode?
- When would you want this sensor on a mission?

**Concept checkpoint:**
Why is a static scene invisible to an event camera?
What information is lost vs gained compared to a regular frame camera at the same frame rate?

---

### Day 10 — LiDAR, Radar, GNSS-R (Steps 13-15)

**Topics:**
- Step 13: LiDAR — time-of-flight ranging, point cloud density, calibration to RGB.
- Step 14: Radar — FMCW chirp, Doppler velocity, range-Doppler maps, CFAR.
- Step 15: GNSS-R and satellite-derived feeds — DOP, GNSS accuracy, AIS/ADS-B.

**Pre-reading:**
- How does time-of-flight laser ranging work? What limits range resolution?
- What is the Doppler effect and how does radar measure velocity from it?
- What does HDOP mean and what is a "good" value?

**Exercise:**
Write one paragraph per modality: what can LiDAR tell you that monocular depth cannot, what can radar tell you that LiDAR cannot, and what can GNSS tell you that neither can.
Focus on the unique information each provides, not their operating principles.

**Concept checkpoint:**
Why does radar perform better than LiDAR in fog?
What physical property of the scatterers matters?

---

### Day 11 — Inertial, Atmospheric, Chemical, Acoustic (Steps 16-19)

**Topics:**
- Step 16: IMU — accelerometer drift, gyroscope bias, complementary filter, barometric altitude.
- Step 17: atmospheric sensing — turbulence effects, sensor degradation from weather.
- Step 18: chemical/gas/radiation sensing — electrochemical vs MOS vs NDIR, geo-tagging.
- Step 19: acoustic sensing — MFCCs, event detection vs scene classification, propeller noise masking.

**Pre-reading:**
- What is sensor bias drift? Why does integrating a biased IMU produce a position error that grows over time?
- What is a MFCC and how does it represent audio?

**Exercise:**
For the mission type you are most likely to work with (drone, vehicle, or fixed sensor), identify which of Steps 16-19 would be most valuable and explain why.
Then identify which would be hardest to integrate and explain the practical obstacle.

**Concept checkpoint:**
What is the difference between acoustic event detection and acoustic scene classification?
Give one example of each that would be useful in a real mission.

---

### Day 12 — Sensor Fusion (Step 20)

**Topics:**
- Step 20: timestamp alignment, lag tolerance, contradiction detection, missing data handling.
- Read the `VideoKnowledge` class again with sensor fusion in mind.
- Read [`pipeline/workflows/local/_common.py`](../../pipeline/workflows/local/_common.py) — `context_for_frame()`.

**Pre-reading:**
- What is a timestamp alignment window and why is ±2 seconds a reasonable default for 30 fps video?
- What is the difference between sensor fusion and sensor integration?

**Exercise:**
Reconstruct the full `context_for_frame()` string for one specific frame from your output.
Identify which line came from which sensor.
Then: intentionally set each line to "absent" (pretend that step failed).
For which absent line does the Qwen output degrade most?

**Concept checkpoint:**
What does "contradiction between modalities" look like in practice?
Give one realistic example where LiDAR and monocular depth would give opposite signals.

---

## Week 3: Structure, Adaptation, And Audit

**Prerequisites:** Weeks 1-2 complete. Familiarity with PyTorch helpful for Days 16-18.

---

### Day 13 — Segmentation, Tracking, Language-Guided Perception (Steps 21-22)

**Topics:**
- Step 21: YOLO vs RF-DETR, SAM box-prompted vs auto-mask, NMS, IoU as mask quality metric.
- Step 22: language-directed tracking, IoU-based greedy matching, track break conditions.
- Read [`pipeline/workflows/local/steps_yolo_sam.py`](../../pipeline/workflows/local/steps_yolo_sam.py).

**Pre-reading:**
- What is Intersection over Union (IoU)? How is it used for both detection quality and tracking matching?
- What is the Segment Anything Model (SAM) and what kinds of prompts does it accept?

**Exercise:**
Open `gemma_tracking_summary.md` from an output directory.
Identify the categories Gemma requested tracking for.
Confirm whether those categories are present in the video.
Find one track that broke (IDs reset) and explain why based on the visual content around that timestamp.

**Concept checkpoint:**
Why does IoU-based tracking break for fast-moving objects?
What alternative tracking method would handle fast motion better?

---

### Day 14 — Temporal Embeddings, Qwen, UniDriveVLA (Steps 23-25)

**Topics:**
- Step 23: clip-level video embeddings vs frame embeddings — what the temporal axis adds.
- Step 24: Qwen multimodal prompt construction, rolling state, JSON parse failure handling.
- Step 25: VLA models vs VLM models, domain-specific vs general reasoning.
- Read [`pipeline/vision/qwen.py`](../../pipeline/vision/qwen.py).

**Pre-reading:**
- What is an exponential moving average? How is it used in the EMA teacher model?
- What is a VLA (Vision-Language-Action) model and how does it differ from a VLM?

**Exercise:**
Compare `detailed_captions.md` (Qwen) and `unidrive_analysis.md` (UniDriveVLA) for the same frame.
Find: one claim that only Qwen makes, one claim that only UniDriveVLA makes, and one claim they agree on.
Evaluate which claim seems more reliable for each case.

**Concept checkpoint:**
What is the rolling state in `VideoKnowledge._last_qwen`?
How is it injected into the next frame's context?
What happens when Qwen returns malformed JSON?

---

### Day 15 — Search Tests And 3D Mapping (Steps 26-27)

**Topics:**
- Step 26: search test design — Precision@K, Recall@K, hard positives vs hard negatives.
- Step 27: SfM (Structure-from-Motion) requirements, 3DGS (Gaussian Splat), scale ambiguity.
- Read [`docs/gaussian_splat.md`](../gaussian_splat.md).

**Pre-reading:**
- What is Structure-from-Motion? What does it require from the input images?
- What is 3D Gaussian Splatting and how does it differ from NeRF and from traditional mesh reconstruction?

**Exercise:**
Run at least three search queries against your mission output.
Record top-5 results for each.
Identify: one retrieval that is clearly correct, one that is clearly wrong, and one that is ambiguous.
For the wrong result, explain whether the failure is a vocabulary problem, a perspective problem, or a domain problem.

**Concept checkpoint:**
Why does SfM require multi-view overlap?
What happens if the camera rotates without translating?

---

### Day 16 — SSL Fine-Tuning (Step 28)

**Topics:**
- Step 28: DINO self-supervised fine-tuning, student-teacher EMA, augmentation strategy.
- SSL gate: what it means when the gate triggers.
- Read [`pipeline/workflows/local/steps_ssl.py`](../../pipeline/workflows/local/steps_ssl.py) and [`pipeline/training/ssl.py`](../../pipeline/training/ssl.py).

**Pre-reading:**
- What is self-supervised learning? How does it differ from supervised learning?
- What is an exponential moving average (EMA) teacher and why does it produce more stable targets than copying student weights directly?

**Exercise:**
Inspect the loss curve from an SSL run (ASCII sparkline in the logs or the `loss_history` JSON field).
Classify the curve: converging, stuck, or oscillating.
Explain what each pattern indicates about the learning rate or data quality.

**Concept checkpoint:**
Why does DINO not need labels?
What provides the training signal if no annotations exist?

---

### Day 17 — Distillation And Export (Steps 29-30)

**Topics:**
- Step 29: soft targets vs hard targets, temperature scaling, teacher-student capacity gap.
- Step 30: ONNX tracing vs scripting, dynamic axes, gallery build.
- Read [`pipeline/workflows/local/steps_distill.py`](../../pipeline/workflows/local/steps_distill.py).

**Pre-reading:**
- What is knowledge distillation? What is "dark knowledge" in the context of soft targets?
- What is ONNX and what problem does it solve for model deployment?

**Exercise:**
Read `student_model.pt` training details from the logs.
Compare the student embedding dimension to the teacher embedding dimension.
Explain: what information must the student preserve to perform well on retrieval? What can it safely compress?

**Concept checkpoint:**
What is the practical limit on how small you can make the student before distillation quality degrades significantly?
What metric would you use to measure whether the student preserved the teacher's structure?

---

### Day 18 — Evaluation And Cross-Model Comparison (Steps 31-33)

**Topics:**
- Step 31: comparing baseline vs fine-tuned retrieval — P@K delta, rank shift, visual inspection.
- Step 32: model comparison dimensions — embedding distance, retrieval rank, subjective quality.
- Step 33: cross-model agreement, sources of disagreement, ensemble false confidence.
- Read `comparison.md` and `multi_model_comparison.md` from an output directory.

**Pre-reading:**
- What is Precision@K? What is Recall@K? When does one matter more than the other?
- What is an "ensemble" and why can unanimous ensemble agreement still be wrong?

**Exercise:**
Find the query with the largest positive delta (fine-tuned improved most) and the query with the largest negative delta (fine-tuned regressed most).
For both: look at the actual retrieved frames and explain the improvement or regression visually.
Then find the frame with lowest cross-model agreement in `multi_model_comparison.md` and inspect it manually.

**Concept checkpoint:**
How do you distinguish "the model improved on this query" from "the test set was too easy"?
What makes a well-designed retrieval test hard to game?

---

### Day 19 — Synthesis And Audit (Steps 34-35)

**Topics:**
- Step 34: synthesis structure, evidence sourcing, active learning tags, change detection.
- Step 35: provenance, silent failures, context contamination, audit trail.
- Read `agentic_flow.md` from an output directory.

**Pre-reading:**
- What is data provenance? Why does it matter for ML system reliability?
- What is a silent failure in a software system? How does it differ from an exception?

**Exercise:**
Read `agentic_flow.md` completely.
Identify every step that had status `skipped` or `failed`.
For each: trace what downstream effect that skip had on the synthesis.
Then: pick one claim from the synthesis and trace it backward through `VideoKnowledge` to its raw source.

**Concept checkpoint:**
What is the most dangerous type of error in this pipeline — a hard failure (exception) or a silent failure (wrong default)?
Explain your reasoning.

---

### Day 20 — End-To-End Review Run

**Topics:**
- End-to-end run of `python main.py --mode local` on a short video (1-5 minutes).

**Exercise:**
Before running: write down what you expect to find in the output for each of the 35 steps.
After running: open each artifact in the expected order and check your predictions.
Record: which predictions were right, which were wrong, and why.

**Focus areas:**
- Is `frames_metadata.json` complete?
- Does `gemma_analysis.md` show the right scene type?
- Does `agentic_flow.md` show any skipped or failed steps?
- Is the retrieval in `comparison.md` better after fine-tuning for at least one query?

---

### Day 21 — Consolidation

**Exercise:**
Write your own one-page explanation of the full local pipeline from memory.
Structure it as: input → evidence extraction → sensor expansion → structured reasoning → adaptation → audit → output.

Then: compare your write-up to [`local_path.md`](../local_path.md) and [`01_runtime_and_study_guide.md`](01_runtime_and_study_guide.md).
Note the gaps — these are the areas where you should go back and re-read.

---

## Week 4: Application And Depth (Optional)

**Audience:** Engineers who want to contribute to or extend the pipeline.
**Prerequisites:** Weeks 1-3 complete.

---

### Day 22 — Code Architecture Walkthrough

**Topics:**
- Full walkthrough of `pipeline/workflows/local/runner.py` as an orchestrator.
- How the `init_models()`, `_process_video()`, and per-step functions interact.
- Where each step's configuration comes from (`pipeline/config.py`).

**Exercise:**
Identify the three places in `runner.py` where a step failure is caught and swallowed.
For each: explain whether this is the right behavior or whether it should raise instead.

---

### Day 23 — Adding A New Step

**Exercise:**
Design a new hypothetical Step 36: "Crowd density estimation from aerial footage."
Write:
1. What it does (one paragraph).
2. What it deposits into `VideoKnowledge` (new field or existing field?).
3. Where in the runner it would be inserted.
4. What the `context_for_frame()` output line would look like.
5. What the audit step should record about it.

You do not need to implement it — just design it and verify the design is coherent with the existing structure.

---

### Day 24 — Failure Mode Inventory

**Exercise:**
Go through Steps 1-35 and for each step, write a one-sentence failure mode statement in this format:
> "If [condition], then [step] produces [output], which causes [downstream effect]."

Focus on silent failures (wrong output, no exception).
You should have 35 sentences by the end.
This is the most practical debugging reference you can produce.

---

### Day 25 — Custom Search Query Design

**Exercise:**
Design a retrieval evaluation suite for your specific mission domain (aerial survey, vehicle tracking, infrastructure inspection, etc.).

Include:
- 5 easy positive queries (a human would expect good results).
- 5 hard positive queries (domain-specific vocabulary, unusual viewpoint).
- 5 hard negative queries (semantically close but should not match your mission footage).
- Expected top-3 results for each query (based on your knowledge of the data).

Run the queries against your output and record where the baseline fails.
These failures are the learning opportunities for SSL fine-tuning (Step 28).

---

### Day 26 — VideoKnowledge Extension

**Exercise:**
Extend `VideoKnowledge` to store sensor fusion results from a new sensor type.
Choose one: RF detection events, LiDAR cluster count, or acoustic event timestamps.

For your chosen extension:
1. Add a `_new_field` dict and `_ts_new_field` list to the class.
2. Write a `add_new_sensor()` method.
3. Modify `context_for_frame()` to include a `[New sensor]` line.
4. Write one unit test for the new method.

Commit nothing — this is a study exercise in understanding the code structure.

---

### Day 27 — Cross-Mission Change Detection

**Topics:**
- `pipeline/change_detection.py`
- GPS bbox Qdrant filter
- Embedding distance threshold (`CHANGE_DETECTION_THRESHOLD`)
- `change_detections` PostgreSQL table

**Exercise:**
Run two missions that cover the same GPS area.
Find the change detection output in the synthesis report.
Evaluate: are the detected changes real or false alarms?
Identify the embedding distance threshold that would suppress the false alarms without missing the real changes.

---

### Day 28 — Architecture Review And Personal Next Steps

**Exercise:**
Write a personal review of the pipeline covering:
1. The three decisions you would change in the current architecture and why.
2. The two steps you understand least well and what you would need to read to understand them.
3. One extension you would add that is not already in the 35-step path.
4. Which step failure would be hardest to detect in a real deployment and why.

Use this review as a map for your ongoing learning.

---

## Summary: Study Milestones

| Milestone | Day | Indicator |
|-----------|-----|-----------|
| Can describe the pipeline end-to-end | 1 | After Day 1 |
| Can trace one frame through Steps 1-8 | 7 | After Day 7 |
| Can explain what each sensor adds | 12 | After Day 12 |
| Can reconstruct a full `context_for_frame()` string | 12 | After Day 12 |
| Can explain SSL and distillation without notes | 17 | After Day 17 |
| Can trace a synthesis claim to its raw source | 19 | After Day 19 |
| Can run a full pipeline and inspect all artifacts | 20 | After Day 20 |
| Can identify and explain any step's failure modes | 24 | After Day 24 |
| Can design a new step that fits the architecture | 23 | After Day 23 |
