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
- Read [`local_path.md`](../local_path.md): the fast study map of the current pipeline.
- Inspect one output directory from a previous run (or ask someone to share one).

**Exercise:**
List every artifact you find in the output directory and guess which step produced it.
Check against [`pipeline.md`](../pipeline.md) and [`architecture.md`](../architecture.md).

**Concept checkpoint:**
Can you describe the pipeline in two sentences — what goes in and what comes out?

---

### Day 2 — Frame Extraction (Step 1)

**Topics:**
- Step 1: how FFmpeg extracts frames from video.
- FPS, keyframes, and video container timestamps.
- Read [`pipeline/media/frames.py`](../../src/selfsuvis/pipeline/media/frames.py).

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
- Read [`models/openclip_model.py`](../../src/selfsuvis/models/openclip_model.py).

**Pre-reading:**
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, 2021) — Sections 1-3. [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
- Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (DINO, 2021) — Abstract and Section 3. [arxiv.org/abs/2104.14294](https://arxiv.org/abs/2104.14294)
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
- Read [`pipeline/workflows/local/_common.py`](../../src/selfsuvis/pipeline/workflows/local/_common.py) — the `VideoKnowledge` class.

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
- Read [`pipeline/vision/florence.py`](../../src/selfsuvis/pipeline/vision/florence.py).

**Pre-reading:**
- Xiao et al., "Florence-2" (2023) — Sections 1-3 (task formulation and FLD-5B dataset). [arxiv.org/abs/2311.06242](https://arxiv.org/abs/2311.06242)
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
- Read [`pipeline/vision/asr.py`](../../src/selfsuvis/pipeline/vision/asr.py).

**Pre-reading:**
- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper, 2022) — Abstract and Section 2.2 (multitask format). [arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)
- Li et al., "TrOCR" (2021) — Abstract and Section 2 (architecture). [arxiv.org/abs/2109.10282](https://arxiv.org/abs/2109.10282)
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
- Read [`pipeline/vision/depth.py`](../../src/selfsuvis/pipeline/vision/depth.py) and [`pipeline/vision/detection.py`](../../src/selfsuvis/pipeline/vision/detection.py).

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

### Interlude — Post-Run Artifact Analysis

**Topics:**
- Read [`analytics.md`](../analytics.md).
- Read [`08_local_run_artifact_analysis.md`](08_local_run_artifact_analysis.md).
- Learn how to move from a completed run directory back to the originating code.

**Exercise:**
Run the analytics CLI on one finished local run.
Write down every warning it emits, then verify each warning manually from the raw artifacts.

**Concept checkpoint:**
Why is a completed run not equivalent to a healthy run?
Which artifact family gives you the fastest signal that a stage silently degraded?

---

## Week 2: Learn The Sensor Expansion

**Prerequisites:** Week 1 complete. Basic physics helpful but not required.

---

### Day 8 — RF / SDR Sensing (Step 9)

**Topics:**
- Step 9: IQ data, spectrograms, SNR, spectral flatness, occupied bandwidth.
- Read [`pipeline/vision/rf_analyzer.py`](../../src/selfsuvis/pipeline/vision/rf_analyzer.py).

**Pre-reading:**
- West & O'Shea, "Deep Architectures for Modulation Recognition" (2017) — Abstract and Section 2 (IQ representation). [arxiv.org/abs/1703.09197](https://arxiv.org/abs/1703.09197)
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
- Gallego et al., "Event-based Vision: A Survey" (2022) — Sections 2-3 (event representations: time surfaces, voxel grids). [arxiv.org/abs/1904.08405](https://arxiv.org/abs/1904.08405)
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
- Read [09_sensor_fusion_fundamentals.md](09_sensor_fusion_fundamentals.md).
- Step 20: timestamp alignment, lag tolerance, contradiction detection, missing data handling.
- Read the `VideoKnowledge` class again with sensor fusion in mind.
- Read [`pipeline/workflows/local/_common.py`](../../src/selfsuvis/pipeline/workflows/local/_common.py) — `context_for_frame()`.

**Pre-reading:**
- Geneva et al., "OpenVINS: A Research Platform for Visual-Inertial Estimation" (2020) — Section 2 (EKF state representation and multi-sensor time alignment). [arxiv.org/abs/1908.01012](https://arxiv.org/abs/1908.01012)
- What is a timestamp alignment window and why is ±2 seconds a reasonable default for 30 fps video?
- What is the difference between sensor fusion and sensor integration?
- Review the five questions from the sensor-fusion fundamentals session:
  physical quantity, coordinate frame, clock base, uncertainty, and downstream consumer.

**Exercise:**
Reconstruct the full `context_for_frame()` string for one specific frame from your output.
Identify which line came from which sensor.
Then: intentionally set each line to "absent" (pretend that step failed).
For which absent line does the Qwen output degrade most?

**Concept checkpoint:**
What does "contradiction between modalities" look like in practice?
Give one realistic example where LiDAR and monocular depth would give opposite signals.
Why is the current `selfsuvis` local pipeline better described as context fusion than full probabilistic state fusion?

---

## Week 3: Structure, Adaptation, And Audit

**Prerequisites:** Weeks 1-2 complete. Familiarity with PyTorch helpful for Days 16-18.

---

### Day 13 — Segmentation, Tracking, Language-Guided Perception (Steps 21-22)

**Topics:**
- Step 21: YOLO vs RF-DETR, SAM box-prompted vs auto-mask, NMS, IoU as mask quality metric.
- Step 22: language-directed tracking, IoU-based greedy matching, track break conditions.
- Read [`pipeline/workflows/local/steps_yolo_sam.py`](../../src/selfsuvis/pipeline/workflows/local/steps_yolo_sam.py).

**Pre-reading:**
- Kirillov et al., "Segment Anything" (SAM, 2023) — Sections 2-3 (task formulation, promptable segmentation). [arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)
- Bewley et al., "SORT" (2016) — all four pages. This is the IoU matching algorithm used in `RFDETRTracker`. [arxiv.org/abs/1602.00763](https://arxiv.org/abs/1602.00763)
- What is Intersection over Union (IoU)? How is it used for both detection quality and tracking matching?

**Exercise:**
Open `gemma_tracking_summary.md` from an output directory.
Identify the categories Gemma requested tracking for.
Confirm whether those categories are present in the video.
Find one track that broke (IDs reset) and explain why based on the visual content around that timestamp.

**Concept checkpoint:**
Why does IoU-based tracking break for fast-moving objects?
What alternative tracking method would handle fast motion better?

---

### Day 14 — Temporal Embeddings, RSSM Surprise, Qwen, UniDriveVLA (Steps 23-25)

**Topics:**
- Step 23 part A: clip-level video embeddings vs frame embeddings — what the temporal axis adds.
- Step 23 part B: RSSM temporal surprise scoring (DreamerV3-inspired). How a lightweight GRU-based world model learns the mission's temporal rhythm and flags frames that break it.
- Step 24: Qwen multimodal prompt construction, rolling state, JSON parse failure handling.
- Step 25: VLA models vs VLM models, domain-specific vs general reasoning.
- Read [`models/rssm_model.py`](../../src/selfsuvis/models/rssm_model.py) and [`pipeline/vision/qwen.py`](../../src/selfsuvis/pipeline/vision/qwen.py).

**Pre-reading:**
- Hafner et al., "Learning Latent Dynamics for Planning from Pixels" (PlaNet, 2019) — Section 3 (RSSM architecture). [arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
- Romero et al., "Dream to Fly" (ICRA 2026) — Abstract and Section III. [rpg.ifi.uzh.ch/docs/ICRA26_Romero.pdf](https://rpg.ifi.uzh.ch/docs/ICRA26_Romero.pdf)
- Team Qwen, "Qwen2.5-VL Technical Report" (2025) — Section 3 (architecture). [arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923)
- What is a Recurrent State Space Model (RSSM)? What is the difference between the *posterior* (observed) and *prior* (predicted) latent?
- What is a VLA (Vision-Language-Action) model and how does it differ from a VLM?

**Exercise:**
Inspect `frame_facts_json["rssm"]["surprise_score"]` for a long mission.
Plot (or print) the surprise scores over time.
Identify: one cluster of high-surprise frames and one stretch of low-surprise frames.
Explain the visual content difference between the two clusters.

**Concept checkpoint:**
What does a high RSSM surprise score mean intuitively?
Why does the RSSM operate on CLIP embeddings rather than raw pixels?
What is the EMA fallback and when does it trigger?
What is the rolling state in `VideoKnowledge._last_qwen`?
How is it injected into the next frame's context?

---

### Day 15 — Search Tests And 3D Mapping (Steps 26-27)

**Topics:**
- Step 26: search test design — Precision@K, Recall@K, hard positives vs hard negatives.
- Step 27: SfM (Structure-from-Motion) requirements, 3DGS (Gaussian Splat), scale ambiguity.
- Read [`docs/gaussian_splat.md`](../gaussian_splat.md).

**Pre-reading:**
- Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields" (2020) — Sections 1-3 (implicit radiance field, volume rendering). [arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934)
- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (2023) — Sections 4-5 (3DGS representation and adaptive density control). [arxiv.org/abs/2308.04079](https://arxiv.org/abs/2308.04079)
- What does "scale ambiguity" mean in monocular SfM and how does GPS break it?

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
- RSSM-guided frame selection: how RSSM surprise scores from Step 23 influence which frames become contrastive pairs in SSL training.
- Read [`pipeline/workflows/local/steps_ssl.py`](../../src/selfsuvis/pipeline/workflows/local/steps_ssl.py) and [`pipeline/training/ssl.py`](../../src/selfsuvis/pipeline/training/ssl.py).

**Pre-reading:**
- Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (DINO, 2021) — Sections 3-4 (multi-crop and student-teacher EMA). [arxiv.org/abs/2104.14294](https://arxiv.org/abs/2104.14294)
- Ericsson et al., "Self-Supervised Representation Learning: Introduction, Advances, and Challenges" (2022) — Sections 2-3 (contrastive vs self-distillation). [arxiv.org/abs/2110.09327](https://arxiv.org/abs/2110.09327)
- What is an exponential moving average (EMA) teacher and why does it produce more stable targets than copying student weights directly?

**Exercise:**
Inspect the loss curve from an SSL run (ASCII sparkline in the logs or the `loss_history` JSON field).
Classify the curve: converging, stuck, or oscillating.
Explain what each pattern indicates about the learning rate or data quality.
Then compare: do the `needs_annotation` frames selected by the RSSM-enhanced AL score look more informative than frames selected by DINO distance alone?

**Concept checkpoint:**
Why does DINO not need labels?
What provides the training signal if no annotations exist?
How does RSSM surprise improve the quality of SSL contrastive pairs compared to random or DINO-only frame selection?

---

### Day 17 — Distillation And Export (Steps 29-30)

**Topics:**
- Step 29: soft targets vs hard targets, temperature scaling, teacher-student capacity gap.
- Step 30: ONNX tracing vs scripting, dynamic axes, gallery build.
- Read [`pipeline/workflows/local/steps_distill.py`](../../src/selfsuvis/pipeline/workflows/local/steps_distill.py).

**Pre-reading:**
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015) — all five pages. [arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)
- ONNX Runtime documentation quickstart: [onnxruntime.ai/docs/get-started/with-python.html](https://onnxruntime.ai/docs/get-started/with-python.html)
- What is "dark knowledge" in the context of soft targets?

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
- Musgrave et al., "A Metric Learning Reality Check" (2020) — Section 3 (evaluation pitfalls). [arxiv.org/abs/2003.08505](https://arxiv.org/abs/2003.08505)
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval* — Chapter 8 (P@K, MAP, nDCG). [nlp.stanford.edu/IR-book](https://nlp.stanford.edu/IR-book) — free.
- What is Precision@K? What is Recall@K? When does one matter more than the other?

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
- End-to-end run of `selfsuvis --mode local` on a short video (1-5 minutes).

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

## Week 5: Future Direction Track (Advanced)

**Audience:** Humans who already understand the current runtime and want to push
the system toward stronger self-supervised vision, physical-world modeling, and
realtime sensor-mesh threat analysis.

**Prerequisites:** Weeks 1-4 complete.

### Day 29 — Temporal Self-Supervised Vision

**Topics:**
- Re-read Step 23 (RSSM surprise) and Step 28 (SSL DINO fine-tuning) as one temporal-learning story.
- Read [15_future_directions_realtime_threat_analysis.md](15_future_directions_realtime_threat_analysis.md), Sections 2 and 10.
- Focus on track-aware, clip-aware, and cross-view self-supervision.

**Exercise:**
Pick one mission and list five failure cases where frame-only embeddings are too weak:
- viewpoint shift
- scale change
- occlusion
- motion blur
- repeated texture

For each case, write which temporal SSL signal would help most:
- track continuity
- clip prediction
- multiview consistency
- cross-modal consistency

**Concept checkpoint:**
Why is temporal SSL a better next step than just swapping in a larger image encoder?

### Day 30 — Cross-Modal Self-Supervision

**Topics:**
- Treat sensor agreement as a training signal, not just a report signal.
- Study RGB-depth, RGB-thermal, camera-IMU, and camera-radar consistency ideas.

**Exercise:**
Choose three modality pairs relevant to your mission domain.
For each pair, define:
1. what should agree physically
2. what typical disagreement means
3. whether that disagreement is more likely to indicate anomaly, calibration error, or sensor failure

**Concept checkpoint:**
What is the difference between using another sensor as a label source and using it as a noisy self-supervised constraint?

### Day 31 — Physical Models And Field Models

**Topics:**
- Move from semantic descriptions to state, flow, occupancy, and field estimates.
- Revisit the fusion docs with environmental fields in mind: wind, RF, plume, thermal spread, visibility.

**Exercise:**
Design one physical model for your mission type:
- plume spread
- RF interference field
- occupancy flow
- terrain traversability

Write:
1. the state variables
2. the measurements
3. the update cadence
4. the failure modes
5. the output artifact you would want in `selfsuvis`

**Concept checkpoint:**
Why are many important hazards better modeled as fields than as detected objects?

### Day 32 — Local Threat Inference

**Topics:**
- Define a platform-centered local threat window.
- Separate threat primitives from high-level threat labels.

**Exercise:**
For one platform (drone, vehicle, or fixed tower), define:
- 5 local threat primitives
- the evidence sources for each
- the confidence and freshness conditions required before escalation

Example primitives:
- collision risk
- pose loss
- RF jamming suspicion
- local gas hotspot
- hidden thermal agent

**Concept checkpoint:**
Why should local threat inference remain causal and low-latency instead of depending on a large reasoning model?

### Day 33 — Global Threat Aggregation

**Topics:**
- Sector-level risk maps, route advisories, and mission-wide hazard persistence.
- Temporal persistence and cross-node confirmation.

**Exercise:**
Design a simple global threat table with columns:
- sector
- threat type
- persistence score
- cross-sensor support
- confidence
- recommended action

Then describe how local threat outputs from multiple nodes would populate it.

**Concept checkpoint:**
Why is a global threat map mostly an aggregation and evidence-management problem, not a single-model prediction problem?

### Day 34 — Realtime Sensor-Mesh Architecture Proposal

**Topics:**
- Draft a concrete extension of `selfsuvis` from current fusion outputs toward realtime threat analysis.
- Use the proposed layers in [15_future_directions_realtime_threat_analysis.md](15_future_directions_realtime_threat_analysis.md).

**Exercise:**
Write a one-page architecture proposal with these layers:
1. ingest
2. representation
3. physical-state
4. threat primitive
5. local threat
6. global threat
7. audit

For each layer, specify:
- input
- output
- latency target
- one hard failure mode

**Concept checkpoint:**
Which layer is the first one that must be trustworthy enough for operator action?

### Day 35 — Personal Research And Build Plan

**Exercise:**
Write a concrete next-quarter plan for yourself:
1. one self-supervised vision improvement
2. one physical-model improvement
3. one realtime threat-analysis improvement
4. one evaluation protocol for each
5. one artifact or dashboard you would add for humans

Force yourself to choose only one item in each category.
Breadth is less useful than a coherent direction.

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
| Can explain a credible next-stage SSL direction | 29 | After Day 29 |
| Can define local vs global threat inference | 33 | After Day 33 |
| Can propose a realtime sensor-mesh architecture | 34 | After Day 34 |

---

## Consolidated Reading List

Organized by depth level. Read in this order if you are starting from scratch. Skip sections where you have strong background.

### Tier 1 — Foundational (read before Week 1)

These establish the mathematical and conceptual vocabulary used throughout the pipeline documentation.

| Resource | Covers | Where to get it |
|---|---|---|
| Goodfellow, Bengio & Courville, *Deep Learning* (2016) | CNNs, RNNs, representation learning, optimization | [deeplearningbook.org](https://www.deeplearningbook.org) — free |
| Prince, *Understanding Deep Learning* (2023) | Modern architectures, transformers, diffusion | [udlbook.github.io/udlbook](https://udlbook.github.io/udlbook) — free |
| Szeliski, *Computer Vision: Algorithms and Applications* (2022) | Feature extraction, SfM, stereo, optical flow | [szeliski.org/Book](https://szeliski.org/Book) — free |
| Thrun, Burgard & Fox, *Probabilistic Robotics* (2005) | Bayes filter, KF/EKF/UKF, particle filter, SLAM | University library or MIT Press |
| Settles, "Active Learning Literature Survey" (2009) | Uncertainty sampling, query-by-committee, BALD | [burrsettles.com/pub/settles.activelearning.pdf](http://burrsettles.com/pub/settles.activelearning.pdf) — free |

### Tier 2 — Core Papers (read during Weeks 1-3, one or two per day)

These are the papers that introduced the models and techniques used directly in the pipeline.

| Paper | Step(s) | arXiv |
|---|---|---|
| Vaswani et al., "Attention Is All You Need" (2017) | All transformer-based models | [1706.03762](https://arxiv.org/abs/1706.03762) |
| Dosovitskiy et al., "An Image is Worth 16×16 Words" (ViT, 2020) | CLIP, DINOv2, Florence-2 backbone | [2010.11929](https://arxiv.org/abs/2010.11929) |
| Radford et al., "Learning Transferable Visual Models" (CLIP, 2021) | Step 2 | [2103.00020](https://arxiv.org/abs/2103.00020) |
| Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (DINO, 2021) | Steps 2, 28 | [2104.14294](https://arxiv.org/abs/2104.14294) |
| Oquab et al., "DINOv2" (2023) | Steps 2, 28 | [2304.07193](https://arxiv.org/abs/2304.07193) |
| Xiao et al., "Florence-2" (2023) | Step 4 | [2311.06242](https://arxiv.org/abs/2311.06242) |
| Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper, 2022) | Step 5 | [2212.04356](https://arxiv.org/abs/2212.04356) |
| Li et al., "TrOCR" (2021) | Step 6 | [2109.10282](https://arxiv.org/abs/2109.10282) |
| Yang et al., "Depth Anything V2" (2024) | Step 7 | [2406.09414](https://arxiv.org/abs/2406.09414) |
| Lv et al., "RT-DETR" (2023) | Steps 8, 22 | [2304.08069](https://arxiv.org/abs/2304.08069) |
| Kirillov et al., "Segment Anything" (SAM, 2023) | Step 21 | [2304.02643](https://arxiv.org/abs/2304.02643) |
| Ravi et al., "SAM 2" (2024) | Step 21 | [2408.00714](https://arxiv.org/abs/2408.00714) |
| Bewley et al., "SORT" (2016) | Step 22 | [1602.00763](https://arxiv.org/abs/1602.00763) |
| Hafner et al., "Learning Latent Dynamics for Planning from Pixels" (PlaNet/RSSM, 2019) | Step 23 | [1811.04551](https://arxiv.org/abs/1811.04551) |
| Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3, 2023) | Step 23 | [2301.04104](https://arxiv.org/abs/2301.04104) |
| Romero et al., "Dream to Fly" (ICRA 2026) | Step 23 | [rpg.ifi.uzh.ch/docs/ICRA26_Romero.pdf](https://rpg.ifi.uzh.ch/docs/ICRA26_Romero.pdf) |
| Team Qwen, "Qwen2.5-VL Technical Report" (2025) | Step 24 | [2502.13923](https://arxiv.org/abs/2502.13923) |
| Mildenhall et al., "NeRF" (2020) | Step 27 | [2003.08934](https://arxiv.org/abs/2003.08934) |
| Kerbl et al., "3D Gaussian Splatting" (2023) | Step 27 | [2308.04079](https://arxiv.org/abs/2308.04079) |
| Chen et al., "SimCLR" (2020) | Step 28 — context | [2002.05709](https://arxiv.org/abs/2002.05709) |
| Grill et al., "Bootstrap Your Own Latent" (BYOL, 2020) | Step 28 / advanced SSL direction | [2006.07733](https://arxiv.org/abs/2006.07733) |
| He et al., "MAE" (2021) | Step 28 — context | [2111.06377](https://arxiv.org/abs/2111.06377) |
| Hinton et al., "Distilling the Knowledge in a Neural Network" (2015) | Step 29 | [1503.02531](https://arxiv.org/abs/1503.02531) |
| Gou et al., "Knowledge Distillation: A Survey" (2021) | Step 29 | [2006.05525](https://arxiv.org/abs/2006.05525) |

### Tier 3 — Deep Dives (read during Week 4 and beyond)

These go deeper into specific subsystems or provide the broader research context.

| Resource | Topic |
|---|---|
| Gallego et al., "Event-based Vision: A Survey" (2022) — [1904.08405](https://arxiv.org/abs/1904.08405) | Event cameras (Step 12) |
| Qi et al., "PointNet" (2017) — [1612.00593](https://arxiv.org/abs/1612.00593) | LiDAR 3D perception (Step 13) |
| Barfoot, *State Estimation for Robotics* (Cambridge, 2017) | IMU fusion, SE(3) pose (Steps 16, 27) |
| Hartley & Zisserman, *Multiple View Geometry in Computer Vision* (2004) | SfM fundamentals (Step 27) |
| Bommasani et al., "On the Opportunities and Risks of Foundation Models" (2021) — [2108.07258](https://arxiv.org/abs/2108.07258) | Architecture-level context for all steps |
| Musgrave et al., "A Metric Learning Reality Check" (2020) — [2003.08505](https://arxiv.org/abs/2003.08505) | Evaluation methodology (Steps 31-33) |
| Park et al., "Generative Agents" (2023) — [2304.03442](https://arxiv.org/abs/2304.03442) | Agent memory design (Step 35 / VideoKnowledge) |
| Mialon et al., "Augmented Language Models" (2023) — [2302.07842](https://arxiv.org/abs/2302.07842) | Tool use and retrieval in LLM systems |
| Nagel et al., "A White Paper on Neural Network Quantization" (2021) — [2106.08295](https://arxiv.org/abs/2106.08295) | INT8 quantization for ONNX export (Step 30) |
| Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking and Navigation* (2001) | Multi-target tracking, IMM, data association for physical-model extensions |
| Yilmaz, Javed & Shah, "Object Tracking: A Survey" (2006) — [cs.rochester.edu/u/omer/PDFs/ObjectTracking.pdf](https://www.cs.rochester.edu/u/omer/PDFs/ObjectTracking.pdf) | Tracking design space before extending threat primitives |

### HuggingFace quick-reference

| Component | Documentation page |
|---|---|
| Transformers library | [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers) |
| CLIP | [/model_doc/clip](https://huggingface.co/docs/transformers/model_doc/clip) |
| DINOv2 | [/model_doc/dinov2](https://huggingface.co/docs/transformers/model_doc/dinov2) |
| Florence-2 | [huggingface.co/microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) |
| Whisper | [/model_doc/whisper](https://huggingface.co/docs/transformers/model_doc/whisper) |
| TrOCR | [/model_doc/trocr](https://huggingface.co/docs/transformers/model_doc/trocr) |
| DPT (depth) | [/model_doc/dpt](https://huggingface.co/docs/transformers/model_doc/dpt) |
| Depth Anything V2 | [depth-anything/Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large) |
| RT-DETR | [/model_doc/rt_detr](https://huggingface.co/docs/transformers/model_doc/rt_detr) |
| SAM2 | [/model_doc/sam2](https://huggingface.co/docs/transformers/model_doc/sam2) |
| VideoMAE | [/model_doc/videomae](https://huggingface.co/docs/transformers/model_doc/videomae) |
| Qwen2.5-VL | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Optimum (ONNX) | [huggingface.co/docs/optimum](https://huggingface.co/docs/optimum) |
| Datasets | [huggingface.co/docs/datasets](https://huggingface.co/docs/datasets) |
