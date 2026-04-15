# Adaptation, Evaluation, And Audit: Steps 28-35

This phase asks the practical engineering questions:

- Can the model adapt to mission-specific data without labels?
- Can it be compressed for edge deployment?
- Did it actually improve, or just change?
- Can a human audit and trust the result?

The key discipline in this phase is **separating adaptation from evaluation**.
You cannot evaluate improvement with the same process that produced the improvement.
Steps 28-31 are the adapt-then-measure loop.
Steps 32-35 are the human-facing outputs of that loop.

---

<a id="step-28-ssl-dino-fine-tuning"></a>
## Step 28. SSL DINO fine-tuning

**What it does:**
Run self-supervised fine-tuning of the DINOv3 backbone on the mission frames extracted in this run.
No labels are required: DINO uses self-distillation between an online (student) and exponential-moving-average (teacher) network.
The result is a backbone whose representations are better calibrated to this specific mission's visual distribution.

**Why it matters:**
DINOv3 pretrained on internet images may encode features useful for general photography but not for specific mission content.
A mission with industrial equipment, military vehicles, or unusual terrain is far from the DINO pretraining distribution.
Self-supervised fine-tuning narrows this distribution gap using the mission data itself, without requiring any human labels.
Even 20-50 training epochs on a single mission's frames often produces measurable improvement in retrieval recall.

**Implementation:**
- [`pipeline/workflows/local/steps_ssl.py`](../../pipeline/workflows/local/steps_ssl.py)
- [`pipeline/training/ssl.py`](../../pipeline/training/ssl.py)

**Key concepts:**

*DINO (Distillation with No labels):*
DINO trains a student network to produce similar embeddings to a teacher network.
The teacher is an exponential moving average (EMA) of the student's weights — it moves slowly and acts as a stable target.
Both networks see the same image through different random augmentations (crops, flips, color jitter).
The student is forced to match the teacher's output despite seeing a different view.
This encourages the student to learn augmentation-invariant features: the essence of the object, not the specific cropping.

*SSL gate (`_SSL_GATE_MAX_LOSS = 10.0`):*
If the training loss never drops below 10.0, the fine-tuning failed.
In that case, the pipeline skips the downstream distillation and export steps and uses the baseline model instead.
This prevents a failed training run from producing a worse model than the baseline.
The gate is logged clearly; inspect the loss sparkline in the run output.

*Loss sparkline:*
The step logs an ASCII sparkline of the training loss curve using Unicode block characters (▁▂▃▄▅▆▇█).
A monotonically decreasing curve is expected.
A flat curve means the optimizer is stuck. An oscillating curve means learning rate is too high.

*Augmentation strategy:*
Mission aerial footage differs from internet photos in: scale, viewpoint (nadir), color distribution, and object size.
The default augmentations (random crop, flip, color jitter, Gaussian blur) are conservative.
For strong domain shift, custom augmentations aligned to the mission type (e.g., scale-preserving crops for aerial data) can help.

**Output artifact:**
Fine-tuned DINO checkpoint: `dino_finetuned.pt` in the video output directory.
Training loss curve: per-epoch loss values and ASCII sparkline.
Loss analysis: `{first_loss, best_loss, best_epoch, drop_pct, converged}`.

**Human focus:**
- Understand why self-supervised learning works without labels: augmentation-invariance is the signal.
- Learn to read the loss sparkline: flat = stuck, steady decrease = converging, spike = too high LR.
- Know when fine-tuning helps: missions with rare domain-specific content.
- Know when it does not help: missions that closely match the pretraining distribution (ordinary street scenes).
- Understand the SSL gate: it exists to protect downstream steps from a failed run.

**Common failure modes:**
- Too few training frames (< 50) → the model overfits to a handful of examples; embeddings collapse.
- Learning rate too high → loss oscillates or diverges; SSL gate triggers.
- GPU OOM during fine-tuning → reduce batch size in `FinetuneConfig`; check VRAM before training.
- All frames are visually identical (static camera) → augmentation crops look too similar; model learns nothing.

---

<a id="step-29-knowledge-distillation"></a>
## Step 29. Knowledge distillation

**What it does:**
Train a smaller student network to reproduce the output distribution of the fine-tuned DINO teacher.
The student sees the mission frames and is trained to match the teacher's embeddings using KL divergence or MSE loss on the embedding vectors.
The result is a compact model that retains the teacher's mission-specific knowledge.

**Why it matters:**
The fine-tuned DINO teacher is a large model (ViT-B/16 or ViT-L/14).
It may not be deployable on edge hardware with strict memory and latency budgets.
Distillation transfers the teacher's learned structure into a smaller architecture:
- ViT-B/16 → MobileViT, EfficientViT, or a 4-layer ViT.
- Speed gain: 3-10x inference speedup.
- Size gain: 5-20x smaller model file.
The student does not need raw pixels from the mission: it learns from the teacher's output on those pixels.

**Implementation:**
- [`pipeline/workflows/local/steps_distill.py`](../../pipeline/workflows/local/steps_distill.py)
- [`pipeline/training/distill.py`](../../pipeline/training/distill.py)

**Key concepts:**

*Soft targets vs hard targets:*
Hard targets: one-hot class labels. The student learns to classify.
Soft targets: the teacher's full output distribution (embedding vector). The student learns the geometric structure.
Soft targets carry much more information: the teacher expresses "this frame is 70% similar to class A and 30% similar to class B" rather than just "class A."

*Temperature scaling:*
In classification distillation, a temperature parameter T softens the teacher's probability distribution.
High T: all classes get more equal weight; student learns relative similarities.
Low T: distribution is peaked; student sees the hard winner.
For embedding distillation (used here), temperature scaling is applied differently: to the cosine similarity kernel.

*Teacher-student capacity gap:*
If the student is too small relative to the teacher, it cannot fit the teacher's knowledge.
The practical rule: student should be at least 1/4 the parameter count of the teacher for good distillation.

**Output artifact:**
Student model checkpoint: `student_model.pt` in the video output directory.
Distillation loss curve: per-epoch loss values.
`{teacher_dim, student_dim, distill_loss_final}`.

**Human focus:**
- Understand the "dark knowledge" idea: teacher soft outputs encode more information than hard labels.
- Learn the practical tradeoff: smaller student = faster inference = higher distillation loss.
- Know that distillation always involves some information loss; the student approximates the teacher, it does not copy it.
- Understand when distillation is worth doing: only when deployment constraints exist. If you have a GPU, just use the teacher.

**Common failure modes:**
- Student architecture is too small → distillation loss is stuck high; student cannot represent teacher's distribution.
- Teacher was poorly fine-tuned (high SSL gate loss) → distillation transfers a bad representation; student is worse than the baseline.
- Training on too few frames → student overfits to a handful of teacher outputs; generalizes poorly.

---

<a id="step-30-onnx-export-and-gallery-build"></a>
## Step 30. ONNX export and gallery build

**What it does:**
Export the student model (or the fine-tuned teacher if no student exists) to ONNX format.
Build a reference gallery: embed all mission frames with the ONNX model and store the embeddings for fast search.

**Why it matters:**
A model that cannot be exported and served is still a research artifact.
ONNX (Open Neural Network Exchange) is the standard intermediate format for deploying PyTorch models on:
- Edge hardware with ONNX Runtime
- TensorRT for GPU-optimized inference
- Mobile devices via CoreML or Android NNAPI
- Any runtime that supports ONNX (most modern ML serving stacks)

The gallery is the pre-computed embedding index that allows real-time search without re-embedding every frame on each query.

**Implementation:**
- [`pipeline/workflows/local/steps_distill.py`](../../pipeline/workflows/local/steps_distill.py)

**Key concepts:**

*ONNX export process:*
1. PyTorch model is traced or scripted with a dummy input tensor.
2. The traced computation graph is exported to an `.onnx` file.
3. Dynamic axes are specified so the model accepts variable batch sizes.
4. The ONNX model is validated by loading it with onnxruntime and comparing outputs to the PyTorch model.

*Dynamic vs static axes:*
A static-axes ONNX model only accepts a fixed batch size and input resolution.
A dynamic-axes ONNX model accepts any batch size but may be slightly less optimizable.
The pipeline exports with dynamic batch axes; input resolution is fixed to the training resolution.

*Gallery build:*
The gallery is a `.npy` array of shape `(n_frames, embedding_dim)` produced by running the exported ONNX model over all mission frames.
Alongside it: a metadata JSON that maps each gallery row to a frame path and timestamp.
Search is then: embed query → cosine similarity against gallery → return top-K.

**Output artifact:**
`model_export.onnx` in the video output directory.
`gallery_embeddings.npy` and `gallery_metadata.json`.

**Human focus:**
- Understand why ONNX is the target: it separates model training (PyTorch) from model serving (any runtime).
- Learn the ONNX tracing vs scripting distinction: tracing captures one execution path; scripting handles control flow.
- Know the gallery as a pre-indexed search space: the trade-off is that adding a new frame requires re-running the gallery build.

**Common failure modes:**
- Dynamic control flow in the model (if statements depending on tensor values) → tracing fails; scripting required.
- Input normalization not included in the model → ONNX model requires the caller to pre-normalize; easy to forget.
- ONNX opset version mismatch → exported model uses ops not supported by the target runtime.
- Gallery size too large for RAM → `.npy` file is too big to load into memory; need chunked search or Qdrant.

---

<a id="step-31-fine-tuned-search-test"></a>
## Step 31. Fine-tuned search test

**What it does:**
Re-run the same test queries from Step 26 against the fine-tuned model's gallery.
Record top-K results and compute P@K, R@K.
Produce a side-by-side comparison: baseline (Step 26) vs fine-tuned (Step 31).

**Why it matters:**
Training a model does not mean it improved.
The fine-tuned model might:
- Improve retrieval for in-distribution mission content (the goal).
- Degrade retrieval for general content (acceptable trade-off if mission-specific is more important).
- Make no meaningful difference (fine-tuning was unnecessary for this mission type).
- Get worse overall (the SSL gate should have caught this; if it did not, the evaluation will).

Step 31 is the only honest measure of whether Steps 28-30 were worth doing.

**Implementation:**
- [`pipeline/workflows/local/steps_embed.py`](../../pipeline/workflows/local/steps_embed.py)

**Key concepts:**

*What counts as improvement:*
- Metric improvement: P@K or R@K increases for the target queries.
- Subjective improvement: retrieved frames are more relevant even if the metric is similar (edge cases near the threshold).
- Distribution shift: fine-tuning changed what the model finds similar, even if the metric looks the same.

*When the metric can lie:*
- P@K is sensitive to K: a model that ranks the one correct result at position 3 scores P@5=0.2, same as one that ranks 1 relevant result at position 5.
- R@K is sensitive to how many "correct" frames exist: if you only manually labeled 3 relevant frames but 30 exist, R@K is underestimated.
- Always combine metric comparison with human inspection of the results.

**Output artifact:**
Search comparison report: per-query table of baseline vs fine-tuned top-K results with scores and delta.
Summary: `{baseline_p_at_k, finetuned_p_at_k, delta, improved_queries, degraded_queries}`.

**Human focus:**
- Run the comparison yourself, not just read the summary table.
- Pick two queries: one where fine-tuning clearly helped and one where it clearly did not.
- Understand why mission-specific adaptation might degrade general queries: the model is less general after fine-tuning.

**Common failure modes:**
- Test queries were not designed before fine-tuning → the queries may now be biased toward what the fine-tuning emphasized.
- Only a few test queries → results are not statistically meaningful.
- No baseline recorded from Step 26 → comparison is impossible; rerun Step 26 with the baseline model first.

---

<a id="step-32-model-comparison-and-video-description"></a>
## Step 32. Model comparison and video description

**What it does:**
Produce a side-by-side comparison of baseline vs fine-tuned model behavior across a sample of frames.
Derive a clip-level video description: a readable natural-language summary of the full mission based on all accumulated evidence.

**Why it matters:**
Two separate outputs:
1. The model comparison gives a quantitative and qualitative picture of what changed between baseline and fine-tuned.
2. The video description is the human-facing mission summary: what happened, what was observed, what matters.

The video description is built from the union of all evidence in `VideoKnowledge`:
Gemma scene analysis, Florence captions, ASR text, Qwen structured observations, and UniDriveVLA expert analysis.
This is where the pipeline becomes a product rather than a tool.

**Implementation:**
- [`pipeline/workflows/local/runner.py`](../../pipeline/workflows/local/runner.py)
- [`pipeline/workflows/local/steps_report.py`](../../pipeline/workflows/local/steps_report.py)

**Key concepts:**

*Comparison dimensions:*
- Embedding distance: for the same frame, how far are the baseline and fine-tuned embeddings from each other?
  Large distance = adaptation changed the representation significantly.
  Near-zero distance = adaptation had no effect.
- Retrieval rank shift: did the same query return different neighbors?
  Which improved, which degraded?
- Visual inspection: for the same query, do the fine-tuned top-5 look more relevant than the baseline top-5?

*Video description quality:*
The video description combines evidence across all modalities and all steps.
Its quality is limited by the weakest upstream step: if Gemma failed, scene context is missing; if ASR was noisy, the transcript adds noise.
The description should be read critically: check which claims are supported by which evidence.

**Output artifact:**
`comparison.md` in the video output directory: side-by-side baseline vs fine-tuned retrieval results.
`video_description.md`: the full mission narrative summary.

**Human focus:**
- Read the video description and evaluate whether it matches your understanding of the mission.
- Identify which claims in the description are well-supported by evidence vs speculative.
- Check the comparison table: find the query with the largest rank improvement and the query with the largest regression.

**Common failure modes:**
- SSL fine-tuning failed → comparison shows no difference; fine-tuned results are identical to baseline.
- Video description is too generic ("a vehicle moved across a road") → domain-specific evidence from UniDriveVLA or sensors is absent.
- Description includes hallucinated claims from Qwen that no other modality supports.

---

<a id="step-33-multi-model-comparison"></a>
## Step 33. Multi-model comparison

**What it does:**
For a sample of key frames, collect outputs from multiple multimodal analyzers:
- Florence (caption)
- Gemma (scene analysis)
- Qwen (detailed structured observation)
- UniDriveVLA (domain expert analysis)

Identify where they agree and where they disagree.
Compute a cross-model agreement score for each frame.

**Why it matters:**
Agreement across independent models increases confidence in an observation.
Disagreement is often more informative than agreement:
- If Qwen says "empty road" and Florence says "a vehicle approaching an intersection", one of them is wrong.
- If Gemma says "industrial terrain" and UniDriveVLA says "highway interchange", the scene is ambiguous and warrants human inspection.
- Systematic disagreement between Florence and Qwen across many frames may indicate that one model's prompting is wrong for this domain.

**Implementation:**
- [`pipeline/workflows/local/runner.py`](../../pipeline/workflows/local/runner.py)

**Key concepts:**

*Sources of disagreement:*
1. Different input: Florence sees a single frame; Qwen sees the frame plus context. They should disagree slightly by design.
2. Different training distribution: Florence and Qwen were trained on different corpora; they use different feature biases.
3. Genuine ambiguity: the scene truly has multiple valid interpretations.
4. Error: one model is simply wrong.

*Agreement metric:*
The pipeline computes a Jaccard-style token overlap between captions from different models.
High overlap = agreement. Low overlap = disagreement.
This is a proxy, not a semantic agreement measure: two models can disagree in meaning with high word overlap, or agree in meaning with low overlap.

**Output artifact:**
`multi_model_comparison.md` in the video output directory: per-frame side-by-side outputs with agreement score.
Summary: list of frames with lowest agreement (most uncertain or ambiguous).

**Human focus:**
- Pick the three frames with the lowest cross-model agreement and inspect them manually.
- Determine whether disagreement is due to genuine ambiguity, different input, or outright error.
- Use the disagreement list as an attention signal: these frames are where the pipeline is least confident.

**Common failure modes:**
- Only one model is available → "comparison" has nothing to compare; the step produces a single-model report.
- All models agree on a wrong interpretation → ensemble agreement gives false confidence; always inspect high-confidence failures.
- Agreement metric treats paraphrase as disagreement ("vehicle" vs "car") → score underestimates true agreement.

---

<a id="step-34-video-synthesis"></a>
## Step 34. Video synthesis

**What it does:**
Collect all intermediate artifacts from the full pipeline run and produce a single structured HTML report:
- Mission metadata (timestamp, duration, GPS bounding box)
- Scene summary from Gemma and Florence
- Key frame gallery with captions, detection overlays, and depth labels
- ASR transcript timeline
- Multi-model comparison table for selected frames
- Model adaptation summary (baseline vs fine-tuned metrics)
- Active learning tags: which frames need human annotation?
- Change detection summary if this is a repeat mission

**Why it matters:**
This is where the pipeline becomes a human-facing product.
All previous steps produce machine-readable intermediate artifacts.
Step 34 assembles them into something a person can read in 10-15 minutes and use to make operational decisions.

**Implementation:**
- [`pipeline/workflows/local/runner.py`](../../pipeline/workflows/local/runner.py)
- [`pipeline/report_generator.py`](../../pipeline/report_generator.py)

**Key concepts:**

*What belongs in the synthesis:*
Include: every claim that is supported by at least two independent sources.
Flag: claims supported by only one source.
Exclude: model outputs that directly contradicted another model or failed quality checks.

*Active learning tags:*
Frames tagged `needs_annotation` (high `active_learning_score`) are surfaced in the report as priority labeling candidates.
The report includes the score formula: `0.6 × DINOv3_dist + 0.4 × (1 - caption_confidence)`.
High score = the model was uncertain about this frame; human labeling here will improve future fine-tuning the most.

*Change detection summary:*
If the same GPS area was covered in a prior mission, the change detection table shows which objects or regions changed.
This is the key cross-mission feature: it makes the pipeline a persistent monitoring system, not a one-time analysis.

**Output artifact:**
`reports/{mission_id}/summary.html` — the full mission report rendered as HTML.
Linked: frame gallery images, ASR transcript, GPS track overlay.

**Human focus:**
- Read one full report end-to-end on a real mission and identify one error or gap in each major section.
- Compare the active learning tags to your own sense of which frames are most unusual.
- Evaluate the report as a product: could an operator make a decision based on this without reading the raw artifacts?

**Common failure modes:**
- Some intermediate artifacts are missing (step failed silently) → report has empty sections with no explanation.
- Report is too long → operator skips most of it; the synthesis failed at its own purpose.
- HTML rendering broken due to missing frame paths or relative path errors.

---

<a id="step-35-agentic-flow-audit"></a>
## Step 35. Agentic flow audit

**What it does:**
Produce a provenance-style audit document that traces how context moved through the pipeline:
- Which step produced which artifact?
- Which artifact was consumed by which later step?
- At which steps did context propagation break (missing input, API failure, silent skip)?
- Which claims in the final report are traceable back to raw sensor data?

**Why it matters:**
This is the inspection layer for debugging, trust calibration, and failure analysis.

Without an audit trail:
- You cannot determine which step caused a wrong output.
- You cannot tell whether a claim in the report is well-evidenced or hallucinated.
- You cannot detect silent failures (a step that appeared to run but produced garbage).
- You cannot reproduce the analysis with different settings.

With an audit trail:
- Every claim in the synthesis traces back to at least one source.
- Every step is timestamped; failures have clear upstream causes.
- An engineer can diagnose failures by following the provenance chain backward.

**Implementation:**
- [`pipeline/workflows/local/runner.py`](../../pipeline/workflows/local/runner.py)
- [`docs/pipeline.md`](../pipeline.md)

**Key concepts:**

*Provenance:*
The record of where a piece of information came from, which transformations it went through, and which later outputs it influenced.
In this pipeline: a Qwen claim in the synthesis traces back to → Qwen input context → which came from VideoKnowledge → which was populated by Florence, ASR, OCR, depth, and detection steps.

*Silent failure:*
A step that returns a default or empty result without raising an exception.
Example: Gemma API is unavailable; the step logs a warning and returns `{}`.
`VideoKnowledge.scene_type` stays empty; all later domain hints are blank.
The synthesis does not know this happened; it just receives an empty domain hint and produces weaker output.
The audit catches this: it lists which steps produced non-empty output and which produced default fallbacks.

*Context contamination:*
Wrong context propagated forward through `VideoKnowledge` corrupts downstream steps.
Example: Gemma misclassifies the scene as "coastal" instead of "desert"; this wrong domain hint propagates into every Qwen call.
The audit traces which Gemma call produced the wrong classification and when it was injected.

**Output artifact:**
`agentic_flow.md` in the video output directory:
- Step-by-step timing table: step name, start time, elapsed, status (success / skipped / failed).
- Context propagation graph: which `VideoKnowledge` fields were populated by which step.
- Silent failure log: steps that returned empty output or default fallbacks.
- Claim provenance: for each section of the synthesis, which steps contributed evidence.

**Human focus:**
- Read `agentic_flow.md` first after a new run, before reading the synthesis.
- Identify any step with status `skipped` or `failed` and determine what downstream effects it had.
- Trace one claim from the synthesis backward to its raw source. Can you do it?
- Know the difference between a hard failure (exception logged) and a silent failure (empty output, no exception).

**Common failure modes:**
- Audit itself fails to run (runner crash in final cleanup) → no provenance exists; debugging is manual.
- Step timing is wrong because CUDA events are asynchronous → GPU steps appear instantaneous; use CPU timing for wall-clock.
- Silent failures not logged → audit is incomplete; some gaps are invisible.
- Context contamination is undetected → audit shows "success" for every step but the synthesis is wrong.

---

## What A Human Should Learn In This Phase

Learn to separate these four questions and ask them independently:

1. **Did the representation get better?**
   Compare P@K from Steps 26 and 31. Not just the numbers, but which queries improved.

2. **Did the smaller model preserve the right structure?**
   Compare the teacher's embeddings to the student's embeddings on held-out frames.
   Low cosine distance means the student learned well; high distance means it lost something.

3. **Did the outputs become more useful for humans?**
   Read the synthesis report. Could an operator use it? What is missing?
   This is a different question from (1) and (2).

4. **Can I trace how the final conclusion was formed?**
   Use the agentic flow audit. Can you follow the provenance chain from any synthesis claim back to raw data?

A system can optimize one of these and fail the others.
Good engineering requires all four.

## Related Docs

- [Tracking and mapping: Steps 21-27](04_tracking_mapping_steps_21_27.md)
- [Agentic knowledge flow](06_agentic_knowledge_flow.md)
- [Runtime and study guide](01_runtime_and_study_guide.md)
- [Pipeline architecture](../pipeline.md)
