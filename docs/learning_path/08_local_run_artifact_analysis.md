# Local Run Artifact Analysis

This deep dive is for the moment after a local run has finished and you need to answer:

- Which artifacts are trustworthy?
- Which stages silently degraded?
- What should I inspect first?
- How do the artifacts relate back to the code?

## Why This Matters

The local pipeline is intentionally ambitious. A run can complete and still contain failures that
only show up in the artifacts:

- empty Florence captions
- zero RF-DETR tracks after successful Gemma segmentation
- world-model clip failures with RSSM fallback still succeeding
- good ONNX export but weak distillation quality

Treat the run directory as evidence, not as proof that every stage worked.

## Recommended Inspection Order

1. `final_stats.md`
   Use this to identify slow stages, skipped stages, and aggregate metrics.
2. `scene_captions.md` and `detailed_captions.md`
   Check whether caption quality is informative or silently empty.
3. `yolo_sam_results.json` and `detection_comparison.md`
   Confirm object counts and detector agreement.
4. `gemma_tracking_results.json`
   Verify whether Gemma-directed tracking actually found tracks or only masks.
5. `rssm_temporal.json`
   Identify surprise peaks and compare them to scene changes.
6. `finetune_stats.md`, `distill_stats.md`, and `edge_models/`
   Determine whether adaptation produced a useful deployable edge artifact.
7. `3d_map/`
   Confirm that map generation produced actual geometry, not just placeholder outputs.

## Use The Analytics Tooling

Generate a report:

```bash
selfsuvis --mode analyse --run-dir data/local_runs/drone_mission
```

Module form:

```bash
python -m selfsuvis --mode analyse --run-dir data/local_runs/drone_mission
```

This gives you:

- a timeline chart
- detection charts
- embedding PCA and similarity plots
- training charts
- an HTML report with warnings and artifact inventory

## Questions To Ask Per Artifact Family

### Per-frame artifacts

- Are frame timestamps monotonic and plausible in `frames_metadata.json`?
- Are Florence captions non-empty and semantically stable?
- Do Qwen captions add information beyond Florence, or just paraphrase?
- Does ASR inject relevant context or obvious contamination?

### Detection and tracking artifacts

- Do YOLO counts match what you can visually verify in annotated frames?
- Did Gemma tracking produce track IDs, or only SAM masks?
- Are zero detections a model problem or a label-vocabulary problem?

### Temporal artifacts

- Do RSSM surprise peaks align with visible state changes?
- If surprise is high everywhere, is the embedder unstable or the scene genuinely dynamic?

### Adaptation artifacts

- Did SSL loss improve meaningfully?
- Did the student retain retrieval quality after compression?
- Does the ONNX artifact exist and match the reported dimensions?

### Mapping artifacts

- Does `3d_map/map_stats.json` show real poses and points?
- Is the Gaussian splat viewable, or was only a placeholder file written?

## Code Traceback Map

When an artifact looks wrong, trace it back from the run directory to code:

- captions: `src/selfsuvis/pipeline/vision/florence.py`
- local caption/report orchestration: `src/selfsuvis/pipeline/workflows/local/steps_caption.py`
- report writers: `src/selfsuvis/pipeline/workflows/local/steps_report.py`
- analytics loader: `src/selfsuvis/analytics/loader.py`
- report generation: `src/selfsuvis/visualization/report.py`

## Practical Exercises

1. Find one completed run with a warning in the analytics report. Explain the warning using raw artifacts.
2. Compare `scene_captions.md` and `detailed_captions.md` for ten frames. Identify where Qwen adds materially new information.
3. For each RSSM peak frame, open the corresponding image and decide whether the peak is justified.
4. Inspect `gemma_tracking_results.json` and explain why tracking succeeded or failed.
5. Verify the reported ONNX and checkpoint sizes against actual files on disk.
