# Local Run Analytics And Visualization Plan

This document is the implementation plan for post-run analytics over artifacts produced by
`selfsuvis --mode local`.

## Goals

1. Convert local-run output directories into a structured Python object model.
2. Detect partial failures that do not abort the pipeline but degrade outputs.
3. Generate reusable plots and a portable HTML report from those artifacts.
4. Expose the workflow through both a repo script and an installed CLI entry point.
5. Use the same artifacts as the basis for a human learning path.

## Scope

Artifacts explicitly targeted:

- `frames_metadata.json`
- `scene_captions.md`
- `asr_subtitles.md`
- `detailed_captions.md`
- `yolo_sam_results.json`
- `gemma_tracking_results.json`
- `rssm_temporal.json`
- `video_ontology.json`
- `finetune_stats.md`
- `distill_stats.md`
- `edge_models/gallery.npz`
- `3d_map/map_stats.json`
- inventory scan of every file under the run directory

## Package Design

### `selfsuvis.analytics`

Responsibilities:

- parse artifact files into typed dataclasses
- compute aggregate statistics
- compute run-health warnings for silent degradation cases
- expose a single `LocalRunLoader(...).load()` entry point

Core types:

- `RunSummary`
- `FrameRecord`
- `DetectionStats`
- `TemporalStats`
- `TrainingStats`
- `TrackingStats`
- `EmbeddingStats`
- `MapStats`
- `ArtifactInventory`
- `RunHealth`

### `selfsuvis.visualization`

Responsibilities:

- render timeline, detection, embedding, and training figures
- assemble a self-contained HTML report
- present warnings and artifact inventory alongside charts

## Failure Detection Rules

These are explicit because the local pipeline can finish successfully while still producing
bad or incomplete artifacts.

- Florence failure: all captions empty in `scene_captions.md`
- World-model failure: `multimodal_features.md` indicates unavailable/error or zero usable clips
- Tracking degradation: tracking artifact exists but `total_detections == 0`
- Missing artifact: expected summary artifact absent after the stage claims completion

## CLI Surface

Supported entry points:

- main CLI: `selfsuvis --mode analyse --run-dir ...`
- module form: `python -m selfsuvis --mode analyse --run-dir ...`

Outputs:

- PNG charts
- `analysis_report.html`
- optional compact summary JSON

## Verification Plan

1. Add unit tests for Florence input sanitization.
2. Add unit tests for world-model input dtype alignment.
3. Add unit tests for analytics loader parsing against a synthetic local-run directory.
4. Run focused pytest targets for the touched areas.

## Out Of Scope

- interactive dashboards
- browser-served artifact explorer
- automatic cross-run comparisons between multiple local runs
- mutation of original artifacts after the pipeline completes
