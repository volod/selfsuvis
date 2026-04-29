# Local Run Analytics & Visualization

After running the local pipeline (`selfsuvis --mode local`), the output directory contains a
mix of JSON, Markdown, model, and media artifacts. The analytics and visualization subpackages
turn those artifacts into structured summaries, warnings, charts, and a portable HTML report.

Implementation plan:

- [Local run analytics and visualization plan](design/local_run_analytics_visualization_plan.md)

## Subpackages

| Package | Location | Purpose |
|---------|----------|---------|
| `selfsuvis.analytics` | `src/selfsuvis/analytics/` | Load artifacts, compute statistics, flag degraded stages |
| `selfsuvis.visualization` | `src/selfsuvis/visualization/` | Produce matplotlib charts and a self-contained HTML report |

## Quick Start

Main CLI:

```bash
selfsuvis --mode analyse --run-dir data/local_runs/drone_mission
```

Module form:

```bash
python -m selfsuvis --mode analyse --run-dir data/local_runs/drone_mission
```

Optional machine-readable summary:

```bash
selfsuvis --mode analyse \
  --run-dir data/local_runs/drone_mission \
  --summary-json data/local_runs/drone_mission/analysis_summary.json
```

This writes PNG charts and `analysis_report.html` into the run directory by default.

Options:

```text
--run-dir PATH         Local run output directory
--charts-dir PATH      Override chart output directory
--no-report            Skip HTML report generation
--report-filename NAME Custom HTML filename
--summary-json PATH    Write compact summary JSON
```

## What Gets Parsed

The loader reads the following artifacts when present:

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
- `3d_map/map_quality_advisor.json`
- every file under the run directory for inventory and size reporting

## What Gets Computed

`RunSummary` now carries:

- frame-level records with timestamps, detections, surprise, captions, ASR, and tracking counts
- aggregate detection, temporal, training, tracking, embedding, and map statistics
- derived diagnostics for modality completeness, run quality, tracking fragmentation, map pose coverage, temporal surprise dispersion, and adaptation efficiency
- artifact inventory by path, suffix, and top-level category
- run-health warnings for silent degradations

Current warning rules include:

- Florence captions empty for all frames
- Qwen detailed captioning produced only parse errors
- Gemma-directed tracking produced zero detections
- Gemma-directed tracking had to retry with a relaxed label filter
- Gemma-directed tracking is highly fragmented
- 3D map quality is degraded
- model restore failures or long VRAM waits occurred
- world-model stage reported failure or zero usable clips

## Diagnostics Methodology

The analytics step now computes a compact diagnostics block in both the Python API
and the local pipeline `analysis_summary.json`.

| Field | Meaning |
|-------|---------|
| `modality_completeness` | Mean availability across Florence, Qwen, ASR, OCR, world model, tracking, mapping, and edge export |
| `quality_score` | Weighted 0-100 health score with penalties for warnings |
| `detection_density_per_frame` | Mean YOLO/SAM detections per frame |
| `detection_count_cv` | Coefficient of variation of per-frame detection counts |
| `detection_entropy_norm` | Normalized Shannon entropy of detected class counts |
| `tracking_fragmentation` | Unique track IDs divided by total tracked detections |
| `track_persistence` | Mean track length divided by frame count |
| `surprise_std` | Standard deviation of RSSM temporal-surprise scores |
| `surprise_peak_rate` | Fraction of frames marked as temporal-surprise peaks |
| `surprise_detection_overlap` | Fraction of surprise peaks that also contain detections or tracks |
| `map_points_per_pose` | Sparse-map point density per recovered pose |
| `map_pose_coverage` | Recovered SfM poses divided by extracted frames, clamped to `[0, 1]` |
| `adaptation_efficiency` | Distilled Recall@1 divided by compression ratio |
| `artifact_density_per_frame` | Number of output files per extracted frame |
| `artifact_mb_per_min` | Artifact size normalized by video duration |

For the equations and interpretation rules, read
[Local analytics math and methodology](learning_path/13_local_analytics_math_methodology.md).

## Mapping Diagnostics And Advisor Artifacts

Mapping now emits two complementary artifacts:

- `3d_map/map_stats.json`
  Reports what reconstruction actually happened: sparse SfM poses, point count, frame anchors,
  fallback method, and whether the map is degraded.
- `3d_map/map_quality_advisor.json`
  Explains why the map succeeded or degraded by measuring video quality and geometry signals:
  duration, resolution, brightness stability, sharpness, feature richness, adjacent-frame
  matchability, parallax proxy, object scale / field size, and a coarse camera-angle hint.

This distinction matters:

- `map_stats.json` answers "what did the mapper recover?"
- `map_quality_advisor.json` answers "was the source video suitable for a high-quality map?"

Typical advisor findings:

- good exposure but poor parallax: the video is visually clean, but camera motion is too close to
  pure forward drift or pure nadir hover for strong geometry
- strong feature richness but tiny object scale: the scene has texture, but the aircraft is too
  high or the FOV is too wide to recover detailed structure
- short clip with sparse SfM poses: the mapping stack is not the main problem; capture coverage is
  insufficient

For the current drone learning-path run, the advisor shows exactly that pattern: exposure,
sharpness, and ORB feature count are healthy, while clip duration, parallax, and field scale are
poor. Treat that as a capture problem first, not as a model-loading problem.

## Charts Produced

| Chart | Source data | Insight |
|-------|-------------|---------|
| `timeline.png` | `rssm_temporal.json` + frame records | Surprise curve and detection density over time |
| `detections.png` | `yolo_sam_results.json` | Class totals and detections per frame |
| `embedding_pca.png` | `edge_models/gallery.npz` | 2-D view of frame embedding structure |
| `similarity_matrix.png` | `edge_models/gallery.npz` | Frame-to-frame cosine similarity |
| `training.png` | `finetune_stats.md` + `distill_stats.md` | SSL loss trend and distillation summary |

## HTML Report Contents

The report combines:

- overview cards
- run-health warnings
- timeline and detection figures
- embedding-space figures
- training summary
- derived-artifact cards for tracking, embeddings, and mapping
- full artifact inventory with file sizes

## Python API

```python
from selfsuvis.analytics import LocalRunLoader
from selfsuvis.visualization import generate_report, plot_timeline

summary = LocalRunLoader("data/local_runs/drone_mission").load()
print(summary.run_health.warnings)
print(summary.artifact_inventory.total_files)

fig = plot_timeline(summary)
report_path = generate_report(summary)
```

## Learning Path Integration

Use the deep-dive companion after your first run:

- [Local run artifact analysis](learning_path/08_local_run_artifact_analysis.md)
- [Local analytics math and methodology](learning_path/13_local_analytics_math_methodology.md)
- [Day-by-day syllabus](learning_path/00_day_by_day_syllabus.md)
