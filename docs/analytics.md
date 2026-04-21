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
- every file under the run directory for inventory and size reporting

## What Gets Computed

`RunSummary` now carries:

- frame-level records with timestamps, detections, surprise, captions, ASR, and tracking counts
- aggregate detection, temporal, training, tracking, embedding, and map statistics
- artifact inventory by path, suffix, and top-level category
- run-health warnings for silent degradations

Current warning rules include:

- Florence captions empty for all frames
- Gemma-directed tracking produced zero detections
- world-model stage reported failure or zero usable clips

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
- [Day-by-day syllabus](learning_path/07_day_by_day_syllabus.md)
