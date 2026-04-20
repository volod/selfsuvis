# Learning Path

The old single-file learning path has been split into a short path plus a deep-dive directory.

Start here:

- [Short local path](local_path.md)
- [Learning path deep dives](learning_path/README.md)
- [Day-by-day syllabus](learning_path/07_day_by_day_syllabus.md)

Recommended reading order:

1. [Short local path](local_path.md)
2. [Runtime and study guide](learning_path/01_runtime_and_study_guide.md)
3. [Perception core, Steps 1-8](learning_path/02_perception_core_steps_01_08.md)
4. [Sensors and fusion, Steps 9-20](learning_path/03_sensor_steps_09_20.md)
5. [Tracking and mapping, Steps 21-27](learning_path/04_tracking_mapping_steps_21_27.md)
6. [Adaptation and audit, Steps 28-35](learning_path/05_adaptation_eval_steps_28_35.md)
7. [Agentic knowledge flow](learning_path/06_agentic_knowledge_flow.md)
8. [Local run artifact analysis](learning_path/08_local_run_artifact_analysis.md)

If you were sent here by an older script or document, use this page as the compatibility entry point.

## Probabilistic State Fusion

The fusion subsystem is fully implemented. Four layers run after the 3D map
step on every local run and write `full_state_fusion.json`:

| Layer | Description |
|---|---|
| Semantic priors | VLM scene type + RSSM surprise → adaptive Kalman noise |
| Platform KF | GPS + IMU + baro constant-velocity Kalman filter |
| Map-state (RTS) | Adds SfM visual-pose constraints; RTS backward smoother |
| Object-state | Per-track KF + Mahalanobis gating + Hungarian + RTS |

Deep dives:

- [Sensor fusion fundamentals](learning_path/09_sensor_fusion_fundamentals.md)
- [Requirements and status](learning_path/10_probabilistic_state_fusion_requirements.md)
- [Architecture and data flow](learning_path/11_probabilistic_state_fusion_architecture.md)
- [Delivery status and gaps](learning_path/12_probabilistic_state_fusion_implementation_order.md)
- [**Mathematical deep dive**](learning_path/13_probabilistic_fusion_deep_dive.md) ← start here if you want to understand the math

## Post-run analysis

After completing your first local run, use the analytics toolkit to inspect results:

- [Analytics & Visualization guide](analytics.md) — charts, HTML report, Python API
