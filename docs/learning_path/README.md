# Learning Path Deep Dives

This directory is the detailed companion to [`local_path.md`](../../local_path.md).
`local_path.md` is the fast study map. These documents are the deeper human-oriented explanations.

## What Is In Here

| Document | Purpose |
|---|---|
| [01_runtime_and_study_guide.md](01_runtime_and_study_guide.md) | How to read the repo, what the runtime actually does, and how to study it without getting lost |
| [02_perception_core_steps_01_08.md](02_perception_core_steps_01_08.md) | Frame extraction, embeddings, captioning, ASR, OCR, depth, and detection |
| [03_sensor_steps_09_20.md](03_sensor_steps_09_20.md) | RF, thermal, multispectral, LiDAR, radar, GNSS-R, environmental sensors, and fusion |
| [04_tracking_mapping_steps_21_27.md](04_tracking_mapping_steps_21_27.md) | Segmentation, tracking, temporal embeddings, Qwen, UniDriveVLA, search, and 3D mapping |
| [05_adaptation_eval_steps_28_35.md](05_adaptation_eval_steps_28_35.md) | SSL, distillation, export, evaluation, synthesis, and audit |
| [06_agentic_knowledge_flow.md](06_agentic_knowledge_flow.md) | How context accumulates and gets reused across steps |
| [07_day_by_day_syllabus.md](07_day_by_day_syllabus.md) | A realistic study plan with daily topics and exercises |

## How To Read These Docs

1. Start with [`local_path.md`](../../local_path.md) to understand the full path.
2. Pick the phase you are working on.
3. Read the matching deep-dive file.
4. Open the implementation modules linked from that file and compare code to the prose.

## Step-To-Document Map

| Steps | Deep-dive document |
|---|---|
| 1-8 | [Perception core](02_perception_core_steps_01_08.md) |
| 9-20 | [Sensors and fusion](03_sensor_steps_09_20.md) |
| 21-27 | [Tracking and mapping](04_tracking_mapping_steps_21_27.md) |
| 28-35 | [Adaptation and audit](05_adaptation_eval_steps_28_35.md) |

## Related Repo Docs

- [Pipeline architecture](../pipeline.md)
- [Architecture](../architecture.md)
- [Configuration](../configuration.md)
- [Setup](../setup.md)
- [3D Gaussian Splat](../gaussian_splat.md)
- [Design docs](../design)
- [ADRs](../adr/README.md)
