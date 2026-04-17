# Learning Path Deep Dives

This directory is the human-oriented companion to [`local_path.md`](../local_path.md),
[architecture.md](../architecture.md), and [pipeline.md](../pipeline.md).

Use these docs when you want to understand the system as a person, not just run it.
They are written to answer:

- What is this stage for?
- What evidence does it create?
- What artifact should I inspect?
- What later stage depends on it?
- What usually goes wrong?

`local_path.md` is the quick route. This directory is the slower, explanatory route.

## What Is In Here

| Document | Purpose |
|---|---|
| [01_runtime_and_study_guide.md](01_runtime_and_study_guide.md) | Best entry point: how the current repo is organized, what runs in production vs local mode, and how to study without getting lost |
| [02_perception_core_steps_01_08.md](02_perception_core_steps_01_08.md) | Frame extraction, embeddings, Gemma, Florence, ASR, OCR, depth, and detection — with key concepts, artifacts, and failure modes |
| [03_sensor_steps_09_20.md](03_sensor_steps_09_20.md) | Optional physical sensor families and fusion thinking — useful when sidecar data exists, ignorable when it does not |
| [04_tracking_mapping_steps_21_27.md](04_tracking_mapping_steps_21_27.md) | Semantic graph construction, Gemma-directed tracking, world-model context, Qwen, UniDriveVLA, and 3D mapping |
| [05_adaptation_eval_steps_28_35.md](05_adaptation_eval_steps_28_35.md) | Fine-tuning, distillation, ONNX export, retrieval evaluation, synthesis, and audit — the “did the system improve?” phase |
| [06_agentic_knowledge_flow.md](06_agentic_knowledge_flow.md) | `VideoKnowledge` structure, evidence accumulation, context reuse, contamination risks, and debugging strategy |
| [07_day_by_day_syllabus.md](07_day_by_day_syllabus.md) | Multi-week human study plan with prerequisites, exercises, checkpoints, and milestones |

## Current Runtime vs Conceptual Path

The current local runner executes **23 top-level steps**.
Some older learning-path documents still group the system using a broader **35-step conceptual map**.
That is intentional:

- the **runtime** view matches the current code in `src/selfsuvis/pipeline/workflows/local/runner.py`
- the **conceptual** view keeps more granular study buckets so related ideas stay separate for a learner

Read the documents as a study decomposition, not a promise that every numbered conceptual step
is a separate top-level function call in the current runner.

## How To Read These Docs

1. Start with [01_runtime_and_study_guide.md](01_runtime_and_study_guide.md).
2. Skim [`local_path.md`](../local_path.md) for the fast path.
3. Read the matching deep-dive file for the phase you care about.
4. Open the implementation modules linked from that file and compare code to the prose.
5. Inspect real output artifacts while you read. This repo makes more sense from outputs back to code than from code outward.

## If You Are New Here

Use this order:

1. [01_runtime_and_study_guide.md](01_runtime_and_study_guide.md)
2. [02_perception_core_steps_01_08.md](02_perception_core_steps_01_08.md)
3. [06_agentic_knowledge_flow.md](06_agentic_knowledge_flow.md)
4. [04_tracking_mapping_steps_21_27.md](04_tracking_mapping_steps_21_27.md)
5. [05_adaptation_eval_steps_28_35.md](05_adaptation_eval_steps_28_35.md)

That route gets you from “what is this repo?” to “how does evidence move?” before
you dive into adaptation and evaluation.

## Step-To-Document Map

| Conceptual phase | Deep-dive document |
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
