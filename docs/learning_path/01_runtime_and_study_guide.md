# Runtime And Study Guide

This guide explains how to approach the repo as a human learner.
The goal is not only to run the pipeline once, but to understand why each stage exists and how evidence moves through the stack.

## Source Of Truth

For runtime behavior, the main local source is:

- [`pipeline/workflows/local/runner.py`](../../pipeline/workflows/local/runner.py)

Supporting step modules live here:

- [`pipeline/workflows/local/steps_embed.py`](../../pipeline/workflows/local/steps_embed.py)
- [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
- [`pipeline/workflows/local/steps_yolo_sam.py`](../../pipeline/workflows/local/steps_yolo_sam.py)
- [`pipeline/workflows/local/steps_gemma_tracking.py`](../../pipeline/workflows/local/steps_gemma_tracking.py)
- [`pipeline/workflows/local/steps_map.py`](../../pipeline/workflows/local/steps_map.py)
- [`pipeline/workflows/local/steps_ssl.py`](../../pipeline/workflows/local/steps_ssl.py)
- [`pipeline/workflows/local/steps_distill.py`](../../pipeline/workflows/local/steps_distill.py)
- [`pipeline/workflows/local/steps_report.py`](../../pipeline/workflows/local/steps_report.py)

The 35-step learning path is broader than the runner implementation itself. That is intentional:

- the runner is the execution path
- the learning path is the conceptual path
- some conceptual steps are grouped or partially optional in code

## Best Mental Model

Read the system in five layers:

1. Input and memory: frames, embeddings, retrieval
2. Evidence extraction: captions, speech, OCR, depth, detections
3. Physical sensing: RF, thermal, inertial, acoustic, environmental side channels
4. Structured reasoning: tracking, temporal embeddings, multimodal experts, map building
5. Adaptation and audit: SSL, distillation, export, evaluation, final synthesis

## How A Human Should Study It

Do not start by reading every file.
Use this order instead:

1. Run or inspect one local mission output directory.
2. Read `local_path.md`.
3. Read the deep-dive doc for the phase you care about.
4. Open the code modules only after you know what question you are trying to answer.

Good questions:

- What evidence is created here?
- What artifact is written here?
- What later step depends on this output?
- What can fail silently?
- What should a human verify manually?

Weak questions:

- What does every line do?
- Which model is coolest?
- Can I memorize all options before I run anything?

## What To Inspect After A Real Run

A strong first pass is to inspect artifacts in this order:

1. `scene_captions.md`
2. `asr_subtitles.md`
3. `multimodal_features.md`
4. `detailed_captions.md`
5. `unidrive_analysis.md`
6. `comparison.md`
7. `multi_model_comparison.md`
8. `3d_map/`
9. `agentic_flow.md`

That order mirrors the move from raw evidence to higher-level reasoning.

## Practical Study Rules

- Study outputs before internals.
- Compare neighboring steps, not isolated steps.
- Separate representation problems from reasoning problems.
- Separate “this step exists in the repo” from “this step is currently enabled in my run”.
- Keep notes on inputs, outputs, and failure modes for each step.

## Where To Go Next

- For Steps 1-8: [02_perception_core_steps_01_08.md](02_perception_core_steps_01_08.md)
- For Steps 9-20: [03_sensor_steps_09_20.md](03_sensor_steps_09_20.md)
- For Steps 21-27: [04_tracking_mapping_steps_21_27.md](04_tracking_mapping_steps_21_27.md)
- For Steps 28-35: [05_adaptation_eval_steps_28_35.md](05_adaptation_eval_steps_28_35.md)
- For context accumulation: [06_agentic_knowledge_flow.md](06_agentic_knowledge_flow.md)
