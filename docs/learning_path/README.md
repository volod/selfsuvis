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
| [00_day_by_day_syllabus.md](00_day_by_day_syllabus.md) | Multi-week human study plan with prerequisites, exercises, checkpoints, and milestones |
| [18_future_directions.md](18_future_directions.md) | Not-yet-implemented advanced themes: full cross-modal temporal SSL, environmental field models, calibration and contradiction handling, global cross-mission threat inference |
| [01_runtime_and_study_guide.md](01_runtime_and_study_guide.md) | Best entry point: how the current repo is organized, what runs in production vs local mode, and how to study without getting lost |
| [02_perception_core_steps_01_08.md](02_perception_core_steps_01_08.md) | Frame extraction, embeddings, Gemma, Florence, ASR, OCR, depth, and detection — with key concepts, artifacts, and failure modes |
| [03_sensor_fusion_fundamentals.md](03_sensor_fusion_fundamentals.md) | Knowledge session on clocks, calibration, uncertainty, contradiction handling, and what fusion means in the current `selfsuvis` architecture |
| [04_sensor_steps_09_20.md](04_sensor_steps_09_20.md) | Optional physical sensor families and fusion thinking — useful when sidecar data exists, ignorable when it does not |
| [05_tracking_mapping_steps_21_27.md](05_tracking_mapping_steps_21_27.md) | Semantic graph construction, Gemma-directed tracking, world-model context, Qwen, UniDriveVLA, and 3D mapping |
| [06_adaptation_eval_steps_28_35.md](06_adaptation_eval_steps_28_35.md) | Fine-tuning, distillation, drone detection edge training, ONNX export, retrieval evaluation, synthesis, and audit — the “did the system improve?” phase |
| [07_agentic_knowledge_flow.md](07_agentic_knowledge_flow.md) | `VideoKnowledge` structure, evidence accumulation, context reuse, contamination risks, and debugging strategy |
| [08_local_run_artifact_analysis.md](08_local_run_artifact_analysis.md) | How to inspect a completed run, detect silent failures, and connect artifacts back to code |
| [09_probabilistic_state_fusion_requirements.md](09_probabilistic_state_fusion_requirements.md) | Requirements for probabilistic state fusion, with implementation status for each requirement |
| [10_probabilistic_state_fusion_architecture.md](10_probabilistic_state_fusion_architecture.md) | Actual subsystem architecture: package layout, data flow, layer responsibilities, configuration, degradation modes |
| [11_probabilistic_state_fusion_implementation_order.md](11_probabilistic_state_fusion_implementation_order.md) | Delivery status for all five phases, what is still missing, validation sequence for future extensions |
| [12_probabilistic_fusion_deep_dive.md](12_probabilistic_fusion_deep_dive.md) | **Mathematical deep dive**: Kalman filter equations, Umeyama Sim(3) derivation, RTS smoother, Mahalanobis gating, Hungarian assignment, semantic noise priors, worked example, artifact reading guide |
| [13_local_analytics_math_methodology.md](13_local_analytics_math_methodology.md) | The math and interpretation rules behind local-run diagnostics |
| [14_temporal_ssl_physical_state.md](14_temporal_ssl_physical_state.md) | Track-aware SSL: why frame augmentation is insufficient, how RF-DETR track IDs produce identity-consistent positive pairs, and how cycle-consistency loss prevents embedding drift along long tracks |
| [15_threat_primitives_local_inference.md](15_threat_primitives_local_inference.md) | Threat primitive layer: structured evidence-gated threat signals from physical state + fusion; schema design, the two-source gate, and why free-text hazards are insufficient |
| [16_coop_pilot_iot_edge_monitoring.md](16_coop_pilot_iot_edge_monitoring.md) | IoT edge monitoring deep dive: MQTT, LoRaWAN/ChirpStack, Frigate, MediaMTX RTSP bridge, acoustic analysis, rolling site state, scene synthesis, and realtime threat ingestion |
| [17_essential_technology_stack.md](17_essential_technology_stack.md) | Extended human guide to every essential technology in the stack: package layout, pipeline.core shared modules (env, preflight, sidecars, logging), API/worker/DB/Qdrant service flow, security boundaries (APP_ENV, HMAC, fail-closed), FFmpeg and sidecar JSONL, CLIP/DINO embeddings and named-vector retrieval, all vision/language models (Gemma, Florence, Whisper, OCR, depth, YOLO/RF-DETR, SAM, Qwen, UniDriveVLA, SceneTok), temporal state and fusion (RSSM, Kalman, Mahalanobis, Hungarian, RTS, SfM, Gaussian Splat), 33-step local runner, preflight startup contract, LangGraph path, runtime gates, analysis_summary.json analytics, Utilyze profiling, SSL fine-tuning, distillation, ONNX/RKNN edge export, drone detection training, drone audio CNN training, and realtime/coop stack (MediaMTX, Mosquitto MQTT, ChirpStack LoRaWAN, Frigate, rolling site state) |
| [19_drone_audio_detection.md](19_drone_audio_detection.md) | Step 32 deep dive: DroneAudioCNN binary classifier, MFCC feature extraction without librosa (scipy-only STFT + mel filterbank + DCT), Conv2d/BN/ReLU/AdaptiveAvgPool architecture (~52k params), ONNX export for edge inference, dataset split, inference example, physics-based audio simulation (inverse-square, atmospheric absorption, Doppler) |

## Probabilistic State Fusion — Quick Reference

The fusion subsystem lives in `src/selfsuvis/pipeline/fusion/` and is fully
implemented. The four active layers are:

1. **Semantic priors** — Gemma/Qwen/RSSM scene type → noise scale factors
2. **Platform Kalman** — GPS + IMU + baro → position/velocity posterior
3. **Map-state fusion** — adds SfM visual-pose constraints + RTS trajectory smoothing
4. **Object-state fusion** — per-track Kalman + Mahalanobis gating + Hungarian assignment + RTS

Output artifact: `full_state_fusion.json` in each video's output directory.

Entry point for the math: [12_probabilistic_fusion_deep_dive.md](12_probabilistic_fusion_deep_dive.md).

## Current Runtime vs Conceptual Path

The current monolithic local runner reports **33 runtime/post-run steps**.
Some older learning-path documents still group the system using a broader **36-step conceptual map**.
That is intentional:

- the **runtime** view matches the current code in `src/selfsuvis/pipeline/workflows/local/runner.py`
- the **conceptual** view keeps more granular study buckets so related ideas stay separate for a learner

Read the documents as a study decomposition, not a promise that every numbered conceptual step
is a separate top-level function call in the current runner.

## How To Read These Docs

1. Start with [01_runtime_and_study_guide.md](01_runtime_and_study_guide.md).
2. Skim [00_day_by_day_syllabus.md](00_day_by_day_syllabus.md) if you want a paced study plan instead of a reference manual.
3. Skim [`local_path.md`](../local_path.md) for the fast path.
4. Read the matching deep-dive file for the phase you care about.
5. Open the implementation modules linked from that file and compare code to the prose.
6. Inspect real output artifacts while you read. This repo makes more sense from outputs back to code than from code outward.

If you are entering the sensor phase for the first time, read
[03_sensor_fusion_fundamentals.md](03_sensor_fusion_fundamentals.md) before
[04_sensor_steps_09_20.md](04_sensor_steps_09_20.md). It gives the minimum
framework for reasoning about clocks, coordinate frames, calibration, and
uncertainty.

If you already understand the current runner and want to reason about where the
system should go next, read
[18_future_directions.md](18_future_directions.md)
after the fusion and adaptation docs.

## If You Are New Here

Use this order:

1. [01_runtime_and_study_guide.md](01_runtime_and_study_guide.md)
2. [02_perception_core_steps_01_08.md](02_perception_core_steps_01_08.md)
3. [03_sensor_fusion_fundamentals.md](03_sensor_fusion_fundamentals.md)
4. [04_sensor_steps_09_20.md](04_sensor_steps_09_20.md)
5. [05_tracking_mapping_steps_21_27.md](05_tracking_mapping_steps_21_27.md)
6. [07_agentic_knowledge_flow.md](07_agentic_knowledge_flow.md)
7. [06_adaptation_eval_steps_28_35.md](06_adaptation_eval_steps_28_35.md)
8. [08_local_run_artifact_analysis.md](08_local_run_artifact_analysis.md)
9. [13_local_analytics_math_methodology.md](13_local_analytics_math_methodology.md)
10. [14_temporal_ssl_physical_state.md](14_temporal_ssl_physical_state.md)
11. [15_threat_primitives_local_inference.md](15_threat_primitives_local_inference.md)
12. [16_coop_pilot_iot_edge_monitoring.md](16_coop_pilot_iot_edge_monitoring.md)
13. [17_essential_technology_stack.md](17_essential_technology_stack.md)
14. [19_drone_audio_detection.md](19_drone_audio_detection.md)
15. [18_future_directions.md](18_future_directions.md)

That route gets you from “what is this repo?” to “how does evidence move?” before
you branch into adaptation, physical-world modeling, and advanced global-threat work.

## Step-To-Document Map

| Conceptual phase | Deep-dive document |
|---|---|
| 1-8 | [Perception core](02_perception_core_steps_01_08.md) |
| 9-20 | [Sensors and fusion](04_sensor_steps_09_20.md) |
| 21-27 | [Tracking and mapping](05_tracking_mapping_steps_21_27.md) |
| 28-36 | [Adaptation and audit](06_adaptation_eval_steps_28_35.md) |
| 37-43 | [coop_pilot IoT edge monitoring](16_coop_pilot_iot_edge_monitoring.md) |

Supporting sessions:

- [Sensor fusion fundamentals](03_sensor_fusion_fundamentals.md): read before
  or during Steps 9-20 if you need the theory of alignment, uncertainty, and
  contradiction handling.
- [Probabilistic state fusion requirements](09_probabilistic_state_fusion_requirements.md): what was required, what was delivered
- [Probabilistic state fusion architecture](10_probabilistic_state_fusion_architecture.md): package layout, data flow, degradation modes
- [Probabilistic state fusion implementation order](11_probabilistic_state_fusion_implementation_order.md): delivery status, gaps, validation sequence
- [**Probabilistic fusion deep dive**](12_probabilistic_fusion_deep_dive.md): full math, worked example, artifact reading guide
- [**Local analytics math and methodology**](13_local_analytics_math_methodology.md): diagnostic equations, quality scoring, and failure interpretation
- [**Future directions: not-yet-implemented advanced themes**](18_future_directions.md): full cross-modal SSL, environmental field models (RF/gas/acoustic), calibration and formal contradiction modeling, global cross-mission threat inference
- [**Temporal SSL and track-aware representation learning**](14_temporal_ssl_physical_state.md): implementation deep dive for the track-pair and cycle-consistency SSL upgrade
- [**Threat primitives and local inference**](15_threat_primitives_local_inference.md): structured evidence-gated threat signals from physical state, the two-source gate, and why free-text hazards are insufficient for decision-making
- [**Drone detection runbook**](../runbooks/drone-detection.md): operational guide for step 30 — YOLOv8n training, hard negative injection, ONNX fp32/int8 export, RKNN NPU conversion, and edge inference on Cortex-A76 and RV1106G3
- [**coop_pilot IoT edge monitoring**](16_coop_pilot_iot_edge_monitoring.md): continuous site-awareness layer with MQTT sensor ingestion, LoRaWAN decoding, Frigate event handling, RTSP bridge sessions, acoustic analysis, scene synthesis, and realtime threat-sector integration
- [**Essential technology stack**](17_essential_technology_stack.md): extended guide covering pipeline.core shared modules (env, preflight, sidecars, logging), API/worker/DB/Qdrant service flow, fail-closed security (APP_ENV, HMAC-SHA256, constant-time comparison, bounded rate-limit buckets), FFmpeg and sidecar JSONL, CLIP/DINO named-vector retrieval, all VLMs and detector families, Kalman/Mahalanobis/Hungarian/RTS fusion math, SfM and Gaussian Splat mapping, 32-step local runner startup contract, analysis_summary.json analytics, Utilyze profiling, SSL fine-tuning, distillation, ONNX/RKNN edge export, and realtime/coop technologies (MediaMTX, Mosquitto, ChirpStack LoRaWAN, Frigate)

## Related Repo Docs

- [Pipeline architecture](../pipeline.md)
- [Architecture](../architecture.md)
- [coop_pilot getting started](../coop/getting-started.md)
- [coop_pilot integration](../coop/integration.md)
- [Configuration](../configuration.md)
- [Setup](../setup.md)
- [3D Gaussian Splat](../gaussian_splat.md)
- [Design docs](../design)
- [ADRs](../adr/README.md)
