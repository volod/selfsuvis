# Learning Path Deep Dives

This directory is the human-oriented companion to [`local_path.md`](../quickstart/local_path.md),
[architecture.md](../reference/architecture.md), and [pipeline.md](../reference/pipeline.md).

Use these docs when you want to understand the system as a person, not just run it.
They are written to answer:

- What is this stage for?
- What evidence does it create?
- What artifact should I inspect?
- What later stage depends on it?
- What usually goes wrong?

`local_path.md` is the quick route. This directory is the slower, explanatory route.

## Document Index

| # | Document | Purpose |
|---|---|---|
| 00 | [00_day_by_day_syllabus.md](00_day_by_day_syllabus.md) | Multi-week human study plan with prerequisites, exercises, checkpoints, and milestones |
| 01 | [01_runtime_and_study_guide.md](01_runtime_and_study_guide.md) | Best entry point: how the current repo is organized, what runs in production vs local mode, and how to study without getting lost |
| 02 | [02_perception_core_steps_01_08.md](02_perception_core_steps_01_08.md) | Frame extraction, embeddings, Gemma, Florence, ASR, OCR, depth, and detection -- with key concepts, artifacts, and failure modes |
| 03 | [03_sensor_fusion_fundamentals.md](03_sensor_fusion_fundamentals.md) | Knowledge session on clocks, calibration, uncertainty, contradiction handling, and what fusion means in the current `selfsuvis` architecture |
| 04 | [04_sensor_steps_09_20.md](04_sensor_steps_09_20.md) | Optional physical sensor families and fusion thinking -- useful when sidecar data exists, ignorable when it does not |
| 05 | [05_tracking_mapping_steps_21_27.md](05_tracking_mapping_steps_21_27.md) | Semantic graph construction, Gemma-directed tracking, world-model context, Qwen, UniDriveVLA, and 3D mapping |
| 06 | [06_adaptation_eval_steps_28_35.md](06_adaptation_eval_steps_28_35.md) | Fine-tuning, distillation, drone detection edge training, ONNX export, retrieval evaluation, synthesis, and audit -- the "did the system improve?" phase |
| 07 | [07_agentic_knowledge_flow.md](07_agentic_knowledge_flow.md) | `VideoKnowledge` structure, evidence accumulation, context reuse, contamination risks, and debugging strategy |
| 08 | [08_local_run_artifact_analysis.md](08_local_run_artifact_analysis.md) | How to inspect a completed run, detect silent failures, and connect artifacts back to code |
| 09 | [09_probabilistic_state_fusion_requirements.md](09_probabilistic_state_fusion_requirements.md) | Requirements for probabilistic state fusion, with implementation status for each requirement |
| 10 | [10_probabilistic_state_fusion_architecture.md](10_probabilistic_state_fusion_architecture.md) | Actual subsystem architecture: package layout, data flow, layer responsibilities, configuration, degradation modes |
| 11 | [11_probabilistic_state_fusion_implementation_order.md](11_probabilistic_state_fusion_implementation_order.md) | Delivery status for all five phases, what is still missing, validation sequence for future extensions |
| 12 | [12_probabilistic_fusion_deep_dive.md](12_probabilistic_fusion_deep_dive.md) | Mathematical deep dive: Kalman filter equations, Umeyama Sim(3) derivation, RTS smoother, Mahalanobis gating, Hungarian assignment, semantic noise priors, worked example, artifact reading guide |
| 13 | [13_local_analytics_math_methodology.md](13_local_analytics_math_methodology.md) | The math and interpretation rules behind local-run diagnostics |
| 14 | [14_temporal_ssl_physical_state.md](14_temporal_ssl_physical_state.md) | Track-aware SSL: why frame augmentation is insufficient, how RF-DETR track IDs produce identity-consistent positive pairs, and how cycle-consistency loss prevents embedding drift along long tracks |
| 15 | [15_threat_primitives_local_inference.md](15_threat_primitives_local_inference.md) | Threat primitive layer: structured evidence-gated threat signals from physical state + fusion; schema design, the two-source gate, and why free-text hazards are insufficient |
| 16 | [16_coop_pilot_iot_edge_monitoring.md](16_coop_pilot_iot_edge_monitoring.md) | IoT edge monitoring deep dive: MQTT, LoRaWAN/ChirpStack, Frigate, MediaMTX RTSP bridge, acoustic analysis, rolling site state, scene synthesis, and realtime threat ingestion |
| 17 | [17_essential_technology_stack.md](17_essential_technology_stack.md) | Extended human guide to every essential technology in the stack: pipeline.core shared modules, API/worker/DB/Qdrant service flow, security boundaries, FFmpeg and sidecar JSONL, CLIP/DINO embeddings, all vision/language models, temporal state and fusion, 33-step local runner, SSL fine-tuning, distillation, ONNX/RKNN edge export, drone detection training, drone audio CNN training, and realtime/coop stack |
| 18 | [18_future_directions.md](18_future_directions.md) | Not-yet-implemented advanced themes: full cross-modal temporal SSL, environmental field models, calibration and contradiction handling, global cross-mission threat inference |
| 19 | [19_drone_audio_detection.md](19_drone_audio_detection.md) | Step 32 deep dive: DroneAudioCNN binary classifier, MFCC feature extraction (scipy-only STFT + mel filterbank + DCT), Conv2d/BN/ReLU/AdaptiveAvgPool architecture (~52k params), ONNX export for edge inference, dataset split, inference example, physics-based audio simulation |
| 20 | [20_drau_range_eval.md](20_drau_range_eval.md) | Step 33 deep dive: drau physics model (github.com/volod/drau), inverse-square amplitude scaling, ISO 9613-1 atmospheric absorption, range-detection curve, standalone edge script (numpy+scipy+onnxruntime only) |

## Probabilistic State Fusion -- Quick Reference

The fusion subsystem lives in `src/selfsuvis/pipeline/fusion/` and is fully
implemented. The four active layers are:

1. **Semantic priors** -- Gemma/Qwen/RSSM scene type -> noise scale factors
2. **Platform Kalman** -- GPS + IMU + baro -> position/velocity posterior
3. **Map-state fusion** -- adds SfM visual-pose constraints + RTS trajectory smoothing
4. **Object-state fusion** -- per-track Kalman + Mahalanobis gating + Hungarian assignment + RTS

Output artifact: `full_state_fusion.json` in each video's output directory.

Entry point for the math: [12_probabilistic_fusion_deep_dive.md](12_probabilistic_fusion_deep_dive.md).

## Current Runtime vs Conceptual Path

The current monolithic local runner reports **34 runtime/post-run steps** (Step 33
is the drau range-detection evaluation added for edge-device acoustic testing;
Step 34 is the global model advisor run after all videos).
Some older learning-path documents still group the system using a broader
**36-step conceptual map** or the previous 33-step numbering.
That is intentional:

- the **runtime** view matches the current code in `src/selfsuvis/pipeline/workflows/local/runner.py`
- the **conceptual** view keeps more granular study buckets so related ideas stay separate for a learner

Read the documents as a study decomposition, not a promise that every numbered conceptual step
is a separate top-level function call in the current runner.

## Step-To-Document Map

| Conceptual pipeline phase | Steps | Deep-dive document |
|---|---|---|
| Perception core | 1-8 | [02_perception_core_steps_01_08.md](02_perception_core_steps_01_08.md) |
| Physical sensors | 9-20 | [04_sensor_steps_09_20.md](04_sensor_steps_09_20.md) |
| Tracking and mapping | 21-27 | [05_tracking_mapping_steps_21_27.md](05_tracking_mapping_steps_21_27.md) |
| Adaptation and audit | 28-36 | [06_adaptation_eval_steps_28_35.md](06_adaptation_eval_steps_28_35.md) |
| Drone visual detection (step 30) | 30 | [Drone detection runbook](../runbooks/drone-detection.md) |
| Drone audio training (step 32) | 32 | [19_drone_audio_detection.md](19_drone_audio_detection.md) |
| drau range evaluation (step 33) | 33 | [20_drau_range_eval.md](20_drau_range_eval.md) |
| coop_pilot IoT edge monitoring | 37-43 | [16_coop_pilot_iot_edge_monitoring.md](16_coop_pilot_iot_edge_monitoring.md) |

Supporting sessions (not tied to a specific pipeline step number):

| Topic cluster | Document |
|---|---|
| Sensor fusion theory | [03_sensor_fusion_fundamentals.md](03_sensor_fusion_fundamentals.md) -- read before or during steps 9-20 |
| Agentic context flow | [07_agentic_knowledge_flow.md](07_agentic_knowledge_flow.md) -- cross-cutting; read after steps 1-8 |
| Run artifact inspection | [08_local_run_artifact_analysis.md](08_local_run_artifact_analysis.md) -- how to read a completed run |
| Probabilistic fusion requirements | [09_probabilistic_state_fusion_requirements.md](09_probabilistic_state_fusion_requirements.md) |
| Probabilistic fusion architecture | [10_probabilistic_state_fusion_architecture.md](10_probabilistic_state_fusion_architecture.md) |
| Probabilistic fusion delivery status | [11_probabilistic_state_fusion_implementation_order.md](11_probabilistic_state_fusion_implementation_order.md) |
| Probabilistic fusion math | [12_probabilistic_fusion_deep_dive.md](12_probabilistic_fusion_deep_dive.md) -- Kalman, RTS, Hungarian |
| Local run analytics | [13_local_analytics_math_methodology.md](13_local_analytics_math_methodology.md) |
| Temporal SSL | [14_temporal_ssl_physical_state.md](14_temporal_ssl_physical_state.md) -- track-aware self-supervision |
| Threat primitives | [15_threat_primitives_local_inference.md](15_threat_primitives_local_inference.md) |
| Essential technology stack | [17_essential_technology_stack.md](17_essential_technology_stack.md) -- comprehensive reference |
| Future directions | [18_future_directions.md](18_future_directions.md) -- not-yet-implemented extensions |

## How To Read These Docs

1. Start with [01_runtime_and_study_guide.md](01_runtime_and_study_guide.md).
2. Skim [00_day_by_day_syllabus.md](00_day_by_day_syllabus.md) if you want a paced study plan instead of a reference manual.
3. Skim [`local_path.md`](../quickstart/local_path.md) for the fast path.
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

## Recommended Reading Order for New Engineers

Follow this sequence to go from "what is this repo?" to "how does evidence move?" before
branching into adaptation, physical-world modeling, and advanced global-threat work.

1. [01_runtime_and_study_guide.md](01_runtime_and_study_guide.md) -- orientation
2. [02_perception_core_steps_01_08.md](02_perception_core_steps_01_08.md) -- steps 1-8
3. [07_agentic_knowledge_flow.md](07_agentic_knowledge_flow.md) -- how context flows
4. [03_sensor_fusion_fundamentals.md](03_sensor_fusion_fundamentals.md) -- theory
5. [04_sensor_steps_09_20.md](04_sensor_steps_09_20.md) -- steps 9-20
6. [05_tracking_mapping_steps_21_27.md](05_tracking_mapping_steps_21_27.md) -- steps 21-27
7. [06_adaptation_eval_steps_28_35.md](06_adaptation_eval_steps_28_35.md) -- steps 28-35
8. [19_drone_audio_detection.md](19_drone_audio_detection.md) -- step 32
9. [20_drau_range_eval.md](20_drau_range_eval.md) -- step 33 (drau edge evaluation)
10. [08_local_run_artifact_analysis.md](08_local_run_artifact_analysis.md) -- inspect a run
11. [13_local_analytics_math_methodology.md](13_local_analytics_math_methodology.md) -- diagnostics
12. [09_probabilistic_state_fusion_requirements.md](09_probabilistic_state_fusion_requirements.md)
13. [12_probabilistic_fusion_deep_dive.md](12_probabilistic_fusion_deep_dive.md) -- math
14. [14_temporal_ssl_physical_state.md](14_temporal_ssl_physical_state.md) -- temporal SSL
15. [15_threat_primitives_local_inference.md](15_threat_primitives_local_inference.md)
16. [16_coop_pilot_iot_edge_monitoring.md](16_coop_pilot_iot_edge_monitoring.md) -- IoT edge
17. [17_essential_technology_stack.md](17_essential_technology_stack.md) -- full reference
18. [18_future_directions.md](18_future_directions.md) -- where to go next

## Related Repo Docs

- [Pipeline architecture](../reference/pipeline.md)
- [Architecture](../reference/architecture.md)
- [coop_pilot getting started](../coop/getting-started.md)
- [coop_pilot integration](../coop/integration.md)
- [Configuration](../reference/configuration.md)
- [Setup](../quickstart/setup.md)
- [3D Gaussian Splat](../reference/gaussian_splat.md)
- [ADRs](../adr/README.md)
