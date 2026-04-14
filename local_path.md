# Local Learning Path

This is the short version of the local `python main.py --mode local` study path.
Use it when you want the essentials for each step first, then jump into deeper material only where needed.

Deep-dive entry points:

- [Learning path index](docs/learning_path/README.md)
- [Runtime and study guide](docs/learning_path/01_runtime_and_study_guide.md)
- [Perception core, Steps 1-8](docs/learning_path/02_perception_core_steps_01_08.md)
- [Physical sensors and fusion, Steps 9-20](docs/learning_path/03_sensor_steps_09_20.md)
- [Tracking, world models, and 3D mapping, Steps 21-27](docs/learning_path/04_tracking_mapping_steps_21_27.md)
- [Adaptation, evaluation, and audit, Steps 28-35](docs/learning_path/05_adaptation_eval_steps_28_35.md)
- [Agentic knowledge flow](docs/learning_path/06_agentic_knowledge_flow.md)
- [Day-by-day syllabus](docs/learning_path/07_day_by_day_syllabus.md)

## How To Use This Path

1. Read the step summary below once from top to bottom.
2. Pick the deep-dive document for the phase you are working on.
3. Use the day-by-day syllabus if you want a realistic study plan instead of reading everything at once.

## The 35 Steps, Short Version

| Step | Essential purpose | Go deeper |
|---|---|---|
| 1. Frame extraction | Decode the video into frames. This fixes the temporal resolution for everything else. | [Perception core](docs/learning_path/02_perception_core_steps_01_08.md#step-1-frame-extraction) |
| 2. Vector store indexing | Embed frames with CLIP and DINO, then index them for retrieval and comparison. | [Perception core](docs/learning_path/02_perception_core_steps_01_08.md#step-2-vector-store-indexing) |
| 3. Gemma multimodal analysis | Build video-level scene understanding, scene changes, and text-image retrieval context. | [Perception core](docs/learning_path/02_perception_core_steps_01_08.md#step-3-gemma-multimodal-analysis) |
| 4. Florence captioning | Produce a readable scene caption for each key frame. | [Perception core](docs/learning_path/02_perception_core_steps_01_08.md#step-4-florence-scene-captioning) |
| 5. ASR transcription | Turn speech into timestamped text aligned with the video. | [Perception core](docs/learning_path/02_perception_core_steps_01_08.md#step-5-asr-transcription) |
| 6. OCR text extraction | Recover visible text from signs, dashboards, overlays, and labels. | [Perception core](docs/learning_path/02_perception_core_steps_01_08.md#step-6-ocr-text-extraction) |
| 7. Depth estimation | Add a cheap geometric prior: near, far, cluttered, open. | [Perception core](docs/learning_path/02_perception_core_steps_01_08.md#step-7-depth-estimation) |
| 8. Object detection | Turn scenes into object instances with boxes and labels. | [Perception core](docs/learning_path/02_perception_core_steps_01_08.md#step-8-object-detection) |
| 9. RF / SDR sensing | Inspect the radio environment around the mission when IQ data exists. | [Sensors and fusion](docs/learning_path/03_sensor_steps_09_20.md#step-9-rf--sdr-sensing) |
| 10. Thermal sensing | Add heat signatures that RGB alone cannot reveal. | [Sensors and fusion](docs/learning_path/03_sensor_steps_09_20.md#step-10-thermal--infrared-imaging) |
| 11. Multispectral sensing | Add non-RGB bands for material and vegetation cues. | [Sensors and fusion](docs/learning_path/03_sensor_steps_09_20.md#step-11-multispectral--hyperspectral-imaging) |
| 12. Event camera sensing | Represent change as asynchronous events instead of normal frames. | [Sensors and fusion](docs/learning_path/03_sensor_steps_09_20.md#step-12-event-camera-neuromorphic-sensing) |
| 13. LiDAR sensing | Add active ranging and point geometry. | [Sensors and fusion](docs/learning_path/03_sensor_steps_09_20.md#step-13-lidar--active-ranging) |
| 14. Radar sensing | Add motion and range structure that works in poor visibility. | [Sensors and fusion](docs/learning_path/03_sensor_steps_09_20.md#step-14-radar-fmcw--doppler--sar) |
| 15. GNSS-R and satellite reception | Add signal-based environmental and traffic evidence beyond the camera. | [Sensors and fusion](docs/learning_path/03_sensor_steps_09_20.md#step-15-gnss-r-and-satellite-signal-reception) |
| 16. Inertial and barometric sensing | Add motion, orientation drift, and altitude pressure context. | [Sensors and fusion](docs/learning_path/03_sensor_steps_09_20.md#step-16-inertial-and-barometric-sensing) |
| 17. Atmospheric sensing | Add weather and ambient environmental context. | [Sensors and fusion](docs/learning_path/03_sensor_steps_09_20.md#step-17-atmospheric--environmental-sensing) |
| 18. Chemical / gas / radiation sensing | Add invisible hazard indicators. | [Sensors and fusion](docs/learning_path/03_sensor_steps_09_20.md#step-18-chemical--gas--radiation-sensing) |
| 19. Acoustic sensing | Add sound evidence from engines, speech, impacts, and ambience. | [Sensors and fusion](docs/learning_path/03_sensor_steps_09_20.md#step-19-acoustic-sensing) |
| 20. Sensor fusion | Merge all side channels into one time-aligned context block. | [Sensors and fusion](docs/learning_path/03_sensor_steps_09_20.md#step-20-sensor-fusion-analysis) |
| 21. YOLO + SAM detection and segmentation | Refine object localization and add masks for spatial structure. | [Tracking and mapping](docs/learning_path/04_tracking_mapping_steps_21_27.md#step-21-yolo--sam-detection-and-segmentation) |
| 22. Gemma directed tracking | Use language-guided context to focus tracking on what matters. | [Tracking and mapping](docs/learning_path/04_tracking_mapping_steps_21_27.md#step-22-gemma-directed-tracking) |
| 23. World model embeddings | Move from isolated frames to temporal clip representations. | [Tracking and mapping](docs/learning_path/04_tracking_mapping_steps_21_27.md#step-23-world-model-video-embeddings) |
| 24. Qwen detailed captioning | Build dense per-frame reasoning from accumulated context. | [Tracking and mapping](docs/learning_path/04_tracking_mapping_steps_21_27.md#step-24-qwen-detailed-captioning) |
| 25. UniDriveVLA expert analysis | Add domain-specific understanding, perception, and planning structure. | [Tracking and mapping](docs/learning_path/04_tracking_mapping_steps_21_27.md#step-25-unidrivevla-expert-analysis) |
| 26. Base model search test | Check whether the baseline embedding space retrieves useful neighbors. | [Tracking and mapping](docs/learning_path/04_tracking_mapping_steps_21_27.md#step-26-base-model-search-test) |
| 27. 3D map and Gaussian Splat | Turn 2D evidence into reusable geometry and scene structure. | [Tracking and mapping](docs/learning_path/04_tracking_mapping_steps_21_27.md#step-27-3d-map-and-gaussian-splat) |
| 28. SSL DINO fine-tuning | Adapt the representation to the current mission without labels. | [Adaptation and audit](docs/learning_path/05_adaptation_eval_steps_28_35.md#step-28-ssl-dino-fine-tuning) |
| 29. Knowledge distillation | Compress the strong teacher into a smaller deployment model. | [Adaptation and audit](docs/learning_path/05_adaptation_eval_steps_28_35.md#step-29-knowledge-distillation) |
| 30. ONNX export and gallery build | Package the adapted model for lightweight inference. | [Adaptation and audit](docs/learning_path/05_adaptation_eval_steps_28_35.md#step-30-onnx-export-and-gallery-build) |
| 31. Fine-tuned search test | Measure whether adaptation actually improved retrieval. | [Adaptation and audit](docs/learning_path/05_adaptation_eval_steps_28_35.md#step-31-fine-tuned-search-test) |
| 32. Model comparison and video description | Compare baseline vs adapted behavior and produce a clip-level summary. | [Adaptation and audit](docs/learning_path/05_adaptation_eval_steps_28_35.md#step-32-model-comparison-and-video-description) |
| 33. Multi-model comparison | Check agreement and disagreement across major multimodal analyzers. | [Adaptation and audit](docs/learning_path/05_adaptation_eval_steps_28_35.md#step-33-multi-model-comparison) |
| 34. Video synthesis | Turn many artifacts into one human-readable report. | [Adaptation and audit](docs/learning_path/05_adaptation_eval_steps_28_35.md#step-34-video-synthesis) |
| 35. Agentic flow audit | Explain how context moved through the pipeline and where risk can propagate. | [Adaptation and audit](docs/learning_path/05_adaptation_eval_steps_28_35.md#step-35-agentic-flow-audit) |

## Realistic Day-By-Day Syllabus

Use this if you want a practical study sequence instead of trying to absorb all 35 steps at once.
For the longer version, see [the full syllabus](docs/learning_path/07_day_by_day_syllabus.md).

| Day | Focus |
|---|---|
| 1 | Repo overview, `README.md`, runtime shape, local outputs |
| 2 | Step 1: video basics, FPS, sampling, `ffmpeg` |
| 3 | Step 2: embeddings, cosine similarity, vector stores |
| 4 | Step 3: Gemma multimodal reasoning and scene analysis |
| 5 | Step 4: captioning; compare human captions vs Florence |
| 6 | Steps 5-6: ASR and OCR as non-visual evidence |
| 7 | Steps 7-8: depth and detection as geometric/object structure |
| 8 | Step 9: RF basics, IQ, spectral features |
| 9 | Steps 10-12: thermal, multispectral, event cameras |
| 10 | Steps 13-15: LiDAR, radar, GNSS-R, satellite-derived signals |
| 11 | Steps 16-19: inertial, weather, gas/radiation, acoustic sensing |
| 12 | Step 20: time alignment and fusion design |
| 13 | Steps 21-22: segmentation, tracking, language-guided perception |
| 14 | Steps 23-25: video embeddings, Qwen, UniDriveVLA |
| 15 | Steps 26-27: retrieval sanity checks and 3D mapping |
| 16 | Step 28: self-supervised adaptation |
| 17 | Steps 29-30: distillation and export |
| 18 | Steps 31-33: evaluation and cross-model comparison |
| 19 | Steps 34-35: synthesis, audit, provenance |
| 20 | Re-run one small video end to end and inspect artifacts |
| 21 | Write your own step-by-step summary of what changed at each phase |

## Recommended Reading Order

If you only have time for a fast pass:

1. [Runtime and study guide](docs/learning_path/01_runtime_and_study_guide.md)
2. [Perception core](docs/learning_path/02_perception_core_steps_01_08.md)
3. [Sensors and fusion](docs/learning_path/03_sensor_steps_09_20.md)
4. [Tracking and mapping](docs/learning_path/04_tracking_mapping_steps_21_27.md)
5. [Adaptation and audit](docs/learning_path/05_adaptation_eval_steps_28_35.md)
6. [Agentic knowledge flow](docs/learning_path/06_agentic_knowledge_flow.md)
