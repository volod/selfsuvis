# selfsuvis — Outdoor Autonomy Perception Stack

Spatial memory engine for outdoor robotics. Ingest mission video from drones, rovers, or vehicles → extract frames → estimate camera poses (pycolmap SfM) → build dense 3D maps (nerfstudio splatfacto) → embed frames (OpenCLIP + DINOv3) → caption with Florence-2 → store in PostgreSQL + Qdrant → search by text or image query.

Self-improvement loop: each mission auto-tags uncertain and novel frames for annotation, building training data for future self-supervised model fine-tuning and edge distillation.

## Local run pipeline

The local pipeline is a 35-step research and training workflow that goes well beyond the Docker stack. It processes a mission video end-to-end through four phases:

| Phase | Steps | What happens |
|---|---|---|
| **Perception core** | 1–8 | Frame extraction, CLIP+DINOv3 embedding, Gemma scene analysis, Florence-2 captioning, Whisper ASR, OCR, depth estimation, object detection |
| **Sensor fusion** | 9–20 | Optional physical sensor sidecars — RF/SDR, thermal, multispectral, event camera, LiDAR, radar, GNSS-R, IMU, barometer, atmospheric, gas/radiation, acoustic — fused into a single time-aligned context block |
| **Tracking and 3D mapping** | 21–27 | YOLO+SAM segmentation, Gemma-directed RF-DETR tracking, world model video embeddings, Qwen+UniDriveVLA dense captioning, pycolmap SfM + nerfstudio 3D Gaussian Splat |
| **Adaptation and evaluation** | 28–35 | SSL fine-tuning, edge model distillation (ONNX), multi-model comparison, Qwen3 reasoning audit, aggregate statistics |

The local pipeline is the primary path for understanding how the system works, building training datasets, and adapting models to a new domain. The Docker stack runs steps 1, 2, 4, and 7–8 continuously as a production service. See the [local learning path](docs/local_path.md) for step-by-step guidance.

Recent local-run builds also apply adaptive runtime controls by default to keep single-video analysis practical on 16 GiB GPUs:

- OCR is prescreened from Florence caption confidence before invoking the OCR sidecar.
- Qwen detailed captioning uses bounded sampled-frame selection instead of captioning every frame.
- Depth `auto` now prefers a fast local profile unless you explicitly switch back to a quality-oriented model.
- The final reasoning audit uses a simple-first flow and only falls back to a second attempt when the first output is incomplete.

---

## Documents

| Document | Contents |
|---|---|
| [Quick start](docs/quickstart.md) | Run the stack with Docker or locally, step by step |
| [Setup](docs/setup.md) | Detailed setup options, GPU, CVAT |
| [Configuration](docs/configuration.md) | All env vars with defaults and security notes |
| [API reference](docs/api.md) | HTTP endpoints, robot pose API |
| [UI guide](docs/ui.md) | Streamlit UI usage |
| [Architecture](docs/architecture.md) | System components and service topology |
| [Pipeline](docs/pipeline.md) | Agentic pipeline architecture and data flow |
| [Data layout](docs/data_layout.md) | Directory structure, sensor sidecars, output artifacts |
| [Examples](docs/examples.md) | Example queries and workflows |
| [Performance](docs/performance.md) | Latency targets and tuning |
| [Troubleshooting](docs/troubleshooting.md) | Common errors and fixes |
| [Tests](docs/tests.md) | Unit and integration test guide |
| [Development](docs/develop.md) | Contributing, code style, project conventions |
| [Runbooks](docs/runbooks/README.md) | Per-component operational runbooks |
| [Local learning path](docs/local_path.md) | 35-step essentials + day-by-day syllabus |
| [Learning path deep dives](docs/learning_path/README.md) | Detailed study set per pipeline phase |
| [Analytics and visualization](docs/analytics.md) | Post-run artifact analysis, charts, HTML report, and CLI usage |
| [Architecture decisions](docs/adr/README.md) | ADR log |
| [Design docs](docs/design/outdoor-autonomy-perception-stack.md) | Original design document |
