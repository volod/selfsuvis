# selfsuvis project memory

## Key architecture
- Robotics/drone perception stack: video → frames → SfM → 3DGS → embeddings → Qdrant/PostgreSQL
- FastAPI app + async worker + Streamlit UI + PostgreSQL + Qdrant + MediaMTX
- Demo pipeline: 21-step runner in `pipeline/workflows/demo/runner.py`

## Gemma directed tracking (P3) — implemented
- New step P3 between P2 (YOLO+SAM) and Q (World model)
- Files: `pipeline/vision/rfdetr.py`, `pipeline/workflows/demo/steps_gemma_tracking.py`
- Settings: `RFDETR_ENABLED`, `RFDETR_MODEL` (base/large), `RFDETR_CONFIDENCE`
- CLI: `--no-rfdetr`, `--rfdetr-model`
- Flow: Gemma API → structured JSON (scene_type + objects + rough_bbox) → SAM box-prompts (Path A) or CLIP-filtered auto-masks (Path B) → RF-DETR tracking with IoU-based persistent IDs
- Production: `VideoIndexer._run_gemma_directed_tracking_pass()` in `indexer.py`
- Requirement: `rfdetr>=1.1.0` added to `requirements/requirements_prod.txt`

## Step numbering (demo pipeline)
Steps 1-21. P3 is step 10; World model Q moved to 11; Qwen R to 12; base search C to 13; 3D map I to 14; SSL D to 15; distillation E to 16; ONNX F to 17; finetuned search G to 18; comparison H to 19; synthesis Z to 20; agentic audit AA to 21.

## Test command
`make test-unit` or `.venv/bin/python -m pytest tests/unit/ -v`

## Config
All settings in `pipeline/core/config.py` as Settings class; reads env vars with defaults.
GEMMA_API_URL required for Gemma sidecar calls (Ollama/vLLM compatible).
