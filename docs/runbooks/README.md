# Runbooks

Operations runbooks for every model in the local pipeline.

## Model runbooks

| Runbook | Pipeline step | On by default | Mode |
|---|---|---|---|
| [clip-dino.md](clip-dino.md) | B — CLIP + DINOv3 embeddings | Yes | Local weights |
| [florence-2.md](florence-2.md) | L — Florence-2 captioning | Yes | Local weights or vLLM sidecar |
| [gemma-api.md](gemma-api.md) | J — Gemma captioning + P3 scene analysis | Optional | Ollama / vLLM sidecar |
| [whisper-asr.md](whisper-asr.md) | M — Whisper ASR transcription | No (`ASR_ENABLED`) | Local weights |
| [ocr.md](ocr.md) | N — OCR text extraction | No (`OCR_ENABLED`) | Local weights or vLLM sidecar |
| [depth.md](depth.md) | O — Monocular depth estimation | No (`DEPTH_ENABLED`) | Local weights |
| [detection-hf.md](detection-hf.md) | P — HF object detection (RT-DETR / Grounding DINO) | No (`DETECTION_ENABLED`) | Local weights |
| [yolo-sam.md](yolo-sam.md) | P2 — YOLO11 detection + SAM2/3 segmentation | Yes | Local weights |
| [rfdetr-tracking.md](rfdetr-tracking.md) | P3 — Gemma-directed RF-DETR tracking | Yes (requires Gemma) | Local weights + Gemma sidecar |
| [world-model.md](world-model.md) | Q — World model video embeddings | No (`WORLD_MODEL_ENABLED`) | Local weights |
| [qwen-api.md](qwen-api.md) | R — Qwen2.5-VL detailed captioning | Optional | Ollama / vLLM sidecar |
| [unidrive-api.md](unidrive-api.md) | S — UniDriveVLA expert analysis | No (`UNIDRIVE_ENABLED`) | Any OpenAI-compatible sidecar |

## Quick: enable all optional sidecars

```bash
python main.py --mode local \
  --gemma-api-url    http://localhost:11434/v1 \
  --qwen-api-url     http://localhost:8010/v1 \
  --unidrive-api-url http://localhost:8010/v1
```

## Quick: download all model weights

```bash
python scripts/prepare_models.py --all
```

## Step-to-runbook map

| Step ID | Description | Runbook |
|---|---|---|
| B | CLIP + DINOv3 | [clip-dino.md](clip-dino.md) |
| J | Gemma captioning | [gemma-api.md](gemma-api.md) |
| L | Florence-2 captioning | [florence-2.md](florence-2.md) |
| M | ASR (Whisper) | [whisper-asr.md](whisper-asr.md) |
| N | OCR | [ocr.md](ocr.md) |
| O | Depth estimation | [depth.md](depth.md) |
| P | HF object detection | [detection-hf.md](detection-hf.md) |
| P2 | YOLO11 + SAM2/3 | [yolo-sam.md](yolo-sam.md) |
| P3 | Gemma-directed RF-DETR tracking | [rfdetr-tracking.md](rfdetr-tracking.md) |
| Q | World model embeddings | [world-model.md](world-model.md) |
| R | Qwen2.5-VL detailed captioning | [qwen-api.md](qwen-api.md) |
| S | UniDriveVLA expert analysis | [unidrive-api.md](unidrive-api.md) |
