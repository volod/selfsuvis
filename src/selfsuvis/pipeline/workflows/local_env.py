"""Local full-analysis environment setup helpers.

This module is intentionally dependency-light so it can be imported by `main.py`
before `pipeline.core.config` is imported. This ensures local orchestration CLI flags are reflected
in environment variables consumed by settings initialization.
"""


import os
from pathlib import Path
from typing import Any


def normalize_local_orchestration_args(args: Any) -> Any:
    """Apply default-on local orchestration flags unless explicitly overridden."""
    if getattr(args, "mode", "") != "local":
        return args

    default_true_flags = (
        "asr",
        "ocr",
        "depth",
        "detection",
        "world_model",
        "qwen",
        "unidrive",
    )
    for flag in default_true_flags:
        if getattr(args, flag, None) is None:
            setattr(args, flag, True)
    return args


def apply_local_env(args: Any) -> None:
    """Set environment variables for local full-analysis mode.

    Args:
        args: Parsed argparse namespace for local full-analysis mode.
    """
    normalize_local_orchestration_args(args)

    # Load the project-root .env FIRST so its values are visible to the
    # setdefault calls below.  Explicit CLI args (direct os.environ assignments
    # further down) still override .env values.
    try:
        from dotenv import load_dotenv as _load_dotenv  # noqa: PLC0415
        _load_dotenv()
    except ImportError:
        pass

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Propagate HF_TOKEN → HUGGING_FACE_HUB_TOKEN so transformers / huggingface_hub
    # authenticate automatically for gated models (Gemma, Llama, …).
    _hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if _hf:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("DATA_DIR", str(output_dir))
    os.environ.setdefault("MODEL_NAME", "gemma")
    os.environ.setdefault("GEMMA_MODEL_ID", "google/gemma-4-it-2b")
    os.environ.setdefault("GEMMA_USE_BF16", "true" if args.device != "cpu" else "false")
    os.environ.setdefault("QDRANT_HOST", "localhost")
    os.environ.setdefault("QDRANT_PORT", "6333")
    os.environ.setdefault("QDRANT_COLLECTION", "local_video_semantic")
    os.environ.setdefault("DEVICE", args.device)
    os.environ.setdefault("USE_FP16", "false")
    os.environ.setdefault("SAMPLE_FPS_MAX", str(args.fps))
    os.environ.setdefault("SFM_FPS", "1")
    os.environ.setdefault("ALLOWED_INDEX_PATHS", "")
    os.environ.setdefault("API_KEY", "")
    # Model enable flags: always force-set from CLI args so --asr/--no-asr etc.
    # take precedence over any pre-existing env var (e.g. from a Docker session).
    # setdefault would silently ignore a pre-set ASR_ENABLED=false.
    os.environ["ASR_ENABLED"] = "true" if args.asr else "false"
    os.environ.setdefault("ASR_MODEL", args.asr_model)
    os.environ.setdefault("ASR_LANGUAGE", args.asr_language)
    os.environ["OCR_ENABLED"] = "true" if args.ocr else "false"
    os.environ.setdefault("OCR_MODEL", args.ocr_model)
    os.environ["DEPTH_ENABLED"] = "true" if args.depth else "false"
    os.environ.setdefault("DEPTH_MODEL", args.depth_model)
    os.environ["DETECTION_ENABLED"] = "true" if args.detection else "false"
    os.environ.setdefault("DETECTION_MODEL", args.detection_model)
    os.environ.setdefault("DETECTION_LABELS", args.detection_labels)
    # YOLO11 + SAM2/3 (step P2) — enabled by default; opt out with --no-yolo / --no-sam
    _no_yolo = getattr(args, "no_yolo", False)
    os.environ["YOLO_ENABLED"] = "false" if _no_yolo else "true"
    _yolo_model = getattr(args, "yolo_model", "yolo11l") or "yolo11l"
    os.environ.setdefault("YOLO_MODEL", _yolo_model)
    _no_sam = getattr(args, "no_sam", False)
    os.environ["SAM_ENABLED"] = "false" if _no_sam else "true"
    _sam_model = getattr(args, "sam_model", "auto") or "auto"
    os.environ.setdefault("SAM_MODEL", _sam_model)
    # Gemma directed tracking (step P3) — enabled by default; opt out with --no-rfdetr
    _no_rfdetr = getattr(args, "no_rfdetr", False)
    os.environ["RFDETR_ENABLED"] = "false" if _no_rfdetr else "true"
    _rfdetr_model = getattr(args, "rfdetr_model", "base") or "base"
    os.environ.setdefault("RFDETR_MODEL", _rfdetr_model)
    os.environ["WORLD_MODEL_ENABLED"] = "true" if args.world_model else "false"
    # Only force-set WORLD_MODEL when the user explicitly passed --world-model-id;
    # if it's still "auto", let the .env value (loaded above) take precedence.
    if args.world_model_id != "auto":
        os.environ["WORLD_MODEL"] = args.world_model_id
    else:
        os.environ.setdefault("WORLD_MODEL", "auto")

    if args.qwen_api_url:
        os.environ["QWEN_API_URL"] = args.qwen_api_url
    os.environ.setdefault("QWEN_API_URL", "")
    if args.qwen_model:
        os.environ["QWEN_MODEL"] = args.qwen_model
    if args.qwen_backend:
        os.environ["QWEN_BACKEND"] = args.qwen_backend
    # Force-set (not setdefault) so --unidrive / --no-unidrive always wins over
    # any pre-existing UNIDRIVE_ENABLED=false in the shell or a prior Docker session.
    os.environ["UNIDRIVE_ENABLED"] = "true" if args.unidrive else "false"
    if getattr(args, "unidrive_api_url", ""):
        os.environ["UNIDRIVE_API_URL"] = args.unidrive_api_url
    os.environ.setdefault("UNIDRIVE_API_URL", "")
    if getattr(args, "unidrive_model", ""):
        os.environ["UNIDRIVE_MODEL"] = args.unidrive_model
    if getattr(args, "unidrive_backend", ""):
        os.environ["UNIDRIVE_BACKEND"] = args.unidrive_backend
    # SceneTok — force-set the enable flag and propagate sidecar URL + checkpoint.
    os.environ["SCENETOK_ENABLED"] = "true" if getattr(args, "scenetok", False) else "false"
    if getattr(args, "scenetok_api_url", ""):
        os.environ["SCENETOK_API_URL"] = args.scenetok_api_url
    os.environ.setdefault("SCENETOK_API_URL", "")
    if getattr(args, "scenetok_checkpoint", ""):
        os.environ["SCENETOK_CHECKPOINT"] = args.scenetok_checkpoint

    gemma_api_url = getattr(args, "gemma_api_url", "")
    if gemma_api_url:
        os.environ["GEMMA_API_URL"] = gemma_api_url
    os.environ.setdefault("GEMMA_API_URL", "")
    gemma_api_model = getattr(args, "gemma_api_model", "")
    if gemma_api_model:
        os.environ["GEMMA_API_MODEL"] = gemma_api_model
    gemma_api_backend = getattr(args, "gemma_api_backend", "")
    if gemma_api_backend:
        os.environ["GEMMA_API_BACKEND"] = gemma_api_backend

    reasoning_api_url = getattr(args, "reasoning_api_url", "")
    if reasoning_api_url:
        os.environ["REASONING_API_URL"] = reasoning_api_url
    reasoning_model = getattr(args, "reasoning_model", "")
    if reasoning_model:
        os.environ["REASONING_MODEL"] = reasoning_model
    reasoning_backend = getattr(args, "reasoning_backend", "")
    if reasoning_backend:
        os.environ["REASONING_BACKEND"] = reasoning_backend

    florence_api_url = getattr(args, "florence_api_url", "")
    if florence_api_url:
        os.environ["FLORENCE_API_URL"] = florence_api_url
    florence_model = getattr(args, "florence_model", "")
    if florence_model:
        os.environ["FLORENCE_MODEL"] = florence_model
