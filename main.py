"""Entry point for the selfsuvis CLI.

Modes:
  file    — process a video file or directory (default)
  stream  — process a live RTSP/device stream
  demo    — run the full end-to-end demonstration pipeline
"""

import argparse

from pipeline.cli_parser import build_parser


def _setup_demo_env(args: argparse.Namespace) -> None:
    """Set env vars from demo args before pipeline.config is imported."""
    import os
    from pathlib import Path

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("DATA_DIR", str(output_dir))
    os.environ.setdefault("MODEL_NAME", "dinov3")
    os.environ.setdefault("QDRANT_HOST", "localhost")
    os.environ.setdefault("QDRANT_PORT", "6333")
    os.environ.setdefault("QDRANT_COLLECTION", "demo_video_semantic")
    os.environ.setdefault("DEVICE", args.device)
    os.environ.setdefault("USE_FP16", "false")
    os.environ.setdefault("SAMPLE_FPS_MAX", str(args.fps))
    os.environ.setdefault("SFM_FPS", "1")
    os.environ.setdefault("ALLOWED_INDEX_PATHS", "")
    os.environ.setdefault("API_KEY", "")
    os.environ.setdefault("ASR_ENABLED",         "true" if args.asr       else "false")
    os.environ.setdefault("ASR_MODEL",            args.asr_model)
    os.environ.setdefault("ASR_LANGUAGE",         args.asr_language)
    os.environ.setdefault("OCR_ENABLED",          "true" if args.ocr       else "false")
    os.environ.setdefault("OCR_MODEL",            args.ocr_model)
    os.environ.setdefault("DEPTH_ENABLED",        "true" if args.depth     else "false")
    os.environ.setdefault("DEPTH_MODEL",          args.depth_model)
    os.environ.setdefault("DETECTION_ENABLED",    "true" if args.detection else "false")
    os.environ.setdefault("DETECTION_MODEL",      args.detection_model)
    os.environ.setdefault("DETECTION_LABELS",     args.detection_labels)
    os.environ.setdefault("WORLD_MODEL_ENABLED",  "true" if args.world_model else "false")
    os.environ.setdefault("WORLD_MODEL",          args.world_model_id)
    if args.qwen_api_url:
        os.environ["QWEN_API_URL"] = args.qwen_api_url
    os.environ.setdefault("QWEN_API_URL", "")
    if args.qwen_model:
        os.environ["QWEN_MODEL"] = args.qwen_model
    if args.qwen_backend:
        os.environ["QWEN_BACKEND"] = args.qwen_backend


def main() -> None:
    parser = build_parser()
    args: argparse.Namespace = parser.parse_args()

    if args.mode == "demo":
        # Env vars must be set before pipeline.config is imported.
        _setup_demo_env(args)
        from pipeline.demo_runner import run_demo  # noqa: PLC0415
        run_demo(args)
    elif args.mode == "file":
        from pipeline.cli_runner import run_file_mode  # noqa: PLC0415
        run_file_mode(args)
    else:
        from pipeline.cli_runner import run_stream_mode  # noqa: PLC0415
        run_stream_mode(args)


if __name__ == "__main__":
    main()
