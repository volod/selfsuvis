"""Argument parser for the agentic video processing pipeline CLI."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(description="Video processing pipeline")
    parser.add_argument("--mode", choices=["file", "stream", "demo"], default="file")
    parser.add_argument("--input", help="Video file path")
    parser.add_argument("--dir", help="Directory containing videos")
    parser.add_argument(
        "--output-dir", default="data_test/videos_test", help="Output directory"
    )

    parser.add_argument(
        "--interval", type=float, default=1.0, help="Fixed interval in seconds"
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable adaptive frame sampling",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=1.0,
        help="Minimum interval for adaptive sampling",
    )
    parser.add_argument(
        "--max-gap", type=float, default=10.0, help="Max gap between kept frames"
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=0.12,
        help="Frame diff threshold",
    )
    parser.add_argument(
        "--probe-fps", type=float, default=5.0, help="Probe fps for adaptive sampling"
    )

    parser.add_argument(
        "--source", help="Stream source (URL or device index)"
    )
    parser.add_argument(
        "--stream-name", help="Name for stream output directory"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Max frames for stream mode"
    )
    parser.add_argument(
        "--steps",
        default="extract,describe,index",
        help="Comma-separated steps: extract,describe,index",
    )

    parser.add_argument(
        "--model-type",
        choices=["openclip_sam", "openclip_only"],
        default="openclip_sam",
        help="Vision model stack",
    )
    parser.add_argument(
        "--sam-checkpoint", help="Path to SAM checkpoint file"
    )
    parser.add_argument(
        "--sam-model-type",
        default=None,
        help="SAM model type (vit_h/vit_l/vit_b)",
    )
    parser.add_argument(
        "--labels-file", help="Path to label vocabulary file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging per frame",
    )

    parser.add_argument(
        "--es-url",
        help="Elasticsearch base URL, e.g. http://localhost:9200",
    )
    parser.add_argument(
        "--es-index", help="Elasticsearch index name"
    )

    # ── Demo mode args (used when --mode demo) ────────────────────────────────
    parser.add_argument(
        "--videos-dir", default="data_test/videos",
        help="[demo] Directory containing input .mp4/.mov/.mkv files",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="[demo] Torch device (auto selects CUDA when available)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="[demo] SSL fine-tuning epochs per video",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="[demo] SSL fine-tuning batch size",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="[demo] Nearest neighbours to show in search tests",
    )
    parser.add_argument(
        "--no-qdrant", action="store_true",
        help="[demo] Skip Qdrant; use in-memory cosine search",
    )
    parser.add_argument(
        "--no-sfm", action="store_true",
        help="[demo] Skip pycolmap SfM; use PCA point-cloud fallback",
    )
    parser.add_argument(
        "--no-gsplat", action="store_true",
        help="[demo] Skip 3D Gaussian Splatting (step I); keep sparse point-cloud only",
    )
    parser.add_argument(
        "--no-caption", action="store_true",
        help="[demo] Skip Florence-2 scene captioning (step L)",
    )
    parser.add_argument(
        "--florence-api-url", default="",
        help="[demo] vLLM endpoint serving Florence-2 (e.g. http://localhost:8020/v1). "
             "When set, Florence is called via API instead of loading locally — "
             "no local VRAM consumed. See README for vLLM setup instructions.",
    )
    parser.add_argument(
        "--florence-model", default="",
        help="[demo] Florence-2 model ID for vLLM API (default: microsoft/Florence-2-large)",
    )
    parser.add_argument(
        "--distill-epochs", type=int, default=5,
        help="[demo] Knowledge distillation epochs (student ViT-S/14)",
    )
    parser.add_argument(
        "--no-distill", action="store_true",
        help="[demo] Skip knowledge distillation; export teacher to ONNX instead",
    )
    parser.add_argument(
        "--no-onnx", action="store_true",
        help="[demo] Skip ONNX export",
    )
    parser.add_argument(
        "--fps", type=float, default=2.0,
        help="[demo] Frame-extraction rate (fps)",
    )
    parser.add_argument(
        "--view-npz", metavar="PATH", nargs="?", const="",
        help="[demo] Visualize existing sparse_map.npz without running pipeline",
    )
    parser.add_argument(
        "--no-view", action="store_true",
        help="[demo] Skip the interactive 3D map viewer",
    )
    # Optional multimodal steps
    parser.add_argument("--asr", action="store_true",
                        help="[demo] Enable Whisper ASR speech-to-text (step M)")
    parser.add_argument("--asr-model", default="auto",
                        help="[demo] Whisper model ID or 'auto'")
    parser.add_argument("--asr-language", default="",
                        help="[demo] Force ASR language code (e.g. 'en'). Empty = auto-detect")
    parser.add_argument("--ocr", action="store_true",
                        help="[demo] Enable OCR text extraction per frame (step N)")
    parser.add_argument("--ocr-model", default="auto",
                        help="[demo] OCR model ID or 'auto'")
    parser.add_argument("--depth", action="store_true",
                        help="[demo] Enable depth estimation per frame (step O)")
    parser.add_argument("--depth-model", default="auto",
                        help="[demo] Depth model ID or 'auto'")
    parser.add_argument("--detection", action="store_true",
                        help="[demo] Enable object detection per frame (step P)")
    parser.add_argument("--detection-model", default="auto",
                        help="[demo] Detection model ID or 'auto'")
    parser.add_argument("--detection-labels", default="",
                        help="[demo] Comma-separated labels for open-vocabulary detection")
    parser.add_argument("--world-model", action="store_true",
                        help="[demo] Enable world model video embeddings (step Q)")
    parser.add_argument("--world-model-id", default="auto",
                        help="[demo] World model ID or 'auto'")
    parser.add_argument("--qwen", action="store_true",
                        help="[demo] Enable Qwen VLM detailed captioning (step R); requires --qwen-api-url")
    parser.add_argument("--qwen-api-url", default="",
                        help="[demo] Qwen vLLM/ollama endpoint (e.g. http://localhost:8010/v1)")
    parser.add_argument("--qwen-model", default="",
                        help="[demo] Qwen model ID; empty = use QWEN_MODEL env var default")
    parser.add_argument("--qwen-backend", default="", choices=["", "vllm", "ollama"],
                        help="[demo] Qwen backend type. Empty = auto-detect")

    return parser
