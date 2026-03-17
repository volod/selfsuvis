"""Argument parser for the agentic video processing pipeline CLI."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(description="Video processing pipeline")
    parser.add_argument("--mode", choices=["file", "stream"], default="file")
    parser.add_argument("--input", help="Video file path")
    parser.add_argument("--dir", help="Directory containing videos")
    parser.add_argument(
        "--output-dir", default="video_test", help="Output directory"
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
    return parser
