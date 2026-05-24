"""Entry point for the selfsuvis CLI.

Modes:
  local   — run the full local analysis / training orchestration
  file    — process a video file or directory (default)
  stream  — process a live RTSP/device stream
  analyse — generate charts / report for an existing local run
"""

import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated",
    category=FutureWarning,
)


def _validate_local_inputs(args) -> None:
    videos_dir = Path(getattr(args, "videos_dir", ".data/videos"))
    if videos_dir.is_dir():
        return
    parser_error = (
        f"Videos directory does not exist: {videos_dir}\n"
        "Use the local data directory:  --videos-dir .data/videos\n"
        "Create it with:  mkdir -p .data/videos"
    )
    raise SystemExit(parser_error)


def main() -> None:
    from selfsuvis.pipeline.workflows import build_parser

    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "local":
        from selfsuvis.pipeline.workflows import apply_local_env  # noqa: PLC0415

        apply_local_env(args)
        _validate_local_inputs(args)
        from selfsuvis.pipeline.core import log_preflight, run_local_preflight  # noqa: PLC0415

        report = run_local_preflight(args)
        log_preflight(report)
        report.raise_for_errors()
        from selfsuvis.pipeline.workflows import run_local  # noqa: PLC0415

        run_local(args)
    elif args.mode == "file":
        from selfsuvis.pipeline.workflows import run_file_mode  # noqa: PLC0415

        run_file_mode(args)
    elif args.mode == "analyse":
        from selfsuvis.scripts.analyse_local_run import run  # noqa: PLC0415

        run(args)
    else:
        from selfsuvis.pipeline.workflows import run_stream_mode  # noqa: PLC0415

        run_stream_mode(args)


if __name__ == "__main__":
    main()
