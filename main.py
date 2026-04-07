"""Entry point for the selfsuvis CLI.

Modes:
  file    — process a video file or directory (default)
  stream  — process a live RTSP/device stream
  demo    — run the full end-to-end demonstration pipeline
"""

from pipeline.workflows import build_parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "demo":
        # Env vars must be set before pipeline.core.config is imported.
        from pipeline.workflows import apply_demo_env  # noqa: PLC0415
        apply_demo_env(args)
        from pipeline.workflows import run_demo  # noqa: PLC0415
        run_demo(args)
    elif args.mode == "file":
        from pipeline.workflows import run_file_mode  # noqa: PLC0415
        run_file_mode(args)
    else:
        from pipeline.workflows import run_stream_mode  # noqa: PLC0415
        run_stream_mode(args)


if __name__ == "__main__":
    main()
