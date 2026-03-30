"""Entry point for the selfsuvis CLI.

Modes:
  file    — process a video file or directory (default)
  stream  — process a live RTSP/device stream
  demo    — run the full end-to-end demonstration pipeline
"""

from pipeline.cli_parser import build_parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "demo":
        # Env vars must be set before pipeline.config is imported.
        from pipeline.demo_env import apply_demo_env  # noqa: PLC0415
        apply_demo_env(args)
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
