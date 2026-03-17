"""Entry point for the agentic video processing pipeline CLI."""

import argparse

from pipeline.cli_parser import build_parser
from pipeline.cli_runner import run_file_mode, run_stream_mode


def main() -> None:
    parser = build_parser()
    args: argparse.Namespace = parser.parse_args()
    if args.mode == "file":
        run_file_mode(args)
    else:
        run_stream_mode(args)


if __name__ == "__main__":
    main()
