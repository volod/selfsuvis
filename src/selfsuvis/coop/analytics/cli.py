#!/usr/bin/env python3
"""CLI for coop-pilot stack analytics."""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from .reporter import ReportRenderer


def _parse_datetime(s: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse datetime: {s}")


def _parse_since(value: str) -> datetime:
    if value.endswith("h"):
        return datetime.now() - timedelta(hours=int(value[:-1]))
    if value.endswith("m"):
        return datetime.now() - timedelta(minutes=int(value[:-1]))
    if value.endswith("d"):
        return datetime.now() - timedelta(days=int(value[:-1]))
    return _parse_datetime(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="coop-pilot stack analytics")
    parser.add_argument("--since", help="Start time ('1h', '30m', '7d' or ISO datetime)")
    parser.add_argument("--until", help="End time (ISO datetime)")
    parser.add_argument("--tail", type=int, default=2000, help="Log lines per service")
    parser.add_argument(
        "--format",
        choices=["console", "json", "html", "markdown"],
        default="console",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    since = _parse_since(args.since) if args.since else None
    until = _parse_datetime(args.until) if args.until else None
    renderer = ReportRenderer()

    if args.health_only:
        if args.format == "json":
            print(json.dumps(renderer.get_health(), indent=2))
        else:
            renderer.print_health_only()
        return

    if args.stats_only:
        if args.format == "json":
            print(json.dumps(renderer.get_stats(), indent=2))
        else:
            renderer.print_stats_only()
        return

    print("Collecting logs and generating report...", file=sys.stderr)
    report = renderer.generate_report(since=since, until=until, tail=args.tail)

    if args.format == "console":
        renderer.print_console_report(report)
    elif args.format == "json":
        if args.output:
            renderer.export_json(report, args.output)
            print(f"Report saved to: {args.output}", file=sys.stderr)
        else:
            print(json.dumps(report, indent=2, default=str))
    elif args.format == "html":
        output = args.output or Path("report.html")
        renderer.export_html(report, output)
        print(f"Report saved to: {output}", file=sys.stderr)
    elif args.format == "markdown":
        output = args.output or Path("report.md")
        renderer.export_markdown(report, output)
        print(f"Report saved to: {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
