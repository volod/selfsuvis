"""Shared Python helpers for repo shell scripts.

This module provides a small CLI so bash scripts can delegate JSON parsing and
numeric calculations to named Python functions instead of inline heredocs.
"""

import argparse
import json
import math
import sys
from pathlib import Path


def pretty_json(stdin_text: str) -> str:
    """Format JSON from stdin with indentation."""
    payload = json.loads(stdin_text)
    return json.dumps(payload, indent=2)


def json_field(stdin_text: str, field: str, default: str = "") -> str:
    """Extract a top-level JSON field from stdin."""
    payload = json.loads(stdin_text)
    value = payload.get(field, default)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def cuda_version_from_json(path: str) -> str:
    """Read CUDA version from `/usr/local/cuda/version.json`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("cuda", {}).get("version", "")
    return version.rsplit(".", 1)[0] if version else ""


def compute_flash_attn_jobs(
    total_kb: int,
    avail_kb: int,
    cpu_cores: int,
    ram_per_job_gb: float = 12.0,
    reserve_frac: float = 0.20,
) -> str:
    """Compute a conservative flash-attn parallel build budget."""
    total_gb = total_kb / 1024 / 1024
    avail_gb = avail_kb / 1024 / 1024
    floor_gb = total_gb * reserve_frac
    usable_gb = max(0.0, avail_gb - floor_gb)
    raw_mem = usable_gb / ram_per_job_gb if ram_per_job_gb > 0 else 0.0
    mem_jobs = math.ceil(raw_mem) if (raw_mem % 1) >= 0.8 else int(raw_mem)
    cpu_jobs = max(1, (cpu_cores - 2) // 2)
    jobs = max(1, min(mem_jobs, cpu_jobs))
    return f"{jobs} {total_gb:.1f} {avail_gb:.1f} {usable_gb:.1f} {mem_jobs} {cpu_jobs}"


def max_jobs(ram_per_job_gb: float = 12.0, reserve_frac: float = 0.20) -> int:
    """Return safe parallel C++/CUDA compilation job count for this machine.

    Canonical heavy-build budget for the main project (sslm, xformers, flash-attn).
    Reads /proc/meminfo and os.cpu_count() directly — no arguments needed.
    See AGENTS.md for policy; nanochat uses src/nanochat/scripts/detect_hw.py max_jobs.
    """
    try:
        meminfo = Path("/proc/meminfo").read_text()
        total_kb = int(next(l.split()[1] for l in meminfo.splitlines() if l.startswith("MemTotal:")))
        avail_kb = int(next(l.split()[1] for l in meminfo.splitlines() if l.startswith("MemAvailable:")))
    except Exception:
        total_kb, avail_kb = 8 * 1024 * 1024, 4 * 1024 * 1024
    cpu_cores = sys.maxsize  # type: ignore[assignment]
    try:
        cpu_cores = __import__("os").cpu_count() or 4
    except Exception:
        cpu_cores = 4
    result = compute_flash_attn_jobs(
        total_kb=total_kb,
        avail_kb=avail_kb,
        cpu_cores=cpu_cores,
        ram_per_job_gb=ram_per_job_gb,
        reserve_frac=reserve_frac,
    )
    return int(result.split()[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shell helper utilities for selfsuvis scripts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("pretty-json", help="Pretty-print JSON from stdin")

    json_field_parser = subparsers.add_parser("json-field", help="Read a JSON field from stdin")
    json_field_parser.add_argument("--field", required=True, help="Top-level JSON field name")
    json_field_parser.add_argument(
        "--default", default="", help="Fallback value when field is absent"
    )

    cuda_parser = subparsers.add_parser(
        "cuda-version-from-json",
        help="Read CUDA version from a version.json file",
    )
    cuda_parser.add_argument("--path", required=True, help="Path to CUDA version.json")

    subparsers.add_parser(
        "max-jobs",
        help="Print safe C++/CUDA parallel build job count for this machine",
    )

    flash_parser = subparsers.add_parser(
        "compute-flash-attn-jobs",
        help="Compute flash-attn parallel build job budget",
    )
    flash_parser.add_argument(
        "--total-kb", required=True, type=int, help="MemTotal from /proc/meminfo"
    )
    flash_parser.add_argument(
        "--avail-kb",
        required=True,
        type=int,
        help="MemAvailable from /proc/meminfo",
    )
    flash_parser.add_argument("--cpu-cores", required=True, type=int, help="Available CPU cores")
    flash_parser.add_argument(
        "--ram-per-job-gb",
        default=12.0,
        type=float,
        help="Estimated peak RAM per compilation job in GiB",
    )
    flash_parser.add_argument(
        "--reserve-frac",
        default=0.20,
        type=float,
        help="Fraction of total RAM to keep free",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "pretty-json":
        sys.stdout.write(pretty_json(sys.stdin.read()))
    elif args.command == "json-field":
        sys.stdout.write(json_field(sys.stdin.read(), field=args.field, default=args.default))
    elif args.command == "cuda-version-from-json":
        sys.stdout.write(cuda_version_from_json(args.path))
    elif args.command == "max-jobs":
        print(max_jobs())
    else:
        print(
            compute_flash_attn_jobs(
                total_kb=args.total_kb,
                avail_kb=args.avail_kb,
                cpu_cores=args.cpu_cores,
                ram_per_job_gb=args.ram_per_job_gb,
                reserve_frac=args.reserve_frac,
            )
        )


if __name__ == "__main__":
    main()
