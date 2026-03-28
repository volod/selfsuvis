#!/usr/bin/env python3
"""Validate Qwen2.5-VL-7B vLLM/ollama serving quality and latency.

Runs N sample frames (default 20) through the configured Qwen sidecar
(QWEN_API_URL) and measures:

  1. JSON validity rate — response parseable + all required keys present.
     Target: ≥ 0.85
  2. Vehicle count accuracy vs optional ground-truth labels.
     Target: ≥ 0.70 (only reported when --ground-truth is provided)
  3. Per-frame latency; reports p50 / p95.
     Target p95: ≤ 30s
  4. GPU VRAM budget — reads nvidia-smi memory.total and memory.used.
     Target with Florence + Qwen active: ≤ 14 GB total

Usage:

    # Basic: pick 20 frames at random from a mission frames directory
    QWEN_API_URL=http://localhost:8010/v1 \\
    python scripts/validate_qwen_serving.py \\
        --frames-dir data/frames/my_mission --count 20

    # With ground-truth labels for vehicle count accuracy
    python scripts/validate_qwen_serving.py \\
        --frames-dir data/frames/my_mission --count 20 \\
        --ground-truth eval/vehicle_gt.jsonl

    # Explicit frame list
    python scripts/validate_qwen_serving.py \\
        --frames data/frames/m1/frame_0001.jpg data/frames/m1/frame_0042.jpg

Ground-truth JSONL format (one frame per line):
    {"frame": "frame_0001.jpg", "vehicle_count": 3}

Exit codes:
    0  All targets met
    1  One or more targets missed (report printed)
    2  Service unreachable or no frames found
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

_env_name = os.getenv("APP_ENV", "prod")
_env_file = Path(__file__).parent.parent / "env" / f"{_env_name}.env"
if _env_file.exists():
    load_dotenv(_env_file)
else:
    load_dotenv()

from pipeline.config import settings
from pipeline.logging_utils import get_logger
from pipeline.qwen_model import QwenModel, _parse_qwen_response

logger = get_logger(__name__)

# ── thresholds ────────────────────────────────────────────────────────────────

TARGET_VALIDITY_RATE = 0.85
TARGET_VEHICLE_ACCURACY = 0.70
TARGET_P95_LATENCY_SEC = 30.0
TARGET_TOTAL_VRAM_GB = 14.0

REQUIRED_KEYS = {"vehicle_groups", "road_surface", "road_condition", "scene_summary"}


# ── GPU VRAM helpers ──────────────────────────────────────────────────────────


def _read_vram_gb() -> Tuple[Optional[float], Optional[float]]:
    """Return (total_gb, used_gb) from nvidia-smi, or (None, None) on failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None, None
        line = result.stdout.strip().splitlines()[0]
        total_mib, used_mib = (int(x.strip()) for x in line.split(","))
        return total_mib / 1024.0, used_mib / 1024.0
    except Exception:
        return None, None


# ── frame discovery ───────────────────────────────────────────────────────────


def _find_frames(frames_dir: Optional[str], count: int) -> List[Path]:
    """Return up to `count` image paths from a directory, shuffled."""
    if not frames_dir:
        return []
    dir_path = Path(frames_dir)
    if not dir_path.is_dir():
        return []
    candidates = sorted(
        p for p in dir_path.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not candidates:
        return []
    random.shuffle(candidates)
    return candidates[:count]


def _load_ground_truth(path: Optional[str]) -> Dict[str, int]:
    """Load frame-level vehicle count ground truth from a JSONL file.

    Format: {"frame": "frame_0001.jpg", "vehicle_count": 3}
    Returns a dict mapping basename → vehicle_count.
    """
    if not path:
        return {}
    gt: Dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = Path(obj["frame"]).name
            gt[key] = int(obj["vehicle_count"])
    return gt


# ── validity check ────────────────────────────────────────────────────────────


def _is_valid_response(result: Dict[str, Any]) -> bool:
    """Return True when result is a successful structured response."""
    if not isinstance(result, dict):
        return False
    # Any error / filtered / disabled key means not a success response
    error_keys = {"disabled", "service_unavailable", "clip_filtered", "timeout", "parse_error"}
    if error_keys & result.keys():
        return False
    return REQUIRED_KEYS.issubset(result.keys())


# ── main validation loop ──────────────────────────────────────────────────────


def run_validation(
    frame_paths: List[Path],
    ground_truth: Dict[str, int],
) -> Dict[str, Any]:
    """Run all frames through QwenModel and collect metrics."""
    from PIL import Image

    # Instantiate without CLIP pre-screening so every frame is sent to Qwen.
    model = QwenModel(clip_prescreen_fn=None)

    if not model.is_enabled():
        print("ERROR: QWEN_API_URL is not set. Cannot validate.", file=sys.stderr)
        sys.exit(2)

    if not model.is_healthy():
        print(
            f"ERROR: Qwen service unreachable at {settings.QWEN_API_URL}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Re-enable CLIP threshold by passing threshold=0 to avoid filtering
    # (we want to send all frames regardless of content for this validation).
    # We achieve this by temporarily overriding the threshold via a no-op prescreen.
    model._clip_prescreen_fn = lambda _img: True  # always pass

    latencies: List[float] = []
    valid_count = 0
    vehicle_correct = 0
    vehicle_total = 0
    frame_results: List[Dict[str, Any]] = []

    for i, fp in enumerate(frame_paths, 1):
        try:
            img = Image.open(fp).convert("RGB")
        except Exception as exc:
            logger.warning("Cannot open %s: %s", fp, exc)
            frame_results.append({"path": str(fp), "error": "cannot_open"})
            continue

        t0 = time.perf_counter()
        result = model.extract_frame_facts(img)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        valid = _is_valid_response(result)
        if valid:
            valid_count += 1

        # Vehicle count accuracy (only when ground truth available)
        gt_count = ground_truth.get(fp.name)
        if gt_count is not None:
            vehicle_total += 1
            if valid:
                predicted = sum(
                    g.get("count", 0)
                    for g in result.get("vehicle_groups", [])
                    if isinstance(g.get("count"), int)
                )
                if predicted == gt_count:
                    vehicle_correct += 1

        status = "valid" if valid else list(result.keys())[0] if result else "empty"
        print(
            f"  [{i:02d}/{len(frame_paths)}] {fp.name:<35} "
            f"{elapsed:.1f}s  {status}"
        )
        frame_results.append({
            "path": str(fp),
            "latency_sec": round(elapsed, 3),
            "valid": valid,
            "result_keys": list(result.keys()),
        })

    n = len(latencies)
    if n == 0:
        print("ERROR: No frames were processed.", file=sys.stderr)
        sys.exit(2)

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[int(n * 0.50)]
    p95 = latencies_sorted[min(int(n * 0.95), n - 1)]
    validity_rate = valid_count / n

    summary: Dict[str, Any] = {
        "frames_total": len(frame_paths),
        "frames_processed": n,
        "validity_rate": round(validity_rate, 4),
        "valid_count": valid_count,
        "latency_p50_sec": round(p50, 3),
        "latency_p95_sec": round(p95, 3),
    }

    if vehicle_total > 0:
        summary["vehicle_accuracy"] = round(vehicle_correct / vehicle_total, 4)
        summary["vehicle_frames_evaluated"] = vehicle_total

    return summary


# ── reporting ─────────────────────────────────────────────────────────────────


def _check(name: str, value: float, target: float, higher_is_better: bool = True) -> bool:
    ok = value >= target if higher_is_better else value <= target
    sym = "PASS ✓" if ok else "FAIL ✗"
    direction = "≥" if higher_is_better else "≤"
    print(f"  {sym}  {name:<35} {value:.4f}  (target {direction} {target})")
    return ok


def print_report(summary: Dict[str, Any], vram_total: Optional[float], vram_used: Optional[float]) -> bool:
    """Print validation report and return True if all targets met."""
    print(f"\n{'═' * 65}")
    print("  Qwen2.5-VL-7B Serving Validation Report")
    print(f"{'═' * 65}\n")

    all_pass = True

    all_pass &= _check(
        "JSON validity rate",
        summary["validity_rate"],
        TARGET_VALIDITY_RATE,
    )
    all_pass &= _check(
        "p95 latency per frame (sec)",
        summary["latency_p95_sec"],
        TARGET_P95_LATENCY_SEC,
        higher_is_better=False,
    )

    if "vehicle_accuracy" in summary:
        all_pass &= _check(
            "Vehicle count accuracy",
            summary["vehicle_accuracy"],
            TARGET_VEHICLE_ACCURACY,
        )

    print()
    print(f"  Frames processed : {summary['frames_processed']} / {summary['frames_total']}")
    print(f"  Valid responses  : {summary['valid_count']} / {summary['frames_processed']}")
    print(f"  Latency p50      : {summary['latency_p50_sec']:.1f}s")
    print(f"  Latency p95      : {summary['latency_p95_sec']:.1f}s")

    if vram_total is not None and vram_used is not None:
        vram_ok = vram_used <= TARGET_TOTAL_VRAM_GB
        sym = "PASS ✓" if vram_ok else "WARN ⚠"
        print(f"\n  GPU VRAM (nvidia-smi):")
        print(f"    Total : {vram_total:.1f} GB")
        print(f"    Used  : {vram_used:.1f} GB  ({sym} target ≤ {TARGET_TOTAL_VRAM_GB} GB with Florence active)")
        # VRAM is a warning, not a hard gate (Florence may not be loaded during this script)
        if not vram_ok:
            print("    NOTE: Start Florence in the worker before re-running to get combined budget.")
    else:
        print("\n  GPU VRAM: nvidia-smi not available — skipped")

    gate = "ALL TARGETS MET" if all_pass else "ONE OR MORE TARGETS MISSED"
    print(f"\n  {'✓ ' + gate if all_pass else '✗ ' + gate}")

    if not all_pass:
        print("\n  Recommended actions:")
        if summary["validity_rate"] < TARGET_VALIDITY_RATE:
            print("  • Validity < 0.85: try --temperature 0 (already default); check prompt version")
            print("    Consider switching quantisation: Q4_K_M → Q8 or vLLM FP16")
        if summary["latency_p95_sec"] > TARGET_P95_LATENCY_SEC:
            print("  • p95 > 30s: reduce VLLM_MAX_NUM_SEQS or VLLM_MAX_MODEL_LEN")
            print("    Check if Florence and Qwen are competing for VRAM (increase --cpu-offload-gb)")
        if summary.get("vehicle_accuracy", 1.0) < TARGET_VEHICLE_ACCURACY:
            print("  • Vehicle accuracy < 0.70: inspect failed frames; prompt may need tuning")
            print("    Check QWEN_MODEL matches the loaded model name in vLLM")

    print()
    return all_pass


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Qwen2.5-VL serving quality, latency, and GPU budget."
    )

    frame_src = parser.add_mutually_exclusive_group(required=True)
    frame_src.add_argument(
        "--frames-dir",
        help="Directory to sample frames from (JPG/PNG recursively).",
    )
    frame_src.add_argument(
        "--frames",
        nargs="+",
        help="Explicit list of frame image paths.",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of frames to sample from --frames-dir (default: 20).",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="JSONL file with per-frame vehicle_count ground truth.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for frame sampling (default: 42).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results (default: print only).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    # Collect frame paths
    if args.frames:
        frame_paths = [Path(p) for p in args.frames]
        missing = [p for p in frame_paths if not p.exists()]
        if missing:
            print(f"ERROR: {len(missing)} frame(s) not found: {missing[:3]}", file=sys.stderr)
            sys.exit(2)
    else:
        frame_paths = _find_frames(args.frames_dir, args.count)
        if not frame_paths:
            print(f"ERROR: No JPG/PNG frames found in {args.frames_dir}", file=sys.stderr)
            sys.exit(2)

    ground_truth = _load_ground_truth(args.ground_truth)

    print(f"\n=== Qwen2.5-VL Serving Validation ===")
    print(f"  Endpoint : {settings.QWEN_API_URL or '(not set)'}")
    print(f"  Model    : {settings.QWEN_MODEL}")
    print(f"  Backend  : {settings.QWEN_BACKEND}")
    print(f"  Frames   : {len(frame_paths)}")
    if ground_truth:
        print(f"  GT labels: {len(ground_truth)} frames")
    print()

    # Read VRAM before inference (Qwen should already be loaded by the sidecar)
    vram_total, vram_used = _read_vram_gb()

    print("── Per-frame results ────────────────────────────────────────────────")
    summary = run_validation(frame_paths, ground_truth)

    all_pass = print_report(summary, vram_total, vram_used)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "vram_total_gb": vram_total,
            "vram_used_gb": vram_used,
            "targets": {
                "validity_rate": TARGET_VALIDITY_RATE,
                "p95_latency_sec": TARGET_P95_LATENCY_SEC,
                "vehicle_accuracy": TARGET_VEHICLE_ACCURACY,
                "total_vram_gb": TARGET_TOTAL_VRAM_GB,
            },
            "all_targets_met": all_pass,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Results written to {out_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
