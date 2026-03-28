#!/usr/bin/env python3
"""Evaluate Florence-2 caption quality against a ground-truth query set.

Runs two retrieval methods side-by-side:
  semantic  — text → OpenCLIP embedding → Qdrant cosine search on `clip` vector
  fts       — keyword search on `caption` column in Postgres (ILIKE)

Outputs per-query Precision@5, per-category aggregates with 95% CI,
overall P@5, false-positive rate on negative controls, and an optional
caption_confidence calibration report.

Usage:
    python scripts/eval_captions.py \\
        --ground-truth eval/ground_truth.jsonl \\
        --method both \\
        --top-k 5 \\
        --output eval/results_$(date +%Y%m%d).json

    python scripts/eval_captions.py \\
        --ground-truth eval/ground_truth.jsonl \\
        --method semantic \\
        --confidence-calibration

Ground-truth JSONL format (one query per line):
    {"query": "five trucks in a row", "category": "vehicle_count",
     "relevant_frame_ids": ["mission_1:42:12500", "mission_2:7:3000"]}

For negative controls use relevant_frame_ids: [].

See docs/design/eval-design-spec.md for the full protocol.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg
from dotenv import load_dotenv

_env_name = os.getenv("APP_ENV", "prod")
_env_file = Path(__file__).parent.parent / "env" / f"{_env_name}.env"
if _env_file.exists():
    load_dotenv(_env_file)
else:
    load_dotenv()

from pipeline.config import settings
from pipeline.logging_utils import get_logger

logger = get_logger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis",
)

NEGATIVE_CONTROL_CATEGORY = "negative_control"


# ── data structures ───────────────────────────────────────────────────────────


@dataclass
class QuerySpec:
    query: str
    category: str
    relevant_frame_ids: Set[str]


@dataclass
class QueryResult:
    query: str
    category: str
    relevant_frame_ids: Set[str]
    retrieved_ids: List[str] = field(default_factory=list)
    precision_at_k: float = 0.0


# ── I/O helpers ───────────────────────────────────────────────────────────────


def load_ground_truth(path: str) -> List[QuerySpec]:
    specs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            specs.append(QuerySpec(
                query=obj["query"],
                category=obj["category"],
                relevant_frame_ids=set(obj.get("relevant_frame_ids", [])),
            ))
    return specs


# ── semantic retrieval (OpenCLIP → Qdrant) ────────────────────────────────────


def _semantic_search(query: str, top_k: int) -> List[str]:
    """Return up to top_k frame_ids via CLIP text embedding → Qdrant search."""
    from models.openclip_model import OpenCLIPEmbedder
    from pipeline.qdrant_utils import QdrantStore
    from qdrant_client.http import models as qmodels

    # Lazy-init to share across calls
    if not hasattr(_semantic_search, "_model"):
        _semantic_search._model = OpenCLIPEmbedder()
        _semantic_search._store = QdrantStore(
            clip_dim=_semantic_search._model.image_dim()
        )

    model: OpenCLIPEmbedder = _semantic_search._model
    store: QdrantStore = _semantic_search._store

    query_vec = model.encode_texts([query])[0]

    # Restrict to frame-type points only
    frame_filter = qmodels.Filter(
        must=[qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="frame"))]
    )
    results = store.search(
        vector_name="clip",
        query_vector=query_vec,
        limit=top_k,
        payload_filter=frame_filter,
    )

    # Qdrant points use numeric IDs; we need the frame_id from the payload.
    # frame_id = mission_id:segment_id:t_ms stored in Qdrant payload as
    # mission_id + segment_id + t_sec — reconstruct from payload if possible,
    # otherwise fall back to the Qdrant point id string.
    frame_ids = []
    for pt in results:
        payload = pt.payload or {}
        mission_id = payload.get("mission_id", "")
        segment_id = payload.get("segment_id")
        t_sec = payload.get("t_sec")
        if mission_id and segment_id is not None and t_sec is not None:
            frame_ids.append(f"{mission_id}:{segment_id}:{int(t_sec * 1000)}")
        else:
            frame_ids.append(str(pt.id))

    return frame_ids


# ── FTS baseline (Postgres ILIKE on caption column) ───────────────────────────


async def _fts_search_async(query: str, top_k: int) -> List[str]:
    """Return up to top_k frame_ids via Postgres caption ILIKE search."""
    conn = await asyncpg.connect(DATABASE_URL, timeout=10)
    try:
        # Build ILIKE pattern from individual words for a simple keyword match.
        # Each word must appear somewhere in the caption (AND logic).
        words = [w.strip() for w in query.split() if len(w.strip()) >= 3]
        if not words:
            return []

        # Build WHERE clause: caption ILIKE '%word1%' AND caption ILIKE '%word2%' ...
        conditions = " AND ".join(
            f"caption ILIKE '%' || ${i+1} || '%'" for i, _ in enumerate(words)
        )
        sql = f"""
            SELECT id FROM frames
            WHERE caption IS NOT NULL
              AND {conditions}
            ORDER BY caption_confidence DESC NULLS LAST
            LIMIT ${len(words) + 1}
        """
        rows = await conn.fetch(sql, *words, top_k)
        return [r["id"] for r in rows]
    finally:
        await conn.close()


def _fts_search(query: str, top_k: int) -> List[str]:
    return asyncio.run(_fts_search_async(query, top_k))


# ── confidence calibration ────────────────────────────────────────────────────


async def _fetch_confidences_async(frame_ids: List[str]) -> Dict[str, Optional[float]]:
    """Fetch caption_confidence from Postgres for a set of frame_ids."""
    if not frame_ids:
        return {}
    conn = await asyncpg.connect(DATABASE_URL, timeout=10)
    try:
        rows = await conn.fetch(
            "SELECT id, caption_confidence FROM frames WHERE id = ANY($1::text[])",
            frame_ids,
        )
        return {r["id"]: r["caption_confidence"] for r in rows}
    finally:
        await conn.close()


def compute_calibration(
    query_results: List[QueryResult],
) -> Optional[float]:
    """Compute Pearson r between caption_confidence and per-frame mean relevance.

    Returns None if there is insufficient data (< 10 frame-query pairs).
    """
    # Build frame → binary relevance list across all queries
    frame_relevance: Dict[str, List[int]] = {}
    all_frame_ids: Set[str] = set()

    for qr in query_results:
        if qr.category == NEGATIVE_CONTROL_CATEGORY:
            continue
        for fid in qr.retrieved_ids:
            all_frame_ids.add(fid)
            frame_relevance.setdefault(fid, []).append(1 if fid in qr.relevant_frame_ids else 0)
        for fid in qr.relevant_frame_ids:
            all_frame_ids.add(fid)
            if fid not in qr.retrieved_ids:
                frame_relevance.setdefault(fid, []).append(0)

    if not all_frame_ids:
        return None

    confidences = asyncio.run(_fetch_confidences_async(list(all_frame_ids)))

    xs, ys = [], []
    for fid, rel_list in frame_relevance.items():
        conf = confidences.get(fid)
        if conf is None:
            continue
        mean_rel = sum(rel_list) / len(rel_list)
        xs.append(float(conf))
        ys.append(mean_rel)

    if len(xs) < 10:
        logger.warning("Insufficient data for calibration: %d frame-query pairs (need ≥10)", len(xs))
        return None

    # Pearson r
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    if den == 0:
        return 0.0
    return num / den


# ── metrics ───────────────────────────────────────────────────────────────────


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k


def agresti_coull_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Agresti-Coull 95% CI for a proportion.

    successes: number of queries where P@5 > 0
    n: total number of queries
    """
    n_tilde = n + z ** 2
    p_tilde = (successes + z ** 2 / 2) / n_tilde
    margin = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return max(0.0, p_tilde - margin), min(1.0, p_tilde + margin)


def compute_metrics(
    results: List[QueryResult], top_k: int
) -> Dict[str, dict]:
    """Return per-category and overall metrics."""
    categories: Dict[str, List[QueryResult]] = {}
    for qr in results:
        categories.setdefault(qr.category, []).append(qr)

    metrics: Dict[str, dict] = {}

    non_negative_all: List[QueryResult] = []

    for cat, qrs in categories.items():
        p_at_k_list = [qr.precision_at_k for qr in qrs]
        mean_p = sum(p_at_k_list) / len(p_at_k_list) if p_at_k_list else 0.0

        if cat == NEGATIVE_CONTROL_CATEGORY:
            # FPR: how often does any result appear for a null query?
            fpr = mean_p  # precision@k on negatives = false positive rate
            metrics[cat] = {
                "query_count": len(qrs),
                "false_positive_rate": round(fpr, 4),
                "queries": [
                    {"query": qr.query, "retrieved": qr.retrieved_ids[:top_k]}
                    for qr in qrs
                ],
            }
        else:
            non_negative_all.extend(qrs)
            successes = sum(1 for p in p_at_k_list if p > 0)
            ci_lo, ci_hi = agresti_coull_ci(successes, len(qrs))
            metrics[cat] = {
                "query_count": len(qrs),
                f"p_at_{top_k}": round(mean_p, 4),
                "ci_95_lo": round(ci_lo, 4),
                "ci_95_hi": round(ci_hi, 4),
                "queries": [
                    {
                        "query": qr.query,
                        f"p_at_{top_k}": round(qr.precision_at_k, 4),
                        "retrieved": qr.retrieved_ids[:top_k],
                        "relevant": list(qr.relevant_frame_ids),
                    }
                    for qr in qrs
                ],
            }

    # Overall (excluding negative controls)
    if non_negative_all:
        overall_p = sum(qr.precision_at_k for qr in non_negative_all) / len(non_negative_all)
        overall_successes = sum(1 for qr in non_negative_all if qr.precision_at_k > 0)
        ci_lo, ci_hi = agresti_coull_ci(overall_successes, len(non_negative_all))
        metrics["__overall__"] = {
            "query_count": len(non_negative_all),
            f"p_at_{top_k}": round(overall_p, 4),
            "ci_95_lo": round(ci_lo, 4),
            "ci_95_hi": round(ci_hi, 4),
        }

    return metrics


# ── reporting ─────────────────────────────────────────────────────────────────


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def print_report(
    metrics: Dict[str, dict],
    top_k: int,
    method: str,
    calibration_r: Optional[float] = None,
) -> None:
    print(f"\n{'═' * 70}")
    print(f"  Caption Eval — method: {method.upper()}    Precision@{top_k}")
    print(f"{'═' * 70}")

    PASS_OVERALL = 0.8
    TRIGGER_VEHICLE_COUNT = 0.6

    overall = metrics.get("__overall__", {})
    overall_p = overall.get(f"p_at_{top_k}", float("nan"))
    overall_ci = f"[{overall.get('ci_95_lo', 0):.2f}, {overall.get('ci_95_hi', 0):.2f}]"

    print(f"\n  {'Category':<25} {'P@' + str(top_k):<8} {'95% CI':<18} {'N':>4}  Bar")
    print(f"  {'-' * 65}")

    for cat, m in metrics.items():
        if cat == "__overall__":
            continue
        if cat == NEGATIVE_CONTROL_CATEGORY:
            fpr = m.get("false_positive_rate", 0)
            flag = " ← FPR should be 0.0" if fpr > 0 else " ✓"
            print(f"  {'[negatives] FPR':<25} {fpr:<8.4f} {'':18} {m['query_count']:>4}  {flag}")
        else:
            p = m.get(f"p_at_{top_k}", 0)
            ci = f"[{m.get('ci_95_lo', 0):.2f}, {m.get('ci_95_hi', 0):.2f}]"
            flag = ""
            if cat == "vehicle_count" and p < TRIGGER_VEHICLE_COUNT:
                flag = " ← TRIGGER Phase 2"
            print(f"  {cat:<25} {p:<8.4f} {ci:<18} {m['query_count']:>4}  {_bar(p)} {flag}")

    print(f"  {'-' * 65}")
    gate_symbol = "PASS ✓" if overall_p >= PASS_OVERALL else "FAIL ✗"
    print(f"  {'OVERALL':<25} {overall_p:<8.4f} {overall_ci:<18} {overall.get('query_count', 0):>4}  "
          f"{_bar(overall_p)}  [{gate_symbol}]")

    print(f"\n  Phase 1 gate:   P@{top_k}(overall) ≥ {PASS_OVERALL}  →  "
          f"{'PASSED — ready to ship' if overall_p >= PASS_OVERALL else 'FAILED — do not ship yet'}")
    print(f"  Phase 2 trigger: P@{top_k}(vehicle_count) < {TRIGGER_VEHICLE_COUNT}  →  "
          f"{'TRIGGERED' if metrics.get('vehicle_count', {}).get(f'p_at_{top_k}', 1) < TRIGGER_VEHICLE_COUNT else 'not triggered'}")

    if calibration_r is not None:
        quality = "GOOD (useful for active learning)" if calibration_r >= 0.5 else "WEAK (consider replacement signal)"
        print(f"\n  caption_confidence calibration: r = {calibration_r:.3f}  →  {quality}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Florence-2 caption quality against a ground-truth query set."
    )
    parser.add_argument(
        "--ground-truth", required=True,
        help="Path to ground_truth.jsonl (one query per line).",
    )
    parser.add_argument(
        "--method", choices=["semantic", "fts", "both"], default="both",
        help="Retrieval method(s) to evaluate (default: both).",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write JSON results (default: print only).",
    )
    parser.add_argument(
        "--confidence-calibration", action="store_true",
        help="Also compute Pearson r between caption_confidence and relevance.",
    )
    return parser.parse_args()


def _run_method(
    method: str,
    specs: List[QuerySpec],
    top_k: int,
) -> List[QueryResult]:
    results = []
    for spec in specs:
        if method == "semantic":
            retrieved = _semantic_search(spec.query, top_k)
        else:
            retrieved = _fts_search(spec.query, top_k)

        p = precision_at_k(retrieved, spec.relevant_frame_ids, top_k)
        results.append(QueryResult(
            query=spec.query,
            category=spec.category,
            relevant_frame_ids=spec.relevant_frame_ids,
            retrieved_ids=retrieved,
            precision_at_k=p,
        ))
    return results


def main() -> None:
    args = _parse_args()

    specs = load_ground_truth(args.ground_truth)
    logger.info("Loaded %d queries from %s", len(specs), args.ground_truth)

    methods = ["semantic", "fts"] if args.method == "both" else [args.method]
    all_results: Dict[str, dict] = {}

    for method in methods:
        logger.info("Running %s retrieval for %d queries …", method, len(specs))
        results = _run_method(method, specs, args.top_k)
        metrics = compute_metrics(results, args.top_k)

        calibration_r = None
        if args.confidence_calibration and method == "semantic":
            calibration_r = compute_calibration(results)
            metrics["__calibration_r__"] = calibration_r

        print_report(metrics, args.top_k, method, calibration_r)
        all_results[method] = metrics

    if args.method == "both" and len(all_results) == 2:
        _print_comparison(all_results["semantic"], all_results["fts"], args.top_k)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Results written to %s", args.output)


def _print_comparison(
    semantic: Dict[str, dict],
    fts: Dict[str, dict],
    top_k: int,
) -> None:
    print(f"\n{'═' * 70}")
    print(f"  Comparison: Semantic (Florence) vs FTS Baseline")
    print(f"{'═' * 70}")
    print(f"  {'Category':<25} {'Semantic P@' + str(top_k):<18} {'FTS P@' + str(top_k):<18} Delta")
    print(f"  {'-' * 65}")

    categories = set(semantic.keys()) | set(fts.keys())
    for cat in sorted(categories):
        if cat.startswith("__"):
            continue
        if cat == NEGATIVE_CONTROL_CATEGORY:
            s_val = semantic.get(cat, {}).get("false_positive_rate", float("nan"))
            f_val = fts.get(cat, {}).get("false_positive_rate", float("nan"))
            print(f"  {'[negatives] FPR':<25} {s_val:<18.4f} {f_val:<18.4f} —")
        else:
            s_val = semantic.get(cat, {}).get(f"p_at_{top_k}", float("nan"))
            f_val = fts.get(cat, {}).get(f"p_at_{top_k}", float("nan"))
            delta = s_val - f_val if not math.isnan(s_val) and not math.isnan(f_val) else float("nan")
            delta_str = f"{delta:+.4f}" if not math.isnan(delta) else "—"
            print(f"  {cat:<25} {s_val:<18.4f} {f_val:<18.4f} {delta_str}")

    s_overall = semantic.get("__overall__", {}).get(f"p_at_{top_k}", float("nan"))
    f_overall = fts.get("__overall__", {}).get(f"p_at_{top_k}", float("nan"))
    delta_overall = s_overall - f_overall if not math.isnan(s_overall) and not math.isnan(f_overall) else float("nan")
    delta_str = f"{delta_overall:+.4f}" if not math.isnan(delta_overall) else "—"
    print(f"  {'-' * 65}")
    print(f"  {'OVERALL':<25} {s_overall:<18.4f} {f_overall:<18.4f} {delta_str}")

    if not math.isnan(delta_overall) and delta_overall <= 0:
        print("\n  ⚠ FTS baseline matches or beats semantic search.")
        print("    Expand eval set to 250+ frames before drawing conclusions.")
    print()


if __name__ == "__main__":
    main()
