#!/usr/bin/env python3
"""Qdrant 2D GPS range-query performance benchmark.

Validates whether simultaneous lat + lon payload filters on a 50K-point
collection meet the p99 < 200ms SLA required by the robot advisory API and
change detection pipeline.

Usage:
    # Start Qdrant first: make up qdrant
    python scripts/benchmark_qdrant_gps.py [--points 50000] [--queries 500] [--radius-m 50]

What this tests:
    1. Inserts N synthetic points with random GPS coordinates (lat, lon, alt)
       into a temporary Qdrant collection with payload indexes on gps.lat and gps.lon.
    2. Runs Q bounding-box queries with the 2D filter (both lat AND lon range).
    3. Runs Q bounding-box queries with the 1D filter (lat-only + Python lon post-filter).
    4. Reports p50 / p95 / p99 latencies and recommends whether 2D or 1D is safer.

The collection is deleted after the benchmark unless --keep is passed.
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from pipeline.logging_utils import get_logger

logger = get_logger("benchmark_qdrant_gps")

_COLLECTION = "gps_benchmark_tmp"
_VECTOR_DIM = 512  # ViT-B-16 CLIP dimension

# Approximate conversion (flat-earth, valid for small radii)
_M_PER_DEG_LAT = 111_320.0


def _bbox(lat: float, lon: float, radius_m: float) -> Tuple[float, float, float, float]:
    dlat = radius_m / _M_PER_DEG_LAT
    dlon = radius_m / (_M_PER_DEG_LAT * abs(np.cos(np.radians(lat))) + 1e-9)
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def _insert_points(client: QdrantClient, n: int) -> List[dict]:
    """Insert n random GPS points and return their metadata for later queries."""
    batch_size = 1000
    points_meta = []

    print(f"  Inserting {n:,} points in batches of {batch_size}...")
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = []
        meta = []
        for i in range(batch_start, batch_end):
            lat = random.uniform(30.0, 70.0)
            lon = random.uniform(-10.0, 50.0)
            alt = random.uniform(0.0, 1000.0)
            vec = np.random.rand(_VECTOR_DIM).astype(np.float32)
            vec /= np.linalg.norm(vec)
            batch.append(
                qmodels.PointStruct(
                    id=i,
                    vector=vec.tolist(),
                    payload={
                        "gps": {"lat": lat, "lon": lon, "alt": alt},
                        "mission_id": f"mission_{i % 100}",
                    },
                )
            )
            meta.append({"id": i, "lat": lat, "lon": lon})

        client.upsert(collection_name=_COLLECTION, points=batch, wait=True)
        points_meta.extend(meta)
        print(f"    {batch_end:,}/{n:,}", end="\r")

    print(f"  Inserted {n:,} points.           ")
    return points_meta


def _run_2d_queries(
    client: QdrantClient,
    query_centers: list,
    radius_m: float,
    n_queries: int,
) -> List[float]:
    """Run 2D (lat AND lon) payload filter queries. Returns latency list (seconds)."""
    latencies = []
    for lat, lon in query_centers[:n_queries]:
        min_lat, max_lat, min_lon, max_lon = _bbox(lat, lon, radius_m)
        filt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="gps.lat",
                    range=qmodels.Range(gte=min_lat, lte=max_lat),
                ),
                qmodels.FieldCondition(
                    key="gps.lon",
                    range=qmodels.Range(gte=min_lon, lte=max_lon),
                ),
            ]
        )
        vec = np.random.rand(_VECTOR_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)

        t0 = time.perf_counter()
        client.search(
            collection_name=_COLLECTION,
            query_vector=vec.tolist(),
            query_filter=filt,
            limit=10,
        )
        latencies.append(time.perf_counter() - t0)
    return latencies


def _run_1d_queries(
    client: QdrantClient,
    query_centers: list,
    radius_m: float,
    n_queries: int,
) -> List[float]:
    """Run 1D (lat-only) Qdrant filter + Python lon post-filter. Returns latency list."""
    latencies = []
    for lat, lon in query_centers[:n_queries]:
        min_lat, max_lat, min_lon, max_lon = _bbox(lat, lon, radius_m)
        filt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="gps.lat",
                    range=qmodels.Range(gte=min_lat, lte=max_lat),
                ),
            ]
        )
        vec = np.random.rand(_VECTOR_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)

        t0 = time.perf_counter()
        results = client.search(
            collection_name=_COLLECTION,
            query_vector=vec.tolist(),
            query_filter=filt,
            limit=100,  # fetch more; post-filter in Python
        )
        # Python lon post-filter
        _ = [
            r for r in results
            if min_lon <= r.payload.get("gps", {}).get("lon", 999) <= max_lon
        ]
        latencies.append(time.perf_counter() - t0)
    return latencies


def _percentile(data: List[float], p: int) -> float:
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(data_sorted) - 1)
    return data_sorted[lo] + (data_sorted[hi] - data_sorted[lo]) * (k - lo)


def _print_stats(label: str, latencies: List[float]) -> None:
    p50 = _percentile(latencies, 50) * 1000
    p95 = _percentile(latencies, 95) * 1000
    p99 = _percentile(latencies, 99) * 1000
    mean = sum(latencies) / len(latencies) * 1000
    sla_ok = p99 < 200
    status = "✓ PASS" if sla_ok else "✗ FAIL"
    print(f"\n  {label}")
    print(f"    mean={mean:.1f}ms  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms")
    print(f"    SLA (p99 < 200ms): {status}")
    return sla_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=50_000, help="Number of synthetic GPS points")
    parser.add_argument("--queries", type=int, default=200, help="Number of benchmark queries per method")
    parser.add_argument("--radius-m", type=float, default=50.0, help="GPS bounding-box radius in metres")
    parser.add_argument("--host", default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    parser.add_argument("--keep", action="store_true", help="Keep the benchmark collection after run")
    args = parser.parse_args()

    client = QdrantClient(host=args.host, port=args.port, timeout=30)

    print(f"\n=== Qdrant GPS range-query benchmark ===")
    print(f"  host={args.host}:{args.port}  points={args.points:,}  "
          f"queries={args.queries}  radius={args.radius_m}m\n")

    # ── Setup ──────────────────────────────────────────────────────────────
    # Drop + recreate collection to start clean
    try:
        client.delete_collection(_COLLECTION)
    except Exception:
        pass

    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config=qmodels.VectorParams(size=_VECTOR_DIM, distance=qmodels.Distance.COSINE),
    )

    # Create payload indexes for lat and lon (required for range queries to be fast)
    print("  Creating payload indexes on gps.lat and gps.lon...")
    client.create_payload_index(
        collection_name=_COLLECTION,
        field_name="gps.lat",
        field_schema=qmodels.PayloadSchemaType.FLOAT,
    )
    client.create_payload_index(
        collection_name=_COLLECTION,
        field_name="gps.lon",
        field_schema=qmodels.PayloadSchemaType.FLOAT,
    )

    # ── Insert ─────────────────────────────────────────────────────────────
    t_insert_start = time.perf_counter()
    points_meta = _insert_points(client, args.points)
    t_insert = time.perf_counter() - t_insert_start
    print(f"  Insert time: {t_insert:.1f}s")

    # Pick random query centers from actual inserted points
    query_centers = [(p["lat"], p["lon"]) for p in random.sample(points_meta, args.queries)]

    # ── Warm-up ────────────────────────────────────────────────────────────
    print("\n  Warming up (10 queries)...")
    _run_2d_queries(client, query_centers[:10], args.radius_m, 10)

    # ── Benchmark ──────────────────────────────────────────────────────────
    print(f"\n  Running {args.queries} queries per method...")

    lat_2d = _run_2d_queries(client, query_centers, args.radius_m, args.queries)
    lat_1d = _run_1d_queries(client, query_centers, args.radius_m, args.queries)

    # ── Results ────────────────────────────────────────────────────────────
    print("\n─── Results ──────────────────────────────────────────────────────────")
    pass_2d = _print_stats("2D filter (lat AND lon in Qdrant)", lat_2d)
    pass_1d = _print_stats("1D filter (lat in Qdrant + Python lon post-filter)", lat_1d)

    print("\n─── Recommendation ───────────────────────────────────────────────────")
    if pass_2d:
        print("  ✓ 2D filter meets SLA — safe to use GPS_FILTER_2D=true in production.")
        print("    Ensure payload indexes on gps.lat and gps.lon are created at collection init.")
    else:
        print("  ✗ 2D filter does NOT meet p99 < 200ms SLA.")
        if pass_1d:
            print("  ✓ 1D filter (default GPS_FILTER_2D=false) meets SLA — keep the default.")
        else:
            print("  ✗ Even 1D filter misses SLA — investigate Qdrant configuration / hardware.")
    print()

    # ── Cleanup ────────────────────────────────────────────────────────────
    if not args.keep:
        client.delete_collection(_COLLECTION)
        print("  (Benchmark collection deleted. Pass --keep to retain it.)")


if __name__ == "__main__":
    main()
