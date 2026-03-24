#!/usr/bin/env python3
"""Synthetic 3DGS PLY generator for testing and development.

Generates realistic-looking 3D Gaussian Splat PLY files without needing a
trained nerfstudio model. Useful for:
  - Testing the Phase 2 ICP fusion pipeline without real mission data.
  - CI/CD fixtures that require splat.ply files.
  - Benchmarking splat_io read/write performance.

Usage:
    # Single scene
    python scripts/generate_test_splat.py \\
        --output data/maps/mission_test/splat.ply \\
        --n-gaussians 500 --lat 48.0 --lon 11.0 --alt 100.0 --radius 15

    # Pre-generate all test assets (CI fixtures)
    python scripts/generate_test_splat.py --test-assets

Output:
    <output>.ply         — binary little-endian 3DGS PLY (59 float32 props/vertex)
    <output>_meta.json  — {"origin_lat", "origin_lon", "origin_alt", "n_gaussians", "radius_m"}
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.splat_io import write_splat_from_arrays, write_splat_metadata


def generate_splat(
    output_path: str,
    n_gaussians: int = 500,
    origin_lat: float = 48.0,
    origin_lon: float = 11.0,
    origin_alt: float = 100.0,
    radius_m: float = 15.0,
    seed: Optional[int] = None,
) -> None:
    """Generate a synthetic 3DGS PLY file centred on the given GPS position.

    Gaussians are distributed in a flattened ellipsoid (outdoor scene geometry):
      - Horizontal extent: ±radius_m
      - Vertical extent:   ±radius_m * 0.3  (flat outdoor terrain)

    Gaussian attributes use physically plausible ranges:
      - Opacity:    logit(0.3..0.9)  — most Gaussians moderately opaque
      - Scale:      log(0.01..0.5m) — small to medium sized splats
      - Rotation:   random unit quaternions
      - Colour:     mild SH-DC values (neutral grey-green, outdoor-ish)

    Args:
        output_path:  path to write splat.ply (dirs created automatically).
        n_gaussians:  number of Gaussian primitives.
        origin_lat/lon/alt: GPS ENU origin written to companion _meta.json.
        radius_m:     horizontal scatter radius in metres (ENU space).
        seed:         numpy random seed (None = random).
    """
    rng = np.random.default_rng(seed)

    # ── positions (ENU metres from origin) ────────────────────────────────────
    theta = rng.uniform(0, 2 * np.pi, n_gaussians)
    r     = rng.uniform(0, radius_m, n_gaussians) ** 0.5 * radius_m ** 0.5  # radial
    x_enu = r * np.cos(theta)
    y_enu = r * np.sin(theta)
    z_enu = rng.uniform(-radius_m * 0.3, radius_m * 0.3, n_gaussians)
    positions = np.column_stack([x_enu, y_enu, z_enu]).astype(np.float32)

    # ── opacity (logit-encoded) ────────────────────────────────────────────────
    raw_opacity = rng.uniform(0.3, 0.9, n_gaussians).astype(np.float32)
    opacities   = np.log(raw_opacity / (1 - raw_opacity)).astype(np.float32)

    # ── scale (log-encoded, metres) ───────────────────────────────────────────
    raw_scale = rng.uniform(0.01, 0.5, (n_gaussians, 3)).astype(np.float32)
    scales    = np.log(raw_scale).astype(np.float32)

    # ── rotation (random unit quaternions, WXYZ order) ────────────────────────
    q = rng.standard_normal((n_gaussians, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rotations = q  # (w, x, y, z)

    # ── SH DC coefficients (mild neutral outdoor colour) ──────────────────────
    base_color = np.array([0.3, 0.35, 0.2], dtype=np.float32)  # grey-green
    sh_dc = (base_color + rng.uniform(-0.1, 0.1, (n_gaussians, 3))).astype(np.float32)

    # ── write PLY + metadata ──────────────────────────────────────────────────
    write_splat_from_arrays(
        path=output_path,
        positions=positions,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        sh_dc=sh_dc,
    )
    write_splat_metadata(
        splat_path=output_path,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        origin_alt=origin_alt,
        extra={"n_gaussians": n_gaussians, "radius_m": radius_m},
    )

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  wrote {output_path}  ({n_gaussians} Gaussians, {size_kb:.1f} KB)")


def generate_test_assets() -> None:
    """Generate the canonical test asset set used by unit and integration tests.

    Three scenes:
      scene_a — 200 Gaussians at (48.0, 11.0),  r=10m  (reference scene)
      scene_b — 200 Gaussians at (48.00005, 11.00005), r=10m  (overlapping with a, ~7m offset)
      scene_c — 200 Gaussians at (48.05, 11.05), r=10m  (non-overlapping, ~6km away)

    scene_a and scene_b overlap by design to test ICP registration.
    scene_c is isolated to test the no-overlap rejection path.
    """
    assets_dir = Path(__file__).resolve().parent.parent / "tests" / "assets" / "splats"
    assets_dir.mkdir(parents=True, exist_ok=True)

    scenes = [
        {
            "name": "scene_a",
            "lat": 48.0, "lon": 11.0, "alt": 100.0,
            "n": 200, "radius": 10.0, "seed": 42,
            "desc": "reference scene",
        },
        {
            "name": "scene_b",
            "lat": 48.00005, "lon": 11.00005, "alt": 100.0,  # ~7m NE of scene_a
            "n": 200, "radius": 10.0, "seed": 43,
            "desc": "overlapping scene (~7m offset from scene_a)",
        },
        {
            "name": "scene_c",
            "lat": 48.05, "lon": 11.05, "alt": 100.0,   # ~6 km away
            "n": 200, "radius": 10.0, "seed": 44,
            "desc": "non-overlapping scene (~6 km from scene_a)",
        },
    ]

    print("Generating test splat assets:")
    for s in scenes:
        out = str(assets_dir / f"{s['name']}.ply")
        generate_splat(
            output_path=out,
            n_gaussians=s["n"],
            origin_lat=s["lat"],
            origin_lon=s["lon"],
            origin_alt=s["alt"],
            radius_m=s["radius"],
            seed=s["seed"],
        )
    print(f"\nAssets written to: {assets_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic 3DGS PLY test data")
    parser.add_argument("--test-assets", action="store_true",
                        help="Generate canonical test assets in tests/assets/splats/")
    parser.add_argument("--output", help="Output PLY path (single scene)")
    parser.add_argument("--n-gaussians", type=int, default=500)
    parser.add_argument("--lat", type=float, default=48.0)
    parser.add_argument("--lon", type=float, default=11.0)
    parser.add_argument("--alt", type=float, default=100.0)
    parser.add_argument("--radius", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.test_assets:
        generate_test_assets()
    elif args.output:
        generate_splat(
            output_path=args.output,
            n_gaussians=args.n_gaussians,
            origin_lat=args.lat,
            origin_lon=args.lon,
            origin_alt=args.alt,
            radius_m=args.radius,
            seed=args.seed,
        )
    else:
        parser.print_help()
