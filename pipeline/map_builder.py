"""Sparse 3D map builder for the demo pipeline.

Builds a point cloud from mission frames either via pycolmap SfM
(camera centres) or a PCA projection of frame embeddings as fallback.
Saves ``sparse_map.npz`` and ``map_stats.json`` into *map_dir*.

Returns
-------
dict with keys:
    npz_path    : str  — absolute path to saved .npz
    points      : np.ndarray shape (N, 3)
    colours     : np.ndarray shape (N, 3)  — RGB in [0, 1]
    sfm_poses   : int  — number of frames with recovered SfM pose
    scene_count : int  — number of SfM connected components
    method      : str  — "sfm" | "pca_dino" | "pca_clip" | "failed"
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from pipeline.logging_utils import get_logger
from pipeline.sfm import run_sfm

logger = get_logger(__name__)


def _sfm_point_cloud(
    video_path: str,
    video_id: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int]:
    """Run pycolmap SfM and return (points, colours, sfm_poses, scene_count).

    Returns (None, None, 0, 0) if SfM fails or recovers no poses.
    """
    sfm_result = run_sfm(video_path, video_id, video_id)
    success_frames = [f for f in sfm_result["frames"] if f["pose_status"] == "success"]
    sfm_poses = len(success_frames)
    scene_count = sfm_result.get("scene_count", 0)
    logger.info("SfM: %d/%d poses recovered, %d scene(s)",
                sfm_poses, len(sfm_result["frames"]), scene_count)

    if sfm_poses == 0:
        return None, None, sfm_poses, scene_count

    pts, cols = [], []
    max_t = max(sfm_result["frames"][-1]["t_sec"], 1.0)
    for f in success_frames:
        pose = f["pose_json"]
        R = np.array(pose["R"])   # 3×3
        t = np.array(pose["t"])   # (3,)
        pts.append(-R.T @ t)      # camera centre in world coords
        ratio = f["t_sec"] / max_t
        cols.append([ratio, 0.3, 1.0 - ratio])

    return (
        np.array(pts, dtype=np.float32),
        np.array(cols, dtype=np.float32),
        sfm_poses,
        scene_count,
    )


def _pca_point_cloud(
    frame_list: List[Tuple[str, float]],
    models: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Project frame embeddings to 3D via PCA. Returns (points, colours, method)."""
    dino_model = models.get("dino")
    model_for_pca = dino_model or models["clip"]
    step = max(1, len(frame_list) // 200)
    imgs = [Image.open(fp).convert("RGB") for fp, _ in frame_list[::step]]
    embeds = model_for_pca.encode_images(imgs)           # (N, D)

    emb_centered = embeds - embeds.mean(axis=0)
    _, _, Vt = np.linalg.svd(emb_centered, full_matrices=False)
    points = (emb_centered @ Vt[:3].T).astype(np.float32)   # (N, 3)

    n = len(points)
    ratios = np.linspace(0, 1, n)
    colours = np.stack([ratios, np.full(n, 0.4), 1.0 - ratios], axis=1).astype(np.float32)

    method = "pca_dino" if dino_model else "pca_clip"
    logger.info("PCA point cloud: %d points (method=%s)", n, method)
    return points, colours, method


def export_ply(
    points: np.ndarray,
    colours: np.ndarray,
    ply_path: Path,
) -> Path:
    """Write a coloured point cloud to a PLY file.

    Parameters
    ----------
    points   : (N, 3) float32 XYZ coordinates
    colours  : (N, 3) float32 RGB values in [0, 1]
    ply_path : destination path (parent dir must exist)

    Returns the path written.  Viewable in MeshLab, CloudCompare, Blender, etc.
    """
    from plyfile import PlyData, PlyElement

    rgb = (colours.clip(0, 1) * 255).astype(np.uint8)
    vertex = np.empty(len(points), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"]   = rgb[:, 0]
    vertex["green"] = rgb[:, 1]
    vertex["blue"]  = rgb[:, 2]

    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(ply_path))
    logger.info("PLY exported: %s (%d points)", ply_path, len(points))
    return Path(ply_path)


def build_sparse_map(
    video_path: str,
    video_id: str,
    map_dir: Path,
    frame_list: List[Tuple[str, float]],
    models: Dict[str, Any],
    run_sfm_flag: bool = True,
) -> Dict[str, Any]:
    """Build sparse 3D map and save to *map_dir*.

    Parameters
    ----------
    video_path   : absolute path to source video
    video_id     : unique video identifier
    map_dir      : output directory (created if absent)
    frame_list   : list of (frame_path, t_sec) tuples from frame extraction
    models       : dict with "clip" and optionally "dino" embedder instances
    run_sfm_flag : attempt pycolmap SfM when True; go straight to PCA when False
    """
    map_dir = Path(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)
    npz_path   = map_dir / "sparse_map.npz"
    stats_path = map_dir / "map_stats.json"

    points3d: Optional[np.ndarray] = None
    colours:  Optional[np.ndarray] = None
    sfm_poses   = 0
    scene_count = 0
    method      = "none"

    if run_sfm_flag:
        logger.info("Running Structure-from-Motion (pycolmap) …")
        try:
            points3d, colours, sfm_poses, scene_count = _sfm_point_cloud(
                video_path, video_id
            )
            if points3d is not None:
                method = "sfm"
                logger.info("SfM point cloud: %d camera centres", len(points3d))
        except Exception as exc:
            logger.warning("SfM failed (%s) — using PCA fallback", exc)
    else:
        logger.info("SfM skipped — using PCA point-cloud fallback")

    if points3d is None:
        logger.info("Building PCA 3D point cloud from frame embeddings …")
        try:
            points3d, colours, method = _pca_point_cloud(frame_list, models)
        except Exception as exc:
            logger.warning("PCA fallback failed (%s)", exc)
            points3d = np.zeros((1, 3), dtype=np.float32)
            colours  = np.ones((1, 3),  dtype=np.float32)
            method   = "failed"

    np.savez(str(npz_path), points=points3d, colours=colours)
    ply_path = export_ply(points3d, colours, map_dir / "sparse_map.ply")
    map_stats = {
        "method":      method,
        "point_count": int(len(points3d)),
        "sfm_poses":   sfm_poses,
        "scene_count": scene_count,
        "npz":         str(npz_path),
        "ply":         str(ply_path),
    }
    stats_path.write_text(json.dumps(map_stats, indent=2), encoding="utf-8")

    logger.info("3D map saved: %d points  method=%s", len(points3d), method)
    logger.info("  NPZ: %s", npz_path)
    logger.info("  PLY: %s  (open in MeshLab / CloudCompare / Blender)", ply_path)
    return {
        "npz_path":    str(npz_path),
        "ply_path":    str(ply_path),
        "points":      points3d,
        "colours":     colours,
        "sfm_poses":   sfm_poses,
        "scene_count": scene_count,
        "method":      method,
    }
