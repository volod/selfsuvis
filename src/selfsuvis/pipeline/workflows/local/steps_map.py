"""3D map step and GPU/VRAM memory helpers."""


from pathlib import Path
from typing import Any, Dict, List, Tuple

from selfsuvis.pipeline.mapping import advise_map_quality, build_sparse_map
from ._common import _log


def step_create_3d_map(
    video_path: Path,
    video_id: str,
    video_dir: Path,
    frame_list: List[Tuple[str, float]],
    models: Dict[str, Any],
    run_sfm_flag: bool,
    run_gsplat_flag: bool = True,
    device: str = "cuda",
    depth_results: List[Dict[str, Any]] | None = None,
    yolo_detection_results: List[Dict[str, Any]] | None = None,
    tracking_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
    """Step 15: build sparse 3D map + 3D Gaussian Splat."""
    return build_sparse_map(
        video_path=str(video_path),
        video_id=video_id,
        map_dir=video_dir / "3d_map",
        frame_list=frame_list,
        models=models,
        run_sfm_flag=run_sfm_flag,
        run_gsplat_flag=run_gsplat_flag,
        device=device,
        depth_results=depth_results,
        yolo_detection_results=yolo_detection_results,
        tracking_results=tracking_results,
    )


def step_advise_3d_map_quality(
    video_path: Path,
    video_dir: Path,
    frame_list: List[Tuple[str, float]],
    map_result: Dict[str, Any],
    caption_results: List[Dict[str, Any]] | None = None,
    tracking_results: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Generate measured map-quality diagnostics and capture guidance."""
    advisor = advise_map_quality(
        video_path=str(video_path),
        frame_list=frame_list,
        map_result=map_result,
        caption_results=caption_results or [],
        tracking_results=tracking_results or {},
        output_dir=video_dir / "3d_map",
    )
    _log.info(
        "  ✓ Map quality advisor → %s (%s, %.1f/100)",
        advisor.get("markdown_path", video_dir / "3d_map" / "map_quality_advisor.md"),
        ((advisor.get("summary", {}) or {}).get("overall") or "unknown"),
        float(advisor.get("readiness_score", 0.0) or 0.0),
    )
    return advisor
