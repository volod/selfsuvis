"""3D map step and GPU/VRAM memory helpers."""


from pathlib import Path
from typing import Any, Dict, List, Tuple

from selfsuvis.pipeline.mapping import build_sparse_map
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
    )
