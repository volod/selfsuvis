"""Phase 2 parallel nodes — steps 4-8.

These five nodes are dispatched concurrently from p2_gemma_analysis.  Each
writes only its own dedicated state key, so there are no reducer conflicts.
VRAM serialisation is handled internally by _prep_vram_for_step /
_guard_min_free_vram inside each step function — no additional locking needed.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict

from ..graph_state import PipelineState

_log = logging.getLogger(__name__)


def node_p2_florence_caption(state: PipelineState) -> Dict[str, Any]:
    from ..steps_caption import step_scene_captioning
    from selfsuvis.pipeline.core.config import settings

    args = state["args"]
    caption_results = []
    t0 = time.monotonic()

    if not args.no_caption:
        knowledge = state.get("knowledge")
        caption_out = step_scene_captioning(
            state["frame_list"],
            state["video_name"],
            Path(state["video_dir"]),
            state["device"],
            models=state["models"],
            qwen_api_url=getattr(args, "qwen_api_url", ""),
            qwen_model=getattr(args, "qwen_model", "") or settings.QWEN_MODEL,
            florence_api_url=getattr(args, "florence_api_url", ""),
            florence_model=getattr(args, "florence_model", ""),
            domain_hint=knowledge.domain_hint() if knowledge else "",
        )
        caption_results = caption_out.get("captions", [])

    stats = dict(state.get("stats", {}))
    stats.setdefault("timings", {})["L_caption"] = time.monotonic() - t0
    return {"caption_results": caption_results, "stats": stats}


def node_p2_asr(state: PipelineState) -> Dict[str, Any]:
    from ..steps_caption import step_asr_transcription, _prep_vram_for_step

    args = state["args"]
    asr_result: Dict[str, Any] = {"skipped": True, "subtitle_map": {}, "segments": []}
    t0 = time.monotonic()

    if args.asr:
        _prep_vram_for_step(state["models"], state["device"])
        asr_result = step_asr_transcription(
            Path(state["video_path"]),
            state["frame_list"],
            state["video_name"],
            Path(state["video_dir"]),
        )

    stats = dict(state.get("stats", {}))
    stats.setdefault("timings", {})["M_asr"] = time.monotonic() - t0
    return {"asr_result": asr_result, "stats": stats}


def node_p2_ocr(state: PipelineState) -> Dict[str, Any]:
    from ..steps_caption import step_ocr_extraction, _prep_vram_for_step

    args = state["args"]
    ocr_result: Dict[str, Any] = {"skipped": True, "ocr_results": []}
    t0 = time.monotonic()

    if args.ocr:
        _prep_vram_for_step(state["models"], state["device"])
        ocr_result = step_ocr_extraction(
            state["frame_list"],
            state["video_name"],
            Path(state["video_dir"]),
            caption_results=state.get("caption_results", []),
        )

    stats = dict(state.get("stats", {}))
    stats.setdefault("timings", {})["N_ocr"] = time.monotonic() - t0
    return {"ocr_result": ocr_result, "stats": stats}


def node_p2_depth(state: PipelineState) -> Dict[str, Any]:
    from ..steps_caption import step_depth_estimation, _prep_vram_for_step

    args = state["args"]
    depth_result: Dict[str, Any] = {"skipped": True, "depth_results": []}
    t0 = time.monotonic()

    if args.depth:
        _prep_vram_for_step(state["models"], state["device"])
        depth_result = step_depth_estimation(
            state["frame_list"],
            state["video_name"],
            Path(state["video_dir"]),
        )

    stats = dict(state.get("stats", {}))
    stats.setdefault("timings", {})["O_depth"] = time.monotonic() - t0
    return {"depth_result": depth_result, "stats": stats}


def node_p2_detection(state: PipelineState) -> Dict[str, Any]:
    from ..steps_caption import step_object_detection, _prep_vram_for_step

    args = state["args"]
    det_result: Dict[str, Any] = {"skipped": True, "detection_results": []}
    t0 = time.monotonic()

    if args.detection:
        _prep_vram_for_step(state["models"], state["device"])
        det_result = step_object_detection(
            state["frame_list"],
            state["video_name"],
            Path(state["video_dir"]),
        )

    stats = dict(state.get("stats", {}))
    stats.setdefault("timings", {})["P_detection"] = time.monotonic() - t0
    return {"det_result": det_result, "stats": stats}
