import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from pipeline.config import settings
from pipeline.logging_utils import get_logger
from pipeline.utils import ensure_dir, now_iso
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.vision_models import OpenCLIPTagger, SAMSegmenter


@dataclass
class Segment:
    segment_id: str
    label: str
    bbox: Tuple[int, int, int, int]
    mean_color: Tuple[int, int, int]
    area: int


@dataclass
class FrameResult:
    description: str
    segments: List[Segment]
    entities: List[Dict[str, Any]]
    tracks: List[Dict[str, Any]]
    warnings: List[str]


def _dominant_color_name(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    if r >= g and r >= b:
        return "red"
    if g >= r and g >= b:
        return "green"
    return "blue"


def _scene_description(frame_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))
    brightness = "bright" if mean_val > 150 else "dim" if mean_val < 80 else "balanced"
    texture = "detailed" if edge_density > 0.08 else "smooth"
    return f"A {brightness}, {texture} frame."


def _segment_kmeans(frame_bgr: np.ndarray, k: int = 4) -> List[Segment]:
    h, w = frame_bgr.shape[:2]
    data = frame_bgr.reshape((-1, 3)).astype(np.float32)
    if data.shape[0] < k:
        k = max(1, data.shape[0])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    centers = centers.astype(np.uint8)

    segments: List[Segment] = []
    for i in range(k):
        mask = labels == i
        if not np.any(mask):
            continue
        ys, xs = np.where(mask.reshape((h, w)))
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        area = int(mask.sum())
        mean_color = tuple(int(c) for c in centers[i].tolist())
        seg = Segment(
            segment_id=f"region_{i}",
            label=f"region_{i}",
            bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
            mean_color=mean_color,
            area=area,
        )
        segments.append(seg)
    return segments


def image_to_text_agent(
    frame_bgr: np.ndarray,
    tagger: Optional[Any] = None,
    segmenter: Optional[Any] = None,
) -> Tuple[str, List[Segment]]:
    segments: List[Segment] = []
    if segmenter and tagger:
        try:
            from pipeline.vision_models import mask_to_segments

            masks = segmenter.segment(frame_bgr)
            sam_segments = mask_to_segments(frame_bgr, masks, tagger)
            segments = [
                Segment(
                    segment_id=s.segment_id,
                    label=s.label,
                    bbox=s.bbox,
                    mean_color=s.mean_color,
                    area=s.area,
                )
                for s in sam_segments
            ]
        except Exception:
            segments = _segment_kmeans(frame_bgr, k=4)
    else:
        segments = _segment_kmeans(frame_bgr, k=4)

    description = _scene_description(frame_bgr)
    if tagger is not None:
        info = tagger.describe_image(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        if info.get("labels"):
            labels = ", ".join([item["label"] for item in info["labels"]])
            description = f"{description} Likely: {labels}."

    if segments:
        dominant = max(segments, key=lambda s: s.area)
        color = _dominant_color_name(dominant.mean_color[::-1])
        description = f"{description} Dominant color: {color}."
    return description, segments


def recognition_correctness_agent(description: str, segments: List[Segment]) -> List[str]:
    warnings = []
    if not segments:
        warnings.append("no_segments_detected")
    if "Dominant color" not in description:
        warnings.append("missing_color_summary")
    return warnings


def ontology_agent(
    ontology: Dict[str, Any],
    segments: List[Segment],
    frame_index: int,
    t_sec: float,
) -> Dict[str, Any]:
    entities = ontology.setdefault("entities", {})
    for seg in segments:
        ent = entities.setdefault(seg.label, {"count": 0, "first_seen": frame_index, "last_seen": frame_index})
        ent["count"] += 1
        ent["last_seen"] = frame_index
        ent.setdefault("colors", {})
        color_name = _dominant_color_name(seg.mean_color[::-1])
        ent["colors"][color_name] = ent["colors"].get(color_name, 0) + 1
    ontology.setdefault("timeline", []).append({"frame_index": frame_index, "t_sec": t_sec})
    return ontology


def matching_agent(
    segments: List[Segment],
    prev_segments: Optional[List[Segment]],
    prev_tracks: Dict[str, int],
    next_track_id: int,
    iou_threshold: float = 0.2,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], int]:
    tracks: List[Dict[str, Any]] = []
    updated_tracks: Dict[str, int] = {}

    def iou(a: Segment, b: Segment) -> float:
        ax, ay, aw, ah = a.bbox
        bx, by, bw, bh = b.bbox
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        inter_x1, inter_y1 = max(ax, bx), max(ay, by)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union = aw * ah + bw * bh - inter
        return inter / max(union, 1)

    for seg in segments:
        track_id = None
        if prev_segments:
            best = 0.0
            best_seg = None
            for prev in prev_segments:
                score = iou(seg, prev)
                if score > best:
                    best = score
                    best_seg = prev
            if best_seg and best >= iou_threshold:
                track_id = prev_tracks.get(best_seg.segment_id)
        if track_id is None:
            track_id = next_track_id
            next_track_id += 1
        updated_tracks[seg.segment_id] = track_id
        tracks.append(
            {
                "track_id": track_id,
                "segment_id": seg.segment_id,
                "label": seg.label,
                "bbox": seg.bbox,
            }
        )
    return tracks, updated_tracks, next_track_id


def _process_frame_to_record(
    rec: Any,
    video_name: str,
    tagger: Optional[Any],
    segmenter: Optional[Any],
    ontology: Dict[str, Any],
    prev_segments: Optional[List[Segment]],
    prev_tracks: Dict[str, int],
    next_track_id: int,
    base_metadata: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Optional[List[Segment]], Dict[str, int], int]:
    """Process one frame through agents and return (record_dict, ontology, prev_segments, prev_tracks, next_track_id).
    Returns (None, ontology, ...) if frame could not be read."""
    frame = cv2.imread(rec.path)
    if frame is None:
        return (None, ontology, prev_segments, prev_tracks, next_track_id)

    description, segments = image_to_text_agent(frame, tagger=tagger, segmenter=segmenter)
    warnings = recognition_correctness_agent(description, segments)
    ontology = ontology_agent(ontology, segments, rec.index, rec.t_sec)

    tracks, prev_tracks, next_track_id = matching_agent(
        segments, prev_segments, prev_tracks, next_track_id
    )
    prev_segments = segments

    entities = [
        {
            "name": seg.label,
            "type": "region",
            "bbox": seg.bbox,
            "dominant_color": _dominant_color_name(seg.mean_color[::-1]),
        }
        for seg in segments
    ]

    ontology_entities = []
    for name, ent in ontology.get("entities", {}).items():
        colors = ent.get("colors", {})
        dominant = None
        if colors:
            dominant = max(colors.items(), key=lambda item: item[1])[0]
        ontology_entities.append(
            {
                "name": name,
                "count": ent.get("count", 0),
                "first_seen": ent.get("first_seen"),
                "last_seen": ent.get("last_seen"),
                "dominant_color": dominant,
            }
        )

    record = formatter_agent(
        video_name=video_name,
        frame_index=rec.index,
        t_sec=rec.t_sec,
        width=rec.width,
        height=rec.height,
        description=description,
        segments=segments,
        entities=entities,
        tracks=tracks,
        warnings=warnings,
        frame_path=rec.path,
        ontology_entities=ontology_entities,
        metadata=base_metadata,
    )
    return (record, ontology, prev_segments, prev_tracks, next_track_id)


def formatter_agent(
    video_name: str,
    frame_index: int,
    t_sec: float,
    width: int,
    height: int,
    description: str,
    segments: List[Segment],
    entities: List[Dict[str, Any]],
    tracks: List[Dict[str, Any]],
    warnings: List[str],
    frame_path: str,
    ontology_entities: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "video_name": video_name,
        "frame_index": frame_index,
        "t_sec": t_sec,
        "width": width,
        "height": height,
        "frame_path": frame_path,
        "description": description,
        "segments": [
            {
                "segment_id": seg.segment_id,
                "label": seg.label,
                "bbox": seg.bbox,
                "mean_color": seg.mean_color,
                "area": seg.area,
            }
            for seg in segments
        ],
        "entities": entities,
        "tracks": tracks,
        "warnings": warnings,
        "ontology_entities": ontology_entities,
        "metadata": metadata,
    }


def process_frames(
    video_name: str,
    frame_records: List[Any],
    output_dir: str,
    run_metadata: Optional[Dict[str, Any]] = None,
    model_type: str = "openclip_sam",
    sam_checkpoint: Optional[str] = None,
    sam_model_type: Optional[str] = None,
    labels_file: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    logger = get_logger(__name__)
    ensure_dir(output_dir)
    jsonl_path = os.path.join(output_dir, f"{video_name}.jsonl")
    ontology_path = os.path.join(output_dir, f"{video_name}.ontology.json")

    ontology: Dict[str, Any] = {"video_name": video_name, "entities": {}, "timeline": []}
    base_metadata: Dict[str, Any] = {
        "created_at": now_iso(),
        "output_dir": output_dir,
        "model_type": model_type,
    }
    if run_metadata:
        base_metadata.update(run_metadata)
    prev_segments: Optional[List[Segment]] = None
    prev_tracks: Dict[str, int] = {}
    next_track_id = 1

    tagger: Optional[Any] = None
    segmenter: Optional[Any] = None
    if model_type == "openclip_sam" and not (sam_checkpoint or settings.SAM_CHECKPOINT):
        logger.warning("SAM checkpoint missing; falling back to openclip_only")
        model_type = "openclip_only"
        base_metadata["model_type"] = model_type

    if model_type in {"openclip_sam", "openclip_only"}:
        from pipeline.vision_models import OpenCLIPTagger

        tagger = OpenCLIPTagger(labels_file=labels_file)
    if model_type == "openclip_sam":
        from pipeline.vision_models import SAMSegmenter

        segmenter = SAMSegmenter(model_type=sam_model_type, checkpoint=sam_checkpoint)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in frame_records:
            record, ontology, prev_segments, prev_tracks, next_track_id = _process_frame_to_record(
                rec,
                video_name,
                tagger,
                segmenter,
                ontology,
                prev_segments,
                prev_tracks,
                next_track_id,
                base_metadata,
            )
            if record is None:
                continue
            if verbose:
                logger.info(
                    "frame=%s t=%.3fs size=%sx%s desc=%s",
                    rec.path,
                    rec.t_sec,
                    rec.width,
                    rec.height,
                    record["description"],
                )
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    with open(ontology_path, "w", encoding="utf-8") as f:
        json.dump(ontology, f, ensure_ascii=True, indent=2)

    logger.info("Wrote jsonl=%s ontology=%s", jsonl_path, ontology_path)
    return {"jsonl_path": jsonl_path, "ontology_path": ontology_path}
