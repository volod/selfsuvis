#!/usr/bin/env python3
"""Generate a synthetic CVAT-format annotated test archive.

Creates 1001 JPEG frames (synthetic drone-view imagery) plus a CVAT XML 1.1
annotation file, then writes them to data_test/cvat_frames/ and packages
everything into data_test/cvat_test_archive.zip.

Categories (VisDrone-inspired dataset labels):
  car (240 frames), truck (180), bus (120),
  pedestrian (200), bicycle (160), motor (101)  → 1001 total

Run once after git clone to create the test fixtures:
    python scripts/make_test_cvat_archive.py
    # → data_test/cvat_frames/         (extracted frames)
    # → data_test/cvat_annotations.xml (CVAT XML 1.1)
    # → data_test/cvat_test_archive.zip

The archive is consumed by tests/unit/test_supervised_finetune.py and by the
supervised fine-tuning pipeline when CVAT_TEST_ARCHIVE_DIR points to data_test/.
"""
import argparse
import os
import random
import struct
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

# VisDrone-2019 object categories (subset relevant to autonomous missions)
LABELS = ["car", "truck", "bus", "pedestrian", "bicycle", "motor"]

# Frames per label: must sum to 1001
FRAMES_PER_LABEL = {
    "car": 240,
    "truck": 180,
    "bus": 120,
    "pedestrian": 200,
    "bicycle": 160,
    "motor": 101,
}

# Frame dimensions (matches typical drone downscaled export)
FRAME_W, FRAME_H = 640, 480
SEED = 42


# ── Minimal JPEG writer (no external deps beyond numpy) ───────────────────────

def _write_minimal_jpeg(path: str, array: np.ndarray) -> None:
    """Write a numpy uint8 HxWx3 array as a minimal valid JPEG using PIL."""
    from PIL import Image
    Image.fromarray(array, mode="RGB").save(path, format="JPEG", quality=75)


def _make_frame(label: str, frame_idx: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic 640x480 RGB drone-view frame.

    Each class gets a distinct background hue.  A synthetic bounding-box region
    is drawn with brighter pixels to simulate an object.  The rest of the image
    is random low-variance noise (simulating ground/vegetation texture).
    """
    # Hue → RGB base colour per class (HSV-inspired, deterministic)
    hue_map = {
        "car":        (200, 80,  80),
        "truck":      (80,  150, 80),
        "bus":        (80,  80,  200),
        "pedestrian": (180, 150, 80),
        "bicycle":    (150, 80,  180),
        "motor":      (80,  180, 180),
    }
    base_rgb = hue_map[label]
    # Low-variance texture background
    noise = rng.integers(-30, 30, (FRAME_H, FRAME_W, 3), dtype=np.int16)
    base = np.array(base_rgb, dtype=np.int16)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)

    # Synthetic object region (brighter rectangle)
    bx1 = int(rng.integers(50, FRAME_W // 3))
    by1 = int(rng.integers(50, FRAME_H // 3))
    bx2 = bx1 + int(rng.integers(80, 200))
    by2 = by1 + int(rng.integers(60, 150))
    bx2 = min(bx2, FRAME_W - 1)
    by2 = min(by2, FRAME_H - 1)
    img[by1:by2, bx1:bx2] = np.clip(img[by1:by2, bx1:bx2].astype(np.int16) + 60, 0, 255).astype(np.uint8)

    return img, (bx1, by1, bx2, by2)


# ── CVAT XML 1.1 builder ──────────────────────────────────────────────────────

def _build_cvat_xml(frame_annotations: list) -> ET.ElementTree:
    """Build a CVAT XML 1.1 ElementTree from a list of annotation dicts.

    Each dict: {id, name, width, height, label, xtl, ytl, xbr, ybr}.
    """
    root = ET.Element("annotations")

    ver = ET.SubElement(root, "version")
    ver.text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")

    ET.SubElement(task, "id").text = "1"
    ET.SubElement(task, "name").text = "selfsuvis_test_drone_vehicles"
    ET.SubElement(task, "size").text = str(len(frame_annotations))
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "flipped").text = "False"
    ET.SubElement(task, "created").text = "2024-01-01 00:00:00.000000+00:00"
    ET.SubElement(task, "updated").text = "2024-01-01 00:00:00.000000+00:00"

    labels_el = ET.SubElement(task, "labels")
    for lname in LABELS:
        lbl = ET.SubElement(labels_el, "label")
        ET.SubElement(lbl, "name").text = lname
        ET.SubElement(lbl, "color").text = "#000000"
        ET.SubElement(lbl, "attributes")

    segs = ET.SubElement(task, "segments")
    seg = ET.SubElement(segs, "segment")
    ET.SubElement(seg, "id").text = "0"
    ET.SubElement(seg, "start").text = "0"
    ET.SubElement(seg, "stop").text = str(len(frame_annotations) - 1)
    ET.SubElement(seg, "url").text = ""

    ET.SubElement(meta, "dumped").text = "2024-01-01 00:00:00.000000+00:00"

    for ann in frame_annotations:
        img_el = ET.SubElement(root, "image")
        img_el.set("id", str(ann["id"]))
        img_el.set("name", ann["name"])
        img_el.set("width", str(ann["width"]))
        img_el.set("height", str(ann["height"]))

        box = ET.SubElement(img_el, "box")
        box.set("label", ann["label"])
        box.set("source", "manual")
        box.set("occluded", "0")
        box.set("xtl", f"{ann['xtl']:.2f}")
        box.set("ytl", f"{ann['ytl']:.2f}")
        box.set("xbr", f"{ann['xbr']:.2f}")
        box.set("ybr", f"{ann['ybr']:.2f}")
        box.set("z_order", "0")

    return ET.ElementTree(root)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(output_dir: str = "data_test") -> None:
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    frames_dir = os.path.join(output_dir, "cvat_frames")
    xml_path = os.path.join(output_dir, "cvat_annotations.xml")
    zip_path = os.path.join(output_dir, "cvat_test_archive.zip")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Build ordered list of (label, count) ensuring total = 1001
    assert sum(FRAMES_PER_LABEL.values()) == 1001, "FRAMES_PER_LABEL must sum to 1001"

    frame_annotations = []
    frame_idx = 0

    for label in LABELS:
        count = FRAMES_PER_LABEL[label]
        for i in range(count):
            name = f"frame_{frame_idx:04d}.jpg"
            img_array, (bx1, by1, bx2, by2) = _make_frame(label, frame_idx, rng)
            _write_minimal_jpeg(os.path.join(frames_dir, name), img_array)

            frame_annotations.append({
                "id": frame_idx,
                "name": name,
                "width": FRAME_W,
                "height": FRAME_H,
                "label": label,
                "xtl": float(bx1),
                "ytl": float(by1),
                "xbr": float(bx2),
                "ybr": float(by2),
            })
            frame_idx += 1

    print(f"Generated {frame_idx} frames in {frames_dir}")

    # Shuffle to avoid label blocks (realistic for a mixed annotation task)
    random.shuffle(frame_annotations)
    # Re-assign sequential IDs after shuffle
    for i, ann in enumerate(frame_annotations):
        ann["id"] = i

    # Write CVAT XML
    tree = _build_cvat_xml(frame_annotations)
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Wrote CVAT XML → {xml_path}")

    # Package into zip
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(xml_path, arcname="cvat_annotations.xml")
        for fname in sorted(os.listdir(frames_dir)):
            fpath = os.path.join(frames_dir, fname)
            zf.write(fpath, arcname=os.path.join("cvat_frames", fname))

    total_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"Archive → {zip_path}  ({total_mb:.1f} MB)")
    print(f"\nLabel distribution:")
    for lbl in LABELS:
        print(f"  {lbl:12s}: {FRAMES_PER_LABEL[lbl]}")
    print(f"  {'TOTAL':12s}: {sum(FRAMES_PER_LABEL.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir", default="data_test",
        help="Directory to write frames, XML, and archive (default: data_test)"
    )
    args = parser.parse_args()
    main(args.output_dir)
