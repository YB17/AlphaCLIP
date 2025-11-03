#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract binary masks per segment for a given COCO Panoptic image.

Requirements:
  pip install pillow numpy

Example:
  python export_panoptic_masks.py \
    --json annotations/panoptic_val2017.json \
    --panoptic-dir panoptic_val2017 \
    --image-file 000000397133.jpg \
    --out-dir out_masks/000000397133 \
    --things-only

Notes:
- COCO Panoptic 的 PNG 用 RGB 编码 segment_id: id = R + (G<<8) + (B<<16)
- 默认导出所有 segment；加 --things-only 仅导出实例类（things）
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image


def parse_args():
    ap = argparse.ArgumentParser(
        description="Export per-segment binary masks from a COCO Panoptic image."
    )
    ap.add_argument("--json", required=True, help="Path to panoptic_*.json")
    ap.add_argument("--panoptic-dir", required=True, help="Dir of panoptic PNGs")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--image-id", type=int, help="Image id (e.g., 397133)")
    grp.add_argument("--image-file", help="Image file name (e.g., 000000397133.jpg)")
    ap.add_argument("--out-dir", required=True, help="Output directory for masks")
    ap.add_argument(
        "--things-only",
        action="store_true",
        help="Export only thing instances (exclude stuff).",
    )
    return ap.parse_args()


def load_panoptic_json(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Build quick lookup maps
    images_by_id = {im["id"]: im for im in data["images"]}
    images_by_name = {im["file_name"]: im for im in data["images"]}
    ann_by_image_id = {ann["image_id"]: ann for ann in data["annotations"]}
    cats_by_id = {c["id"]: c for c in data["categories"]}
    return {
        "raw": data,
        "images_by_id": images_by_id,
        "images_by_name": images_by_name,
        "ann_by_image_id": ann_by_image_id,
        "cats_by_id": cats_by_id,
    }


def rgb_png_to_id_map(png: Image.Image) -> np.ndarray:
    """Decode COCO panoptic PNG to a 2D uint32 id map."""
    if png.mode != "RGB":
        png = png.convert("RGB")
    arr = np.array(png, dtype=np.uint32)  # (H,W,3)
    id_map = arr[..., 0] + (arr[..., 1] << 8) + (arr[..., 2] << 16)
    return id_map  # shape (H, W), dtype uint32


def find_image_entry(meta: Dict[str, Any], image_id: Optional[int], image_file: Optional[str]) -> Tuple[Dict, Dict]:
    if image_id is not None:
        im = meta["images_by_id"].get(image_id)
        if im is None:
            raise FileNotFoundError(f"image id {image_id} not found in JSON.")
    else:
        im = meta["images_by_name"].get(image_file)
        if im is None:
            raise FileNotFoundError(f"image file {image_file} not found in JSON.")
    ann = meta["ann_by_image_id"].get(im["id"])
    if ann is None:
        raise FileNotFoundError(f"No panoptic annotation for image id={im['id']}.")
    return im, ann


def export_masks_for_image(
    ann: Dict[str, Any],
    cats_by_id: Dict[int, Dict[str, Any]],
    panoptic_dir: Path,
    out_dir: Path,
    things_only: bool = False,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load panoptic PNG and decode id map
    pan_png_path = panoptic_dir / ann["file_name"]
    with Image.open(pan_png_path) as png:
        id_map = rgb_png_to_id_map(png)

    # Iterate segments
    num_saved = 0
    for seg in ann["segments_info"]:
        seg_id = int(seg["id"])
        cat_id = int(seg["category_id"])
        cat = cats_by_id[cat_id]
        isthing = bool(cat.get("isthing", 0))
        if things_only and not isthing:
            continue

        mask = (id_map == seg_id).astype(np.uint8) * 255  # 0/255
        if mask.max() == 0:
            # Should not happen, but be robust to mismatch
            continue

        # Use category name + seg_id for uniqueness
        cat_name = cat.get("name", f"cat_{cat_id}").replace(" ", "_").replace("/", "_")
        fname = f"{Path(ann['file_name']).stem}__{cat_name}__seg{seg_id}.png"
        out_path = out_dir / fname

        Image.fromarray(mask, mode="L").convert("RGB").save(out_path)
        num_saved += 1

    return num_saved


def main():
    args = parse_args()
    json_path = Path(args.json)
    panoptic_dir = Path(args.panoptic_dir)
    out_dir = Path(args.out_dir)

    meta = load_panoptic_json(json_path)
    im, ann = find_image_entry(meta, args.image_id, args.image_file)

    saved = export_masks_for_image(
        ann=ann,
        cats_by_id=meta["cats_by_id"],
        panoptic_dir=panoptic_dir,
        out_dir=out_dir,
        things_only=args.things_only,
    )

    im_name = im["file_name"]
    print(
        f"Done. Image: {im_name} (id={im['id']}) | "
        f"PNG: {ann['file_name']} | "
        f"Masks saved: {saved} | Output dir: {out_dir}"
    )


if __name__ == "__main__":
    main()
