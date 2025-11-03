
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize hard cases from CSV, overlaying GT masks on original images with labels.

Features:
- Visualize all rows or a random subset (--mode all|sample, --sample-n N).
- Works with CSVs produced by the Alpha-CLIP panoptic evaluator (must include columns:
  ["image_id", "file_name", "segment_id", "gt_name", "pred", "hit"]).
- Overlays the GT instance mask (derived from panoptic PNG via segment_id) on the original image.
- Annotates GT label and predicted label (and highlights if wrong). Optionally draws bbox if present.
- Supports multiple CSV inputs (they will be concatenated).

Usage example:
python viz_hard_cases.py \
  --csv ./hard_cases_wrong_top200.csv \
  --images-dir /home/host/coco/val2017 \
  --panoptic-json /home/host/coco/annotations/panoptic_val2017.json \
  --panoptic-seg-dir /home/host/coco/panoptic_val2017 \
  --out-dir ./hardcase_viz \
  --mode sample --sample-n 50 --seed 42 --alpha 0.6 --draw-bbox

Notes:
- We implement rgb2id here to avoid requiring panopticapi at runtime.
- Panoptic file names are sourced from the panoptic JSON ("annotations": [{"image_id", "file_name", ...}]).
"""

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def rgb2id(color: np.ndarray) -> np.ndarray:
    """Convert RGB panoptic encoding to segment id map (panopticapi.utils.rgb2id equivalent)."""
    if color.ndim == 3:
        return color[:, :, 0].astype(np.int32) + \
               256 * color[:, :, 1].astype(np.int32) + \
               256 * 256 * color[:, :, 2].astype(np.int32)
    else:
        return color.astype(np.int32)


def load_panoptic_index(panoptic_json_path: Path) -> Dict[int, str]:
    """Build a mapping: image_id -> panoptic PNG file_name from the panoptic JSON."""
    with open(panoptic_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    id2png = {}
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        file_name = ann.get("file_name")
        if img_id is not None and file_name:
            id2png[int(img_id)] = file_name
    if not id2png:
        raise RuntimeError("No entries found in panoptic JSON 'annotations'. Please verify the file.")
    return id2png


def parse_bbox(bbox_str: str):
    """Parse bbox field from CSV if present; expected formats: '[x, y, w, h]' or '(x, y, w, h)'."""
    if bbox_str is None or (isinstance(bbox_str, float) and np.isnan(bbox_str)):
        return None
    try:
        val = ast.literal_eval(bbox_str)
        if isinstance(val, (list, tuple)) and len(val) == 4:
            x, y, w, h = val
            return float(x), float(y), float(w), float(h)
    except Exception:
        pass
    return None


def colorize_mask(mask: np.ndarray, color=(255, 0, 0), alpha=0.6) -> Image.Image:
    """
    Convert a binary mask (H, W) to an RGBA overlay image with the given color and alpha.
    """
    h, w = mask.shape
    # Create RGBA overlay array
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    a = int(alpha * 255)
    rgba[..., 3] = a
    rgba[~mask, 3] = 0  # transparent where mask is False
    return Image.fromarray(rgba, mode="RGBA")


def draw_text_box(img: Image.Image, lines: List[str], corner=(8, 8), fill=(0, 0, 0), text=(255, 255, 255)) -> None:
    """
    Draw a semi-transparent textbox with several lines of text.
    """
    draw = ImageDraw.Draw(img, "RGBA")
    # Basic font fallback
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # Compute text box size
    padding = 6
    widths = []
    heights = []
    for line in lines:
        # 使用 textbbox 替代 textsize
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        widths.append(w)
        heights.append(h)
    box_w = max(widths) + padding * 2
    box_h = sum(heights) + padding * (len(lines) + 1)

    x, y = corner
    draw.rectangle([x, y, x + box_w, y + box_h], fill=(fill[0], fill[1], fill[2], 160))
    ty = y + padding
    for line in lines:
        draw.text((x + padding, ty), line, font=font, fill=text)
        # 使用 getbbox 替代 getsize
        bbox = font.getbbox(line)
        ty += (bbox[3] - bbox[1]) + padding // 2

def visualize_rows(df: pd.DataFrame,
                   images_dir: Path,
                   panoptic_seg_dir: Path,
                   id2png: Dict[int, str],
                   out_dir: Path,
                   alpha: float = 0.6,
                   draw_bbox: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Visualizing"):
        try:
            image_id = int(row["image_id"])
            img_path = images_dir / str(row["file_name"])
            if not img_path.exists():
                # try basename only
                img_path = images_dir / Path(str(row["file_name"])).name
            if not img_path.exists():
                print(f"[WARN] Image not found: {row['file_name']}")
                continue

            # Load image
            img = Image.open(img_path).convert("RGB")
            W, H = img.size

            # Load panoptic PNG for this image
            pan_png_name = id2png.get(image_id, None)
            if pan_png_name is None:
                print(f"[WARN] No panoptic PNG mapping for image_id={image_id}")
                continue
            pan_path = panoptic_seg_dir / pan_png_name
            if not pan_path.exists():
                # also try basename only
                pan_path = panoptic_seg_dir / Path(pan_png_name).name
            if not pan_path.exists():
                print(f"[WARN] Panoptic PNG not found: {pan_png_name}")
                continue
            pan = np.array(Image.open(pan_path).convert("RGB"))
            seg_map = rgb2id(pan)

            seg_id = int(row["segment_id"])
            mask = (seg_map == seg_id)

            if not mask.any():
                print(f"[WARN] Empty mask for image_id={image_id}, segment_id={seg_id}")
                continue

            # Resize mask to image size if needed
            if mask.shape[0] != H or mask.shape[1] != W:
                # Panoptic PNG should match original image size; but just in case
                mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize((W, H), Image.NEAREST)
                mask = np.array(mask_img) > 127

            # Compose overlay
            overlay = colorize_mask(mask, color=(255, 0, 0), alpha=alpha)
            vis = img.convert("RGBA")
            vis = Image.alpha_composite(vis, overlay)

            # Draw bbox if present
            bbox = None
            if "bbox" in df.columns:
                bbox = parse_bbox(row["bbox"])
            if draw_bbox and bbox is not None:
                x, y, w, h = bbox
                draw = ImageDraw.Draw(vis, "RGBA")
                draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 0, 255), width=3)

            # Annotations
            gt = str(row.get("gt_name", "NA"))
            pred = str(row.get("pred", "NA"))
            hit_val = row.get("hit", False)
            try:
                hit = bool(int(hit_val)) if isinstance(hit_val, (int, np.integer, str)) else bool(hit_val)
            except Exception:
                hit = bool(hit_val)

            label_line = f"GT: {gt}"
            if hit:
                lines = [label_line, f"Pred: {pred} (correct)"]
            else:
                lines = [label_line, f"Pred: {pred} (WRONG)"]
            draw_text_box(vis, lines, corner=(8, 8))

            # Save
            safe_gt = gt.replace("/", "_")
            safe_pred = pred.replace("/", "_")
            base_name = f"{image_id}_{seg_id}_{safe_gt}_pred-{safe_pred}.png"
            out_path = out_dir / base_name
            vis.convert("RGB").save(out_path, quality=95)

        except Exception as e:
            print(f"[ERROR] {e}")


def main():
    parser = argparse.ArgumentParser(description="Visualize hard cases with GT mask overlay.")
    parser.add_argument("--csv", nargs="+", required=True, help="Path(s) to CSV file(s) with hard cases.")
    parser.add_argument("--images-dir", required=True, help="COCO images directory, e.g., /path/to/coco/val2017")
    parser.add_argument("--panoptic-json", required=True, help="COCO panoptic JSON, e.g., panoptic_val2017.json")
    parser.add_argument("--panoptic-seg-dir", required=True, help="COCO panoptic PNG dir, e.g., panoptic_val2017/")
    parser.add_argument("--out-dir", default="./hardcase_viz", help="Output directory")
    parser.add_argument("--mode", choices=["all", "sample"], default="all")
    parser.add_argument("--sample-n", type=int, default=50, help="Number of samples if mode=sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.6, help="Mask overlay alpha in [0,1]")
    parser.add_argument("--draw-bbox", action="store_true", help="Draw bbox if present in CSV")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    panoptic_json_path = Path(args.panoptic_json)
    panoptic_seg_dir = Path(args.panoptic_seg_dir)
    out_dir = Path(args.out_dir)

    # Load CSVs
    frames = []
    for p in args.csv:
        fp = Path(p)
        if not fp.exists():
            raise FileNotFoundError(f"CSV not found: {fp}")
        frames.append(pd.read_csv(fp))
    df = pd.concat(frames, ignore_index=True)

    # Basic checks
    required_cols = {"image_id", "file_name", "segment_id", "gt_name", "pred", "hit"}
    missing = required_cols - set(map(str, df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Select subset
    if args.mode == "sample":
        df = df.sample(n=min(args.sample_n, len(df)), random_state=args.seed)

    # Build mapping image_id -> panoptic PNG
    id2png = load_panoptic_index(panoptic_json_path)

    # Visualize
    visualize_rows(df, images_dir, panoptic_seg_dir, id2png, out_dir, alpha=args.alpha, draw_bbox=args.draw_bbox)


if __name__ == "__main__":
    main()
