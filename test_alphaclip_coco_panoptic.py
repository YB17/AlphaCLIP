"""Alpha-CLIP COCO Panoptic evaluation script.

This script evaluates Alpha-CLIP on the COCO Panoptic dataset by extracting
region embeddings for every segment (things and stuff) and matching them
against a small candidate text set consisting of the ground-truth label and
generic rejection classes.

The implementation follows the requirements provided in the task description
and mirrors the Alpha-CLIP README usage patterns for masked region features.

Example usages
--------------

# 冒烟测试（仅跑 5 张图）：
python test_alphaclip_coco_panoptic.py \
  --images-dir /path/to/coco/val2017 \
  --panoptic-json /path/to/annotations/panoptic_val2017.json \
  --panoptic-seg-dir /path/to/annotations/panoptic_val2017 \
  --alpha-ckpt ./checkpoints/clip_l14_grit20m_fultune_2xe.pth \
  --model-name "ViT-L/14" \
  --image-size 336 \
  --limit-images 5 \
  --out-dir ./alphaclip_eval_smoke

# 全量评测（无可视化）：
python test_alphaclip_coco_panoptic.py \
  --images-dir /path/to/coco/val2017 \
  --panoptic-json /path/to/annotations/panoptic_val2017.json \
  --panoptic-seg-dir /path/to/annotations/panoptic_val2017 \
  --alpha-ckpt ./checkpoints/clip_l14_grit20m_fultune_2xe.pth \
  --model-name "ViT-L/14" \
  --image-size 336 \
  --batch-size 64 \
  --num-workers 8 \
  --out-dir ./alphaclip_panoptic_eval

# 启用标签自然化与短语聚合（举例）：
python test_alphaclip_coco_panoptic.py \
  --images-dir /path/to/coco/val2017 \
  --panoptic-json /path/to/annotations/panoptic_val2017.json \
  --panoptic-seg-dir /path/to/annotations/panoptic_val2017 \
  --alpha-ckpt ./checkpoints/clip_b16_grit1m_fultune_8xe.pth \
  --model-name "ViT-B/16" \
  --image-size 224 \
  --label-map-json ./natural_label_map.json \
  --agg max \
  --visualize-n 20 \
  --out-dir ./alphaclip_panoptic_eval_nat
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from panopticapi.utils import rgb2id
from tqdm import tqdm

import torch
import torchvision.transforms as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Alpha-CLIP on COCO Panoptic segments."
    )
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--panoptic-json", required=True, type=Path)
    parser.add_argument("--panoptic-seg-dir", required=True, type=Path)
    parser.add_argument("--alpha-ckpt", required=True, type=Path)

    parser.add_argument("--model-name", default="ViT-L/14")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument("--reject-threshold", type=float)
    parser.add_argument("--prompt-template", default="{label}")
    parser.add_argument("--label-map-json", type=Path)
    parser.add_argument("--agg", choices=["max", "mean"], default="max")
    parser.add_argument("--visualize-n", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("./alphaclip_panoptic_eval"))
    parser.add_argument("--save-csv", dest="save_csv", action="store_true", default=True)
    parser.add_argument("--no-save-csv", dest="save_csv", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-images", type=int)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class PanopticSample:
    image_id: int
    file_name: str
    image_path: Path
    segment_id: int
    mask: np.ndarray  # bool mask
    category_id: int
    category_name: str
    isthing: bool
    area: int
    bbox: Sequence[float]


def load_alpha_clip(
    model_name: str,
    alpha_ckpt: Path,
    device: torch.device,
    image_size: int,
) -> Tuple[torch.nn.Module, T.Compose, T.Compose]:
    try:
        from alpha_clip import load as alpha_clip_load
    except ImportError as exc:  # pragma: no cover - informative message
        raise RuntimeError(
            "Failed to import alpha_clip. Please install Alpha-CLIP according to the "
            "repository README before running this script."
        ) from exc

    model, preprocess = alpha_clip_load(
        model_name,
        alpha_vision_ckpt_pth=str(alpha_ckpt),
        device=str(device),
    )
    model.eval()

    mask_transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.Normalize((0.5,), (0.26,)),
        ]
    )

    return model, preprocess, mask_transform


def load_panoptic(panoptic_json: Path, panoptic_seg_dir: Path) -> Tuple[dict, dict, dict]:
    with panoptic_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {c["id"]: c for c in data["categories"]}
    images = {img["id"]: img for img in data["images"]}
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)

    # Ensure segmentation directory is correct.
    seg_dir = panoptic_seg_dir
    if not seg_dir.exists():
        raise FileNotFoundError(f"Panoptic segmentation directory not found: {seg_dir}")

    return images, annotations, categories


def iter_image_instances(
    image_record: dict,
    annotations: Sequence[dict],
    images_dir: Path,
    panoptic_seg_dir: Path,
    categories: Dict[int, dict],
    min_area: int,
) -> Iterator[PanopticSample]:
    image_path = images_dir / image_record["file_name"]
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    for ann in annotations:
        seg_path = panoptic_seg_dir / ann["file_name"]
        if not seg_path.exists():
            raise FileNotFoundError(f"Panoptic segmentation PNG not found: {seg_path}")

        with Image.open(seg_path) as seg_image:
            panoptic_img = np.array(seg_image, dtype=np.uint8)
        seg_map = rgb2id(panoptic_img)

        for seg_info in ann.get("segments_info", []):
            area = seg_info.get("area", 0)
            if area < min_area:
                continue

            category_id = seg_info["category_id"]
            category_meta = categories.get(category_id)
            if category_meta is None:
                continue

            mask = seg_map == seg_info["id"]
            if not np.any(mask):
                continue

            yield PanopticSample(
                image_id=image_record["id"],
                file_name=image_record["file_name"],
                image_path=image_path,
                segment_id=seg_info["id"],
                mask=mask,
                category_id=category_id,
                category_name=category_meta.get("name", str(category_id)),
                isthing=bool(category_meta.get("isthing", 0)),
                area=int(area),
                bbox=seg_info.get("bbox", [0, 0, 0, 0]),
            )


def encode_region(
    model: torch.nn.Module,
    preprocess: T.Compose,
    mask_transform: T.Compose,
    image_pil: Image.Image,
    mask_bool: np.ndarray,
    device: torch.device,
    use_half: bool,
    log_dtype: bool,
) -> torch.Tensor:
    image_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    mask_pil = Image.fromarray(mask_uint8, mode="L")
    alpha_tensor = mask_transform(mask_pil).unsqueeze(0).to(device)

    if use_half:
        image_tensor = image_tensor.half()
        alpha_tensor = alpha_tensor.half()

    if log_dtype:
        print(
            f"[Debug] image tensor dtype={image_tensor.dtype} device={image_tensor.device}; "
            f"alpha dtype={alpha_tensor.dtype} device={alpha_tensor.device}"
        )

    with torch.no_grad():
        region_feat = model.visual(image_tensor, alpha_tensor)

    region_feat = region_feat / region_feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return region_feat.squeeze(0)


def apply_template(template: str, label: str) -> str:
    if "{label}" in template:
        return template.replace("{label}", label)
    # Fallback to append label for robustness
    return template.strip() + " " + label


def build_text_candidates(
    gt_label: str,
    prompt_template: str,
    label_map: Optional[Dict[str, List[str]]],
) -> Tuple[List[str], List[List[str]]]:
    gt_phrases: List[str] = [apply_template(prompt_template, gt_label)]
    if label_map and gt_label in label_map:
        for phrase in label_map[gt_label]:
            formatted = apply_template(prompt_template, phrase)
            if formatted not in gt_phrases:
                gt_phrases.append(formatted)

    candidate_labels = [gt_label, "other", "unknown", "none"]
    phrase_groups = [gt_phrases, ["other"], ["unknown"], ["none"]]
    return candidate_labels, phrase_groups


def load_label_map(label_map_path: Optional[Path]) -> Optional[Dict[str, List[str]]]:
    if not label_map_path:
        return None
    with label_map_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: list(v) for k, v in data.items()}


class TextEncoder:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        batch_size: int,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.cache: Dict[str, torch.Tensor] = {}
        self.tokenize = self._resolve_tokenizer()

    def _resolve_tokenizer(self):
        try:
            from alpha_clip import tokenize as alpha_tokenize

            return alpha_tokenize
        except ImportError:
            pass

        try:
            import clip  # type: ignore

            return clip.tokenize  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import tokenizer from alpha_clip or clip. "
                "Install one of them to proceed."
            ) from exc

    def encode(self, phrases: Sequence[str]) -> Dict[str, torch.Tensor]:
        missing = [p for p in phrases if p not in self.cache]
        if missing:
            with torch.no_grad():
                for start in range(0, len(missing), self.batch_size):
                    chunk = missing[start : start + self.batch_size]
                    tokens = self.tokenize(chunk).to(self.device)
                    text_features = self.model.encode_text(tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    for phrase, feature in zip(chunk, text_features):
                        self.cache[phrase] = feature.detach()
        return {p: self.cache[p] for p in phrases}


def aggregate_similarity(
    region_feature: torch.Tensor,
    phrase_groups: List[List[str]],
    phrase_embeddings: Dict[str, torch.Tensor],
    agg: str,
) -> List[float]:
    sims: List[float] = []
    for group in phrase_groups:
        group_scores = [float(torch.matmul(region_feature, phrase_embeddings[phrase])) for phrase in group]
        if agg == "max":
            sims.append(float(max(group_scores)))
        else:
            sims.append(float(sum(group_scores) / max(len(group_scores), 1)))
    return sims


class EvaluationAccumulator:
    def __init__(self) -> None:
        self.total = 0
        self.hits = 0
        self.thing_total = 0
        self.thing_hits = 0
        self.stuff_total = 0
        self.stuff_hits = 0
        self.skipped = 0
        self.category_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"total": 0, "hits": 0})
        self.confusion: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"gt": 0, "other": 0, "unknown": 0, "none": 0}
        )

    def update(
        self,
        category_name: str,
        isthing: bool,
        hit: bool,
        pred_label: str,
    ) -> None:
        self.total += 1
        if hit:
            self.hits += 1
        if isthing:
            self.thing_total += 1
            if hit:
                self.thing_hits += 1
        else:
            self.stuff_total += 1
            if hit:
                self.stuff_hits += 1

        stats = self.category_stats[category_name]
        stats["total"] += 1
        if hit:
            stats["hits"] += 1

        col = "gt" if hit else pred_label
        if col not in self.confusion[category_name]:
            self.confusion[category_name][col] = 0
        self.confusion[category_name][col] += 1

    def register_skip(self) -> None:
        self.skipped += 1

    def summary(self) -> Dict[str, object]:
        overall_acc = self.hits / self.total if self.total else 0.0
        thing_acc = self.thing_hits / self.thing_total if self.thing_total else None
        stuff_acc = self.stuff_hits / self.stuff_total if self.stuff_total else None

        per_category = {
            cat: {
                "total": stats["total"],
                "hits": stats["hits"],
                "accuracy": stats["hits"] / stats["total"] if stats["total"] else None,
            }
            for cat, stats in self.category_stats.items()
        }

        return {
            "total_instances": self.total,
            "matched": self.hits,
            "accuracy": overall_acc,
            "thing_total": self.thing_total,
            "thing_hits": self.thing_hits,
            "thing_accuracy": thing_acc,
            "stuff_total": self.stuff_total,
            "stuff_hits": self.stuff_hits,
            "stuff_accuracy": stuff_acc,
            "skipped_instances": self.skipped,
            "per_category": per_category,
        }


def visualize_instance(
    image: Image.Image,
    mask: np.ndarray,
    output_path: Path,
    gt_label: str,
    pred_label: str,
    scores: Sequence[float],
    candidate_labels: Sequence[str],
) -> None:
    overlay_color = np.array([255, 0, 0], dtype=np.uint8)
    alpha = 0.5
    image_np = np.array(image.convert("RGB"), dtype=np.float32)
    overlay = image_np.copy()
    overlay[mask] = overlay[mask] * (1 - alpha) + overlay_color * alpha
    blended = Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))

    blended = blended.convert("RGBA")
    draw = ImageDraw.Draw(blended)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=16)
    except IOError:
        font = ImageFont.load_default()

    text_lines = [f"GT: {gt_label}", f"Pred: {pred_label}"]
    for label, score in zip(candidate_labels, scores):
        text_lines.append(f"{label}: {score:.3f}")
    text = "\n".join(text_lines)
    bbox = draw.multiline_textbbox((10, 10), text, font=font)
    padding = 6
    background = [
        max(0, bbox[0] - padding),
        max(0, bbox[1] - padding),
        bbox[2] + padding,
        bbox[3] + padding,
    ]
    draw.rectangle(background, fill=(0, 0, 0, 180))
    draw.multiline_text((10, 10), text, fill=(255, 255, 255, 255), font=font)

    blended.convert("RGB").save(output_path)


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    use_half = device.type == "cuda"

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz"
    if args.visualize_n > 0:
        viz_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(args.label_map_json)

    model, preprocess, mask_transform = load_alpha_clip(
        args.model_name, args.alpha_ckpt, device, args.image_size
    )
    text_encoder = TextEncoder(model, device, args.batch_size)

    images, annotations, categories = load_panoptic(args.panoptic_json, args.panoptic_seg_dir)

    summary_writer = EvaluationAccumulator()

    results_path = out_dir / "results.jsonl"
    csv_rows: List[dict] = []
    jsonl_file = results_path.open("w", encoding="utf-8")

    visualize_budget = args.visualize_n

    image_ids = list(images.keys())
    image_ids.sort()
    if args.limit_images is not None:
        image_ids = image_ids[: args.limit_images]

    logged_dtype = False

    try:
        for image_id in tqdm(image_ids, desc="Images", unit="img"):
            image_record = images[image_id]
            image_path = args.images_dir / image_record["file_name"]
            try:
                image_pil = Image.open(image_path).convert("RGB")
            except FileNotFoundError:
                print(f"[Warning] Missing image: {image_path}")
                summary_writer.register_skip()
                continue

            anns = annotations.get(image_id, [])
            if not anns:
                summary_writer.register_skip()
                continue

            instance_iter = iter_image_instances(
                image_record,
                anns,
                args.images_dir,
                args.panoptic_seg_dir,
                categories,
                args.min_area,
            )

            for instance in instance_iter:
                try:
                    region_feature = encode_region(
                        model,
                        preprocess,
                        mask_transform,
                        image_pil,
                        instance.mask,
                        device,
                        use_half,
                        log_dtype=not logged_dtype,
                    )
                    logged_dtype = True
                except Exception as exc:
                    print(f"[Warning] Failed to encode region {instance.segment_id}: {exc}")
                    summary_writer.register_skip()
                    continue

                candidate_labels, phrase_groups = build_text_candidates(
                    instance.category_name, args.prompt_template, label_map
                )
                flattened_phrases = []
                for group in phrase_groups:
                    for phrase in group:
                        if phrase not in flattened_phrases:
                            flattened_phrases.append(phrase)
                phrase_embeddings = text_encoder.encode(flattened_phrases)
                sims = aggregate_similarity(
                    region_feature, phrase_groups, phrase_embeddings, args.agg
                )

                max_score = max(sims) if sims else -float("inf")
                if args.reject_threshold is not None and max_score < args.reject_threshold:
                    pred_idx = 2  # unknown
                else:
                    pred_idx = int(np.argmax(sims))

                pred_label = candidate_labels[pred_idx]
                hit = pred_label == instance.category_name

                summary_writer.update(instance.category_name, instance.isthing, hit, pred_label)

                if summary_writer.total % 1000 == 0:
                    acc = summary_writer.hits / summary_writer.total
                    print(
                        f"Processed {summary_writer.total} instances - accuracy: {acc:.4f}"
                    )

                score_dict = {
                    "gt": sims[0],
                    "other": sims[1],
                    "unknown": sims[2],
                    "none": sims[3],
                }

                result_entry = {
                    "image_id": instance.image_id,
                    "file_name": instance.file_name,
                    "segment_id": instance.segment_id,
                    "category_id": instance.category_id,
                    "gt_name": instance.category_name,
                    "candidate_labels": candidate_labels,
                    "thing_or_stuff": "thing" if instance.isthing else "stuff",
                    "scores": score_dict,
                    "pred": pred_label,
                    "hit": hit,
                    "area": instance.area,
                    "bbox": list(instance.bbox),
                }
                jsonl_file.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

                if args.save_csv:
                    csv_rows.append(result_entry)

                if visualize_budget > 0:
                    viz_path = viz_dir / f"{instance.image_id}_{instance.segment_id}.png"
                    visualize_instance(
                        image_pil,
                        instance.mask,
                        viz_path,
                        instance.category_name,
                        pred_label,
                        sims,
                        candidate_labels,
                    )
                    visualize_budget -= 1

            image_pil.close()
    finally:
        jsonl_file.close()

    summary = summary_writer.summary()

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    confusion_df = pd.DataFrame.from_dict(summary_writer.confusion, orient="index")
    confusion_df.index.name = "gt_label"
    confusion_df = confusion_df.fillna(0).astype(int)
    confusion_path = out_dir / "confusion.csv"
    confusion_df.to_csv(confusion_path, encoding="utf-8")

    if args.save_csv:
        results_csv_path = out_dir / "results.csv"
        results_df = pd.DataFrame(csv_rows)
        results_df.to_csv(results_csv_path, index=False, encoding="utf-8")

    print("==== Evaluation Summary ====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results JSONL: {results_path}")
    if args.save_csv:
        print(f"Results CSV: {results_csv_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Confusion CSV: {confusion_path}")
    if args.visualize_n > 0:
        print(f"Visualization samples saved to: {viz_dir}")


if __name__ == "__main__":
    main()

