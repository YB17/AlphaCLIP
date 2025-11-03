"""Enhanced Alpha-CLIP COCO Panoptic evaluation script.

This script extends the vanilla ``test_alphaclip_coco_panoptic.py`` utility with
advanced preprocessing and postprocessing capabilities aimed at improving
zero-shot recognition of text-weak categories such as ``person``.  The
implementation keeps the original entry points compatible while adding
fine-grained command line toggles and logging for easy ablations.

The script only performs preprocessing / postprocessing (no training) and keeps
all outputs compatible with the legacy version while appending additional
statistics for transparency.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from panopticapi.utils import rgb2id
from tqdm import tqdm

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with defaults closely matching the legacy script.
    """

    parser = argparse.ArgumentParser(
        description="Enhanced Alpha-CLIP evaluation on COCO Panoptic segments."
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
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--seed", type=int, default=42)

    # Legacy compatibility switches
    parser.add_argument("--prompt-template", default="{label}")
    parser.add_argument("--agg", choices=["max", "mean"], default="max")
    parser.add_argument("--reject-threshold", type=float)

    # New preprocessing / geometry toggles
    parser.add_argument("--alpha-soft", action="store_true", default=False)
    parser.add_argument("--alpha-blur", type=float, default=0.0)

    # Two-view options
    parser.add_argument("--two-view", action="store_true", default=False)
    parser.add_argument("--bbox-pad", type=float, default=0.08)
    parser.add_argument(
        "--two-view-merge", choices=["mean", "max"], default="mean"
    )

    # Text processing options
    parser.add_argument("--label-map-json", type=Path)
    parser.add_argument(
        "--stuff-extra-surface", dest="stuff_extra_surface", action="store_true", default=True
    )
    parser.add_argument("--no-stuff-extra-surface", dest="stuff_extra_surface", action="store_false")
    parser.add_argument("--with-rejects", dest="with_rejects", action="store_true", default=True)
    parser.add_argument("--no-rejects", dest="with_rejects", action="store_false")
    parser.add_argument("--warmup-text-embeddings", action="store_true", default=False)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--logit-bias-json", type=Path)
    parser.add_argument("--per-class-thresholds-json", type=Path)
    parser.add_argument("--default-threshold", type=float, default=0.0)

    # Debugging / visualization helpers
    parser.add_argument("--visualize-n", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("./alphaclip_panoptic_eval"))
    parser.add_argument("--save-csv", dest="save_csv", action="store_true", default=True)
    parser.add_argument("--no-save-csv", dest="save_csv", action="store_false")
    parser.add_argument("--dump-debug-overlays", action="store_true", default=False)
    parser.add_argument("--debug-every", type=int, default=200)

    # Ablation toggles
    parser.add_argument("--no-naturalize", action="store_true", default=False)
    parser.add_argument("--no-synonyms", action="store_true", default=False)
    parser.add_argument("--no-two-view", action="store_true", default=False)
    parser.add_argument("--no-bias", action="store_true", default=False)
    parser.add_argument("--no-thresholds", action="store_true", default=False)

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class PanopticSample:
    """Light-weight container describing a single panoptic segment."""

    image_id: int
    file_name: str
    image_path: Path
    segment_id: int
    mask: np.ndarray
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
) -> Tuple[torch.nn.Module, T.Compose]:
    """Load Alpha-CLIP model and preprocessing pipeline.

    Returns
    -------
    Tuple[torch.nn.Module, torchvision.transforms.Compose]
        The model and its image preprocessing pipeline.

    Complexity
    ----------
    Dominated by model loading time.
    """

    try:
        import alpha_clip
    except ImportError as exc:  # pragma: no cover - informative
        raise RuntimeError(
            "Failed to import alpha_clip. Please install Alpha-CLIP before running this script."
        ) from exc

    model, preprocess = alpha_clip.load(
        model_name,
        device=str(device),
        alpha_vision_ckpt_pth=str(alpha_ckpt),
        lora_adapt=False,
        rank=-1,
    )
    model.eval()

    return model, preprocess


def summarize_preprocess(preprocess: T.Compose) -> List[str]:
    """Return a human readable summary of torchvision preprocessing pipeline.

    Complexity is O(N) where N is the number of transforms in the compose.
    """

    summary: List[str] = []
    for idx, transform in enumerate(getattr(preprocess, "transforms", [])):
        summary.append(f"[{idx}] {transform}")
    return summary


def clone_geometry_from_preprocess(
    preprocess: T.Compose,
    use_soft_alpha: bool,
    soft_alpha_blur: float = 0.0,
) -> Callable[[Image.Image], Image.Image]:
    """Clone geometric preprocessing ops from an image pipeline for alpha masks.

    Parameters
    ----------
    preprocess: torchvision.transforms.Compose
        Full preprocessing pipeline for the RGB image.
    use_soft_alpha: bool
        If ``True`` the resize interpolation uses bilinear filtering and an
        optional blur is applied to produce soft mask edges.
    soft_alpha_blur: float, default ``0.0``
        Optional Gaussian blur radius applied after all geometric transforms
        when ``use_soft_alpha`` is enabled.

    Notes
    -----
    Non-geometric operations (Normalize, ToTensor, etc.) are ignored. Only
    ``Resize`` and ``CenterCrop`` are mirrored. Other transform types pass
    through untouched.

    Returns
    -------
    Callable[[PIL.Image.Image], PIL.Image.Image]
        Function that applies the cloned geometry operations to a mask image.
    
    Complexity
    ----------
    O(N) over the number of transforms in ``preprocess`` for extraction and per
    call application.
    """

    geometry_ops: List[Tuple[str, dict]] = []
    for transform in getattr(preprocess, "transforms", []):
        if isinstance(transform, T.Resize):
            size = transform.size
            antialias = getattr(transform, "antialias", None)
            geometry_ops.append(
                (
                    "resize",
                    {
                        "size": size,
                        "antialias": antialias,
                    },
                )
            )
        elif isinstance(transform, T.CenterCrop):
            geometry_ops.append(("center_crop", {"size": transform.size}))

    resample = Image.BILINEAR if use_soft_alpha else Image.NEAREST

    def _apply(mask: Image.Image) -> Image.Image:
        out = mask
        for kind, kwargs in geometry_ops:
            if kind == "resize":
                size = kwargs["size"]
                antialias = kwargs.get("antialias", None)
                out = F.resize(out, size=size, interpolation=resample, antialias=antialias)
            elif kind == "center_crop":
                out = F.center_crop(out, kwargs["size"])
        if use_soft_alpha and soft_alpha_blur > 0:
            out = out.filter(ImageFilter.GaussianBlur(radius=soft_alpha_blur))
        return out

    return _apply


def compute_bbox_from_mask(
    mask_np: np.ndarray,
    pad_ratio: float,
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Compute a padded bounding box from a binary mask.

    Parameters
    ----------
    mask_np: np.ndarray
        Binary/boolean mask of shape ``(H, W)``.
    pad_ratio: float
        Padding factor relative to the mask box size. ``0.08`` means an 8%
        expansion on each side.
    image_size: Tuple[int, int]
        Target image size in ``(width, height)`` order to clamp the bbox.

    Boundary Conditions
    -------------------
    If the mask is empty the function returns the full-image bounds.

    Returns
    -------
    Tuple[int, int, int, int]
        Padded bounding box ``(x1, y1, x2, y2)`` with ``x2``/``y2`` exclusive.

    Complexity
    ----------
    O(P) where ``P`` is the number of positive pixels (dominated by
    ``np.argwhere``).
    """

    if mask_np.dtype != np.bool_:
        mask_bool = mask_np.astype(bool)
    else:
        mask_bool = mask_np

    positions = np.argwhere(mask_bool)
    if positions.size == 0:
        width, height = image_size
        return (0, 0, width, height)

    y_min, x_min = positions.min(axis=0)
    y_max, x_max = positions.max(axis=0)

    width = image_size[0]
    height = image_size[1]

    # Convert to inclusive integer coords then pad
    x1 = float(x_min)
    y1 = float(y_min)
    x2 = float(x_max + 1)
    y2 = float(y_max + 1)

    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    pad_w = box_w * pad_ratio
    pad_h = box_h * pad_ratio

    x1 = max(0.0, x1 - pad_w)
    y1 = max(0.0, y1 - pad_h)
    x2 = min(float(width), x2 + pad_w)
    y2 = min(float(height), y2 + pad_h)

    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)

    return (int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2)), int(math.ceil(y2)))


SPECIAL_LABEL_MAP = {
    "wall-other-merged": "wall",
    "sky-other-merged": "sky",
    "pavement-merged": "pavement",
    "window-blind": "window blinds",
    "mirror-stuff": "mirror",
    "wall-brick": "brick wall",
}


def naturalize_label(raw: str) -> str:
    """Convert non-natural COCO labels into more readable phrases.

    The heuristic removes dataset-specific suffixes (``-merged``, ``-stuff``,
    ``-other``) and replaces hyphenated tokens with spaces. A small lookup table
    handles particularly odd labels.

    Parameters
    ----------
    raw: str
        Original category name from COCO panoptic metadata.

    Returns
    -------
    str
        Human-friendly label name.

    Complexity
    ----------
    O(L) for label length due to string replacements.
    """

    lowered = raw.lower()
    if lowered in SPECIAL_LABEL_MAP:
        return SPECIAL_LABEL_MAP[lowered]

    cleaned = lowered
    for suffix in ("-merged", "-stuff", "-other"):
        cleaned = cleaned.replace(suffix, "")
    cleaned = cleaned.replace("-", " ")
    cleaned = cleaned.replace("_", " ")
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("  ", " ")
    return cleaned


def load_label_map(label_map_path: Optional[Path]) -> Optional[Dict[str, List[str]]]:
    """Load external label map JSON describing synonym lists.

    Parameters
    ----------
    label_map_path: pathlib.Path or None
        Path to the JSON file mapping labels to synonym phrases.

    Returns
    -------
    Optional[Dict[str, List[str]]]
        Lower-cased keys with list of synonyms, or ``None`` when path missing.

    Complexity
    ----------
    O(M) where ``M`` is the number of entries in the JSON file.
    """

    if not label_map_path:
        return None
    with label_map_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {k.lower(): list(v) for k, v in data.items()}


def build_default_synonyms() -> Dict[str, List[str]]:
    """Return built-in synonym lists focusing on people-centric categories.

    Returns
    -------
    Dict[str, List[str]]
        Hard-coded mapping emphasising ``person`` variants.

    Complexity
    ----------
    O(1) since the mapping is static.
    """

    return {
        "person": [
            "person",
            "people",
            "a person",
            "a man",
            "a woman",
            "a boy",
            "a girl",
            "a human",
            "a pedestrian",
            "a crowd",
        ]
    }


class TextEncoder:
    """Encode text phrases with caching and optional warmup."""

    def __init__(self, model: torch.nn.Module, device: torch.device, batch_size: int) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.cache: Dict[str, torch.Tensor] = {}
        self._resolve_tokenizer()
        self.requests = 0
        self.cache_hits = 0

    def _resolve_tokenizer(self) -> None:
        try:
            from alpha_clip import tokenize as alpha_tokenize

            self.tokenize = alpha_tokenize
            return
        except ImportError:
            pass

        try:
            import clip  # type: ignore

            self.tokenize = clip.tokenize  # type: ignore[attr-defined]
            return
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import tokenizer from alpha_clip or clip."
            ) from exc

    def encode(self, phrases: Sequence[str]) -> Dict[str, torch.Tensor]:
        """Encode phrases, caching previous results.

        Parameters
        ----------
        phrases: Sequence[str]
            Iterable of phrases to encode.

        Returns
        -------
        Dict[str, torch.Tensor]
            Mapping phrase to normalized embedding on ``self.device``.

        Complexity
        ----------
        O(U) for the number of uncached phrases ``U`` plus O(K) dictionary
        lookups for ``K`` requested phrases.
        """

        missing = [p for p in phrases if p not in self.cache]
        self.requests += len(phrases)
        self.cache_hits += len(phrases) - len(missing)

        if missing:
            with torch.no_grad():
                for start in range(0, len(missing), self.batch_size):
                    chunk = missing[start : start + self.batch_size]
                    tokens = self.tokenize(chunk).to(self.device)
                    text_features = self.model.encode_text(tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    for phrase, feature in zip(chunk, text_features):
                        self.cache[phrase] = feature.detach().to(self.device)
        return {p: self.cache[p] for p in phrases}

    def warmup(self, phrases: Sequence[str]) -> None:
        """Pre-encode a list of phrases to populate the cache.

        Complexity
        ----------
        O(N) for ``N`` phrases; leverages batching for efficiency.
        """

        start = time.time()
        self.encode(list(phrases))
        duration = time.time() - start
        print(
            f"[Warmup] Encoded {len(set(phrases))} unique phrases in {duration:.2f}s; "
            f"cache size now {len(self.cache)}"
        )

    def cache_hit_rate(self) -> float:
        """Return current cache hit rate in percentage."""

        return (self.cache_hits / max(self.requests, 1)) * 100.0


def apply_template(template: str, label: str) -> str:
    """Legacy compatibility template application."""

    if "{label}" in template:
        return template.replace("{label}", label)
    return template.strip() + " " + label


class LabelVocabulary:
    """Handle label naturalization, templating and synonym expansion."""

    def __init__(
        self,
        categories: Dict[int, dict],
        label_map_external: Optional[Dict[str, List[str]]],
        prompt_template: str,
        stuff_extra_surface: bool,
        with_rejects: bool,
        naturalize_enabled: bool,
        synonyms_enabled: bool,
    ) -> None:
        self.categories = categories
        self.prompt_template = prompt_template
        self.label_map_external = label_map_external or {}
        self.stuff_extra_surface = stuff_extra_surface
        self.with_rejects = with_rejects
        self.naturalize_enabled = naturalize_enabled
        self.synonyms_enabled = synonyms_enabled

        self.candidate_labels: List[str] = []
        self.phrase_groups: List[List[str]] = []
        self.template_types: List[str] = []
        self.label_to_phrases: Dict[str, List[str]] = {}
        self.label_to_template: Dict[str, str] = {}

        self._build()

    def _build(self) -> None:
        ordered_ids = sorted(self.categories.keys())
        default_synonyms = build_default_synonyms()

        for cid in ordered_ids:
            meta = self.categories[cid]
            raw_name = meta.get("name", str(cid))
            natural_name = naturalize_label(raw_name) if self.naturalize_enabled else raw_name
            lookup_key = natural_name.lower()
            isthing = bool(meta.get("isthing", 0))
            base_template = "a photo of a {label}" if isthing else "a photo of the {label}"

            phrases: List[str] = []
            templated = apply_template(base_template, natural_name)
            phrases.append(templated)

            synonym_sources: List[str] = []
            if self.synonyms_enabled:
                if lookup_key in default_synonyms:
                    synonym_sources.extend(default_synonyms[lookup_key])
                if lookup_key in self.label_map_external:
                    synonym_sources.extend(self.label_map_external[lookup_key])

            for phrase in synonym_sources:
                formatted = self._template_phrase(base_template, phrase)
                if formatted not in phrases:
                    phrases.append(formatted)

            if not self.synonyms_enabled and lookup_key in self.label_map_external:
                # Keep deterministic behaviour for ablation: use primary templated phrase only
                pass

            if not isthing and self.stuff_extra_surface:
                extra = self._template_phrase("a photo of the {label}", f"{natural_name} surface")
                if extra not in phrases:
                    phrases.append(extra)

            self.candidate_labels.append(raw_name)
            self.phrase_groups.append(phrases)
            self.template_types.append("thing" if isthing else "stuff")
            self.label_to_phrases[raw_name] = phrases
            self.label_to_template[raw_name] = base_template

        if self.with_rejects:
            for reject in ["other", "unknown", "none"]:
                self.candidate_labels.append(reject)
                self.phrase_groups.append([reject])
                self.template_types.append("reject")
                self.label_to_phrases[reject] = [reject]
                self.label_to_template[reject] = "identity"

    @staticmethod
    def _template_phrase(template: str, phrase: str) -> str:
        if "{label}" in template:
            return template.replace("{label}", phrase)
        # If the phrase already starts with an article, return directly
        lowered = phrase.lower().strip()
        if lowered.startswith(("a ", "an ", "the ", "this ", "that ", "these ", "those ")):
            return phrase
        return f"{template.strip()} {phrase}".strip()

    def flatten_phrases(self) -> List[str]:
        seen = set()
        flat: List[str] = []
        for group in self.phrase_groups:
            for phrase in group:
                if phrase not in seen:
                    flat.append(phrase)
                    seen.add(phrase)
        return flat


def load_panoptic(panoptic_json: Path, panoptic_seg_dir: Path) -> Tuple[dict, dict, dict]:
    """Load COCO panoptic annotations."""

    with panoptic_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {c["id"]: c for c in data["categories"]}
    images = {img["id"]: img for img in data["images"]}
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)

    if not panoptic_seg_dir.exists():
        raise FileNotFoundError(f"Panoptic segmentation directory not found: {panoptic_seg_dir}")

    return images, annotations, categories


def iter_image_instances(
    image_record: dict,
    annotations: Sequence[dict],
    images_dir: Path,
    panoptic_seg_dir: Path,
    categories: Dict[int, dict],
    min_area: int,
) -> Iterator[PanopticSample]:
    """Yield panoptic segments for a given image."""

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
    geometry_cloner: Callable[[Image.Image], Image.Image],
    image_pil: Image.Image,
    mask_bool: np.ndarray,
    device: torch.device,
    use_half: bool,
    log_dtype: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode the masked view and return tensors ready for fusion.

    Parameters
    ----------
    model: torch.nn.Module
        Alpha-CLIP model instance.
    preprocess: torchvision.transforms.Compose
        RGB preprocessing pipeline.
    geometry_cloner: Callable[[PIL.Image.Image], PIL.Image.Image]
        Function that applies cloned geometry ops to the mask.
    image_pil: PIL.Image.Image
        Original RGB image.
    mask_bool: np.ndarray
        Boolean mask in original resolution.
    device: torch.device
        Target device for the tensors.
    use_half: bool
        Whether to cast tensors to ``float16``.
    log_dtype: bool
        Whether to print tensor dtype/device debug information.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Normalized region feature vector and processed alpha tensor.

    Complexity
    ----------
    Dominated by the forward pass through the vision encoder.
    """

    image_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    mask_pil = Image.fromarray(mask_uint8, mode="L")
    aligned_mask = geometry_cloner(mask_pil)
    tensor_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.26])])
    alpha_tensor = tensor_transform(aligned_mask).unsqueeze(0).to(device)

    if use_half:
        image_tensor = image_tensor.half()
        alpha_tensor = alpha_tensor.half()

    if log_dtype:
        print(
            f"[Debug] image tensor dtype={image_tensor.dtype} device={image_tensor.device}; "
            f"alpha dtype={alpha_tensor.dtype} device={alpha_tensor.device}"
        )

    with torch.no_grad():
        feat = model.visual(image_tensor, alpha_tensor)
    feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return feat.squeeze(0), alpha_tensor.squeeze(0)


def crop_and_preprocess(
    image_pil: Image.Image,
    bbox: Tuple[int, int, int, int],
    preprocess: T.Compose,
    device: torch.device,
    use_half: bool,
) -> torch.Tensor:
    """Crop image by bbox and preprocess with the same pipeline.

    Parameters
    ----------
    image_pil: PIL.Image.Image
        Input RGB image.
    bbox: Tuple[int, int, int, int]
        Bounding box ``(x1, y1, x2, y2)`` in pixel coordinates.
    preprocess: torchvision.transforms.Compose
        Image preprocessing pipeline identical to the main view.
    device: torch.device
        Target device for the resulting tensor.
    use_half: bool
        If ``True`` cast tensor to ``float16`` to mirror the main pipeline.

    Returns
    -------
    torch.Tensor
        Preprocessed tensor of shape ``(1, C, H, W)``.

    Complexity
    ----------
    O(P) where ``P`` is the number of pixels in the cropped region.
    """

    x1, y1, x2, y2 = bbox
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)
    crop = image_pil.crop((x1, y1, x2, y2))
    tensor = preprocess(crop).unsqueeze(0).to(device)
    if use_half:
        tensor = tensor.half()
    return tensor


def apply_temperature_and_bias(
    sim_vec: torch.Tensor,
    candidate_labels: List[str],
    temperature: float,
    bias_map: Dict[str, float],
) -> torch.Tensor:
    """Apply temperature scaling and additive bias to similarity scores.

    Parameters
    ----------
    sim_vec: torch.Tensor
        Similarity scores (cosine) for each candidate label.
    candidate_labels: List[str]
        Label order corresponding to ``sim_vec``.
    temperature: float
        Softmax temperature; values >1.0 sharpen logits, <1.0 smooth them.
    bias_map: Dict[str, float]
        Optional additive bias per label. Missing keys default to zero.

    Returns
    -------
    torch.Tensor
        Adjusted logits respecting the input device/dtype.

    Complexity
    ----------
    O(K) where ``K`` is the number of labels.
    """

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    scaled = sim_vec / temperature
    bias_tensor = torch.tensor([
        bias_map.get(label, 0.0) for label in candidate_labels
    ], device=sim_vec.device, dtype=sim_vec.dtype)
    return scaled + bias_tensor


def prepare_bias_map(
    candidate_labels: List[str],
    bias_json: Optional[Path],
    disable_bias: bool,
) -> Dict[str, float]:
    """Create a label->bias map with sensible defaults.

    Parameters
    ----------
    candidate_labels: List[str]
        Labels participating in inference.
    bias_json: pathlib.Path or None
        Optional JSON file overriding biases.
    disable_bias: bool
        When ``True`` forces all biases to zero regardless of JSON input.

    Returns
    -------
    Dict[str, float]
        Mapping label -> additive bias in logit space. When no JSON is provided
        the default assigns ``-0.007`` to ``other/unknown/none`` and ``0``
        elsewhere.

    Complexity
    ----------
    O(K) for ``K`` labels.
    """

    if disable_bias:
        return {label: 0.0 for label in candidate_labels}

    if bias_json:
        with bias_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: float(v) for k, v in data.items()}

    default_bias = {label: 0.0 for label in candidate_labels}
    for label in ("other", "unknown", "none"):
        if label in default_bias:
            default_bias[label] = -0.007
    return default_bias


def prepare_thresholds(
    candidate_labels: List[str],
    default_threshold: float,
    thresholds_json: Optional[Path],
    disable_thresholds: bool,
) -> Dict[str, float]:
    """Prepare per-class decision thresholds.

    Parameters
    ----------
    candidate_labels: List[str]
        Labels to evaluate.
    default_threshold: float
        Base threshold used when per-class value is absent.
    thresholds_json: pathlib.Path or None
        Optional JSON overriding thresholds.
    disable_thresholds: bool
        When ``True`` every threshold becomes ``-inf``.

    Returns
    -------
    Dict[str, float]
        Mapping label -> decision threshold with ``person`` defaulting to ``0.24``.

    Complexity
    ----------
    O(K) for ``K`` labels.
    """

    thresholds = {label: default_threshold for label in candidate_labels}
    if disable_thresholds:
        return {label: -float("inf") for label in candidate_labels}

    if thresholds_json:
        with thresholds_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for label, value in data.items():
            thresholds[label] = float(value)

    if "person" in thresholds and thresholds["person"] == default_threshold:
        thresholds["person"] = 0.24

    return thresholds


class EvaluationAccumulator:
    """Accumulate evaluation statistics including ablation metrics."""

    def __init__(self) -> None:
        self.total = 0
        self.hits = 0
        self.thing_total = 0
        self.thing_hits = 0
        self.stuff_total = 0
        self.stuff_hits = 0
        self.skipped = 0
        self.category_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"total": 0, "hits": 0})
        self.confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.person_routes = {"other": 0, "unknown": 0, "none": 0}
        self.person_total = 0
        self.person_hits = 0
        self.two_view_enabled = 0
        self.two_view_better = 0
        self.two_view_worse = 0
        self.two_view_same = 0
        self.two_view_mask_hits = 0
        self.two_view_merged_hits = 0

    def register_skip(self) -> None:
        self.skipped += 1

    def update(
        self,
        sample: PanopticSample,
        pred_label: str,
        hit: bool,
        mask_only_hit: Optional[bool],
        used_two_view: bool,
    ) -> None:
        self.total += 1
        if hit:
            self.hits += 1
        if sample.isthing:
            self.thing_total += 1
            if hit:
                self.thing_hits += 1
        else:
            self.stuff_total += 1
            if hit:
                self.stuff_hits += 1

        stats = self.category_stats[sample.category_name]
        stats["total"] += 1
        if hit:
            stats["hits"] += 1

        if sample.category_name == "person":
            self.person_total += 1
            if hit:
                self.person_hits += 1
            elif pred_label in self.person_routes:
                self.person_routes[pred_label] += 1

        col = "gt" if hit else pred_label
        self.confusion[sample.category_name][col] += 1

        if used_two_view:
            self.two_view_enabled += 1
            if mask_only_hit is not None:
                if mask_only_hit and hit:
                    self.two_view_same += 1
                elif mask_only_hit and not hit:
                    self.two_view_worse += 1
                elif (not mask_only_hit) and hit:
                    self.two_view_better += 1
                else:
                    self.two_view_same += 1
                if mask_only_hit:
                    self.two_view_mask_hits += 1
                if hit:
                    self.two_view_merged_hits += 1

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

        two_view_stats = {
            "enabled": self.two_view_enabled,
            "better": self.two_view_better,
            "worse": self.two_view_worse,
            "same": self.two_view_same,
            "mask_hits": self.two_view_mask_hits,
            "merged_hits": self.two_view_merged_hits,
            "accuracy_gain": (
                (self.two_view_merged_hits - self.two_view_mask_hits) / max(self.two_view_enabled, 1)
                if self.two_view_enabled
                else 0.0
            ),
        }

        routes_ratio = {
            k: (v / self.person_total if self.person_total else 0.0)
            for k, v in self.person_routes.items()
        }
        person_summary = {
            "total": self.person_total,
            "hits": self.person_hits,
            "routes": self.person_routes,
            "routes_ratio": routes_ratio,
        }

        return {
            "total_instances": self.total,
            "skipped_instances": self.skipped,
            "accuracy": overall_acc,
            "thing_accuracy": thing_acc,
            "stuff_accuracy": stuff_acc,
            "per_category": per_category,
            "two_view": two_view_stats,
            "person": person_summary,
        }


def visualize_debug_overlay(
    image_pil: Image.Image,
    mask_bool: np.ndarray,
    output_path: Path,
    gt_label: str,
    pred_label: str,
    top_score: float,
    top_score_adj: float,
) -> None:
    """Save overlay visualizations for debugging.

    Parameters
    ----------
    image_pil: PIL.Image.Image
        Original RGB image.
    mask_bool: np.ndarray
        Boolean mask aligned with the image.
    output_path: pathlib.Path
        Target path for the visualization.
    gt_label: str
        Ground-truth category name.
    pred_label: str
        Predicted category name after calibration.
    top_score: float
        Raw cosine similarity of the winning label.
    top_score_adj: float
        Calibrated logit for the winning label.

    Complexity
    ----------
    O(P) where ``P`` is the number of pixels in the image.
    """

    mask_img = Image.fromarray(mask_bool.astype(np.uint8) * 255)
    edges = mask_img.filter(ImageFilter.FIND_EDGES)
    overlay = image_pil.convert("RGBA")
    edge_rgba = Image.new("RGBA", overlay.size, (255, 0, 0, 0))
    edge_rgba.paste((255, 0, 0, 255), mask=edges)
    blended = Image.alpha_composite(overlay, edge_rgba)

    draw = ImageDraw.Draw(blended)
    text = f"GT: {gt_label}\nPred: {pred_label}\nscore={top_score:.3f} adj={top_score_adj:.3f}"
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=16)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.multiline_textbbox((10, 10), text, font=font)
    draw.rectangle(
        [bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4],
        fill=(0, 0, 0, 160),
    )
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
    debug_dir = out_dir / "debug_overlays"
    if args.dump_debug_overlays:
        debug_dir.mkdir(parents=True, exist_ok=True)

    model, preprocess = load_alpha_clip(args.model_name, args.alpha_ckpt, device, args.image_size)

    preprocess_summary = summarize_preprocess(preprocess)
    print("[Preprocess] Pipeline summary:")
    for line in preprocess_summary:
        print("  " + line)

    geometry_cloner = clone_geometry_from_preprocess(
        preprocess,
        use_soft_alpha=args.alpha_soft,
        soft_alpha_blur=args.alpha_blur,
    )
    print(
        f"[Alpha] Soft={args.alpha_soft} blur={args.alpha_blur}; geometry ops cloned: {len(preprocess_summary)}"
    )

    images, annotations, categories = load_panoptic(args.panoptic_json, args.panoptic_seg_dir)
    label_map_external = load_label_map(args.label_map_json)

    vocab = LabelVocabulary(
        categories,
        label_map_external,
        args.prompt_template,
        stuff_extra_surface=args.stuff_extra_surface,
        with_rejects=args.with_rejects,
        naturalize_enabled=not args.no_naturalize,
        synonyms_enabled=not args.no_synonyms,
    )

    print(f"[Labels] Total candidate labels: {len(vocab.candidate_labels)}")
    person_phrases = vocab.label_to_phrases.get("person", [])[:5]
    print(f"[Labels] Person phrase examples: {person_phrases}")

    text_encoder = TextEncoder(model, device, args.batch_size)
    all_phrases = vocab.flatten_phrases()
    phrase_groups = vocab.phrase_groups
    candidate_labels = vocab.candidate_labels
    print(f"[Text] Total unique phrases: {len(all_phrases)}")

    if args.warmup_text_embeddings:
        text_encoder.warmup(all_phrases)

    bias_map = prepare_bias_map(candidate_labels, args.logit_bias_json, args.no_bias)
    thresholds = prepare_thresholds(
        candidate_labels,
        default_threshold=args.default_threshold,
        thresholds_json=args.per_class_thresholds_json,
        disable_thresholds=args.no_thresholds,
    )
    print("[Bias] First entries:", list(bias_map.items())[:5])
    print("[Thresholds] First entries:", list(thresholds.items())[:5])

    summary_writer = EvaluationAccumulator()

    results_path = out_dir / "results.jsonl"
    jsonl_file = results_path.open("w", encoding="utf-8")
    csv_rows: List[dict] = []

    visualize_budget = args.visualize_n
    image_ids = sorted(images.keys())
    if args.limit_images is not None:
        image_ids = image_ids[: args.limit_images]

    logged_dtype = False
    processed_samples = 0

    two_view_enabled = args.two_view and not args.no_two_view

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
                processed_samples += 1
                try:
                    region_feature, alpha_tensor = encode_region(
                        model,
                        preprocess,
                        geometry_cloner,
                        image_pil,
                        instance.mask,
                        device,
                        use_half,
                        log_dtype=not logged_dtype,
                    )
                    if not logged_dtype:
                        logged_dtype = True
                        print(
                            f"[Shapes] Image tensor spatial dims: {region_feature.shape if region_feature.ndim > 1 else 'N/A'}"
                        )
                except Exception as exc:
                    print(f"[Warning] Failed to encode region {instance.segment_id}: {exc}")
                    summary_writer.register_skip()
                    continue

                if processed_samples <= 2:
                    image_tensor = preprocess(image_pil)
                    print(
                        f"[Check] Sample {processed_samples} image shape: {tuple(image_tensor.shape[-2:])}, "
                        f"alpha shape: {tuple(alpha_tensor.shape[-2:])}"
                    )
                    assert image_tensor.shape[-2:] == alpha_tensor.shape[-2:]

                phrase_embeddings = text_encoder.encode(all_phrases)
                if processed_samples == 1 or processed_samples % 2000 == 0:
                    print(
                        f"[Cache] After {processed_samples} samples hit rate={text_encoder.cache_hit_rate():.2f}%"
                    )

                # Compute similarities for each label group
                sims_raw: List[float] = []
                # Pre-compute region features for two-view if enabled
                mask_feature = region_feature
                merged_feature = region_feature
                mask_only_hit: Optional[bool] = None

                if two_view_enabled:
                    bbox = compute_bbox_from_mask(
                        instance.mask,
                        pad_ratio=args.bbox_pad,
                        image_size=image_pil.size,
                    )
                    box_tensor = crop_and_preprocess(
                        image_pil,
                        bbox,
                        preprocess,
                        device,
                        use_half,
                    )
                    with torch.no_grad():
                        box_feat = model.visual(box_tensor)
                    box_feat = box_feat / box_feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    box_feat = box_feat.squeeze(0)
                    if args.two_view_merge == "mean":
                        merged_feature = (mask_feature + box_feat) / 2.0
                    else:
                        merged_feature = torch.maximum(mask_feature, box_feat)
                    merged_feature = merged_feature / merged_feature.norm().clamp_min(1e-6)

                feature_for_scores = merged_feature if two_view_enabled else mask_feature

                group_scores = []
                for group in phrase_groups:
                    scores = []
                    for p in group:
                        emb = phrase_embeddings[p].to(device=feature_for_scores.device, dtype=feature_for_scores.dtype)
                        scores.append(float(torch.matmul(feature_for_scores, emb)))
                    if args.agg == "mean":
                        group_scores.append(sum(scores) / max(len(scores), 1))
                    else:
                        group_scores.append(max(scores))
                sims_raw = group_scores

                if two_view_enabled:
                    mask_only_scores = []
                    for group in phrase_groups:
                        scores = []
                        for p in group:
                            emb = phrase_embeddings[p].to(device=mask_feature.device, dtype=mask_feature.dtype)
                            scores.append(float(torch.matmul(mask_feature, emb)))
                        if args.agg == "mean":
                            mask_only_scores.append(sum(scores) / max(len(scores), 1))
                        else:
                            mask_only_scores.append(max(scores))
                    mask_only_idx = int(np.argmax(mask_only_scores))
                    mask_only_label = candidate_labels[mask_only_idx]
                    mask_only_hit = mask_only_label == instance.category_name

                sim_tensor = torch.tensor(sims_raw, device=device, dtype=region_feature.dtype)
                adjusted = apply_temperature_and_bias(sim_tensor, candidate_labels, args.temperature, bias_map)

                top1_idx = int(torch.argmax(adjusted).item())
                top1_label = candidate_labels[top1_idx]
                top1_score_adj = float(adjusted[top1_idx].item())
                top1_score_raw = sims_raw[top1_idx]
                top1_label_raw = top1_label

                sorted_adj, indices = torch.sort(adjusted, descending=True)
                margin_adj = float(sorted_adj[0] - sorted_adj[1]) if len(sorted_adj) > 1 else 0.0

                threshold = thresholds.get(top1_label, args.default_threshold)
                if args.no_thresholds:
                    threshold = -float("inf")
                below_threshold = top1_score_adj < threshold
                if args.reject_threshold is not None and top1_score_raw < args.reject_threshold:
                    below_threshold = True

                fallback_label = top1_label
                if below_threshold and args.with_rejects:
                    if "unknown" in candidate_labels:
                        fallback_label = "unknown"
                    elif "other" in candidate_labels:
                        fallback_label = "other"
                pred_label = fallback_label

                hit = pred_label == instance.category_name

                scores_raw_dict = {
                    label: float(sims_raw[idx]) for idx, label in enumerate(candidate_labels)
                }
                scores_adj_dict = {
                    label: float(adjusted[idx].item()) for idx, label in enumerate(candidate_labels)
                }

                summary_writer.update(
                    instance,
                    pred_label,
                    hit,
                    mask_only_hit,
                    used_two_view=two_view_enabled,
                )

                if summary_writer.total % 25 == 0:
                    acc = summary_writer.hits / summary_writer.total
                    print(f"[Progress] {summary_writer.total} instances processed - acc={acc:.4f}")

                result_entry = {
                    "image_id": instance.image_id,
                    "file_name": instance.file_name,
                    "segment_id": instance.segment_id,
                    "category_id": instance.category_id,
                    "gt_name": instance.category_name,
                    "candidate_labels": candidate_labels,
                    "thing_or_stuff": "thing" if instance.isthing else "stuff",
                    "pred": pred_label,
                    "hit": hit,
                    "area": instance.area,
                    "bbox": list(instance.bbox),
                    "top1_label_raw": top1_label_raw,
                    "top1_score_raw": float(top1_score_raw),
                    "top1_score_adj": float(top1_score_adj),
                    "margin_adj": float(margin_adj),
                    "used_two_view": two_view_enabled,
                    "alpha_soft": args.alpha_soft,
                    "bbox_pad": args.bbox_pad,
                    "temperature": args.temperature,
                    "threshold_used": threshold,
                    "below_threshold": below_threshold,
                    "chosen_template_type": vocab.template_types[top1_idx] if top1_idx < len(vocab.template_types) else "unknown",
                    "bias_other": bias_map.get("other", 0.0),
                    "bias_unknown": bias_map.get("unknown", 0.0),
                    "bias_none": bias_map.get("none", 0.0),
                    "person_threshold": thresholds.get("person", args.default_threshold),
                    "scores_raw": scores_raw_dict,
                    "scores_adj": scores_adj_dict,
                    "legacy_reject_threshold": args.reject_threshold,
                }

                jsonl_file.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

                if args.save_csv:
                    csv_rows.append(result_entry)

                if visualize_budget > 0:
                    viz_path = viz_dir / f"{instance.image_id}_{instance.segment_id}.png"
                    visualize_debug_overlay(
                        image_pil,
                        instance.mask,
                        viz_path,
                        instance.category_name,
                        pred_label,
                        top1_score_raw,
                        top1_score_adj,
                    )
                    visualize_budget -= 1

                if args.dump_debug_overlays and (summary_writer.total % args.debug_every == 0):
                    debug_path = debug_dir / f"debug_{instance.image_id}_{instance.segment_id}.png"
                    visualize_debug_overlay(
                        image_pil,
                        instance.mask,
                        debug_path,
                        instance.category_name,
                        pred_label,
                        top1_score_raw,
                        top1_score_adj,
                    )

            image_pil.close()
    finally:
        jsonl_file.close()

    summary = summary_writer.summary()
    summary["config"] = {
        "alpha_soft": args.alpha_soft,
        "alpha_blur": args.alpha_blur,
        "two_view": two_view_enabled,
        "bbox_pad": args.bbox_pad,
        "two_view_merge": args.two_view_merge,
        "temperature": args.temperature,
        "with_rejects": args.with_rejects,
        "stuff_extra_surface": args.stuff_extra_surface,
        "naturalize": not args.no_naturalize,
        "synonyms": not args.no_synonyms,
        "bias_disabled": args.no_bias,
        "thresholds_disabled": args.no_thresholds,
        "legacy_reject_threshold": args.reject_threshold,
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    confusion_df = pd.DataFrame.from_dict(summary_writer.confusion, orient="index").fillna(0).astype(int)
    confusion_df.index.name = "gt_label"
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
    if args.dump_debug_overlays:
        print(f"Debug overlays saved to: {debug_dir}")


if __name__ == "__main__":
    main()
