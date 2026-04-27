"""Ego-centered 360° minimap pipeline for the CSE 5509 final project.

This module implements a qualitative visualization pipeline for fixed-location,
in-place camera rotations (``direction 0`` ... ``direction 7``). The final
output is an object-level minimap built from projected detection rows
(``*_minimap_instances`` tables). It also produces a stitched BEV diagnostic
that is pixel-level and created by rotating/merging per-image BEV diagnostic
rasters.

Key assumptions:
    * Monocular depth (DPT) is approximate and not calibrated metric depth.
    * DPT output is treated as inverse depth by default.
    * Bounding-box bottom-center is used as a ground-contact proxy.
    * Direction indexing is clockwise from ``direction 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import csv
import importlib.util
import inspect
import json
import math
import re
import shutil


@dataclass
class PipelineConfig:
    """Configuration for dataset paths, model behavior, geometry, and outputs.

    Notes:
        * ``detection_threshold`` filters Mask R-CNN detections before
          projection.
        * ``minimap_min_confidence`` is the final confidence filter before
          writing rows to the minimap tables.
        * These two thresholds should usually be tuned together:
            - ``0.70`` is a balanced demo setting.
            - ``0.75`` is cleaner but can miss objects.
            - ``0.65`` may recover more objects but needs visual validation.
        * Zero-shot thresholds are candidate-generation controls and are kept
          slightly lower by default for recall.
    """
    repo_root: Path
    data_dir: Path
    output_dir: Path
    run_small_demo: bool = False
    demo_locations: int = 1
    demo_images_per_location: Optional[int] = None
    clean_output_dir: bool = False

    # Geometric / depth assumptions.
    # Xiaomi 13 main camera is ~23mm equivalent; ~76° horizontal FOV is a
    # practical starting point for 1x captures. Ultra-wide shots need larger FOV.
    horizontal_fov_deg: float = 76.0
    min_relative_range: float = 0.0
    max_relative_range: float = 10.0
    depth_is_inverse: bool = True

    # Per-image BEV diagnostic settings (stitched later for diagnostics).
    bev_max_range: float = 10.0
    bev_scale_px_per_range_unit: Optional[float] = None
    bev_height_px: int = 1400
    bev_width_px: int = 2000

    # Minimap settings (primary output).
    minimap_size_px: int = 1400
    minimap_max_range: float = 10.0
    minimap_scale_px_per_range_unit: Optional[float] = None
    minimap_draw_dense_points: bool = True
    minimap_draw_object_labels: bool = True
    minimap_min_confidence: float = 0.70
    minimap_label_top_k_per_location: int = 60
    minimap_marker_alpha: float = 0.85
    minimap_jitter_overlapping_markers: bool = True
    minimap_merge_nearby_same_class: bool = True
    minimap_merge_radius_units: float = 0.75
    minimap_merge_radius_by_class: Optional[Dict[str, float]] = None

    stitched_draw_dense_points: bool = True
    stitched_dense_alpha: float = 0.45

    # Direction convention.
    # heading_deg is clockwise degrees from +y (north/forward).
    direction_zero_heading_deg: float = 0.0
    direction_step_deg: float = 45.0
    direction_turn: str = "right"

    detection_threshold: float = 0.70
    image_nms_iou_threshold: float = 0.5
    use_homography_diagnostics: bool = True
    use_zero_shot_detector: bool = True
    zero_shot_model_name: str = "IDEA-Research/grounding-dino-tiny"
    zero_shot_box_threshold: float = 0.35
    zero_shot_text_threshold: float = 0.30


COCO_ID_TO_NAME = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
    10: "traffic light",
    13: "stop sign",
}

CLASS_COLORS = {
    "person": (247, 111, 111),
    "bicycle": (255, 194, 87),
    "car": (96, 187, 255),
    "motorcycle": (184, 143, 255),
    "bus": (255, 136, 52),
    "truck": (120, 232, 168),
    "dumpster": (166, 149, 121),
    "road_sign": (255, 230, 130),
}

GROUND_CLASS_NAMES = {"road", "sidewalk", "terrain"}
MOVABLE_CLASS_NAMES = {
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
}

TARGET_CLASS_NAMES = {"person", "car", "bus", "motorcycle", "bicycle", "dumpster", "road_sign"}
ZERO_SHOT_LABEL_PROMPTS = [
    "person",
    "car",
    "bus",
    "motorcycle",
    "bicycle",
    "dumpster",
    "trash dumpster",
    "road sign",
    "traffic sign",
    "stop sign",
]
ZERO_SHOT_CLASS_ALIASES = {
    "trash dumpster": "dumpster",
    "road sign": "road_sign",
    "traffic sign": "road_sign",
    "stop sign": "road_sign",
    "traffic light": "road_sign",
}
DEDUP_RADIUS_BY_CLASS_DEFAULT = {
    "person": 0.9,
    "bicycle": 0.45,
    "motorcycle": 0.45,
    "car": 0.75,
    "bus": 1.0,
    "dumpster": 0.65,
    "road_sign": 0.5,
}


def module_available(name: str) -> bool:
    """Return ``True`` when an importable module with ``name`` exists."""
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def get_cv2():
    """Import OpenCV if available; otherwise return ``None``."""
    try:
        import cv2

        return cv2
    except Exception:
        return None


def in_colab() -> bool:
    """Return whether the runtime appears to be Google Colab."""
    return module_available("google.colab")


def resolve_project_paths(project_dir: str | Path | None = None) -> Dict[str, Path]:
    """Resolve repository, data, and output paths.

    Args:
        project_dir: Optional repository root. If omitted, current working
            directory is used.

    Returns:
        Dictionary with keys ``repo_root``, ``data_dir`` (repo_root/data), and
        ``output_dir`` (repo_root/outputs).
    """
    if project_dir is not None:
        repo_root = Path(project_dir).expanduser().resolve()
    else:
        repo_root = Path.cwd().resolve()
    data_dir = repo_root / "data"
    output_dir = repo_root / "outputs"
    return {"repo_root": repo_root, "data_dir": data_dir, "output_dir": output_dir}


def discover_dataset(data_dir: Path) -> Dict[str, Any]:
    """Discover location folders and image files inside ``repo_root/data``.

    Args:
        data_dir: Dataset root expected to contain ``loc*/direction*.jpg``.

    Returns:
        Summary dictionary with location names, counts, and per-location files.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist.
        ValueError: If no locations or images are found.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. "
            "Expected repo_root/data/loc*/direction*.jpg."
        )

    locations = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not locations:
        raise ValueError(
            f"Dataset directory has no location folders: {data_dir}. "
            "Expected repo_root/data/loc*/direction*.jpg."
        )

    per_location: Dict[str, List[Path]] = {}
    total_images = 0

    for loc in locations:
        images = sorted(loc.glob("*.jpg")) + sorted(loc.glob("*.png"))
        per_location[loc.name] = images
        total_images += len(images)

    if total_images == 0:
        raise ValueError(
            f"Dataset directory contains no images: {data_dir}. "
            "Expected repo_root/data/loc*/direction*.jpg."
        )

    return {
        "location_count": len(locations),
        "total_images": total_images,
        "locations": [loc.name for loc in locations],
        "images_per_location": {k: len(v) for k, v in per_location.items()},
        "files": per_location,
    }


def ensure_output_dirs(base: Path, clean_output_dir: bool = False) -> Dict[str, Path]:
    """Create output directory structure used by the pipeline.

    Args:
        base: Root output directory.
        clean_output_dir: If ``True``, remove and recreate ``base``.

    Returns:
        Dictionary of concrete output paths.
    """
    if clean_output_dir and base.exists():
        shutil.rmtree(base)

    dirs = {
        "base": base,
        "per_image": base / "per_image",
        "per_location": base / "per_location",
        "tables": base / "tables",
        "diagnostics": base / "diagnostics",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def default_device() -> str:
    """Return ``cuda`` when available, otherwise ``cpu``."""
    if module_available("torch"):
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def load_rgb_image(path: Path):
    """Load a file as an RGB PIL image.

    Raises:
        FileNotFoundError: If the image path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    from PIL import Image

    return Image.open(path).convert("RGB")


def init_segmentation_model(device: str = "cpu") -> Dict[str, Any]:
    """Initialize SegFormer semantic segmentation model state."""
    if not module_available("transformers") or not module_available("torch"):
        return {"available": False, "reason": "transformers/torch not installed"}
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    model_name = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(model_name, do_reduce_labels=False)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return {
        "available": True,
        "processor": processor,
        "model": model,
        "id2label": model.config.id2label,
        "device": device,
    }


def init_depth_model(device: str = "cpu") -> Dict[str, Any]:
    """Initialize DPT depth estimation pipeline state."""
    if not module_available("transformers"):
        return {"available": False, "reason": "transformers not installed"}
    from transformers import pipeline

    pipe = pipeline("depth-estimation", model="Intel/dpt-large", device=0 if device.startswith("cuda") else -1)
    return {"available": True, "pipeline": pipe}


def init_instance_detector(device: str = "cpu") -> Dict[str, Any]:
    """Initialize Mask R-CNN instance detector state."""
    if not module_available("torchvision") or not module_available("torch"):
        return {"available": False, "reason": "torchvision/torch not installed", "fallback": "no_detections"}
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    preprocess = weights.transforms()
    categories = list(weights.meta.get("categories", []))
    return {"available": True, "model": model, "preprocess": preprocess, "device": device, "categories": categories}


def init_zero_shot_detector(cfg: PipelineConfig, device: str = "cpu") -> Dict[str, Any]:
    """Initialize optional Grounding DINO detector state."""
    if not cfg.use_zero_shot_detector:
        return {"available": False, "reason": "disabled by config"}
    if not module_available("transformers") or not module_available("torch"):
        return {"available": False, "reason": "transformers/torch not installed"}
    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        processor = AutoProcessor.from_pretrained(cfg.zero_shot_model_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.zero_shot_model_name)
        model.to(device)
        model.eval()
    except Exception as exc:
        print(f"[WARN] Zero-shot detector unavailable ({cfg.zero_shot_model_name}): {exc}. Falling back to Mask R-CNN only.")
        return {"available": False, "reason": str(exc)}
    return {"available": True, "processor": processor, "model": model, "device": device}


def infer_segmentation(image, seg_state: Dict[str, Any]):
    """Run semantic segmentation and derive ground/object masks.

    Returns fallback empty masks when segmentation is unavailable.
    """
    import numpy as np

    if not seg_state.get("available", False):
        h, w = image.size[1], image.size[0]
        return {
            "label_map": np.zeros((h, w), dtype=np.int32),
            "id2label": {0: "unknown"},
            "ground_mask": np.zeros((h, w), dtype=bool),
            "object_mask": np.zeros((h, w), dtype=bool),
            "clean_ground_mask": np.zeros((h, w), dtype=bool),
            "clean_object_mask": np.zeros((h, w), dtype=bool),
            "warning": seg_state.get("reason", "segmentation unavailable"),
        }

    import torch
    import torch.nn.functional as F

    processor = seg_state["processor"]
    model = seg_state["model"]
    id2label = seg_state["id2label"]
    device = seg_state.get("device", "cpu")

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        logits = F.interpolate(logits, size=(image.size[1], image.size[0]), mode="bilinear", align_corners=False)
        label_map = logits.argmax(dim=1)[0].cpu().numpy()

    label_lut = {int(k): v.lower() for k, v in id2label.items()}
    ground_ids = {i for i, n in label_lut.items() if n in GROUND_CLASS_NAMES}
    object_ids = {i for i, n in label_lut.items() if n in MOVABLE_CLASS_NAMES}

    ground_mask = np.isin(label_map, list(ground_ids))
    object_mask = np.isin(label_map, list(object_ids))

    clean_ground, clean_obj = cleanup_masks(ground_mask, object_mask)
    return {
        "label_map": label_map,
        "id2label": id2label,
        "ground_mask": ground_mask,
        "object_mask": object_mask,
        "clean_ground_mask": clean_ground,
        "clean_object_mask": clean_obj,
    }


def infer_depth(image, depth_state: Dict[str, Any]):
    """Run monocular depth and normalize to [0, 1].

    The normalized map is not metric depth. It is later converted with
    ``normalized_depth_to_distance`` using configured min/max depth bounds.
    """
    import numpy as np

    h, w = image.size[1], image.size[0]
    if not depth_state.get("available", False):
        return {"depth": np.ones((h, w), dtype=np.float32) * 0.5, "warning": depth_state.get("reason", "depth unavailable")}

    depth_pipe = depth_state["pipeline"]
    pred = depth_pipe(image)
    depth_im = pred["depth"]
    depth_np = np.array(depth_im).astype("float32")
    if depth_np.shape != (h, w):
        from PIL import Image

        depth_np = np.array(Image.fromarray(depth_np).resize((w, h)))

    mn, mx = float(depth_np.min()), float(depth_np.max())
    if mx - mn < 1e-6:
        depth_norm = np.zeros_like(depth_np, dtype=np.float32)
    else:
        depth_norm = (depth_np - mn) / (mx - mn)

    return {"depth": depth_norm.astype(np.float32)}


def cleanup_masks(ground_mask, object_mask):
    """Apply morphology and small-component cleanup to segmentation masks."""
    import numpy as np

    cv2 = get_cv2()
    if cv2 is not None:

        g = ground_mask.astype("uint8")
        o = object_mask.astype("uint8")
        kernel3 = np.ones((3, 3), np.uint8)
        kernel5 = np.ones((5, 5), np.uint8)

        g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel5)
        g = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel3)
        o = cv2.morphologyEx(o, cv2.MORPH_CLOSE, kernel3)

        g = remove_small_components(g.astype(bool), min_size=200)
        o = remove_small_components(o.astype(bool), min_size=80)
        return g, o

    return ground_mask, object_mask


def remove_small_components(mask, min_size: int = 100):
    """Remove connected components smaller than ``min_size`` pixels."""
    import numpy as np

    cv2 = get_cv2()
    if cv2 is None:
        return mask

    out = np.zeros_like(mask, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"), connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out[labels == i] = True
    return out


def normalize_class_name(class_name: str) -> str:
    """Normalize detector label text into canonical class names.

    Examples:
        ``traffic sign`` / ``stop sign`` / ``road sign`` -> ``road_sign``
        ``trash dumpster`` -> ``dumpster``
    """
    key = str(class_name).strip().lower()
    key = re.sub(r"^[\s\.,\-_]+|[\s\.,\-_]+$", "", key)
    key = re.sub(r"[\.,\-_]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    key = re.sub(r"^(a|an)\s+", "", key)
    return ZERO_SHOT_CLASS_ALIASES.get(key, key.replace(" ", "_"))


def _instance_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute IoU of two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = max(area_a + area_b - inter, 1e-8)
    return inter / denom


def _dedup_image_space_instances(instances: Sequence[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Apply image-space NMS per class, keeping highest-confidence boxes."""
    kept: List[Dict[str, Any]] = []
    for candidate in sorted(instances, key=lambda i: -float(i["confidence"])):
        suppress = False
        for existing in kept:
            if normalize_class_name(candidate["class_name"]) != normalize_class_name(existing["class_name"]):
                continue
            if _instance_iou_xyxy(candidate["bbox"], existing["bbox"]) >= iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(candidate)
    return kept


def _infer_instances_maskrcnn(image, detector_state: Dict[str, Any], threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Infer object instances using Mask R-CNN and normalize class labels."""
    if not detector_state.get("available", False):
        return []

    import torch

    model = detector_state["model"]
    preprocess = detector_state["preprocess"]
    device = detector_state.get("device", "cpu")

    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor)[0]

    instances: List[Dict[str, Any]] = []
    for i in range(len(pred["scores"])):
        score = float(pred["scores"][i].cpu().item())
        label_id = int(pred["labels"][i].cpu().item())
        if score < threshold or label_id not in COCO_ID_TO_NAME:
            continue

        bbox = pred["boxes"][i].detach().cpu().numpy().tolist()
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        contact = ((x1 + x2) / 2.0, y2)
        instances.append(
            {
                "class_name": normalize_class_name(COCO_ID_TO_NAME[label_id]),
                "confidence": score,
                "bbox": bbox,
                "center_xy": center,
                "contact_xy": contact,
                "detector_source": "mask_rcnn",
            }
        )
    return [i for i in instances if i["class_name"] in TARGET_CLASS_NAMES]


def _infer_instances_zero_shot(image, zero_shot_state: Dict[str, Any], cfg: PipelineConfig) -> List[Dict[str, Any]]:
    """Infer open-vocabulary detections with Grounding DINO when available."""
    if not zero_shot_state.get("available", False):
        return []
    import torch

    processor = zero_shot_state["processor"]
    model = zero_shot_state["model"]
    device = zero_shot_state.get("device", "cpu")
    text_prompt = ". ".join(ZERO_SHOT_LABEL_PROMPTS) + "."
    try:
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    except Exception:
        try:
            inputs = processor(images=image, text=[ZERO_SHOT_LABEL_PROMPTS], return_tensors="pt").to(device)
        except Exception as exc:
            print(f"[WARN] Grounding DINO input formatting failed ({exc}); continuing with Mask R-CNN detections only.")
            return []
    with torch.no_grad():
        outputs = model(**inputs)
    post = _post_process_grounding_dino(processor, outputs, inputs.input_ids, cfg, image)
    instances: List[Dict[str, Any]] = []
    labels = post.get("text_labels", post.get("labels", []))
    for score, box, label in zip(post.get("scores", []), post.get("boxes", []), labels):
        cls = normalize_class_name(label if isinstance(label, str) else str(label))
        if cls not in TARGET_CLASS_NAMES:
            continue
        bbox_raw = box.tolist() if hasattr(box, "tolist") else list(box)
        bbox = [float(v) for v in bbox_raw]
        x1, y1, x2, y2 = bbox
        instances.append(
            {
                "class_name": cls,
                "confidence": float(score),
                "bbox": bbox,
                "center_xy": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                "contact_xy": [(x1 + x2) / 2.0, y2],
                "detector_source": "grounding_dino",
            }
        )
    return instances


def _post_process_grounding_dino(processor, outputs, input_ids, cfg: PipelineConfig, image) -> Dict[str, Any]:
    """Compatibility wrapper for Grounding DINO post-process API variants."""
    target_sizes = [(image.height, image.width)]
    empty_result: Dict[str, Any] = {"scores": [], "boxes": [], "labels": []}
    fn = processor.post_process_grounded_object_detection

    try:
        params = inspect.signature(fn).parameters
        supports_input_ids = "input_ids" in params
        if "box_threshold" in params:
            results = fn(
                outputs,
                input_ids,
                box_threshold=cfg.zero_shot_box_threshold,
                text_threshold=cfg.zero_shot_text_threshold,
                target_sizes=target_sizes,
            )
            return results[0] if results else empty_result
        if "threshold" in params:
            kwargs: Dict[str, Any] = {
                "threshold": cfg.zero_shot_box_threshold,
                "text_threshold": cfg.zero_shot_text_threshold,
                "target_sizes": target_sizes,
            }
            if supports_input_ids:
                kwargs["input_ids"] = input_ids
            results = fn(outputs, **kwargs)
            return results[0] if results else empty_result
    except Exception:
        pass

    try:
        results = fn(
            outputs,
            input_ids,
            box_threshold=cfg.zero_shot_box_threshold,
            text_threshold=cfg.zero_shot_text_threshold,
            target_sizes=target_sizes,
        )
        return results[0] if results else empty_result
    except TypeError as exc:
        msg = str(exc).lower()
        if "box_threshold" not in msg and "unexpected keyword" not in msg:
            print(
                "[WARN] Grounding DINO post-processing failed; continuing with Mask R-CNN detections only."
            )
            return empty_result
    except Exception:
        print(
            "[WARN] Grounding DINO post-processing failed; continuing with Mask R-CNN detections only."
        )
        return empty_result

    try:
        results = fn(
            outputs,
            input_ids=input_ids,
            threshold=cfg.zero_shot_box_threshold,
            text_threshold=cfg.zero_shot_text_threshold,
            target_sizes=target_sizes,
        )
        return results[0] if results else empty_result
    except TypeError:
        try:
            results = fn(
                outputs,
                threshold=cfg.zero_shot_box_threshold,
                text_threshold=cfg.zero_shot_text_threshold,
                target_sizes=target_sizes,
            )
            return results[0] if results else empty_result
        except Exception:
            print(
                "[WARN] Grounding DINO post-processing failed; continuing with Mask R-CNN detections only."
            )
            return empty_result
    except Exception:
        print(
            "[WARN] Grounding DINO post-processing failed; continuing with Mask R-CNN detections only."
        )
        return empty_result


def infer_instances(
    image,
    detector_state: Dict[str, Any],
    zero_shot_state: Optional[Dict[str, Any]] = None,
    cfg: Optional[PipelineConfig] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run available detectors and return deduplicated, labeled instances."""
    if cfg is None:
        raise ValueError("infer_instances requires cfg")
    if not detector_state.get("available", False) and not (zero_shot_state or {}).get("available", False):
        return {"instances": [], "warning": detector_state.get("reason", "detector unavailable")}
    instances = _infer_instances_maskrcnn(image, detector_state, threshold=threshold)
    instances.extend(_infer_instances_zero_shot(image, zero_shot_state or {}, cfg))
    instances = _dedup_image_space_instances(instances, iou_threshold=cfg.image_nms_iou_threshold)

    instances = assign_instance_labels(instances)
    return {"instances": instances}


def assign_instance_labels(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign per-image readable labels like ``car1``, ``person2``."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for inst in instances:
        grouped.setdefault(inst["class_name"], []).append(inst)

    output: List[Dict[str, Any]] = []
    for cls, group in grouped.items():
        group_sorted = sorted(group, key=lambda g: (g["contact_xy"][0], -g["contact_xy"][1]))
        for i, inst in enumerate(group_sorted, start=1):
            inst["instance_label"] = f"{cls}{i}"
            output.append(inst)
    return output


def assert_unique_labels(instances: Sequence[Dict[str, Any]]) -> None:
    """Validate that ``instance_label`` values are unique."""
    labels = [i["instance_label"] for i in instances]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate instance labels found")


def get_intrinsics(width: int, height: int, hfov_deg: float) -> Dict[str, float]:
    """Approximate pinhole intrinsics from width and horizontal FOV."""
    fx = width / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
    fy = fx
    return {"fx": fx, "fy": fy, "cx": width / 2.0, "cy": height / 2.0}


def normalized_depth_to_distance(depth_norm, depth_min: float, depth_max: float, inverse: bool = True):
    """Map normalized depth to relative range units.

    Args:
        depth_norm: Normalized depth in [0, 1].
        depth_min: Near bound in relative units.
        depth_max: Far bound in relative units.
        inverse: When ``True`` treat larger normalized values as nearer
            (DPT-style inverse-depth assumption).
    """
    import numpy as np

    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    distance_norm = 1.0 - depth_norm if inverse else depth_norm
    return depth_min + (depth_max - depth_min) * distance_norm


def build_bev(seg_result: Dict[str, Any], depth_result: Dict[str, Any], cfg: PipelineConfig):
    """Build per-image diagnostic BEV pixels from segmentation + depth.

    This output is pixel-level diagnostic content; it is not the final
    object-level minimap table.
    """
    import numpy as np

    depth = depth_result["depth"]
    ground = seg_result["clean_ground_mask"]
    obstacles = seg_result["clean_object_mask"]
    if depth.shape != ground.shape:
        raise ValueError(f"Depth and segmentation shape mismatch: {depth.shape} vs {ground.shape}")

    h, w = depth.shape
    intr = get_intrinsics(w, h, cfg.horizontal_fov_deg)
    dist = normalized_depth_to_distance(depth, cfg.min_relative_range, cfg.max_relative_range, inverse=cfg.depth_is_inverse)

    bev = np.zeros((cfg.bev_height_px, cfg.bev_width_px, 3), dtype=np.uint8)
    scale = bev_scale_px_per_range_unit(cfg)
    bottom_margin_px = 40
    ego_x = cfg.bev_width_px // 2
    ego_y = cfg.bev_height_px - bottom_margin_px

    ys, xs = np.where(ground | obstacles)
    for y, x in zip(ys[::2], xs[::2]):
        z_forward = float(dist[y, x])
        if z_forward <= 0.01 or z_forward > cfg.bev_max_range:
            continue
        x_cam = (x - intr["cx"]) / intr["fx"] * z_forward

        bev_x = int(round(ego_x + x_cam * scale))
        bev_y = int(round(ego_y - z_forward * scale))
        if 0 <= bev_x < cfg.bev_width_px and 0 <= bev_y < cfg.bev_height_px:
            bev[bev_y, bev_x] = (60, 170, 60) if ground[y, x] else (200, 90, 60)

    draw_bev_guides(bev, ego_x, ego_y, cfg)
    return {"bev": bev, "ego_xy": (ego_x, ego_y), "distance_units": dist}


def draw_bev_guides(bev, ego_x: int, ego_y: int, cfg: PipelineConfig) -> None:
    """Draw ego marker and range rings on a per-image BEV diagnostic."""
    cv2 = get_cv2()
    if cv2 is None:
        return

    scale = bev_scale_px_per_range_unit(cfg)
    cv2.circle(bev, (ego_x, ego_y), 8, (255, 255, 255), -1)
    cv2.arrowedLine(bev, (ego_x, ego_y), (ego_x, max(5, ego_y - 70)), (255, 255, 0), 2, tipLength=0.2)

    for rel in [2, 4, 6, 8, 10]:
        r = int(rel * scale)
        y = ego_y - r
        if y > 0:
            cv2.ellipse(bev, (ego_x, ego_y), (r, r), 0, 180, 360, (80, 80, 80), 1)
            cv2.putText(bev, f"{rel} rel", (ego_x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)


def add_instance_markers(bev_result: Dict[str, Any], instances: List[Dict[str, Any]], depth_m, cfg: PipelineConfig):
    """Overlay detected instance markers on a BEV diagnostic image."""
    bev = bev_result["bev"]
    ego_x, ego_y = bev_result["ego_xy"]
    scale = bev_scale_px_per_range_unit(cfg)
    intr = get_intrinsics(depth_m.shape[1], depth_m.shape[0], cfg.horizontal_fov_deg)

    cv2 = get_cv2()
    if cv2 is None:
        return bev, []

    records = []
    for inst in instances:
        cx, cy = inst["contact_xy"]
        ix = int(max(0, min(depth_m.shape[1] - 1, round(cx))))
        iy = int(max(0, min(depth_m.shape[0] - 1, round(cy))))
        z_forward = float(depth_m[iy, ix])
        x_cam = (cx - intr["cx"]) / intr["fx"] * z_forward
        bev_x = int(round(ego_x + x_cam * scale))
        bev_y = int(round(ego_y - z_forward * scale))

        inst["estimated_relative_range"] = z_forward
        inst["bev_xy"] = (bev_x, bev_y)

        if 0 <= bev_x < cfg.bev_width_px and 0 <= bev_y < cfg.bev_height_px:
            cv2.circle(bev, (bev_x, bev_y), 5, (235, 50, 50), -1)
            cv2.putText(bev, inst["instance_label"], (bev_x + 6, bev_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        records.append(
            {
                "instance_label": inst["instance_label"],
                "class_name": inst["class_name"],
                "detector_source": inst.get("detector_source", "unknown"),
                "confidence": round(float(inst["confidence"]), 4),
                "bbox": [round(v, 2) for v in inst["bbox"]],
                "center_xy": [round(v, 2) for v in inst["center_xy"]],
                "contact_xy": [round(v, 2) for v in inst["contact_xy"]],
                "estimated_relative_range": round(float(z_forward), 3),
                "bev_xy": [int(bev_x), int(bev_y)],
            }
        )

    return bev, records


def draw_detection_overlay(image, instances: Sequence[Dict[str, Any]]):
    """Render detection boxes and labels on the input RGB image."""
    import numpy as np

    arr = np.array(image).copy()
    cv2 = get_cv2()
    if cv2 is None:
        return arr

    for inst in instances:
        x1, y1, x2, y2 = [int(v) for v in inst["bbox"]]
        cv2.rectangle(arr, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(arr, f"{inst['instance_label']} {inst['confidence']:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return arr


def extract_direction_index(path_or_name: str) -> Optional[int]:
    """Extract integer direction index from filenames like ``direction 3``."""
    text = Path(path_or_name).stem.lower()
    m = re.search(r"direction[\s_\-]*(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"\bdir[\s_\-]*(\d+)\b", text)
    if m:
        return int(m.group(1))
    return None


def heading_for_direction(direction_index: int, cfg: PipelineConfig) -> float:
    """Map direction index to heading in clockwise degrees from +y."""
    sign = 1.0 if cfg.direction_turn.lower() == "right" else -1.0
    return cfg.direction_zero_heading_deg + sign * direction_index * cfg.direction_step_deg


def estimate_camera_relative_position(instance: Dict[str, Any], image_width: int, cfg: PipelineConfig) -> Tuple[float, float]:
    """Estimate camera-relative (x right, y forward) coordinates for an object.

    Uses the bounding-box bottom-center x-position and estimated depth.
    """
    contact_x = float(instance["contact_xy"][0])
    depth_m = float(instance["estimated_relative_range"])
    normalized_x = (contact_x - image_width / 2.0) / (image_width / 2.0)
    angle_offset_deg = normalized_x * (cfg.horizontal_fov_deg / 2.0)
    lateral_x_units = depth_m * math.tan(math.radians(angle_offset_deg))
    forward_units = depth_m
    return lateral_x_units, forward_units


def direction_vector_from_heading(heading_deg: float) -> Tuple[float, float]:
    """Return (x, y) unit vector for heading clockwise from +y (north)."""
    theta = math.radians(heading_deg)
    return math.sin(theta), math.cos(theta)


def rotate_clockwise_from_camera_to_ego(lateral_x_units: float, forward_units: float, heading_deg: float) -> Tuple[float, float]:
    """Rotate camera-local (x right, y forward) into ego/world (x right, y forward)."""
    theta = math.radians(heading_deg)
    ego_x = lateral_x_units * math.cos(theta) + forward_units * math.sin(theta)
    ego_y = -lateral_x_units * math.sin(theta) + forward_units * math.cos(theta)
    return ego_x, ego_y


def rotate_camera_relative_to_ego(lateral_x_units: float, forward_units: float, heading_deg: float, cfg: PipelineConfig) -> Tuple[float, float]:
    """Rotate camera-relative coordinates into ego/world relative units."""
    _ = cfg
    return rotate_clockwise_from_camera_to_ego(lateral_x_units, forward_units, heading_deg)


def bev_scale_px_per_range_unit(cfg: PipelineConfig) -> float:
    """Return BEV pixel-per-relative-unit scale."""
    if cfg.bev_scale_px_per_range_unit is not None:
        return cfg.bev_scale_px_per_range_unit

    top_margin_px = 60
    bottom_margin_px = 40
    side_margin_px = 60
    ego_y = cfg.bev_height_px - bottom_margin_px

    vertical_scale = (ego_y - top_margin_px) / max(cfg.bev_max_range, 1e-6)
    half_fov_rad = math.radians(cfg.horizontal_fov_deg / 2.0)
    lateral_extent = cfg.bev_max_range * math.tan(half_fov_rad)
    horizontal_scale = ((cfg.bev_width_px / 2.0) - side_margin_px) / max(lateral_extent, 1e-6)
    return max(1.0, min(vertical_scale, horizontal_scale))


def minimap_scale_px_per_range_unit(cfg: PipelineConfig) -> float:
    """Return minimap pixel-per-relative-unit scale."""
    if cfg.minimap_scale_px_per_range_unit is not None:
        return cfg.minimap_scale_px_per_range_unit
    return (cfg.minimap_size_px * 0.48) / cfg.minimap_max_range


def ego_meters_to_minimap_px(x_units: float, y_units: float, cfg: PipelineConfig) -> Tuple[int, int]:
    """Convert ego-frame relative units to minimap pixel coordinates."""
    # Ego/world: +x right, +y forward. Image y increases downward.
    scale = minimap_scale_px_per_range_unit(cfg)
    c = cfg.minimap_size_px // 2
    px = int(round(c + x_units * scale))
    py = int(round(c - y_units * scale))
    return px, py


def draw_minimap_guides(canvas, cfg: PipelineConfig) -> None:
    """Draw ego marker, rings, and direction spokes on the minimap canvas."""
    cv2 = get_cv2()
    if cv2 is None:
        return

    c = cfg.minimap_size_px // 2
    scale = minimap_scale_px_per_range_unit(cfg)
    ring_units = [2, 4, 6, 8, 10]

    cv2.circle(canvas, (c, c), 9, (255, 255, 255), -1)
    cv2.arrowedLine(canvas, (c, c), (c, c - 70), (0, 255, 255), 3, tipLength=0.25)
    cv2.putText(canvas, "dir0", (c + 8, c - 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)

    for rel in ring_units:
        r = int(rel * scale)
        cv2.circle(canvas, (c, c), r, (75, 75, 75), 1)
        cv2.putText(canvas, f"{rel} rel", (c + 6, c - r - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    for i in range(8):
        heading = heading_for_direction(i, cfg)
        theta = math.radians(heading)
        x = int(round(c + math.sin(theta) * (cfg.minimap_size_px * 0.45)))
        y = int(round(c - math.cos(theta) * (cfg.minimap_size_px * 0.45)))
        cv2.line(canvas, (c, c), (x, y), (45, 45, 45), 1)
        cv2.putText(canvas, f"dir{i}", (x - 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)


def draw_stitched_bev_guides(canvas, center_px: int, scale_px_per_range_unit: float, max_distance_units: float, cfg: PipelineConfig) -> None:
    """Draw rings/spokes on stitched BEV diagnostic canvas."""
    cv2 = get_cv2()
    if cv2 is None:
        return

    cv2.circle(canvas, (center_px, center_px), 6, (255, 255, 255), -1)
    cv2.arrowedLine(canvas, (center_px, center_px), (center_px, int(round(center_px - 0.22 * canvas.shape[0]))), (0, 190, 190), 1, tipLength=0.2)
    cv2.putText(canvas, "dir0", (center_px + 6, int(round(center_px - 0.24 * canvas.shape[0]))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

    for rel in [2, 4, 6, 8, 10]:
        if rel > max_distance_units:
            continue
        radius = int(round(rel * scale_px_per_range_unit))
        cv2.circle(canvas, (center_px, center_px), radius, (58, 58, 58), 1)

    spoke_radius = int(round(max_distance_units * scale_px_per_range_unit))
    for i in range(8):
        theta = math.radians(heading_for_direction(i, cfg))
        spoke_x = int(round(center_px + math.sin(theta) * spoke_radius))
        spoke_y = int(round(center_px - math.cos(theta) * spoke_radius))
        cv2.line(canvas, (center_px, center_px), (spoke_x, spoke_y), (45, 45, 45), 1)
        cv2.putText(canvas, f"dir{i}", (spoke_x - 14, spoke_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (165, 165, 165), 1)


def render_direction_debug_plot(cfg: PipelineConfig, output_path: Path) -> Path:
    """Render a direction-convention guide image and save it."""
    import numpy as np

    canvas = np.zeros((cfg.minimap_size_px, cfg.minimap_size_px, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)
    draw_minimap_guides(canvas, cfg)
    save_array_image(output_path, canvas)
    return output_path


def _bearing_deg(x_units: float, y_units: float) -> float:
    """Return clockwise bearing degrees from +y for ego-frame coordinates."""
    return (math.degrees(math.atan2(x_units, y_units)) + 360.0) % 360.0


def deduplicate_location_rows(rows: List[Dict[str, Any]], cfg: PipelineConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Heuristically deduplicate projected objects in ego-frame relative-unit space.

    This is location-level class-aware NMS using distance radii. It reduces
    repeated detections across overlapping directions but does not prove two
    rows are the same physical object.
    """
    if not cfg.minimap_merge_nearby_same_class:
        return rows, []

    radius_by_class = dict(DEDUP_RADIUS_BY_CLASS_DEFAULT)
    if cfg.minimap_merge_radius_by_class:
        radius_by_class.update(cfg.minimap_merge_radius_by_class)

    kept: List[Dict[str, Any]] = []
    suppressed_records: List[Dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: -float(r["confidence"])):
        cls = normalize_class_name(row["class_name"])
        threshold_units = float(radius_by_class.get(cls, cfg.minimap_merge_radius_units))
        suppressed = False
        for existing in kept:
            if normalize_class_name(existing["class_name"]) != cls:
                continue
            distance_units = float(math.hypot(row["ego_x_units"] - existing["ego_x_units"], row["ego_y_units"] - existing["ego_y_units"]))
            if distance_units <= threshold_units:
                suppressed = True
                suppressed_records.append(
                    {
                        "class_name": cls,
                        "distance_units": round(distance_units, 4),
                        "threshold_units": round(threshold_units, 4),
                        "kept_label": existing.get("instance_label"),
                        "kept_source": existing.get("detector_source", "unknown"),
                        "kept_confidence": float(existing.get("confidence", 0.0)),
                        "suppressed_label": row.get("instance_label"),
                        "suppressed_source": row.get("detector_source", "unknown"),
                        "suppressed_confidence": float(row.get("confidence", 0.0)),
                    }
                )
                break
        if not suppressed:
            kept.append(row)
    return kept, suppressed_records


def relabel_instances_per_class(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Relabel deduplicated rows per class for stable readable labels."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(normalize_class_name(row["class_name"]), []).append(row)
    updated: List[Dict[str, Any]] = []
    for cls in sorted(grouped.keys()):
        group = sorted(grouped[cls], key=lambda r: (r["bearing_deg"], r["source_image"], r["contact_xy"][0]))
        for idx, row in enumerate(group, start=1):
            row["class_name"] = cls
            row["instance_label"] = f"{cls}{idx}"
            updated.append(row)
    return updated


def render_location_minimap(location_name: str, rows: List[Dict[str, Any]], cfg: PipelineConfig, output_path: Path) -> Path:
    """Render and save final object-level minimap for one location.

    The minimap is produced from projected detection rows (not BEV pixels) and
    uses the deduplicated location table.
    """
    import numpy as np

    cv2 = get_cv2()
    if cv2 is None:
        canvas = np.zeros((cfg.minimap_size_px, cfg.minimap_size_px, 3), dtype=np.uint8)
        save_array_image(output_path, canvas)
        return output_path

    canvas = np.zeros((cfg.minimap_size_px, cfg.minimap_size_px, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)
    draw_minimap_guides(canvas, cfg)

    rows_for_labels = sorted(rows, key=lambda r: (-r["confidence"], r["range_units"]))[: cfg.minimap_label_top_k_per_location]
    label_keys = {(r["source_image"], r["instance_label"]) for r in rows_for_labels}

    occupancy: Dict[Tuple[int, int], int] = {}
    for row in rows:
        x, y = row["minimap_xy"]
        if cfg.minimap_jitter_overlapping_markers:
            key = (x, y)
            n = occupancy.get(key, 0)
            if n > 0:
                ang = n * 2.399963
                jitter_px = max(8, int(round(cfg.minimap_size_px * 0.007)))
                x += int(round(jitter_px * math.cos(ang)))
                y += int(round(jitter_px * math.sin(ang)))
            occupancy[key] = n + 1

        color = CLASS_COLORS.get(row["class_name"], (220, 220, 220))
        alpha = float(max(0.0, min(1.0, cfg.minimap_marker_alpha)))
        overlay = canvas.copy()
        cv2.circle(overlay, (x, y), 6, color, -1)
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0, dst=canvas)

        if cfg.minimap_draw_object_labels and (row["source_image"], row["instance_label"]) in label_keys:
            text = row["instance_label"]
            tx, ty = x + 10, y - 8
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            cv2.rectangle(canvas, (tx - 3, ty - th - 3), (tx + tw + 3, ty + baseline + 2), (20, 20, 20), -1)
            cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2)

    cv2.putText(canvas, f"{location_name} minimap (ego-centered)", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "distance scale: relative DPT units", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)

    legend_items = ["person", "car", "bus", "motorcycle", "bicycle", "dumpster", "road_sign"]
    lx, ly = 20, cfg.minimap_size_px - 220
    cv2.rectangle(canvas, (lx - 10, ly - 30), (lx + 280, ly + 180), (35, 35, 35), -1)
    cv2.putText(canvas, "Legend", (lx, ly - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1)
    for i, name in enumerate(legend_items):
        y = ly + i * 23
        cv2.circle(canvas, (lx + 12, y + 6), 6, CLASS_COLORS.get(name, (200, 200, 200)), -1)
        cv2.putText(canvas, name, (lx + 26, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    save_array_image(output_path, canvas)
    return output_path


def _serialize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an internal row to a JSON/CSV-friendly serializable row."""
    return {
        "source_image": row["source_image"],
        "location": row["location"],
        "direction_index": row["direction_index"],
        "heading_deg": round(row["heading_deg"], 3),
        "class_name": row["class_name"],
        "detector_source": row.get("detector_source", "unknown"),
        "instance_label": row["instance_label"],
        "confidence": round(row["confidence"], 4),
        "bbox": [round(v, 2) for v in row["bbox"]],
        "center_xy": [round(v, 2) for v in row["center_xy"]],
        "contact_xy": [round(v, 2) for v in row["contact_xy"]],
        "estimated_relative_range": round(row["estimated_relative_range"], 3),
        "camera_lateral_x_units": round(row["camera_lateral_x_units"], 3),
        "camera_forward_units": round(row["camera_forward_units"], 3),
        "ego_x_units": round(row["ego_x_units"], 3),
        "ego_y_units": round(row["ego_y_units"], 3),
        "bearing_deg": round(row["bearing_deg"], 3),
        "range_units": round(row["range_units"], 3),
        "minimap_xy": [int(row["minimap_xy"][0]), int(row["minimap_xy"][1])],
        "clipped_to_minimap": bool(row["clipped_to_minimap"]),
    }


def save_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write rows to CSV, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compose_location_bev(bev_images: Sequence, labels: Sequence[str], cfg: PipelineConfig):
    """Compose stitched pixel-level BEV diagnostic by fixed-center rotation.

    Each per-image BEV raster is rotated into a shared ego frame according to
    the clockwise direction convention. Black/background pixels do not erase
    previously accumulated content.
    """
    import numpy as np

    if len(bev_images) == 0:
        raise ValueError("No BEV images provided for stitching")

    output_scale = bev_scale_px_per_range_unit(cfg)
    radius_px = int(math.ceil(cfg.bev_max_range * output_scale))
    canvas_size = max(bev_images[0].shape[0], bev_images[0].shape[1], 2 * radius_px + 1)
    center = canvas_size // 2
    accum = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    counts = np.zeros((canvas_size, canvas_size), dtype=np.float32)

    diagnostics = []
    for idx, bev in enumerate(bev_images):
        bev_u8 = bev.astype(np.uint8)
        h, w = bev_u8.shape[:2]
        src_ego_x = w // 2
        bottom_margin_px = 40
        src_ego_y = h - bottom_margin_px

        ys, xs = np.where(np.any(bev_u8 > 20, axis=2))
        if ys.size == 0:
            diagnostics.append({"view": labels[idx], "valid_pixels": 0, "status": "no_valid_pixels"})
            continue

        local_x_units = (xs.astype(np.float32) - src_ego_x) / output_scale
        local_y_units = (src_ego_y - ys.astype(np.float32)) / output_scale

        direction_idx = extract_direction_index(labels[idx])
        if direction_idx is None:
            direction_idx = idx
        heading_deg = heading_for_direction(direction_idx, cfg)
        world_x_units, world_y_units = rotate_clockwise_from_camera_to_ego(local_x_units, local_y_units, heading_deg)

        out_x = np.rint(center + world_x_units * output_scale).astype(np.int32)
        out_y = np.rint(center - world_y_units * output_scale).astype(np.int32)
        in_bounds = (out_x >= 0) & (out_x < canvas_size) & (out_y >= 0) & (out_y < canvas_size)
        if not np.any(in_bounds):
            diagnostics.append({"view": labels[idx], "valid_pixels": int(ys.size), "painted_pixels": 0, "heading_deg": round(heading_deg, 1)})
            continue

        out_x = out_x[in_bounds]
        out_y = out_y[in_bounds]
        src_pixels = bev_u8[ys[in_bounds], xs[in_bounds]].astype(np.float32)

        if not cfg.stitched_draw_dense_points:
            src_mask = np.any(src_pixels > 20, axis=1)
            out_x = out_x[src_mask]
            out_y = out_y[src_mask]
            src_pixels = src_pixels[src_mask]
        if src_pixels.size == 0:
            diagnostics.append({"view": labels[idx], "valid_pixels": int(ys.size), "painted_pixels": 0, "heading_deg": round(heading_deg, 1)})
            continue

        np.add.at(accum, (out_y, out_x, slice(None)), src_pixels * float(cfg.stitched_dense_alpha))
        np.add.at(counts, (out_y, out_x), float(cfg.stitched_dense_alpha))
        diagnostics.append({
            "view": labels[idx],
            "heading_deg": round(heading_deg, 1),
            "valid_pixels": int(ys.size),
            "painted_pixels": int(np.count_nonzero(in_bounds)),
            "status": "fixed_center_rotation",
        })

    stitched = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    painted = counts > 0
    stitched[painted] = np.clip(accum[painted] / counts[painted, None], 0, 255).astype(np.uint8)
    draw_stitched_bev_guides(stitched, center, output_scale, cfg.bev_max_range, cfg)

    return stitched, diagnostics


def compute_alignment_diagnostics(images: Sequence, labels: Sequence[str]) -> List[Dict[str, Any]]:
    """Compute optional ORB feature-match diagnostics between adjacent views."""
    cv2 = get_cv2()
    if cv2 is None:
        return [{"status": "skipped_cv2_missing", "pair": "n/a", "matches": 0, "inliers": 0}]
    import numpy as np

    out: List[Dict[str, Any]] = []
    orb = cv2.ORB_create(1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i in range(len(images) - 1):
        a, b = images[i], images[i + 1]
        a_u8 = np.asarray(a).astype(np.uint8)
        b_u8 = np.asarray(b).astype(np.uint8)
        if a_u8.ndim == 3:
            g1 = cv2.cvtColor(a_u8, cv2.COLOR_RGB2GRAY)
        else:
            g1 = a_u8
        if b_u8.ndim == 3:
            g2 = cv2.cvtColor(b_u8, cv2.COLOR_RGB2GRAY)
        else:
            g2 = b_u8
        k1, d1 = orb.detectAndCompute(g1, None)
        k2, d2 = orb.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
            out.append({"pair": f"{labels[i]}->{labels[i+1]}", "matches": 0, "inliers": 0, "status": "fallback_fixed_angle", "diagnostic_source": "rgb_original"})
            continue

        matches = sorted(matcher.match(d1, d2), key=lambda m: m.distance)
        pts1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        out.append(
            {
                "pair": f"{labels[i]}->{labels[i+1]}",
                "matches": len(matches),
                "inliers": inliers,
                "status": "diagnostic_only" if H is not None else "fallback_fixed_angle",
                "diagnostic_source": "rgb_original",
            }
        )
    return out


def save_array_image(path: Path, arr) -> None:
    """Save a numpy image array to disk."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def save_json(path: Path, obj: Any) -> None:
    """Write JSON to disk with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def process_image(image_path: Path, cfg: PipelineConfig, model_states: Dict[str, Any], output_dirs: Dict[str, Path]):
    """Process one source image and write per-image outputs.

    Side effects:
        Saves detection overlays, per-image BEV diagnostics, segmentation
        diagnostics, and per-image instance JSON.
    """
    image = load_rgb_image(image_path)
    seg = infer_segmentation(image, model_states["seg"])
    depth = infer_depth(image, model_states["depth"])
    det = infer_instances(
        image,
        model_states["det"],
        zero_shot_state=model_states.get("det_zero_shot"),
        cfg=cfg,
        threshold=cfg.detection_threshold,
    )

    instances = det["instances"]
    assert_unique_labels(instances)

    bev_res = build_bev(seg, depth, cfg)
    bev_img, _ = add_instance_markers(bev_res, instances, bev_res["distance_units"], cfg)
    overlay = draw_detection_overlay(image, instances)

    stem = image_path.stem.replace(" ", "_")
    loc = image_path.parent.name
    bev_path = output_dirs["per_image"] / f"{loc}_{stem}_bev.png"
    det_path = output_dirs["per_image"] / f"{loc}_{stem}_det.png"
    save_array_image(bev_path, bev_img)
    save_array_image(det_path, overlay)

    import numpy as np

    raw_seg_vis = ((seg["label_map"] % 20) * 12).astype(np.uint8)
    raw_seg_vis = np.stack([raw_seg_vis, raw_seg_vis, raw_seg_vis], axis=-1)
    ground_vis = np.stack([seg["clean_ground_mask"] * 255] * 3, axis=-1).astype(np.uint8)
    obj_vis = np.stack([seg["clean_object_mask"] * 255] * 3, axis=-1).astype(np.uint8)
    save_array_image(output_dirs["diagnostics"] / f"{loc}_{stem}_raw_seg.png", raw_seg_vis)
    save_array_image(output_dirs["diagnostics"] / f"{loc}_{stem}_ground_mask.png", ground_vis)
    save_array_image(output_dirs["diagnostics"] / f"{loc}_{stem}_object_mask.png", obj_vis)

    h, w = image.size[1], image.size[0]
    distance_units = normalized_depth_to_distance(depth["depth"], cfg.min_relative_range, cfg.max_relative_range, inverse=cfg.depth_is_inverse)
    direction_index = extract_direction_index(image_path.name)
    if direction_index is None:
        direction_index = 0
    heading_deg = heading_for_direction(direction_index, cfg)

    rows: List[Dict[str, Any]] = []
    for inst in instances:
        if inst["confidence"] < cfg.minimap_min_confidence:
            continue

        cx, cy = inst["contact_xy"]
        ix = int(max(0, min(w - 1, round(cx))))
        iy = int(max(0, min(h - 1, round(cy))))
        depth_m = float(distance_units[iy, ix])
        depth_m = max(cfg.min_relative_range, min(cfg.max_relative_range, depth_m))
        inst["estimated_relative_range"] = depth_m

        cam_x_units, cam_y_units = estimate_camera_relative_position(inst, w, cfg)
        ego_x_units, ego_y_units = rotate_camera_relative_to_ego(cam_x_units, cam_y_units, heading_deg, cfg)
        range_units = math.hypot(ego_x_units, ego_y_units)
        bearing_deg = _bearing_deg(ego_x_units, ego_y_units)

        px, py = ego_meters_to_minimap_px(ego_x_units, ego_y_units, cfg)
        clipped = False
        if range_units > cfg.minimap_max_range:
            clipped = True
            s = cfg.minimap_max_range / max(range_units, 1e-6)
            ego_x_units *= s
            ego_y_units *= s
            px, py = ego_meters_to_minimap_px(ego_x_units, ego_y_units, cfg)

        row = {
            "source_image": image_path.name,
            "location": loc,
            "direction_index": int(direction_index),
            "heading_deg": float(heading_deg),
            "class_name": normalize_class_name(inst["class_name"]),
            "detector_source": inst.get("detector_source", "unknown"),
            "instance_label": inst["instance_label"],
            "confidence": float(inst["confidence"]),
            "bbox": list(inst["bbox"]),
            "center_xy": list(inst["center_xy"]),
            "contact_xy": list(inst["contact_xy"]),
            "estimated_relative_range": depth_m,
            "camera_lateral_x_units": cam_x_units,
            "camera_forward_units": cam_y_units,
            "ego_x_units": ego_x_units,
            "ego_y_units": ego_y_units,
            "bearing_deg": bearing_deg,
            "range_units": range_units,
            "minimap_xy": [int(px), int(py)],
            "clipped_to_minimap": clipped,
        }
        rows.append(row)

    image_records = [_serialize_row(r) for r in rows]
    save_json(output_dirs["tables"] / f"{loc}_{stem}_instances.json", image_records)

    return {
        "image_path": str(image_path),
        "bev_path": str(bev_path),
        "det_path": str(det_path),
        "instance_count": len(image_records),
        "direction_index": direction_index,
        "heading_deg": heading_deg,
        "instances": image_records,
    }


def process_location(location_name: str, image_paths: Sequence[Path], cfg: PipelineConfig, model_states: Dict[str, Any], output_dirs: Dict[str, Path]):
    """Process one location and write final minimap/stitch/table artifacts."""
    image_results = []
    bev_stack = []
    rgb_stack = []
    labels = []
    loc_rows: List[Dict[str, Any]] = []

    for path in image_paths:
        result = process_image(path, cfg, model_states, output_dirs)
        image_results.append(result)
        loc_rows.extend(result["instances"])

        from PIL import Image
        import numpy as np

        bev_stack.append(np.array(Image.open(result["bev_path"]).convert("RGB")))
        rgb_stack.append(np.array(Image.open(path).convert("RGB")))
        labels.append(path.stem)

    dedup_rows, dedup_diag = deduplicate_location_rows(loc_rows, cfg)
    dedup_rows = relabel_instances_per_class(dedup_rows)

    minimap_path = output_dirs["per_location"] / f"{location_name}_minimap.png"
    render_location_minimap(location_name, dedup_rows, cfg, minimap_path)

    direction_debug_path = output_dirs["diagnostics"] / f"{location_name}_direction_debug.png"
    render_direction_debug_plot(cfg, direction_debug_path)

    stitched, diagnostics = compose_location_bev(bev_stack, labels, cfg)
    diagnostics.extend(compute_alignment_diagnostics(rgb_stack, labels))
    stitched_path = output_dirs["per_location"] / f"{location_name}_stitched_bev.png"
    save_array_image(stitched_path, stitched)
    save_json(output_dirs["tables"] / f"{location_name}_stitch_diagnostics.json", diagnostics)
    save_json(output_dirs["tables"] / f"{location_name}_dedup_diagnostics.json", dedup_diag)

    loc_json_rows = [_serialize_row(r) for r in dedup_rows]
    loc_json_path = output_dirs["tables"] / f"{location_name}_minimap_instances.json"
    loc_csv_path = output_dirs["tables"] / f"{location_name}_minimap_instances.csv"
    save_json(loc_json_path, loc_json_rows)
    save_csv(loc_csv_path, [
        {
            **r,
            "bbox": json.dumps(r["bbox"]),
            "center_xy": json.dumps(r["center_xy"]),
            "contact_xy": json.dumps(r["contact_xy"]),
            "minimap_xy": json.dumps(r["minimap_xy"]),
        }
        for r in loc_json_rows
    ])

    return {
        "location": location_name,
        "num_images": len(image_results),
        "images": image_results,
        "minimap": str(minimap_path),
        "direction_debug_plot": str(direction_debug_path),
        "stitched_bev": str(stitched_path),
        "minimap_instance_table_json": str(loc_json_path),
        "minimap_instance_table_csv": str(loc_csv_path),
        "rows": loc_json_rows,
        "total_instances": len(dedup_rows),
        "dedup_suppressed": len(dedup_diag),
    }


def run_pipeline(cfg: PipelineConfig, summary: Dict[str, Any], model_states: Dict[str, Any]) -> Dict[str, Any]:
    """Run full pipeline over selected dataset locations.

    Supports small-demo mode where ``demo_images_per_location=None`` means
    "use all images for selected locations".
    """
    output_dirs = ensure_output_dirs(cfg.output_dir, clean_output_dir=cfg.clean_output_dir)

    location_names = summary["locations"]
    if cfg.run_small_demo:
        location_names = location_names[: cfg.demo_locations]

    results = []
    all_rows: List[Dict[str, Any]] = []
    for loc in location_names:
        image_paths = summary["files"][loc]
        if cfg.run_small_demo:
            if cfg.demo_images_per_location is not None:
                image_paths = image_paths[: cfg.demo_images_per_location]
        res = process_location(loc, image_paths, cfg, model_states, output_dirs)
        results.append(res)
        all_rows.extend(res.get("rows", []))

    combined_csv = cfg.output_dir / "tables" / "all_minimap_instances.csv"
    save_csv(combined_csv, [
        {
            **r,
            "bbox": json.dumps(r["bbox"]),
            "center_xy": json.dumps(r["center_xy"]),
            "contact_xy": json.dumps(r["contact_xy"]),
            "minimap_xy": json.dumps(r["minimap_xy"]),
        }
        for r in all_rows
    ])

    report = {
        "config": {
            "horizontal_fov_deg": cfg.horizontal_fov_deg,
            "min_relative_range": cfg.min_relative_range,
            "max_relative_range": cfg.max_relative_range,
            "minimap_size_px": cfg.minimap_size_px,
            "minimap_max_range": cfg.minimap_max_range,
            "direction_zero_heading_deg": cfg.direction_zero_heading_deg,
            "direction_step_deg": cfg.direction_step_deg,
            "direction_turn": cfg.direction_turn,
        },
        "locations_processed": len(results),
        "images_processed": sum(r["num_images"] for r in results),
        "total_detections": len(all_rows),
        "location_results": results,
        "all_minimap_instances_csv": str(combined_csv),
        "output_dirs": {k: str(v) for k, v in output_dirs.items()},
    }
    save_json(cfg.output_dir / "run_report.json", report)
    return report


def default_model_states(cfg: Optional[PipelineConfig] = None, device: Optional[str] = None) -> Dict[str, Any]:
    """Initialize all model states with graceful fallbacks."""
    resolved = device or default_device()
    if cfg is None:
        tmp_root = Path(".")
        cfg = PipelineConfig(repo_root=tmp_root, data_dir=tmp_root / "data", output_dir=tmp_root / "outputs")
    return {
        "seg": init_segmentation_model(device=resolved),
        "depth": init_depth_model(device=resolved),
        "det": init_instance_detector(device=resolved),
        "det_zero_shot": init_zero_shot_detector(cfg, device=resolved),
    }


def summarize_detection_table(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize final detection rows by class and direction index."""
    by_class: Dict[str, int] = {}
    by_direction: Dict[int, int] = {}
    for row in rows:
        cls = normalize_class_name(row.get("class_name", "unknown"))
        by_class[cls] = by_class.get(cls, 0) + 1
        d = int(row.get("direction_index", -1))
        by_direction[d] = by_direction.get(d, 0) + 1
    return {
        "final_detection_count": len(rows),
        "detections_per_class": dict(sorted(by_class.items())),
        "detections_per_direction": dict(sorted(by_direction.items())),
    }
