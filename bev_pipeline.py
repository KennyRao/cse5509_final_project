"""Helper functions for the CSE 5509 BEV/minimap final project.

The pipeline combines pretrained semantic segmentation, monocular depth, and
instance detection, then projects detections into an ego-centered minimap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import csv
import importlib.util
import json
import math
import re
import shutil


@dataclass
class PipelineConfig:
    repo_root: Path
    data_dir: Path
    output_dir: Path
    run_small_demo: bool = True
    demo_locations: int = 1
    demo_images_per_location: int = 2
    clean_output_dir: bool = False

    # Geometric / depth assumptions.
    horizontal_fov_deg: float = 70.0
    min_depth_m: float = 2.0
    max_depth_m: float = 40.0

    # Legacy per-image BEV diagnostic settings.
    bev_max_distance_m: float = 40.0
    bev_scale_px_per_m: float = 12.0
    bev_height_px: int = 720
    bev_width_px: int = 720

    # Minimap settings (primary output).
    minimap_size_px: int = 900
    minimap_max_distance_m: float = 40.0
    minimap_scale_px_per_m: Optional[float] = None
    minimap_draw_dense_points: bool = True
    minimap_draw_object_labels: bool = True
    minimap_min_confidence: float = 0.5
    minimap_label_top_k_per_location: int = 30
    minimap_marker_alpha: float = 0.85
    minimap_jitter_overlapping_markers: bool = True
    minimap_merge_nearby_same_class: bool = False
    minimap_merge_radius_m: float = 1.0

    # Direction convention.
    direction_zero_heading_deg: float = 0.0
    direction_step_deg: float = 45.0
    direction_turn: str = "left"

    detection_threshold: float = 0.5
    use_homography_diagnostics: bool = True


COCO_ID_TO_NAME = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

CLASS_COLORS = {
    "person": (247, 111, 111),
    "bicycle": (255, 194, 87),
    "car": (96, 187, 255),
    "motorcycle": (184, 143, 255),
    "bus": (255, 136, 52),
    "truck": (120, 232, 168),
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


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def in_colab() -> bool:
    return module_available("google.colab")


def resolve_project_paths(project_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Return project/data/output paths for local and Colab execution."""
    cwd = Path.cwd()
    repo_root = project_dir or cwd

    if project_dir is None:
        if not (repo_root / "data").exists() and (repo_root / "cse5509_final_project").exists():
            repo_root = repo_root / "cse5509_final_project"
        if in_colab():
            candidates = [
                Path("/content/drive/MyDrive/CSE 5509 Final Project"),
                Path("/content/drive/MyDrive/cse5509_final_project"),
            ]
            for candidate in candidates:
                if candidate.exists() and (candidate / "bev_pipeline.py").exists():
                    repo_root = candidate
                    break

    data_dir = repo_root / "data"
    if (data_dir / "data").exists():
        data_dir = data_dir / "data"
    output_dir = repo_root / "outputs"
    return {"repo_root": repo_root, "data_dir": data_dir, "output_dir": output_dir}


def discover_dataset(data_dir: Path) -> Dict[str, Any]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    locations = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    per_location: Dict[str, List[Path]] = {}
    total_images = 0

    for loc in locations:
        images = sorted(loc.glob("*.jpg")) + sorted(loc.glob("*.png"))
        per_location[loc.name] = images
        total_images += len(images)

    return {
        "location_count": len(locations),
        "total_images": total_images,
        "locations": [loc.name for loc in locations],
        "images_per_location": {k: len(v) for k, v in per_location.items()},
        "files": per_location,
    }


def ensure_output_dirs(base: Path, clean_output_dir: bool = False) -> Dict[str, Path]:
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
    if module_available("torch"):
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def load_rgb_image(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    from PIL import Image

    return Image.open(path).convert("RGB")


def init_segmentation_model(device: str = "cpu") -> Dict[str, Any]:
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
    if not module_available("transformers"):
        return {"available": False, "reason": "transformers not installed"}
    from transformers import pipeline

    pipe = pipeline("depth-estimation", model="Intel/dpt-large", device=0 if device.startswith("cuda") else -1)
    return {"available": True, "pipeline": pipe}


def init_instance_detector(device: str = "cpu") -> Dict[str, Any]:
    if not module_available("torchvision") or not module_available("torch"):
        return {"available": False, "reason": "torchvision/torch not installed", "fallback": "no_detections"}
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    preprocess = weights.transforms()
    return {"available": True, "model": model, "preprocess": preprocess, "device": device}


def infer_segmentation(image, seg_state: Dict[str, Any]):
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
    import numpy as np

    if module_available("cv2"):
        import cv2

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
    import numpy as np

    if not module_available("cv2"):
        return mask
    import cv2

    out = np.zeros_like(mask, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"), connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out[labels == i] = True
    return out


def infer_instances(image, detector_state: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
    if not detector_state.get("available", False):
        return {"instances": [], "warning": detector_state.get("reason", "detector unavailable")}

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
                "class_name": COCO_ID_TO_NAME[label_id],
                "confidence": score,
                "bbox": bbox,
                "center_xy": center,
                "contact_xy": contact,
            }
        )

    instances = assign_instance_labels(instances)
    return {"instances": instances}


def assign_instance_labels(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    labels = [i["instance_label"] for i in instances]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate instance labels found")


def get_intrinsics(width: int, height: int, hfov_deg: float) -> Dict[str, float]:
    fx = width / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
    fy = fx
    return {"fx": fx, "fy": fy, "cx": width / 2.0, "cy": height / 2.0}


def normalized_depth_to_distance(depth_norm, depth_min: float, depth_max: float):
    import numpy as np

    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    return depth_min + (depth_max - depth_min) * depth_norm


def build_bev(seg_result: Dict[str, Any], depth_result: Dict[str, Any], cfg: PipelineConfig):
    import numpy as np

    depth = depth_result["depth"]
    ground = seg_result["clean_ground_mask"]
    obstacles = seg_result["clean_object_mask"]
    if depth.shape != ground.shape:
        raise ValueError(f"Depth and segmentation shape mismatch: {depth.shape} vs {ground.shape}")

    h, w = depth.shape
    intr = get_intrinsics(w, h, cfg.horizontal_fov_deg)
    dist = normalized_depth_to_distance(depth, cfg.min_depth_m, cfg.max_depth_m)

    bev = np.zeros((cfg.bev_height_px, cfg.bev_width_px, 3), dtype=np.uint8)
    ego_x = cfg.bev_width_px // 2
    ego_y = cfg.bev_height_px - 40

    ys, xs = np.where(ground | obstacles)
    for y, x in zip(ys[::2], xs[::2]):
        z_forward = float(dist[y, x])
        if z_forward <= 0.01 or z_forward > cfg.bev_max_distance_m:
            continue
        x_cam = (x - intr["cx"]) / intr["fx"] * z_forward

        bev_x = int(round(ego_x + x_cam * cfg.bev_scale_px_per_m))
        bev_y = int(round(ego_y - z_forward * cfg.bev_scale_px_per_m))
        if 0 <= bev_x < cfg.bev_width_px and 0 <= bev_y < cfg.bev_height_px:
            bev[bev_y, bev_x] = (60, 170, 60) if ground[y, x] else (200, 90, 60)

    draw_bev_guides(bev, ego_x, ego_y, cfg)
    return {"bev": bev, "ego_xy": (ego_x, ego_y), "distance_m": dist}


def draw_bev_guides(bev, ego_x: int, ego_y: int, cfg: PipelineConfig) -> None:
    if not module_available("cv2"):
        return
    import cv2

    cv2.circle(bev, (ego_x, ego_y), 8, (255, 255, 255), -1)
    cv2.arrowedLine(bev, (ego_x, ego_y), (ego_x, max(5, ego_y - 70)), (255, 255, 0), 2, tipLength=0.2)

    for meters in [5, 10, 20, 30, 40]:
        r = int(meters * cfg.bev_scale_px_per_m)
        y = ego_y - r
        if y > 0:
            cv2.ellipse(bev, (ego_x, ego_y), (r, r), 0, 180, 360, (80, 80, 80), 1)
            cv2.putText(bev, f"{meters}m", (ego_x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)


def add_instance_markers(bev_result: Dict[str, Any], instances: List[Dict[str, Any]], depth_m, cfg: PipelineConfig):
    bev = bev_result["bev"]
    ego_x, ego_y = bev_result["ego_xy"]
    intr = get_intrinsics(depth_m.shape[1], depth_m.shape[0], cfg.horizontal_fov_deg)

    if not module_available("cv2"):
        return bev, []
    import cv2

    records = []
    for inst in instances:
        cx, cy = inst["contact_xy"]
        ix = int(max(0, min(depth_m.shape[1] - 1, round(cx))))
        iy = int(max(0, min(depth_m.shape[0] - 1, round(cy))))
        z_forward = float(depth_m[iy, ix])
        x_cam = (cx - intr["cx"]) / intr["fx"] * z_forward
        bev_x = int(round(ego_x + x_cam * cfg.bev_scale_px_per_m))
        bev_y = int(round(ego_y - z_forward * cfg.bev_scale_px_per_m))

        inst["estimated_depth_m"] = z_forward
        inst["bev_xy"] = (bev_x, bev_y)

        if 0 <= bev_x < cfg.bev_width_px and 0 <= bev_y < cfg.bev_height_px:
            cv2.circle(bev, (bev_x, bev_y), 5, (235, 50, 50), -1)
            cv2.putText(bev, inst["instance_label"], (bev_x + 6, bev_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        records.append(
            {
                "instance_label": inst["instance_label"],
                "class_name": inst["class_name"],
                "confidence": round(float(inst["confidence"]), 4),
                "bbox": [round(v, 2) for v in inst["bbox"]],
                "center_xy": [round(v, 2) for v in inst["center_xy"]],
                "contact_xy": [round(v, 2) for v in inst["contact_xy"]],
                "estimated_depth_m": round(float(z_forward), 3),
                "bev_xy": [int(bev_x), int(bev_y)],
            }
        )

    return bev, records


def draw_detection_overlay(image, instances: Sequence[Dict[str, Any]]):
    import numpy as np

    arr = np.array(image).copy()
    if not module_available("cv2"):
        return arr

    import cv2

    for inst in instances:
        x1, y1, x2, y2 = [int(v) for v in inst["bbox"]]
        cv2.rectangle(arr, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(arr, f"{inst['instance_label']} {inst['confidence']:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return arr


def extract_direction_index(path_or_name: str) -> Optional[int]:
    text = Path(path_or_name).stem.lower()
    m = re.search(r"direction[\s_\-]*(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"\bdir[\s_\-]*(\d+)\b", text)
    if m:
        return int(m.group(1))
    return None


def heading_for_direction(direction_index: int, cfg: PipelineConfig) -> float:
    sign = 1.0 if cfg.direction_turn.lower() == "left" else -1.0
    return cfg.direction_zero_heading_deg + sign * direction_index * cfg.direction_step_deg


def estimate_camera_relative_position(instance: Dict[str, Any], image_width: int, cfg: PipelineConfig) -> Tuple[float, float]:
    contact_x = float(instance["contact_xy"][0])
    depth_m = float(instance["estimated_depth_m"])
    normalized_x = (contact_x - image_width / 2.0) / (image_width / 2.0)
    angle_offset_deg = normalized_x * (cfg.horizontal_fov_deg / 2.0)
    lateral_x_m = depth_m * math.tan(math.radians(angle_offset_deg))
    forward_m = depth_m
    return lateral_x_m, forward_m


def rotate_camera_relative_to_ego(lateral_x_m: float, forward_m: float, heading_deg: float, cfg: PipelineConfig) -> Tuple[float, float]:
    _ = cfg
    theta = math.radians(heading_deg)
    ego_x = lateral_x_m * math.cos(theta) - forward_m * math.sin(theta)
    ego_y = lateral_x_m * math.sin(theta) + forward_m * math.cos(theta)
    return ego_x, ego_y


def minimap_scale_px_per_m(cfg: PipelineConfig) -> float:
    if cfg.minimap_scale_px_per_m is not None:
        return cfg.minimap_scale_px_per_m
    return (cfg.minimap_size_px * 0.48) / cfg.minimap_max_distance_m


def ego_meters_to_minimap_px(x_m: float, y_m: float, cfg: PipelineConfig) -> Tuple[int, int]:
    scale = minimap_scale_px_per_m(cfg)
    c = cfg.minimap_size_px // 2
    px = int(round(c + x_m * scale))
    py = int(round(c - y_m * scale))
    return px, py


def draw_minimap_guides(canvas, cfg: PipelineConfig) -> None:
    if not module_available("cv2"):
        return
    import cv2

    c = cfg.minimap_size_px // 2
    scale = minimap_scale_px_per_m(cfg)
    ring_m = [5, 10, 20, 30, 40]

    cv2.circle(canvas, (c, c), 9, (255, 255, 255), -1)
    cv2.arrowedLine(canvas, (c, c), (c, c - 70), (0, 255, 255), 3, tipLength=0.25)
    cv2.putText(canvas, "dir0", (c + 8, c - 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)

    for meters in ring_m:
        r = int(meters * scale)
        cv2.circle(canvas, (c, c), r, (75, 75, 75), 1)
        cv2.putText(canvas, f"{meters}m", (c + 6, c - r - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    for i in range(8):
        heading = heading_for_direction(i, cfg)
        theta = math.radians(heading)
        x = int(round(c + math.sin(theta) * (cfg.minimap_size_px * 0.45)))
        y = int(round(c - math.cos(theta) * (cfg.minimap_size_px * 0.45)))
        cv2.line(canvas, (c, c), (x, y), (45, 45, 45), 1)
        cv2.putText(canvas, f"dir{i}", (x - 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)


def render_direction_debug_plot(cfg: PipelineConfig, output_path: Path) -> Path:
    import numpy as np

    canvas = np.zeros((cfg.minimap_size_px, cfg.minimap_size_px, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)
    draw_minimap_guides(canvas, cfg)
    save_array_image(output_path, canvas)
    return output_path


def _bearing_deg(x_m: float, y_m: float) -> float:
    return math.degrees(math.atan2(-x_m, y_m))


def _apply_optional_merge(rows: List[Dict[str, Any]], cfg: PipelineConfig) -> List[Dict[str, Any]]:
    if not cfg.minimap_merge_nearby_same_class:
        return rows

    merged: List[Dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: -r["confidence"]):
        found = False
        for m in merged:
            if row["class_name"] != m["class_name"]:
                continue
            dist = math.hypot(row["ego_x_m"] - m["ego_x_m"], row["ego_y_m"] - m["ego_y_m"])
            if dist <= cfg.minimap_merge_radius_m:
                found = True
                break
        if not found:
            merged.append(row)
    return merged


def render_location_minimap(location_name: str, rows: List[Dict[str, Any]], cfg: PipelineConfig, output_path: Path) -> Path:
    import numpy as np

    if not module_available("cv2"):
        canvas = np.zeros((cfg.minimap_size_px, cfg.minimap_size_px, 3), dtype=np.uint8)
        save_array_image(output_path, canvas)
        return output_path

    import cv2

    canvas = np.zeros((cfg.minimap_size_px, cfg.minimap_size_px, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)
    draw_minimap_guides(canvas, cfg)

    rows = _apply_optional_merge(rows, cfg)
    rows_for_labels = sorted(rows, key=lambda r: (-r["confidence"], r["range_m"]))[: cfg.minimap_label_top_k_per_location]
    label_keys = {(r["source_image"], r["instance_label"]) for r in rows_for_labels}

    occupancy: Dict[Tuple[int, int], int] = {}
    for row in rows:
        x, y = row["minimap_xy"]
        if cfg.minimap_jitter_overlapping_markers:
            key = (x, y)
            n = occupancy.get(key, 0)
            if n > 0:
                ang = n * 2.399963
                x += int(round(5 * math.cos(ang)))
                y += int(round(5 * math.sin(ang)))
            occupancy[key] = n + 1

        color = CLASS_COLORS.get(row["class_name"], (220, 220, 220))
        alpha = float(max(0.0, min(1.0, cfg.minimap_marker_alpha)))
        overlay = canvas.copy()
        cv2.circle(overlay, (x, y), 6, color, -1)
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0, dst=canvas)

        if cfg.minimap_draw_object_labels and (row["source_image"], row["instance_label"]) in label_keys:
            cv2.putText(canvas, row["instance_label"], (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1)

    cv2.putText(canvas, f"{location_name} minimap (ego-centered)", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    legend_items = ["person", "car", "truck", "bus", "bicycle", "motorcycle"]
    lx, ly = 20, cfg.minimap_size_px - 180
    cv2.rectangle(canvas, (lx - 10, ly - 30), (lx + 260, ly + 145), (35, 35, 35), -1)
    cv2.putText(canvas, "Legend", (lx, ly - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1)
    for i, name in enumerate(legend_items):
        y = ly + i * 23
        cv2.circle(canvas, (lx + 12, y + 6), 6, CLASS_COLORS.get(name, (200, 200, 200)), -1)
        cv2.putText(canvas, name, (lx + 26, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    save_array_image(output_path, canvas)
    return output_path


def _serialize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_image": row["source_image"],
        "location": row["location"],
        "direction_index": row["direction_index"],
        "heading_deg": round(row["heading_deg"], 3),
        "class_name": row["class_name"],
        "instance_label": row["instance_label"],
        "confidence": round(row["confidence"], 4),
        "bbox": [round(v, 2) for v in row["bbox"]],
        "center_xy": [round(v, 2) for v in row["center_xy"]],
        "contact_xy": [round(v, 2) for v in row["contact_xy"]],
        "estimated_depth_m": round(row["estimated_depth_m"], 3),
        "camera_lateral_x_m": round(row["camera_lateral_x_m"], 3),
        "camera_forward_m": round(row["camera_forward_m"], 3),
        "ego_x_m": round(row["ego_x_m"], 3),
        "ego_y_m": round(row["ego_y_m"], 3),
        "bearing_deg": round(row["bearing_deg"], 3),
        "range_m": round(row["range_m"], 3),
        "minimap_xy": [int(row["minimap_xy"][0]), int(row["minimap_xy"][1])],
        "clipped_to_minimap": bool(row["clipped_to_minimap"]),
    }


def save_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
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
    import numpy as np

    if len(bev_images) == 0:
        raise ValueError("No BEV images provided for stitching")

    canvas = np.zeros_like(bev_images[0], dtype=np.float32)
    diagnostics = []
    for idx, bev in enumerate(bev_images):
        direction_idx = extract_direction_index(labels[idx])
        if direction_idx is None:
            direction_idx = idx
        rotation_deg = heading_for_direction(direction_idx, cfg)
        rotated = bev
        if module_available("cv2"):
            import cv2

            h, w = bev.shape[:2]
            center = (w / 2.0, h / 2.0)
            matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
            rotated = cv2.warpAffine(bev.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        alpha = 0.65
        canvas = canvas * (1.0 - alpha) + rotated.astype(np.float32) * alpha
        diagnostics.append({"view": labels[idx], "rotation_deg": round(rotation_deg, 1), "alpha": alpha})

    stitched = np.clip(canvas, 0, 255).astype("uint8")
    if module_available("cv2") and cfg.use_homography_diagnostics and len(bev_images) > 1:
        diagnostics.extend(compute_alignment_diagnostics(bev_images, labels))
    return stitched, diagnostics


def compute_alignment_diagnostics(images: Sequence, labels: Sequence[str]) -> List[Dict[str, Any]]:
    import cv2
    import numpy as np

    out: List[Dict[str, Any]] = []
    orb = cv2.ORB_create(1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i in range(len(images) - 1):
        a, b = images[i], images[i + 1]
        g1 = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        k1, d1 = orb.detectAndCompute(g1, None)
        k2, d2 = orb.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
            out.append({"pair": f"{labels[i]}->{labels[i+1]}", "matches": 0, "inliers": 0, "status": "fallback_fixed_angle"})
            continue

        matches = sorted(matcher.match(d1, d2), key=lambda m: m.distance)
        pts1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        out.append({"pair": f"{labels[i]}->{labels[i+1]}", "matches": len(matches), "inliers": inliers, "status": "diagnostic_only" if H is not None else "fallback_fixed_angle"})
    return out


def save_array_image(path: Path, arr) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def process_image(image_path: Path, cfg: PipelineConfig, model_states: Dict[str, Any], output_dirs: Dict[str, Path]):
    image = load_rgb_image(image_path)
    seg = infer_segmentation(image, model_states["seg"])
    depth = infer_depth(image, model_states["depth"])
    det = infer_instances(image, model_states["det"], threshold=cfg.detection_threshold)

    instances = det["instances"]
    assert_unique_labels(instances)

    bev_res = build_bev(seg, depth, cfg)
    bev_img, _ = add_instance_markers(bev_res, instances, bev_res["distance_m"], cfg)
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
    distance_m = normalized_depth_to_distance(depth["depth"], cfg.min_depth_m, cfg.max_depth_m)
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
        depth_m = float(distance_m[iy, ix])
        depth_m = max(cfg.min_depth_m, min(cfg.max_depth_m, depth_m))
        inst["estimated_depth_m"] = depth_m

        cam_x_m, cam_y_m = estimate_camera_relative_position(inst, w, cfg)
        ego_x_m, ego_y_m = rotate_camera_relative_to_ego(cam_x_m, cam_y_m, heading_deg, cfg)
        range_m = math.hypot(ego_x_m, ego_y_m)
        bearing_deg = _bearing_deg(ego_x_m, ego_y_m)

        px, py = ego_meters_to_minimap_px(ego_x_m, ego_y_m, cfg)
        clipped = False
        if range_m > cfg.minimap_max_distance_m:
            clipped = True
            s = cfg.minimap_max_distance_m / max(range_m, 1e-6)
            ego_x_m *= s
            ego_y_m *= s
            px, py = ego_meters_to_minimap_px(ego_x_m, ego_y_m, cfg)

        row = {
            "source_image": image_path.name,
            "location": loc,
            "direction_index": int(direction_index),
            "heading_deg": float(heading_deg),
            "class_name": inst["class_name"],
            "instance_label": inst["instance_label"],
            "confidence": float(inst["confidence"]),
            "bbox": list(inst["bbox"]),
            "center_xy": list(inst["center_xy"]),
            "contact_xy": list(inst["contact_xy"]),
            "estimated_depth_m": depth_m,
            "camera_lateral_x_m": cam_x_m,
            "camera_forward_m": cam_y_m,
            "ego_x_m": ego_x_m,
            "ego_y_m": ego_y_m,
            "bearing_deg": bearing_deg,
            "range_m": range_m,
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
    image_results = []
    bev_stack = []
    labels = []
    loc_rows: List[Dict[str, Any]] = []

    for path in image_paths:
        result = process_image(path, cfg, model_states, output_dirs)
        image_results.append(result)
        loc_rows.extend(result["instances"])

        from PIL import Image
        import numpy as np

        bev_stack.append(np.array(Image.open(result["bev_path"]).convert("RGB")))
        labels.append(path.stem)

    minimap_path = output_dirs["per_location"] / f"{location_name}_minimap.png"
    render_location_minimap(location_name, loc_rows, cfg, minimap_path)

    direction_debug_path = output_dirs["diagnostics"] / f"{location_name}_direction_debug.png"
    render_direction_debug_plot(cfg, direction_debug_path)

    stitched, diagnostics = compose_location_bev(bev_stack, labels, cfg)
    stitched_path = output_dirs["per_location"] / f"{location_name}_stitched_bev.png"
    save_array_image(stitched_path, stitched)
    save_json(output_dirs["tables"] / f"{location_name}_stitch_diagnostics.json", diagnostics)

    loc_json_rows = [_serialize_row(r) for r in loc_rows]
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
        "total_instances": len(loc_rows),
    }


def run_pipeline(cfg: PipelineConfig, summary: Dict[str, Any], model_states: Dict[str, Any]) -> Dict[str, Any]:
    output_dirs = ensure_output_dirs(cfg.output_dir, clean_output_dir=cfg.clean_output_dir)

    location_names = summary["locations"]
    if cfg.run_small_demo:
        location_names = location_names[: cfg.demo_locations]

    results = []
    all_rows: List[Dict[str, Any]] = []
    for loc in location_names:
        image_paths = summary["files"][loc]
        if cfg.run_small_demo:
            image_paths = image_paths[: cfg.demo_images_per_location]
        res = process_location(loc, image_paths, cfg, model_states, output_dirs)
        results.append(res)
        for image_result in res["images"]:
            all_rows.extend(image_result["instances"])

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
            "min_depth_m": cfg.min_depth_m,
            "max_depth_m": cfg.max_depth_m,
            "minimap_size_px": cfg.minimap_size_px,
            "minimap_max_distance_m": cfg.minimap_max_distance_m,
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


def default_model_states(device: Optional[str] = None) -> Dict[str, Any]:
    resolved = device or default_device()
    return {
        "seg": init_segmentation_model(device=resolved),
        "depth": init_depth_model(device=resolved),
        "det": init_instance_detector(device=resolved),
    }
