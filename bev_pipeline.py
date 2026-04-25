"""Helper functions for the CSE 5509 BEV final project.

The code loads the dataset, runs pretrained vision models, projects results
into an approximate BEV canvas, and saves visualizations used in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence
import importlib.util
import json
import math


@dataclass
class PipelineConfig:
    repo_root: Path
    data_dir: Path
    output_dir: Path
    run_small_demo: bool = True
    demo_locations: int = 1
    demo_images_per_location: int = 2
    assumed_hfov_deg: float = 90.0
    depth_clip_min: float = 0.05
    depth_clip_max: float = 1.0
    bev_max_distance_m: float = 40.0
    bev_scale_px_per_m: float = 12.0
    bev_height_px: int = 720
    bev_width_px: int = 720
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


def resolve_project_paths() -> Dict[str, Path]:
    """Return project/data/output paths for both local and Colab modes."""
    cwd = Path.cwd()
    repo_root = cwd
    if not (repo_root / "data").exists() and (repo_root / "cse5509_final_project").exists():
        repo_root = repo_root / "cse5509_final_project"

    colab_drive_repo = Path("/content/drive/MyDrive/cse5509_final_project")
    if in_colab() and colab_drive_repo.exists():
        repo_root = colab_drive_repo

    data_dir = repo_root / "data"
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

    summary = {
        "location_count": len(locations),
        "total_images": total_images,
        "locations": [loc.name for loc in locations],
        "images_per_location": {k: len(v) for k, v in per_location.items()},
        "files": per_location,
    }
    return summary


def ensure_output_dirs(base: Path) -> Dict[str, Path]:
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
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return {"available": True, "processor": processor, "model": model, "id2label": model.config.id2label, "device": device}


def init_depth_model(device: str = "cpu") -> Dict[str, Any]:
    if not module_available("transformers"):
        return {"available": False, "reason": "transformers not installed"}
    from transformers import pipeline

    pipe = pipeline("depth-estimation", model="Intel/dpt-large", device=0 if device.startswith("cuda") else -1)
    return {"available": True, "pipeline": pipe}


def init_instance_detector(device: str = "cpu") -> Dict[str, Any]:
    if not module_available("torchvision") or not module_available("torch"):
        return {"available": False, "reason": "torchvision/torch not installed", "fallback": "no_detections"}
    import torch
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_Weights,
        maskrcnn_resnet50_fpn,
    )

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

    # Normalize per image so depth can be mapped into a fixed BEV distance range.
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
        mask = None
        if "masks" in pred:
            mask = pred["masks"][i, 0].detach().cpu().numpy()

        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        # Bottom-center of the box is a simple ground contact proxy.
        contact = ((x1 + x2) / 2.0, y2)
        instances.append(
            {
                "class_name": COCO_ID_TO_NAME[label_id],
                "confidence": score,
                "bbox": bbox,
                "mask": mask,
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
    cx = width / 2.0
    cy = height / 2.0
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


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
    intr = get_intrinsics(w, h, cfg.assumed_hfov_deg)
    dist = normalized_depth_to_distance(depth, cfg.depth_clip_min, cfg.depth_clip_max)

    bev = np.zeros((cfg.bev_height_px, cfg.bev_width_px, 3), dtype=np.uint8)

    ego_x = cfg.bev_width_px // 2
    ego_y = cfg.bev_height_px - 40

    ys, xs = np.where(ground | obstacles)
    for y, x in zip(ys[::2], xs[::2]):
        z_forward = float(dist[y, x] * cfg.bev_max_distance_m)
        if z_forward <= 0.01 or z_forward > cfg.bev_max_distance_m:
            continue

        x_cam = (x - intr["cx"]) / intr["fx"] * z_forward

        bev_x = int(round(ego_x + x_cam * cfg.bev_scale_px_per_m))
        bev_y = int(round(ego_y - z_forward * cfg.bev_scale_px_per_m))
        if 0 <= bev_x < cfg.bev_width_px and 0 <= bev_y < cfg.bev_height_px:
            if ground[y, x]:
                bev[bev_y, bev_x] = (60, 170, 60)
            elif obstacles[y, x]:
                bev[bev_y, bev_x] = (200, 90, 60)

    draw_bev_guides(bev, ego_x, ego_y, cfg)
    return {"bev": bev, "ego_xy": (ego_x, ego_y), "distance_m": dist * cfg.bev_max_distance_m}


def draw_bev_guides(bev, ego_x: int, ego_y: int, cfg: PipelineConfig) -> None:
    if not module_available("cv2"):
        return
    import cv2

    # OpenCV draws onto an RGB array here; colors are specified as RGB tuples.
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
    intr = get_intrinsics(depth_m.shape[1], depth_m.shape[0], cfg.assumed_hfov_deg)

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
    # OpenCV draws onto an RGB array here; colors are specified as RGB tuples.
    for inst in instances:
        x1, y1, x2, y2 = [int(v) for v in inst["bbox"]]
        cv2.rectangle(arr, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(
            arr,
            f"{inst['instance_label']} {inst['confidence']:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
    return arr


def _extract_direction_index(label: str):
    parts = label.lower().replace("_", " ").split()
    for i, token in enumerate(parts[:-1]):
        if token == "direction":
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def compose_location_bev(bev_images: Sequence, labels: Sequence[str], cfg: PipelineConfig):
    import numpy as np

    if len(bev_images) == 0:
        raise ValueError("No BEV images provided for stitching")

    canvas = np.zeros_like(bev_images[0], dtype=np.float32)
    step = 360.0 / max(1, len(bev_images))

    diagnostics = []
    for idx, bev in enumerate(bev_images):
        direction_idx = _extract_direction_index(labels[idx])
        rotation_deg = direction_idx * step if direction_idx is not None else idx * step
        rotated = bev
        if module_available("cv2"):
            import cv2

            h, w = bev.shape[:2]
            center = (w / 2.0, h / 2.0)
            matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
            rotated = cv2.warpAffine(
                bev.astype(np.uint8),
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        # This is approximate compositing, not true geometry-based stitching.
        alpha = 0.65
        canvas = canvas * (1.0 - alpha) + rotated.astype(np.float32) * alpha
        diagnostics.append({"view": labels[idx], "rotation_deg": round(rotation_deg, 1), "alpha": alpha})

    if module_available("cv2") and cfg.use_homography_diagnostics and len(bev_images) > 1:
        diagnostics.extend(compute_alignment_diagnostics(bev_images, labels))

    stitched = np.clip(canvas, 0, 255).astype("uint8")
    return stitched, diagnostics


def compute_alignment_diagnostics(images: Sequence, labels: Sequence[str]) -> List[Dict[str, Any]]:
    import cv2
    import numpy as np

    out: List[Dict[str, Any]] = []
    orb = cv2.ORB_create(1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i in range(len(images) - 1):
        a = images[i]
        b = images[i + 1]
        g1 = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        k1, d1 = orb.detectAndCompute(g1, None)
        k2, d2 = orb.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
            out.append(
                {
                    "pair": f"{labels[i]}->{labels[i+1]}",
                    "matches": 0,
                    "inliers": 0,
                    "status": "fallback_fixed_angle",
                }
            )
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
            }
        )
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
    bev_img, records = add_instance_markers(bev_res, instances, bev_res["distance_m"], cfg)

    overlay = draw_detection_overlay(image, instances)

    stem = image_path.stem.replace(" ", "_")
    loc = image_path.parent.name
    save_array_image(output_dirs["per_image"] / f"{loc}_{stem}_bev.png", bev_img)
    save_array_image(output_dirs["per_image"] / f"{loc}_{stem}_det.png", overlay)

    # Intermediate outputs
    import numpy as np

    raw_seg_vis = ((seg["label_map"] % 20) * 12).astype(np.uint8)
    raw_seg_vis = np.stack([raw_seg_vis, raw_seg_vis, raw_seg_vis], axis=-1)
    ground_vis = np.stack([seg["clean_ground_mask"] * 255] * 3, axis=-1).astype(np.uint8)
    obj_vis = np.stack([seg["clean_object_mask"] * 255] * 3, axis=-1).astype(np.uint8)
    save_array_image(output_dirs["diagnostics"] / f"{loc}_{stem}_raw_seg.png", raw_seg_vis)
    save_array_image(output_dirs["diagnostics"] / f"{loc}_{stem}_ground_mask.png", ground_vis)
    save_array_image(output_dirs["diagnostics"] / f"{loc}_{stem}_object_mask.png", obj_vis)

    save_json(output_dirs["tables"] / f"{loc}_{stem}_instances.json", records)

    return {
        "image_path": str(image_path),
        "bev_path": str(output_dirs["per_image"] / f"{loc}_{stem}_bev.png"),
        "det_path": str(output_dirs["per_image"] / f"{loc}_{stem}_det.png"),
        "instance_count": len(records),
        "instances": records,
    }


def process_location(location_name: str, image_paths: Sequence[Path], cfg: PipelineConfig, model_states: Dict[str, Any], output_dirs: Dict[str, Path]):
    image_results = []
    bev_stack = []
    labels = []

    for path in image_paths:
        result = process_image(path, cfg, model_states, output_dirs)
        image_results.append(result)

        from PIL import Image
        import numpy as np

        bev_stack.append(np.array(Image.open(result["bev_path"]).convert("RGB")))
        labels.append(path.stem)

    stitched, diagnostics = compose_location_bev(bev_stack, labels, cfg)
    save_array_image(output_dirs["per_location"] / f"{location_name}_stitched_bev.png", stitched)
    save_json(output_dirs["tables"] / f"{location_name}_stitch_diagnostics.json", diagnostics)

    return {
        "location": location_name,
        "num_images": len(image_results),
        "images": image_results,
        "stitched_bev": str(output_dirs["per_location"] / f"{location_name}_stitched_bev.png"),
    }


def default_model_states(device: str = "cpu") -> Dict[str, Any]:
    return {
        "seg": init_segmentation_model(device=device),
        "depth": init_depth_model(device=device),
        "det": init_instance_detector(device=device),
    }
