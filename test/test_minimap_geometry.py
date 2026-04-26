import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from bev_pipeline import (
    PipelineConfig,
    _post_process_grounding_dino,
    assign_instance_labels,
    compute_alignment_diagnostics,
    compose_location_bev,
    deduplicate_location_rows,
    ego_meters_to_minimap_px,
    extract_direction_index,
    heading_for_direction,
    normalized_depth_to_distance,
    normalize_class_name,
    resolve_project_paths,
    render_location_minimap,
    run_pipeline,
    rotate_camera_relative_to_ego,
)


def _cfg() -> PipelineConfig:
    root = Path('.')
    return PipelineConfig(repo_root=root, data_dir=root / 'data', output_dir=root / 'outputs')


def test_direction_index_parsing() -> None:
    assert extract_direction_index('direction 0.jpg') == 0
    assert extract_direction_index('direction_1.jpg') == 1
    assert extract_direction_index('loc1_direction_7.jpg') == 7


def test_heading_for_direction_clockwise() -> None:
    cfg = _cfg()
    assert heading_for_direction(0, cfg) == 0.0
    assert heading_for_direction(1, cfg) == 45.0
    assert heading_for_direction(2, cfg) == 90.0
    assert heading_for_direction(4, cfg) == 180.0
    assert heading_for_direction(6, cfg) == 270.0
    assert heading_for_direction(7, cfg) == 315.0


def test_rotation_and_minimap_axes() -> None:
    cfg = _cfg()
    # Direction 0: forward should map upward (smaller pixel y).
    x0, y0 = rotate_camera_relative_to_ego(0.0, 10.0, heading_for_direction(0, cfg), cfg)
    # Direction 1: forward should map up-right in ego frame.
    x1, y1 = rotate_camera_relative_to_ego(0.0, 10.0, heading_for_direction(1, cfg), cfg)
    # Direction 2: forward should map right in ego frame.
    x2, y2 = rotate_camera_relative_to_ego(0.0, 10.0, heading_for_direction(2, cfg), cfg)
    # Direction 6: forward should map left in ego frame.
    x6, y6 = rotate_camera_relative_to_ego(0.0, 10.0, heading_for_direction(6, cfg), cfg)
    assert y0 > 0
    assert x1 > 0 and y1 > 0
    assert x2 > 0 and abs(y2) < 1e-6
    assert x6 < 0 and abs(y6) < 1e-6
    cx, cy = ego_meters_to_minimap_px(0.0, 0.0, cfg)
    p0 = ego_meters_to_minimap_px(x0, y0, cfg)
    p1 = ego_meters_to_minimap_px(x1, y1, cfg)
    p2 = ego_meters_to_minimap_px(x2, y2, cfg)
    p6 = ego_meters_to_minimap_px(x6, y6, cfg)
    assert p0[1] < cy
    assert p1[0] > cx and p1[1] < cy
    assert p2[0] > cx
    assert p6[0] < cx


def test_compose_location_bev_rotates_from_fixed_center() -> None:
    import importlib.util
    if importlib.util.find_spec('numpy') is None:
        return
    import numpy as np

    cfg = _cfg()
    cfg.bev_width_px = 100
    cfg.bev_height_px = 100
    cfg.bev_scale_px_per_m = 10.0
    cfg.bev_max_distance_m = 10.0
    ego_x = cfg.bev_width_px // 2
    ego_y = cfg.bev_height_px - 40
    patch_y = ego_y - 20  # 2m forward from source ego origin.

    def mk_bev(color):
        bev = np.zeros((cfg.bev_height_px, cfg.bev_width_px, 3), dtype=np.uint8)
        bev[patch_y - 1: patch_y + 2, ego_x - 1: ego_x + 2] = color
        return bev

    bevs = [mk_bev((255, 0, 0)), mk_bev((0, 255, 0)), mk_bev((0, 0, 255)), mk_bev((255, 255, 0))]
    labels = ["direction 0", "direction 1", "direction 2", "direction 6"]
    stitched, _ = compose_location_bev(bevs, labels, cfg)

    center = stitched.shape[0] // 2
    tol = 4
    # dir0 should be above center.
    assert np.any(stitched[center - 20 - tol:center - 20 + tol + 1, center - tol:center + tol + 1] > 0)
    # dir1 should be above-right.
    assert np.any(stitched[center - 14 - tol:center - 14 + tol + 1, center + 14 - tol:center + 14 + tol + 1] > 0)
    # dir2 should be right.
    assert np.any(stitched[center - tol:center + tol + 1, center + 20 - tol:center + 20 + tol + 1] > 0)
    # dir6 should be left.
    assert np.any(stitched[center - tol:center + tol + 1, center - 20 - tol:center - 20 + tol + 1] > 0)


def test_compose_location_bev_ignores_black_overwrite() -> None:
    import importlib.util
    if importlib.util.find_spec('numpy') is None:
        return
    import numpy as np

    cfg = _cfg()
    cfg.bev_width_px = 100
    cfg.bev_height_px = 100
    cfg.bev_scale_px_per_m = 10.0
    cfg.bev_max_distance_m = 10.0
    ego_x = cfg.bev_width_px // 2
    ego_y = cfg.bev_height_px - 40

    first = np.zeros((cfg.bev_height_px, cfg.bev_width_px, 3), dtype=np.uint8)
    first[ego_y - 20: ego_y - 17, ego_x - 1: ego_x + 2] = (255, 0, 0)
    second = np.zeros_like(first)  # all black should not erase first contribution
    stitched, _ = compose_location_bev([first, second], ["direction 0", "direction 1"], cfg)

    center = stitched.shape[0] // 2
    assert np.any(stitched[center - 24:center - 16, center - 4:center + 4, 0] > 0)


def test_unique_instance_labels() -> None:
    instances = [
        {'class_name': 'car', 'contact_xy': (10, 20)},
        {'class_name': 'car', 'contact_xy': (15, 20)},
        {'class_name': 'person', 'contact_xy': (12, 25)},
    ]
    labeled = assign_instance_labels(instances)
    labels = [i['instance_label'] for i in labeled]
    assert len(labels) == len(set(labels))


def test_minimap_rendering_synthetic(tmp_path: Path) -> None:
    import importlib.util
    if importlib.util.find_spec('numpy') is None:
        return
    cfg = _cfg()
    out = tmp_path / 'synthetic_minimap.png'
    rows = [
        {
            'source_image': 'direction 0.jpg',
            'location': 'locX',
            'direction_index': 0,
            'heading_deg': 0.0,
            'class_name': 'car',
            'instance_label': 'car1',
            'confidence': 0.9,
            'bbox': [0, 0, 1, 1],
            'center_xy': [0, 0],
            'contact_xy': [0, 0],
            'estimated_depth_m': 10.0,
            'camera_lateral_x_m': 0.0,
            'camera_forward_m': 10.0,
            'ego_x_m': 0.0,
            'ego_y_m': 10.0,
            'bearing_deg': 0.0,
            'range_m': 10.0,
            'minimap_xy': [cfg.minimap_size_px // 2, cfg.minimap_size_px // 2 - 100],
            'clipped_to_minimap': False,
        }
    ]
    render_location_minimap('locX', rows, cfg, out)
    assert out.exists()


def test_resolve_project_paths_uses_repo_data_dir_only(tmp_path: Path) -> None:
    (tmp_path / "data" / "data").mkdir(parents=True)
    paths = resolve_project_paths(tmp_path)
    resolved_tmp = tmp_path.resolve()
    assert paths["repo_root"] == resolved_tmp
    assert paths["data_dir"] == resolved_tmp / "data"
    assert paths["output_dir"] == resolved_tmp / "outputs"


def test_depth_inverse_mapping() -> None:
    import numpy as np

    d = normalized_depth_to_distance(np.array([0.0, 1.0]), 2.0, 40.0, inverse=True)
    assert np.allclose(d, np.array([40.0, 2.0]), atol=1e-4)


def test_dedup_keeps_highest_confidence() -> None:
    cfg = _cfg()
    rows = [
        {"class_name": "car", "confidence": 0.80, "ego_x_m": 0.0, "ego_y_m": 0.0, "instance_label": "car_old", "detector_source": "a"},
        {"class_name": "car", "confidence": 0.95, "ego_x_m": 0.2, "ego_y_m": 0.0, "instance_label": "car_new", "detector_source": "b"},
    ]
    kept, _ = deduplicate_location_rows(rows, cfg)
    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.95


def test_dedup_does_not_merge_different_classes() -> None:
    cfg = _cfg()
    rows = [
        {"class_name": "car", "confidence": 0.9, "ego_x_m": 0.0, "ego_y_m": 0.0, "instance_label": "car1", "detector_source": "a"},
        {"class_name": "bus", "confidence": 0.85, "ego_x_m": 0.0, "ego_y_m": 0.0, "instance_label": "bus1", "detector_source": "b"},
    ]
    kept, _ = deduplicate_location_rows(rows, cfg)
    assert len(kept) == 2


def test_alignment_diagnostics_accepts_rgb_images() -> None:
    import importlib.util
    if importlib.util.find_spec('numpy') is None:
        return
    import numpy as np
    images = [np.zeros((20, 20, 3), dtype=np.uint8), np.zeros((20, 20, 3), dtype=np.uint8)]
    diag = compute_alignment_diagnostics(images, ["direction 0", "direction 1"])
    assert len(diag) >= 1
    assert "status" in diag[0]


def test_small_demo_one_full_location(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    loc1 = data_dir / "loc1"
    loc2 = data_dir / "loc2"
    loc1.mkdir(parents=True)
    loc2.mkdir(parents=True)
    for i in range(3):
        (loc1 / f"direction {i}.jpg").write_bytes(b"x")
    for i in range(2):
        (loc2 / f"direction {i}.jpg").write_bytes(b"x")

    summary = {
        "locations": ["loc1", "loc2"],
        "files": {"loc1": [loc1 / f"direction {i}.jpg" for i in range(3)], "loc2": [loc2 / f"direction {i}.jpg" for i in range(2)]},
    }
    cfg = PipelineConfig(
        repo_root=tmp_path,
        data_dir=data_dir,
        output_dir=tmp_path / "outputs",
        run_small_demo=True,
        demo_locations=1,
        demo_images_per_location=None,
    )

    import bev_pipeline

    calls = []

    def fake_process_location(location_name, image_paths, cfg, model_states, output_dirs):
        calls.append((location_name, len(image_paths)))
        return {"num_images": len(image_paths), "images": [], "rows": []}

    original = bev_pipeline.process_location
    bev_pipeline.process_location = fake_process_location
    try:
        run_pipeline(cfg, summary, {"seg": {}, "depth": {}, "det": {}, "det_zero_shot": {}})
    finally:
        bev_pipeline.process_location = original

    assert calls == [("loc1", 3)]


def test_grounding_dino_postprocess_accepts_box_threshold_api() -> None:
    class FakeProcessor:
        def post_process_grounded_object_detection(
            self,
            outputs,
            input_ids,
            box_threshold,
            text_threshold,
            target_sizes,
        ):
            assert box_threshold == 0.35
            assert text_threshold == 0.30
            assert target_sizes == [(60, 80)]
            return [{"scores": [0.91], "boxes": [[1, 2, 3, 4]], "labels": ["car"]}]

    class FakeImage:
        width = 80
        height = 60

    cfg = _cfg()
    post = _post_process_grounding_dino(FakeProcessor(), outputs={}, input_ids=[1, 2], cfg=cfg, image=FakeImage())
    assert post["labels"] == ["car"]


def test_grounding_dino_postprocess_accepts_threshold_api() -> None:
    class FakeProcessor:
        def post_process_grounded_object_detection(
            self,
            outputs,
            input_ids=None,
            threshold=None,
            text_threshold=None,
            target_sizes=None,
            box_threshold=None,
        ):
            if box_threshold is not None:
                raise TypeError("unexpected keyword argument 'box_threshold'")
            assert threshold == 0.35
            assert text_threshold == 0.30
            assert target_sizes == [(100, 200)]
            assert input_ids == [42]
            return [{"scores": [0.88], "boxes": [[10, 20, 30, 40]], "text_labels": ["a car"]}]

    class FakeImage:
        width = 200
        height = 100

    cfg = _cfg()
    post = _post_process_grounding_dino(FakeProcessor(), outputs={}, input_ids=[42], cfg=cfg, image=FakeImage())
    assert post["text_labels"] == ["a car"]


def test_normalize_class_name_zero_shot_variants() -> None:
    assert normalize_class_name("road sign.") == "road_sign"
    assert normalize_class_name("traffic sign") == "road_sign"
    assert normalize_class_name("stop sign") == "road_sign"
    assert normalize_class_name("trash dumpster") == "dumpster"
    assert normalize_class_name("a car") == "car"
