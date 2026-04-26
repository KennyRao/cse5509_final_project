import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from bev_pipeline import (
    PipelineConfig,
    assign_instance_labels,
    ego_meters_to_minimap_px,
    extract_direction_index,
    heading_for_direction,
    render_location_minimap,
    rotate_camera_relative_to_ego,
)


def _cfg() -> PipelineConfig:
    root = Path('.')
    return PipelineConfig(repo_root=root, data_dir=root / 'data', output_dir=root / 'outputs')


def test_direction_index_parsing() -> None:
    assert extract_direction_index('direction 0.jpg') == 0
    assert extract_direction_index('direction_1.jpg') == 1
    assert extract_direction_index('loc1_direction_7.jpg') == 7


def test_heading_for_direction_left_turn() -> None:
    cfg = _cfg()
    assert heading_for_direction(0, cfg) == 0.0
    assert heading_for_direction(1, cfg) == 45.0
    assert heading_for_direction(2, cfg) == 90.0
    assert heading_for_direction(4, cfg) == 180.0


def test_rotation_and_minimap_axes() -> None:
    cfg = _cfg()
    # Direction 0: forward should map upward (smaller pixel y).
    x0, y0 = rotate_camera_relative_to_ego(0.0, 10.0, heading_for_direction(0, cfg), cfg)
    # Direction 2: forward should map left (negative x in ego frame).
    x2, y2 = rotate_camera_relative_to_ego(0.0, 10.0, heading_for_direction(2, cfg), cfg)
    assert y0 > 0
    assert x2 < 0
    cx, cy = ego_meters_to_minimap_px(0.0, 0.0, cfg)
    p0 = ego_meters_to_minimap_px(x0, y0, cfg)
    p2 = ego_meters_to_minimap_px(x2, y2, cfg)
    assert p0[1] < cy
    assert p2[0] < cx


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
