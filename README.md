# CSE 5509 Final Project: Ego-Centered 360° Minimap from Single-Camera Views

## Project Overview
This project builds an interpretable, game-style minimap/radar from handheld image captures at street locations. For each location, the camera remains at one fixed physical spot while the photographer rotates in place and captures eight views (`direction 0` through `direction 7`). The system combines pretrained semantic segmentation, monocular depth estimation, and object detection, then projects detections into a shared ego-centered top-down map. The final output is a per-location minimap with the ego marker at the center, distance rings, direction guides, and labeled object markers.

This is a course project visualization pipeline (qualitative analysis), not a calibrated vehicle-grade mapping system.

## Data Collection and Direction Convention
For each location:
- The camera position is fixed (no intentional translation between views).
- `direction 0` is the reference forward view.
- Each next image is captured after turning left by approximately 45°.
- Therefore, direction index increases with left turns:
  - `direction 1` ≈ +45° left,
  - `direction 2` ≈ +90° left,
  - ...,
  - `direction 7` ≈ +315° left.

The pipeline uses this known capture convention directly instead of trying to estimate per-view camera translation.

## Methods and Course Topic Coverage
The pipeline intentionally combines multiple CSE 5509-relevant computer vision tasks:
1. **Pretrained semantic segmentation (SegFormer)** for scene structure and ground/object diagnostics.
2. **Pretrained monocular depth estimation (DPT)** for approximate distance cues.
3. **Pretrained object detection (Mask R-CNN)** for object categories and bounding boxes.
4. **Geometric minimap projection** using assumed horizontal FOV + contact-point depth + known direction headings.

## Dataset Format
Expected layout:

```text
repo_root/
  data/
    loc1/
      direction 0.jpg
      direction 1.jpg
      ...
      direction 7.jpg
    loc2/
      ...
```

Supported direction-like filename patterns include examples such as:
- `direction 0.jpg`
- `direction_1.jpg`
- `loc1_direction_7.jpg`

## How the Minimap Projection Works (Approximate)
For each detection:
1. Use the bottom-center of the bounding box as a ground contact proxy.
2. Read monocular depth at that contact point; map to approximate meters.
3. Convert contact x-position into lateral angle using an assumed horizontal FOV.
4. Estimate camera-relative coordinates:
   - lateral offset (right/left)
   - forward range
5. Rotate that 2D point by known heading from the image direction index.
6. Draw all rotated detections into one shared ego-centered minimap canvas.

This avoids the old confusing approach of rotating whole rendered BEV canvases with off-center ego points.

## Running Locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open `final-project-cse5509-v2.ipynb` and run cells top-to-bottom.

## Running in Google Colab
1. Open the notebook in Colab.
2. Mount Drive when prompted.
3. Ensure your folder contains:
   - `bev_pipeline.py`
   - `final-project-cse5509-v2.ipynb`
   - `data/`
4. Run cells top-to-bottom.

The notebook uses `PROJECT_DIR` and `resolve_project_paths(...)` so the same workflow works for local and Colab usage with minimal manual edits.

## Main Outputs
- **Primary final output (per location)**:
  - `outputs/per_location/loc1_minimap.png`
  - `outputs/per_location/loc2_minimap.png`
  - ...
- Per-image diagnostics:
  - `outputs/per_image/*_det.png` (detection overlays)
  - `outputs/per_image/*_bev.png` (BEV diagnostics)
- Direction convention debug plot:
  - `outputs/diagnostics/loc*_direction_debug.png`
- Instance tables:
  - `outputs/tables/loc*_minimap_instances.json`
  - `outputs/tables/loc*_minimap_instances.csv`
  - `outputs/tables/all_minimap_instances.csv`
- Full run summary:
  - `outputs/run_report.json`

## How to Interpret the Minimap
- Ego/player marker is at the image center.
- `dir0` arrow points forward/reference direction.
- Rings indicate approximate distance (e.g., 5m, 10m, 20m, 30m, 40m).
- Marker colors indicate object class (person, car, truck, bus, bicycle, motorcycle).
- Labels (e.g., `car1`, `person2`) are shown for the most readable subset.

## Intermediate Diagnostics
The project keeps intermediate outputs so TAs can inspect failure modes:
- segmentation-derived masks,
- per-image detections and BEV diagnostics,
- direction-convention debug rays,
- per-location object tables with bearing/range and projected minimap coordinates.

## Assumptions and Limitations
- Monocular depth is not true calibrated metric distance.
- Camera intrinsics are approximated from image width + assumed horizontal FOV.
- Bounding-box bottom center is only an approximate ground-contact estimate.
- Adjacent direction views overlap, so repeated detections can occur.
- Dense rows of cars/bicycles can yield many close detections, potentially cluttering the minimap.
- The minimap is best interpreted qualitatively.

## AI Usage Statement
AI tools were used to assist with code restructuring, documentation drafting, and notebook organization. Final design decisions, limitations, and project claims were reviewed and edited for accurate course-project reporting.
