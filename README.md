# CSE 5509 Final Project: Single-Camera BEV Scene Visualization

## Overview
This project takes multiple street-view images per location (for example, `direction 0.jpg` to `direction 7.jpg`) and generates Bird's-Eye View (BEV)-style visualizations. The outputs combine a segmented scene layout with object markers from instance detection.

This is a course project pipeline, not a calibrated vehicle BEV system. Distances and object positions are approximate because camera intrinsics are assumed (not calibrated) and monocular depth has scale ambiguity.

## Methods
- **Semantic segmentation**: A pretrained SegFormer model predicts per-pixel classes. We use those labels to separate likely ground regions (such as road/sidewalk/terrain) from movable-object regions.
- **Monocular depth estimation**: A pretrained DPT model predicts relative depth from one image. The depth map is normalized and used for approximate forward range placement in the BEV canvas.
- **Instance detection**: A pretrained Mask R-CNN detector finds people/vehicles and keeps supported classes (`person`, `car`, `bus`, etc.). Instances are labeled per image as `car1`, `car2`, `person1`, and so on.
- **Approximate pinhole projection**: Image points are mapped into a top-down canvas using assumed camera intrinsics from image width and a horizontal FOV setting.
- **Per-location compositing**: Per-direction BEV images are combined with a fixed-angle assumption. When filenames include `direction N`, that index is used for simple rotation before alpha compositing.
- **Optional ORB/homography checks**: If OpenCV is available and enabled, the pipeline saves feature-match and homography inlier summaries as alignment checks.

## Repository Structure
```text
cse5509_final_project/
├── data/
├── bev_pipeline.py
├── final-project-cse5509-v2.ipynb
├── requirements.txt
└── outputs/
```

## Dataset Format
Expected input format:

```text
data/
  loc1/
    direction 0.jpg
    direction 1.jpg
    ...
  loc2/
    direction 0.jpg
    ...
```

The notebook counts folders and images from `data/` at runtime, so it does not rely on hard-coded totals.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Windows activation command:

```bash
.venv\Scripts\activate
```

On the first run, pretrained model weights may be downloaded, so model initialization can take a while.

## Running the Notebook
1. Open `final-project-cse5509-v2.ipynb`.
2. Run cells from top to bottom.
3. `run_small_demo=True` runs a small subset for quick checking.
4. Set `run_small_demo=False` for the full dataset.
5. Confirm the printed `Data dir` path before running the pipeline.
6. Outputs are saved under `outputs/`.

## Outputs
- `outputs/per_image/`: per-image BEV images and detection overlays.
- `outputs/per_location/`: combined per-location BEV composite image.
- `outputs/diagnostics/`: segmentation masks and intermediate visualizations.
- `outputs/tables/`: JSON summaries for detections and compositing/alignment checks.
- `outputs/run_report.json`: run summary for processed locations/images.

## Assumptions and Limitations
- Monocular depth is relative, not metric.
- Camera intrinsics are assumed from image size and horizontal FOV.
- Ground-plane projection is approximate.
- Segmentation can misclassify road/sidewalk/object regions.
- The detector can miss small, distant, or occluded objects.
- Fixed-angle compositing assumes directional images are evenly spaced.
- Results should be evaluated qualitatively, not as a precise map.

## AI Usage
AI tools were used to help refactor code and polish documentation. The final implementation, outputs, and limitations were reviewed by the project members.
