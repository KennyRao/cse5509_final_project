# CSE 5509 Final Project: Ego-Centered 360° Minimap from Single-Camera Views

## 1) Overview
This repository implements a qualitative course-project pipeline that builds an ego-centered 360° minimap from single-camera images captured at a fixed location while rotating in place. The primary output is an **object-level minimap** generated from projected detection rows. A separate **stitched BEV** image is also generated as a **pixel-level diagnostic** from rotated per-image BEV rasters. The system is intended for explainable visualization and demo discussion, not calibrated autonomous-driving mapping.

## 2) Course requirement alignment
This project uses multiple CV tasks from class:
- semantic segmentation (SegFormer),
- monocular depth estimation (DPT),
- object detection (Mask R-CNN, optional Grounding DINO),
- geometric projection into an ego-centered frame,
- optional feature-matching diagnostics (ORB) for alignment checks.

## 3) Dataset format
Expected layout:

```text
repo_root/
  data/
    loc1/
      direction 0.jpg
      ...
      direction 7.jpg
    loc2/
      ...
```

The pipeline expects `repo_root/data/` (not `data/data/`).

## 4) Direction convention (clockwise)
- `direction 0` = forward/up on minimap.
- `direction 1` = up-right (+45° clockwise).
- `direction 2` = right (+90° clockwise).
- ...
- `direction 7` = up-left (+315° clockwise).

## 5) Pipeline overview
1. Load RGB image.
2. Run SegFormer for scene masks (diagnostic).
3. Run DPT monocular depth (approximate; treated as inverse depth by default).
4. Run Mask R-CNN (+ optional Grounding DINO for open-vocabulary classes like `dumpster`, `road_sign`).
5. Use bbox bottom-center as a ground-contact proxy.
6. Project detections to ego meters using FOV + depth approximation.
7. Apply image-space class-aware NMS, then location-level heuristic deduplication in ego-meter space.
8. Render final minimap from deduplicated rows.
9. Render stitched BEV diagnostic from rotated per-image BEV pixels.

## 6) Run in Google Colab
1. Open `final-project-cse5509-v2.ipynb` in Colab.
2. Install dependencies if needed.
3. Mount Drive when prompted.
4. Set `PROJECT_DIR` in the notebook.
5. Run cells top-to-bottom.

## 7) Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```
Then open `final-project-cse5509-v2.ipynb` and run top-to-bottom.

## 8) Configuration notes
- Balanced demo defaults:
  - `detection_threshold = 0.70`
  - `minimap_min_confidence = 0.70`
- Recommended tuning behavior:
  - `0.75` is cleaner but may miss objects,
  - `0.65` may recover more objects but should be visually checked for false positives.
- Zero-shot candidate thresholds (default):
  - `zero_shot_box_threshold = 0.35`
  - `zero_shot_text_threshold = 0.30`
- FOV is configurable (`horizontal_fov_deg`) and affects projection geometry.
- Demo mode:
  - `run_small_demo=True`, `demo_locations=1`, `demo_images_per_location=None` uses all images in first location.

## 9) Outputs
- Per-image detection overlays: `outputs/per_image/*_det.png`
- Per-image BEV diagnostics: `outputs/per_image/*_bev.png`
- Final minimap (primary output): `outputs/per_location/loc*_minimap.png`
- Stitched BEV (diagnostic): `outputs/per_location/loc*_stitched_bev.png`
- Tables/JSON/CSV:
  - `outputs/tables/loc*_minimap_instances.json`
  - `outputs/tables/loc*_minimap_instances.csv`
  - `outputs/tables/all_minimap_instances.csv`
  - `outputs/tables/loc*_dedup_diagnostics.json`
  - `outputs/tables/loc*_stitch_diagnostics.json`
- Run summary: `outputs/run_report.json`

## 10) How to interpret the minimap
- Center marker = ego position.
- Rings = approximate distance intervals.
- Spokes/labels indicate direction convention.
- Colored markers = object class.
- Labels (e.g., `car1`) are unique after deduplication/relabeling.

## 11) Minimap vs stitched BEV
- **Minimap (final)**: object-level, generated from projected detection rows, deduplicated and labeled.
- **Stitched BEV (diagnostic)**: pixel-level, generated from rotated per-image BEV diagnostic images; does not directly consume final minimap table rows.

## 12) Assumptions and limitations
- Monocular depth is approximate and not calibrated metric depth.
- DPT normalization is mapped to pseudo-metric distance via configured bounds.
- Object projection uses bbox bottom-center as ground-contact proxy.
- FOV is approximate unless calibrated for the capture camera.
- Deduplication is heuristic and reduces repeats; it does not prove object identity.
- False positives/missed detections can occur.

## 13) AI usage statement
AI tools were used for code cleanup, docstring/documentation drafting, and notebook organization. Final claims and technical limitations were reviewed and kept consistent with a qualitative course-project scope.

## 14) Troubleshooting
- **Missing `data/`**: verify dataset is in `repo_root/data/loc*/direction*.jpg`.
- **Colab path issue**: check `PROJECT_DIR` points to folder containing `bev_pipeline.py`.
- **Grounding DINO unavailable**: pipeline continues with Mask R-CNN-only detections.
- **Slow model loading**: first run may download model weights.
- **No CUDA**: CPU fallback is supported but slower.
