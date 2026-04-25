# CSE 5509 Computer Vision Final Project
## Improved Single-Camera BEV Driver-Assistance Visualization

This repository contains a **class-aligned CSE 5509 Computer Vision final project** that builds a mock Bird’s-Eye View (BEV) visualization from monocular street images.

The updated pipeline focuses on reproducibility and interpretability:
- dual-mode path setup (**Colab + local repo**)
- dynamic dataset discovery (no stale hard-coded image counts)
- semantic segmentation + cleanup for ground/obstacle separation
- monocular depth estimation for approximate distance reasoning
- instance detection with unique labels (`car1`, `car2`, `person1`, ...)
- geometry-aware BEV projection using an approximate pinhole model
- per-image outputs + location-level 360° stitched compositing

---

## Motivation
The original notebook produced a BEV-like output but mixed assumptions and hard-coded paths, and it lacked robust instance-level markers.

This revision makes the project easier to run, inspect, and explain in CV terms while staying lightweight and using pretrained models.

---

## CSE 5509 topic alignment
The implementation covers multiple course-relevant computer vision methods:

1. **Semantic segmentation / dense prediction**
   - SegFormer (Cityscapes) used to infer pixel-level classes.
2. **Monocular depth estimation / dense prediction**
   - DPT depth model used to estimate per-pixel relative depth.
3. **Object detection + instance segmentation (model output includes masks)**
   - Torchvision Mask R-CNN ResNet50-FPN for `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`.
4. **Camera geometry + pinhole back-projection**
   - Approximate intrinsics estimated from image width and assumed horizontal FOV.
5. **Image warping / inverse perspective intuition for BEV rendering**
   - Ground/object pixels projected into a top-view canvas using depth + camera model.
6. **Failure analysis and diagnostics**
   - Intermediate masks and overlays are saved for debugging and limitations analysis.

---

## Repository layout

```text
cse5509_final_project/
├── data/
│   ├── loc1/ ... loc6/
│   │   └── direction 0.jpg ... direction 7.jpg
├── outputs/                    # Generated after running notebook
│   ├── per_image/
│   ├── per_location/
│   ├── tables/
│   └── diagnostics/
├── bev_pipeline.py             # Modular helper functions
├── final-project-cse5509-v2.ipynb
└── README.md
```

Dataset counts are discovered at runtime and printed automatically.

---

## Input data format
Expected structure per location:
- folder name: `locX`
- image naming: `direction N.jpg` where `N` is typically `0..7`
- fixed-angle assumption: for 8 images, each direction is treated as ~45° apart in stitched compositing

---

## Output structure and examples
After execution, check:

- `outputs/per_image/*_bev.png` — BEV with ego marker, range guides, and instance labels.
- `outputs/per_image/*_det.png` — detection overlay in camera image space.
- `outputs/diagnostics/*_raw_seg.png` — raw segmentation preview.
- `outputs/diagnostics/*_ground_mask.png` — cleaned ground mask.
- `outputs/diagnostics/*_object_mask.png` — cleaned object mask.
- `outputs/tables/*_instances.json` — per-image instance table with class, confidence, bbox, contact point, depth, BEV coordinate.
- `outputs/per_location/*_stitched_bev.png` — stitched location-level BEV composite.

---

## How to run (local)

1. Create and activate a Python environment.
2. Install dependencies (see `requirements.txt`).
3. From repository root, run Jupyter and open notebook:
   - `final-project-cse5509-v2.ipynb`
4. In the config cell:
   - set `RUN_SMALL_DEMO=True` (default) for quick validation
   - set `RUN_SMALL_DEMO=False` for full dataset pass
5. Run all cells.

---

## How to run (Google Colab)

1. Upload or clone this repo to your Colab runtime / Drive.
2. Open `final-project-cse5509-v2.ipynb`.
3. The notebook attempts optional Drive mount if Colab is detected.
4. Ensure the discovered `data/` path is correct in the config output.
5. Run cells top-to-bottom.

---

## Key assumptions and limitations

- **Monocular depth scale ambiguity:** metric distances are approximate.
- **Approximate intrinsics:** focal length derived from assumed horizontal FOV.
- **Ground-plane simplification:** uneven terrain and slopes can degrade BEV accuracy.
- **Detection availability:** if torchvision models are unavailable, the pipeline falls back gracefully and records warnings.
- **Compositing simplification:** stitched 360° BEV currently uses fixed-angle alpha blending; optional feature-based alignment diagnostics can be added later.

This project intentionally avoids overclaiming physical metric precision.

---

## Reproducibility and sanity checks
The notebook includes checks for:
- image existence and loadability
- segmentation/depth shape compatibility
- BEV output shape and save success
- unique per-image instance labels
- empty-safe detection handling
- automatic output directory creation

---

## AI usage note (placeholder)
Add your course-required disclosure here, for example:

> Portions of code structure and documentation were assisted by AI tools, then reviewed and validated by the student authors.

