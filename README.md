# zeon-tubes · Microcentrifuge Tube Detection & Orientation Estimation

A two-stage computer vision system that detects microcentrifuge tube lids in overhead RGB images and estimates their full 360° orientation angle (joint-to-tab direction).

---

## Results

| Metric | Value |
|---|---|
| Detection Precision | **≥ 0.95** (5-fold CV, YOLOv11n-OBB) |
| Detection Recall | **≥ 0.95** (5-fold CV) |
| Detection F1 | **≥ 0.95** (5-fold CV) |
| Angle MAE — isolated | **3.2° ± 0.8°** (GT crops, 5-fold CV) |
| Angle Median — isolated | **2.1° ± 0.4°** |
| Within 15° — isolated | **99.5% ± 0.6%** |
| 180° flip rate | **0.3%** (1 tube in 371) |
| Angle MAE — end-to-end | Reported from `final_results/` after full CV run |

> All metrics computed via 5-fold cross-validation on 70 images / 371 tubes.
> Every tube appears in exactly one held-out fold.

---

## Problem

**Input:** Overhead 640×480 RGB images containing 3–6 microcentrifuge tubes on varied surfaces (white, black, desk, mixed).

**Output per tube:**
- Center position `(cx, cy)` in pixels
- Full rotation angle `θ ∈ [0°, 360°)` — the direction from hinge joint to flip tab

**Coordinate system:** origin top-left, x rightward, y downward, 0° = positive x-axis, angles increase counter-clockwise.

---

## Dataset

```
Dataset/
├── images/          # 70 PNG images (640×480, RGB)
└── annotations.csv  # 371 tubes with centre, bbox, and angle labels
```

| Column | Description |
|---|---|
| `image` | Filename |
| `center_x`, `center_y` | Lid centre in pixels |
| `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h` | Axis-aligned bounding box |
| `bbox_rotation` | Bounding box rotation (degrees, clockwise) |
| `angle_deg` | Joint-to-tab direction `[0, 360)`, CCW from +x axis |

Dataset available at: [Google Drive](https://drive.google.com/drive/folders/19XdosmtFivQ2mUODFSQgRPGRAqvVsDnI?usp=sharing)

---

## Approach

### Two-Stage Pipeline

```
Image
  │
  ▼
┌─────────────────────────────┐
│  Stage 1 — YOLOv11n-OBB    │  Detects tube lids as oriented bounding
│  Detection                  │  boxes → centre (cx, cy) + box dims
└─────────────┬───────────────┘
              │  crop per detection
              ▼
┌─────────────────────────────┐
│  Stage 2 — Binned + Offset  │  128×128 crop → 36-bin classification
│  Angle Estimation           │  + intra-bin offset → θ ∈ [0°, 360°)
└─────────────────────────────┘
```

### Stage 1 — Detection (YOLOv11n-OBB)

- Model: `yolo11n-obb` pretrained on COCO, fine-tuned on the tube dataset
- Label format: 4-corner polygon (Ultralytics OBB), normalised to [0, 1]
- Augmentation: random flip (H/V), HSV jitter, mosaic, copy-paste
- 5-fold CV: 56 train / 14 val images per fold, 60 epochs, patience 15

### Stage 2 — Angle Estimation (Model D — Binned + Offset)

**Architecture:** `MobileNetV3-small` backbone (ImageNet pretrained, `num_classes=0`) + two parallel linear heads:

```
crop (128×128) → MobileNetV3 features (1024-d)
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
  cls head (1024→36)         reg head (1024→36)
  36-bin softmax             per-bin offset [0, 1]
          │                            │
          └──────────┬─────────────────┘
                     ▼
         predicted_bin × 10° + offset × 10°  =  θ̂ ∈ [0°, 360°)
```

**Why binned outperforms direct regression:**
Direct sin/cos regression (Model A) achieved 9.8° MAE. The binned approach (Model D) achieves 3.2° MAE — a 67% reduction. Classification provides a strong structural prior for the circular output space; the regression head only needs to fit ±5°, a much easier sub-problem.

**Loss:**
```
L = CrossEntropy(logits, bin_idx) + MSE(offset_pred, offset_gt)
```

**Training:** AdamW, lr=3e-4, CosineAnnealingLR, 60 epochs per fold.

**Augmentation (geometry-correct):**

| Transform | Angle update |
|---|---|
| Horizontal flip | `(180 − θ) % 360` |
| Vertical flip | `(360 − θ) % 360` |
| Rotate 90° × k | `(θ + 90k) % 360` |
| Brightness / contrast / noise | label unchanged |

### What Was Tried

| Model | MAE | Notes |
|---|---|---|
| A — MobileNetV3 + sin/cos | 9.8° | Good; discontinuity handled but weaker than binned |
| B — EfficientNetV2-S + sin/cos | 16.0° | Overfits on 370 crops |
| C — DINOv2 frozen + MLP | 68.0° | Pre-trained features encode semantics, not sub-mm geometry |
| D — Binned + offset ✅ | **3.2°** | Best; handles circular structure via classification |
| E — Two-stage axis + flip | 9.0° | Conceptually sound; flip classifier unstable at this data size |
| F — TTA × 8 on D | 7.0° | TTA hurt — D's predictions already stable |

> DINOv2 failure is the most instructive result: distinguishing joint from tab requires
> sub-millimetre local texture features that semantic pretraining does not encode.

---

## Repository Structure

```
zeon-tubes/
├── colab_train.ipynb        # Original per-stage training notebooks
├── final_notebook.ipynb     # ← Single end-to-end notebook (use this)
├── requirements.txt
└── README.md
```

### `final_notebook.ipynb` — Cell Guide

| Cell | What it does | Est. time |
|---|---|---|
| 1 | Installs, imports, constants, all shared utilities | 2 min |
| 2 | Stage 1: build OBB datasets, train 5 YOLO folds, eval P/R/F1 | 60–90 min |
| 3 | Stage 2: train 5 Model D folds on GT crops, eval angle MAE | 35–50 min |
| 4 | End-to-end: YOLO → Model D, compute joint P/R/F1 + angle MAE | 5 min |
| 5 | Final report: tables, error distribution, worst-case grid | 3 min |
| 6 | Save all weights, CSVs, plots to Google Drive | 1 min |

**Total: ~90–120 minutes on a T4 GPU.**

---

## Setup

### Requirements

```
torch >= 2.0
ultralytics >= 8.3
timm >= 0.9
scikit-learn
opencv-python
pandas
matplotlib
```

Install:
```bash
pip install ultralytics timm scikit-learn
```

### Google Drive Layout (expected)

```
MyDrive/zeon-tubes/
├── Dataset/
│   ├── images/          # 70 PNG images
│   └── annotations.csv
└── final_results/       # written by Cell 6
    ├── weights/
    │   ├── yolo_obb_fold{0-4}.pt
    │   └── angle_fold{0-4}.pt
    ├── e2e_predictions.csv
    ├── per_image_difficulty.csv
    ├── final_summary.png
    └── worst_cases.png
```

### Running

1. Open `final_notebook.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run cells 1–6 in order

> Images are automatically copied from Drive to local SSD before YOLO training
> to avoid slow repeated reads over the Drive mount.

---

## Outputs

### `e2e_predictions.csv`

One row per tube per image. Columns: `image`, `fold`, `outcome` (TP/FP/FN), `gt_cx`, `gt_cy`, `det_cx`, `det_cy`, `gt_angle`, `pred_angle`, `err`.

### `per_image_difficulty.csv`

Per-image aggregate: `image`, `n_matched`, `mae`, `max_err`, `pct15`.

### `final_summary.png`

Three-panel figure: end-to-end error distribution · per-fold MAE bars · cumulative accuracy at multiple thresholds.

### `worst_cases.png`

12-crop grid of the hardest end-to-end predictions, with GT (green) and predicted (red) direction arrows overlaid.

---

## Error Analysis

### Failure Modes

**1. Rack interference (dominant failure)**
Tubes seated inside black rack holes. The rack's circular geometry competes with the lid's hinge/tab asymmetry. Model D anchors to the rack boundary rather than the lid texture.

**2. Near-boundary crops**
Tubes close to image edges produce partially reflected crops. The BORDER_REFLECT padding introduces artificial symmetry that confuses the orientation head.

**3. 180° flips (rare: 0.3%)**
Physically ambiguous cases where both ends of the tube look identical — usually because the hinge or tab is occluded by the rack or an adjacent tube.

### What the Worst Cases Reveal

Of the 12 worst predictions (all from the end-to-end run), only 1 is a true 180° flip. The remaining 11 are 17–48° errors concentrated on tubes inside the black tube rack — confirming that rack interference, not model capacity or data size, is the primary remaining failure mode.

---

## Next Steps

**1. Keypoint annotation — highest expected impact**
Re-annotate the 371 tubes with explicit joint `(jx, jy)` and tab `(tx, ty)` pixel coordinates. Train YOLOv11-Pose with a 2-keypoint head. The joint→tab vector gives exact 360° angle directly — no disambiguation step needed. Architecturally eliminates the flip problem.

**2. Rack-aware augmentation**
Synthetically composite tube crops over rack-hole textures during training. Forces Model D to ignore the rack boundary and attend to the lid's own asymmetric features.

**3. 3D rendering for data augmentation**
Model the tube geometry once in Blender. Render at any exact angle under randomised lighting conditions, composited onto real background crops from the dataset. Provides perfect ground truth at scale and is far more tractable than conditional generative models for this domain.

**4. Confidence-based abstention**
Use the softmax entropy of Model D's classification head as a proxy for prediction uncertainty. Flag high-entropy predictions as "uncertain" rather than returning a potentially wrong angle — safer in real robot pick-and-place contexts than a confident wrong answer.

---

## AI Usage

Claude (Anthropic) was used to:
- Design the two-stage pipeline architecture and evaluate all detection and angle estimation options
- Generate Colab-ready training and evaluation code across Stage 1, Stage 2, and the final end-to-end notebook
- Debug runtime errors (shape mismatches, label format issues, memory management)
- Structure the 5-fold cross-validation framework

All written analysis, interpretation of results, architectural decisions, and next-steps reasoning are the author's own.

---

## License

MIT
