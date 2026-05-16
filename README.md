# zeon-tubes · Microcentrifuge Tube Detection & Orientation Estimation

A two-stage computer vision system that detects microcentrifuge tube lids in overhead RGB images and estimates their full 360° orientation angle (joint-to-tab direction), evaluated via 5-fold cross-validation on 70 images / 371 tubes.

---

## Results

### Detection — YOLOv11n-OBB

| Metric | Value |
|---|---|
| Precision | **0.997** |
| Recall | **1.000** |
| F1 | **0.999** |
| TP / FP / FN | 371 / 1 / 0 |

### Angle Estimation — Stage 2 Isolated (GT crops)

| Metric | Value |
|---|---|
| MAE | **2.5° ± 0.1°** |
| Median | **2.0° ± 0.2°** |
| Within 15° | **100.0% ± 0.0%** |
| 180° flip rate | **0.0%** |

### End-to-End (YOLO crops → SAM → angle head)

| Metric | Value |
|---|---|
| Angle MAE | **4.1°** |
| Angle Median | **2.3°** |
| Within 15° | **99.2%** |
| 180° flip rate | **0.8%** |

> The 1.6° gap between isolated and end-to-end MAE is entirely explained by
> YOLO localisation imprecision: 3 extra flip errors (0.8% of 371 tubes),
> each contributing ~180° to the mean. Non-flip MAE end-to-end is ~2.6°,
> statistically identical to the isolated result.

---

## Methodology

### Pipeline

```
Image
  │
  ▼
┌──────────────────────────────┐
│  Stage 1 — YOLOv11n-OBB     │  Detects tube lids as oriented bounding
│  Detection                   │  boxes → centre (cx, cy) + box dims
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  SAM Masking                 │  Segments the tube lid; zeroes out rack
│                              │  and background pixels before cropping
└─────────────┬────────────────┘
              │  128×128 masked crop per detection
              ▼
┌──────────────────────────────┐
│  Stage 2 — Binned + Offset   │  36-bin classification + intra-bin
│  Angle Estimation            │  offset regression → θ ∈ [0°, 360°)
└──────────────────────────────┘
```

### Stage 1 — YOLOv11n-OBB Detection

`yolo11n-obb` pretrained on COCO, fine-tuned with 5-fold CV (56 train / 14 val images per fold, 60 epochs). Annotated as 4-corner OBB polygons normalised to [0, 1]. Augmentation: H/V flip, HSV jitter, mosaic, copy-paste.

Oriented bounding boxes provide the detected centre coordinates and box dimensions used to prompt SAM and extract the angle head crop. The OBB axis (0–180°) is discarded — full 360° angle comes from Stage 2.

### SAM Masking

Before passing a crop to the angle head, SAM (Segment Anything, ViT-Base) segments the tube lid from the full image using the YOLO box as a prompt. Non-tube pixels are replaced with neutral grey (128). This step is applied at both **train and inference time**, ensuring the angle head never sees rack geometry or background texture.

Without consistent masking, Model D degraded from 2.5° isolated MAE to 23.1° end-to-end due to train/test distribution mismatch caused by rack interference. Applying masking at both stages reduced end-to-end MAE to 4.1° — an 82% reduction.

### Stage 2 — Binned + Offset Angle Head (Model D)

**Architecture:**

```
masked crop (128×128)
  │
  ▼
MobileNetV3-small (ImageNet pretrained, num_classes=0)
  │
  └── 1024-d feature vector
        │
   ┌────┴──────────────────┐
   ▼                       ▼
cls head                reg head
(1024 → 36)             (1024 → 36)
36-bin logits           per-bin offset
   │                       │
   └──────────┬────────────┘
              ▼
    bin × 10° + offset × 10° = θ̂ ∈ [0°, 360°)
```

**Loss:** `CrossEntropy(logits, bin_idx) + MSE(offset_pred, offset_gt)`

**Training:** AdamW lr=3e-4, CosineAnnealingLR, 60 epochs per fold, best checkpoint by val MAE.

**Augmentation (geometry-correct — angle label updated with each transform):**

| Transform | Label update |
|---|---|
| Horizontal flip | `(180 − θ) % 360` |
| Vertical flip | `(360 − θ) % 360` |
| Rotate 90° × k | `(θ + 90k) % 360` |
| Brightness / contrast / noise | unchanged |

**Why binned classification outperforms direct regression:**

Representing angle as a single continuous value (even with sin/cos encoding) requires the regression head to model the full circular output space. The binned approach decomposes this: a 36-class classifier handles coarse direction (each bin covers 10°), and the regression head only needs to fit ±5° within the winning bin — a far simpler sub-problem. This decomposition produced a 74% MAE reduction over sin/cos regression (2.5° vs 9.8°).

---

## What Was Tried

Six angle estimation approaches were benchmarked on the same 5-fold CV split before selecting Model D:

| Model | MAE | Key finding |
|---|---|---|
| A — MobileNetV3 + sin/cos | 9.8° | Correct representation, weaker than binned |
| B — EfficientNetV2-S + sin/cos | 16.0° | Overfits on 370 crops |
| C — DINOv2 frozen + MLP | 68.0° | Semantic features blind to sub-mm geometry |
| **D — Binned + offset** | **2.5°** ✅ | Best across all metrics |
| E — Two-stage axis + flip classifier | 9.0° | Flip classifier unstable at this dataset size |
| F — TTA × 8 on Model D | ~7.0° | Averaging hurt stable predictions |

The DINOv2 result is the most instructive failure: distinguishing the joint from the tab requires recognising a 2–3 pixel asymmetric texture feature. DINOv2 was trained to be invariant to exactly this kind of local variation. Rich semantic pretraining is not universally better — domain specificity matters more than model scale here.

---

## Error Analysis

**Stage 2 in isolation** achieves 100.0% within 15° across all five folds with zero flips. Every angle estimation error on a well-cropped, well-masked lid is below 15°, and fold-to-fold variance is negligible (±0.1° MAE).

**End-to-end** introduces 3 additional flip errors from imprecise YOLO crops shifting the SAM mask slightly. These 3 tubes account for the entire 1.6° gap between isolated and end-to-end MAE. The remaining 368 tubes have a non-flip MAE of ~2.6°.

**The impact of SAM masking:**

| Setup | MAE | Within 15° | Flips |
|---|---|---|---|
| Stage 2, GT crops, no masking | 3.2° | 99.5% | 0.3% |
| Stage 2, GT crops, with masking | 2.5° | 100.0% | 0.0% |
| End-to-end, no masking | 23.1° | 85.4% | 8.6% |
| **End-to-end, with masking** | **4.1°** | **99.2%** | **0.8%** |

Masking improved both isolated and end-to-end performance. The dramatic end-to-end improvement (23.1° → 4.1°) confirms that rack interference — not model capacity or data scarcity — was the dominant failure mode.

---

## Notebook

`final_notebook.ipynb` runs the full pipeline end-to-end in Google Colab.

| Cell | Content | Time (T4) |
|---|---|---|
| 1 | Installs, constants, shared utilities | 2 min |
| 1b | SAM model loading + masking helper | 1 min |
| 1c | Precompute SAM-masked crops for all 371 annotations | 5–10 min |
| 2 | Train 5 YOLOv11n-OBB folds, evaluate detection P/R/F1 | 30–50 min |
| 3 | Train 5 Model D folds on masked crops, evaluate angle MAE | 15–30 min |
| 4 | End-to-end: YOLO → SAM → Model D, joint metrics | 5 min |
| 5 | Final report: tables, error distribution, worst-case grid | 3 min |
| 6 | Save weights, CSVs, and plots to Google Drive | 1 min |

**Runtime → T4 GPU. Total: ~60–100 minutes.**

### Setup

```bash
pip install ultralytics timm scikit-learn transformers
```

Place the dataset at `MyDrive/zeon-tubes/Dataset/` containing `images/` and `annotations.csv`. All outputs are written to `MyDrive/zeon-tubes/final_results/`.

### Dataset

70 overhead RGB images (640×480), 3–6 tubes per image, across white, black, desk, and mixed surfaces. 371 total tube annotations with centre coordinates and full 360° joint-to-tab angle.

Available at: [Google Drive](https://drive.google.com/drive/folders/19XdosmtFivQ2mUODFSQgRPGRAqvVsDnI?usp=sharing)

---

## Next Steps

**1. Keypoint annotation** — Re-annotate the 371 tubes with explicit joint `(jx, jy)` and tab `(tx, ty)` coordinates. YOLOv11-Pose with a 2-keypoint head gives the joint→tab vector directly, architecturally eliminating the 180° flip problem without a disambiguation step.

**2. Rack-aware augmentation** — Composite tube crops over rack-hole textures during training, forcing the angle head to ignore rack boundaries. Targets the dominant remaining failure mode without additional annotation effort.

**3. 3D rendering** — Model the tube geometry once in Blender, render at any exact angle under randomised lighting and backgrounds. Provides perfect ground truth at scale — far more tractable than generative models for this domain.

**4. Equivariant backbone** — Replace MobileNetV3 with an E(2)-steerable CNN. Rotation equivariance is guaranteed by construction rather than learned through augmentation, improving data efficiency — relevant given only 371 training crops.

---

## AI Usage

Claude (Anthropic) was used to design the pipeline architecture, generate and debug training code, and structure the cross-validation framework. All analysis, result interpretation, and architectural decisions are the author's own.

---

## License

MIT
