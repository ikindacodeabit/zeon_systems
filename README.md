# zeon_systems — Tube Detection & Orientation

Detect microcentrifuge tube positions and orientations in overhead RGB images. Take-home assignment for Zeon Systems.

## Approach

Two-stage hybrid system:

1. **Detection** — Fine-tuned **YOLOv11n** finds cap centers (axis-aligned bbox).
2. **Angle** — Small CNN (MobileNetV3-Small) regresses `(sin θ, cos θ)` per cap crop → recovers the full **0–360°** joint→tab direction.

A **classical CV baseline** (HoughCircles + PCA + tab-side disambiguation via convexity defects) is included for comparison and analysis.

The two-stage design is chosen because every off-the-shelf rotated-detector (YOLO-OBB, Oriented R-CNN, ...) predicts angle **mod 180°** (rectangles are 180°-symmetric), while the labels are **0–360°**. The angle CNN trained with a circular loss handles this natively.

## Repo layout

```
src/
├── dataio.py        # CSV parsing, OBB→AABB, 5-fold splits, YOLO label export
├── metrics.py       # IoU, Hungarian, P/R/F1, circular angle error
├── viz.py           # overlay predictions/GT, failure-mode collage
├── classical.py     # CV baseline
├── train_detector.py
├── train_angle.py
├── infer.py         # end-to-end inference → predictions CSV + annotated PNG
└── evaluate.py      # predictions CSV + GT CSV → metric table
notebooks/colab_train.ipynb
report/report.md
```

## Reproducing results

```bash
pip install -r requirements.txt

# Sanity check eval harness with GT-as-predictions
python -m src.evaluate --pred data/annotations.csv --gt data/annotations.csv

# Classical baseline over the whole dataset
python -m src.classical --images data/images --out outputs/classical_preds.csv
python -m src.evaluate --pred outputs/classical_preds.csv --gt data/annotations.csv

# Two-stage system: train on Colab via notebooks/colab_train.ipynb
# Then download weights to outputs/ and run inference locally:
python -m src.infer --images data/images --weights outputs/ --out outputs/hybrid_preds.csv
python -m src.evaluate --pred outputs/hybrid_preds.csv --gt data/annotations.csv
```

## Results

| System | Precision | Recall | F1 | Mean angle err (°) | Median angle err (°) | Within 10° | Within 30° |
|---|---|---|---|---|---|---|---|
| Classical baseline | 0.393 | 0.445 | 0.417 | 107.8 | 107.9 | 9.7% | 14.5% |
| Two-stage (oracle + angle, 5-fold CV) | 1.000* | 1.000* | 1.000* | 46.4 | 31.5 | 18.1% | 48.0% |
| Two-stage (oracle + angle, single-fold 50 ep) | 1.000* | 1.000* | 1.000* | 38.0 | 22.0 | 21.4% | — |

\* Oracle = GT centers as detector; isolates angle quality. End-to-end YOLO+angle metrics are produced by `notebooks/colab_train.ipynb`. See `report/report.md` for full analysis.

## Dataset

`Dataset/` (gitignored) contains 70 PNG images (640×480) and `annotations.csv` with 371 oriented tube annotations.
