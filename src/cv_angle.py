"""5-fold cross-validation for the angle head.

For each fold:
  - Train the angle head on the 4 training folds.
  - Predict held-out fold using the oracle detector (GT centers) so the metric
    isolates angle quality.
  - Concatenate held-out predictions across folds → full-dataset predictions
    where every prediction was made by a model that did NOT see that image.

The output CSV can then be fed into src/evaluate.py for an honest report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from .dataio import (
    Tube, load_annotations, group_by_image, kfold_image_splits, write_predictions_csv,
)
from .train_angle import train as train_angle
from .infer import AnglePredictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--weights-dir", default="outputs/angle_cv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    Path(args.weights_dir).mkdir(parents=True, exist_ok=True)

    by_image = group_by_image(load_annotations(args.ann))
    image_names = sorted(by_image.keys())
    folds = kfold_image_splits(image_names, k=5, seed=args.seed)

    all_preds: list[Tube] = []
    fold_metrics: list[dict] = []

    for fi in range(5):
        train_imgs, val_imgs = folds[fi]
        weights_path = str(Path(args.weights_dir) / f"angle_fold{fi}.pt")
        print(f"\n=== fold {fi}: train={len(train_imgs)} val={len(val_imgs)} ===")
        res = train_angle(
            args.images, args.ann, weights_path,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            val_fold=fi, seed=args.seed,
        )
        fold_metrics.append({"fold": fi, "best_val_mean": res.val_mean_err, "final_val_med": res.val_med_err})

        predictor = AnglePredictor(weights_path, tta=args.tta)
        for img_name in val_imgs:
            img_path = str(Path(args.images) / img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            gts = by_image[img_name]
            angles = predictor.predict(img, [(t.center_x, t.center_y) for t in gts])
            for t, a in zip(gts, angles):
                all_preds.append(Tube(
                    image=t.image, center_x=t.center_x, center_y=t.center_y,
                    bbox_x=t.bbox_x, bbox_y=t.bbox_y, bbox_w=t.bbox_w, bbox_h=t.bbox_h,
                    bbox_rotation=t.bbox_rotation, angle_deg=a,
                ))
        del predictor

    write_predictions_csv(args.out_csv, all_preds)
    print(f"\nwrote {len(all_preds)} predictions → {args.out_csv}")
    print(json.dumps(fold_metrics, indent=2))


if __name__ == "__main__":
    main()
