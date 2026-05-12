"""Evaluation CLI: predictions CSV + GT CSV → metric table.

Usage:
    python -m src.evaluate --pred outputs/preds.csv --gt Dataset/annotations.csv
    python -m src.evaluate --pred Dataset/annotations.csv --gt Dataset/annotations.csv  # sanity
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataio import load_annotations, group_by_image
from .metrics import evaluate_dataset
from .viz import overlay_pred_vs_gt, worst_n_collage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Predictions CSV (same schema as annotations.csv).")
    ap.add_argument("--gt", required=True, help="Ground-truth CSV.")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching.")
    ap.add_argument("--viz-dir", default=None, help="Optional output dir for per-image overlays.")
    ap.add_argument("--images-dir", default=None, help="Image dir, required if --viz-dir is set.")
    ap.add_argument("--collage", default=None, help="Optional output path for worst-N angle-error collage.")
    args = ap.parse_args()

    preds = load_annotations(args.pred)
    gts = load_annotations(args.gt)
    preds_by_image = group_by_image(preds)
    gts_by_image = group_by_image(gts)

    metrics, per_image = evaluate_dataset(preds_by_image, gts_by_image, iou_thr=args.iou)
    print(metrics.as_table())

    if args.viz_dir:
        assert args.images_dir, "--images-dir required with --viz-dir"
        out_root = Path(args.viz_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        for img in sorted(set(preds_by_image) | set(gts_by_image)):
            src = Path(args.images_dir) / img
            if not src.exists():
                continue
            overlay_pred_vs_gt(
                str(src),
                preds_by_image.get(img, []),
                gts_by_image.get(img, []),
                str(out_root / f"viz_{img}"),
            )
        print(f"per-image overlays → {out_root}")

    if args.collage:
        assert args.images_dir, "--images-dir required with --collage"
        worst_n_collage(preds_by_image, gts_by_image, per_image, args.images_dir, args.collage, n=12)
        print(f"worst-N collage → {args.collage}")


if __name__ == "__main__":
    main()
