"""Train the YOLOv11n cap detector via Ultralytics.

Designed for Colab/Kaggle (free GPU). On CPU/MPS this will still run but
takes much longer; we recommend the Colab notebook for the full schedule.

Workflow:
  1. Build a YOLO-format dataset folder for one train/val split (image-level).
  2. Call `YOLO("yolo11n.pt").train(...)` with rotation-friendly augmentation.
  3. Save best.pt to the requested output path.

Run:
    python -m src.train_detector --images Dataset/images --ann Dataset/annotations.csv \
        --out outputs/detector_fold0.pt --val-fold 0 --epochs 80
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataio import (
    load_annotations, group_by_image, kfold_image_splits, make_yolo_dataset,
)


def train(
    images_dir: str,
    ann_csv: str,
    out_path: str,
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    val_fold: int = 0,
    seed: int = 0,
    workdir: str = "outputs/yolo_run",
) -> str:
    # Late import so the rest of the package doesn't require ultralytics.
    from ultralytics import YOLO

    by_image = group_by_image(load_annotations(ann_csv))
    image_names = sorted(by_image.keys())
    folds = kfold_image_splits(image_names, k=5, seed=seed)
    train_imgs, val_imgs = folds[val_fold]

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    yaml_path = make_yolo_dataset(images_dir, by_image, train_imgs, val_imgs, workdir / "dataset")

    model = YOLO("yolo11n.pt")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,  # auto-pick CUDA GPU; ignore if absent
        project=str(workdir),
        name="train",
        exist_ok=True,
        # Augmentation: heavy rotation is safe because GT angles are ~uniform.
        degrees=180,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        translate=0.05,
        scale=0.2,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        # Detection setup
        single_cls=True,
        lr0=1e-3,
        cos_lr=True,
        seed=seed,
        verbose=True,
    )
    # `results` is a Results object; the best weights live under save_dir/weights/best.pt
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(best.read_bytes())
        print(f"copied {best} → {out}")
    return str(best)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workdir", default="outputs/yolo_run")
    args = ap.parse_args()
    train(
        args.images, args.ann, args.out,
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        val_fold=args.val_fold, seed=args.seed, workdir=args.workdir,
    )


if __name__ == "__main__":
    main()
