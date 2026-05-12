"""End-to-end inference: image → list of (center, bbox, angle).

The detector is pluggable: GT-as-detector (oracle for angle-only evaluation),
classical baseline, or a trained YOLOv11n. Same angle head is reused in all
configurations.

Usages:

    # Evaluate angle head in isolation using GT centers as the "detector":
    python -m src.infer --mode oracle --ann Dataset/annotations.csv \
        --images Dataset/images --angle outputs/angle_fold0.pt \
        --out outputs/hybrid_oracle.csv

    # Use classical detector + learned angle head:
    python -m src.infer --mode classical --images Dataset/images \
        --angle outputs/angle_fold0.pt --out outputs/hybrid_classical.csv

    # Use trained YOLOv11n + learned angle head:
    python -m src.infer --mode yolo --images Dataset/images \
        --yolo outputs/detector.pt --angle outputs/angle_fold0.pt \
        --out outputs/hybrid_yolo.csv

    # Single-image demo with annotated PNG output:
    python -m src.infer --mode yolo --image one.png --yolo det.pt --angle a.pt \
        --viz-out one_pred.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from .classical import detect_one_image as classical_detect
from .dataio import Tube, load_annotations, group_by_image, write_predictions_csv
from .train_angle import AngleHead, CROP_SIZE, _rotate_crop, pick_device


def _normalize_crop(crop: np.ndarray) -> np.ndarray:
    """BGR uint8 patch → normalized RGB float tensor input."""
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    crop = (crop - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return crop.transpose(2, 0, 1)


class AnglePredictor:
    def __init__(self, weights_path: str, tta: bool = True, device: torch.device | None = None):
        self.device = device or pick_device()
        self.model = AngleHead().to(self.device)
        state = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.tta = tta

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray, centers: list[tuple[float, float]]) -> list[float]:
        """Return predicted angle (deg, [0,360)) for each center."""
        if not centers:
            return []
        # Test-time augmentation: predict on N rotations of each crop, average
        # on the unit circle, then un-rotate.
        alphas = [0.0, 90.0, 180.0, 270.0] if self.tta else [0.0]

        crops_all = []  # shape: [n_centers * n_alphas, 3, H, W]
        for cx, cy in centers:
            for a in alphas:
                crop = _rotate_crop(img_bgr, cx, cy, a, CROP_SIZE)
                crops_all.append(_normalize_crop(crop))
        x = torch.tensor(np.stack(crops_all), dtype=torch.float32, device=self.device)
        pred = self.model(x).cpu().numpy()  # (n_centers * n_alphas, 2)  unit (sin, cos)

        out_angles: list[float] = []
        per = len(alphas)
        for i in range(len(centers)):
            block = pred[i * per:(i + 1) * per]  # (per, 2)
            # Un-rotate each prediction: model saw image rotated by α, so
            # predicted angle in world coords is (model_output_angle - α).
            sins, coss = [], []
            for k, a in enumerate(alphas):
                sin_p, cos_p = float(block[k, 0]), float(block[k, 1])
                model_angle = math.degrees(math.atan2(sin_p, cos_p)) % 360.0
                world = (model_angle - a) % 360.0
                sins.append(math.sin(math.radians(world)))
                coss.append(math.cos(math.radians(world)))
            # Average on unit circle
            ang = math.degrees(math.atan2(float(np.mean(sins)), float(np.mean(coss)))) % 360.0
            out_angles.append(ang)
        return out_angles


def _oracle_tubes(gt_by_image: dict[str, list[Tube]], image_name: str) -> list[Tube]:
    """Oracle "detector": return GT tubes (centers + bboxes) with angle=0 (to be
    replaced by the angle head)."""
    return [
        Tube(
            image=t.image, center_x=t.center_x, center_y=t.center_y,
            bbox_x=t.bbox_x, bbox_y=t.bbox_y, bbox_w=t.bbox_w, bbox_h=t.bbox_h,
            bbox_rotation=t.bbox_rotation, angle_deg=0.0,
        )
        for t in gt_by_image.get(image_name, [])
    ]


def _yolo_detect(yolo_model, img_path: str, image_name: str) -> list[Tube]:
    results = yolo_model.predict(source=img_path, verbose=False, imgsz=640, conf=0.25)
    tubes: list[Tube] = []
    for r in results:
        boxes = r.boxes
        if boxes is None or boxes.xyxy is None:
            continue
        for xyxy, conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
            x0, y0, x1, y1 = map(float, xyxy)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            tubes.append(Tube(
                image=image_name,
                center_x=cx, center_y=cy,
                bbox_x=x0, bbox_y=y0,
                bbox_w=x1 - x0, bbox_h=y1 - y0,
                bbox_rotation=0.0,
                angle_deg=0.0,
            ))
    return tubes


def run(
    mode: str,
    images_dir: str | None,
    image_path: str | None,
    ann_csv: str | None,
    yolo_weights: str | None,
    angle_weights: str,
    out_csv: str | None,
    viz_out: str | None,
    tta: bool = True,
) -> list[Tube]:
    angle_pred = AnglePredictor(angle_weights, tta=tta)

    yolo_model = None
    if mode == "yolo":
        from ultralytics import YOLO
        assert yolo_weights, "--yolo required for mode=yolo"
        yolo_model = YOLO(yolo_weights)

    gt_by_image: dict[str, list[Tube]] = {}
    if mode == "oracle":
        assert ann_csv, "--ann required for mode=oracle"
        gt_by_image = group_by_image(load_annotations(ann_csv))

    targets: list[str]
    if image_path:
        targets = [image_path]
        images_dir = str(Path(image_path).parent)
    else:
        assert images_dir, "--images required when --image not given"
        targets = sorted(str(p) for p in Path(images_dir).glob("*.png"))

    all_preds: list[Tube] = []
    for path in targets:
        name = Path(path).name
        img = cv2.imread(path)
        if img is None:
            continue
        if mode == "oracle":
            tubes = _oracle_tubes(gt_by_image, name)
        elif mode == "classical":
            tubes = classical_detect(path)
        elif mode == "yolo":
            assert yolo_model is not None
            tubes = _yolo_detect(yolo_model, path, name)
        else:
            raise ValueError(f"unknown mode: {mode}")

        if tubes:
            angles = angle_pred.predict(img, [(t.center_x, t.center_y) for t in tubes])
            for t, a in zip(tubes, angles):
                t.angle_deg = a
        all_preds.extend(tubes)

        if viz_out and image_path:
            _draw_demo(path, tubes, viz_out)

    if out_csv:
        write_predictions_csv(out_csv, all_preds)
        print(f"wrote {len(all_preds)} predictions → {out_csv}")
    return all_preds


def _draw_demo(image_path: str, tubes: list[Tube], out_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    for t in tubes:
        x0, y0 = t.bbox_x, t.bbox_y
        x1, y1 = x0 + t.bbox_w, y0 + t.bbox_h
        d.rectangle([x0, y0, x1, y1], outline=(255, 60, 60), width=2)
        rad = math.radians(t.angle_deg)
        ex = t.center_x + 22 * math.cos(rad)
        ey = t.center_y - 22 * math.sin(rad)
        d.line([t.center_x, t.center_y, ex, ey], fill=(255, 220, 0), width=2)
        d.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=(255, 220, 0))
        d.text((x0, max(0, y0 - 12)), f"{t.angle_deg:.0f}°", fill=(255, 220, 0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "classical", "yolo"], required=True)
    ap.add_argument("--images", help="Directory of PNG images.")
    ap.add_argument("--image", help="Single image path (for demo).")
    ap.add_argument("--ann", help="Annotations CSV (required for mode=oracle).")
    ap.add_argument("--yolo", help="YOLOv11n weights (required for mode=yolo).")
    ap.add_argument("--angle", required=True, help="Angle head weights (.pt).")
    ap.add_argument("--out", help="Predictions CSV output.")
    ap.add_argument("--viz-out", help="Annotated PNG output for --image.")
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()
    run(
        mode=args.mode, images_dir=args.images, image_path=args.image,
        ann_csv=args.ann, yolo_weights=args.yolo, angle_weights=args.angle,
        out_csv=args.out, viz_out=args.viz_out, tta=not args.no_tta,
    )


if __name__ == "__main__":
    main()
