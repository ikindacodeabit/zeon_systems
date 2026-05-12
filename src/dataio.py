"""Dataset I/O: parse annotations.csv, OBB→AABB, image-level k-fold splits, YOLO export."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np


IMG_W = 640
IMG_H = 480


@dataclass
class Tube:
    image: str
    center_x: float
    center_y: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    bbox_rotation: float
    angle_deg: float

    @classmethod
    def from_row(cls, row: dict) -> "Tube":
        return cls(
            image=row["image"],
            center_x=float(row["center_x"]),
            center_y=float(row["center_y"]),
            bbox_x=float(row["bbox_x"]),
            bbox_y=float(row["bbox_y"]),
            bbox_w=float(row["bbox_w"]),
            bbox_h=float(row["bbox_h"]),
            bbox_rotation=float(row["bbox_rotation"]),
            angle_deg=float(row["angle_deg"]),
        )

    def to_row(self) -> dict:
        return asdict(self)


CSV_FIELDS = [
    "image", "center_x", "center_y",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "bbox_rotation", "angle_deg",
]


def load_annotations(csv_path: str | os.PathLike) -> list[Tube]:
    """Load every annotation row as a Tube."""
    with open(csv_path, newline="") as f:
        return [Tube.from_row(r) for r in csv.DictReader(f)]


def group_by_image(tubes: Iterable[Tube]) -> dict[str, list[Tube]]:
    out: dict[str, list[Tube]] = {}
    for t in tubes:
        out.setdefault(t.image, []).append(t)
    return out


def write_predictions_csv(path: str | os.PathLike, tubes: Iterable[Tube]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for t in tubes:
            w.writerow(t.to_row())


def obb_corners(cx: float, cy: float, w: float, h: float, rot_deg: float) -> np.ndarray:
    """Return 4 corner points of an oriented bbox centered at (cx, cy).

    Convention: rot_deg is rotation in screen coordinates (y-down). Annotation
    docs call this clockwise; we apply it consistently regardless of sign because
    we only consume the AABB that wraps the result.
    """
    a = math.radians(rot_deg)
    ca, sa = math.cos(a), math.sin(a)
    hw, hh = w / 2, h / 2
    local = np.array([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
    R = np.array([[ca, -sa], [sa, ca]])
    rotated = local @ R.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return rotated


def obb_to_aabb(cx: float, cy: float, w: float, h: float, rot_deg: float) -> tuple[float, float, float, float]:
    """Return axis-aligned bbox (x_min, y_min, x_max, y_max) that wraps the OBB."""
    pts = obb_corners(cx, cy, w, h, rot_deg)
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def tube_aabb(t: Tube) -> tuple[float, float, float, float]:
    """AABB wrapping a Tube's oriented bbox."""
    return obb_to_aabb(t.center_x, t.center_y, t.bbox_w, t.bbox_h, t.bbox_rotation)


def kfold_image_splits(images: list[str], k: int = 5, seed: int = 0) -> list[tuple[list[str], list[str]]]:
    """Image-level k-fold. Returns list of (train_images, val_images) per fold."""
    rng = np.random.default_rng(seed)
    shuffled = list(images)
    rng.shuffle(shuffled)
    folds: list[list[str]] = [[] for _ in range(k)]
    for i, name in enumerate(shuffled):
        folds[i % k].append(name)
    out = []
    for i in range(k):
        val = folds[i]
        train = [n for j in range(k) if j != i for n in folds[j]]
        out.append((train, val))
    return out


def export_yolo_labels(
    tubes_by_image: dict[str, list[Tube]],
    image_list: list[str],
    out_dir: str | os.PathLike,
    img_w: int = IMG_W,
    img_h: int = IMG_H,
) -> None:
    """Write one .txt per image in YOLO axis-aligned format: 'class cx cy w h' normalized."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in image_list:
        stem = Path(name).stem
        lines = []
        for t in tubes_by_image.get(name, []):
            x0, y0, x1, y1 = tube_aabb(t)
            cx = ((x0 + x1) / 2) / img_w
            cy = ((y0 + y1) / 2) / img_h
            w = (x1 - x0) / img_w
            h = (y1 - y0) / img_h
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (out / f"{stem}.txt").write_text("\n".join(lines))


def make_yolo_dataset(
    images_dir: str | os.PathLike,
    tubes_by_image: dict[str, list[Tube]],
    train_imgs: list[str],
    val_imgs: list[str],
    root: str | os.PathLike,
) -> str:
    """Materialize a YOLO-format dataset folder with symlinks. Returns path to data.yaml."""
    root = Path(root)
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    images_dir = Path(images_dir).resolve()

    def link(name: str, split: str) -> None:
        src = images_dir / name
        dst = root / "images" / split / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)

    for name in train_imgs:
        link(name, "train")
    for name in val_imgs:
        link(name, "val")
    export_yolo_labels(tubes_by_image, train_imgs, root / "labels" / "train")
    export_yolo_labels(tubes_by_image, val_imgs, root / "labels" / "val")

    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: tube_cap\n"
    )
    return str(yaml_path)
