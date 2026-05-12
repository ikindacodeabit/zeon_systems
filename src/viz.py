"""Overlay GT and predictions on images. Includes a failure-mode collage helper."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .dataio import Tube, tube_aabb
from .metrics import ImageEval


GREEN = (0, 220, 0)
RED = (255, 60, 60)
YELLOW = (255, 220, 0)
CYAN = (0, 220, 220)


def _arrow(d: ImageDraw.ImageDraw, cx: float, cy: float, angle_deg: float, length: float, color):
    rad = math.radians(angle_deg)
    # CCW angle in screen coords (y-down) → flip y
    ex = cx + length * math.cos(rad)
    ey = cy - length * math.sin(rad)
    d.line([cx, cy, ex, ey], fill=color, width=2)
    d.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=color)


def draw_tubes(img: Image.Image, tubes: list[Tube], box_color, arrow_color, label: str | None = None) -> Image.Image:
    d = ImageDraw.Draw(img)
    for t in tubes:
        x0, y0, x1, y1 = tube_aabb(t)
        d.rectangle([x0, y0, x1, y1], outline=box_color, width=2)
        d.ellipse([t.center_x - 3, t.center_y - 3, t.center_x + 3, t.center_y + 3], fill=box_color)
        _arrow(d, t.center_x, t.center_y, t.angle_deg, length=22, color=arrow_color)
    if label:
        d.text((4, 4), label, fill=box_color)
    return img


def overlay_pred_vs_gt(image_path: str, preds: list[Tube], gts: list[Tube], out_path: str) -> None:
    """Single image: GT in green, predictions in red. Both get an arrow."""
    img = Image.open(image_path).convert("RGB")
    draw_tubes(img, gts, box_color=GREEN, arrow_color=GREEN)
    draw_tubes(img, preds, box_color=RED, arrow_color=YELLOW)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def worst_n_collage(
    preds_by_image: dict[str, list[Tube]],
    gts_by_image: dict[str, list[Tube]],
    per_image_evals: list[ImageEval],
    images_dir: str,
    out_path: str,
    n: int = 9,
) -> None:
    """Compose an N-up grid of the worst-angle-error matches across the dataset."""
    cells: list[Image.Image] = []
    # Flatten matches with their image and angle error
    items = []
    for ev in per_image_evals:
        for m in ev.matches:
            items.append((ev.image, m))
    items.sort(key=lambda x: x[1].angle_err, reverse=True)

    for image_name, m in items[:n]:
        img = Image.open(Path(images_dir) / image_name).convert("RGB")
        gt = gts_by_image[image_name][m.gt_idx]
        pr = preds_by_image[image_name][m.pred_idx]
        x0, y0, x1, y1 = tube_aabb(gt)
        pad = 18
        x0 = max(0, int(x0 - pad)); y0 = max(0, int(y0 - pad))
        x1 = min(img.width, int(x1 + pad)); y1 = min(img.height, int(y1 + pad))
        crop = img.crop((x0, y0, x1, y1)).resize((160, 160))
        d = ImageDraw.Draw(crop)
        # Re-draw arrows in crop coords
        sx = 160 / (x1 - x0); sy = 160 / (y1 - y0)
        cx_g = (gt.center_x - x0) * sx; cy_g = (gt.center_y - y0) * sy
        cx_p = (pr.center_x - x0) * sx; cy_p = (pr.center_y - y0) * sy
        _arrow(d, cx_g, cy_g, gt.angle_deg, 26, GREEN)
        _arrow(d, cx_p, cy_p, pr.angle_deg, 26, YELLOW)
        d.text((4, 4), f"err={m.angle_err:.0f}°", fill=CYAN)
        cells.append(crop)

    if not cells:
        return
    cols = int(math.ceil(math.sqrt(len(cells))))
    rows = int(math.ceil(len(cells) / cols))
    sheet = Image.new("RGB", (cols * 160, rows * 160), (20, 20, 20))
    for i, c in enumerate(cells):
        r, k = divmod(i, cols)
        sheet.paste(c, (k * 160, r * 160))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
