"""Classical CV baseline.

Intentionally simple — a sanity reference for the learned system, not a
competitive solution. We use a strong ROI prior (every tube center in the
training data lies inside a tight rectangle) plus HoughCircles, then keep the
top candidates per image and estimate angle from a local intensity ring.

The report explicitly calls out the limitations: the ROI prior would not
generalize beyond this dataset, and the tab-direction cue is weak when cap and
rack are similar in intensity.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from .dataio import Tube, write_predictions_csv


# Rack ROI prior, taken from the GT center distribution across all 70 images
# (xs: 341-520, ys: 46-231) with a small margin.
ROI_X0, ROI_Y0, ROI_X1, ROI_Y1 = 320, 30, 560, 260

HOUGH_MIN_R = 12
HOUGH_MAX_R = 22
HOUGH_MIN_DIST = 26


def _hough_circles(gray: np.ndarray) -> np.ndarray:
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=HOUGH_MIN_DIST, param1=90, param2=16,
        minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R,
    )
    if circles is None:
        return np.zeros((0, 3), dtype=float)
    return circles[0].astype(float)


def _tab_sector_scan(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    """Estimate the tab direction in [0, 360°) by scanning angular sectors just
    outside the cap. Empirically (probed on GT examples) the tab is the
    **brightest** sector outside the cap — the plastic flap reflects more light
    than the dark cap/rack body around it. We sample at r*1.25 and r*1.55,
    smooth circularly, and take argmax.
    """
    H, W = gray.shape
    n_angles = 36
    angs_deg = np.linspace(0, 360, n_angles, endpoint=False)
    radii = [r * 1.25, r * 1.55]

    sector_means = np.zeros(n_angles, dtype=np.float32)
    for i, a_deg in enumerate(angs_deg):
        a = math.radians(a_deg)
        vals = []
        for rr in radii:
            x = int(round(cx + rr * math.cos(a)))
            y = int(round(cy - rr * math.sin(a)))
            x0p, y0p = max(0, x - 1), max(0, y - 1)
            x1p, y1p = min(W, x + 2), min(H, y + 2)
            block = gray[y0p:y1p, x0p:x1p]
            if block.size:
                vals.append(float(block.mean()))
        sector_means[i] = float(np.mean(vals)) if vals else float(gray.mean())

    pad = 2
    extended = np.r_[sector_means[-pad:], sector_means, sector_means[:pad]]
    kernel = np.ones(2 * pad + 1, dtype=np.float32) / (2 * pad + 1)
    sector_smooth = np.convolve(extended, kernel, mode="valid")

    best_idx = int(np.argmax(sector_smooth))
    return float(angs_deg[best_idx]) % 360.0


def _hough_score(blurred: np.ndarray, cx: int, cy: int, r: int) -> float:
    """Crude "circularity" score: average gradient magnitude along the
    candidate circle. Higher = more circular edge support."""
    H, W = blurred.shape
    n = 48
    angs = np.linspace(0, 2 * math.pi, n, endpoint=False)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    xs = np.clip(cx + r * np.cos(angs), 0, W - 1).astype(np.int32)
    ys = np.clip(cy + r * np.sin(angs), 0, H - 1).astype(np.int32)
    return float(mag[ys, xs].mean())


def detect_one_image(image_path: str, max_per_image: int = 6) -> list[Tube]:
    name = Path(image_path).name
    img = cv2.imread(image_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    # ROI mask: replace outside with median so Hough has nothing to find there.
    roi_gray = np.full_like(gray, int(np.median(gray)))
    roi_gray[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1] = gray[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]

    cands = _hough_circles(roi_gray)
    scored: list[tuple[float, Tube]] = []
    for (x, y, r) in cands:
        cx, cy, r_i = int(round(x)), int(round(y)), int(round(r))
        if not (ROI_X0 <= cx <= ROI_X1 and ROI_Y0 <= cy <= ROI_Y1):
            continue
        score = _hough_score(blur, cx, cy, r_i)
        angle_full = _tab_sector_scan(gray, cx, cy, r_i)
        scored.append((
            score,
            Tube(
                image=name,
                center_x=float(cx),
                center_y=float(cy),
                bbox_x=float(cx - r_i),
                bbox_y=float(cy - r_i),
                bbox_w=float(2 * r_i),
                bbox_h=float(2 * r_i),
                bbox_rotation=0.0,
                angle_deg=float(angle_full),
            ),
        ))

    scored.sort(key=lambda x: x[0], reverse=True)
    kept: list[Tube] = []
    for _, t in scored:
        if all(math.hypot(t.center_x - k.center_x, t.center_y - k.center_y) >= HOUGH_MIN_DIST for k in kept):
            kept.append(t)
        if len(kept) >= max_per_image:
            break
    return kept


def run(images_dir: str, out_csv: str, max_per_image: int = 6) -> None:
    images_dir = Path(images_dir)
    out: list[Tube] = []
    image_names = sorted(p.name for p in images_dir.glob("*.png"))
    for name in image_names:
        out.extend(detect_one_image(str(images_dir / name), max_per_image=max_per_image))
    write_predictions_csv(out_csv, out)
    print(f"wrote {len(out)} detections → {out_csv}  ({len(image_names)} images)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-image", type=int, default=6)
    args = ap.parse_args()
    run(args.images, args.out, max_per_image=args.max_per_image)


if __name__ == "__main__":
    main()
