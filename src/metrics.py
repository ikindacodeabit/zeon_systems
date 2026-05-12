"""Evaluation metrics: IoU, Hungarian matching, P/R/F1 and circular angle error."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .dataio import Tube, tube_aabb


def iou_aabb(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def circular_diff_deg(a: float, b: float) -> float:
    """Smallest angular distance between two angles in degrees, in [0, 180]."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


@dataclass
class Match:
    pred_idx: int
    gt_idx: int
    iou: float
    angle_err: float


@dataclass
class ImageEval:
    image: str
    tp: int
    fp: int
    fn: int
    matches: list[Match] = field(default_factory=list)


def match_image(preds: list[Tube], gts: list[Tube], iou_thr: float = 0.5) -> ImageEval:
    """Greedy-optimal one-to-one matching via Hungarian on the IoU matrix."""
    if not preds and not gts:
        return ImageEval(image=(preds + gts)[0].image if (preds + gts) else "", tp=0, fp=0, fn=0)
    image_name = (preds[0].image if preds else gts[0].image)
    n_p, n_g = len(preds), len(gts)
    if n_p == 0:
        return ImageEval(image=image_name, tp=0, fp=0, fn=n_g)
    if n_g == 0:
        return ImageEval(image=image_name, tp=0, fp=n_p, fn=0)

    iou_mat = np.zeros((n_p, n_g), dtype=float)
    p_boxes = [tube_aabb(p) for p in preds]
    g_boxes = [tube_aabb(g) for g in gts]
    for i in range(n_p):
        for j in range(n_g):
            iou_mat[i, j] = iou_aabb(p_boxes[i], g_boxes[j])

    # Hungarian on cost = -IoU. Pad to square via large cost to allow unmatched.
    cost = -iou_mat
    r_idx, c_idx = linear_sum_assignment(cost)

    matches: list[Match] = []
    matched_p, matched_g = set(), set()
    for i, j in zip(r_idx, c_idx):
        if iou_mat[i, j] >= iou_thr:
            err = circular_diff_deg(preds[i].angle_deg, gts[j].angle_deg)
            matches.append(Match(pred_idx=int(i), gt_idx=int(j), iou=float(iou_mat[i, j]), angle_err=err))
            matched_p.add(int(i))
            matched_g.add(int(j))

    tp = len(matches)
    fp = n_p - tp
    fn = n_g - tp
    return ImageEval(image=image_name, tp=tp, fp=fp, fn=fn, matches=matches)


@dataclass
class DatasetMetrics:
    precision: float
    recall: float
    f1: float
    mean_angle_err: float
    median_angle_err: float
    pct_within_10: float
    pct_within_30: float
    tp: int
    fp: int
    fn: int
    matched_count: int

    def as_table(self) -> str:
        return (
            f"P={self.precision:.3f}  R={self.recall:.3f}  F1={self.f1:.3f}   "
            f"angle: mean={self.mean_angle_err:.2f}°  med={self.median_angle_err:.2f}°  "
            f"<10°={self.pct_within_10:.1%}  <30°={self.pct_within_30:.1%}   "
            f"(TP={self.tp} FP={self.fp} FN={self.fn})"
        )


def evaluate_dataset(
    preds_by_image: dict[str, list[Tube]],
    gts_by_image: dict[str, list[Tube]],
    iou_thr: float = 0.5,
) -> tuple[DatasetMetrics, list[ImageEval]]:
    all_images = sorted(set(preds_by_image) | set(gts_by_image))
    per_image: list[ImageEval] = []
    tp = fp = fn = 0
    angle_errs: list[float] = []
    for img in all_images:
        p = preds_by_image.get(img, [])
        g = gts_by_image.get(img, [])
        ev = match_image(p, g, iou_thr=iou_thr)
        per_image.append(ev)
        tp += ev.tp
        fp += ev.fp
        fn += ev.fn
        angle_errs.extend(m.angle_err for m in ev.matches)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    if angle_errs:
        arr = np.array(angle_errs)
        mean_e = float(arr.mean())
        med_e = float(np.median(arr))
        within_10 = float((arr <= 10).mean())
        within_30 = float((arr <= 30).mean())
    else:
        mean_e = med_e = within_10 = within_30 = 0.0

    return (
        DatasetMetrics(
            precision=precision, recall=recall, f1=f1,
            mean_angle_err=mean_e, median_angle_err=med_e,
            pct_within_10=within_10, pct_within_30=within_30,
            tp=tp, fp=fp, fn=fn, matched_count=len(angle_errs),
        ),
        per_image,
    )
