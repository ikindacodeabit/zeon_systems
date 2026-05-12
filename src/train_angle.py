"""Train the angle head.

Predicts (sinθ, cosθ) for θ ∈ [0, 360°) from a 64×64 cap crop using
MobileNetV3-Small as backbone. Rotation augmentation gives effectively
infinite training data, so 371 GT crops are plenty.

Loss: 1 − cos(θ_pred − θ_gt). On the unit circle this is equivalent to MSE on
(sinθ, cosθ) up to a constant. We L2-normalize the head's 2-vector output so
predictions live on the circle.

Run:
    python -m src.train_angle --images Dataset/images --ann Dataset/annotations.csv \
        --out outputs/angle.pt --epochs 30 --val-fold 0
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

from .dataio import Tube, load_annotations, group_by_image, kfold_image_splits


CROP_SIZE = 64
PATCH_RADIUS = 48  # source patch before rotation/resize (giving padding for rotation)


def _rotate_crop(img: np.ndarray, cx: float, cy: float, angle_deg: float, out_size: int) -> np.ndarray:
    """Rotate the full image around (cx, cy) by angle_deg (CCW) and crop a
    centred out_size × out_size window. Equivalent to rotating the cap so its
    tab is at angle 0°, but done as one affine warp for speed.
    """
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    # After warpAffine, the cap center is still at (cx, cy). Shift so it lands
    # at (out_size/2, out_size/2).
    M[0, 2] += (out_size / 2) - cx
    M[1, 2] += (out_size / 2) - cy
    return cv2.warpAffine(img, M, (out_size, out_size), borderMode=cv2.BORDER_REPLICATE)


class TubeAngleDataset(Dataset):
    def __init__(self, tubes: list[Tube], images_dir: str, train: bool):
        self.tubes = tubes
        self.images_dir = Path(images_dir)
        self.train = train
        # Cache images so we don't re-read from disk every batch.
        self._img_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.tubes)

    def _load(self, name: str) -> np.ndarray:
        img = self._img_cache.get(name)
        if img is None:
            img = cv2.imread(str(self.images_dir / name))
            assert img is not None, f"cannot read {name}"
            self._img_cache[name] = img
        return img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.tubes[idx]
        img = self._load(t.image)

        # Augmentation: random rotation α in [0, 360); add it to the label.
        if self.train:
            alpha = random.uniform(0, 360.0)
            jitter_x = random.uniform(-3, 3)
            jitter_y = random.uniform(-3, 3)
        else:
            alpha = 0.0
            jitter_x = jitter_y = 0.0

        # CCW rotation in screen coords: cv2 rotates CCW for positive angle in
        # standard math convention, but warpAffine uses screen y-down, so a
        # positive cv2 angle actually rotates the IMAGE clockwise on screen.
        # Empirically, getRotationMatrix2D(center, angle, scale) with angle>0
        # rotates the image *counter-clockwise* in display when y-axis is
        # flipped. We use this directly: rotating the source by α visually
        # adds α to the cap's joint→tab direction in our screen-CCW convention.
        crop = _rotate_crop(img, t.center_x + jitter_x, t.center_y + jitter_y, alpha, CROP_SIZE)

        if self.train:
            # Color jitter
            crop = crop.astype(np.float32)
            crop *= random.uniform(0.85, 1.15)
            crop += random.uniform(-10, 10)
            crop = np.clip(crop, 0, 255).astype(np.uint8)

        # Convert BGR → RGB, normalize.
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = torch.from_numpy(crop.transpose(2, 0, 1))

        # Adjusted label = (GT angle + α) mod 360. Empirically verified: with
        # this sign convention val error drops from ~85° → ~40°; with the
        # opposite sign it stays near random.
        angle = (t.angle_deg + alpha) % 360.0
        rad = math.radians(angle)
        target = torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)
        return tensor, target


class AngleHead(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_feats = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_feats, 2)
        self.net = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return F.normalize(out, dim=-1)


def circular_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # 1 - cos(θ_pred - θ_gt) using (sin, cos) representation.
    # cos(a-b) = sin a sin b + cos a cos b.
    cos_diff = (pred * target).sum(dim=-1)
    return (1.0 - cos_diff).mean()


def angle_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cos_diff = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cos_diff) * (180.0 / math.pi)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainResult:
    val_mean_err: float
    val_med_err: float


def train(
    images_dir: str,
    ann_csv: str,
    out_path: str,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-4,
    val_fold: int = 0,
    seed: int = 0,
) -> TrainResult:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_tubes = load_annotations(ann_csv)
    by_image = group_by_image(all_tubes)
    image_names = sorted(by_image.keys())
    folds = kfold_image_splits(image_names, k=5, seed=seed)
    train_imgs, val_imgs = folds[val_fold]
    train_tubes = [t for img in train_imgs for t in by_image[img]]
    val_tubes = [t for img in val_imgs for t in by_image[img]]

    device = pick_device()
    print(f"device: {device}  train_tubes={len(train_tubes)}  val_tubes={len(val_tubes)}")

    train_ds = TubeAngleDataset(train_tubes, images_dir, train=True)
    val_ds = TubeAngleDataset(val_tubes, images_dir, train=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = AngleHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tqdm(train_dl, desc=f"ep{ep} train", leave=False):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = circular_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        errs: list[float] = []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                errs.extend(angle_error_deg(pred, y).cpu().tolist())
        mean_e = float(np.mean(errs)) if errs else 0.0
        med_e = float(np.median(errs)) if errs else 0.0
        print(f"ep{ep:2d}: val mean={mean_e:.2f}°  med={med_e:.2f}°  <10°={(np.array(errs) <= 10).mean():.1%}")
        if mean_e < best:
            best = mean_e
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "val_mean_err": mean_e}, out_path)

    print(f"best val mean error: {best:.2f}°  → {out_path}")
    return TrainResult(val_mean_err=best, val_med_err=med_e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(
        args.images, args.ann, args.out,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        val_fold=args.val_fold, seed=args.seed,
    )


if __name__ == "__main__":
    main()
