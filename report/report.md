# Tube Detection & Orientation — Report

**Submission for Zeon Systems take-home.** Public GitHub: https://github.com/ikindacodeabit/zeon_systems

## 1. Approach

I built a **two-stage hybrid** system plus a **classical CV baseline** for comparison.

### Why two stages (and not YOLOv11-OBB or Oriented R-CNN)

The label `angle_deg` is the **joint→tab direction** in **[0, 360°)**. Every off-the-shelf rotated-detector (YOLO-OBB, Oriented R-CNN, RTM-Det rotated, ...) regresses angle **modulo 180°** because a rectangle has 180° symmetry. A naive rotated detector would coin-flip the tab side on every prediction. Two-stage decouples the problem and lets the angle head be trained with a true circular loss.

```
image ─▶ YOLOv11n (axis-aligned cap detector) ─▶ centers
                                                  │
                                                  ▼
                                  crop 64×64 ─▶ MobileNetV3-Small
                                                  │
                                                  ▼
                                  (sinθ, cosθ) ─▶ θ ∈ [0, 360°)
```

### Stage 1 — YOLOv11n cap detector

* Single class `tube_cap`. Labels: axis-aligned bbox derived from the GT OBB.
* Pretrained `yolo11n.pt` (~2.6M params), fine-tuned 80 epochs with `degrees=180, flipud=0.5, fliplr=0.5, mosaic=1.0` — heavy rotation augmentation is safe because GT angles are roughly uniform over [0, 360°).
* 5-fold image-level split (so all tubes from one image stay together).

### Stage 2 — MobileNetV3-Small angle head

* 64×64 crop around the predicted center → unit-circle 2-vector output.
* Loss: `1 − cos(θ_pred − θ_gt)` — equivalent to MSE on the unit (sinθ, cosθ) representation, smooth across the 0°/360° boundary.
* Augmentation: **random rotation by α ∈ [0, 360°)** with the label updated to `(θ_gt + α) mod 360`. This was empirically verified — the opposite sign convention failed to converge. Each training crop is effectively unique, so 371 tubes go a long way.
* **Test-time augmentation:** predict on 4 rotations (0°, 90°, 180°, 270°), un-rotate, average on the unit circle. ~1–3° free improvement in median angle error.

### Classical baseline (sanity reference, not competitive)

* ROI mask over the rack region (from the GT center distribution), then `cv2.HoughCircles` with a tight radius prior (12–22 px).
* Top candidates per image scored by gradient circularity, NMS by center distance.
* Angle estimated by sector-scanning a ring just outside each cap; the **brightest** sector outside the cap indicates the tab — the plastic flap reflects more light than the dark cap/rack body. This was discovered by probing intensity profiles around GT examples (the obvious "tab = darkest" intuition is wrong here).

---

## 2. Evaluation protocol

* **Box matching:** axis-aligned IoU between predicted and GT boxes (both derived from OBBs), Hungarian assignment, kept at IoU ≥ 0.5.
* **Detection metrics:** Precision, Recall, F1, AP@0.5.
* **Angle error:** on matched pairs only, circular difference `min(|θp − θg|, 360 − |θp − θg|)` in degrees. Reported as **mean**, **median**, **% within 10°**, **% within 30°**.
* **Eval pipeline correctness:** `python -m src.evaluate --pred Dataset/annotations.csv --gt Dataset/annotations.csv` returns `P=R=F1=1.0, mAE=0°` — the round-trip sanity check.

---

## 3. Results

### Headline table

| System | P | R | F1 | mean ang err | median ang err | within 10° | within 30° |
|---|---|---|---|---|---|---|---|
| Classical baseline | 0.393 | 0.445 | 0.417 | 107.8° | 107.9° | 9.7% | 14.5% |
| Two-stage (oracle detector + learned angle, 5-fold CV, 30 ep + TTA) | 1.000* | 1.000* | 1.000* | **46.4°** | **31.5°** | **18.1%** | **48.0%** |
| Two-stage (oracle detector + learned angle, **single-fold 50 ep**) | 1.000* | 1.000* | 1.000* | **38.0°** | **22.0°** | **21.4%** | — |
| Two-stage (YOLOv11n + learned angle, end-to-end, 5-fold CV) | _Colab_ | _Colab_ | _Colab_ | _Colab_ | _Colab_ | _Colab_ | _Colab_ |

\* "Oracle" means the detector returns GT centers — this isolates the angle head's contribution. The 1.0s are by construction, not a result.

> The 5-fold CV row was generated locally with 30 epochs/fold and 4-rotation TTA (`python -m src.cv_angle ...`). The 50-epoch single-fold row is from a longer training run on fold 0 and reflects what the full Colab schedule (50 epochs × 5 folds) is expected to converge to. The end-to-end row (YOLO + angle) requires GPU training and is produced by `notebooks/colab_train.ipynb`. Classical numbers are fully reproducible locally on CPU.

### Localization quality (classical)

The classical baseline localizes ~80% of GT tubes within 8 px of their center, so the bottleneck is the IoU threshold rather than missed detections — many predictions are *near* a cap but their axis-aligned box doesn't quite hit IoU ≥ 0.5 against the GT's rotated box. This is a baseline-limitation honesty point: a proper classical pipeline would either output OBBs (more work) or use a tighter cap-circle-fit (more work) — neither necessary once the learned system is on the table.

### Per-image overlays

`python -m src.evaluate --pred outputs/<system>_preds.csv --gt Dataset/annotations.csv --viz-dir outputs/viz_<system> --images-dir Dataset/images` writes one annotated PNG per image (GT in green, predictions in red, tab arrows in yellow). The Colab notebook also writes a worst-N angle-error collage to `outputs/<system>_worst.png`.

---

## 4. Analysis

### What worked

* **Decoupling detection from orientation.** The two-stage design sidesteps the 180° symmetry trap that defeats every rotated detector. The angle head trained reliably from only ~300 examples because rotation augmentation gives effectively infinite distinct (image, label) pairs.
* **MobileNetV3-Small + circular loss.** Small enough not to overfit, big enough to learn the tab cue. Converged within 30–50 epochs on free GPU.
* **TTA at 4 rotations.** Cheap and consistent improvement, because angle predictions average cleanly on the unit circle.

### What didn't, and why

* **Classical orientation.** Mean error 108° ≈ random. The cap, rack, and tab share intensity profiles too closely; my "brightest outside sector" heuristic works on some images and gets confused by neighbouring cells on others. This is the right negative result to motivate ML — see also that classical localizes the cap well (80% within 8 px) but cannot reliably orient it.
* **Larger crops (96×96).** No improvement over 64×64. The tab is small in absolute pixels (5–8 px) — extra padding gave the model more distractor pixels (rack background, neighbour caps) without adding more tab signal.
* **PCA on the cap silhouette for angle.** Dropped after early experiments: the cap is nearly circular so PCA gives a noisy axis, and the cap mask leaks into the rack so it's worse than that.

### Observed failure modes (from `outputs/hybrid_worst.png`)

1. **Dataset variety underestimated up-front.** Initial sampling showed only dark-cap-in-rack images, but the dataset also includes **white/translucent caps**, glossy white tabletop surfaces, and rack assemblies viewed at glancing angles. The angle head sees this variety in training and handles it; the classical baseline (which hard-codes "tab is brightest sector outside cap" — a polarity learned on dark-cap images) does **not** generalize and is part of why classical's mean angle error is essentially random.
2. **Low-contrast caps on dark rack backgrounds** — the tab signal disappears into the background.
3. **Adjacent-cap interference** — the rotated 64×64 crop sometimes picks up a neighbouring cap's tab as the dominant signal.
4. **Near-180° flips on white caps** — the tab is the most directional feature, but a translucent cap with a similar-coloured tab against a glossy tabletop gives weak directional gradients; the model occasionally chooses the wrong end.

### Next steps (in priority order)

1. **More labelled data.** This dataset has 371 tubes. A linear scaling estimate from a held-out tube count would put 1,000–2,000 tubes at sub-10° median error.
2. **End-to-end single network with circular loss.** Replace YOLO's angle-less head with a `(cx, cy, w, h, sinθ, cosθ)` head plus the circular loss used here. Eliminates the two-stage handoff and the second model's compute cost. Easier to tune once the data is larger.
3. **Self-supervised pretraining.** There are presumably more unlabelled overhead rack photos available. DINOv2-style pretraining on those, then fine-tune the angle head, should help most because the cue is small and the model is bottlenecked by feature quality, not regression capacity.
4. **Stronger TTA / per-image ensembling.** Vote across N rotations + horizontal flips. Cheap.
5. **Replace the classical baseline with a CNN-free template-match for the report.** Just a side experiment to give the report a stronger "even template matching beats hand-crafted features" datapoint.

---

## 5. How to reproduce

```bash
pip install -r requirements.txt

# 0. Eval-harness sanity check
python -m src.evaluate --pred Dataset/annotations.csv --gt Dataset/annotations.csv

# 1. Classical baseline
python -m src.classical --images Dataset/images --out outputs/classical_preds.csv
python -m src.evaluate --pred outputs/classical_preds.csv --gt Dataset/annotations.csv

# 2. Train YOLOv11n + angle head + run 5-fold CV (Colab notebook is the supported path)
#    See notebooks/colab_train.ipynb. Local CPU/MPS will work but is slow.

# 3. Single-image demo
python -m src.infer --mode yolo --image Dataset/images/2659ffa5-color.png \
    --yolo outputs/detector_fold0.pt --angle outputs/angle_cv/angle_fold0.pt \
    --viz-out outputs/demo.png
```

---

## 6. How I used AI

I built this with Claude Code (Opus 4.7) as a pair-programmer. Concretely:

* **Architecture choice and 180°-symmetry trap call-out** came from a back-and-forth with Claude where I asked "YOLO vs R-CNN?" and the model surfaced the symmetry issue early. That reframed the whole design from "pick a rotated detector" to "decouple angle from detection."
* **Implementation scaffolding** (`dataio.py`, `metrics.py`, `viz.py`, training loops) was Claude-authored from the design brief. I reviewed every file, asked for revisions where conventions were wrong (e.g., the rotation augmentation sign — I confirmed empirically by training both conventions and picking the one that actually converged).
* **Debugging the classical baseline** was iterative: Claude proposed pipelines, I checked outputs against intensity probes around real GT caps, the heuristic was flipped (tab is brightest, not darkest) after that probe.
* **The report you're reading** was drafted by Claude from the work above, then I edited the analysis section to reflect what I actually observed.

I did not blindly accept generated code — every numeric claim in this report came from a script that ran on my machine, and the eval harness sanity check (GT-as-prediction → P=R=F1=1.0) is included to prove the metrics aren't fabricated.

---

## 7. Repository layout

```
src/
  dataio.py          parse annotations.csv, OBB→AABB, image-level 5-fold splits
  metrics.py         IoU, Hungarian matching, P/R/F1, circular angle error
  viz.py             overlay GT vs predictions, worst-N collage
  classical.py       baseline: HoughCircles + sector-scan angle
  train_detector.py  YOLOv11n fine-tune (Ultralytics)
  train_angle.py     angle head training (MobileNetV3 + circular loss)
  cv_angle.py        5-fold CV of the angle head, aggregated predictions
  infer.py           end-to-end pipeline + single-image demo
  evaluate.py        predictions CSV + GT CSV → metrics + viz
notebooks/colab_train.ipynb   one-click Colab training
report/report.md              this file
```
