#!/usr/bin/env python3
"""Is ``Q_seg`` an objective a policy can follow in the image plane?

DESIGN_NOTES.md section 5.3 assigns ``Q_seg`` the in-plane axes and makes the
image policy's target ``Q_seg`` itself. The correction direction handed to the
operator -- human or robot -- is therefore the in-plane gradient of ``Q_seg``,
not a separately designed signal. That places one requirement on ``Q_seg`` which
has never been tested: **its gradient must point at a good view.** The same
document already refuses to assume this for ``Q_raw`` ("None of the four
sub-scores has been shown to be monotone or unimodal in contact force -- that is
an experiment, not an assumption").

Simulating in-plane motion from archived frames
-----------------------------------------------
In-plane probe translation moves the anatomy through a sector that stays fixed
in image coordinates, so it is simulated by translating the frame while holding
the ROI. Two limits come with that and are enforced here rather than hidden:

* Tissue shifted in from outside the frame was never imaged. Edge replication
  keeps a black band from masquerading as anechoic lumen, but the simulation is
  only trustworthy while the fabricated margin stays small -- hence the modest
  ``--extent``.
* The ROI is held fixed, so ``mask_area_ratio`` keeps one denominator across the
  grid. Letting the ROI shrink with the shift would move ``mask_completeness``
  for reasons that have nothing to do with the probe.

The sweeps in this dataset cannot substitute: the bladder centroid moves 0.32 px
between sampled frames, well under the segmentation's own centroid error, so
real probe motion here carries no signal.

Ground-truth reference view
---------------------------
A good in-plane view is one where the true lumen is fully insonified and sits on
the beam axis, near mid-field. In-plane translation cannot change the imaging
plane, so those are the only things it can improve. Both are read from the
ground-truth mask, never from a prediction:

``visible``   fraction of the ground-truth lumen inside the ROI
``centred``   distance from the ground-truth centroid to the ROI centroid

The reference offset is where the lumen would be centred, which is a fixed
vector per frame; the test is whether the ``Q_seg`` gradient points along it.

Example:
    python scripts/qseg_landscape.py --frames-per-patient 5 --extent 9 --step 3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

from rus_perception.control.features import FeatureExtractionConfig, extract_control_state
from rus_perception.control.roi import RoiConfig, build_roi_mask
from rus_perception.data.io import load_grayscale, load_mask, resize_image, resize_mask
from rus_perception.data.manifest import load_manifest
from rus_perception.inference.predictor import Predictor, PredictorConfig

logger = logging.getLogger(__name__)


def translate(array: np.ndarray, dx: int, dy: int, size: tuple[int, int]) -> np.ndarray:
    """Shift by ``(dx, dy)``, replicating the edge rather than filling with black.

    A zero fill would introduce a maximally dark band, which the intensity
    features would read as lumen; replication produces tissue-like values that
    are wrong in detail but not systematically biased toward "anechoic".
    """
    import cv2

    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    interpolation = cv2.INTER_NEAREST if array.dtype == np.uint8 else cv2.INTER_LINEAR
    return cv2.warpAffine(array, matrix, (size[1], size[0]),
                          flags=interpolation, borderMode=cv2.BORDER_REPLICATE)


def landscape(predictor, feature_config, roi: np.ndarray, image: np.ndarray,
              truth: np.ndarray, offsets: Sequence[int]) -> dict:
    """``Q_seg`` and the ground-truth view measures over a grid of translations."""
    size = image.shape
    ys, xs = np.nonzero(roi)
    roi_centre = np.array([xs.mean(), ys.mean()])
    n = len(offsets)
    quality = np.zeros((n, n))
    visible = np.zeros((n, n))
    centred = np.zeros((n, n))
    dice = np.zeros((n, n))

    for iy, dy in enumerate(offsets):
        for ix, dx in enumerate(offsets):
            frame = translate(image, dx, dy, size)
            target = translate(truth, dx, dy, size)
            probability, _ = predictor.predict_probability(frame)
            state = extract_control_state(probability, image=frame,
                                          config=feature_config, roi_mask=roi)
            quality[iy, ix] = state.control_quality_score

            target_px = int(target.sum())
            visible[iy, ix] = (float((target & (roi > 0)).sum()) / target_px
                               if target_px else np.nan)
            gy, gx = np.nonzero(target)
            centred[iy, ix] = (-math.hypot(gx.mean() - roi_centre[0], gy.mean() - roi_centre[1])
                               if gx.size else np.nan)
            prediction = np.asarray(state.binary_mask, np.uint8)
            total = float(prediction.sum() + target_px)
            dice[iy, ix] = 2.0 * float((prediction & target).sum()) / total if total else np.nan

    return {"quality": quality, "visible": visible, "centred": centred, "dice": dice,
            "roi_centre": roi_centre}


def gradient_at_centre(grid: np.ndarray, offsets: Sequence[int]) -> Optional[np.ndarray]:
    """Central-difference gradient of a landscape at the recorded pose."""
    centre = len(offsets) // 2
    if centre < 1:
        return None
    step = offsets[centre + 1] - offsets[centre]
    return np.array([
        (grid[centre, centre + 1] - grid[centre, centre - 1]) / (2.0 * step),
        (grid[centre + 1, centre] - grid[centre - 1, centre]) / (2.0 * step),
    ])


def smoothness(grid: np.ndarray) -> float:
    """Mean |Laplacian| divided by the landscape's range.

    A gradient-following policy needs a surface whose local slope means
    something. This is 0 for a plane and grows as the surface breaks into
    noise-scale bumps; it is reported rather than thresholded.
    """
    import cv2

    span = float(np.nanmax(grid) - np.nanmin(grid))
    if span < 1e-9:
        return float("nan")
    return float(np.abs(cv2.Laplacian(grid.astype(np.float32), cv2.CV_32F)).mean() / span)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/exp_seed43_retro.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/exp_seed43/best.pt")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--frames-per-patient", type=int, default=5)
    parser.add_argument("--extent", type=int, default=9,
                        help="Largest translation simulated, in pixels at 256x256.")
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/qseg_landscape"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from rus_perception.utils.config import load_config
    config = load_config(args.config)
    size = tuple(int(v) for v in config.section("data")["image_size"])
    roi_config = RoiConfig.from_dict(config.section("control").get("roi"))
    roi = np.asarray(build_roi_mask(size, roi_config)) > 0
    feature_config = FeatureExtractionConfig.from_dict(
        {"postprocess": config.section("postprocess"), **config.section("control")}
    )
    predictor = Predictor.from_checkpoint(
        args.checkpoint, model_config=config.section("model"),
        predictor_config=PredictorConfig(
            input_size=size,
            intensity_normalization=str(config.get("data.intensity_normalization")),
            device=str(config.get("train.device", "auto")).replace("auto", "cuda"),
            restore_original_size=False, roi=roi_config),
        feature_config=feature_config,
    )

    data = config.section("data")
    manifest = load_manifest(str(data["manifest"]), root=str(data["root"]))
    by_patient: dict[str, list] = defaultdict(list)
    for record in manifest:
        if record.split in args.splits and record.mask_path:
            by_patient[record.patient_id].append(record)

    offsets = list(range(-args.extent, args.extent + 1, args.step))
    rows = []
    for patient, records in sorted(by_patient.items()):
        records.sort(key=lambda r: r.frame_id)
        chosen = records[:: max(1, len(records) // args.frames_per_patient)][:args.frames_per_patient]
        for record in chosen:
            image = np.asarray(resize_image(load_grayscale(str(manifest.resolve(record.image_path))),
                                            size), np.float32)
            truth = (np.asarray(resize_mask(load_mask(str(manifest.resolve(record.mask_path))),
                                            size)) > 0.5).astype(np.uint8)
            result = landscape(predictor, feature_config, roi, image, truth, offsets)

            gy, gx = np.nonzero(truth)
            if gx.size == 0:
                continue
            # Where the probe would have to move to bring the true lumen onto the
            # beam axis. The image shifts with the probe, so this is the offset
            # that carries the ground-truth centroid onto the ROI centre.
            reference = result["roi_centre"] - np.array([gx.mean(), gy.mean()])
            gradient = gradient_at_centre(result["quality"], offsets)
            if gradient is None or np.linalg.norm(gradient) < 1e-12:
                continue
            cosine = float(gradient @ reference /
                           (np.linalg.norm(gradient) * np.linalg.norm(reference)))
            best = np.unravel_index(int(np.argmax(result["quality"])), result["quality"].shape)
            rows.append({
                "split": record.split, "patient_id": patient, "frame_id": record.frame_id,
                "lever_arm_px": float(np.linalg.norm(reference)),
                "cos_gradient_vs_reference": cosine,
                "angle_deg": float(math.degrees(math.acos(max(-1.0, min(1.0, cosine))))),
                "quality_at_pose": float(result["quality"][len(offsets) // 2, len(offsets) // 2]),
                "quality_range": float(result["quality"].max() - result["quality"].min()),
                "argmax_offset": [offsets[best[1]], offsets[best[0]]],
                "argmax_on_edge": bool(best[0] in (0, len(offsets) - 1)
                                       or best[1] in (0, len(offsets) - 1)),
                "smoothness": smoothness(result["quality"]),
                "visible_at_pose": float(result["visible"][len(offsets) // 2, len(offsets) // 2]),
                "dice_at_pose": float(result["dice"][len(offsets) // 2, len(offsets) // 2]),
            })
        logger.info("%s: %d frames", patient, len(chosen))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "landscape.json").write_text(json.dumps(
        {"offsets": offsets, "config": args.config, "rows": rows}, indent=2))

    cos = np.array([r["cos_gradient_vs_reference"] for r in rows])
    ang = np.array([r["angle_deg"] for r in rows])
    print(f"\nFrames: {len(rows)}   grid ±{args.extent} px step {args.step}")
    print(f"gradient of Q_seg vs direction to the centred view:")
    print(f"  cos > 0 : {(cos > 0).mean() * 100:.1f}%   (chance 50%)")
    print(f"  angle   : median {np.median(ang):.1f}°  p90 {np.percentile(ang, 90):.1f}°"
          f"   (chance 90°)")
    print(f"  smoothness (|Laplacian| / range): median "
          f"{np.median([r['smoothness'] for r in rows]):.2f}")
    print(f"  argmax on grid edge: "
          f"{np.mean([r['argmax_on_edge'] for r in rows]) * 100:.0f}% of frames")
    for split in args.splits:
        s = np.array([r["cos_gradient_vs_reference"] for r in rows if r["split"] == split])
        if s.size:
            a = np.degrees(np.arccos(np.clip(s, -1, 1)))
            print(f"  {split}: cos>0 {(s > 0).mean() * 100:.1f}%  median angle {np.median(a):.1f}°"
                  f"  (n={s.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
