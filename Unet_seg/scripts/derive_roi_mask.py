#!/usr/bin/env python3
"""Measure the imaged-sector ROI from the data and write it as a mask file.

Why this is needed
------------------
``rus_perception/control/roi.py`` ships with ``mode: full`` and says of every
other mode: "nothing here is calibrated". With ``full``, three control features
are silently inert, exactly as that module's docstring warns:

* ``segmentation_confidence`` averages ``|2p-1|`` over the whole frame, so the
  dead region outside the sector -- which any model calls background with total
  confidence -- dominates the mean and hides hedging at the lumen boundary.
* ``border_contact_ratio`` counts mask perimeter on the *image rectangle*. A fan
  inscribed in that rectangle never reaches it, so the value is 0.0 on every
  frame and the criterion never fires.
* ``mask_area_ratio`` divides by ``H*W``, making the frame grabber's crop an
  input to ``mask_completeness`` and hence to the contact force the search
  settles on.

This script closes that gap empirically. The sector is a property of the
acquisition, not of the model, so it is measured once from the images.

Method
------
For each patient, the pixelwise maximum over a sample of frames is thresholded:
a pixel that is *never* bright in any frame of a sweep was never insonified.
Per-patient masks are then combined by a consensus vote, which rejects both a
patient whose sweep happened to leave a corner dark and a stray bright artefact
outside the sector.

Only the ``train`` split is read. The ROI is acquisition geometry rather than a
fitted parameter, so leakage is a weak concern here, but deriving it from train
costs nothing and keeps the val/test frames untouched.

Example:
    python scripts/derive_roi_mask.py --output configs/roi/pfus_sector_256.npy
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

from rus_perception.data.io import load_grayscale, resize_image
from rus_perception.data.manifest import load_manifest

logger = logging.getLogger(__name__)

#: Intensity above which a pixel counts as insonified. B-mode dead regions are
#: written as exact zeros by the scan converter; the margin absorbs the ringing
#: that JPEG-style compression leaves around the sector edge.
BRIGHT_THRESHOLD = 0.02

#: Fraction of patients that must mark a pixel as imaged for it to enter the ROI.
CONSENSUS = 0.5


def patient_sector(paths: list[str], size: tuple[int, int]) -> np.ndarray:
    """Pixelwise-maximum sector mask for one patient, at the working resolution."""
    accumulator = None
    for path in paths:
        frame = resize_image(load_grayscale(path), size)
        accumulator = frame if accumulator is None else np.maximum(accumulator, frame)
    return np.asarray(accumulator) > BRIGHT_THRESHOLD


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/exp_seed43.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("configs/roi/pfus_sector_256.npy"))
    parser.add_argument("--frames-per-patient", type=int, default=40,
                        help="Frames sampled per patient for the maximum projection.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from rus_perception.utils.config import load_config
    data = load_config(args.config).section("data")
    size = tuple(int(v) for v in data["image_size"])
    # data.root may be null ("paths are relative to the manifest's own directory");
    # str(None) would resolve every path against a literal "None" directory.
    root = data.get("root")
    manifest = load_manifest(str(data["manifest"]), root=str(root) if root else None)

    by_patient: dict[str, list[str]] = defaultdict(list)
    for record in manifest:
        if record.split == args.split:
            by_patient[record.patient_id].append(str(manifest.resolve(record.image_path)))
    if not by_patient:
        raise ValueError(f"No frame in split {args.split!r}; nothing to measure.")

    masks = {patient: patient_sector(paths[: args.frames_per_patient], size)
             for patient, paths in sorted(by_patient.items())}
    stack = np.stack(list(masks.values()))
    roi = stack.mean(axis=0) >= CONSENSUS

    # Agreement is the evidence that one ROI is valid for the whole cohort. A
    # low minimum would mean the geometry varies per patient and a single mask
    # is the wrong abstraction -- so it is reported, not assumed.
    agreement = {
        patient: float(np.logical_and(mask, roi).sum() / np.logical_or(mask, roi).sum())
        for patient, mask in masks.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, roi)
    report = {
        "split": args.split,
        "n_patients": len(masks),
        "size": list(size),
        "bright_threshold": BRIGHT_THRESHOLD,
        "consensus": CONSENSUS,
        "roi_coverage": float(roi.mean()),
        "per_patient_iou": agreement,
        "iou_min": min(agreement.values()),
        "iou_median": float(np.median(list(agreement.values()))),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))

    logger.info("ROI covers %.2f%% of the frame, from %d %s patients.",
                roi.mean() * 100, len(masks), args.split)
    logger.info("Per-patient IoU against the consensus: min %.4f, median %.4f",
                report["iou_min"], report["iou_median"])
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
