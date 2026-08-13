#!/usr/bin/env python3
"""Generate a small synthetic ultrasound-like dataset with a manifest.

The generator imitates the structure of a bladder B-mode frame (dark anechoic
lumen, brighter boundary, speckle, depth gain, occasional shadowing) so the
whole pipeline can be exercised without any clinical data. It is NOT a physical
ultrasound simulator and must not be used to make accuracy claims.

Example:
    python scripts/make_synthetic_dataset.py --output-dir data/synthetic \
        --patients 6 --frames 12 --size 128 128 --labeled-fraction 0.8
"""

from __future__ import annotations

import argparse
import logging

from _common import REPO_ROOT  # noqa: F401  (path setup)

from src.data.splits import apply_split, split_by_patient, split_report
from src.data.synthetic import SyntheticSequenceConfig, generate_dataset
from src.data.manifest import write_manifest
from src.utils.logging_utils import setup_logging

logger = logging.getLogger("make_synthetic_dataset")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic ultrasound-like dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", required=True, help="Directory to populate")
    parser.add_argument("--patients", type=int, default=6, help="Number of patients")
    parser.add_argument("--sequences", type=int, default=1, help="Sequences per patient")
    parser.add_argument("--frames", type=int, default=12, help="Frames per sequence")
    parser.add_argument("--size", type=int, nargs=2, default=[128, 128], metavar=("H", "W"))
    parser.add_argument("--labeled-fraction", type=float, default=1.0)
    parser.add_argument("--motion", type=float, default=1.5, help="Lumen motion in px/frame")
    parser.add_argument("--speckle-std", type=float, default=0.15)
    parser.add_argument("--frame-rate", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=[0.6, 0.2, 0.2],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Patient-level split ratios written into the manifest",
    )
    parser.add_argument("--no-split", action="store_true", help="Leave the split column empty")
    return parser.parse_args()


def main() -> int:
    args = get_args()
    setup_logging("INFO")

    config = SyntheticSequenceConfig(
        num_patients=args.patients,
        sequences_per_patient=args.sequences,
        frames_per_sequence=args.frames,
        image_size=(args.size[0], args.size[1]),
        labeled_fraction=args.labeled_fraction,
        motion_px_per_frame=args.motion,
        speckle_std=args.speckle_std,
        frame_rate=args.frame_rate,
    )
    manifest_path, manifest = generate_dataset(args.output_dir, config, seed=args.seed)

    if not args.no_split:
        assignment = split_by_patient(manifest.patients, tuple(args.split_ratios), seed=args.seed)
        manifest = apply_split(manifest, assignment)
        write_manifest(manifest_path, manifest.records)

    logger.info("Manifest written to %s", manifest_path)
    print(split_report(manifest, compute_class_occupancy=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
