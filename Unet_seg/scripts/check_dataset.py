#!/usr/bin/env python
"""Preflight a dataset before spending a training run on it.

    python scripts/check_dataset.py --manifest data/bladder/manifest.csv

Everything here fails *fast and loudly* rather than during epoch 3. The checks,
in the order a real dataset breaks them:

1. **Manifest structure** -- required columns, parseable rows.
2. **Files exist** -- every image, every referenced mask, every flow file.
3. **Ordering** -- frame indices increasing within a sequence, timestamps not
   disagreeing with frame order. Temporal pairs are formed from this ordering,
   so a scrambled sequence trains the temporal loss on nonsense.
4. **Enough patients for the split** -- 7:3 over *patients*, not frames. Two
   patients cannot make a 7:3 split that means anything.
5. **Split assignment and leakage** -- the same patient must never appear in
   both halves.
6. **Label coverage** -- unlabeled frames are a first-class case, but a
   validation split with no labels silently makes model selection meaningless.

Exit code 0 = ready to train. 1 = something must be fixed first.

``--write-split`` freezes the computed assignment into the manifest's ``split``
column, so the same partition survives even if patients are added later. After
that, set ``split.strategy: manifest`` to pin it.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from _common import REPO_ROOT  # noqa: F401  (path setup)

from rus_perception.data.manifest import load_manifest, write_manifest
from rus_perception.data.splits import (
    PatientLeakageError,
    apply_split,
    assert_no_patient_leakage,
    split_by_patient,
)
from rus_perception.utils.config import load_config
from rus_perception.utils.logging_utils import setup_logging

logger = logging.getLogger("check_dataset")

MIN_PATIENTS_WARN = 8


class Findings:
    """Accumulates problems so every check runs before the script gives up."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)
        logger.error("%s", message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning("%s", message)

    def ok(self, message: str) -> None:
        logger.info("ok   %s", message)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, help="Manifest CSV. Defaults to data.manifest.")
    parser.add_argument("--config", type=Path, default=Path("configs/slim_unet_temporal.yaml"))
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=None,
        metavar=("TRAIN", "VAL", "TEST"),
        help="Override split.ratios.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override split.seed.")
    parser.add_argument(
        "--write-split",
        action="store_true",
        help="Write the computed assignment into the manifest's split column.",
    )
    parser.add_argument("--skip-files", action="store_true", help="Skip the file-existence check.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    found = Findings()

    # -- config -------------------------------------------------------------
    config = None
    if args.config and args.config.is_file():
        config = load_config(args.config, validate=False)
    elif args.manifest is None:
        parser.error("Give --manifest, or a --config that names one.")

    manifest_path = args.manifest
    if manifest_path is None and config is not None:
        manifest_path = Path(str(config.get("data.manifest", "")))
    if not manifest_path or not manifest_path.is_file():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    ratios = args.ratios or (
        list(config.get("split.ratios", (0.70, 0.30, 0.0))) if config else (0.70, 0.30, 0.0)
    )
    seed = args.seed if args.seed is not None else int(config.get("split.seed", 42) if config else 42)

    # -- 1. structure -------------------------------------------------------
    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:  # noqa: BLE001 - report any parse failure verbatim
        logger.error("Could not read %s: %s", manifest_path, exc)
        return 1
    if len(manifest) == 0:
        logger.error("%s contains no rows.", manifest_path)
        return 1
    found.ok(
        f"manifest      {len(manifest)} frames, {len(manifest.patients)} patients, "
        f"{len(manifest.sequences())} sequences"
    )

    # -- 2. files -----------------------------------------------------------
    if args.skip_files:
        found.warn("file-existence check skipped (--skip-files)")
    else:
        try:
            manifest.validate_files(check_masks=True, check_flow=True)
            found.ok("files         every image, mask and flow path resolves")
        except FileNotFoundError as exc:
            found.error(str(exc))

    # -- 3. ordering --------------------------------------------------------
    try:
        manifest.validate_ordering()
        found.ok("ordering      frame indices and timestamps agree within every sequence")
    except Exception as exc:  # noqa: BLE001
        found.error(f"ordering: {exc}")

    # -- 4. enough patients -------------------------------------------------
    patients = list(manifest.patients)
    non_zero = [name for name, value in zip(("train", "val", "test"), ratios) if value > 0]
    if len(patients) < len(non_zero):
        found.error(
            f"{len(patients)} patient(s) cannot fill {len(non_zero)} non-empty split(s) {non_zero}."
        )
    elif len(patients) < MIN_PATIENTS_WARN:
        found.warn(
            f"only {len(patients)} patients. A {ratios[0]:.0%}/{ratios[1]:.0%} split over this few "
            "gives a validation set of a handful of subjects, so every reported number carries a "
            "very wide error bar. Treat the results as directional until the cohort grows."
        )
    else:
        found.ok(f"cohort        {len(patients)} patients")

    # -- 5. split + leakage -------------------------------------------------
    assignment = None
    if not found.errors:
        try:
            assignment = split_by_patient(patients, ratios, seed)
            split_manifest = apply_split(manifest, assignment)
            assert_no_patient_leakage(split_manifest)
            counts = {
                name: len([r for r in split_manifest if r.split == name])
                for name in ("train", "val", "test")
            }
            found.ok(
                f"split         train {len(assignment.train)}p/{counts['train']}f · "
                f"val {len(assignment.val)}p/{counts['val']}f · "
                f"test {len(assignment.test)}p/{counts['test']}f  (seed {seed})"
            )
            found.ok("leakage       no patient appears in more than one split")
        except (ValueError, PatientLeakageError) as exc:
            found.error(f"split: {exc}")
            split_manifest = None
    else:
        split_manifest = None

    # -- 6. label coverage --------------------------------------------------
    if split_manifest is not None:
        for name in ("train", "val"):
            rows = [r for r in split_manifest if r.split == name]
            if not rows:
                continue
            labeled = sum(1 for r in rows if r.mask_path)
            fraction = labeled / len(rows)
            if labeled == 0:
                found.error(
                    f"{name} split has no labeled frames. "
                    + ("Model selection would be meaningless." if name == "val" else
                       "There would be no segmentation supervision at all.")
                )
            elif fraction < 0.5:
                found.warn(f"{name} split is only {fraction:.0%} labeled ({labeled}/{len(rows)}).")
            else:
                found.ok(f"labels {name:<6} {labeled}/{len(rows)} frames labeled ({fraction:.0%})")

        flow = sum(1 for r in split_manifest if r.flow_backward_path)
        if flow == 0:
            found.warn(
                "no precomputed optical flow. The temporal loss will fall back to computing "
                "dense flow every epoch, or to no alignment at all -- run "
                "scripts/precompute_flow.py first."
            )
        else:
            found.ok(f"flow          {flow}/{len(split_manifest)} frames have precomputed flow")

    # -- write --------------------------------------------------------------
    if args.write_split:
        if split_manifest is None:
            found.error("--write-split refused: the split could not be computed.")
        else:
            write_manifest(manifest_path, list(split_manifest))
            logger.info(
                "wrote the split column into %s. Set split.strategy: manifest to pin it.",
                manifest_path,
            )

    # -- verdict ------------------------------------------------------------
    logger.info("-" * 68)
    if found.errors:
        logger.error("NOT READY -- %d error(s), %d warning(s)", len(found.errors), len(found.warnings))
        return 1
    if found.warnings:
        logger.warning("READY, with %d warning(s)", len(found.warnings))
    else:
        logger.info("READY -- training can start")
    return 0


if __name__ == "__main__":
    sys.exit(main())
