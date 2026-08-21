#!/usr/bin/env python3
"""Phase 9 (part 1) -- predict every test frame in sequence order.

Needed because the audit subset is 5 scattered frames per patient, which says
nothing about temporal stability. Runs at 256x256 (the model's working
resolution, matching runs/*/eval_test) and stores one boolean stack per patient
so the optical-flow work can happen in the environment that has scipy/cv2.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from rus_perception.control.features import FeatureExtractionConfig  # noqa: E402
from rus_perception.data.io import load_grayscale, load_mask, resize_image, resize_mask  # noqa: E402
from rus_perception.inference.predictor import Predictor, PredictorConfig  # noqa: E402
from rus_perception.utils.config import load_config  # noqa: E402

MODELS = {
    "standard": ("configs/pfus_bladder.yaml", "checkpoints/pfus_bladder/best.pt"),
    "slim": ("configs/pfus_bladder_slim.yaml", "checkpoints/pfus_bladder_slim/best.pt"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--manifest", default="/home/rosotauser/datasets/pfus/manifest.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=str(here))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    root = Path(args.manifest).parent
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    size = (args.size, args.size)

    with open(args.manifest, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == args.split]
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(row)
    for records in by_patient.values():
        records.sort(key=lambda r: int(r["frame_index"]))

    # Images and ground truth do not depend on the model: cache them once.
    images_path = out / f"images_{args.split}_{args.size}.npz"
    if not images_path.exists():
        cache = {}
        for patient, records in sorted(by_patient.items()):
            # load_grayscale returns float32 in [0, 1]; rescale before the uint8
            # cast or every pixel truncates to zero.
            cache[f"{patient}__image"] = (
                np.stack([resize_image(load_grayscale(root / r["image_path"]), size) for r in records])
                * 255.0
            ).clip(0, 255).astype(np.uint8)
            cache[f"{patient}__gt"] = np.stack(
                [resize_mask(load_mask(root / r["mask_path"]), size) for r in records]
            ).astype(bool)
        np.savez_compressed(images_path, **cache)
        print(f"cached images/gt -> {images_path.name}")

    for name, (config_path, checkpoint) in MODELS.items():
        target = out / f"masks_{name}_{args.split}_{args.size}.npz"
        if target.exists():
            print(f"[{name}] {target.name} exists, skipping")
            continue
        config = load_config(str(REPO / config_path))
        predictor = Predictor.from_checkpoint(
            str(REPO / checkpoint),
            model_config=config.section("model"),
            predictor_config=PredictorConfig(
                input_size=size,
                intensity_normalization=str(config.get("data.intensity_normalization", "zero_one")),
                normalization_stats=config.get("data.normalization_stats"),
                device=args.device,
                restore_original_size=False,
            ),
            feature_config=FeatureExtractionConfig.from_dict(
                {"postprocess": config.section("postprocess"), **config.section("control")}
            ),
        )
        store = {}
        for patient, records in sorted(by_patient.items()):
            masks = []
            for record in records:
                image = resize_image(load_grayscale(root / record["image_path"]), size)
                state = predictor.predict_control_state(image)
                masks.append(np.asarray(state.binary_mask).astype(bool))
            store[patient] = np.stack(masks)
            print(f"  [{name}] {patient}: {len(masks)} frames")
        np.savez_compressed(target, **store)
        print(f"[{name}] wrote {target.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
