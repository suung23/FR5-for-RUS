#!/usr/bin/env python3
"""Phase 6 -- run the existing checkpoints on the audit subset, unchanged.

Checkpoints are loaded exactly as scripts/evaluate.py loads them: architecture,
normalisation and postprocessing all come from the checkpoint's own stored
config, so nothing here can silently change what is being measured.

Predictions are produced at TWO resolutions on purpose:

  native  -- the frame's original size. Manual lumen annotation happens here, so
             every Phase 4/6/7 comparison against a manual mask must use it.
  r256    -- 256x256, the model's working resolution and the one
             runs/*/eval_test/frame_metrics.csv was computed at. Kept so the
             audit numbers can be reconciled with the existing report.

Masks are saved as .npz; metric computation lives in metrics/score_model_vs_pfus.py
because the only environment here with CUDA-capable torch has no scipy and the
only one with scipy has no torch. Splitting them keeps both untouched.
"""

from __future__ import annotations

import argparse
import csv
import sys
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


def build_predictor(config_path: Path, checkpoint: Path, device: str, native: bool) -> Predictor:
    config = load_config(str(config_path))
    size = config.get("model.input_size") or config.get("data.image_size") or [128, 128]
    predictor_config = PredictorConfig(
        input_size=(int(size[0]), int(size[1])),
        intensity_normalization=str(config.get("data.intensity_normalization", "zero_one")),
        normalization_stats=config.get("data.normalization_stats"),
        device=device,
        restore_original_size=native,
    )
    feature_config = FeatureExtractionConfig.from_dict(
        {"postprocess": config.section("postprocess"), **config.section("control")}
    )
    return Predictor.from_checkpoint(
        str(checkpoint), model_config=config.section("model"),
        predictor_config=predictor_config, feature_config=feature_config,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--audit-dir", default=str(here.parent / "annotation_audit"))
    parser.add_argument("--output-dir", default=str(here))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()

    audit = Path(args.audit_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = list(csv.DictReader((audit / "manifest.csv").open(encoding="utf-8")))

    rows: list[dict] = []
    for name in args.models:
        config_path, checkpoint = MODELS[name]
        print(f"[{name}] {checkpoint} on {args.device}")
        native_predictor = build_predictor(REPO / config_path, REPO / checkpoint, args.device, native=True)
        r256_predictor = build_predictor(REPO / config_path, REPO / checkpoint, args.device, native=False)
        store: dict[str, np.ndarray] = {}

        for index, record in enumerate(manifest):
            image = load_grayscale(audit / record["image_path"])
            pfus = load_mask(audit / record["original_mask_path"]) > 0

            state = native_predictor.predict_control_state(image)
            prediction = np.asarray(state.binary_mask).astype(bool)
            store[record["sample_id"]] = prediction

            small_image = resize_image(image, (256, 256))
            small_state = r256_predictor.predict_control_state(small_image)
            small_prediction = np.asarray(small_state.binary_mask).astype(bool)
            small_pfus = resize_mask(pfus.astype(np.uint8) * 255, (256, 256)) > 0
            store[record["sample_id"] + "__r256"] = small_prediction

            for resolution, prediction_mask, reference, height, width in (
                ("native", prediction, pfus, record["height"], record["width"]),
                ("r256", small_prediction, small_pfus, 256, 256),
            ):
                rows.append(
                    {
                        "model": name,
                        "sample_id": record["sample_id"],
                        "patient_id": record["patient_id"],
                        "frame_id": record["frame_id"],
                        "resolution": resolution,
                        "height": height,
                        "width": width,
                        "pred_area": int(prediction_mask.sum()),
                        "gt_area": int(reference.sum()),
                    }
                )
            if (index + 1) % 20 == 0:
                print(f"  {index + 1}/{len(manifest)}")

        np.savez_compressed(out / f"predictions_{name}.npz", **store)
        print(f"  saved predictions_{name}.npz")

    fields = ["model", "sample_id", "patient_id", "frame_id", "resolution",
              "height", "width", "pred_area", "gt_area"]
    with (out / "prediction_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote prediction_index.csv ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
