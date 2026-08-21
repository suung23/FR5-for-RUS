#!/usr/bin/env python3
"""Phase 2 -- build the PFUS1 annotation audit subset.

Samples frames for expert re-annotation of the urine-filled bladder LUMEN.

Design decisions worth knowing before changing anything here:

* Frames come from the **test** split only. The audit feeds the Phase 6/7
  ceiling analysis, which compares the trained model against both the PFUS1
  label and the manual label on the same frames; sampling frames the model was
  trained on would make that comparison meaningless.
* Selection is **blind to model predictions**. Frames are chosen by image and
  label geometry alone, so the audit cannot be accused of being seeded with the
  model's own failure cases. Model Dice is attached afterwards, in a separate
  file, never in the annotator-facing package.
* Within each patient, frames are chosen by farthest-point sampling in a
  globally z-scored feature space, so each patient contributes its own extremes
  and the union spans the imaging conditions listed in the protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

FEATURES = (
    "gt_area_ratio",
    "lumen_contrast",
    "boundary_gradient",
    "shadow_ratio",
    "irregularity",
    "global_contrast",
    "area_change",
)


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    gx = ndimage.sobel(image, axis=1, mode="nearest")
    gy = ndimage.sobel(image, axis=0, mode="nearest")
    return np.hypot(gx, gy)


def frame_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Geometry and appearance descriptors used only for stratified sampling."""
    fov = image > 8.0
    fov_area = max(int(fov.sum()), 1)
    gt = mask > 0
    gt_area = int(gt.sum())

    # Local contrast: lumen interior against a ring just outside it, not against
    # the whole frame, so bright near-field structures do not dominate.
    ring = ndimage.binary_dilation(gt, iterations=15) & ~ndimage.binary_erosion(gt, iterations=2) & fov & ~gt
    inside = image[gt] if gt_area else np.array([0.0])
    outside = image[ring] if ring.any() else image[fov & ~gt]
    lumen_contrast = float(outside.mean() - inside.mean())

    band = ndimage.binary_dilation(gt, iterations=3) & ~ndimage.binary_erosion(gt, iterations=3)
    grad = sobel_magnitude(image)
    boundary_gradient = float(grad[band].mean()) if band.any() else 0.0

    # Acoustic shadow proxy: A-lines (columns) whose deep half is far darker
    # than the 90th-percentile column, computed inside the FOV only.
    half = image.shape[0] // 2
    deep = np.where(fov[half:], image[half:], np.nan)
    with np.errstate(invalid="ignore"):
        col = np.nanmean(deep, axis=0)
    valid = np.isfinite(col)
    if valid.sum() > 8:
        reference = float(np.percentile(col[valid], 90))
        shadow_ratio = float((col[valid] < 0.25 * reference).mean())
    else:
        shadow_ratio = 0.0

    perimeter = float(band.sum()) / 6.0 if band.any() else 0.0
    irregularity = float(perimeter**2 / (4 * np.pi * gt_area)) if gt_area and perimeter else 1.0

    return {
        "gt_area_px": float(gt_area),
        "gt_area_ratio": gt_area / fov_area,
        "lumen_contrast": lumen_contrast,
        "boundary_gradient": boundary_gradient,
        "shadow_ratio": shadow_ratio,
        "irregularity": irregularity,
        "global_contrast": float(image[fov].std()),
        "mean_intensity": float(image[fov].mean()),
    }


def farthest_point_sample(matrix: np.ndarray, k: int) -> list[int]:
    """Pick k rows spanning the feature space, starting from the medoid."""
    if len(matrix) <= k:
        return list(range(len(matrix)))
    centre = matrix.mean(axis=0)
    chosen = [int(np.argmin(np.linalg.norm(matrix - centre, axis=1)))]
    while len(chosen) < k:
        distance = np.min(
            np.linalg.norm(matrix[:, None, :] - matrix[chosen][None, :, :], axis=2), axis=1
        )
        distance[chosen] = -1.0
        chosen.append(int(np.argmax(distance)))
    return sorted(chosen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/home/rosotauser/datasets/pfus/manifest.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--frames-per-patient", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    out = Path(args.output_dir)
    root = Path(args.manifest).parent
    for sub in ("images", "original_masks", "overlays", "images_blinded", "labelme", "manual_masks"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    with open(args.manifest, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == args.split]
    by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(row)
    for records in by_patient.values():
        records.sort(key=lambda r: int(r["frame_index"]))

    print(f"[features] {len(rows)} frames / {len(by_patient)} patients in split '{args.split}'")
    records: list[dict] = []
    for patient in sorted(by_patient):
        previous_area = None
        for row in by_patient[patient]:
            image = np.asarray(Image.open(root / row["image_path"]).convert("L"), dtype=np.float32)
            mask = np.asarray(Image.open(root / row["mask_path"]))
            if mask.shape != image.shape:  # defensive: never silently mis-pair
                raise RuntimeError(f"shape mismatch for {row['image_path']}")
            feature = frame_features(image, mask)
            feature["area_change"] = (
                abs(feature["gt_area_px"] - previous_area) / max(previous_area, 1.0)
                if previous_area is not None
                else 0.0
            )
            previous_area = feature["gt_area_px"]
            records.append(
                {
                    "patient_id": patient,
                    "frame_index": int(row["frame_index"]),
                    "frame_id": Path(row["image_path"]).stem,
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "height": image.shape[0],
                    "width": image.shape[1],
                    **feature,
                }
            )
        print(f"  {patient}: {len(by_patient[patient])} frames")

    matrix = np.array([[r[f] for f in FEATURES] for r in records], dtype=float)
    mu, sd = matrix.mean(axis=0), matrix.std(axis=0)
    sd[sd == 0] = 1.0
    z = (matrix - mu) / sd

    # Global condition flags, so the manifest states which situations are covered.
    def tertile(name: str) -> tuple[float, float]:
        values = np.array([r[name] for r in records])
        return float(np.percentile(values, 33)), float(np.percentile(values, 67))

    cuts = {name: tertile(name) for name in ("gt_area_ratio", "lumen_contrast", "boundary_gradient", "shadow_ratio", "irregularity", "global_contrast", "area_change")}
    for record in records:
        lo, hi = cuts["gt_area_ratio"]
        record["bladder_size"] = "small" if record["gt_area_ratio"] < lo else ("large" if record["gt_area_ratio"] > hi else "medium")
        lo, hi = cuts["lumen_contrast"]
        record["contrast_level"] = "low" if record["lumen_contrast"] < lo else ("high" if record["lumen_contrast"] > hi else "medium")
        lo, hi = cuts["boundary_gradient"]
        record["boundary_visibility"] = "weak" if record["boundary_gradient"] < lo else ("clear" if record["boundary_gradient"] > hi else "moderate")
        record["shadow_present"] = int(record["shadow_ratio"] > cuts["shadow_ratio"][1])
        record["irregular_contour"] = int(record["irregularity"] > cuts["irregularity"][1])
        record["deformation"] = int(record["area_change"] > cuts["area_change"][1])

    index_by_patient: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        index_by_patient[record["patient_id"]].append(i)

    selected: list[int] = []
    for patient in sorted(index_by_patient):
        idx = np.array(index_by_patient[patient])
        picked = farthest_point_sample(z[idx], args.frames_per_patient)
        selected.extend(int(idx[p]) for p in picked)
    selected.sort(key=lambda i: (records[i]["patient_id"], records[i]["frame_index"]))
    print(f"[select] {len(selected)} frames from {len(index_by_patient)} patients")

    rng = random.Random(args.seed)
    blind_order = list(range(len(selected)))
    rng.shuffle(blind_order)
    blind_id = {sample: f"blind_{position:04d}" for position, sample in enumerate(blind_order)}

    manifest_rows, key_rows = [], []
    for position, index in enumerate(selected):
        record = records[index]
        sample_id = f"S{position:04d}_{record['patient_id']}_{record['frame_id']}"
        image = Image.open(root / record["image_path"]).convert("L")
        mask = Image.open(root / record["mask_path"]).convert("L")

        image.save(out / "images" / f"{sample_id}.png")
        mask.save(out / "original_masks" / f"{sample_id}.png")
        image.save(out / "images_blinded" / f"{blind_id[position]}.png")

        overlay = Image.merge("RGB", (image, image, image))
        binary = np.asarray(mask) > 0
        edge = binary & ~ndimage.binary_erosion(binary, iterations=2)
        pixels = np.asarray(overlay).copy()
        pixels[edge] = (0, 229, 214)
        Image.fromarray(pixels).save(out / "overlays" / f"{sample_id}.png")

        (out / "labelme" / f"{blind_id[position]}.json").write_text(
            json.dumps(
                {
                    "version": "5.4.1",
                    "flags": {"uncertain_boundary": False},
                    "shapes": [],
                    "imagePath": f"../images_blinded/{blind_id[position]}.png",
                    "imageData": None,
                    "imageHeight": record["height"],
                    "imageWidth": record["width"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        manifest_rows.append(
            {
                "sample_id": sample_id,
                "patient_id": record["patient_id"],
                "frame_id": record["frame_id"],
                "image_path": f"images/{sample_id}.png",
                "original_mask_path": f"original_masks/{sample_id}.png",
                "manual_mask_path": f"manual_masks/{sample_id}.png",
                "blinded_image_path": f"images_blinded/{blind_id[position]}.png",
                "labelme_json_path": f"labelme/{blind_id[position]}.json",
                "frame_index": record["frame_index"],
                "height": record["height"],
                "width": record["width"],
                "gt_area_px": int(record["gt_area_px"]),
                "gt_area_ratio": round(record["gt_area_ratio"], 5),
                "lumen_contrast": round(record["lumen_contrast"], 2),
                "boundary_gradient": round(record["boundary_gradient"], 2),
                "shadow_ratio": round(record["shadow_ratio"], 4),
                "irregularity": round(record["irregularity"], 3),
                "bladder_size": record["bladder_size"],
                "contrast_level": record["contrast_level"],
                "boundary_visibility": record["boundary_visibility"],
                "shadow_present": record["shadow_present"],
                "irregular_contour": record["irregular_contour"],
                "deformation": record["deformation"],
                "annotation_quality": "",
                "uncertain_boundary": "",
            }
        )
        key_rows.append({"blind_id": blind_id[position], "sample_id": sample_id, "patient_id": record["patient_id"], "frame_id": record["frame_id"]})

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (out / "blind_key.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(key_rows, key=lambda r: r["blind_id"]))

    feature_columns = [
        "patient_id", "frame_index", "frame_id", "gt_area_px", "gt_area_ratio", "lumen_contrast",
        "boundary_gradient", "shadow_ratio", "irregularity", "global_contrast", "mean_intensity",
        "area_change", "bladder_size", "contrast_level", "boundary_visibility",
        "shadow_present", "irregular_contour", "deformation",
    ]
    with (out.parent / "dataset_audit" / f"frame_features_{args.split}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=feature_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    (out / "sampling_provenance.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "seed": args.seed,
                "frames_per_patient": args.frames_per_patient,
                "num_patients": len(index_by_patient),
                "num_samples": len(selected),
                "selection": "farthest-point sampling in globally z-scored feature space, per patient",
                "features": list(FEATURES),
                "tertile_cuts": cuts,
                "blind_to_model_predictions": True,
                "rationale": "test split only, so Phase 6/7 model-vs-label comparison uses unseen data",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[coverage]")
    for field in ("bladder_size", "contrast_level", "boundary_visibility"):
        counts = defaultdict(int)
        for row in manifest_rows:
            counts[row[field]] += 1
        print(f"  {field:20s} {dict(counts)}")
    for field in ("shadow_present", "irregular_contour", "deformation"):
        print(f"  {field:20s} {sum(r[field] for r in manifest_rows)}/{len(manifest_rows)}")
    print(f"  patients             {len({r['patient_id'] for r in manifest_rows})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
