#!/usr/bin/env python3
"""Phase 10 -- controlled architecture comparison.

Standard U-Net and Slim U-Net were trained on the SAME patient split, the same
preprocessing, the same augmentation, the same 40-epoch schedule and the same
loss, so they already form a controlled comparison; this script only collects
what those runs produced into one table and adds the ASSD the original
evaluation never computed.

U-Net++ and FPN+ResNet50 are NOT trained here. Doing so honestly requires the
same recipe on the same GPU, and the only torch in this environment is CPU-only
(2.13.0+cpu) -- the original runs took ~15 min each on an RTX 4090, which on CPU
would be hours per model and would still not be comparable on inference time.
The table below marks them as not run rather than filling them with numbers
produced under different conditions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "metrics"))
from segmentation_metrics import patient_bootstrap_ci  # noqa: E402

def _scalar(value):
    """evaluation.json stores some temporal fields as a scalar, some as a summary dict."""
    return float(value["mean"]) if isinstance(value, dict) else float(value)


RUNS = {
    "standard_unet": ("pfus_bladder", "checkpoints/pfus_bladder"),
    "slim_unet": ("pfus_bladder_slim", "checkpoints/pfus_bladder_slim"),
}
NOT_RUN = ("unet_plus_plus", "fpn_resnet50")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    repo = root.parents[1]
    parser.add_argument("--audit-metrics", default=str(root / "model_label_comparison.csv"))
    parser.add_argument("--output", default=str(root / "baseline_comparison" / "baseline_table.csv"))
    parser.add_argument("--summary", default=str(root / "baseline_comparison" / "baseline_comparison.json"))
    args = parser.parse_args()

    audit = list(csv.DictReader(open(args.audit_metrics, encoding="utf-8")))
    rows, payload = [], {}
    for name, (run, checkpoint_dir) in RUNS.items():
        architecture = json.loads((repo / checkpoint_dir / "architecture_report.json").read_text())
        evaluation = json.loads((repo / "runs" / run / "eval_test" / "evaluation.json").read_text())
        spatial = evaluation["spatial"]["overall"]
        latency = evaluation["latency"]["stages"]
        temporal = evaluation["temporal"]

        short = "standard" if name == "standard_unet" else "slim"
        subset = [r for r in audit if r["model"] == short and r["resolution"] == "r256"
                  and r["reference"] == "pfus"]
        patients = [r["patient_id"] for r in subset]
        assd_values = [float(r["assd"]) for r in subset if r["assd"] not in ("", "None")]
        assd_patients = [p for p, r in zip(patients, subset) if r["assd"] not in ("", "None")]
        assd = patient_bootstrap_ci(assd_values, assd_patients, n_boot=4000)

        entry = {
            "model": name,
            "params": architecture["trainable_parameters"],
            "gflops_256": round(architecture["estimated_flops"] / 1e9, 2),
            "dice_mean": round(spatial["dice"]["mean"], 4),
            "dice_median": round(spatial["dice"]["median"], 4),
            "iou_mean": round(spatial["iou"]["mean"], 4),
            "hd95_median_px": round(spatial["hd95"]["median"], 2),
            "assd_audit_px_256": round(assd["point"], 2) if assd["point"] else None,
            "assd_audit_ci95": [round(assd["ci_low"], 2), round(assd["ci_high"], 2)] if assd["point"] else None,
            "missed_bladder_rate": round(spatial["missed_bladder_rate"], 5),
            "component_failure_rate": round(spatial["connected_component_failure_rate"], 5),
            # Labelled "warped" upstream, but the flow columns are empty so no
            # motion compensation actually happened -- see Phase 9.
            "temporal_dice_uncompensated": round(_scalar(temporal["warped_temporal_dice"]), 4),
            "mask_dropout_rate": round(_scalar(temporal["mask_dropout_rate"]), 5),
            "inference_ms_mean_gpu": round(latency["inference"]["mean_ms"], 3),
            "end_to_end_ms_mean_gpu": round(latency["end_to_end"]["mean_ms"], 3),
            "end_to_end_p95_gpu": round(latency["end_to_end"]["p95_ms"], 3),
            "trained": True,
            "notes": "40 epochs, seed 42, identical split/preprocessing/augmentation",
        }
        rows.append(entry)
        payload[name] = entry

    for name in NOT_RUN:
        rows.append({
            "model": name, "params": None, "gflops_256": None, "dice_mean": None,
            "dice_median": None, "iou_mean": None, "hd95_median_px": None,
            "assd_audit_px_256": None, "assd_audit_ci95": None, "missed_bladder_rate": None,
            "component_failure_rate": None,
            "temporal_dice_uncompensated": None, "mask_dropout_rate": None,
            "inference_ms_mean_gpu": None,
            "end_to_end_ms_mean_gpu": None, "end_to_end_p95_gpu": None, "trained": False,
            "notes": "not trained: CPU-only torch in this environment; see module docstring",
        })

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(args.summary).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    header = f"{'Model':<16} {'Params':>10} {'GFLOPs':>7} {'Dice':>7} {'IoU':>7} {'HD95':>7} {'ASSD':>7} {'infer':>8} {'e2e p95':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if not row["trained"]:
            print(f"{row['model']:<16} {'-- not trained (CPU-only environment) --':>60}")
            continue
        print(f"{row['model']:<16} {row['params']:>10,} {row['gflops_256']:>7.2f} "
              f"{row['dice_mean']:>7.4f} {row['iou_mean']:>7.4f} {row['hd95_median_px']:>7.2f} "
              f"{row['assd_audit_px_256']:>7.2f} {row['inference_ms_mean_gpu']:>7.3f}m {row['end_to_end_p95_gpu']:>7.3f}")
    print("\nHD95/ASSD in pixels at 256x256. Latency from the original GPU run (RTX 4090),")
    print("not re-measured here: this environment has no CUDA torch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
