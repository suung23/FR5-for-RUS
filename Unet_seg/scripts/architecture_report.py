#!/usr/bin/env python3
"""Print an architecture report and compare parameter counts with the paper.

Without arguments it reports every reference configuration, including the two
Slim U-Net decoder variants that were compared to resolve the paper's
under-specified decoder (see README, "Parameter-count reconstruction").

Examples:
    python scripts/architecture_report.py
    python scripts/architecture_report.py --config configs/slim_unet_paper.yaml --layers 40
"""

from __future__ import annotations

import argparse
import json

from _common import REPO_ROOT  # noqa: F401  (path setup)

from rus_perception.models.registry import build_model
from rus_perception.models.report import build_architecture_report
from rus_perception.models.slim_unet import INFERRED_DETAILS, PAPER_PARAMETER_COUNTS
from rus_perception.utils.config import load_config

REFERENCE_CONFIGS: dict[str, dict] = {
    "StandardUNet (paper preset)": {
        "name": "standard_unet",
        "preset": "paper",
        "input_channels": 1,
        "output_channels": 1,
    },
    "StandardUNet (original milesial)": {
        "name": "standard_unet",
        "preset": "original",
        "input_channels": 1,
        "output_channels": 1,
    },
    "SlimUNet (paper preset, single-conv decoder)": {"name": "slim_unet", "preset": "paper"},
    "SlimUNet (double-conv decoder variant)": {
        "name": "slim_unet",
        "preset": "paper",
        "decoder_convs": 2,
    },
}

REFERENCE_TARGET = {
    "StandardUNet (paper preset)": PAPER_PARAMETER_COUNTS["standard_unet"],
    "SlimUNet (paper preset, single-conv decoder)": PAPER_PARAMETER_COUNTS["slim_unet"],
    "SlimUNet (double-conv decoder variant)": PAPER_PARAMETER_COUNTS["slim_unet"],
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report model architecture, parameter counts and estimated MACs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Report a single config instead")
    parser.add_argument("--input-size", type=int, nargs=2, default=[128, 128], metavar=("H", "W"))
    parser.add_argument("--layers", type=int, default=0, help="Number of layer rows to print")
    parser.add_argument("--json", default=None, help="Write the full report to this JSON file")
    return parser.parse_args()


def main() -> int:
    args = get_args()
    height, width = args.input_size

    if args.config:
        config = load_config(args.config)
        model_config = config.section("model")
        size = model_config.get("input_size") or [height, width]
        model = build_model(model_config)
        report = build_architecture_report(
            model, (1, int(model_config.get("input_channels", 1)), int(size[0]), int(size[1]))
        )
        print(report.to_table(max_layers=args.layers or None))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as handle:
                handle.write(report.to_json())
        return 0

    print("=" * 100)
    print("Parameter-count reconstruction against the values reported by Raina et al.")
    print(f"  Standard U-Net reported: {PAPER_PARAMETER_COUNTS['standard_unet']:,}")
    print(f"  Slim U-Net     reported: {PAPER_PARAMETER_COUNTS['slim_unet']:,}")
    print("=" * 100)

    reports = {}
    for label, model_config in REFERENCE_CONFIGS.items():
        model = build_model(model_config)
        report = build_architecture_report(
            model,
            (1, int(model_config.get("input_channels", 1)), height, width),
            reference_parameter_count=REFERENCE_TARGET.get(label),
            model_name=label,
        )
        reports[label] = report.to_dict()
        target = REFERENCE_TARGET.get(label)
        difference = "" if target is None else f"  diff={report.trainable_parameters - target:+,}"
        verdict = ""
        if target is not None:
            verdict = "  EXACT MATCH" if report.trainable_parameters == target else "  (rejected)"
        macs = f"{report.estimated_macs / 1e9:.3f} GMACs" if report.estimated_macs else "n/a"
        print(
            f"{label:<46} params={report.trainable_parameters:>11,}{difference}{verdict}\n"
            f"{'':<46} output={tuple(report.output_shape)}  {macs} at {height}x{width}"
        )
        if args.layers:
            print(report.to_table(max_layers=args.layers))
            print()

    print("\nImplementation details inferred from the paper:")
    for index, detail in enumerate(INFERRED_DETAILS, start=1):
        print(f"  {index}. {detail}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(reports, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
