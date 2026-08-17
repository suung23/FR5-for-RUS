#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX.

The exported graph emits **raw logits**, exactly like the PyTorch forward pass.
Sigmoid, thresholding and postprocessing stay outside the model so the deployed
graph matches the trained one and stays fusion-friendly for TensorRT.

Example:
    python scripts/export_onnx.py --checkpoint checkpoints/slim_unet_production/best.pt \
        --output slim_unet.onnx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _common import REPO_ROOT  # noqa: F401  (path setup)

from rus_perception.models.registry import build_model
from rus_perception.utils.checkpoint import load_checkpoint
from rus_perception.utils.logging_utils import setup_logging

logger = logging.getLogger("export_onnx")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a checkpoint to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to export")
    parser.add_argument("--output", required=True, help="Destination .onnx file")
    parser.add_argument("--config", default=None, help="Config overriding the stored model section")
    parser.add_argument(
        "--input-size", type=int, nargs=2, default=None, metavar=("H", "W"),
        help="Export resolution; defaults to the checkpoint's model.input_size",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size of the dummy input")
    parser.add_argument(
        "--static-batch", action="store_true", help="Disable the dynamic batch axis"
    )
    parser.add_argument(
        "--dynamic-spatial", action="store_true",
        help="Also mark height and width dynamic (requires runtime shapes divisible by 2**depth)",
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip onnxruntime verification")
    parser.add_argument(
        "--external-data", action="store_true",
        help="Store weights in a companion .onnx.data file instead of a single "
             "self-contained file. Only needed for models above the 2 GB protobuf limit; "
             "remember to ship both files.",
    )
    return parser.parse_args()


def main() -> int:
    args = get_args()
    setup_logging("INFO")

    import torch

    payload = load_checkpoint(args.checkpoint, map_location="cpu")
    stored = payload.get("config") or {}

    model_config = stored.get("model")
    if args.config:
        from rus_perception.utils.config import load_config

        model_config = load_config(args.config).section("model")
    if not model_config:
        raise SystemExit(
            f"{args.checkpoint} has no stored model config; pass --config explicitly."
        )

    model = build_model(model_config)
    model.load_state_dict(payload["model_state"])
    model.eval()

    size = args.input_size or model_config.get("input_size") or [128, 128]
    channels = int(model_config.get("input_channels", model_config.get("in_channels", 1)))
    dummy = torch.zeros(args.batch_size, channels, int(size[0]), int(size[1]))

    dynamic_axes = None if args.static_batch and not args.dynamic_spatial else {}
    if dynamic_axes is not None:
        if not args.static_batch:
            dynamic_axes["input"] = {0: "batch"}
            dynamic_axes["logits"] = {0: "batch"}
        if args.dynamic_spatial:
            dynamic_axes.setdefault("input", {}).update({2: "height", 3: "width"})
            dynamic_axes.setdefault("logits", {}).update({2: "height", 3: "width"})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    try:
        # A single self-contained file is the default: an .onnx that silently
        # depends on a companion .onnx.data file breaks if only the .onnx is
        # deployed. Pass --external-data to opt into the split layout.
        torch.onnx.export(
            model, dummy, str(output), external_data=args.external_data, **export_kwargs
        )
    except TypeError:
        # Older torch builds have no external_data parameter.
        torch.onnx.export(model, dummy, str(output), **export_kwargs)

    companion = Path(str(output) + ".data")
    if companion.exists() and not args.external_data:
        logger.warning(
            "The exporter still produced %s; both files must be deployed together.",
            companion,
        )
    logger.info(
        "Exported %s -> %s (%.1f MB, input %s, opset %d, raw logits output)%s",
        args.checkpoint,
        output,
        output.stat().st_size / (1024 ** 2),
        tuple(dummy.shape),
        args.opset,
        f" + {companion.name}" if companion.exists() else "",
    )

    if args.no_verify:
        return 0

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        logger.warning(
            "onnxruntime is not installed; skipping numerical verification. "
            "Install it with: pip install onnx onnxruntime"
        )
        return 0

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    reference = model(dummy).detach().numpy()
    exported = session.run(["logits"], {"input": dummy.numpy()})[0]
    difference = float(np.abs(reference - exported).max())
    logger.info("ONNX vs PyTorch max absolute logit difference: %.3e", difference)
    if difference > 1e-3:
        logger.error(
            "ONNX output differs from PyTorch by %.3e, which is larger than the 1e-3 "
            "tolerance. Do not deploy this export.", difference,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
