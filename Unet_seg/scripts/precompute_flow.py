#!/usr/bin/env python3
"""Precompute forward and backward optical flow for every adjacent frame pair.

Recomputing dense flow every epoch is wasteful, so flow is computed once, stored
as .npz files with direction metadata, and referenced from the manifest through
the flow_forward_path / flow_backward_path columns.

Direction convention written into every file:
    forward : previous-grid, previous -> current
    backward: current-grid,  current  -> previous   (grid_sample sampling field)

Example:
    python scripts/precompute_flow.py --manifest data/synthetic/manifest.csv \
        --backend farneback --output-dir data/synthetic/flow --interval 1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from _common import REPO_ROOT  # noqa: F401  (path setup)

from src.data.io import load_grayscale
from src.data.manifest import ManifestRecord, load_manifest, write_manifest
from src.flow.backends import build_flow_backend
from src.flow.precomputed import is_valid_flow_file, save_flow_pair
from src.flow.reliability import ReliabilityConfig, compute_reliability
from src.utils.logging_utils import setup_logging

logger = logging.getLogger("precompute_flow")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute optical flow for a sequential ultrasound manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="Manifest CSV to process")
    parser.add_argument(
        "--backend",
        default="farneback",
        choices=["farneback", "raft_small", "identity"],
        help="Flow estimator",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for the .npz flow files")
    parser.add_argument("--interval", type=int, default=1, help="Frame gap of each pair")
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Write the flow paths back into the manifest",
    )
    parser.add_argument(
        "--manifest-out", default=None, help="Where to write the updated manifest"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip pairs whose flow file is already valid"
    )
    parser.add_argument(
        "--with-reliability",
        action="store_true",
        help="Also store a reliability map computed from the flow and the two frames",
    )
    parser.add_argument("--device", default="cpu", help="Device for the raft_small backend")
    return parser.parse_args()


def main() -> int:
    args = get_args()
    setup_logging("INFO")

    if args.interval < 1:
        raise SystemExit(f"--interval must be >= 1, got {args.interval}.")

    manifest = load_manifest(args.manifest)
    manifest.validate_ordering()
    manifest.validate_files(check_masks=False)

    params = {"device": args.device} if args.backend == "raft_small" else {}
    backend = build_flow_backend({"backend": args.backend, "params": params})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reliability_config = ReliabilityConfig()
    updated: dict[tuple[str, str, int], tuple[str, str]] = {}
    written = skipped = failed = 0

    for (patient, sequence), records in sorted(manifest.sequences().items()):
        if len(records) <= args.interval:
            logger.warning(
                "Sequence %s/%s has %d frame(s); too short for interval %d.",
                patient, sequence, len(records), args.interval,
            )
            continue
        for position in range(args.interval, len(records)):
            previous, current = records[position - args.interval], records[position]
            stem = f"{patient}_{sequence}_{previous.frame_index:06d}_to_{current.frame_index:06d}"
            path = output_dir / f"{stem}.npz"
            relative = str(path.relative_to(output_dir.parent)) if output_dir.parent in path.parents else str(path)

            image_previous = load_grayscale(manifest.resolve(previous.image_path))
            image_current = load_grayscale(manifest.resolve(current.image_path))
            if image_previous.shape != image_current.shape:
                logger.error(
                    "Frames %s and %s differ in size (%s vs %s); skipping this pair.",
                    previous.frame_id, current.frame_id, image_previous.shape, image_current.shape,
                )
                failed += 1
                continue

            if args.skip_existing and is_valid_flow_file(path, image_current.shape):
                skipped += 1
                updated[current.key] = (relative, relative)
                continue

            try:
                pair = backend.compute(image_previous, image_current)
            except (ImportError, ValueError, RuntimeError) as exc:
                logger.error("Flow failed for %s: %s", current.frame_id, exc)
                failed += 1
                continue

            if pair.shape != image_current.shape:
                logger.error(
                    "Backend returned flow of size %s for frames of size %s; skipping.",
                    pair.shape, image_current.shape,
                )
                failed += 1
                continue

            if args.with_reliability:
                import torch

                to_tensor = lambda flow: torch.from_numpy(flow).permute(2, 0, 1)[None].float()
                output = compute_reliability(
                    flow_backward=to_tensor(pair.backward),
                    flow_forward=to_tensor(pair.forward),
                    image_current=torch.from_numpy(image_current)[None, None].float(),
                    image_previous=torch.from_numpy(image_previous)[None, None].float(),
                    config=reliability_config,
                )
                pair.reliability = output.reliability[0, 0].cpu().numpy().astype(np.float32)

            save_flow_pair(
                path, pair, frame_id=current.frame_id, previous_frame_id=previous.frame_id
            )
            updated[current.key] = (relative, relative)
            written += 1

    logger.info("Flow files written=%d skipped=%d failed=%d -> %s", written, skipped, failed, output_dir)

    if args.update_manifest:
        records = []
        for record in manifest:
            paths = updated.get(record.key)
            records.append(
                ManifestRecord(
                    patient_id=record.patient_id,
                    sequence_id=record.sequence_id,
                    frame_index=record.frame_index,
                    image_path=record.image_path,
                    mask_path=record.mask_path,
                    timestamp=record.timestamp,
                    split=record.split,
                    flow_forward_path=paths[0] if paths else record.flow_forward_path,
                    flow_backward_path=paths[1] if paths else record.flow_backward_path,
                )
            )
        destination = Path(args.manifest_out or args.manifest)
        write_manifest(destination, records)
        logger.info("Manifest updated with flow paths: %s", destination)

    return 1 if failed and not written else 0


if __name__ == "__main__":
    raise SystemExit(main())
