#!/usr/bin/env python3
"""Phase 4 helper -- turn LabelMe polygons into binary manual_masks/*.png.

Accepts the blinded LabelMe JSONs the annotator fills in and writes one binary
mask per sample_id, at the original frame resolution. Frames the annotator left
empty become all-zero masks: an intentionally empty lumen annotation is a
result, not a missing file, and Phase 4 scores it as such.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LABEL = "bladder_lumen"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--audit-dir", default=str(root / "annotation_audit"))
    parser.add_argument("--label", default=LABEL)
    args = parser.parse_args()

    audit = Path(args.audit_dir)
    manifest = list(csv.DictReader((audit / "manifest.csv").open(encoding="utf-8")))
    key = {r["blind_id"]: r["sample_id"] for r in csv.DictReader((audit / "blind_key.csv").open(encoding="utf-8"))}
    by_sample = {r["sample_id"]: r for r in manifest}
    (audit / "manual_masks").mkdir(exist_ok=True)

    written = empty = missing = 0
    for blind_id, sample_id in sorted(key.items()):
        path = audit / "labelme" / f"{blind_id}.json"
        if not path.exists():
            missing += 1
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        shapes = [s for s in payload.get("shapes", []) if s.get("label") == args.label]
        if not shapes and payload.get("shapes"):
            labels = sorted({s.get("label") for s in payload["shapes"]})
            print(f"  ! {blind_id}: no '{args.label}' shape; found {labels}")
        record = by_sample[sample_id]
        size = (int(record["width"]), int(record["height"]))
        canvas = Image.new("L", size, 0)
        draw = ImageDraw.Draw(canvas)
        for shape in shapes:
            points = [(float(x), float(y)) for x, y in shape["points"]]
            if len(points) >= 3:
                draw.polygon(points, fill=255)
        array = np.asarray(canvas)
        if not array.any():
            empty += 1
        Image.fromarray(array).save(audit / "manual_masks" / f"{sample_id}.png")
        written += 1

    print(f"wrote {written} manual masks ({empty} intentionally empty); {missing} LabelMe files absent")
    if missing:
        print("  -> annotation is incomplete; Phase 4 will score only the frames present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
