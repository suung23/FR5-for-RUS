#!/usr/bin/env python3
"""Word accuracy of the lateral instruction for graded vocabularies.

Reads the sweep (frames.csv) and the derived thresholds
(lateral_instruction.json), classifies every frame's emitted instruction and
ground-truth displacement into words, and writes vocabulary_accuracy.json for
the manuscript's Table 6.

Vocabularies
------------
2 words   move / hold                    (direction ignored)
3 words   move left / move right / hold
5 words   3 words + two magnitude grades per direction
7 words   3 words + three magnitude grades per direction

The hold band is |x| < H with H the derived 5% boundary (1.64 sigma).  The
magnitude grades cut |x| at successive multiples of that same H (2H, 3H, ...),
so the grading introduces no constant that is not already derived; the
resulting class width H = 1.64 sigma is below the 2 sigma a class needs to be
assignable even at its centre, which is the point the numbers make.

A word matches when hold/move state, direction and grade all agree.  The
direction-error rate counts frames whose emitted and true words are both moves
with opposite signs, as a fraction of all frames.

    python scripts/vocabulary_accuracy.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "lateral_instruction"

VOCABULARIES = (
    ("move / hold", 2, 0),
    ("move left / move right / hold", 3, 0),
    ("+ two magnitude grades", 5, 1),
    ("+ three magnitude grades", 7, 2),
)


def words(x: np.ndarray, hold: float, extra_bounds: int) -> np.ndarray:
    """0 = hold; otherwise sign(x) * grade, grades cut at 2H, 3H, ..."""
    magnitude = np.abs(x)
    grade = np.ones(len(x), int)
    for k in range(extra_bounds):
        grade += (magnitude >= (k + 2) * hold).astype(int)
    return np.where(magnitude < hold, 0, np.sign(x).astype(int) * grade)


def main() -> int:
    stats = json.loads((EXPERIMENT / "lateral_instruction.json").read_text())
    hold = stats["thresholds"]["5%"]["px"]
    with (EXPERIMENT / "frames.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    emitted = np.array([float(r["instruction_px"]) for r in rows])
    truth = np.array([float(r["truth_px"]) for r in rows])

    result = {"n_frames": len(rows), "hold_threshold_px": hold,
              "sigma_px": stats["sigma_px"], "grade_step_px": hold,
              "vocabularies": []}
    for name, size, extra in VOCABULARIES:
        wp, wt = words(emitted, hold, extra), words(truth, hold, extra)
        if size == 2:
            wp, wt = np.abs(np.sign(wp)), np.abs(np.sign(wt))
        wrong_direction = (wp * wt) < 0
        result["vocabularies"].append({
            "name": name, "size": size,
            "grade_boundaries_px": [(k + 2) * hold for k in range(extra)],
            "accuracy": float((wp == wt).mean()),
            "direction_error": float(wrong_direction.mean()),
        })
        print(f"{name:32s} size={size}  accuracy={(wp == wt).mean()*100:5.1f}%  "
              f"direction error={wrong_direction.mean()*100:.2f}%")

    out = EXPERIMENT / "vocabulary_accuracy.json"
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
