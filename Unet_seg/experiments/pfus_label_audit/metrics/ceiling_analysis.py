#!/usr/bin/env python3
"""Phase 7 -- ceiling analysis: is the shortfall the model or the label?

Compares three Dice distributions on the SAME frames:

    D_model_pfus    Dice(prediction, PFUS1 label)
    D_model_manual  Dice(prediction, manual lumen label)
    D_pfus_manual   Dice(PFUS1 label, manual lumen label)   <- the ceiling

The last one is the ceiling: no model trained on the PFUS1 label can score
higher than D_pfus_manual against a manual lumen target, however good it is.

Case assignment (thresholds are explicit and adjustable, not hidden):

    A  label problem      D_pfus_manual low,  D_model_manual high
    B  model problem      D_pfus_manual high, D_model_manual low
    C  both               both low
    D  already fine       all high -- remaining gap is Dice's own ceiling
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmentation_metrics import compare_masks, patient_bootstrap_ci, summarise  # noqa: E402


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--audit-dir", default=str(root / "annotation_audit"))
    parser.add_argument("--predictions-dir", default=str(root / "predictions"))
    parser.add_argument("--model", default="standard")
    parser.add_argument("--output", default=str(root / "metrics" / "ceiling_analysis.json"))
    parser.add_argument("--high", type=float, default=0.80, help="Dice at or above this counts as high")
    parser.add_argument("--low", type=float, default=0.70, help="Dice below this counts as low")
    args = parser.parse_args()

    audit = Path(args.audit_dir)
    manifest = list(csv.DictReader((audit / "manifest.csv").open(encoding="utf-8")))
    missing = [r["sample_id"] for r in manifest if not (audit / r["manual_mask_path"]).exists()]
    if missing:
        payload = {
            "status": "blocked",
            "reason": "manual lumen annotation not available",
            "missing": len(missing),
            "expected": len(manifest),
            "unblocks": "run Phase 3 annotation, then metrics/rasterise_manual.py",
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Phase 7 BLOCKED: {len(missing)}/{len(manifest)} manual masks missing.")
        print("  D_pfus_manual is the whole point of this phase and cannot be estimated")
        print("  from PFUS1 alone -- there is no second opinion in the dataset.")
        return 0

    store = np.load(Path(args.predictions_dir) / f"predictions_{args.model}.npz")
    rows = []
    for record in manifest:
        pfus = load(audit / record["original_mask_path"])
        manual = load(audit / record["manual_mask_path"])
        prediction = store[record["sample_id"]].astype(bool)
        rows.append(
            {
                "sample_id": record["sample_id"],
                "patient_id": record["patient_id"],
                "d_model_pfus": compare_masks(prediction, pfus).dice,
                "d_model_manual": compare_masks(prediction, manual).dice,
                "d_pfus_manual": compare_masks(pfus, manual).dice,
            }
        )

    patients = [r["patient_id"] for r in rows]
    stats = {k: summarise([r[k] for r in rows], patients)
             for k in ("d_model_pfus", "d_model_manual", "d_pfus_manual")}
    ceiling = stats["d_pfus_manual"]["patient_bootstrap_ci_95"]["point"]
    model_manual = stats["d_model_manual"]["patient_bootstrap_ci_95"]["point"]
    model_pfus = stats["d_model_pfus"]["patient_bootstrap_ci_95"]["point"]

    if ceiling < args.low and model_manual >= args.high:
        case, verdict = "A", "PFUS1 label quality / target definition is the dominant cause."
    elif ceiling >= args.high and model_manual < args.low:
        case, verdict = "B", "Model or training strategy is the dominant cause."
    elif ceiling < args.low and model_manual < args.low:
        case, verdict = "C", "Label noise and model limitation are both present."
    elif min(ceiling, model_manual, model_pfus) >= args.high:
        case, verdict = "D", "Performance is already high; the residual gap is Dice's own ceiling."
    else:
        case, verdict = "mixed", "Falls between the defined cases; read the three numbers directly."

    exceeds = sum(1 for r in rows if r["d_model_manual"] > r["d_pfus_manual"])
    payload = {
        "status": "complete",
        "model": args.model,
        "n_samples": len(rows),
        "n_patients": len(set(patients)),
        "thresholds": {"high": args.high, "low": args.low},
        "statistics": stats,
        "case": case,
        "verdict": verdict,
        "frames_where_model_beats_pfus_label": exceeds,
        "frames_where_model_beats_pfus_label_ratio": exceeds / len(rows),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Ceiling analysis -- {args.model}, {len(rows)} frames / {len(set(patients))} patients")
    for key in ("d_model_pfus", "d_model_manual", "d_pfus_manual"):
        ci = stats[key]["patient_bootstrap_ci_95"]
        print(f"  {key:<16} {ci['point']:.4f}  95%CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")
    print(f"\n  Case {case}: {verdict}")
    print(f"  Model agreed with the manual lumen better than PFUS1 did on "
          f"{exceeds}/{len(rows)} frames ({100 * exceeds / len(rows):.0f}%)  [answers Q4]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
