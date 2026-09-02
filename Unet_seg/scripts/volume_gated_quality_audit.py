#!/usr/bin/env python3
"""Audit segmentation and control quality on distended-bladder frames only.

What it produces
----------------
1. The cohort that survives a bladder-volume floor, at several floors, so the
   cost of the filter is visible before any number is read from it.
2. Segmentation performance on the surviving frames -- Dice, IoU, HD95 and the
   centroid error the controller actually servos on.
3. ``Q`` and every sub-score on those same frames.
4. A per-frame label table (CSV) joining all of the above, so any row can be
   traced back to a patient and frame.
5. A verdict per sub-score: what is wrong with it and what kind of change would
   fix it.

Why the verdicts need two cohorts
---------------------------------
Val and test are 11 patients each and have repeatedly disagreed on this data --
``mean_boundary_entropy`` reaches AUROC 0.772 on one and 0.522 on the other. A
verdict from a single cohort is therefore not evidence. Every term is judged on
both, and a verdict is issued only where they agree; where they do not, the term
is reported as UNSTABLE, which is itself the finding.

Volume floor
------------
``--min-gt-area-ratio`` is ground-truth bladder area over the ROI, so it is
independent of the frame grabber's crop. The default corresponds to a quarter of
the largest bladder in the PFUS test split, which is the loosest floor that
still leaves more than one patient standing; the sweep printed first shows what
tighter floors cost.

Example:
    python scripts/volume_gated_quality_audit.py --min-gt-area-ratio 0.0574
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

from rus_perception.control.quality import QUALITY_COMPONENT_NAMES

logger = logging.getLogger(__name__)

#: Dice at or above which a frame counts as usably segmented.
GOOD_DICE = 0.80

#: Centroid error, in normalized image widths, above which the measurement is
#: too far off to steer on. 0.02 is ~5 px at 256 x 256.
GOOD_CENTROID_ERROR = 0.02

#: Q at or above which the gate would admit a frame. Chosen from the operating
#: point table, not fitted; pass --gate to move it.
DEFAULT_GATE = 0.85

#: Volume floors swept for the cohort-composition table, as ground-truth
#: bladder area over the ROI.
FLOOR_SWEEP = (0.0, 0.03, 0.0574, 0.08, 0.12, 0.16)

#: A sub-score is called saturated when it spends this much of its mass at the
#: ceiling: it cannot separate anything the rest of the score does not.
CEILING = 0.99
SATURATED_FRACTION = 0.90


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """AUROC with mid-ranks. Higher ``scores`` are taken to predict ``True``."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    ok = ~np.isnan(scores)
    scores, labels = scores[ok], labels[ok]
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    sorted_scores, start = scores[order], 0
    for end in range(1, len(sorted_scores) + 1):
        if end == len(sorted_scores) or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _describe(values: Sequence[float]) -> Optional[dict]:
    clean = np.asarray([v for v in values if not np.isnan(v)], float)
    if clean.size == 0:
        return None
    return {"n": int(clean.size), "mean": float(clean.mean()),
            "sd": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
            "median": float(np.median(clean)),
            "iqr": [float(np.percentile(clean, 25)), float(np.percentile(clean, 75))]}


def load_split(run_dir: Path, manifest: str, root: str, size: tuple[int, int],
               workers: int) -> list[dict]:
    """Every evaluated frame with its metrics, quality terms and centroid error."""
    from centroid_accuracy import load_ground_truth_centroids

    states = [json.loads(line) for line in (run_dir / "control_states.jsonl").open()]
    with (run_dir / "frame_metrics.csv").open(newline="") as handle:
        metrics = {row["frame_id"]: row for row in csv.DictReader(handle)}
    truth = load_ground_truth_centroids(manifest, root,
                                        {s["frame_id"] for s in states}, size, workers)

    frames = []
    for state in states:
        row = metrics[state["frame_id"]]
        centre = truth[state["frame_id"]]
        error = (math.hypot(state["centroid_x_normalized"] - centre[0],
                            state["centroid_y_normalized"] - centre[1])
                 if centre and state["centroid_x_normalized"] is not None else float("nan"))
        roi_area = float(state["roi_area_px"])
        hd95 = row.get("hd95", "")
        frames.append({
            "frame_id": state["frame_id"],
            "patient_id": state["metadata"]["patient_id"],
            "frame_index": int(row["frame_index"]),
            "gt_area_px": int(row["target_area_px"]),
            "gt_area_ratio": int(row["target_area_px"]) / roi_area,
            "dice": float(row["dice"]),
            "iou": float(row["iou"]),
            "hd95": float(hd95) if hd95 not in ("", "nan") else float("nan"),
            "centroid_error": error,
            "centroid_error_px": error * float(state["image_width"]),
            "quality": float(state["control_quality_score"]),
            "components": {name: state["quality_components"].get(name, float("nan"))
                           for name in QUALITY_COMPONENT_NAMES},
            "valid": bool(state["valid_for_control"]),
            "rejection_reasons": list(state["rejection_reasons"]),
        })
    return frames


def label(frame: dict, gate: float, weights: dict[str, float]) -> dict:
    """Per-frame labels: how the frame did, and how the gate judged it.

    ``false_accept`` is the label that matters. Those are the frames the
    controller would have acted on and should not have, and the sub-score
    values on them are what says whether the score can be repaired by
    reweighting or needs a feature it does not have.

    Only weighted terms can be *limiting*: a zero-weight sub-score is absent
    from the aggregate, so naming it as what held a frame back would point at a
    term that cannot move the number.
    """
    seg_ok = frame["dice"] >= GOOD_DICE
    centroid_ok = (frame["centroid_error"] <= GOOD_CENTROID_ERROR
                   if not np.isnan(frame["centroid_error"]) else None)
    admitted = frame["quality"] >= gate and frame["valid"]
    available = {n: v for n, v in frame["components"].items()
                 if not np.isnan(v) and weights.get(n, 0.0) > 0.0}
    limiting = min(available, key=available.get) if available else None
    return {
        "seg_ok": seg_ok,
        "centroid_ok": centroid_ok,
        "admitted": admitted,
        "gate_outcome": ("true_accept" if admitted and seg_ok else
                         "false_accept" if admitted else
                         "false_reject" if seg_ok else "true_reject"),
        # With a geometric mean the smallest sub-score dominates, so "which term
        # is holding this frame back" is a real question with a real answer.
        "limiting_component": limiting,
        "limiting_value": available.get(limiting) if limiting else float("nan"),
    }


def cohort_sweep(frames: Sequence[dict]) -> list[dict]:
    """What each volume floor leaves standing."""
    out = []
    for floor in FLOOR_SWEEP:
        kept = [f for f in frames if f["gt_area_ratio"] >= floor]
        patients = sorted({f["patient_id"] for f in kept})
        out.append({"floor": floor, "n_frames": len(kept), "n_patients": len(patients),
                    "patients": patients,
                    "dice": _describe([f["dice"] for f in kept]),
                    "frames_fraction": len(kept) / len(frames) if frames else 0.0})
    return out


def component_evidence(frames: Sequence[dict], gate: float,
                       weights: dict[str, float]) -> dict:
    """Per sub-score: does it vary, does it rank, and does it catch failures."""
    seg_ok = np.asarray([f["dice"] >= GOOD_DICE for f in frames])
    dice = np.asarray([f["dice"] for f in frames])
    error = np.asarray([f["centroid_error"] for f in frames])
    labels = [label(f, gate, weights) for f in frames]
    false_accept = np.asarray([lab["gate_outcome"] == "false_accept" for lab in labels])

    evidence = {}
    for name in QUALITY_COMPONENT_NAMES:
        values = np.asarray([f["components"].get(name, np.nan) for f in frames], float)
        finite = values[~np.isnan(values)]
        if finite.size == 0:
            evidence[name] = {"available": 0}
            continue
        evidence[name] = {
            "available": int(finite.size),
            "mean": float(finite.mean()),
            "sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "at_ceiling": float((finite >= CEILING).mean()),
            "auroc_seg_ok": _auroc(values, seg_ok),
            "spearman_dice": _spearman(values, dice),
            "auroc_centroid_ok": _auroc(values, error <= GOOD_CENTROID_ERROR),
            # On the frames the gate wrongly admitted, was this term already low
            # (so more weight would have helped) or high like everything else
            # (so no reweighting could have caught them)?
            "mean_on_false_accepts": (float(np.nanmean(values[false_accept]))
                                      if false_accept.any() else float("nan")),
            "limiting_share": float(np.mean([lab["limiting_component"] == name
                                             for lab in labels])),
        }
    return evidence


def verdict(name: str, cohorts: dict[str, dict], weight: float,
            unfiltered: Optional[dict[str, dict]] = None) -> dict:
    """Reconcile one sub-score's evidence across cohorts into an action.

    A judgement is issued only where the cohorts agree. Disagreement is reported
    as UNSTABLE rather than averaged away: on 11 patients a term can look strong
    on one split and useless on the other, and that fact is more informative
    than the mean of the two.

    ``unfiltered`` is the same evidence computed before the volume floor. A term
    that varies there and is pinned at the ceiling here is not broken -- the
    filter has already removed the frames it exists to catch, and calling that
    saturation would be a tautology. It is reported as redundant *under this
    filter* instead, which is a statement about the experiment, not the term.
    """
    entries = [c[name] for c in cohorts.values() if c[name].get("available")]
    if not entries:
        return {"verdict": "ABSENT", "action": "Sub-score never produced; check the feature."}

    saturated = [e["at_ceiling"] >= SATURATED_FRACTION for e in entries]
    if all(saturated) and unfiltered:
        before = [unfiltered[split][name] for split in cohorts
                  if unfiltered.get(split, {}).get(name, {}).get("available")]
        if before and all(e["at_ceiling"] < SATURATED_FRACTION for e in before):
            return {"verdict": "REDUNDANT UNDER THIS FILTER",
                    "action": ("Varies on the full cohort (ceiling mass " +
                               ", ".join(f"{e['at_ceiling'] * 100:.0f}%" for e in before) +
                               ") but is pinned at 1.0 once the volume floor is applied: "
                               "the filter already selects for what it measures. Judge it "
                               "on unfiltered frames, or drop it when a volume floor is "
                               "enforced upstream.")}
    aurocs = [e["auroc_seg_ok"] for e in entries if not np.isnan(e["auroc_seg_ok"])]
    inverted = [a < 0.45 for a in aurocs]
    informative = [a >= 0.65 for a in aurocs]

    if all(saturated):
        return {"verdict": "SATURATED",
                "action": ("Spends >=90% of its mass at the ceiling on both cohorts: it "
                           "cannot separate anything. Either measure it over a region "
                           "where the decision is hard, or drop it and reclaim the weight.")}
    if all(inverted):
        return {"verdict": "INVERTED",
                "action": ("Ranks the wrong way on both cohorts. Reweighting cannot fix a "
                           "sign; remove it from the score and keep it, if at all, as a "
                           "hard validity floor.")}
    if any(inverted) or (any(informative) and not all(informative)):
        return {"verdict": "UNSTABLE",
                "action": ("Cohorts disagree (AUROC " +
                           ", ".join(f"{a:.3f}" for a in aurocs) +
                           "). Do not act on this term until a larger cohort settles it.")}
    if all(informative):
        if weight <= 0.0:
            return {"verdict": "INFORMATIVE BUT UNUSED",
                    "action": "Carries signal on both cohorts at weight 0. Give it weight."}
        return {"verdict": "WORKING", "action": "Ranks consistently; keep as is."}
    return {"verdict": "WEAK",
            "action": ("AUROC near chance on both cohorts. Low priority: extra weight "
                       "here buys nothing.")}


def failure_attribution(frames: Sequence[dict], gate: float,
                        weights: dict[str, float]) -> dict:
    """Can the wrongly-admitted frames be caught by any reweighting at all?

    If every sub-score is high on a false accept, no combination of weights over
    the current terms separates it, and the gap is a missing measurement rather
    than a tuning error. That distinction decides what work comes next.
    """
    labels = [label(f, gate, weights) for f in frames]
    bad = [(f, lab) for f, lab in zip(frames, labels)
           if lab["gate_outcome"] == "false_accept"]
    if not bad:
        return {"n_false_accepts": 0}

    all_high = [lab["limiting_value"] >= 0.8 for _, lab in bad]
    blame: dict[str, int] = defaultdict(int)
    for _, lab in bad:
        if lab["limiting_value"] < 0.8 and lab["limiting_component"]:
            blame[lab["limiting_component"]] += 1
    by_patient: dict[str, int] = defaultdict(int)
    for frame, _ in bad:
        by_patient[frame["patient_id"]] += 1

    return {
        "n_false_accepts": len(bad),
        "fraction_of_admitted": len(bad) / max(sum(lab["admitted"] for lab in labels), 1),
        "invisible_fraction": float(np.mean(all_high)),
        "dice": _describe([f["dice"] for f, _ in bad]),
        "centroid_error_px": _describe([f["centroid_error_px"] for f, _ in bad]),
        "blamed_component": dict(sorted(blame.items(), key=lambda kv: -kv[1])),
        "by_patient": dict(sorted(by_patient.items(), key=lambda kv: -kv[1])),
    }


def write_frame_labels(path: Path, cohorts: dict[str, list[dict]], gate: float,
                       weights: dict[str, float]) -> int:
    """One row per surviving frame: metrics, every sub-score, and its labels."""
    columns = (["split", "frame_id", "patient_id", "frame_index", "gt_area_px",
                "gt_area_ratio", "dice", "iou", "hd95", "centroid_error_px",
                "quality", "valid", "rejection_reasons"]
               + list(QUALITY_COMPONENT_NAMES)
               + ["seg_ok", "centroid_ok", "admitted", "gate_outcome",
                  "limiting_component", "limiting_value"])
    written = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for split, frames in cohorts.items():
            for frame in frames:
                lab = label(frame, gate, weights)
                row = {
                    "split": split, "frame_id": frame["frame_id"],
                    "patient_id": frame["patient_id"], "frame_index": frame["frame_index"],
                    "gt_area_px": frame["gt_area_px"],
                    "gt_area_ratio": round(frame["gt_area_ratio"], 5),
                    "dice": round(frame["dice"], 4), "iou": round(frame["iou"], 4),
                    "hd95": "" if np.isnan(frame["hd95"]) else round(frame["hd95"], 3),
                    "centroid_error_px": round(frame["centroid_error_px"], 3),
                    "quality": round(frame["quality"], 4), "valid": frame["valid"],
                    "rejection_reasons": ";".join(frame["rejection_reasons"]),
                    **{name: ("" if np.isnan(frame["components"][name])
                              else round(frame["components"][name], 4))
                       for name in QUALITY_COMPONENT_NAMES},
                    "seg_ok": lab["seg_ok"],
                    "centroid_ok": "" if lab["centroid_ok"] is None else lab["centroid_ok"],
                    "admitted": lab["admitted"], "gate_outcome": lab["gate_outcome"],
                    "limiting_component": lab["limiting_component"] or "",
                    "limiting_value": ("" if np.isnan(lab["limiting_value"])
                                       else round(lab["limiting_value"], 4)),
                }
                writer.writerow(row)
                written += 1
    return written


def _fmt(stat: Optional[dict], digits: int = 3) -> str:
    if stat is None:
        return "--"
    return (f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f} "
            f"({stat['median']:.{digits}f} [{stat['iqr'][0]:.{digits}f}–{stat['iqr'][1]:.{digits}f}])")


def render(report: dict, floor: float, gate: float, weights: dict[str, float]) -> str:
    lines = [
        "# Volume-gated segmentation and control-quality audit",
        "",
        f"- Volume floor: ground-truth bladder area ≥ **{floor:.4f}** of the ROI",
        f"- Gate: `valid_for_control` **and** Q ≥ **{gate:.2f}**",
        f"- A frame counts as segmented when Dice ≥ {GOOD_DICE:.2f}, and as steerable "
        f"when centroid error ≤ {GOOD_CENTROID_ERROR:.3f} "
        f"({GOOD_CENTROID_ERROR * 256:.1f} px at 256×256)",
        "",
        "## 1. What the volume floor costs",
        "",
        "| floor (GT area / ROI) | val frames | val patients | test frames | test patients | test Dice |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for val_row, test_row in zip(report["sweep"]["val"], report["sweep"]["test"]):
        mark = " **←**" if abs(val_row["floor"] - floor) < 1e-9 else ""
        lines.append(
            f"| {val_row['floor']:.4f}{mark} | {val_row['n_frames']} | {val_row['n_patients']} "
            f"| {test_row['n_frames']} | {test_row['n_patients']} "
            f"| {_fmt(test_row['dice'])} |"
        )

    lines += ["", "## 2. Segmentation on the surviving frames", "",
              "| split | frames | patients | Dice | IoU | HD95 (px) | centroid error (px) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for split, entry in report["segmentation"].items():
        lines.append(
            f"| {split} | {entry['n_frames']} | {entry['n_patients']} "
            f"| {_fmt(entry['dice'])} | {_fmt(entry['iou'])} | {_fmt(entry['hd95'], 2)} "
            f"| {_fmt(entry['centroid_error_px'], 2)} |"
        )

    lines += ["", "## 3. Q and its sub-scores on those frames", "",
              "| term | weight | val | test | val AUROC | test AUROC | limiting share (test) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    val_ev, test_ev = report["evidence"]["val"], report["evidence"]["test"]
    lines.append(
        f"| **Q (aggregate)** | — | {_fmt(report['quality']['val'])} "
        f"| {_fmt(report['quality']['test'])} "
        f"| {report['quality']['val_auroc']:.3f} | {report['quality']['test_auroc']:.3f} | — |"
    )
    for name in QUALITY_COMPONENT_NAMES:
        v, t = val_ev.get(name, {}), test_ev.get(name, {})
        if not v.get("available") or not t.get("available"):
            lines.append(f"| `{name}` | {weights.get(name, 0.0):.1f} | -- | -- | -- | -- | -- |")
            continue
        lines.append(
            f"| `{name}` | {weights.get(name, 0.0):.1f} "
            f"| {v['mean']:.3f} ± {v['sd']:.3f} | {t['mean']:.3f} ± {t['sd']:.3f} "
            f"| {v['auroc_seg_ok']:.3f} | {t['auroc_seg_ok']:.3f} "
            f"| {t['limiting_share'] * 100:.0f}% |"
        )

    lines += ["", "## 4. Metrics that need work", "",
              "Verdicts are issued only where val and test agree; where they do not, "
              "that disagreement is the finding.", "",
              "| term | weight | verdict | what to do |",
              "| --- | --- | --- | --- |"]
    for name, entry in report["verdicts"].items():
        lines.append(f"| `{name}` | {weights.get(name, 0.0):.1f} | **{entry['verdict']}** "
                     f"| {entry['action']} |")

    lines += ["", "## 5. What the gate lets through", ""]
    for split, entry in report["failures"].items():
        if not entry.get("n_false_accepts"):
            lines += [f"**{split}**: the gate admitted no poorly-segmented frame.", ""]
            continue
        lines += [
            f"**{split}** — {entry['n_false_accepts']} frames admitted with Dice < "
            f"{GOOD_DICE:.2f} ({entry['fraction_of_admitted'] * 100:.1f}% of everything "
            "admitted).",
            "",
            f"- Their Dice: {_fmt(entry['dice'])}; centroid error "
            f"{_fmt(entry['centroid_error_px'], 2)} px",
            f"- **{entry['invisible_fraction'] * 100:.0f}% have every sub-score above 0.8** "
            "— no reweighting of the current terms separates them",
            f"- Where the rest were held back: "
            + (", ".join(f"`{k}` ×{v}" for k, v in entry["blamed_component"].items())
               or "nowhere"),
            f"- Concentrated in: "
            + ", ".join(f"{k} ×{v}" for k, v in list(entry["by_patient"].items())[:5]),
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-run", type=Path, default=Path("runs/exp_seed43_qfix/best_val"))
    parser.add_argument("--test-run", type=Path, default=Path("runs/exp_seed43_qfix/best_test"))
    parser.add_argument("--config", default="configs/exp_seed43_qfix.yaml")
    parser.add_argument("--min-gt-area-ratio", type=float, default=0.0574,
                        help="Volume floor: ground-truth bladder area over the ROI.")
    parser.add_argument("--gate", type=float, default=DEFAULT_GATE)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/volume_gated_audit"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from rus_perception.utils.config import load_config
    config = load_config(args.config)
    data = config.section("data")
    size = tuple(int(v) for v in data["image_size"])
    weights = dict(config.section("control")["quality"].get("weights") or {})

    everything = {
        "val": load_split(args.val_run, str(data["manifest"]), str(data["root"]),
                          size, args.workers),
        "test": load_split(args.test_run, str(data["manifest"]), str(data["root"]),
                           size, args.workers),
    }
    kept = {split: [f for f in frames if f["gt_area_ratio"] >= args.min_gt_area_ratio]
            for split, frames in everything.items()}
    for split, frames in kept.items():
        if not frames:
            raise SystemExit(
                f"No {split} frame has a bladder at or above {args.min_gt_area_ratio}; "
                "lower --min-gt-area-ratio (the sweep in the report shows the trade-off)."
            )

    evidence = {split: component_evidence(frames, args.gate, weights)
                for split, frames in kept.items()}
    # The same evidence before the volume floor, so a term pinned only *because*
    # of the filter is not mistaken for a term that carries no information.
    unfiltered_evidence = {split: component_evidence(frames, args.gate, weights)
                           for split, frames in everything.items()}
    report = {
        "min_gt_area_ratio": args.min_gt_area_ratio,
        "gate": args.gate,
        "sweep": {split: cohort_sweep(frames) for split, frames in everything.items()},
        "segmentation": {
            split: {
                "n_frames": len(frames),
                "n_patients": len({f["patient_id"] for f in frames}),
                "patients": sorted({f["patient_id"] for f in frames}),
                "dice": _describe([f["dice"] for f in frames]),
                "iou": _describe([f["iou"] for f in frames]),
                "hd95": _describe([f["hd95"] for f in frames]),
                "centroid_error_px": _describe([f["centroid_error_px"] for f in frames]),
            } for split, frames in kept.items()
        },
        "quality": {
            "val": _describe([f["quality"] for f in kept["val"]]),
            "test": _describe([f["quality"] for f in kept["test"]]),
            "val_auroc": _auroc([f["quality"] for f in kept["val"]],
                                [f["dice"] >= GOOD_DICE for f in kept["val"]]),
            "test_auroc": _auroc([f["quality"] for f in kept["test"]],
                                 [f["dice"] >= GOOD_DICE for f in kept["test"]]),
        },
        "evidence": evidence,
        "evidence_unfiltered": unfiltered_evidence,
        "verdicts": {name: verdict(name, evidence, weights.get(name, 0.0),
                                   unfiltered_evidence)
                     for name in QUALITY_COMPONENT_NAMES},
        "failures": {split: failure_attribution(frames, args.gate, weights)
                     for split, frames in kept.items()},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2))
    rows = write_frame_labels(args.output_dir / "frame_labels.csv", kept,
                              args.gate, weights)
    text = render(report, args.min_gt_area_ratio, args.gate, weights)
    (args.output_dir / "audit.md").write_text(text)
    print(text)
    logger.info("Labelled %d frames -> %s", rows, args.output_dir / "frame_labels.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
