#!/usr/bin/env python
"""Draw the training and trust-validation report from a run's artefacts.

    python scripts/plot_report.py --run-dir runs/slim_unet_temporal --out runs/.../figures

Consumes what a run already writes -- nothing new has to be logged:

===========================  ==========================================
artefact                     figures
===========================  ==========================================
``history.json``             A1 loss, A2 validation, A3 flow, A4 optimisation
``frame_metrics.csv``        joined with the states below on ``frame_id``
``control_states.jsonl``     B1-B7 -- every trust figure
``force_log.json``           B8 -- Q_raw against contact force
``control_states.jsonl``     B9 -- latency against the frame budget
===========================  ==========================================

The one artefact a run does not produce automatically is ``force_log.json``:
``{"3.0": [q, q, ...], "3.5": [...]}``, one entry per force level with every
sample from its hold window. It comes from the Stage 1 search on the robot, not
from the perception repository, so it is optional here.

Writes ``trust_report.json`` beside the figures -- every number in them, so a
claim can be checked without re-reading a PNG.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from _common import REPO_ROOT  # noqa: F401  (path setup)

from rus_perception.control.quality import QualityConfig
from rus_perception.metrics.trust import (
    DEFAULT_ACCURACY_FLOOR,
    FrameOutcome,
    build_trust_report,
    dice_to_iou,
    force_response,
    iou_to_dice,
)
from rus_perception.utils.logging_utils import setup_logging

logger = logging.getLogger("plot_report")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines."""
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _as_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # drop NaN


def build_outcomes(
    states: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    accuracy_key: str = "dice",
) -> list[FrameOutcome]:
    """Join per-frame control states with ground-truth metrics on ``frame_id``.

    Frames present in only one file are dropped and counted: a state without a
    metric has no ground truth to be scored against, and a metric without a state
    has no quality score. Silently keeping either would put a half-populated row
    into every statistic.
    """
    by_id = {str(record.get("frame_id")): record for record in metrics}
    outcomes: list[FrameOutcome] = []
    unmatched = 0

    for state in states:
        frame_id = str(state.get("frame_id"))
        metric = by_id.get(frame_id)
        if metric is None:
            unmatched += 1
            continue
        components = state.get("quality_components") or {}
        outcomes.append(
            FrameOutcome(
                quality=_as_float(state.get("control_quality_score")),
                valid=bool(state.get("valid_for_control", False)),
                accuracy=_as_float(metric.get(accuracy_key)),
                components={k: float(v) for k, v in components.items() if v is not None},
                reasons=list(state.get("rejection_reasons") or []),
                patient_id=str(metric.get("patient_id", "")),
                sequence_id=str(metric.get("sequence_id", "")),
                frame_index=int(float(metric.get("frame_index", 0) or 0)),
            )
        )

    if unmatched:
        logger.warning(
            "%d control state(s) had no matching row in the metrics file and were dropped.",
            unmatched,
        )
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, help="Directory holding the run's artefacts.")
    parser.add_argument("--history", type=Path, help="Override: path to history.json.")
    parser.add_argument("--control-states", type=Path, help="Override: control_states.jsonl.")
    parser.add_argument("--frame-metrics", type=Path, help="Override: frame_metrics.csv.")
    parser.add_argument("--force-log", type=Path, help="Optional: {force: [q, ...]} JSON.")
    parser.add_argument("--out", type=Path, help="Output directory (default: <run-dir>/figures).")
    parser.add_argument(
        "--accuracy-floor",
        type=float,
        default=DEFAULT_ACCURACY_FLOOR,
        help="Dice at or above which a frame counts as usable. A DECISION, not a measurement.",
    )
    parser.add_argument(
        "--target-bad-rate",
        type=float,
        default=0.02,
        help="Safety budget: tolerable fraction of accepted frames that are actually bad.",
    )
    parser.add_argument(
        "--accuracy-key",
        default="dice",
        choices=["dice", "iou"],
        help=(
            "Which metric column is the ground truth. Rank-based results are identical "
            "either way -- but --accuracy-floor is read in THESE units, so switching the "
            "key without converting the floor silently changes the question."
        ),
    )
    parser.add_argument("--name", default="Q_seg", help="Name of the score under test.")
    parser.add_argument("--frame-budget-ms", type=float, default=1000.0 / 30.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    # State the floor in both units every run. Dice 0.70 is IoU 0.538; carrying a
    # Dice-shaped number over to --accuracy-key iou roughly doubles the reported
    # trusted-bad rate, and nothing in the output would otherwise say why.
    if args.accuracy_key == "dice":
        logger.info(
            "accuracy floor: Dice >= %.3f  (equivalently IoU >= %.3f)",
            args.accuracy_floor,
            dice_to_iou(args.accuracy_floor),
        )
    else:
        logger.info(
            "accuracy floor: IoU >= %.3f  (equivalently Dice >= %.3f)",
            args.accuracy_floor,
            iou_to_dice(args.accuracy_floor),
        )
        if args.accuracy_floor >= DEFAULT_ACCURACY_FLOOR:
            logger.warning(
                "--accuracy-key iou with a floor of %.2f is a much stricter criterion than "
                "the Dice floor of the same number (Dice %.2f is IoU %.3f). If %.2f was "
                "copied from a Dice threshold, convert it.",
                args.accuracy_floor,
                args.accuracy_floor,
                dice_to_iou(args.accuracy_floor),
                args.accuracy_floor,
            )

    run_dir = args.run_dir
    def resolve(explicit: Optional[Path], name: str) -> Optional[Path]:
        if explicit is not None:
            return explicit
        return (run_dir / name) if run_dir else None

    history_path = resolve(args.history, "history.json")
    states_path = resolve(args.control_states, "control_states.jsonl")
    metrics_path = resolve(args.frame_metrics, "frame_metrics.csv")
    out_dir = args.out or ((run_dir / "figures") if run_dir else Path("figures"))

    if not any([history_path, states_path]):
        parser.error("Give --run-dir, or at least one of --history / --control-states.")

    # Imported here so that --help works without matplotlib installed.
    try:
        from rus_perception.reporting import plots as P
    except ImportError as exc:
        logger.error("%s", exc)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # ---- A. training curves ------------------------------------------------
    if history_path and history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if isinstance(history, dict):
            history = history.get("history", [])
        if history:
            written += [
                P.save(P.plot_loss_decomposition(history), out_dir / "A1-loss-decomposition.png"),
                P.save(P.plot_validation_curves(history), out_dir / "A2-validation-curves.png"),
                P.save(P.plot_temporal_diagnostics(history), out_dir / "A3-temporal-diagnostics.png"),
                P.save(P.plot_optimization_health(history), out_dir / "A4-optimization-health.png"),
            ]
        else:
            logger.warning("%s contained no epochs; skipping the training figures.", history_path)
    else:
        logger.info("No history.json; skipping the training figures.")

    # ---- B. trust validation ----------------------------------------------
    report = None
    if states_path and states_path.is_file() and metrics_path and metrics_path.is_file():
        states = load_jsonl(states_path)
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            metrics = list(csv.DictReader(handle))
        outcomes = build_outcomes(states, metrics, args.accuracy_key)
        logger.info("Joined %d frames with both a quality score and ground truth.", len(outcomes))

        if outcomes:
            report = build_trust_report(
                outcomes,
                name=args.name,
                accuracy_floor=args.accuracy_floor,
                weights=QualityConfig().weights,
                target_bad_rate=args.target_bad_rate,
            )
            for line in report.summary_lines():
                logger.info("%s", line)

            quality = [o.quality for o in outcomes if o.is_scored]
            accuracy = [o.accuracy for o in outcomes if o.is_scored]
            valid = [o.valid for o in outcomes if o.is_scored]
            written += [
                P.save(
                    P.plot_quality_vs_accuracy(report, quality, accuracy, valid),
                    out_dir / "B1-quality-vs-accuracy.png",
                ),
                P.save(P.plot_risk_coverage(report), out_dir / "B2-risk-coverage.png"),
                P.save(P.plot_reliability(report), out_dir / "B3-reliability.png"),
                P.save(P.plot_roc(report), out_dir / "B4-roc.png"),
                P.save(P.plot_component_attribution(report), out_dir / "B5-component-attribution.png"),
                P.save(P.plot_reason_codes(report), out_dir / "B6-reason-codes.png"),
            ]

            by_patient: dict[str, list[float]] = {}
            for outcome in outcomes:
                if outcome.accuracy is not None and outcome.patient_id:
                    by_patient.setdefault(outcome.patient_id, []).append(outcome.accuracy)
            if by_patient:
                written.append(
                    P.save(
                        P.plot_per_patient(by_patient, args.accuracy_floor),
                        out_dir / "B7-per-patient.png",
                    )
                )

        latency = {
            "model": [
                v for v in (_as_float(s.get("inference_latency_ms")) for s in states) if v is not None
            ],
            "end-to-end": [
                v for v in (_as_float(s.get("end_to_end_latency_ms")) for s in states) if v is not None
            ],
        }
        latency = {k: v for k, v in latency.items() if v}
        if latency:
            written.append(
                P.save(P.plot_latency(latency, args.frame_budget_ms), out_dir / "B9-latency.png")
            )
    else:
        logger.info(
            "Need both control_states.jsonl and frame_metrics.csv for the trust figures; skipping."
        )

    # ---- B8. force response ------------------------------------------------
    if args.force_log and args.force_log.is_file():
        raw = json.loads(args.force_log.read_text(encoding="utf-8"))
        samples = {float(force): list(values) for force, values in raw.items()}
        response = force_response(samples)
        written.append(P.save(P.plot_force_response(response), out_dir / "B8-force-response.png"))
        logger.info(
            "Q_raw force response: unimodal=%s, turns=%s, F*=%s",
            response.is_unimodal,
            response.sign_changes,
            response.f_star,
        )
        if report is not None:
            payload = report.to_dict()
            payload["force_response"] = response.to_dict()
            (out_dir / "trust_report.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
    if report is not None and not (out_dir / "trust_report.json").is_file():
        (out_dir / "trust_report.json").write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )

    if not written:
        logger.error("Nothing was drawn. Check the input paths.")
        return 1
    for path in written:
        logger.info("wrote %s", path)
    logger.info("%d figure(s) -> %s", len(written), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
