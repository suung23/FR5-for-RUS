#!/usr/bin/env python3
"""정속 힘 유지 검증 — 캡처를 표와 그림으로 바꾼다.

    python3 run_force_hold_analysis.py --runs runs --output-dir outputs

**임의로 만들지 않고 임의로 버리지 않는다.** 모든 실행이 보고서에 남고, 제외된
것은 사유가 함께 적힌다. 힘은 원값으로 판정하며 표를 위해 평활하지 않는다.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

from fh import plotting
from fh.analysis import by_target, load_run, summarise
from fh.report import report, write_csv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs", help="캡처 디렉터리")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    paths = sorted(glob.glob(os.path.join(args.runs, "*_samples.csv")))
    if not paths:
        print(f"캡처가 없다: {args.runs}/*_samples.csv", file=sys.stderr)
        return 2

    runs = [load_run(path) for path in paths]
    summary = summarise(runs)
    grouped = by_target(summary.hold)

    os.makedirs(args.output_dir, exist_ok=True)
    plotting.OUT_DIR = args.output_dir

    write_csv(os.path.join(args.output_dir, "hold_by_target.csv"), grouped,
              ["target_n", "band_n", "settling_point_n", "runs", "samples", "seconds",
               "sd_n", "error_vs_settling_n", "mean_error_n", "rmse_n",
               "rmse_pct_of_target", "in_band_pct", "p95_abs_error_n", "max_force_n",
               "travel_mm"])
    write_csv(os.path.join(args.output_dir, "disturbance_events.csv"), summary.events,
              ["run", "target_n", "direction", "step_ml", "onset_s", "peak_error_n",
               "peak_force_n", "time_to_peak_s", "settle_s", "recovered",
               "travel_mm", "peak_travel_mm"])
    write_csv(os.path.join(args.output_dir, "safety_margin.csv"), summary.safety,
              ["run", "run_type", "target_n", "peak_force_n", "warn_force_n",
               "max_force_n", "margin_to_limit_n", "reached_warn", "exceeded_limit",
               "samples_over_limit", "seconds_over_warn"])

    figures = []
    figures += plotting.force_traces(runs)
    figures += plotting.hold_quality(grouped)
    figures += plotting.disturbance(runs, summary.events)
    figures += plotting.safety_margin(summary.safety)
    figures += plotting.regulation_evidence(runs, summary.evidence, summary.stiffness)
    figures += plotting.excursion_recovery(runs, summary.excursions, summary.exposure)

    write_csv(os.path.join(args.output_dir, "band_excursions.csv"), summary.excursions,
              ["run", "loop", "target_n", "band_top_n", "onset_s", "duration_s",
               "peak_n", "over_band_n"])
    write_csv(os.path.join(args.output_dir, "force_exposure.csv"), summary.exposure,
              ["run", "target_n", "level_n", "episodes", "longest_s", "total_s"])

    write_csv(os.path.join(args.output_dir, "regulation_evidence.csv"), summary.evidence,
              ["run", "target_n", "direction", "onset_s", "travel_mm", "held_mean_n",
               "held_max_n", "absorbed_n", "counterfactual_n"])
    write_csv(os.path.join(args.output_dir, "disturbance_rejection.csv"), summary.rejection,
              ["run", "loop", "target_n", "band_n", "direction", "step_ml",
               "peak_delta_n", "residual_delta_n", "recovered_pct", "outside_band"])
    write_csv(os.path.join(args.output_dir, "phantom_stiffness.csv"), summary.stiffness,
              ["run", "points", "k_n_per_m", "k_n_per_mm", "intercept_n", "r_squared",
               "force_span_n", "travel_span_mm", "ok", "reason"])

    report(summary, grouped, figures, os.path.join(args.output_dir, "analysis_report.md"))

    included = sum(1 for r in runs if r.included)
    print(f"실행 {len(runs)}개 중 {included}개 포함 · 목표 {len(grouped)}개 · "
          f"교란 사건 {len(summary.events)}개 → {args.output_dir}/")
    for run in runs:
        if not run.included:
            print(f"  제외 {run.label}: {run.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
