#!/usr/bin/env python3
"""정속 힘 유지 검증 — 캡처를 표와 그림으로 바꾼다.

    python3 run_force_hold_analysis.py --runs runs --output-dir outputs

다른 회차의 정지 유지 실행을 반복으로 합치려면::

    python3 run_force_hold_analysis.py --runs runs_lap2 --pool-holds runs_sweep500 --output-dir outputs_lap2

``--pool-holds`` 디렉터리에서는 **hold 실행만**, 그것도 기준 회차와 조절기 설정이
같은 것만 유지 지표(fig2, hold_by_target)에 들어간다. 설정이 다른 것은 사유와
함께 실행 표에 남는다. 교란·안전·노출 그림은 기준 회차만으로 그린다.

**임의로 만들지 않고 임의로 버리지 않는다.** 모든 실행이 보고서에 남고, 제외된
것은 사유가 함께 적힌다. 힘은 원값으로 판정하며 표를 위해 평활하지 않는다.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

from fh import plotting
from fh.analysis import by_target, load_run, pool_holds, rank_sum_p, summarise
from fh.report import report, write_csv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs", help="캡처 디렉터리")
    parser.add_argument("--pool-holds", action="append", default=[], metavar="DIR",
                        help="이 디렉터리의 정지 유지(hold) 실행을 같은 설정일 때만 "
                             "반복으로 합친다. 여러 번 줄 수 있다.")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    paths = sorted(glob.glob(os.path.join(args.runs, "*_samples.csv")))
    if not paths:
        print(f"캡처가 없다: {args.runs}/*_samples.csv", file=sys.stderr)
        return 2

    runs = [load_run(path) for path in paths]
    pooled = []
    for directory in args.pool_holds:
        extra = [load_run(path)
                 for path in sorted(glob.glob(os.path.join(directory, "*_samples.csv")))]
        if not extra:
            print(f"합칠 캡처가 없다: {directory}/*_samples.csv", file=sys.stderr)
            return 2
        pooled += pool_holds(runs, extra)
    summary = summarise(runs, pooled_holds=pooled)
    grouped = by_target(summary.hold)

    os.makedirs(args.output_dir, exist_ok=True)
    plotting.OUT_DIR = args.output_dir

    write_csv(os.path.join(args.output_dir, "hold_by_target.csv"), grouped,
              ["target_n", "band_n", "settling_point_n", "runs", "sessions", "samples",
               "seconds", "sd_n", "error_vs_settling_n", "mean_error_n", "rmse_n",
               "rmse_pct_of_target", "in_band_pct", "p95_abs_error_n", "max_force_n",
               "travel_mm"])
    # 실행 하나가 한 행 — 합쳐진 회차별 값은 여기서 본다.
    write_csv(os.path.join(args.output_dir, "hold_runs.csv"),
              [r for r in summary.hold if r["run_type"] == "hold"],
              ["run", "session", "target_n", "band_n", "settling_point_n", "samples",
               "seconds", "sd_n", "error_vs_settling_n", "mean_error_n", "rmse_n",
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

    # 안전 한계에 걸린 대역은 개루프가 아니다 — fig5(b)·fig7 에서 회색으로 덮는다.
    contaminated = sorted({
        float(row["target_n"]) for row in summary.safety
        if str(row["exceeded_limit"]).lower() in ("true", "1")
    })

    figures = []
    figures += plotting.force_traces(runs)
    figures += plotting.hold_quality(grouped, summary.hold)
    figures += plotting.disturbance(runs, summary.events)
    figures += plotting.safety_margin(summary.safety)
    figures += plotting.regulation_evidence(runs, summary.evidence, summary.stiffness,
                                           curve=summary.stiffness_curve,
                                           events=summary.events,
                                           excluded_targets=contaminated)
    figures += plotting.excursion_recovery(runs, summary.excursions, summary.exposure)

    # 위약 대조. 두 팔이 다 있을 때만 그린다 — 한쪽만 있는 회차에서는 그릴 것이
    # 없고, 빈 축을 내보내면 보고서에 "없음" 이 아니라 "0" 처럼 보인다.
    arms = {str(row["run"])[:1] for row in summary.events}
    if {"B", "C"} <= arms:
        on = [abs(float(r["peak_error_n"])) for r in summary.events
              if str(r["run"])[:1] == "B"]
        off = [abs(float(r["peak_error_n"])) for r in summary.events
               if str(r["run"])[:1] == "C"]
        figures += plotting.placebo_comparison(
            summary.events, excluded_targets=contaminated,
            p_value=rank_sum_p(on, off),
        )

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
              ["run", "method", "points", "k_n_per_m", "k_n_per_mm", "intercept_n", "r_squared",
               "force_span_n", "travel_span_mm", "contact_depth_mm", "monotonic",
               "axis_spread_deg", "ok", "reason"])
    if summary.stiffness_curve.get("ok"):
        write_csv(os.path.join(args.output_dir, "phantom_stiffness_curve.csv"),
                  summary.stiffness_curve["curve"],
                  ["run", "target_n", "force_n", "depth_mm", "creep_mm"])

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
