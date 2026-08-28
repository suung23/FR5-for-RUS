#!/usr/bin/env python3
r"""Robot normal force 를 전자저울 ground truth 와 대조한다.

    python3 run_force_validation_analysis.py \\
      --robot-captures captures/robot_force_captures.csv \\
      --scale-ground-truth captures/ground_truth_scale_template.csv \\
      --gravity-validation captures/gravity_compensation_validation.csv \\
      --scale-resolution-g 0.1 \\
      --output-dir outputs

``--gravity-validation`` 이 없으면 멈추지 않는다. "gravity compensation
uncertainty unavailable" 을 명시하고 robot–scale 대조만 한다.

**임의로 만들지 않고 임의로 버리지 않는다.** 값이 빠진 trial 은 사유와 함께 표에
남고, 중력 잔차는 robot 값에서 다시 빼지 않는다.
"""

from __future__ import annotations

import argparse
import os
import sys

from fv.analysis import (
    TRIAL_COLUMNS,
    GravityMetrics,
    accuracy_metrics,
    assign_gravity_error,
    build_trials,
    finalise_uncertainty,
    load_gravity_validation,
    repeatability,
    scale_standard_uncertainty,
)
from fv.report import analysis_report, gravity_report
from fv.tables import InputError, read_rows, write_rows

GRAVITY_METRIC_COLUMNS = ("metric", "unit", "value", "definition")


def _pipeline_state(robot_path: str) -> dict:
    """Capture 표가 기록한 파이프라인 상태를 요약한다."""
    rows = read_rows(robot_path)
    out = {}
    for column, label in (
        ("sensor_calibration_valid", "Sensor calibration"),
        ("gravity_compensation_valid", "Gravity compensation"),
        ("probe_frame_transform_valid", "Probe-frame transform"),
    ):
        values = {(row.get(column) or "").strip().upper() for row in rows}
        values.discard("")
        out[label] = ", ".join(sorted(values)) if values else "not recorded"
    return out


def _gravity_metric_rows(gravity: GravityMetrics, scale_u: float, combined_u: float) -> list:
    """Tidy 형식의 중력 지표. 값이 없으면 빈 칸으로 두고 정의는 남긴다."""
    have = gravity.available
    spec = [
        ("gravity_bias_N", "N", gravity.bias_n if have else None,
         "Mean of the unloaded normal-axis residual"),
        ("gravity_residual_sd_N", "N", gravity.sd_n if have else None,
         "Sample standard deviation of the residual"),
        ("gravity_rmse_N", "N", gravity.rmse_n if have else None,
         "Root mean square of the residual"),
        ("gravity_max_abs_error_N", "N", gravity.max_abs_n if have else None,
         "Largest absolute residual over validation poses"),
        ("gravity_95pct_lower_N", "N", gravity.lower_95_n if have else None,
         "mean - 1.96 x SD"),
        ("gravity_95pct_upper_N", "N", gravity.upper_95_n if have else None,
         "mean + 1.96 x SD"),
        ("gravity_robust_2p5pct_N", "N", gravity.robust_2p5_n if have else None,
         "2.5th percentile of the residual"),
        ("gravity_robust_97p5pct_N", "N", gravity.robust_97p5_n if have else None,
         "97.5th percentile of the residual"),
        ("scale_standard_uncertainty_N", "N", scale_u,
         "Scale resolution / sqrt(12)"),
        ("combined_reference_uncertainty_N", "N", combined_u,
         "sqrt(u_scale^2 + u_gravity^2); reported, never subtracted"),
        ("n_gravity_validation_poses", "count", gravity.n_poses if have else 0,
         "Unloaded validation poses used"),
    ]
    return [
        {
            "metric": name,
            "unit": unit,
            "value": "" if value is None else (
                f"{value:.6f}" if unit == "N" else str(int(value))
            ),
            "definition": definition,
        }
        for name, unit, value, definition in spec
    ]


def main(argv=None) -> int:
    """분석을 돌려 표·문서·그림을 낸다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot-captures", required=True)
    parser.add_argument("--scale-ground-truth", required=True)
    parser.add_argument("--gravity-validation", default=None)
    parser.add_argument("--scale-resolution-g", type=float, default=0.1)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--sign-note", default="negated at capture (compression positive)",
        help="capture 표에 부호 규약이 없을 때 문서에 적을 문장",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.scale_resolution_g <= 0:
        print("--scale-resolution-g 는 0 보다 커야 한다", file=sys.stderr)
        return 2

    try:
        trials = build_trials(args.robot_captures, args.scale_ground_truth)
        pipeline_state = _pipeline_state(args.robot_captures)
    except InputError as exc:
        print(f"입력 오류:\n  {exc}", file=sys.stderr)
        return 2

    gravity = GravityMetrics()
    if args.gravity_validation:
        if not os.path.exists(args.gravity_validation):
            print(f"  ⚠️ gravity compensation uncertainty unavailable — "
                  f"파일이 없다: {args.gravity_validation}")
        else:
            try:
                gravity = load_gravity_validation(args.gravity_validation)
            except InputError as exc:
                print(f"중력 검증 파일 오류:\n  {exc}", file=sys.stderr)
                return 2
    else:
        print("  ⚠️ gravity compensation uncertainty unavailable — "
              "--gravity-validation 이 없다")

    orientation_available = any(t.orientation is not None for t in trials)
    method = assign_gravity_error(trials, gravity, orientation_available)
    combined_u = finalise_uncertainty(trials, gravity, args.scale_resolution_g)
    scale_u = scale_standard_uncertainty(args.scale_resolution_g)

    accuracy = accuracy_metrics(trials)
    repeat_levels = repeatability(trials)

    sign_notes = {
        (row.get("force_sign_convention") or "").strip()
        for row in read_rows(args.robot_captures)
    }
    sign_notes.discard("")
    sign_note = ", ".join(sorted(sign_notes)) if sign_notes else args.sign_note

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    write_rows(os.path.join(out, "trial_results.csv"), TRIAL_COLUMNS, [
        {
            "sample_id": t.sample_id,
            "scale_mass_g": "" if t.scale_mass_g is None else f"{t.scale_mass_g:.4f}",
            "scale_reference_force_N": "" if t.scale_n is None else f"{t.scale_n:.6f}",
            "robot_normal_force_instant_N": "" if t.robot_n is None else f"{t.robot_n:.6f}",
            "residual_robot_minus_scale_N": (
                "" if t.residual_n is None else f"{t.residual_n:.6f}"),
            "estimated_gravity_error_normal_N": (
                "" if t.gravity_error_n is None else f"{t.gravity_error_n:.6f}"),
            "estimated_gravity_uncertainty_normal_N": (
                "" if t.gravity_uncertainty_n is None
                else f"{t.gravity_uncertainty_n:.6f}"),
            "scale_standard_uncertainty_N": f"{t.scale_uncertainty_n:.6f}",
            "combined_reference_uncertainty_N": (
                "" if t.combined_uncertainty_n is None
                else f"{t.combined_uncertainty_n:.6f}"),
            "included_in_primary_analysis": "TRUE" if t.included else "FALSE",
            "exclusion_reason": t.exclusion_reason,
        }
        for t in trials
    ])

    metric_rows = []
    if accuracy is not None:
        metric_rows = [
            ("n_included", "count", accuracy.n),
            ("mean_bias_N", "N", accuracy.bias_n),
            ("mae_N", "N", accuracy.mae_n),
            ("rmse_N", "N", accuracy.rmse_n),
            ("max_abs_error_N", "N", accuracy.max_abs_n),
            ("residual_sd_N", "N", accuracy.residual_sd_n),
            ("full_scale_N", "N", accuracy.full_scale_n),
            ("mae_pct_fs", "%", accuracy.mae_pct_fs),
            ("rmse_pct_fs", "%", accuracy.rmse_pct_fs),
            ("max_error_pct_fs", "%", accuracy.max_pct_fs),
            ("regression_slope", "-", accuracy.slope),
            ("regression_intercept_N", "N", accuracy.intercept),
            ("regression_r_squared", "-", accuracy.r_squared),
            ("slope_ci_low", "-", accuracy.slope_ci[0]),
            ("slope_ci_high", "-", accuracy.slope_ci[1]),
            ("intercept_ci_low_N", "N", accuracy.intercept_ci[0]),
            ("intercept_ci_high_N", "N", accuracy.intercept_ci[1]),
            ("bland_altman_bias_N", "N", accuracy.ba_bias_n),
            ("bland_altman_loa_lower_N", "N", accuracy.ba_lower_n),
            ("bland_altman_loa_upper_N", "N", accuracy.ba_upper_n),
        ]
    write_rows(os.path.join(out, "metrics_summary.csv"), ("metric", "unit", "value"), [
        {"metric": name, "unit": unit,
         "value": f"{value:.6f}" if isinstance(value, float) else str(value)}
        for name, unit, value in metric_rows
    ])

    write_rows(os.path.join(out, "gravity_compensation_metrics.csv"),
               GRAVITY_METRIC_COLUMNS,
               _gravity_metric_rows(gravity, scale_u, combined_u))

    with open(os.path.join(out, "analysis_report.md"), "w", encoding="utf-8") as handle:
        handle.write(analysis_report(
            trials, accuracy, gravity, repeat_levels, combined_u, scale_u,
            args.scale_resolution_g, method, sign_note, pipeline_state,
        ))
    with open(os.path.join(out, "gravity_compensation_report.md"), "w",
              encoding="utf-8") as handle:
        handle.write(gravity_report(gravity, method, scale_u, combined_u))

    from fv import plotting

    written = []
    if accuracy is not None:
        written += plotting.normal_force_validation(
            trials, accuracy, combined_u, repeat_levels,
            os.path.join(out, "normal_force_validation"),
        )
    else:
        print("  ⚠️ 유효 trial 이 2 개 미만이라 Figure 1 을 그리지 않는다")
    if gravity.available:
        written += plotting.gravity_validation(
            gravity, os.path.join(out, "gravity_compensation_validation"))
    else:
        print("  ⚠️ 중력 검증 자료가 없어 Figure 2 를 그리지 않는다")

    used = sum(1 for t in trials if t.included)
    print()
    print(f"  trial {len(trials)} 개 · 유효 {used} · 제외 {len(trials) - used}")
    if accuracy is not None:
        print(f"  bias {accuracy.bias_n:+.4f} N · MAE {accuracy.mae_n:.4f} N · "
              f"RMSE {accuracy.rmse_n:.4f} N · max {accuracy.max_abs_n:.4f} N")
        print(f"  slope {accuracy.slope:.4f} · intercept {accuracy.intercept:+.4f} N · "
              f"R² {accuracy.r_squared:.5f}")
        print(f"  Bland–Altman bias {accuracy.ba_bias_n:+.4f} N · "
              f"LoA {accuracy.ba_lower_n:+.4f} to {accuracy.ba_upper_n:+.4f} N")
    if gravity.available:
        print(f"  gravity {gravity.bias_n:+.4f} ± {gravity.sd_n:.4f} N · "
              f"RMSE {gravity.rmse_n:.4f} N · {method}")
    print(f"  u_scale {scale_u:.5f} N · u_combined {combined_u:.5f} N")
    print()
    for name in sorted(os.listdir(out)):
        print(f"    {os.path.join(out, name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
