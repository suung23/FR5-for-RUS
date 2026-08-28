"""분석 결과를 사람이 읽는 문서로.

수를 나열하는 것으로 끝내지 않는다. 무엇을 몇 개 뺐고 왜 뺐는지, 중력 불확도를
자세별로 줬는지 전체값으로 줬는지, 부호를 어디서 뒤집었는지 — 그것을 적지 않으면
같은 표를 두 사람이 다르게 읽는다.
"""

from __future__ import annotations

import math
from datetime import datetime


def _fmt(value, digits=3, unit="") -> str:
    """숫자를 문서용으로. ``None`` 이나 NaN 은 대시."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}{unit}"


def gravity_sentence(gravity) -> str:
    """사양이 지정한 문장 형식 그대로."""
    return (
        f"Unloaded multi-pose validation showed a gravity-compensation residual of "
        f"{gravity.bias_n:.3f} ± {gravity.sd_n:.3f} N (mean ± SD) along the probe "
        f"normal axis, with an RMSE of {gravity.rmse_n:.3f} N and a 95% error range "
        f"of {gravity.lower_95_n:.3f} to {gravity.upper_95_n:.3f} N."
    )


LIMITATION = (
    "Gravity-compensation error was estimated from unloaded multi-pose residuals. "
    "This estimate does not include contact-induced structural deformation, lateral "
    "friction, dynamic motion effects, or mechanical response uncertainty of the "
    "electronic scale."
)


def _trend_sentence(gravity) -> str:
    """자세 순서에 대한 추세를 한 문장으로.

    잔차가 잡음이면 순서와 상관이 없다. 상관이 있으면 SD 로 요약할 수 없는 계통
    성분이 남아 있다는 뜻이고, 그것을 적지 않으면 SD 만 보고 "이만큼 흔들린다" 로
    읽게 된다.
    """
    if not math.isfinite(gravity.trend_rho):
        return ("Pose-order trend was not computed (fewer than four validation "
                "poses).")
    strong = abs(gravity.trend_rho) > 0.5 and gravity.trend_p < 0.05
    verdict = (
        "This is a systematic component, not scatter: the residual is not "
        "independent of when the pose was taken, so the standard deviation "
        "understates what a single pose can be off by."
        if strong else
        "No strong ordering effect was found."
    )
    return (
        f"Residual versus pose order: Spearman rho = {gravity.trend_rho:+.3f} "
        f"(p = {gravity.trend_p:.4f}). {verdict}"
    )


def analysis_report(trials, accuracy, gravity, repeat_levels, combined_u,
                    scale_u, resolution_g, gravity_method, sign_note,
                    pipeline_state) -> str:
    """``analysis_report.md`` 본문."""
    used = [t for t in trials if t.included]
    dropped = [t for t in trials if not t.included]

    lines = [
        "# Probe normal-force validation against an electronic scale",
        "",
        f"Generated {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Captures",
        "",
        f"- Captures recorded: **{len(trials)}**",
        f"- Included in primary analysis: **{len(used)}**",
        f"- Excluded: **{len(dropped)}**",
        "",
    ]
    if dropped:
        lines += ["| sample_id | reason |", "|---|---|"]
        lines += [f"| {t.sample_id} | {t.exclusion_reason} |" for t in dropped]
        lines.append("")
    else:
        lines += ["No captures were excluded.", ""]

    lines += [
        "## Force pipeline state",
        "",
        "The capture mode reads the final corrected probe-frame wrench from the "
        "existing pipeline. It does not re-apply bias subtraction, gravity "
        "compensation, or the sensor-to-probe transform.",
        "",
        f"- Sign convention: **{sign_note}**",
    ]
    for name, value in pipeline_state.items():
        lines.append(f"- {name}: {value}")
    lines.append("")

    lines += ["## Robot versus scale accuracy", ""]
    if accuracy is None:
        lines += ["Not enough valid trials to compute accuracy (need at least 2).", ""]
    else:
        lines += [
            "| metric | value |",
            "|---|---|",
            f"| n | {accuracy.n} |",
            f"| Mean bias | {_fmt(accuracy.bias_n, 4, ' N')} |",
            f"| MAE | {_fmt(accuracy.mae_n, 4, ' N')} |",
            f"| RMSE | {_fmt(accuracy.rmse_n, 4, ' N')} |",
            f"| Max absolute error | {_fmt(accuracy.max_abs_n, 4, ' N')} |",
            f"| Residual SD | {_fmt(accuracy.residual_sd_n, 4, ' N')} |",
            f"| Full-scale reference | {_fmt(accuracy.full_scale_n, 3, ' N')} |",
            f"| MAE (%FS) | {_fmt(accuracy.mae_pct_fs, 2, ' %')} |",
            f"| RMSE (%FS) | {_fmt(accuracy.rmse_pct_fs, 2, ' %')} |",
            f"| Max error (%FS) | {_fmt(accuracy.max_pct_fs, 2, ' %')} |",
            "",
            "### Linear regression",
            "",
            "`F_robot = a · F_scale + b`",
            "",
            "| term | value | 95% CI |",
            "|---|---|---|",
            f"| slope a | {_fmt(accuracy.slope, 4)} | "
            f"{_fmt(accuracy.slope_ci[0], 4)} to {_fmt(accuracy.slope_ci[1], 4)} |",
            f"| intercept b | {_fmt(accuracy.intercept, 4, ' N')} | "
            f"{_fmt(accuracy.intercept_ci[0], 4)} to {_fmt(accuracy.intercept_ci[1], 4)} |",
            f"| R² | {_fmt(accuracy.r_squared, 5)} | |",
            "",
            "### Bland–Altman",
            "",
            f"- Bias: **{_fmt(accuracy.ba_bias_n, 4, ' N')}**",
            f"- 95% limits of agreement: "
            f"{_fmt(accuracy.ba_lower_n, 4)} to {_fmt(accuracy.ba_upper_n, 4)} N",
            "",
        ]

    lines += ["### Repeatability", ""]
    if repeat_levels:
        lines += ["| target level | n | SD | CV |", "|---|---|---|---|"]
        for level, count, sd, cv in repeat_levels:
            lines.append(f"| {level:.2f} N | {count} | {_fmt(sd, 4, ' N')} | {_fmt(cv, 2, ' %')} |")
        lines.append("")
    else:
        lines += [
            "No force level was measured more than once, so repeatability is not "
            "reported. Grouping was by scale reference force within 0.25 N.",
            "",
        ]

    lines += ["## Reference-side uncertainty", ""]
    lines += [
        f"- Scale resolution: {resolution_g:g} g "
        f"({resolution_g / 1000.0 * 9.80665:.4f} N)",
        f"- Scale standard uncertainty `u_scale = Δ/√12`: **{_fmt(scale_u, 5, ' N')}**",
    ]
    if gravity.available:
        lines.append(
            f"- Gravity-compensation uncertainty `u_gravity = σ_gravity`: "
            f"**{_fmt(gravity.sd_n, 5, ' N')}**"
        )
        lines.append(f"- Combined `u_combined = √(u_scale² + u_gravity²)`: "
                     f"**{_fmt(combined_u, 5, ' N')}**")
    else:
        lines.append("- **Gravity compensation uncertainty unavailable** — no "
                     "validation file was supplied, so only the scale term is known.")
        lines.append(f"- Combined (scale term only): **{_fmt(combined_u, 5, ' N')}**")
    lines += [
        "",
        "This uncertainty does **not** correct the robot–scale residual. It is the "
        "range within which the reference itself is known, and is reported alongside "
        "the residual rather than subtracted from it.",
        "",
    ]

    lines += ["## Gravity compensation", ""]
    if gravity.available:
        lines += [
            gravity_sentence(gravity),
            "",
            f"- Poses: {gravity.n_poses}",
            f"- Max absolute residual: {_fmt(gravity.max_abs_n, 4, ' N')}",
            f"- Robust 95% interval (2.5–97.5 percentile): "
            f"{_fmt(gravity.robust_2p5_n, 4)} to {_fmt(gravity.robust_97p5_n, 4)} N",
            f"- Per-capture gravity uncertainty applied as: **{gravity_method}**",
            "",
            _trend_sentence(gravity),
            "",
        ]
        if gravity_method == "pose-specific":
            lines += [
                "Pose-specific assignment used the nearest validation pose in "
                "orientation space (Euclidean distance over the fixed-axis XYZ "
                "angles), with a 30° cut-off. Captures with no validation pose "
                "inside that cut-off fell back to the global residual. No "
                "interpolation was performed: with this many poses a fitted "
                "surface would state more than the data supports.",
                "",
            ]
    else:
        lines += [
            "**Gravity compensation uncertainty unavailable.** No validation file "
            "was supplied, so the gravity term is absent from the combined "
            "uncertainty and no per-capture gravity error is reported.",
            "",
        ]

    lines += ["## Limitations", "", LIMITATION, ""]
    return "\n".join(lines)


def gravity_report(gravity, gravity_method: str, scale_u: float, combined_u: float) -> str:
    """``gravity_compensation_report.md`` 본문."""
    if not gravity.available:
        return (
            "# Gravity compensation validation\n\n"
            "**Gravity compensation uncertainty unavailable.** No validation file "
            "was supplied.\n"
        )
    return "\n".join([
        "# Gravity compensation validation",
        "",
        f"Generated {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        gravity_sentence(gravity),
        "",
        "| metric | value |",
        "|---|---|",
        f"| Validation poses | {gravity.n_poses} |",
        f"| Bias | {_fmt(gravity.bias_n, 4, ' N')} |",
        f"| Residual SD | {_fmt(gravity.sd_n, 4, ' N')} |",
        f"| RMSE | {_fmt(gravity.rmse_n, 4, ' N')} |",
        f"| Max absolute residual | {_fmt(gravity.max_abs_n, 4, ' N')} |",
        f"| 95% error range (mean ± 1.96 SD) | "
        f"{_fmt(gravity.lower_95_n, 4)} to {_fmt(gravity.upper_95_n, 4)} N |",
        f"| Robust 95% (2.5–97.5 pct) | "
        f"{_fmt(gravity.robust_2p5_n, 4)} to {_fmt(gravity.robust_97p5_n, 4)} N |",
        f"| Scale standard uncertainty | {_fmt(scale_u, 5, ' N')} |",
        f"| Combined reference uncertainty | {_fmt(combined_u, 5, ' N')} |",
        f"| Per-capture assignment | {gravity_method} |",
        f"| Pose-order trend (Spearman rho) | {_fmt(gravity.trend_rho, 3)} "
        f"(p = {_fmt(gravity.trend_p, 4)}) |",
        "",
        _trend_sentence(gravity),
        "",
        "The residual is reported, not removed. Subtracting it from the captured "
        "robot force would shrink the robot–scale residual without improving the "
        "measurement, and the resulting number would mean nothing.",
        "",
        "## Limitations",
        "",
        LIMITATION,
        "",
    ])
