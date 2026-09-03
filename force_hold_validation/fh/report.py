"""표와 보고서. CSV 는 기계가, 마크다운은 사람이 읽는다."""
from __future__ import annotations

import csv
import datetime
import os


def write_csv(path: str, rows: list, columns: list) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _table(rows: list, columns: list, fmt: dict) -> str:
    if not rows:
        return "_(없음)_\n"
    head = "| " + " | ".join(columns) + " |\n"
    rule = "|" + "|".join(["---"] * len(columns)) + "|\n"
    body = ""
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            if value is None:
                cells.append("—")
            elif column in fmt:
                cells.append(fmt[column](value))
            else:
                cells.append(str(value))
        body += "| " + " | ".join(cells) + " |\n"
    return head + rule + body


def report(summary, grouped: list, figures: list, path: str) -> None:
    """무엇을 쟀고 무엇을 못 쟀는지를 같은 무게로 적는다."""
    n = lambda v: f"{v:+.3f}"
    p = lambda v: f"{v:.1f}"
    f2 = lambda v: f"{v:.2f}"

    lines = [
        "# Constant-force hold validation",
        "",
        f"Generated {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Runs",
        "",
        _table(
            [{"run": r.label, "type": r.run_type,
              "target_n": r.target, "samples": int(r.t.size),
              "included": "yes" if r.included else "no",
              "reason": r.reason or ""} for r in summary.runs],
            ["run", "type", "target_n", "samples", "included", "reason"],
            {"target_n": f2},
        ),
        "",
        "## Holding, by target",
        "",
        "**품질은 SD 로 본다.** 목표 대비 평균 편차는 대부분 데드밴드가 설계대로",
        "낸 것이지 추종 실패가 아니다 — 조절기는 밴드 안에 들어서면 `v_z = 0` 을",
        "내고 더 가지 않으므로, **밴드 안에서는 멈춘 자리가 곧 목표다.** 목표는",
        "도달할 지점이 아니라 그 둘레 밴드를 정의하는 값이다.",
        "",
        "아래에서 올라오는 경우 그 운전점은 `max(진입 문턱, 목표 − 밴드)` 이고,",
        "`error_vs_settling_n` 이 그 자리를 기준으로 한 실제 추종 오차다. SD 는",
        "밴드 폭과 무관하므로 밴드가 서로 다른 실행끼리도 **직접 비교된다** —",
        "이 표에서 대역 간 비교가 가능한 열은 그것 하나다.",
        "",
        "**판정은 접촉력 ‖F‖ 로 한다** — `us_diff_ik` 가 문턱에 대는 값과 같은 스칼라이고,",
        "`safety.max_contact_force_n` 도 같은 값에 걸린다. 법선 성분은 `fn_n` 열에",
        "증거로만 남는다.",
        "",
        "정착 구간(프로빙 진입 직후 3 s)은 버렸다 — 목표로 가는 과도이지 유지가 아니다.",
        "",
        _table(grouped,
               ["target_n", "band_n", "settling_point_n", "sd_n",
                "error_vs_settling_n", "mean_error_n", "seconds", "in_band_pct",
                "max_force_n", "travel_mm"],
               {"target_n": f2, "band_n": f2, "settling_point_n": f2,
                "sd_n": lambda v: f"{v:.3f}", "error_vs_settling_n": n,
                "mean_error_n": n, "seconds": p, "in_band_pct": p,
                "max_force_n": f2,
                "travel_mm": lambda v: "—" if v is None else f"{v:+.2f}"}),
        "",
        "`travel_mm` 은 유지 구간 동안 로봇이 프로브 축으로 간 순거리다 (+ = 전진).",
        "팬텀은 유지 중에도 이완하며 물러나므로, 힘이 잡혀 있는데 이 값이 0 이면",
        "지령이 실행되지 않은 것이다 — 힘 열만으로는 그 둘이 구별되지 않는다.",
        "",
        "## Disturbance response",
        "",
        "`travel_mm` = 교란 직전 → 창 끝의 로봇 순변위, `peak_travel_mm` = 창 안",
        "최대 변위 (+ = 조직 쪽 전진, − = 후퇴). 힘이 돌아온 것이 로봇이 받아 낸",
        "것인지는 이 두 열이 말한다.",
        "",
        _table(summary.events,
               ["run", "target_n", "direction", "step_ml", "peak_error_n",
                "time_to_peak_s", "settle_s", "recovered", "travel_mm",
                "peak_travel_mm"],
               {"target_n": f2, "peak_error_n": n, "time_to_peak_s": f2,
                "settle_s": lambda v: "—" if v is None else f"{v:.2f}",
                "travel_mm": lambda v: "—" if v is None else f"{v:+.2f}",
                "peak_travel_mm": lambda v: "—" if v is None else f"{v:+.2f}"}),
        "",
        "## Regulation evidence — did the robot actually do it?",
        "",
        "\"힘이 안 올랐다\" 는 두 가지를 못 가른다: **로봇이 흡수했는가**, 아니면",
        "**애초에 오를 상황이 아니었는가.** 물러난 거리가 그 둘을 가른다 — 힘이",
        "일정한데 팔이 뒤로 갔다면, 그 변위가 로봇이 받아 낸 양이다.",
        "",
        ("**팬텀 강성 k = %s N/mm** (개루프 실행에서 측정)."
         % ("—" if not (summary.k_n_per_m == summary.k_n_per_m)
            else f"{summary.k_n_per_m / 1000.0:.3f}")),
        "",
        _table(summary.stiffness,
               ["run", "points", "k_n_per_mm", "r_squared", "force_span_n",
                "travel_span_mm", "reason"],
               {"k_n_per_mm": lambda v: f"{v:.3f}", "r_squared": lambda v: f"{v:.4f}",
                "force_span_n": f2, "travel_span_mm": f2}),
        "",
        "`counterfactual_n` = 유지한 힘 + (물러난 거리 × k). **팔이 가만히 있었다면",
        "실렸을 힘** 이며, 이것이 안전 한계와 비교할 값이다.",
        "",
        _table(summary.evidence,
               ["run", "target_n", "direction", "travel_mm", "held_mean_n",
                "held_max_n", "absorbed_n", "counterfactual_n"],
               {"target_n": f2, "travel_mm": f2, "held_mean_n": f2,
                "held_max_n": f2, "absorbed_n": f2,
                "counterfactual_n": lambda v: "—" if v is None else f"{v:.2f}"}),
        "",
        "⚠️ 강성은 개루프 실행의 **한 지점 기울기**다. 팬텀이 비선형이면 외삽한 만큼",
        "틀리고, 누른 자리가 다르면 값도 다르다. `r_squared` 와 `force_span_n` 이",
        "그 외삽이 얼마나 먼지를 말해 준다 — 측정 구간 밖으로 크게 벗어난 반사실은",
        "근거가 아니라 추정이다.",
        "",
        "## Safety margin",
        "",
        "**전 구간을 본다** — 접근 중에 넘는 것도 넘는 것이다.",
        "",
        _table(summary.safety,
               ["run", "run_type", "target_n", "peak_force_n", "warn_force_n",
                "max_force_n", "margin_to_limit_n", "reached_warn", "exceeded_limit",
                "samples_over_limit"],
               {"target_n": f2, "peak_force_n": f2, "warn_force_n": f2,
                "max_force_n": f2, "margin_to_limit_n": f2}),
        "",
    ]

    exceeded = [r for r in summary.safety if r.get("exceeded_limit")]
    lines += [
        "## Verdict",
        "",
        ("⚠️ **한계를 넘은 실행이 있다** — "
         + ", ".join(f"{r['run']} ({r['peak_force_n']:.2f} N > {r['max_force_n']:.2f} N)"
                     for r in exceeded))
        if exceeded else
        "이 실행들에서 한계를 넘은 표본은 없다. **한계가 도달 가능했는지**는 별개 질문이며,"
        " 아래 최악 힘과 한계의 거리로 판단할 것 — 거리가 크면 시험한 것은 한계가 아니라"
        " 그 아래 대역이다.",
        "",
        "## Figures",
        "",
    ] + [f"- `{os.path.basename(path)}`" for path in figures] + [""]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
