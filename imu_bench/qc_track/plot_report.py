#!/usr/bin/env python3
"""보고서용 그림 두 장 — "프로브의 움직임을 얼마나 정확히 기록하나" 에 답하는 것만.

    python3 plot_report.py

`plot_track.py` 는 한 장에 전부 담는 진단용이다. 여기서는 사람에게 보여줄 두
질문만 그린다.

  1. 변위를 얼마나 믿을 수 있나 — 창 길이별 오차. 이 QC 의 답 자체다.
  2. 왜 6 축인가 — 같은 원시 데이터를 자력계 쓰고/안 쓰고 나눠 비교.

숫자는 전부 결과 JSON 에서 읽는다. 손으로 적은 값은 없다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import qc_common as qc                                          # noqa: E402
import reference as ref                                         # noqa: E402

BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c8c8c"


def korean_font():
    """이 PC 에 실제로 있는 한글 글꼴. monitor.py 와 같은 이유로 파일에서 등록한다."""
    import matplotlib.font_manager as fm
    for query in ("Noto Sans CJK KR", "NanumGothic", ":lang=ko"):
        try:
            r = subprocess.run(["fc-match", "-f", "%{file}", query],
                               capture_output=True, text=True, timeout=5)
        except Exception:                                       # noqa: BLE001
            continue
        path = r.stdout.strip()
        if path and os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                return fm.FontProperties(fname=path).get_name()
            except Exception:                                   # noqa: BLE001
                continue
    return "DejaVu Sans"


def fig_displacement(res, out):
    """창 길이 -> 변위 오차. **이 QC 가 묻는 것에 대한 답이 이 그림이다.**

    두 곡선을 겹친다.

      · 오차 바닥 — 정지 구간에 같은 파이프라인을 돌린 값. 이보다 잘 나올 수 없다.
      · 실제 이동 — teleop 으로 실제로 움직인 구간의 오차 중앙값.

    둘의 벌어짐이 곧 '움직임 자체가 만드는 오차' 다. 오차 바닥은 T^2 로 자라고
    (ZUPT 가 상수 바이어스를 지우면 남는 것은 자세오차가 만드는 중력 누설 a 이고
    오차가 a*T^2/8 로 자란다), 실제 이동은 그보다 가파르다.
    """
    tr = res["translation"]
    # 키가 "0.5"/"1.0" 처럼 저장돼 있어 f"{w:g}" 로 되돌리면 "1" 이 되어 어긋난다.
    floor = tr["still"]["error_floor_mm"]
    items = sorted((float(k), v) for k, v in floor.items())
    fw = [w for w, _ in items]
    fy = [v["C"] for _, v in items]
    fn = [v["n"] for _, v in items]

    # **실제로 움직인 칸만 '이동' 으로 그린다.** probe 의 0.5~1 s 칸은 이동거리
    # 중앙값이 0.005 mm 다 — 움직이지 않은 구간이라 여기에 그대로 찍으면
    # "1 초 이하는 1 mm 로 정확하다" 로 읽힌다. 그 칸은 아직 비어 있는 것이지
    # 좋은 것이 아니다. 08-21 세션에 0.5~1.5 s 짜리 실제 이동이 하나도 안 들어갔다.
    MOVED_MM = 5.0
    moved, empty = [], []
    for b in tr["probe"]["by_duration"]:
        mid = float(np.sqrt(b["lo"] * b["hi"]))
        (moved if b["dist_med_mm"] >= MOVED_MM else empty).append(
            (mid, b["eps_C_med_mm"], b["n"], b["dist_med_mm"], b["lo"], b["hi"]))

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    t_gap = min(m[4] for m in moved) if moved else 1.0
    ax.axvspan(0.3, t_gap, color="#9a9a9a", alpha=0.13, zorder=0)
    ax.text(np.sqrt(0.3 * t_gap), 0.10, "실제 이동이 없어\n아직 못 잰 구간",
            color="#555555", fontsize=9.5, ha="center", va="bottom")

    ax.plot(fw, fy, "o-", color=GREY, lw=2, ms=7, label="오차 바닥 (정지 5 분, n≥295)")
    for w, y in zip(fw, fy):
        ax.annotate(f"{y:.2f}", (w, y), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8.5, color=GREY)

    kw = sorted(ref.DISPLACEMENT_FLOOR_MM)
    ax.plot(kw, [ref.DISPLACEMENT_FLOOR_MM[k]["C"] for k in kw], "s--",
            color="#c0c0c0", lw=1.4, ms=6, label="기준 바닥 (imu_bench §9)")

    ax.plot([m[0] for m in moved], [m[1] for m in moved], "o-", color=ORANGE,
            lw=2.4, ms=9, label="실제 이동 (probe, 중앙값)")
    for mid, y, n, d, lo, hi in moved:
        ax.annotate(f"{y:.0f} mm\n{lo:g}~{hi:g}s · n={n}\n이동 {d:.0f} mm",
                    (mid, y), textcoords="offset points", xytext=(8, -4),
                    fontsize=8.5, color="#a34f00", va="center")
    for mid, y, n, d, lo, hi in empty:
        ax.plot([mid], [y], "o", mfc="white", mec=ORANGE, mew=1.8, ms=9)
        ax.annotate(f"이동 {d:.2f} mm — 안 움직인 구간이다\n(이 칸은 비어 있다)",
                    (mid, y), textcoords="offset points", xytext=(10, -2),
                    fontsize=8.5, color="#a34f00", va="center")

    # T^2 안내선 — 오차 바닥의 첫 점을 지나게 맞춘다
    tt = np.array([0.35, 7.0])
    ax.plot(tt, fy[0] * (tt / fw[0]) ** 2, ":", color="#444444", lw=1.2,
            label=r"$T^2$ 기울기")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("구간 길이 T [s]")
    ax.set_ylabel("변위 오차 [mm]  (ZUPT 적용, 중앙값)")
    ax.set_ylim(0.08, 400)
    ax.set_title("멈춤과 멈춤 사이가 길수록 변위는 급격히 못 믿게 된다",
                 fontsize=13, pad=10)
    ax.grid(alpha=0.3, which="both", lw=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  -> {out}")


def fig_sources(res6, res9, out):
    """같은 원시 데이터, 자력계만 쓰고/안 쓰고. **9 축이 왜 안 되는지 한 장.**"""
    ch = ["tilt", "tip_y", "tip_x", "spin"]
    name = {"tilt": "기울기 크기\n(tilt)", "tip_y": "기울기 y", "tip_x": "기울기 x",
            "spin": "축둘레 회전\n(spin)"}
    r6 = res6["rotation"]["host6"]["channels"]
    r9 = res9["rotation"]["chip"]["channels"]

    fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.6),
                           gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})

    y = np.arange(len(ch)); h = 0.36
    ax[0].barh(y + h / 2, [r6[c]["err_rms_delay_corrected_deg"] for c in ch], h,
               color=BLUE, label="6축 (자력계 안 씀)")
    ax[0].barh(y - h / 2, [r9[c]["err_rms_delay_corrected_deg"] for c in ch], h,
               color=RED, label="9축 (칩 RV)")
    for i, c in enumerate(ch):
        ax[0].text(r6[c]["err_rms_delay_corrected_deg"] + 0.15, i + h / 2,
                   f"{r6[c]['err_rms_delay_corrected_deg']:.2f}", va="center", fontsize=9)
        ax[0].text(r9[c]["err_rms_delay_corrected_deg"] + 0.15, i - h / 2,
                   f"{r9[c]['err_rms_delay_corrected_deg']:.2f}", va="center", fontsize=9)
    ax[0].axvline(2.0, color="#555", ls="--", lw=1.2)
    ax[0].text(2.1, -0.62, "기준 2°", fontsize=9, color="#555")
    ax[0].set_yticks(y); ax[0].set_yticklabels([name[c] for c in ch], fontsize=9)
    ax[0].set_xlabel("각오차 RMS [deg]  (지연 보정 후)")
    ax[0].set_title("자세를 얼마나 정확히 따라가나", fontsize=12)
    ax[0].legend(fontsize=9, loc="lower right")
    ax[0].grid(axis="x", alpha=0.3, lw=0.5)

    a6, a9 = res6["alignment"], res9["alignment"]
    ax[1].bar([0, 1], [a6["resid_rms_deg"], a9["resid_rms_deg"]],
              color=[BLUE, RED], width=0.55)
    for i, v in enumerate([a6["resid_rms_deg"], a9["resid_rms_deg"]]):
        ax[1].text(i, v + 0.2, f"{v:.2f}°", ha="center", fontsize=10)
    ax[1].axhline(3.0, color="#555", ls="--", lw=1.2)
    ax[1].text(-0.42, 3.12, "기준 3°", fontsize=9, color="#555", ha="left")
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["6축", "9축"])
    ax[1].set_ylabel("정렬 잔차 RMS [deg]")
    ax[1].set_title("고정회전이 풀리나", fontsize=12)
    ax[1].grid(axis="y", alpha=0.3, lw=0.5)

    d6 = abs(r6["spin"]["drift_deg_per_min"]); d9 = abs(r9["spin"]["drift_deg_per_min"])
    ax[2].bar([0, 1], [d6, d9], color=[BLUE, RED], width=0.55)
    for i, v in enumerate([d6, d9]):
        ax[2].text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=10)
    ax[2].axhline(1.0, color="#555", ls="--", lw=1.2)
    ax[2].text(-0.42, 1.12, "기준 1 °/min", fontsize=9, color="#555", ha="left")
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["6축", "9축"])
    ax[2].set_ylabel("heading 드리프트 [°/min]")
    ax[2].set_title("방위가 흐르나", fontsize=12)
    ax[2].grid(axis="y", alpha=0.3, lw=0.5)

    fig.suptitle("같은 원시 데이터 · 같은 해법 — 자력계를 쓰느냐만 다르다",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    print(f"  -> {out}")


def main() -> int:
    k = korean_font()
    plt.rcParams["font.family"] = [k, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    d = os.path.join(qc.DIR_META, "analysis")
    os.makedirs(d, exist_ok=True)
    res6 = json.load(open(os.path.join(qc.DIR_META, "qc_host6.json")))
    res9 = json.load(open(os.path.join(qc.DIR_META, "qc_chip.json")))
    fig_displacement(res6, os.path.join(d, "report_displacement.png"))
    fig_sources(res6, res9, os.path.join(d, "report_sources.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
