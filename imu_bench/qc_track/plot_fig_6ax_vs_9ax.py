#!/usr/bin/env python3
"""의료정보학회 발표용 그림 — 6축 IMU vs 9축 IMU 정확도 차이 한 장.

    python3 plot_fig_6ax_vs_9ax.py

같은 원시 데이터·같은 해법이고 자력계를 쓰느냐만 다르다. 숫자는 전부 QC 결과
JSON 에서 읽는다 (손으로 적은 값 없음). plot_report.fig_sources 와 같은 값을
학회 슬라이드/논문 규격(300 dpi, 벡터 원본 동시 저장)으로 다시 그린 것이다.
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
from matplotlib.patches import FancyArrowPatch                  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import qc_common as qc                                          # noqa: E402

# 하늘색 = 6축(채택), 회색 = 9축. 흑백 인쇄에서도 명도가 갈리도록 회색을 조금 어둡게.
SKY, GREY = "#4FA3D1", "#9AA1A9"
EDGE_SKY, EDGE_GREY = "#2E7CA8", "#71787F"
INK, SUB = "#1a1f26", "#5a6068"          # 본문 먹색 / 보조 텍스트
RULE = "#000000"                          # 기준선 — 검은 파선 (텍스트 없음)

OUT = os.path.join(_HERE, "report", "figures", "fig_imu6_vs_imu9_accuracy")

CH = ["spin", "tip_x", "tip_y", "tilt"]   # 위에서 아래로 (오차 큰 순)
CH_NAME = {
    "spin":  "Axial rotation\n(spin)",
    "tip_x": "Tilt x",
    "tip_y": "Tilt y",
    "tilt":  "Tilt magnitude\n(tilt)",
}


def korean_font() -> str:
    """이 PC 에 실제로 있는 한글 글꼴을 파일에서 등록한다 (plot_report.py 와 동일)."""
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


def panel_tag(ax, tag: str) -> None:
    ax.text(-0.02, 1.055, tag, transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK, ha="right", va="baseline")


def ratio_bracket(ax, x0, x1, y, text) -> None:
    """두 막대 사이에 배수를 적는다 — '얼마나 차이 나나' 가 이 그림의 요지다."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<->", color=SUB, lw=1.1,
                                shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, y, text, ha="center", va="center", fontsize=11,
            color=INK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none"))


def two_bar(ax, v6, v9, thr, ylabel, tag, unit_fmt):
    bars = ax.bar([0, 1], [v6, v9], width=0.52,
                  color=[SKY, GREY], edgecolor=[EDGE_SKY, EDGE_GREY], linewidth=0.9,
                  zorder=3)
    top = max(v6, v9)
    for b, v in zip(bars, (v6, v9)):
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.028, unit_fmt.format(v),
                ha="center", va="bottom", fontsize=11.5, color=INK,
                fontweight="bold", zorder=4,
                bbox=dict(boxstyle="square,pad=0.14", fc="white", ec="none"))
    ax.axhline(thr, color=RULE, ls=(0, (5, 3)), lw=1.2, zorder=2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["6-axis\n(without\nmagnetometer)", "9-axis\n(reference RV)"],
                       fontsize=10, color=INK)      # 영문이 길어 한 줄 더 접는다
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(0, top * 1.30)
    ax.set_ylabel(ylabel, fontsize=11.5, color=INK)
    ax.grid(axis="y", alpha=0.28, lw=0.6, color="#b6bbc2", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8b9198")
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(labelsize=10.5, colors=INK, length=3.5, width=0.9)
    panel_tag(ax, tag)
    return top


def build(res6, res9, out_base):
    r6 = res6["rotation"]["host6"]["channels"]
    r9 = res9["rotation"]["chip"]["channels"]
    e6 = [r6[c]["err_rms_delay_corrected_deg"] for c in CH]
    e9 = [r9[c]["err_rms_delay_corrected_deg"] for c in CH]

    fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.9),
                           gridspec_kw={"width_ratios": [1.62, 1.0, 1.0]})
    fig.patch.set_facecolor("white")

    # ---- (a) 채널별 각오차 RMS -------------------------------------------
    a = ax[0]
    y = np.arange(len(CH))[::-1]                     # 위 -> 아래 = CH 순서
    h = 0.34
    a.barh(y + h / 2 + 0.02, e6, h, color=SKY, edgecolor=EDGE_SKY, lw=0.9,
           label="6-axis\n(without magnetometer)", zorder=3)
    a.barh(y - h / 2 - 0.02, e9, h, color=GREY, edgecolor=EDGE_GREY, lw=0.9,
           label="9-axis\n(reference RV)", zorder=3)
    xmax = max(e9) * 1.30
    box = dict(boxstyle="square,pad=0.14", fc="white", ec="none")   # 기준선 위에서도 읽히게
    for yy, v6, v9 in zip(y, e6, e9):
        a.annotate(f"{v6:.2f}", (v6, yy + h / 2 + 0.02), textcoords="offset points",
                   xytext=(5, 0), va="center", fontsize=10.5, color=INK,
                   fontweight="bold", zorder=4, bbox=box)
        a.annotate(f"{v9:.2f}", (v9, yy - h / 2 - 0.02), textcoords="offset points",
                   xytext=(5, 0), va="center", fontsize=10.5, color=INK,
                   fontweight="bold", zorder=4, bbox=box)
        a.annotate(f"×{v9 / v6:.1f}", (v9, yy - h / 2 - 0.02), textcoords="offset points",
                   xytext=(52, 0), va="center", fontsize=10, color=SUB,
                   zorder=4, bbox=box)
    a.axvline(2.0, color=RULE, ls=(0, (5, 3)), lw=1.2, zorder=2)

    a.set_yticks(y)
    a.set_yticklabels([CH_NAME[c] for c in CH], fontsize=11, color=INK)
    a.set_ylim(-0.72, len(CH) - 0.28)
    a.set_xlim(0, xmax)
    a.set_xlabel("Angular RMS error [deg]  (latency-corrected)", fontsize=11.5,
                 color=INK)
    a.grid(axis="x", alpha=0.28, lw=0.6, color="#b6bbc2", zorder=0)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color("#8b9198")
        a.spines[s].set_linewidth(0.9)
    a.tick_params(labelsize=10.5, colors=INK, length=3.5, width=0.9)
    leg = a.legend(fontsize=10.5, loc="lower right", framealpha=1.0,
                   edgecolor="#c9ced4", borderpad=0.6)
    leg.get_frame().set_linewidth(0.8)
    panel_tag(a, "(a)")

    # ---- (b) 정렬 잔차 ----------------------------------------------------
    a6 = res6["alignment"]["resid_rms_deg"]
    a9 = res9["alignment"]["resid_rms_deg"]
    top_b = two_bar(ax[1], a6, a9, 3.0,
                    "Absolute attitude RMS [deg]", "(b)", "{:.2f}°")
    ratio_bracket(ax[1], 0, 1, top_b * 1.14, f"×{a9 / a6:.1f}")

    # ---- (c) heading 드리프트 --------------------------------------------
    d6 = abs(r6["spin"]["drift_deg_per_min"])
    d9 = abs(r9["spin"]["drift_deg_per_min"])
    top_c = two_bar(ax[2], d6, d9, 1.0,
                    "Heading drift [°/min]", "(c)", "{:.2f}")
    ratio_bracket(ax[2], 0, 1, top_c * 1.14, f"×{d9 / d6:.1f}")

    fig.subplots_adjust(left=0.088, right=0.988, top=0.815, bottom=0.145,
                        wspace=0.30)

    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    png, svg, pdf = out_base + ".png", out_base + ".svg", out_base + ".pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    try:                                       # 논문 규격: PNG 는 RGB
        from PIL import Image
        Image.open(png).convert("RGB").save(png, dpi=(300, 300))
    except Exception:                          # noqa: BLE001
        pass
    for p in (png, svg, pdf):
        print(f"  -> {p}")
    plt.close(fig)


def main() -> int:
    k = korean_font()
    plt.rcParams["font.family"] = [k, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "path"      # 글꼴 없는 PC 에서도 SVG 가 깨지지 않게
    plt.rcParams["pdf.fonttype"] = 42

    res6 = json.load(open(os.path.join(qc.DIR_META, "qc_host6.json")))
    res9 = json.load(open(os.path.join(qc.DIR_META, "qc_chip.json")))
    build(res6, res9, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
