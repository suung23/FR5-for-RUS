"""리포트 공통 지면 규약. imu_bench/qc_track 의 그림들과 같은 계열을 쓴다."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 남색–회색 한 계열. 회색 = 기준·문맥, 남색 = 측정.
NAVY_DK, NAVY, NAVY_LT = "#1F3A5F", "#2E6E96", "#A7CFE6"
GREY_DK, GREY, GREY_LT = "#666C74", "#9AA1A9", "#C7CCD1"
INK, SUB = "#1A1F26", "#5A6068"
BAND = "#DCE6EF"

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlesize": 9,
        "axes.titleweight": "regular",
        "axes.grid": True,
        "grid.color": GREY_LT,
        "grid.linewidth": 0.5,
        "xtick.color": SUB,
        "ytick.color": SUB,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 8.5,
        "font.family": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def save(fig, stem: str) -> list:
    """PNG(300 dpi)·PDF 로 낸다. 저장소의 다른 그림들과 같은 규약."""
    os.makedirs(FIG_DIR, exist_ok=True)
    out = []
    for suffix in (".png", ".pdf"):
        path = os.path.join(FIG_DIR, stem + suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03,
                    facecolor="white")
        out.append(path)
    plt.close(fig)
    return out
