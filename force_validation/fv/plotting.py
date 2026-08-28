"""논문에 넣을 수 있는 그림 두 장.

seaborn 을 쓰지 않는다. 여기서 필요한 것은 산점도·오차막대·히스토그램·strip plot
뿐이고 matplotlib 로 다 된다. 의존성 하나를 줄이면 그림을 못 그려서 분석이 멈추는
경우도 하나 줄어든다.

전체 제목은 넣지 않는다. 논문 그림은 캡션이 제목을 대신하고, 그림 안의 제목은
편집에서 지워야 할 것이 된다. panel label 만 좌측 상단에 둔다.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - 백엔드 지정 뒤에 와야 한다
import numpy as np  # noqa: E402

#: robot 쪽 자료.
ROBOT = "#1B6B5C"
#: 불확도와 보조 자료.
ACCENT = "#D9822B"
#: identity·zero·guide line.
GUIDE = "#7A7A7A"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _label(axes, text: str) -> None:
    """Panel label 을 축 바깥 좌측 상단에."""
    axes.text(-0.16, 1.06, text, transform=axes.transAxes,
              fontsize=11, fontweight="bold", va="top", ha="left")


def _save(figure, stem: str) -> list:
    """PNG(300 dpi) 와 PDF 로 저장한다. 잘림을 막으려 여백을 두고 자른다."""
    written = []
    for suffix, kwargs in ((".png", {"dpi": 300}), (".pdf", {})):
        path = stem + suffix
        figure.savefig(path, bbox_inches="tight", pad_inches=0.05, **kwargs)
        written.append(path)
    plt.close(figure)
    return written


def normal_force_validation(trials, accuracy, combined_uncertainty_n: float,
                            repeat_levels, stem: str) -> list:
    """Figure 1 — robot 값과 전자저울 기준값의 비교."""
    used = [t for t in trials if t.included]
    scale = np.array([t.scale_n for t in used], dtype=float)
    robot = np.array([t.robot_n for t in used], dtype=float)
    error = robot - scale
    yerr = np.array([t.gravity_uncertainty_n or 0.0 for t in used], dtype=float)

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))

    # --- A: 측정 대 기준 -------------------------------------------------
    ax = axes[0, 0]
    span = [0.0, float(max(scale.max(), robot.max())) * 1.08 if used else 1.0]
    ax.plot(span, span, color=GUIDE, lw=0.9, ls="--", label="Identity", zorder=1)
    if accuracy is not None:
        grid = np.linspace(span[0], span[1], 100)
        fit = accuracy.slope * grid + accuracy.intercept
        ax.plot(grid, fit, color=ROBOT, lw=1.4, label="Least squares", zorder=2)
        if all(math.isfinite(v) for v in accuracy.slope_ci + accuracy.intercept_ci):
            low = accuracy.slope_ci[0] * grid + accuracy.intercept_ci[0]
            high = accuracy.slope_ci[1] * grid + accuracy.intercept_ci[1]
            ax.fill_between(grid, np.minimum(low, high), np.maximum(low, high),
                            color=ROBOT, alpha=0.13, lw=0, label="95% CI", zorder=1)
    ax.errorbar(scale, robot, yerr=yerr, fmt="o", ms=4.5, mfc=ROBOT, mec=ROBOT,
                ecolor=ACCENT, elinewidth=1.0, capsize=2.5, lw=0, zorder=3)
    ax.set_xlabel("Scale reference normal force (N)")
    ax.set_ylabel("Robot-measured normal force (N)")
    if accuracy is not None:
        ax.text(0.04, 0.96,
                f"slope {accuracy.slope:.3f}\nintercept {accuracy.intercept:+.3f} N\n"
                f"$R^2$ {accuracy.r_squared:.4f}\nRMSE {accuracy.rmse_n:.3f} N\n"
                f"n = {accuracy.n}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                linespacing=1.5)
    ax.legend(loc="lower right", frameon=False)
    _label(ax, "A")

    # --- B: 잔차 ---------------------------------------------------------
    ax = axes[0, 1]
    ax.axhspan(-combined_uncertainty_n, combined_uncertainty_n,
               color=GUIDE, alpha=0.16, lw=0,
               label=r"$\pm u_{\mathrm{combined}}$")
    ax.axhline(0.0, color=GUIDE, lw=0.9, ls="--")
    if accuracy is not None:
        ax.axhline(accuracy.bias_n, color=ACCENT, lw=1.2,
                   label=f"Mean bias {accuracy.bias_n:+.3f} N")
    # 겹침을 줄이는 작은 흔들림. 자료를 바꾸지 않도록 x 에만, 눈금 폭의 1% 만.
    jitter = (span[1] - span[0]) * 0.006
    rng = np.random.default_rng(0)
    ax.plot(scale + rng.uniform(-jitter, jitter, scale.size), error, "o",
            ms=4.5, color=ROBOT, lw=0)
    ax.set_xlabel("Scale reference normal force (N)")
    ax.set_ylabel("Robot $-$ scale error (N)")
    ax.legend(loc="best", frameon=False)
    _label(ax, "B")

    # --- C: Bland-Altman -------------------------------------------------
    ax = axes[1, 0]
    mean_pair = (robot + scale) / 2.0
    ax.plot(mean_pair, error, "o", ms=4.5, color=ROBOT, lw=0)
    if accuracy is not None:
        ax.axhline(accuracy.ba_bias_n, color=ACCENT, lw=1.2,
                   label=f"Bias {accuracy.ba_bias_n:+.3f} N")
        for bound, name in ((accuracy.ba_upper_n, "+1.96 SD"),
                            (accuracy.ba_lower_n, "$-$1.96 SD")):
            ax.axhline(bound, color=GUIDE, lw=0.9, ls="--")
            ax.text(0.99, bound, f" {name} {bound:+.3f}", transform=ax.get_yaxis_transform(),
                    va="bottom", ha="right", fontsize=7, color=GUIDE)
    ax.set_xlabel("Mean of robot and scale normal force (N)")
    ax.set_ylabel("Robot $-$ scale difference (N)")
    # LoA 라벨은 오른쪽 끝에 붙는다. 범례를 오른쪽 어디에 두든 그 라벨이나
    # bias 근처의 점과 겹치므로 왼쪽 위로 보낸다. 그 구석은 LoA 라벨이 없고,
    # 차이값이 상한 근처에 몰리지 않는 한 비어 있다.
    ax.margins(y=0.14)
    ax.legend(loc="upper left", frameon=False)
    _label(ax, "C")

    # --- D: 반복성 또는 capture 안정성 -----------------------------------
    ax = axes[1, 1]
    if repeat_levels:
        for index, (level, count, sd, _cv) in enumerate(repeat_levels):
            members = [t.robot_n for t in used if abs(t.scale_n - level) <= 0.25]
            xs = np.full(len(members), index, dtype=float)
            xs += rng.uniform(-0.06, 0.06, xs.size)
            ax.plot(xs, members, "o", ms=4, color=ROBOT, alpha=0.75, lw=0)
            mean = float(np.mean(members))
            ax.errorbar([index], [mean], yerr=[sd], fmt="_", ms=16,
                        color=ACCENT, elinewidth=1.3, capsize=4)
        ax.set_xticks(range(len(repeat_levels)))
        ax.set_xticklabels([f"{level:.1f} N\n(n={count})"
                            for level, count, _, _ in repeat_levels])
        ax.set_xlabel("Target force level")
        ax.set_ylabel("Robot normal force (N)")
    else:
        spreads = [t.std_250ms_n for t in used if t.std_250ms_n is not None]
        if spreads:
            ax.hist(spreads, bins=min(12, max(4, len(spreads) // 2)),
                    color=ROBOT, alpha=0.85, edgecolor="white", linewidth=0.6)
            ax.set_xlabel("Capture 250 ms force SD (N)")
            ax.set_ylabel("Captures")
        else:
            ax.text(0.5, 0.5, "no repeat levels\nand no capture SD recorded",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color=GUIDE)
            ax.set_xticks([])
            ax.set_yticks([])
    _label(ax, "D")

    figure.tight_layout(pad=1.4, w_pad=2.4, h_pad=2.2)
    return _save(figure, stem)


def gravity_validation(gravity, stem: str) -> list:
    """Figure 2 — 중력보상 자체의 성능. Figure 1 과 분리해 제시한다."""
    residuals = gravity.residuals
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))

    # --- A: 잔차 분포 ----------------------------------------------------
    ax = axes[0, 0]
    ax.boxplot([residuals], vert=True, widths=0.4, showfliers=False,
               medianprops={"color": ACCENT, "linewidth": 1.4},
               boxprops={"color": GUIDE}, whiskerprops={"color": GUIDE},
               capprops={"color": GUIDE})
    rng = np.random.default_rng(1)
    ax.plot(1 + rng.uniform(-0.09, 0.09, residuals.size), residuals, "o",
            ms=4, color=ROBOT, alpha=0.75, lw=0)
    ax.axhline(0.0, color=GUIDE, lw=0.9, ls="--")
    ax.set_xticks([1])
    ax.set_xticklabels([f"n = {gravity.n_poses}"])
    ax.set_ylabel("Gravity residual, normal axis (N)")
    _label(ax, "A")

    # --- B: 자세별 잔차 --------------------------------------------------
    ax = axes[0, 1]
    if np.any(gravity.orientations):
        # 자세를 한 축으로 요약한다: 회전벡터의 크기. 세 각을 그대로 그리면
        # 축이 셋이 되고 한 패널에 들어가지 않는다.
        angle = np.linalg.norm(gravity.orientations, axis=1)
        ax.plot(angle, residuals, "o", ms=4.5, color=ROBOT, lw=0)
        ax.set_xlabel("Probe orientation magnitude (rad)")
    else:
        ax.plot(np.arange(1, residuals.size + 1), residuals, "o", ms=4.5,
                color=ROBOT, lw=0)
        ax.set_xlabel("Validation pose index")
    ax.axhline(0.0, color=GUIDE, lw=0.9, ls="--")
    ax.set_ylabel("Gravity residual (N)")
    _label(ax, "B")

    # --- C: 히스토그램 ---------------------------------------------------
    ax = axes[1, 0]
    ax.hist(residuals, bins=min(12, max(4, residuals.size // 2)),
            color=ROBOT, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.axvline(gravity.bias_n, color=ACCENT, lw=1.3,
               label=f"Mean {gravity.bias_n:+.3f} N")
    for bound in (gravity.lower_95_n, gravity.upper_95_n):
        ax.axvline(bound, color=GUIDE, lw=0.9, ls="--")
    ax.text(0.03, 0.95,
            f"SD {gravity.sd_n:.3f} N\n95% {gravity.lower_95_n:+.3f} to "
            f"{gravity.upper_95_n:+.3f} N",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5, linespacing=1.5)
    ax.set_xlabel("Gravity residual, normal axis (N)")
    ax.set_ylabel("Poses")
    ax.legend(loc="upper right", frameon=False)
    _label(ax, "C")

    # --- D: 자세별 산점 --------------------------------------------------
    ax = axes[1, 1]
    ax.plot(np.arange(1, residuals.size + 1), residuals, "o", ms=4.5,
            color=ROBOT, lw=0)
    ax.axhline(gravity.bias_n, color=ACCENT, lw=1.2)
    ax.axhspan(gravity.lower_95_n, gravity.upper_95_n, color=GUIDE, alpha=0.14, lw=0)
    ax.axhline(0.0, color=GUIDE, lw=0.9, ls="--")
    ax.set_xlabel("Validation pose")
    ax.set_ylabel("Gravity residual (N)")
    if residuals.size <= 20:
        ax.set_xticks(np.arange(1, residuals.size + 1))
        ax.set_xticklabels([str(p) for p in gravity.pose_ids], fontsize=6.5, rotation=90)
    _label(ax, "D")

    figure.tight_layout(pad=1.4, w_pad=2.4, h_pad=2.2)
    return _save(figure, stem)
