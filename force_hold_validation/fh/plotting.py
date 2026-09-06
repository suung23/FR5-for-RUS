"""그림. force_validation/fv/plotting.py 와 같은 지면 규약을 쓴다."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fh.analysis import axial_travel_series

INK = "#111111"
ACCENT = "#17683a"
BAND = "#17683a"
WARN = "#4a4a4a"
#: 위약 대조 두 팔. 켠 쪽은 기존 ACCENT 그대로여서 fig1~6 과 이어진다.
#: 짝은 dataviz 검증기를 통과한 값이다 — CVD ΔE 11.4 (protan) ·
#: 정상시야 ΔE 31.8 · 대비 3:1 이상, 여섯 검사 모두 PASS.
#: 색만으로 구분하지 않는다: 표식 모양도 함께 바꾼다 (인쇄 흑백 대비).
HOLD_ON = "#17683a"
HOLD_OFF = "#eb6834"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#d6d9de",
    "grid.linewidth": 0.6,
    "font.size": 8.5,
    "savefig.facecolor": "white",
})

OUT_DIR = "outputs"


def _save(figure, stem: str) -> list:
    paths = []
    for suffix, kwargs in ((".png", {"dpi": 300}), (".pdf", {})):
        path = os.path.join(OUT_DIR, stem + suffix)
        figure.savefig(path, bbox_inches="tight", pad_inches=0.05, **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def force_traces(runs: list, stem: str = "fig1_force_traces") -> list:
    """목표별 힘 시계열. 밴드와 한계를 함께 그려 판단 근거를 한 장에 둔다."""
    usable = [r for r in runs if r.included and r.t.size]
    if not usable:
        return []
    columns = min(2, len(usable))
    rows = int(np.ceil(len(usable) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(7.2, 2.0 * rows + 0.6),
                                squeeze=False, sharex=False)
    for index, run in enumerate(usable):
        ax = axes[index // columns][index % columns]
        probing = run.probing()
        ax.plot(run.t, run.force, color=ACCENT, lw=0.6)
        # 프로빙 구간을 옅게 깔아 "언제부터 로봇이 잡고 있었나" 를 보이게 한다.
        ax.fill_between(run.t, 0, 1, where=probing, transform=ax.get_xaxis_transform(),
                        color=BAND, alpha=0.06, lw=0)
        ax.axhspan(run.target - run.band, run.target + run.band,
                   color=BAND, alpha=0.12, lw=0)
        ax.axhline(run.target, color=ACCENT, ls="--", lw=0.8)
        for ts, key in run.events:
            if key in ("i", "w"):
                ax.axvline(ts, color=INK, ls=":", lw=0.7)
        warn = run.meta.get("warn_force_n")
        if warn and run.force.max() > warn * 0.6:
            ax.axhline(warn, color=WARN, ls="-.", lw=0.8)
        # 힘 옆에 로봇 변위를 겹친다. 힘이 평평한 것이 "로봇이 잡고 있다" 인지
        # "로봇이 얼어 있는데 팬텀이 가만히 있다" 인지는 이 곡선 없이는 못 가른다
        # (2026-09-03 — 미소 지령 정체가 정확히 그렇게 숨어 있었다).
        probing_idx = np.where(probing)[0]
        if probing_idx.size:
            travel = axial_travel_series(run, int(probing_idx[0])) * 1000.0
            if np.isfinite(travel).any():
                twin = ax.twinx()
                twin.plot(run.t, travel, color=INK, lw=0.6, ls="--", alpha=0.75)
                twin.set_ylabel("travel [mm]", fontsize=7)
                twin.grid(False)
        ax.set_title(f"{run.label} · target {run.target:.2f} N", fontsize=8)
        ax.set_ylabel("‖F‖ [N]")
        ax.set_xlabel("t [s]")
    for spare in range(len(usable), rows * columns):
        axes[spare // columns][spare % columns].axis("off")
    figure.suptitle("Force held (solid) and axial travel (dashed), per run. "
                    "Shaded = deadband, dotted = syringe step", fontsize=9)
    figure.tight_layout()
    return _save(figure, stem)


def hold_quality(grouped: list, stem: str = "fig2_hold_quality") -> list:
    """대역별 유지 품질 — 절대 오차와 상대 오차, 그리고 밴드 안 체류율."""
    if not grouped:
        return []
    target = np.array([g["target_n"] for g in grouped])
    figure, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))

    axes[0].errorbar(target, [g["mean_error_n"] for g in grouped],
                     yerr=[g["sd_n"] for g in grouped],
                     fmt="o", color=ACCENT, ecolor=WARN, elinewidth=1.0, capsize=3, ms=4)
    axes[0].axhline(0, color=INK, lw=0.8)
    axes[0].set_xlabel("target [N]"); axes[0].set_ylabel("error [N]")
    axes[0].set_title("Steady-state error (mean ± SD)", fontsize=8)

    axes[1].plot(target, [g["rmse_n"] for g in grouped], "o-", color=ACCENT, ms=4, lw=1)
    twin = axes[1].twinx()
    twin.plot(target, [g["rmse_pct_of_target"] for g in grouped], "s--",
              color=WARN, ms=3.5, lw=0.9)
    twin.set_ylabel("RMSE [% of target]", color=WARN)
    axes[1].set_xlabel("target [N]"); axes[1].set_ylabel("RMSE [N]", color=ACCENT)
    axes[1].set_title("Absolute and relative", fontsize=8)

    # 폭은 가장 좁은 목표 간격에서 정한다. 간격마다 다른 폭을 주면 막대 넓이가
    # 값처럼 읽혀, 목표가 촘촘한 저대역이 작아 보인다.
    spacing = float(np.min(np.diff(target))) if target.size > 1 else 1.0
    axes[2].bar(target, [g["in_band_pct"] for g in grouped],
                width=spacing * 0.6,
                color=ACCENT, alpha=0.85, edgecolor="white", lw=0.6)
    axes[2].axhline(100, color=INK, lw=0.8)
    axes[2].set_ylim(0, 105)
    axes[2].set_xlabel("target [N]"); axes[2].set_ylabel("time in band [%]")
    axes[2].set_title("Time inside the deadband", fontsize=8)
    figure.tight_layout()
    return _save(figure, stem)


def disturbance(runs: list, events: list, stem: str = "fig3_disturbance") -> list:
    """주사기 조작에 정렬한 응답 — 위는 힘 오차, 아래는 로봇 변위.

    두 행을 같은 시간축에 두는 이유: 힘이 돌아오는 동안 변위가 계단으로 움직이면
    로봇이 받아 낸 것이고, 변위가 평평하면 팬텀이 스스로 누운 것이다. 힘 행
    하나로는 그 둘이 같은 그림이 된다.
    """
    if not events:
        return []
    by_label = {run.label: run for run in runs}
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex="col")
    for col, direction in enumerate(("in", "withdraw")):
        force_ax, travel_ax = axes[0][col], axes[1][col]
        drawn = 0
        labelled = set()          # 목표당 한 줄만 범례에 올린다
        for event in [e for e in events if e["direction"] == direction]:
            run = by_label.get(event["run"])
            if run is None:
                continue
            window = (run.t >= event["onset_s"] - 1.0) & \
                     (run.t <= event["onset_s"] + max(6.0, event["window_s"]))
            if window.sum() < 5:
                continue
            t = run.t[window] - event["onset_s"]
            tag = f"{event['target_n']:.1f} N"
            line, = force_ax.plot(
                t, run.force[window] - event["target_n"],
                lw=0.7, alpha=0.85,
                label=None if tag in labelled else tag)
            labelled.add(tag)
            # 변위는 교란 직전 표본을 기준으로 한다 — 힘 오차와 같은 정렬이다.
            idx = np.where(window)[0]
            travel = axial_travel_series(run, int(idx[0])) * 1000.0
            if np.isfinite(travel[idx]).any():
                travel_ax.plot(t, travel[window], lw=0.7, alpha=0.85,
                               color=line.get_color())
            drawn += 1
        force_ax.axhline(0, color=INK, lw=0.8)
        force_ax.axvline(0, color=INK, ls=":", lw=0.8)
        force_ax.set_ylabel("error [N]")
        force_ax.set_title(f"syringe {direction}", fontsize=8)
        travel_ax.axhline(0, color=INK, lw=0.8)
        travel_ax.axvline(0, color=INK, ls=":", lw=0.8)
        travel_ax.set_xlabel("t since step [s]")
        travel_ax.set_ylabel("axial travel [mm]")
        if drawn:
            force_ax.legend(fontsize=6.5, frameon=False, ncol=2)
    figure.suptitle("Response to the surface moving — force (top), robot travel (bottom)",
                    fontsize=9)
    figure.tight_layout()
    return _save(figure, stem)


def excursion_recovery(runs: list, excursions: list, exposure: list,
                       stem: str = "fig6_excursion_recovery") -> list:
    """안전을 노출 시간으로 논증하는 그림.

    한계에 닿은 적이 없으면 "넘지 않았다" 는 시험된 사실이 아니다. 남는 질문은
    **벗어나 있던 동안이 얼마나 길었는가** 이고, 조직이 실제로 겪는 것도 그것이다.

    왼쪽은 밴드를 벗어난 구간 하나하나의 복귀 시간, 가운데는 문턱을 올려 가며
    잰 **한 번에 가장 오래** 머문 시간, 오른쪽은 가장 세게 눌린 실행의 시계열에
    그 구간을 칠한 것이다.
    """
    if not excursions:
        return []
    figure, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))

    # --- 복귀 시간 --------------------------------------------------------
    labels = sorted({e["run"] for e in excursions})
    for index, label in enumerate(labels):
        d = [e["duration_s"] for e in excursions if e["run"] == label]
        axes[0].scatter(np.full(len(d), index) + np.random.uniform(-.08, .08, len(d)),
                        d, s=16, color=ACCENT, alpha=.8, edgecolor="white", lw=.5)
        axes[0].plot([index - .22, index + .22], [np.median(d)] * 2, color=INK, lw=1.4)
    worst = max(e["duration_s"] for e in excursions)
    axes[0].axhline(worst, color=WARN, ls="--", lw=.8)
    axes[0].text(len(labels) - .5, worst, f" longest {worst:.2f} s", fontsize=7,
                 va="bottom", ha="right", color=WARN)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=6.5)
    axes[0].set_ylabel("time to return [s]")
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("recovery time per band excursion", fontsize=8)

    # --- 노출 곡선 --------------------------------------------------------
    for label in sorted({e["run"] for e in exposure}):
        rows = sorted([e for e in exposure if e["run"] == label],
                      key=lambda e: e["level_n"])
        if max(r["longest_s"] for r in rows) <= 0:
            continue
        axes[1].plot([r["level_n"] for r in rows], [r["longest_s"] for r in rows],
                     "o-", ms=3, lw=1, alpha=.85, label=label)
    warn = next((r.meta.get("warn_contact_force_n") for r in runs
                 if r.meta.get("warn_contact_force_n")), None)
    limit = next((r.meta.get("max_contact_force_n") for r in runs
                  if r.meta.get("max_contact_force_n")), None)
    if warn:
        axes[1].axvline(warn, color=WARN, ls="-.", lw=.9)
        axes[1].text(warn, axes[1].get_ylim()[1], " warn", fontsize=7,
                     va="top", color=WARN)
    if limit:
        axes[1].axvline(limit, color="#a3231f", ls="--", lw=1.1)
        axes[1].text(limit, axes[1].get_ylim()[1], " limit", fontsize=7,
                     va="top", color="#a3231f")
    axes[1].set_xlabel("threshold [N]"); axes[1].set_ylabel("longest single episode [s]")
    axes[1].set_title("time held continuously above a force", fontsize=8)
    axes[1].legend(fontsize=6, frameon=False)

    # --- 가장 세게 눌린 실행 ----------------------------------------------
    hardest = max(excursions, key=lambda e: e["peak_n"])
    run = next((r for r in runs if r.label == hardest["run"]), None)
    if run is not None:
        mask = run.probing()
        t, f = run.t[mask], run.force[mask]
        top = hardest["band_top_n"]
        axes[2].plot(t, f, color=ACCENT, lw=.6)
        axes[2].fill_between(t, top, f, where=f > top, color="#a3231f", alpha=.25, lw=0)
        axes[2].axhline(top, color=INK, ls="--", lw=.8)
        if warn:
            axes[2].axhline(warn, color=WARN, ls="-.", lw=.9)
        if limit:
            axes[2].axhline(limit, color="#a3231f", ls="--", lw=1.1)
        axes[2].set_title(f"{run.label} · peak {f.max():.2f} N", fontsize=8)
    axes[2].set_xlabel("t [s]"); axes[2].set_ylabel("‖F‖ [N]")

    figure.suptitle("Safety as exposure time — how long above the band, not whether the limit was hit",
                    fontsize=9)
    figure.tight_layout()
    return _save(figure, stem)


def regulation_evidence(runs: list, evidence: list, stiff: list,
                        stem: str = "fig5_regulation_evidence") -> list:
    """힘은 일정한데 팔이 물러났다는 것을 한 장에 둔다.

    왼쪽은 개루프에서 잰 강성 직선, 오른쪽은 교란 중의 힘과 축방향 변위를 겹친
    시계열이다. 오른쪽에서 힘이 평평하고 변위가 계단으로 내려가면, 그 계단이
    로봇이 흡수한 양이다.
    """
    usable = [s for s in stiff if s.get("ok")]
    by_label = {r.label: r for r in runs}
    picked = next((e for e in evidence if e["direction"] == "in"), None)
    if not usable and picked is None:
        return []

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    if usable:
        s0 = usable[0]
        k = s0["k_n_per_m"]
        travel = np.linspace(0, max(s0["travel_span_mm"], 0.1) / 1000.0, 20)
        axes[0].plot(travel * 1000.0, k * travel + s0["intercept_n"],
                     color=ACCENT, lw=1.2)
        axes[0].set_title(f"phantom stiffness · {k/1000.0:.3f} N/mm "
                          f"(R²={s0['r_squared']:.3f})", fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "no open-loop run", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=8, color=WARN)
        axes[0].set_title("phantom stiffness — not measured", fontsize=8)
    axes[0].set_xlabel("axial travel [mm]"); axes[0].set_ylabel("‖F‖ [N]")

    if picked is not None:
        run = by_label.get(picked["run"])
        if run is not None and run.tip.size:
            window = (run.t >= picked["onset_s"] - 1.0) & (run.t <= picked["onset_s"] + 8.0)
            if window.sum() > 5:
                first = int(np.argmax(window))
                travel = np.array([
                    (axial_travel_safe(run, first, i) or 0.0) * 1000.0
                    for i in np.where(window)[0]])
                t = run.t[window] - picked["onset_s"]
                axes[1].plot(t, run.force[window], color=ACCENT, lw=0.8, label="‖F‖ [N]")
                twin = axes[1].twinx()
                twin.plot(t, travel, color=INK, lw=0.9, ls="--", label="travel [mm]")
                twin.set_ylabel("axial travel [mm]")
                axes[1].axhline(picked["target_n"], color=ACCENT, ls=":", lw=0.8)
                axes[1].axvline(0, color=INK, ls=":", lw=0.8)
                axes[1].set_title(
                    f"{picked['run']} · held {picked['held_mean_n']:.2f} N while "
                    f"retreating {abs(picked['travel_mm']):.1f} mm", fontsize=8)
    axes[1].set_xlabel("t since step [s]"); axes[1].set_ylabel("‖F‖ [N]")
    figure.suptitle("Force flat, arm moving — the disturbance was absorbed", fontsize=9)
    figure.tight_layout()
    return _save(figure, stem)


def axial_travel_safe(run, start, stop):
    from fh.analysis import axial_travel
    try:
        return axial_travel(run, start, stop)
    except Exception:
        return None


def safety_margin(rows: list, stem: str = "fig4_safety_margin") -> list:
    """최악 힘과 한계. 넘었는지 아닌지가 한눈에 보여야 한다."""
    rows = [r for r in rows if np.isfinite(r.get("max_force_n", float("nan")))]
    if not rows:
        return []
    figure, ax = plt.subplots(figsize=(7.2, 2.8))
    labels = [r["run"] for r in rows]
    peak = [r["peak_force_n"] for r in rows]
    over = [r.get("exceeded_limit") for r in rows]
    ax.bar(range(len(rows)), peak,
           color=[("#a3231f" if o else ACCENT) for o in over],
           alpha=0.9, edgecolor="white", lw=0.6)
    ax.axhline(rows[0]["max_force_n"], color="#a3231f", ls="--", lw=1.1)
    ax.axhline(rows[0]["warn_force_n"], color=WARN, ls="-.", lw=0.9)
    ax.text(len(rows) - 0.4, rows[0]["max_force_n"], " limit", va="bottom",
            fontsize=7, color="#a3231f")
    ax.text(len(rows) - 0.4, rows[0]["warn_force_n"], " warn", va="bottom",
            fontsize=7, color=WARN)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel("peak ‖F‖ [N]")
    ax.set_title("Worst force seen in each run, against the configured limits", fontsize=9)
    figure.tight_layout()
    return _save(figure, stem)


def placebo_comparison(events: list, excluded_targets=(), p_value=float("nan"),
                       stem: str = "fig7_placebo") -> list:
    """위약 대조 — 같은 교란을 제어 켜고 한 번, 끄고 한 번.

    이 그림 하나가 "힘이 목표 근처에 머물렀다" 를 해석 가능한 문장으로 바꾼다.
    켠 쪽만 있으면 그 값이 제어 덕분인지 교란이 원래 그 정도였는지 알 수 없다.

    왼쪽은 대역마다 두 팔을 나란히 놓는다 (중앙값과 IQR). 막대가 아니라 점과
    수염인 이유는, 사건이 대역마다 10 개뿐이라 분포를 감추면 안 되기 때문이다.
    오른쪽은 두 팔의 누적분포를 통째로 겹친다 — 겹치면 겹치는 대로 보인다.

    Args:
        events: ``disturbance_events`` 행. ``run`` 첫 글자로 팔을 가른다
            (``B`` 제어 켬 · ``C`` 제어 끔).
        excluded_targets: 안전 한계에 걸려 개루프가 아니게 된 대역. 회색으로
            덮고 이유를 적는다 — 지우지 않는 이유는 지운 것이 보고서에서
            사라지면 안 되기 때문이다.
        p_value: 두 팔 전체의 순위합 양측 p.
        stem: 파일 이름.

    Returns:
        저장된 경로 목록.
    """
    def peaks(arm, target=None):
        out = [abs(float(row["peak_error_n"])) for row in events
               if str(row["run"])[:1] == arm
               and (target is None or float(row["target_n"]) == target)]
        return np.asarray(out, dtype=float)

    targets = sorted({float(row["target_n"]) for row in events})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # -- (a) 대역별 -----------------------------------------------------
    ax = axes[0]
    for target in excluded_targets:
        ax.axvspan(target - 0.22, target + 0.22, color="#e9ebee", zorder=0, lw=0)
    arms = (("B", "force hold ON", HOLD_ON, "o"), ("C", "force hold OFF (control)", HOLD_OFF, "s"))
    series = []
    for index, (arm, label, color, marker) in enumerate(arms):
        offset = (index - 0.5) * 0.15
        xs, meds, los, his = [], [], [], []
        for target in targets:
            values = peaks(arm, target)
            if not values.size:
                continue
            xs.append(target + offset)
            meds.append(np.median(values))
            los.append(np.percentile(values, 25))
            his.append(np.percentile(values, 75))
        ax.vlines(xs, los, his, color=color, lw=1.4, zorder=2)
        ax.plot(xs, meds, marker, color=color, ms=5.0, mec="white", mew=0.8,
                ls="none", label=label, zorder=3)
        series.append((xs, meds, color, marker, offset))

    # 축은 **비교가 성립하는 대역**에 맞춘다.
    #
    # 제외 대역(안전 한계에 걸린 쪽)이 축을 3 배 넘게 늘리면, 정작 읽어야 할
    # 0.5~3.5 N 이 아래 3 분의 1 로 눌린다. 그렇다고 그 점을 빼면 보고서에서
    # 사라지므로, 축 위에 얹고 값을 글자로 적는다 — 화면 밖이라는 사실이
    # 보이는 채로 남는다.
    keep = [row for row in events
            if float(row["target_n"]) not in set(excluded_targets)]
    if keep:
        top = max(abs(float(row["peak_error_n"])) for row in keep) * 1.25
        ax.set_ylim(0, top)
        for xs, meds, color, marker, offset in series:
            for x, median in zip(xs, meds):
                if median <= top:
                    continue
                ax.plot([x], [top * 0.97], marker, color=color, ms=5.0,
                        mec="white", mew=0.8, clip_on=False, zorder=4)
                # 두 팔의 글자가 겹치지 않게 각자 바깥쪽으로 민다. 가운데
                # 정렬하면 두 값이 한 자리에 겹쳐 찍힌다.
                side = "right" if offset < 0 else "left"
                ax.annotate(f"{median:.1f}", xy=(x, top * 0.97),
                            xytext=(-3 if offset < 0 else 3, 7),
                            textcoords="offset points",
                            ha=side, fontsize=6.5, color=color)
    if excluded_targets:
        edge = min(excluded_targets)
        ax.annotate("force limit reached\n— not open-loop", xy=(edge, 0),
                    xytext=(edge - 0.12, ax.get_ylim()[1] * 0.30),
                    ha="right", va="center", fontsize=7, color=WARN)
    ax.set_xticks(targets)
    ax.set_xticklabels([f"{t:g}" for t in targets], fontsize=7.5)
    ax.set_xlabel("target [N]")
    ax.set_ylabel("peak |F − target| per event [N]")
    ax.set_title("(a) By band — median and IQR", loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    # -- (b) 전체 분포 --------------------------------------------------
    ax = axes[1]
    for arm, label, color, marker in arms:
        values = np.sort(peaks(arm))
        if not values.size:
            continue
        fraction = np.arange(1, values.size + 1) / values.size
        ax.step(values, fraction, where="post", color=color, lw=1.6,
                label=f"{label}  (n={values.size})")
        median = float(np.median(values))
        ax.plot([median], [0.5], marker, color=color, ms=5.0, mec="white", mew=0.8)
    ax.set_xlabel("peak |F − target| per event [N]")
    ax.set_ylabel("cumulative fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title("(b) All events, pooled", loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    if p_value == p_value:
        ax.text(0.44, 0.44, f"rank-sum two-sided\np = {p_value:.2f}",
                transform=ax.transAxes, fontsize=7.5, color=INK, va="center")

    figure.tight_layout()
    return _save(figure, stem)
