"""그림. force_validation/fv/plotting.py 와 같은 지면 규약을 쓴다.

지면 규약 (2026-09-07, 논문 그림용으로 정리):

* **팔(arm)은 검정·회색 명도로만 가른다.** 힘 유지를 켠 교란은 검정, 끈
  교란(위약)은 회색, 교란 없는 유지는 그보다 옅은 회색. 켬/끔은 같은 표식(●)을
  쓴다 — 표식 모양으로 켬/끔을 구분하지 않는다. 한계선·초과 구간만 진한
  주황으로 따로 띄운다.
* **실행 코드(A_t0p5_r2 …)는 그림에 나오지 않는다.** 모든 패널은 "어떤 목표
  힘을, 어느 팔로 유지했는가" 로 이름 붙는다 (:data:`ARM_NAMES`).
* **대역마다 패널을 따로 둔다.** 여덟 대역을 한 축에 겹치면 읽을 수 없다 —
  fig1·fig3·fig6 은 목표 힘별 소형 다중 패널(small multiples)이다.
* 범례는 축 안에 넣지 않는다. 그림 맨 아래 한 줄에 한 번만 둔다.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from fh.analysis import axial_travel_series

# ---- 팔레트: 검정 · 회색 (팔), 남색 (음영), 주황 (한계) -------------------
# 팔은 색으로만 가른다 — 표식 모양은 켬/끔을 구분하지 않는다.
INK = "#111111"          # 글자, 축, 기준선, 힘 유지 켬(B)
GRAY = "#6b6f76"         # 힘 유지 끔 · 위약 대조(C), 경고선
GRAY_PALE = "#a8acb3"    # 교란 없는 유지(A) — 켬/끔보다 옅게
NAVY = "#1f3a68"         # 요약 그림의 기본 곡선, 데드밴드 음영
ORANGE = "#e3701e"       # (팔 색에서는 빠졌다) 강조가 필요할 때만
ORANGE_DARK = "#b8501a"  # 힘 한계선, 한계 초과 구간
GRAY_LIGHT = "#c4c7cc"   # 보조 곡선(축방향 변위) — 어느 팔 색보다도 밝게
LIGHT = "#e9ebee"        # 제외 대역 덮개
GRID = "#d6d9de"

#: 옛 이름. 다른 모듈이 참조할 수 있어 남긴다 — 값만 새 팔레트로 바뀌었다.
ACCENT = NAVY
BAND = NAVY
WARN = GRAY
HOLD_ON = INK
HOLD_OFF = GRAY
LIMIT = ORANGE_DARK

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    # 그림 안 모든 글자(제목·축 이름·눈금·범례·주석)는 Times New Roman.
    # 뒤의 이름들은 그 글꼴이 없는 기계에서의 대체 순서다.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",     # 수식 글꼴도 Times 계열로
    "font.size": 8.5,
    "savefig.facecolor": "white",
    # PDF/PS 에 TrueType 그대로 묻는다 — 편집기에서 글자가 글자로 남게.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT_DIR = "outputs"

#: 논문용 명칭. 실행 코드의 첫 글자가 팔이다: A = 교란 없는 정지 유지,
#: B = 교란 중 힘 유지 켬, C = 교란 중 힘 유지 끔(위약 대조).
ARM_NAMES = {
    "A": ("undisturbed hold", GRAY_PALE, "^"),
    "B": ("disturbed, force hold on", INK, "o"),
    "C": ("disturbed, force hold off (placebo)", GRAY, "o"),
}


def _arm(label: str):
    """실행 코드 → (논문용 이름, 색, 표식)."""
    return ARM_NAMES.get(str(label)[:1], (str(label), INK, "o"))


def _arm_legend(figure, arms=("A", "B", "C"), extra=(), bottom=0.06,
                style="line", **kwargs):
    """그림 맨 아래 한 줄 범례. ``extra`` 는 (handle) 목록으로 뒤에 붙는다.

    ``style``: "line" 은 선+표식, "trace" 는 표식 없는 선(시계열), "patch" 는 막대.
    """
    handles = []
    for a in arms:
        if a not in ARM_NAMES:
            continue
        name, color, marker = ARM_NAMES[a]
        if style == "patch":
            handles.append(Patch(facecolor=color, edgecolor="white", label=name))
        else:
            handles.append(Line2D([], [], color=color, ls="-", lw=1, label=name,
                                  marker=None if style == "trace" else marker,
                                  ms=4, mec="white", mew=.4))
    handles += list(extra)
    figure.legend(handles=handles, loc="lower center", ncol=len(handles),
                  fontsize=7, frameon=False, handlelength=2.2, columnspacing=2.0,
                  bbox_to_anchor=(0.5, 0.0), **kwargs)
    return bottom


def _targets(runs: list) -> list:
    return sorted({float(r.target) for r in runs if np.isfinite(r.target)})


def _save(figure, stem: str) -> list:
    paths = []
    for suffix, kwargs in ((".png", {"dpi": 300}), (".pdf", {})):
        path = os.path.join(OUT_DIR, stem + suffix)
        figure.savefig(path, bbox_inches="tight", pad_inches=0.05, **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


# =========================================================================
# fig1 — 목표 힘별 힘 시계열 (행 = 목표 힘, 열 = 팔)
# =========================================================================
def force_traces(runs: list, stem: str = "fig1_force_traces") -> list:
    """목표 힘 하나에 한 행, 팔 하나에 한 열. 밴드와 한계를 함께 그린다.

    행이 목표 힘이므로 "이 힘을 잡으려 할 때 세 조건에서 각각 어땠나" 를 한 줄에서
    읽는다. 실행 코드는 쓰지 않는다.
    """
    usable = [r for r in runs if r.included and r.t.size and np.isfinite(r.target)]
    if not usable:
        return []
    targets = _targets(usable)
    arms = [a for a in ("A", "B", "C") if any(str(r.label)[:1] == a for r in usable)]
    rows, columns = len(targets), len(arms)
    figure, axes = plt.subplots(rows, columns, figsize=(2.45 * columns + 0.4, 1.55 * rows + 0.7),
                                squeeze=False)
    for row, target in enumerate(targets):
        for col, arm in enumerate(arms):
            ax = axes[row][col]
            run = next((r for r in usable
                        if str(r.label)[:1] == arm and float(r.target) == target), None)
            name, color, _ = ARM_NAMES[arm]
            if run is None:
                ax.text(0.5, 0.5, "no run", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color=GRAY)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                probing = run.probing()
                ax.plot(run.t, run.force, color=color, lw=0.6)
                # 프로빙 구간을 옅게 깔아 "언제부터 로봇이 잡고 있었나" 를 보이게 한다.
                ax.fill_between(run.t, 0, 1, where=probing, transform=ax.get_xaxis_transform(),
                                color=NAVY, alpha=0.05, lw=0)
                ax.axhspan(run.target - run.band, run.target + run.band,
                           color=NAVY, alpha=0.12, lw=0)
                ax.axhline(run.target, color=INK, ls="--", lw=0.7)
                for ts, key in run.events:
                    if key in ("i", "w"):
                        ax.axvline(ts, color=GRAY, ls=":", lw=0.6)
                warn = run.meta.get("warn_contact_force_n") or run.meta.get("warn_force_n")
                if warn and run.force.max() > warn * 0.6:
                    ax.axhline(warn, color=GRAY, ls="-.", lw=0.7)
                # 힘 옆에 로봇 변위를 겹친다. 힘이 평평한 것이 "로봇이 잡고 있다" 인지
                # "로봇이 얼어 있는데 팬텀이 가만히 있다" 인지는 이 곡선 없이는 못 가른다.
                probing_idx = np.where(probing)[0]
                if probing_idx.size:
                    travel = axial_travel_series(run, int(probing_idx[0])) * 1000.0
                    if np.isfinite(travel).any():
                        twin = ax.twinx()
                        twin.plot(run.t, travel, color=GRAY_LIGHT, lw=0.55, ls="--", alpha=0.9)
                        twin.set_ylabel("travel [mm]", fontsize=6.5, color=GRAY)
                        twin.tick_params(labelsize=6.5, colors=GRAY)
                        twin.grid(False)
                ax.set_xlabel("t [s]", fontsize=7)
            ax.tick_params(labelsize=6.5)
            # 제목은 없다. 행(목표 힘)은 왼쪽 축 이름이, 열(조건)은 맨 아래 범례가 말한다.
            if col == 0:
                ax.set_ylabel(f"target {target:.1f} N\n‖F‖ [N]", fontsize=7)
    bottom = _arm_legend(figure, arms=tuple(arms), bottom=0.05 / max(rows, 1) + 0.012,
                         style="trace")
    figure.tight_layout(rect=(0, bottom, 1, 1))
    return _save(figure, stem)


# =========================================================================
# fig2 — 대역별 유지 품질
# =========================================================================
#: 회차 표식. 팔은 색으로 가르므로 회차는 채움으로 가른다 — 1회차 빈 원, 2회차
#: 찬 원, 그 다음은 세모. 합친 값은 검정 큰 원이다.
SESSION_STYLES = (
    dict(marker="o", mfc="white", mec=GRAY, mew=0.9),
    dict(marker="o", mfc=GRAY, mec="white", mew=0.5),
    dict(marker="^", mfc=GRAY_PALE, mec=GRAY, mew=0.7),
    dict(marker="v", mfc="white", mec=GRAY_PALE, mew=0.9),
)


def hold_quality(grouped: list, hold_rows: list = (), stem: str = "fig2_hold_quality") -> list:
    """대역별 유지 품질 — 절대 오차와 상대 오차, 그리고 밴드 안 체류율.

    ``hold_rows`` 가 여러 회차를 담고 있으면 회차별 값을 작은 표식으로, 합친 값을
    검정으로 겹쳐 그린다. 반복의 목적이 신뢰성이므로 **회차가 서로 얼마나
    가까운가**가 이 그림이 보여야 할 것이고, 합친 값만 그리면 그것이 사라진다.
    회차가 하나면 예전과 같은 그림이다.
    """
    if not grouped:
        return []
    target = np.array([g["target_n"] for g in grouped])
    rows = [r for r in hold_rows if r.get("run_type") == "hold"]
    sessions = sorted({r.get("session", "?") for r in rows})
    multi = len(sessions) > 1
    spacing = float(np.min(np.diff(target))) if target.size > 1 else 1.0
    # 회차 표식을 목표 힘 좌우로 조금씩 벌린다. 겹치면 회차가 하나로 보인다.
    offsets = (np.arange(len(sessions)) - (len(sessions) - 1) / 2) * spacing * 0.16

    def per_session(key):
        """회차 → (목표 힘, 값) 배열. 없는 목표는 건너뛴다."""
        out = {}
        for i, session in enumerate(sessions):
            pts = sorted((r["target_n"], r[key]) for r in rows if r.get("session") == session)
            if pts:
                out[session] = (np.array([p[0] for p in pts]) + offsets[i],
                                np.array([p[1] for p in pts]), SESSION_STYLES[i % len(SESSION_STYLES)])
        return out

    figure, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))

    # --- (a) 정상 상태 오차 ------------------------------------------------
    band = np.array([g.get("band_n") or 0.0 for g in grouped], float)
    if band.size and np.isfinite(band).all() and band.max() > 0:
        axes[0].fill_between(target, -band, band, color=NAVY, alpha=0.10, lw=0, step=None)
    if multi:
        for xs, ys, style in per_session("mean_error_n").values():
            axes[0].plot(xs, ys, ls="none", ms=3.6, **style)
    axes[0].errorbar(target, [g["mean_error_n"] for g in grouped],
                     yerr=[g["sd_n"] for g in grouped],
                     fmt="o", color=INK, ecolor=GRAY, elinewidth=1.0, capsize=3, ms=4.2,
                     zorder=3)
    axes[0].axhline(0, color=INK, lw=0.8)
    axes[0].set_xlabel("target force [N]")
    axes[0].set_ylabel("steady-state error, mean ± SD [N]")

    # --- (b) RMSE, 절대와 상대 ----------------------------------------------
    if multi:
        for xs, ys, style in per_session("rmse_n").values():
            axes[1].plot(xs, ys, ls="none", ms=3.6, **style)
    axes[1].plot(target, [g["rmse_n"] for g in grouped], "o-", color=INK, ms=4.2, lw=1,
                 zorder=3)
    twin = axes[1].twinx()
    twin.plot(target, [g["rmse_pct_of_target"] for g in grouped], ls="--", lw=0.9,
              color=GRAY, marker=None if multi else "s", ms=3.5)
    twin.set_ylabel("RMSE [% of target] (dashed)", color=GRAY)
    twin.tick_params(axis="y", colors=GRAY)
    twin.grid(False)
    axes[1].set_xlabel("target force [N]"); axes[1].set_ylabel("RMSE [N] (solid)")

    # --- (c) 밴드 안 체류율 --------------------------------------------------
    # 폭은 가장 좁은 목표 간격에서 정한다. 간격마다 다른 폭을 주면 막대 넓이가
    # 값처럼 읽혀, 목표가 촘촘한 저대역이 작아 보인다.
    if multi:
        width = spacing * 0.7 / len(sessions)
        for i, session in enumerate(sessions):
            pts = sorted((r["target_n"], r["in_band_pct"]) for r in rows
                         if r.get("session") == session)
            if not pts:
                continue
            style = SESSION_STYLES[i % len(SESSION_STYLES)]
            axes[2].bar([p[0] + offsets[i] for p in pts], [p[1] for p in pts], width=width,
                        color=style["mfc"], edgecolor=style["mec"] if style["mfc"] == "white" else "white",
                        lw=0.7)
        # 합친 값은 묶음 위를 가로지르는 검정 눈금.
        axes[2].hlines([g["in_band_pct"] for g in grouped],
                       target - spacing * 0.36, target + spacing * 0.36,
                       color=INK, lw=1.3, zorder=3)
    else:
        axes[2].bar(target, [g["in_band_pct"] for g in grouped], width=spacing * 0.6,
                    color=INK, alpha=0.9, edgecolor="white", lw=0.6)
    axes[2].axhline(100, color=INK, lw=0.8)
    axes[2].set_ylim(0, 105)
    axes[2].set_xlabel("target force [N]"); axes[2].set_ylabel("time inside the deadband [%]")

    if multi:
        handles = []
        for i, session in enumerate(sessions):
            style = SESSION_STYLES[i % len(SESSION_STYLES)]
            handles.append(Line2D([], [], ls="none", ms=4.5, label=f"session {i + 1} ({session})",
                                  **style))
        handles.append(Line2D([], [], color=INK, marker="o", ms=4.5, lw=1,
                              label="pooled (weighted by samples)"))
        figure.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7,
                      frameon=False, handlelength=2.0, columnspacing=2.0,
                      bbox_to_anchor=(0.5, 0.0))
        figure.tight_layout(rect=(0, 0.09, 1, 1))
    else:
        figure.tight_layout()
    return _save(figure, stem)


# =========================================================================
# fig3 — 교란 응답, 목표 힘별 소형 다중 패널
# =========================================================================
def disturbance(runs: list, events: list, stem: str = "fig3_disturbance") -> list:
    """주사기 조작에 정렬한 응답. 행 = 목표 힘, 열 = 주입 / 인출.

    두 장을 만든다: 힘 오차(``fig3_disturbance``)와 로봇 축방향 변위
    (``fig3b_disturbance_travel``). 같은 배치라 겹쳐 읽을 수 있다 — 힘이 돌아오는
    동안 변위가 계단으로 움직이면 로봇이 받아 낸 것이고, 변위가 평평하면 팬텀이
    스스로 누운 것이다. 힘 한 장으로는 그 둘이 같은 그림이 된다.

    한 패널에는 그 대역의 두 팔(켬 = 남색, 끔 = 주황)만 겹친다. 사건은 하나가
    선 하나다.
    """
    if not events:
        return []
    by_label = {run.label: run for run in runs}
    targets = sorted({float(e["target_n"]) for e in events})
    directions = ("in", "withdraw")

    # 먼저 창(window)을 다 잘라 둔다 — 두 장이 같은 사건 집합을 쓴다.
    cuts = []                     # (target, direction, arm, t, error, travel)
    for event in events:
        run = by_label.get(event["run"])
        if run is None:
            continue
        window = (run.t >= event["onset_s"] - 1.0) & \
                 (run.t <= event["onset_s"] + max(6.0, event["window_s"]))
        if window.sum() < 5:
            continue
        t = run.t[window] - event["onset_s"]
        idx = np.where(window)[0]
        travel = axial_travel_series(run, int(idx[0])) * 1000.0
        travel = travel[window] if np.isfinite(travel[idx]).any() else None
        cuts.append((float(event["target_n"]), event["direction"], str(event["run"])[:1],
                     t, run.force[window] - float(event["target_n"]), travel))

    def grid(kind, ylabel, title, out_stem):
        rows, columns = len(targets), len(directions)
        figure, axes = plt.subplots(rows, columns, sharex=True, sharey="row",
                                    figsize=(3.4 * columns + 0.3, 1.35 * rows + 0.9),
                                    squeeze=False)
        for row, target in enumerate(targets):
            for col, direction in enumerate(directions):
                ax = axes[row][col]
                for tgt, direc, arm, t, error, travel in cuts:
                    if tgt != target or direc != direction:
                        continue
                    y = error if kind == "error" else travel
                    if y is None:
                        continue
                    ax.plot(t, y, color=ARM_NAMES[arm][1], lw=0.6, alpha=0.8)
                ax.axhline(0, color=INK, lw=0.7)
                ax.axvline(0, color=INK, ls=":", lw=0.7)
                ax.tick_params(labelsize=6.5)
                # 제목은 없다. 열(주입/인출)은 맨 아래 축 이름이, 행(목표 힘)은
                # 왼쪽 축 이름이 말한다.
                if col == 0:
                    ax.set_ylabel(f"target {target:.1f} N\n{ylabel}", fontsize=7)
                if row == rows - 1:
                    ax.set_xlabel(f"t since syringe {direction} [s]", fontsize=7)
        bottom = _arm_legend(figure, arms=("B", "C"), bottom=0.025, style="trace")
        figure.tight_layout(rect=(0, bottom, 1, 1))
        return _save(figure, out_stem)

    paths = grid("error", "error [N]",
                 "Force error after the surface moves, per target force",
                 stem)
    paths += grid("travel", "travel [mm]",
                  "Robot axial travel after the surface moves, per target force",
                  stem.replace("fig3_", "fig3b_") + "_travel")
    return paths


# =========================================================================
# fig4 — 실행별 최대 힘과 한계
# =========================================================================
def safety_margin(rows: list, stem: str = "fig4_safety_margin") -> list:
    """최악 힘과 한계. 넘었는지 아닌지가 한눈에 보여야 한다.

    막대는 팔별로 묶고 목표 힘 순으로 늘어놓는다. 색은 팔, 빗금은 한계 초과.
    """
    rows = [r for r in rows if np.isfinite(r.get("max_force_n", float("nan")))]
    if not rows:
        return []
    order = sorted(rows, key=lambda r: (str(r["run"])[:1], float(r["target_n"])))
    figure, ax = plt.subplots(figsize=(7.2, 2.9))
    ticks, ticklabels, x, prev = [], [], 0, None
    for r in order:
        arm = str(r["run"])[:1]
        if prev is not None and arm != prev:
            ax.axvline(x, color=GRID, lw=.8)
            x += 1
        prev = arm
        name, color, _ = _arm(r["run"])
        exceeded = str(r.get("exceeded_limit")).lower() in ("true", "1")
        ax.bar(x, r["peak_force_n"], color=color, alpha=0.9, edgecolor="white", lw=0.6,
               hatch="////" if exceeded else None)
        if exceeded:
            ax.bar(x, r["peak_force_n"], color="none", edgecolor=ORANGE_DARK, lw=0.9)
        ticks.append(x); ticklabels.append(f"{float(r['target_n']):.1f}")
        x += 1
    ax.axhline(rows[0]["max_force_n"], color=ORANGE_DARK, ls="--", lw=1.1)
    ax.axhline(rows[0]["warn_force_n"], color=GRAY, ls="-.", lw=0.9)
    ax.text(x - 0.4, rows[0]["max_force_n"], " limit", va="bottom",
            fontsize=7, color=ORANGE_DARK)
    ax.text(x - 0.4, rows[0]["warn_force_n"], " warn", va="bottom",
            fontsize=7, color=GRAY)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, rotation=90, fontsize=6.5)
    ax.set_xlabel("target force [N]")
    ax.set_ylabel("peak ‖F‖ in the run [N]")
    exceeded_proxy = Patch(facecolor="white", edgecolor=ORANGE_DARK, hatch="////",
                           label="exceeded the force limit")
    bottom = _arm_legend(figure, extra=(exceeded_proxy,), bottom=0.1, style="patch")
    figure.tight_layout(rect=(0, bottom, 1, 1))
    return _save(figure, stem)


# =========================================================================
# fig5 — 힘은 평평, 팔은 물러남
# =========================================================================
def regulation_evidence(runs: list, evidence: list, stiff: list,
                        stem: str = "fig5_regulation_evidence",
                        curve: dict | None = None, events: list = (),
                        excluded_targets=()) -> list:
    """힘은 일정한데 팔이 물러났다는 것을, 강성과 함께 한 장에 둔다.

    (a) 팬텀 강성 — 무교란 유지 여덟 번의 **평형점**(힘, 압입 깊이)과 직선 맞춤.
        제어기가 어느 점에 앉을지만 정하므로 점 자체는 팬텀 F(z) 위에 있다.
    (b) 개루프 교란 크기 — 힘 유지를 끄고 팬텀만 움직인 C 팔에서 주사 한 스텝당
        힘 변화(대역별 중앙값·IQR). 로봇이 고정돼 변위가 없으니 강성은 못 주지만,
        (a)의 k 로 나누면 표면이 스텝당 몇 mm 올라왔는지가 된다(오른쪽 눈금).
    (c) 교란 중 힘과 축방향 변위 시계열 — 힘이 평평하고 변위가 계단으로
        내려가면 그 계단이 로봇이 흡수한 양이다.
    """
    usable = [s for s in stiff if s.get("ok") and s.get("method") != "steady-state equilibria"]
    by_label = {r.label: r for r in runs}
    picked = next((e for e in evidence if e["direction"] == "in"), None)
    curve = curve if (curve and curve.get("ok")) else None
    if not usable and picked is None and curve is None:
        return []

    figure, axes = plt.subplots(1, 3, figsize=(9.4, 3.0))

    # -- (a) 강성 -------------------------------------------------------
    ax = axes[0]
    if curve is not None:
        pts = curve["curve"]
        depth = np.array([p["depth_mm"] for p in pts])
        force = np.array([p["force_n"] for p in pts])
        k_mm = curve["k_n_per_mm"]
        name, color, marker = ARM_NAMES["A"]
        line_x = np.linspace(min(depth.min(), curve["contact_depth_mm"]), depth.max(), 2)
        ax.plot(line_x, k_mm * line_x + curve["intercept_n"], color=NAVY, lw=1.1, zorder=2)
        ax.plot(depth, force, "o", color=color, ms=5, mec="white", mew=.6,
                ls="none", zorder=3)
        ax.axhline(0, color=INK, lw=.6)
        ax.text(0.04, 0.96,
                f"k = {k_mm:.3f} N/mm\nR² = {curve['r_squared']:.3f}\n"
                f"{curve['points']} equilibria, last {curve['settle_s']:.0f} s of each hold",
                transform=ax.transAxes, fontsize=6.8, va="top", color=INK)
        ax.set_xlabel(f"indentation past the {pts[0]['target_n']:.1f} N equilibrium [mm]")
    elif usable:
        s0 = usable[0]
        k = s0["k_n_per_m"]
        travel = np.linspace(0, max(s0["travel_span_mm"], 0.1) / 1000.0, 20)
        ax.plot(travel * 1000.0, k * travel + s0["intercept_n"], color=NAVY, lw=1.2)
        ax.text(0.04, 0.96, f"k = {k/1000.0:.3f} N/mm\nR² = {s0['r_squared']:.3f}",
                transform=ax.transAxes, fontsize=6.8, va="top", color=INK)
        ax.set_xlabel("axial travel [mm]")
    else:
        ax.text(0.5, 0.5, "no stiffness estimate", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color=GRAY)
        ax.set_xlabel("axial travel [mm]")
    ax.set_ylabel("‖F‖ [N]")

    # -- (b) 개루프 교란 크기 (C 팔) ----------------------------------------
    ax = axes[1]
    off = [e for e in events if str(e["run"])[:1] == "C"]
    if off:
        targets = sorted({float(e["target_n"]) for e in off})
        excluded = {float(t) for t in excluded_targets}
        name, color, marker = ARM_NAMES["C"]
        for target in sorted(excluded & set(targets)):
            ax.axvspan(target - 0.22, target + 0.22, color=LIGHT, zorder=0, lw=0)
        for direction, filled, dx in (("in", True, -0.06), ("withdraw", False, 0.06)):
            xs, meds, los, his = [], [], [], []
            for target in targets:
                v = np.array([abs(float(e["peak_error_n"])) for e in off
                              if float(e["target_n"]) == target and e["direction"] == direction])
                if not v.size:
                    continue
                xs.append(target + dx); meds.append(np.median(v))
                los.append(np.percentile(v, 25)); his.append(np.percentile(v, 75))
            ax.vlines(xs, los, his, color=color, lw=1.2, zorder=2)
            ax.plot(xs, meds, marker, color=color if filled else "white", mec=color,
                    mew=1.0, ms=3.5, ls="none", zorder=3,
                    label=f"syringe {direction}")
        ax.set_xticks(targets)
        ax.set_xticklabels([f"{t:.1f}" for t in targets], fontsize=7)
        ax.set_xlabel("target force [N]")
        ax.set_ylabel("|ΔF| per syringe step, force hold off [N]")
        # 축은 개루프가 성립한 대역에 맞춘다. 안전 한계가 개입한 대역은 회색으로
        # 덮고 값을 글자로 적는다 — 지우면 보고서에서 사라진다.
        kept = [abs(float(e["peak_error_n"])) for e in off
                if float(e["target_n"]) not in excluded]
        top = (max(kept) if kept else max(abs(float(e["peak_error_n"])) for e in off)) * 1.3
        ax.set_ylim(0, top)
        for target in sorted(excluded & set(targets)):
            v = [abs(float(e["peak_error_n"])) for e in off if float(e["target_n"]) == target]
            ax.annotate(f"{np.median(v):.1f}", xy=(target, top * 0.97),
                        xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=6.5, color=color, annotation_clip=False)
            ax.plot([target], [top * 0.97], marker, color=color, ms=3.5, mec="white",
                    mew=.8, clip_on=False, zorder=4)
        ax.legend(frameon=False, fontsize=6.8, loc="upper left", handletextpad=0.4)
        if curve is not None and curve["k_n_per_mm"] > 0:
            k_mm = curve["k_n_per_mm"]
            right = ax.secondary_yaxis("right", functions=(lambda f: f / k_mm,
                                                           lambda z: z * k_mm))
            right.set_ylabel("surface rise ΔF / k [mm]", color=GRAY, fontsize=7)
            right.tick_params(labelsize=7, colors=GRAY)
    else:
        ax.text(0.5, 0.5, "no force-hold-off runs", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color=GRAY)

    # -- (c) 흡수 증거 시계열 ----------------------------------------------
    ax = axes[2]
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
                name, color, _ = _arm(run.label)
                ax.plot(t, run.force[window], color=color, lw=0.8, label="‖F‖ [N]")
                twin = ax.twinx()
                twin.plot(t, travel, color=GRAY_LIGHT, lw=1.0, ls="--", label="travel [mm]")
                twin.set_ylabel("axial travel [mm]", color=GRAY)
                twin.tick_params(colors=GRAY)
                twin.grid(False)
                ax.axhline(picked["target_n"], color=INK, ls=":", lw=0.8)
                ax.axvline(0, color=INK, ls=":", lw=0.8)
                # 곡선이 왼쪽 위에서 시작해 오른쪽 아래로 내려오므로 글은 오른쪽 위에.
                ax.text(0.97, 0.96,
                        f"{name}\ntarget {float(picked['target_n']):.1f} N\n"
                        f"held {picked['held_mean_n']:.2f} N while retreating "
                        f"{abs(picked['travel_mm']):.1f} mm",
                        transform=ax.transAxes, fontsize=6.8, va="top", ha="right",
                        color=INK)
    ax.set_xlabel("t since syringe in [s]"); ax.set_ylabel("‖F‖ [N]")
    figure.tight_layout(w_pad=1.6)
    return _save(figure, stem)


def axial_travel_safe(run, start, stop):
    from fh.analysis import axial_travel
    try:
        return axial_travel(run, start, stop)
    except Exception:
        return None


# =========================================================================
# fig6 — 노출 시간으로 본 안전
# =========================================================================
def excursion_recovery(runs: list, excursions: list, exposure: list,
                       stem: str = "fig6_excursion_recovery") -> list:
    """안전을 노출 시간으로 논증하는 그림.

    한계에 닿은 적이 없으면 "넘지 않았다" 는 시험된 사실이 아니다. 남는 질문은
    **벗어나 있던 동안이 얼마나 길었는가** 이고, 조직이 실제로 겪는 것도 그것이다.

    위 줄은 목표 힘마다 패널 하나 — 문턱을 올려 가며 잰 **한 번에 가장 오래**
    머문 시간. 아래 왼쪽은 밴드를 벗어난 구간 하나하나의 복귀 시간(목표 힘별,
    팔별), 아래 오른쪽은 가장 세게 눌린 실행의 시계열에 그 구간을 칠한 것이다.
    범례는 그림 맨 아래 한 줄에 한 번만 둔다.
    """
    if not excursions:
        return []
    targets = sorted({float(e["target_n"]) for e in exposure} |
                     {float(e["target_n"]) for e in excursions})
    n = max(len(targets), 1)
    figure = plt.figure(figsize=(8.6, 5.6))
    # 위 줄은 n 개, 아래 줄은 5:3 — 두 줄의 열 수가 달라 tight_layout 이 못 다루므로
    # 여백을 직접 준다 (아래 0.14 는 범례 줄 + 세운 눈금 글자 몫).
    gs = figure.add_gridspec(2, n, height_ratios=(1.0, 1.3), hspace=0.5, wspace=0.14,
                             left=0.075, right=0.985, top=0.975, bottom=0.14)
    top = [figure.add_subplot(gs[0, i]) for i in range(n)]
    # 아래 줄은 위 줄의 격자를 다시 5:3 으로 나누되, 두 패널 사이를 넓게 띄운다 —
    # 오른쪽 패널의 y 축 이름이 왼쪽 패널 위로 올라가지 않게.
    sub = gs[1, :].subgridspec(1, 2, width_ratios=(5, 3), wspace=0.28)
    left = figure.add_subplot(sub[0, 0])
    right = figure.add_subplot(sub[0, 1])

    warn = next((r.meta.get("warn_contact_force_n") for r in runs
                 if r.meta.get("warn_contact_force_n")), None)
    limit = next((r.meta.get("max_contact_force_n") for r in runs
                  if r.meta.get("max_contact_force_n")), None)

    # --- 위: 목표 힘별 노출 곡선 -----------------------------------------
    ymax = max((r["longest_s"] for r in exposure), default=1.0) * 1.12
    xmax = max(limit or 0, max((r["level_n"] for r in exposure), default=1.0)) + 0.3
    for ax, target in zip(top, targets):
        for label in sorted({e["run"] for e in exposure
                             if float(e["target_n"]) == target}):
            rows = sorted([e for e in exposure if e["run"] == label],
                          key=lambda e: e["level_n"])
            if max(r["longest_s"] for r in rows) <= 0:
                continue
            name, color, marker = _arm(label)
            ax.plot([r["level_n"] for r in rows], [r["longest_s"] for r in rows],
                    marker=marker, ls="-", ms=2.6, lw=0.9, alpha=.9, color=color,
                    mec="white", mew=.3)
        if warn:
            ax.axvline(warn, color=GRAY, ls="-.", lw=.7)
        if limit:
            ax.axvline(limit, color=ORANGE_DARK, ls="--", lw=.9)
        ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
        ax.text(0.06, 0.95, f"target {target:.1f} N", transform=ax.transAxes,
                fontsize=6.8, va="top", ha="left", color=INK)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("threshold [N]", fontsize=7)
        if ax is top[0]:
            ax.set_ylabel("longest episode\nabove threshold [s]", fontsize=7)
        else:
            ax.tick_params(labelleft=False)

    # --- 아래 왼쪽: 복귀 시간 ---------------------------------------------
    by_run: dict = {}
    for e in excursions:
        by_run.setdefault(e["run"], []).append(e)
    order = sorted(by_run, key=lambda r: (str(r)[:1], float(by_run[r][0]["target_n"])))
    ticks, ticklabels, x, prev = [], [], 0, None
    rng = np.random.default_rng(0)                      # 지터를 재현 가능하게
    for label in order:
        arm = str(label)[:1]
        if prev is not None and arm != prev:
            left.axvline(x, color=GRID, lw=.8)
            x += 1
        prev = arm
        name, color, marker = _arm(label)
        d = [e["duration_s"] for e in by_run[label]]
        left.scatter(np.full(len(d), x) + rng.uniform(-.08, .08, len(d)), d,
                     s=16, color=color, marker=marker, alpha=.8,
                     edgecolor="white", lw=.5)
        left.plot([x - .22, x + .22], [np.median(d)] * 2, color=INK, lw=1.4)
        ticks.append(x); ticklabels.append(f"{float(by_run[label][0]['target_n']):.1f}")
        x += 1
    worst = max(e["duration_s"] for e in excursions)
    left.axhline(worst, color=GRAY, ls="--", lw=.8)
    left.text(x - .5, worst, f" longest {worst:.2f} s", fontsize=7,
              va="bottom", ha="right", color=GRAY)
    left.set_xticks(ticks)
    left.set_xticklabels(ticklabels, fontsize=6.5, rotation=90)
    left.set_xlabel("target force [N]")
    left.set_ylabel("time to return to band\nper excursion [s]")
    left.set_ylim(bottom=0)

    # --- 아래 오른쪽: 가장 세게 눌린 실행 --------------------------------
    hardest = max(excursions, key=lambda e: e["peak_n"])
    run = next((r for r in runs if r.label == hardest["run"]), None)
    if run is not None:
        mask = run.probing()
        t, f = run.t[mask], run.force[mask]
        band_top = hardest["band_top_n"]
        name, color, marker = _arm(run.label)
        right.plot(t, f, color=color, lw=.6)
        right.fill_between(t, band_top, f, where=f > band_top, color=ORANGE_DARK,
                           alpha=.3, lw=0)
        right.axhline(band_top, color=INK, ls="--", lw=.8)
        if warn:
            right.axhline(warn, color=GRAY, ls="-.", lw=.9)
        if limit:
            right.axhline(limit, color=ORANGE_DARK, ls="--", lw=1.1)
        right.text(0.97, 0.05, f"{name}\ntarget {float(run.target):.1f} N, peak {f.max():.2f} N",
                   transform=right.transAxes, fontsize=6.8, va="bottom", ha="right", color=INK)
    right.set_xlabel("t [s]"); right.set_ylabel("‖F‖ [N]")

    # 문턱선 설명은 축 안에 쓰지 않는다 — 마지막 패널의 곡선 위에 얹히므로.
    # 팔 범례와 같은 줄, 그림 맨 아래에 함께 둔다.
    guides = []
    if warn:
        guides.append(Line2D([], [], color=GRAY, ls="-.", lw=.8,
                             label=f"warn ({warn:g} N)"))
    if limit:
        guides.append(Line2D([], [], color=ORANGE_DARK, ls="--", lw=.9,
                             label=f"limit ({limit:g} N)"))
    _arm_legend(figure, extra=tuple(guides))
    return _save(figure, stem)


# =========================================================================
# fig7 — 위약 대조
# =========================================================================
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
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # -- (a) 대역별 -----------------------------------------------------
    ax = axes[0]
    for target in excluded_targets:
        ax.axvspan(target - 0.22, target + 0.22, color=LIGHT, zorder=0, lw=0)
    arms = tuple((a, *ARM_NAMES[a]) for a in ("B", "C"))
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
        ax.annotate("force limit reached,\nnot open-loop", xy=(edge, 0),
                    xytext=(edge - 0.12, ax.get_ylim()[1] * 0.30),
                    ha="right", va="center", fontsize=7, color=GRAY)
    ax.set_xticks(targets)
    ax.set_xticklabels([f"{t:.1f}" for t in targets], fontsize=7.5)
    ax.set_xlabel("target force [N]")
    ax.set_ylabel("peak |F − target| per event,\nmedian and IQR [N]")

    # -- (b) 전체 분포 --------------------------------------------------
    ax = axes[1]
    counts = {}
    for arm, label, color, marker in arms:
        values = np.sort(peaks(arm))
        if not values.size:
            continue
        counts[arm] = values.size
        fraction = np.arange(1, values.size + 1) / values.size
        ax.step(values, fraction, where="post", color=color, lw=1.6)
        median = float(np.median(values))
        ax.plot([median], [0.5], marker, color=color, ms=5.0, mec="white", mew=0.8)
    ax.set_xlabel("peak |F − target| per event [N]")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("cumulative fraction of all events")
    if p_value == p_value:
        n_text = "\n".join(f"n = {counts[a]} ({'on' if a == 'B' else 'off'})"
                           for a in ("B", "C") if a in counts)
        ax.text(0.44, 0.40, f"rank-sum two-sided\np = {p_value:.2f}\n{n_text}",
                transform=ax.transAxes, fontsize=7.5, color=INK, va="center")

    bottom = _arm_legend(figure, arms=("B", "C"), bottom=0.1)
    figure.tight_layout(rect=(0, bottom, 1, 1))
    return _save(figure, stem)
