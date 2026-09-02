"""캡처를 지표로 바꾼다.

**원값으로 잰다.** 조절기는 표본 하나하나에 반응하므로, 걸러낸 신호로 판정하면
존재하지 않는 루프를 평가하게 된다. 평활은 그림에만 쓰고 표에는 쓰지 않는다.
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

#: 접촉 프로빙으로 인정하는 모드 문자열. 면내 회전 모드도 힘 루프는 같다.
PROBING_MODES = ("contact_probing", "contact_probing_inplane")

#: 교란 직전 기준을 잡는 구간 [s]. 조작자가 키를 누르는 것과 손이 주사기를
#: 움직이는 것 사이의 지연을 덮을 만큼은 길고, 이전 사건의 잔여가 섞이지 않을
#: 만큼은 짧아야 한다.
PRE_EVENT_S = 0.3


@dataclass
class Run:
    """캡처 한 번."""

    label: str
    run_type: str
    meta: dict
    t: np.ndarray
    force: np.ndarray            # ‖F‖, 접촉력 크기 [N]
    normal: np.ndarray           # F_n [N]
    mode: np.ndarray             # 문자열 배열
    calibration_valid: np.ndarray
    events: list                 # (t_s, key) 목록
    #: 플랜지 위치 [m] 와 자세 쿼터니언. 자세를 못 받았으면 NaN 이다.
    tip: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    quat: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    included: bool = True
    reason: str = ""
    #: 모드를 기록이 아니라 힘에서 되짚었는가.
    mode_reconstructed: bool = False

    @property
    def target(self) -> float:
        return float(self.meta.get("target_force_n", float("nan")))

    @property
    def band(self) -> float:
        return float(self.meta.get("deadband_n", float("nan")))

    def probing(self) -> np.ndarray:
        """접촉 프로빙 구간 마스크. 힘 루프가 돌고 있던 표본만 판정한다.

        모드가 기록돼 있으면 그것을 쓴다. 비어 있으면 **재구성한다** — 그 실행의
        진입·이탈 문턱을 기록된 힘 위에 다시 돌린다. 재구성은 기록보다 약한
        근거이므로 :attr:`mode_reconstructed` 로 표시하고 보고서가 그렇게 적는다.
        """
        recorded = np.isin(self.mode, PROBING_MODES)
        if recorded.any():
            return recorded
        enter = float(self.meta.get("contact_probing_force_n", float("nan")))
        release = float(self.meta.get("contact_probing_release_n", 0.0))
        if not math.isfinite(enter):
            return recorded
        self.mode_reconstructed = True
        out = np.zeros(self.force.size, bool)
        inside = False
        for i, f in enumerate(self.force):
            inside = f >= enter if not inside else f > release
            out[i] = inside
        return out


def load_run(samples_path: str) -> Run:
    """``*_samples.csv`` 와 짝인 ``*_meta.json`` 을 읽는다."""
    meta_path = samples_path.replace("_samples.csv", "_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)

    t, force, normal, mode, valid, events = [], [], [], [], [], []
    tip, quat = [], []
    with open(samples_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t.append(float(row["t_s"]))
            force.append(float(row["mag_n"]))
            normal.append(float(row["fn_n"]))
            mode.append(row.get("probing_mode", ""))
            # 빈 칸은 "무효" 가 아니라 **아직 못 받았다** 이다. 이 신호는 1 Hz
            # 타이머로 오므로 캡처 첫 0.5 초는 늘 비어 있고, 그것을 무효로 읽으면
            # 멀쩡한 실행이 통째로 버려진다. 셋으로 나눠 둔다.
            cell = row.get("calibration_valid", "").upper()
            valid.append(True if cell == "TRUE" else (False if cell == "FALSE" else None))
            key = row.get("event", "")
            if key:
                events.append((float(row["t_s"]), key))
            # 자세는 있으면 쓰고 없으면 NaN 이다. 0 으로 채우면 원점에 있었다는
            # 뜻이 되어, 변위를 재는 쪽에서 조용히 거대한 값이 나온다.
            def cell(name):
                raw = row.get(name, "")
                return float(raw) if raw else float("nan")
            tip.append([cell("tip_x_m"), cell("tip_y_m"), cell("tip_z_m")])
            quat.append([cell("q_x"), cell("q_y"), cell("q_z"), cell("q_w")])

    run = Run(
        label=meta.get("label", os.path.basename(samples_path)),
        run_type=meta.get("run_type", "unknown"),
        meta=meta,
        t=np.asarray(t, float),
        force=np.asarray(force, float),
        normal=np.asarray(normal, float),
        mode=np.asarray(mode, dtype=object),
        calibration_valid=np.asarray(valid, dtype=object),
        tip=np.asarray(tip, float),
        quat=np.asarray(quat, float),
        events=events,
    )
    _screen(run)
    return run


def _screen(run: Run) -> None:
    """받아들일 수 있는 캡처인가. 거절은 사유와 함께 남는다."""
    if run.t.size < 100:
        run.included, run.reason = False, f"표본 부족 {run.t.size}"
        return
    if not np.isfinite(run.target):
        run.included, run.reason = False, "목표 힘이 기록되지 않았다"
        return
    if run.run_type == "stiffness":
        # 개루프가 이 실행의 정의다. 접촉 프로빙에 들어갔다면 그쪽이 잘못이며,
        # stiffness() 가 그 사유로 거절한다.
        return
    if not run.probing().any():
        run.included, run.reason = False, "접촉 프로빙에 한 번도 들어가지 않았다"
        return
    # 명시적으로 거짓인 표본만 결격이다. 아직 못 받은 것(None)은 모른다는 뜻이고,
    # 모르는 것을 무효로 세면 캡처 시작 순간 때문에 실행이 버려진다.
    during = run.calibration_valid[run.probing()]
    if any(v is False for v in during):
        run.included, run.reason = False, "프로빙 중 교정이 무효가 된 구간이 있다"
        return
    if len(during) and all(v is None for v in during):
        run.included, run.reason = False, "교정 유효성을 한 번도 못 받았다"
        return


def loop_state(run: Run) -> str:
    """이 실행에서 힘 루프가 실제로 돌았는가.

    모드 기록이 없는 실행이 많아(2026-09-02 캡처의 QoS 결함) 힘과 그 실행이
    **기록해 둔** 진입 문턱으로 되짚는다. 목표가 문턱보다 낮게 설정된 실행은
    로봇이 한 번도 조절하지 않았고, 그 기록은 조작자가 팔을 고정한 개루프다 —
    버릴 데이터가 아니라 **다른 데이터**이므로 그렇게 이름 붙인다.
    """
    enter = float(run.meta.get("contact_probing_force_n", float("nan")))
    if not math.isfinite(enter):
        return "unknown"
    share = float((run.force >= enter).mean())
    if share > 0.5:
        return "closed"
    if share < 0.05:
        return "open"
    return "marginal"


def rejection(run: Run, tail_s: float = 0.8) -> list:
    """사건마다 얼마나 되돌아왔는가.

    피크만으로는 조절을 못 본다 — 개루프도 주입을 멈추면 팬텀이 스스로 눕는다.
    가르는 것은 **창 끝에 남은 값**이다: 루프가 있으면 목표로 돌아오고, 없으면
    올라간 채로 남는다.

    ⚠️ 교란이 데드밴드보다 작으면 조절기는 **설계상 아무것도 하지 않는다.**
    그때 개루프와 폐루프가 같아 보이는 것은 루프가 없어서가 아니라 부를 일이
    없어서다. `outside_band` 가 그 구분을 들고 있다.
    """
    out = []
    marks = [(ts, key) for ts, key in run.events if key in ("i", "w")]
    state = loop_state(run)
    for index, (onset, key) in enumerate(marks):
        end = marks[index + 1][0] if index + 1 < len(marks) else run.t[-1]
        base = (run.t >= onset - PRE_EVENT_S) & (run.t < onset)
        window = (run.t >= onset) & (run.t < end)
        tail = (run.t >= end - tail_s) & (run.t < end)
        if base.sum() < 5 or window.sum() < 50 or tail.sum() < 5:
            continue
        b = float(run.force[base].mean())
        delta = run.force[window] - b
        peak = float(delta[np.argmax(np.abs(delta))])
        residual = float(run.force[tail].mean() - b)
        out.append({
            "run": run.label,
            "loop": state,
            "target_n": run.target,
            "band_n": run.band,
            "direction": "in" if key == "i" else "withdraw",
            "step_ml": run.meta.get("step_ml"),
            "peak_delta_n": peak,
            "residual_delta_n": residual,
            "recovered_pct": (1 - residual / peak) * 100 if abs(peak) > 1e-6 else None,
            "outside_band": bool(abs(peak) > run.band),
        })
    return out


def settling_point(run: Run) -> float:
    """조절기가 실제로 멈추는 힘 [N].

    **밴드 안에서는 멈춘 자리가 곧 목표다.** `ForceRegulator.update` 는 오차가
    데드밴드 안이면 ``v_z = 0`` 을 내고 더 가지 않는다 — 목표는 도달할 지점이
    아니라 그 둘레 밴드를 정의하는 값이며, 밴드에 처음 들어선 자리가 그 실행의
    운전점이 된다.

    아래에서 올라오면 그 자리는 밴드 아래끝이다. 다만 진입 문턱이 그보다 높으면
    조절기는 문턱에서야 처음 불리므로, 둘 중 큰 쪽이 정착점이다::

        정착점 = max(진입 문턱, 목표 − 밴드)

    ⚠️ 위에서 내려오는 경우(조작자가 세게 누른 뒤 조절기가 물러나는 경우)에는
    밴드 **위**끝에 정착하므로 이 식이 맞지 않는다. 2026-09-02 실행은 전부 아래에서
    올라온 경우였고, 첫 1 초부터 정착값에 있었던 것이 그 증거다.
    """
    enter = float(run.meta.get("contact_probing_force_n", float("nan")))
    bottom = run.target - run.band
    if not math.isfinite(enter):
        return bottom
    return max(enter, bottom)


def hold_metrics(run: Run, settle_s: float = 3.0) -> dict:
    """정상 유지 구간의 지표.

    ``settle_s`` 만큼은 버린다 — 프로빙에 막 들어간 구간은 목표로 가는 과도이지
    유지가 아니다. 무엇을 버렸는지는 표에 남는다.
    """
    mask = run.probing()
    if mask.any():
        start = run.t[mask][0] + settle_s
        mask &= run.t >= start
    if not mask.any():
        return {"samples": 0}

    force = run.force[mask]
    error = force - run.target
    inside = np.abs(error) <= run.band
    # 운전점 대비 오차. 목표 대비 편차는 대부분 데드밴드가 설계대로 낸 것이므로,
    # 추종이 얼마나 정확했는지는 이쪽으로 봐야 한다.
    point = settling_point(run)
    return {
        "samples": int(force.size),
        "settling_point_n": float(point),
        "error_vs_settling_n": float((force - point).mean()),
        "seconds": float(run.t[mask][-1] - run.t[mask][0]),
        "settle_discarded_s": settle_s,
        "mean_error_n": float(error.mean()),
        "sd_n": float(error.std(ddof=1)) if error.size > 1 else 0.0,
        "rmse_n": float(np.sqrt((error**2).mean())),
        "mae_n": float(np.abs(error).mean()),
        "in_band_pct": float(inside.mean() * 100.0),
        "p95_abs_error_n": float(np.percentile(np.abs(error), 95)),
        "max_force_n": float(force.max()),
        "min_force_n": float(force.min()),
        # 상대 오차. "어느 대역을 가장 잘 잡는가" 는 절대와 상대가 다른 답을 준다.
        "rmse_pct_of_target": float(np.sqrt((error**2).mean()) / run.target * 100.0),
    }


def disturbance_events(run: Run, settle_band_s: float = 0.5) -> list:
    """주사기 조작 하나하나의 응답.

    되돌아온 시점은 밴드 안에 ``settle_band_s`` 동안 **연속으로** 머문 첫 순간이다.
    한 표본만 스쳐도 복귀로 치면 진동하는 루프가 즉시 복귀한 것처럼 보인다.
    """
    out = []
    marks = [(ts, key) for ts, key in run.events if key in ("i", "w")]
    for index, (onset, key) in enumerate(marks):
        end = marks[index + 1][0] if index + 1 < len(marks) else run.t[-1]
        window = (run.t >= onset) & (run.t <= end) & run.probing()
        if window.sum() < 10:
            continue
        t = run.t[window]
        force = run.force[window]
        error = force - run.target

        peak_index = int(np.argmax(np.abs(error)))
        inside = np.abs(error) <= run.band
        settle_t = _first_sustained(t, inside, settle_band_s)
        out.append({
            "run": run.label,
            "target_n": run.target,
            "onset_s": float(onset),
            "direction": "in" if key == "i" else "withdraw",
            "step_ml": run.meta.get("step_ml"),
            "peak_error_n": float(error[peak_index]),
            "peak_force_n": float(force[peak_index]),
            "time_to_peak_s": float(t[peak_index] - onset),
            "settle_s": None if settle_t is None else float(settle_t - onset),
            "recovered": settle_t is not None,
            "window_s": float(t[-1] - t[0]),
        })
    return out


def _first_sustained(t: np.ndarray, inside: np.ndarray, hold_s: float):
    """``inside`` 가 ``hold_s`` 동안 이어지는 첫 시각. 없으면 ``None``."""
    start = None
    for index, ok in enumerate(inside):
        if not ok:
            start = None
            continue
        if start is None:
            start = index
        if t[index] - t[start] >= hold_s:
            return t[start]
    return None


def probe_axis(run: Run, index: int) -> np.ndarray:
    """그 시점 프로브 침투축(+z)의 베이스 프레임 방향. 자세가 없으면 ``None``."""
    q = run.quat[index] if index < len(run.quat) else None
    if q is None or not np.all(np.isfinite(q)):
        return None
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return None
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    # 회전행렬의 3열 = 도구 z 축이 베이스에서 보는 방향.
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w),
                     1 - 2 * (x * x + y * y)])


def axial_travel(run: Run, start: int, stop: int):
    """``start`` 에서 ``stop`` 까지 프로브 축 방향으로 간 거리 [m].

    양수 = 조직 쪽으로 파고들었다. 음수 = 물러났다.

    축에 **투영**하는 이유는, 조절기가 여는 축이 프로브 z 하나이기 때문이다.
    베이스 z 변위를 그냥 쓰면 프로브가 기울어 있을 때 그만큼 틀린다.
    """
    if run.tip.size == 0:
        return None
    a, b = run.tip[start], run.tip[stop]
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    axis = probe_axis(run, start)
    if axis is None:
        return None
    return float(np.dot(b - a, axis))


def stiffness(run: Run, settle_s: float = 0.5) -> dict:
    """개루프 실행에서 팬텀 강성 ``k = ΔF/Δz`` 를 잰다 [N/m].

    **힘 축이 닫혀 있어야 뜻이 있다.** 조절기가 돌고 있으면 팔이 힘을 흡수하므로
    여기서 나오는 기울기는 팬텀이 아니라 루프의 성질이 된다. 그래서 이 실행은
    ``teleop.contact_probing_enabled: false`` 로 잡고, 접촉 프로빙 구간이 하나라도
    있으면 거절한다.

    표시 사이 구간마다 (힘, 축방향 위치) 한 쌍을 만들고 그 위에 직선을 맞춘다.
    부피를 넣은 만큼 표면이 올라오고, 팔이 가만히 있으므로 그 전부가 힘이 된다.
    """
    if run.probing().any():
        return {"ok": False, "reason": "접촉 프로빙 구간이 있다 — 개루프가 아니다"}
    marks = [ts for ts, key in run.events if key in ("i", "s", " ")]
    if len(marks) < 3:
        return {"ok": False, "reason": f"표시가 부족하다 ({len(marks)}개, 3개 이상 필요)"}

    force, travel = [], []
    base = int(np.argmin(np.abs(run.t - marks[0])))
    for ts in marks:
        index = int(np.argmin(np.abs(run.t - ts)))
        window = (run.t >= ts - settle_s) & (run.t <= ts)
        if window.sum() < 5:
            continue
        z = axial_travel(run, base, index)
        if z is None:
            continue
        force.append(float(run.force[window].mean()))
        travel.append(z)
    if len(force) < 3:
        return {"ok": False, "reason": "자세가 없어 변위를 못 잰다 — ee_wrt_base 가 왔는가"}

    force = np.asarray(force)
    travel = np.asarray(travel)
    slope, intercept = np.polyfit(travel, force, 1)
    predicted = slope * travel + intercept
    ss_res = float(((force - predicted) ** 2).sum())
    ss_tot = float(((force - force.mean()) ** 2).sum())
    return {
        "ok": True,
        "run": run.label,
        "points": len(force),
        "k_n_per_m": float(slope),
        "k_n_per_mm": float(slope) / 1000.0,
        "intercept_n": float(intercept),
        "r_squared": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "force_span_n": float(force.max() - force.min()),
        "travel_span_mm": float((travel.max() - travel.min()) * 1000.0),
    }


def regulation_evidence(run: Run, k_n_per_m: float) -> list:
    """조절이 일어났다는 직접 증거 — 사건마다 팔이 얼마나 물러났는가.

    "힘이 안 올랐다" 는 두 가지를 못 가른다: 로봇이 흡수했는가, 애초에 오를 상황이
    아니었는가. 물러난 거리는 그 둘을 가른다. 팬텀 강성을 곱하면 **팔이 가만히
    있었다면 실렸을 힘** 이 나오고, 그것이 반사실의 값이다.

    ⚠️ 강성은 개루프 실행에서 잰 한 지점의 기울기다. 팬텀이 비선형이면 외삽한
    만큼 틀리며, 그 사실은 보고서가 함께 적는다.
    """
    out = []
    marks = [(ts, key) for ts, key in run.events if key in ("i", "w")]
    for index, (onset, key) in enumerate(marks):
        end = marks[index + 1][0] if index + 1 < len(marks) else run.t[-1]
        window = (run.t >= onset) & (run.t < end)
        if window.sum() < 10:
            continue
        # 기준점은 교란 **직전**이다. 시작 시점에서 재면 그 순간 이미 일어난
        # 이동을 놓쳐, 빠른 계단일수록 변위가 0 으로 나온다.
        baseline = (run.t >= onset - PRE_EVENT_S) & (run.t < onset)
        first = int(np.argmax(baseline)) if baseline.any() else int(np.argmax(window))
        last = int(len(window) - 1 - np.argmax(window[::-1]))
        travel = axial_travel(run, first, last)
        if travel is None:
            continue
        force = run.force[window]
        # 물러남은 음의 축방향 이동이다. 그만큼을 강성에 곱하면 흡수한 힘이 된다.
        absorbed = -travel * k_n_per_m if np.isfinite(k_n_per_m) else float("nan")
        out.append({
            "run": run.label,
            "target_n": run.target,
            "direction": "in" if key == "i" else "withdraw",
            "onset_s": float(onset),
            "travel_mm": travel * 1000.0,
            "held_mean_n": float(force.mean()),
            "held_max_n": float(force.max()),
            "absorbed_n": absorbed,
            "counterfactual_n": float(force.mean() + absorbed) if np.isfinite(absorbed) else None,
        })
    return out


def _above_runs(t: np.ndarray, f: np.ndarray, level: float, min_s: float = 0.05):
    """``level`` 을 넘은 연속 구간들. ``(시작, 지속, 그 구간 최대힘)`` 목록."""
    above = f > level
    out, i = [], 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(above) and above[j + 1]:
            j += 1
        span = float(t[j] - t[i])
        if span >= min_s:
            out.append((float(t[i]), span, float(f[i:j + 1].max())))
        i = j + 1
    return out


def excursions(run: Run) -> list:
    """유지 밴드를 벗어난 구간 하나하나와 **되돌아오기까지 걸린 시간**.

    한계에 닿았는지로 안전을 말할 수 없을 때 남는 질문은 노출 시간이다: 힘이
    의도한 대역을 벗어나 있던 동안이 얼마나 길었는가. 강제 후퇴가 한 번도 걸리지
    않은 실행에서도 이 값은 잴 수 있고, 조직이 실제로 겪는 것은 이쪽이다.

    구간은 밴드 **위**끝 기준이다. 아래로 벗어나는 것은 접촉이 옅어지는 것이라
    안전 질문이 아니다.
    """
    top = run.target + run.band
    mask = run.probing()
    if not mask.any():
        return []
    t, f = run.t[mask], run.force[mask]
    return [{
        "run": run.label,
        "loop": loop_state(run),
        "target_n": run.target,
        "band_top_n": top,
        "onset_s": onset,
        "duration_s": span,
        "peak_n": peak,
        "over_band_n": peak - top,
    } for onset, span, peak in _above_runs(t, f, top)]


def exposure(run: Run, levels=None) -> list:
    """문턱마다 **가장 긴 한 번의 초과 시간**과 총 초과 시간.

    합계가 아니라 최댓값을 앞에 두는 이유는, 안전에서 중요한 것이 "합쳐서 몇 초"
    가 아니라 "한 번에 얼마나 오래" 이기 때문이다. 짧은 스침 열 번과 5 초 연속은
    같은 합계를 낼 수 있지만 같은 위험이 아니다.
    """
    mask = run.probing()
    if not mask.any():
        return []
    t, f = run.t[mask], run.force[mask]
    top = run.target + run.band
    warn = float(run.meta.get("warn_contact_force_n",
                              run.meta.get("warn_force_n", float("nan"))))
    limit = float(run.meta.get("max_contact_force_n",
                               run.meta.get("max_force_n", float("nan"))))
    if levels is None:
        highest = max(float(f.max()), top)
        levels = sorted({round(v, 2) for v in
                         list(np.linspace(top, highest, 8)) + [warn, limit]
                         if math.isfinite(v)})
    out = []
    for level in levels:
        spans = _above_runs(t, f, level, min_s=0.0)
        out.append({
            "run": run.label,
            "target_n": run.target,
            "level_n": level,
            "episodes": len(spans),
            "longest_s": max((s for _, s, _ in spans), default=0.0),
            "total_s": float(sum(s for _, s, _ in spans)),
        })
    return out


def safety_margin(run: Run) -> dict:
    """한계 대비 최악값. **프로빙 구간만이 아니라 전 구간**을 본다.

    접근 중에 한계를 넘는 것도 넘는 것이다. 힘 루프가 도는 동안만 검사하면 그
    루프가 열리기 전에 일어난 일을 못 본다.
    """
    # 2026-08-31 에 safety.*_normal_force_n 이 *_contact_force_n 으로 바뀌었다.
    # 캡처는 새 이름으로 적고, 그 전에 찍힌 파일은 옛 이름을 들고 있다 — 둘 다 읽는다.
    warn = float(run.meta.get("warn_contact_force_n",
                              run.meta.get("warn_force_n", float("nan"))))
    limit = float(run.meta.get("max_contact_force_n",
                               run.meta.get("max_force_n", float("nan"))))
    peak = float(run.force.max())
    return {
        "run": run.label,
        "run_type": run.run_type,
        "target_n": run.target,
        "peak_force_n": peak,
        "warn_force_n": warn,
        "max_force_n": limit,
        "margin_to_limit_n": limit - peak,
        "reached_warn": bool(peak >= warn) if math.isfinite(warn) else None,
        "exceeded_limit": bool(peak > limit) if math.isfinite(limit) else None,
        "samples_over_limit": int((run.force > limit).sum()) if math.isfinite(limit) else 0,
        "seconds_over_warn": _seconds_where(run, run.force >= warn) if math.isfinite(warn) else 0.0,
    }


def _seconds_where(run: Run, mask: np.ndarray) -> float:
    if not mask.any() or run.t.size < 2:
        return 0.0
    dt = float(np.median(np.diff(run.t)))
    return float(mask.sum() * dt)


@dataclass
class Summary:
    runs: list = field(default_factory=list)
    hold: list = field(default_factory=list)
    events: list = field(default_factory=list)
    safety: list = field(default_factory=list)
    stiffness: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    rejection: list = field(default_factory=list)
    excursions: list = field(default_factory=list)
    exposure: list = field(default_factory=list)
    k_n_per_m: float = float("nan")


def summarise(runs: list) -> Summary:
    out = Summary(runs=runs)

    # 강성을 먼저 낸다 — 반사실이 그 값 위에 서기 때문이다.
    for run in runs:
        if run.run_type == "stiffness":
            out.stiffness.append(stiffness(run))
    usable = [s for s in out.stiffness if s.get("ok")]
    if usable:
        # 여러 개면 R² 로 가중하지 않고 **중앙값**을 쓴다. 적합이 잘 된 것이 반드시
        # 대표적인 것은 아니고, 팬텀은 누른 자리마다 다르다.
        out.k_n_per_m = float(np.median([s["k_n_per_m"] for s in usable]))

    for run in runs:
        out.safety.append(safety_margin(run))
        if not run.included:
            continue
        if run.run_type in ("hold", "disturbance"):
            metrics = hold_metrics(run)
            if metrics.get("samples"):
                out.hold.append({"run": run.label, "run_type": run.run_type,
                                 "target_n": run.target, "band_n": run.band, **metrics})
        if run.run_type == "disturbance":
            out.events.extend(disturbance_events(run))
            out.evidence.extend(regulation_evidence(run, out.k_n_per_m))
    # 되돌아옴은 제외된 실행에서도 잰다 — 힘 제어가 안 걸린 실행이야말로
    # 비교의 반쪽(개루프)이고, 그것을 버리면 비교 자체가 사라진다.
    for run in runs:
        if run.run_type == "disturbance":
            out.rejection.extend(rejection(run))
        if run.run_type in ("hold", "disturbance", "safety"):
            out.excursions.extend(excursions(run))
            out.exposure.extend(exposure(run))
    return out


def by_target(hold_rows: list) -> list:
    """목표별로 묶는다 — 반복 실행을 하나의 행으로.

    **정상 유지 실행만 쓴다.** 교란 실행의 이탈은 일부러 만든 것이므로, 같은 표에
    넣으면 "이 대역을 얼마나 잘 잡는가" 를 묻는 자리에서 "얼마나 세게 흔들었는가" 를
    답하게 된다. 교란 실행은 사건별로 따로 본다 (:func:`disturbance_events`).
    """
    rows_all = [r for r in hold_rows if r["run_type"] == "hold"]
    targets = sorted({row["target_n"] for row in rows_all})
    grouped = []
    for target in targets:
        rows = [r for r in rows_all if r["target_n"] == target]
        weights = np.array([r["samples"] for r in rows], float)

        def pick(key):
            return np.array([r[key] for r in rows], float)

        grouped.append({
            "target_n": target,
            "band_n": rows[0].get("band_n"),
            "settling_point_n": rows[0].get("settling_point_n"),
            "error_vs_settling_n": float(
                np.average(pick("error_vs_settling_n"), weights=weights)),
            "runs": len(rows),
            "samples": int(weights.sum()),
            "seconds": float(sum(r["seconds"] for r in rows)),
            "mean_error_n": float(np.average(pick("mean_error_n"), weights=weights)),
            "sd_n": float(np.average(pick("sd_n"), weights=weights)),
            "rmse_n": float(np.sqrt(np.average(pick("rmse_n") ** 2, weights=weights))),
            "in_band_pct": float(np.average(pick("in_band_pct"), weights=weights)),
            "p95_abs_error_n": float(pick("p95_abs_error_n").max()),
            "rmse_pct_of_target": float(np.average(pick("rmse_pct_of_target"), weights=weights)),
            "max_force_n": float(pick("max_force_n").max()),
        })
    return grouped
