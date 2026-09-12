"""에피소드 판정 — 시작 조건, 성공 판정, 위약 조건.

실험 설계 (2026-09-11)
---------------------
팬텀 물풍선의 부피를 바꾸는 동안 정책이 영상 품질을 유지하는가를 본다. 에피소드 하나는
시작 자세에서 90 s 이고, 조건은 넷이다.

    hold      정지. 아무것도 지령하지 않는다 — **바닥선**. 풍선이 변형되는 동안 가만히
              있으면 품질이 얼마나 떨어지는가.
    placebo   정책과 **같은 분포의 움직임**을 내되 지금 영상과 무관하게. 위약이 없으면
              "움직이니까 좋아졌다" 와 "옳게 움직여서 좋아졌다" 를 못 가른다.
    policy    학습된 정책.
    expert    숙련자 텔레오퍼레이션. 상한선 — 본 시험에서만.

**판정 임계는 파일럿이 정한다.** 여기 있는 기본값은 자리표시자다 (면적비 8 %, 연결성분
80 %). 예비 촬영으로 확정한 뒤 설정으로 넘긴다 — 코드를 고치지 않는다.

시작 조건
--------
"방광이 잘 안 보이는 자세" 에서 출발해야 찾는 능력을 잰다. 이미 잘 보이면 아무것도 안 해도
성공이다. 그래서 **면적비 < 2 % 이면서 Q_raw ≥ 0.6** 을 요구한다 — 앞은 "안 보인다",
뒤는 "그래도 영상은 쓸 만하다"(접촉이 있고 화면이 잡음이 아니다) 이다. 둘 중 하나라도
안 맞으면 그 자세는 **폐기**하고 다음으로 간다. 폐기율 자체가 파일럿의 산출물이다.

``rclpy`` 없이 돈다. 이 판정의 버그는 실기 앞이 아니라 pytest 로 찾아야 하는 종류다.
시간은 밖에서 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .perception import STATE_FEATURE_NAMES

#: 조건. expert 는 로봇이 아니라 사람이 지령하므로 러너는 기록만 한다.
#: search = 학습된 Q̂ 로 고르지 않고 **실제로 움직여 재 본 Q** 로 방향을 고른다
#: (rus_policy.search — Q̂ 의 방향 판별이 52~54 % 로 우연 수준이라 2026-09-12 에 더했다).
CONDITIONS = ("hold", "placebo", "policy", "expert", "search")

_IDX = {name: i for i, name in enumerate(STATE_FEATURE_NAMES)}
AREA = _IDX["area_ratio"]
CENTROID_DX = _IDX["centroid_dx"]      # 정규화 중심 − 0.5 → |dx| ≤ 0.30 이면 중앙 60 % 안
COMPONENT = _IDX["largest_component_ratio"]
QUALITY = _IDX["quality"]
HAS_MASK = _IDX["has_mask"]


@dataclass
class StartGate:
    """시작 조건을 연속으로 만족했는가.

    한 프레임짜리 만족으로 열지 않는다 — 분할이 한 장 튄 것과 "그 자세가 실제로 그렇다"
    를 가르려면 창이 필요하다.
    """

    area_max: float = 0.02          # 면적비 < 2 % — 방광이 잘 안 보인다 (Q_seg 쪽 재료)
    quality_min: float = 0.6        # **Q_raw** ≥ 0.6 — 접촉은 좋다 (힘 축 신호)
    confirm_s: float = 1.0
    _held_s: float = field(default=0.0, init=False)
    _last_t: Optional[float] = field(default=None, init=False)

    def reset(self) -> None:
        self._held_s = 0.0
        self._last_t = None

    def update(self, t: float, state: Sequence[float], q_raw: float) -> bool:
        """상태 표본 하나. 조건이 확인 창만큼 이어졌으면 참.

        Args:
            state: 지각 상태 벡터. 면적비를 여기서 읽는다.
            q_raw: **Q_raw** (접촉·에코 품질). 상태 벡터의 ``quality`` 는 Q_seg 이므로
                여기 쓰면 안 된다 — 계획서 §3 의 통제는 "접촉은 좋은데 방광이 없다" 이고,
                그것을 분할 점수로 재면 "분할이 뭔가 봤다" 를 요구하게 되어 뜻이 뒤집힌다.
                측정 불가(NaN)는 조건 불충족으로 센다.
        """
        dt = 0.0 if self._last_t is None else max(0.0, float(t) - self._last_t)
        self._last_t = float(t)
        s = np.asarray(state, float)
        # 마스크가 **없는 것도 시작 조건이다.** "방광이 안 보인다" 의 가장 극단이 면적비 0 이고,
        # has_mask 를 요구하면 그 자세가 영영 채택되지 않는다 (2026-09-11 실기에서 확인:
        # 면적비 0.000 · Q_raw 0.72 인 자세가 계속 폐기됐다). 계획서 §3 도 마스크 존재를
        # 요구하지 않는다 — 면적비 < 2 % 이면서 Q_raw ≥ 0.6 둘뿐이다.
        # 성공 판정(judge)은 반대다: 거기서는 마스크가 있어야 진단 가능 뷰다.
        ok = (s[AREA] < self.area_max
              and np.isfinite(q_raw) and float(q_raw) >= self.quality_min)
        self._held_s = self._held_s + dt if ok else 0.0
        return self.is_open

    @property
    def is_open(self) -> bool:
        return self._held_s >= self.confirm_s

    @property
    def held_s(self) -> float:
        return self._held_s


@dataclass
class Thresholds:
    """진단 가능 뷰 (EVAL_PLAN_POLICY_RESCUE §4). **수치는 예비 촬영으로 고정**한다 —
    결과를 보고 고치면 그 순간 사전 등록이 아니다."""

    area_min: float = 0.08          # 마스크 면적비 ≥ 8 %
    component_min: float = 0.80     # 가장 큰 연결성분 ≥ 80 % (조각난 오검출 배제)
    centroid_max: float = 0.30      # |중심 − 0.5| ≤ 0.30 → 중앙 60 % 폭 (가장자리 뷰 배제)
    hold_s: float = 3.0             # 셋을 **동시에** 연속 3 s
    # 힘 안전 (probe.yaml safety 와 같은 수). 유지 구간에서 이 선을 얼마나 넘었는지가
    # "그때의 힘은 위험하지 않은가" 의 답이다.
    warn_force_n: float = 4.5       # 상승 금지 선
    limit_force_n: float = 5.0      # 강제 후퇴 선


def longest_run_s(t: Sequence[float], ok: Sequence[bool]) -> float:
    """``ok`` 가 연속으로 참인 가장 긴 구간의 길이 [s]."""
    t = np.asarray(t, float)
    ok = np.asarray(ok, bool)
    if t.size < 2:
        return 0.0
    best = run = 0.0
    for i in range(1, t.size):
        if ok[i] and ok[i - 1]:
            run += max(0.0, t[i] - t[i - 1])
            best = max(best, run)
        else:
            run = 0.0
    return float(best)


def _first_sustained(t: np.ndarray, ok: np.ndarray, hold_s: float) -> Optional[int]:
    """``ok`` 가 ``hold_s`` 동안 끊기지 않고 이어진 **첫 지점**의 인덱스 (그 구간의 끝)."""
    start = None
    for i, good in enumerate(ok):
        if not good:
            start = None
            continue
        if start is None:
            start = i
        if t[i] - t[start] >= hold_s:
            return i
    return None


def judge_two_phase(t: Sequence[float], state: np.ndarray, thr: Thresholds,
                    force_t: Optional[Sequence[float]] = None,
                    force_n: Optional[Sequence[float]] = None) -> dict:
    """찾기 → 유지, 두 단계로 잰다.

    평가하려는 것이 두 가지이기 때문이다 (2026-09-12 조작자 정의): **방광이 없는 자리에서
    시작해 찾아내는가**, 그리고 **찾은 뒤 주사기로 용적을 바꿔도 뷰를 유지하는가 · 그때 힘이
    위험하지 않은가.** 하나의 성공/실패로는 둘을 가릴 수 없다 — 찾자마자 놓쳐도 "성공" 이
    되고, 못 찾으면 유지 능력은 재지도 못한다.

    단계를 가르는 지점은 **진단 가능 뷰가 처음으로 hold_s 만큼 이어진 순간**이다. 그 뒤가
    유지 구간이고, 주사기 조작은 그 안에서 일어난다.
    """
    t = np.asarray(t, float)
    state = np.asarray(state, float)
    good = ((state[:, HAS_MASK] > 0.5)
            & (state[:, AREA] >= thr.area_min)
            & (state[:, COMPONENT] >= thr.component_min)
            & (np.abs(state[:, CENTROID_DX]) <= thr.centroid_max))
    from .view_quality import view_quality
    qv = np.array([view_quality(row)[0] for row in state]) if state.size else np.array([])

    idx = _first_sustained(t, good, thr.hold_s)
    out = {
        "found": idx is not None,
        "find_time_s": float(t[idx] - t[0]) if idx is not None else float("nan"),
        "hold_window_s": float(t[-1] - t[idx]) if idx is not None and len(t) else 0.0,
    }
    if idx is not None and idx < len(t) - 1:
        h = slice(idx, None)
        out["hold_good_fraction"] = float(good[h].mean())
        # 가장 오래 놓친 구간. "얼마나 잘 유지하는가" 는 평균보다 이쪽이 말해 준다 —
        # 90 % 를 유지해도 한 번에 8 s 를 놓치면 그때 화면은 쓸 수 없다.
        out["hold_worst_loss_s"] = longest_run_s(t[h], ~good[h])
        out["hold_q_mean"] = float(np.mean(qv[h])) if qv.size else float("nan")
        out["hold_q_min"] = float(np.min(qv[h])) if qv.size else float("nan")
    else:
        out.update({"hold_good_fraction": float("nan"), "hold_worst_loss_s": float("nan"),
                    "hold_q_mean": float("nan"), "hold_q_min": float("nan")})

    if force_t is not None and force_n is not None and len(force_t) > 1:
        ft, fn = np.asarray(force_t, float), np.asarray(force_n, float)
        keep = np.isfinite(fn)
        ft, fn = ft[keep], fn[keep]
        if ft.size > 1:
            dt = np.diff(ft, prepend=ft[0])
            in_hold = ft >= t[idx] if idx is not None else np.ones(len(ft), bool)
            out.update({
                "force_max_n": float(fn.max()),
                "force_mean_n": float(fn.mean()),
                "time_above_warn_s": float(dt[(fn >= thr.warn_force_n)].sum()),
                "time_above_limit_s": float(dt[(fn >= thr.limit_force_n)].sum()),
                "hold_force_max_n": float(fn[in_hold].max()) if in_hold.any() else float("nan"),
            })
    return out


def judge(t: Sequence[float], state: np.ndarray, thr: Thresholds) -> dict:
    """에피소드 하나의 결과.

    성공은 "언젠가 한 프레임 좋았다" 가 아니라 **연속으로 ``hold_s`` 동안 좋았다** 이다.
    한 프레임짜리 성공을 세면 무작위로 흔들기만 해도 성공률이 올라간다 — 위약이 이기게 된다.
    """
    state = np.asarray(state, float)
    if state.ndim != 2 or state.shape[0] != len(t):
        raise ValueError(f"state 는 (N, {len(STATE_FEATURE_NAMES)}) 여야 한다: {state.shape}")
    good = ((state[:, HAS_MASK] > 0.5)
            & (state[:, AREA] >= thr.area_min)
            & (state[:, COMPONENT] >= thr.component_min)
            & (np.abs(state[:, CENTROID_DX]) <= thr.centroid_max))
    best = longest_run_s(t, good)
    q = state[:, QUALITY]
    finite = np.isfinite(q)
    # 방광 뷰 품질(면적·중심 63 %) — 화면의 Q_seg 와 같은 정의. q_mean 은 옛 정의 그대로
    # 두어 이전 에피소드와 비교가 된다.
    from .view_quality import view_quality
    qv = np.array([view_quality(row)[0] for row in state]) if state.size else np.array([])
    return {
        "success": bool(best >= thr.hold_s),
        "best_run_s": best,
        "good_fraction": float(good.mean()) if good.size else 0.0,
        "q_mean": float(q[finite].mean()) if finite.any() else float("nan"),
        "q_final_10s": _tail_mean(t, q, 10.0),
        "q_view_mean": float(np.mean(qv)) if qv.size else float("nan"),
        "q_view_final_10s": _tail_mean(t, qv, 10.0) if qv.size else float("nan"),
        "area_max": float(state[:, AREA].max()) if state.size else 0.0,
        "n_samples": int(state.shape[0]),
    }


def _tail_mean(t: Sequence[float], v: Sequence[float], window_s: float) -> float:
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    if t.size == 0:
        return float("nan")
    m = (t >= t[-1] - window_s) & np.isfinite(v)
    return float(v[m].mean()) if m.any() else float("nan")


def randomize_direction(action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """회전 성분의 **크기는 그대로, 방향만 무의미하게** (EVAL_PLAN §5).

    축을 섞고 부호를 무작위로 뒤집는다 — 크기의 다중집합이 정확히 보존되므로 "같은 속도·
    같은 지속시간" 이 성립하고, 정책이 고른 것은 방향뿐이라는 것이 통제된다. 병진은 건드리지
    않는다 (정책이 애초에 지령하지 않는다, 런북 §4).

    스케일을 다시 뽑거나 정규분포에서 새로 그리지 않는다. 그러면 위약의 **속도 분포가 달라져**
    "움직임의 양" 이 조건 사이에서 어긋난다 — 그것이 이 대조군이 통제하려던 바로 그 변수다.
    """
    out = np.array(action, dtype=float, copy=True)
    rot = out[3:6]
    out[3:6] = rng.permutation(rot) * rng.choice([-1.0, 1.0], size=rot.shape)
    return out


class PlaceboBuffer:
    """위약 조건 — 정책에게 **지금이 아닌 관측**을 먹인다.

    같은 정책, 같은 후보 분포, 같은 지령 크기. 다른 것은 그 지령이 지금 화면과 무관하다는
    것뿐이다. 그래서 "움직여서 좋아졌다" 와 "옳게 움직여서 좋아졌다" 가 갈린다.

    지연을 쓰는 이유는 재생(replay)과 달리 **부트스트랩이 필요 없기** 때문이다. 조건 순서를
    무작위로 돌릴 수 있고, 앞선 policy 에피소드가 없어도 첫 에피소드부터 돈다.

    ⚠️ 지연이 짧으면 위약이 아니다. 풍선 변형의 시간 규모보다 길어야 한다 — 기본 30 s 는
    자리표시자이고, 파일럿에서 Q_raw 자기상관을 보고 정한다.
    """

    def __init__(self, delay_s: float = 30.0) -> None:
        if delay_s <= 0.0:
            raise ValueError(f"delay_s 는 양수여야 한다: {delay_s}")
        self.delay_s = float(delay_s)
        self._items: list = []

    def push(self, t: float, item) -> None:
        self._items.append((float(t), item))

    def take(self, t: float):
        """``t − delay`` 이전의 가장 최근 항목. 아직 그만큼 안 쌓였으면 ``None``.

        ``t`` 는 **단조** 여야 한다 — 쓰이지 않을 과거를 버리므로 뒤로 가면 이미 없다.
        실시간 루프가 부르는 방식이 그렇고, 되감아 다시 판정할 일이 있으면 새로 만든다.
        """
        cutoff = float(t) - self.delay_s
        chosen = None
        drop = 0
        for i, (ts, item) in enumerate(self._items):
            if ts <= cutoff:
                chosen, drop = item, i
            else:
                break
        if drop > 0:                      # 쓰이지 않을 과거는 버린다 (90 s 에피소드 × 8 fps)
            del self._items[:drop]
        return chosen

    @property
    def n_buffered(self) -> int:
        return len(self._items)


# ---- 실현 확인 ---------------------------------------------------------------

def _quat_angle_deg(q1: Sequence[float], q2: Sequence[float]) -> float:
    """두 단위 사원수 사이의 회전각 [도]. 부호가 반대인 같은 자세(q ≡ −q)도 0 이다."""
    a = np.asarray(q1, float)
    b = np.asarray(q2, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    d = abs(float(np.dot(a / na, b / nb)))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))


def motion_check(t: Sequence[float], cmd_w_deg_s: np.ndarray, quat: np.ndarray,
                 *, min_commanded_deg: float = 1.0) -> dict:
    """지령한 회전이 **실제로** 일어났는가.

    지령만 기록하면 "로봇이 정책대로 움직였는가" 에 답할 수 없다. 2026-09-11 까지 세 번
    "지령은 나가는데 로봇이 안 움직인다" 가 있었고 (축 게이트가 회전을 0 으로 만듦 ·
    시작 게이트가 전량 차단 · 5 Hz 발행이 워치독에 매 주기 절반씩 버려짐), 셋 다 코드를
    읽어서야 찾았다. 기록이 있었으면 첫 에피소드에서 드러났다.

    **경로 길이**로 비교한다. 방향과 무관하므로 위약(크기는 같고 방향만 무작위)에서도
    그대로 성립한다. 알짜 회전(처음↔끝)은 왕복하면 0 이 되어 판정에 못 쓴다.

    Args:
        t: 각 결정의 시각 [s].
        cmd_w_deg_s: (N, 3) 로봇에 **실제로 보낸** 각속도 [°/s]. 조건을 지난 뒤의 값.
        quat: (N, 4) 그때의 프로브 자세 (w, x, y, z). 모르면 NaN.
        min_commanded_deg: 지령 경로가 이보다 짧으면 "지령 없음" 으로 본다.

    Returns:
        commanded_deg · realized_deg · realized_net_deg · ratio · verdict.
    """
    t = np.asarray(t, float)
    w = np.asarray(cmd_w_deg_s, float).reshape(-1, 3)
    q = np.asarray(quat, float).reshape(-1, 4)
    out = {"commanded_deg": float("nan"), "realized_deg": float("nan"),
           "realized_net_deg": float("nan"), "ratio": float("nan"), "verdict": ""}
    if len(t) < 2:
        out["verdict"] = "표본 부족"
        return out

    dt = np.diff(t)
    speed = np.linalg.norm(np.nan_to_num(w[:-1]), axis=1)
    commanded = float(np.sum(speed * dt))
    out["commanded_deg"] = commanded

    ok = np.all(np.isfinite(q), axis=1)
    if ok.sum() < 2:
        out["verdict"] = "자세 없음 — ee_wrt_base 가 안 왔다 (제어 스택이 떠 있는가)"
        return out
    qv = q[ok]
    steps = [_quat_angle_deg(qv[i], qv[i + 1]) for i in range(len(qv) - 1)]
    realized = float(np.nansum(steps))
    out["realized_deg"] = realized
    out["realized_net_deg"] = _quat_angle_deg(qv[0], qv[-1])

    if commanded < min_commanded_deg:
        out["verdict"] = (f"지령 없음 (경로 {commanded:.1f}°) — hold·expert 이거나 정책이 "
                          f"거의 0 을 냈다. 그 사이 실제 회전 {realized:.1f}°")
        return out
    ratio = realized / commanded
    out["ratio"] = ratio
    if ratio >= 0.5:
        out["verdict"] = f"지령대로 움직였다 — 실현 {ratio:.0%}"
    elif ratio >= 0.2:
        out["verdict"] = (f"⚠️ 일부만 실현됐다 ({ratio:.0%}) — 상한·가속 제한·감쇠를 의심하라")
    else:
        out["verdict"] = (f"⚠️ 지령했는데 거의 안 움직였다 ({ratio:.0%}) — 지령이 사슬 어딘가에서 "
                          "끊겼다: probing_mode 가 contact_probing_policy 인가, desired_twist 가 "
                          "50 Hz 로 오는가")
    return out
