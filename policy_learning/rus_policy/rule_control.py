"""규칙 기반 제어기 — 학습된 정책 없이 U-Net 마스크만 보고 돈다.

    c = RuleController(gains)
    vel6, note = c.update(t, state)        # [vx,vy,vz mm/s, wx,wy,wz °/s]

왜 규칙인가
-----------
로봇이 해야 하는 일은 "방광을 찾아 화면 가운데에 두고 유지" 다. 2026-09-12 실측이 말하는 것:

* **면내 방향은 계산된다.** x 병진이 방광 중심을 0.00708 /mm 로 옮긴다 (r=+1.00). 영상에서
  중심이 어디로 치우쳤는지 보이므로 어느 쪽으로 갈지 **추론할 필요가 없다.**
* **면외 방향은 원리적으로 안 보인다.** 2D 한 장은 ±면외 기울임에 대칭이다. 프리핸드
  데이터에서 면외 방향과 영상의 상관은 −0.026 — 없다. 재는 수밖에 없다.
* **학습된 정책은 이 일을 못 했다.** eval_v1 17 에피소드에서 policy 의 Q 변화는 −0.285 로,
  아무것도 안 하는 hold(+0.017)보다 나빴다. 관측→행동 정보가 사전분포에서 사라져서다.

그래서 **보이는 것은 계산하고(면내), 안 보이는 것은 측정한다(면외).** 학습되는 모델은
U-Net 하나뿐이고, 축과 계수는 로봇이 자기 자신을 60 s 재서 얻는다 (rus_policy.axis_probe).

두 국면
-------
``찾기``  면외 축(보통 y 병진)을 **끝에서 끝까지 훑고** 면적이 가장 컸던 자리로 돌아간다.
        오르막 타기가 아니다 — 방광이 안 보이면 면적이 0 이라 경사가 없고, 오르막은
        한 걸음 앞이 0 인지 아닌지만 보므로 제자리를 왕복한다 (2026-09-14 실기).
``유지``  면내 축(실측이 고른 축, 보통 thy)으로 중심을 0 으로 되돌리는 비례 제어.

        **중심을 맞추다 시야를 잃으면 안 된다.** 면내 축이라 해도 면적을 함께 바꾼다
        (2026-09-14 실측: thy 의 면적 이득 +0.00206 /° 로 면외축 thx 의 −0.00189 /° 와
        맞먹는다). 그날 실기에서 중심 보정이 9° 돌아가며 면적을 0 으로 만들었고, 찾기
        축(thx·thz)만으로는 되돌아올 수 없어 나머지 90 s 를 헤맸다.

        그래서 유지 중에 **면적이 가장 좋았던 자리와 그때까지의 누적 변위를 기억**하고,
        면적이 그 절반 아래로 떨어지면 **되돌아간 뒤** 찾기로 넘어간다. 누적 변위 자체도
        ``align_max`` 로 묶는다 — 한 번의 보정이 시야를 벗어날 만큼 크면 안 된다.

힘 축(빔 방향)은 건드리지 않는다 — admittance 와 force_search 의 몫이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .axis_gains import EXCLUDED_AXES, AxisGains
from .normal_align import NormalAlign
from .view_quality import WEIGHTS as Q_WEIGHTS

#: vel6 안에서의 자리 (axis_probe 와 같은 규약).
AXIS_INDEX = {"x": 0, "y": 1, "thx": 3, "thy": 4, "thz": 5}


def is_rotation(axis: str) -> bool:
    """회전 축인가. 단위(°/s 대 mm/s)와 상한이 갈리므로 섞으면 조용히 틀린다."""
    return AXIS_INDEX.get(axis, 0) >= 3


def unit(axis: str) -> str:
    return "°" if is_rotation(axis) else "mm"

FIND, HOLD, HOME = "찾기", "유지", "복귀"

#: Q_seg 의 각 항을 **어떤 움직임이 고칠 수 있는가.**
#:
#: 점수가 낮다는 것만으로는 무엇을 해야 할지 모른다. 중심이 모자라서 낮은 것과 면적이
#: 모자라서 낮은 것은 정반대 동작을 요구한다 — 전자는 정렬 축을 돌려야 하고, 후자는
#: 면외 축을 다시 훑어야 한다. 가중 손실이 가장 큰 항을 보고 고른다.
ALIGN, SWEEP = "정렬", "훑기"
Q_REMEDY = {
    "centering": ALIGN,     # 좌우로 치우쳤다 — 정렬 축
    "border": ALIGN,        # 화면 가장자리에 물렸다 — 같은 축으로 안쪽으로
    "presence": SWEEP,      # 안 보인다 — 면외로 다시 훑는다
    "area": SWEEP,          # 작게 보인다 — 같은 이유
}
#: 제어기가 어쩔 수 없는 항. 무엇 때문인지는 말해 준다.
Q_NOT_OURS = {
    "contrast": "접촉·이득 (force_search 의 몫)",
    "confidence": "분할 신뢰도 (U-Net)",
    "component": "마스크가 조각났다 (U-Net)",
}


# --- 훑기 -------------------------------------------------------------------

GO_START, SCAN, RETURN, DONE = "끝으로", "훑기", "복귀", "끝"


@dataclass
class SweepScan:
    """한 축을 **끝에서 끝까지 훑고** 값이 가장 컸던 자리로 되돌아간다.

    오르막 타기가 아니다. 방광이 안 보이면 면적이 0 이고, 한 걸음 움직여도 여전히 0 이라
    "나아짐" 이 없다 — 오르막은 경사가 있을 때만 돈다. 2026-09-14 실기에서 ±4 mm 를
    영원히 왕복하며 못 찾은 것이 그 때문이다. 훑기는 경사가 필요 없다.

    자리를 아는 방법: 지령 속도를 시간으로 적분한다. 로봇이 지령을 따른다는 가정이고,
    접촉 상한 안에서 도는 저속 병진에서는 성립한다.
    """

    half_range: float = 25.0      # 중심에서 ± 이만큼 훑는다 [mm 또는 °]
    rate: float = 3.0             # 훑는 속도 [단위/s]
    settle_s: float = 0.4         # 끝에서 영상이 따라올 때까지
    min_value: float = 0.0        # 이보다 커야 "봤다" 로 친다
    #: 훑는 도중 값이 이만큼 되면 **거기서 멈춘다.** 목적은 최댓값을 찾는 것이 아니라
    #: 쓸 만한 뷰에 닿는 것이다. 끝까지 훑고 돌아오면 그 사이에 팬텀도 프로브도
    #: 움직인다 (2026-09-14 attempt_008: 108 s 를 훑고서야 유지로 갔다).
    #: inf 면 예전처럼 끝까지 훑는다.
    good_enough: float = float("inf")

    phase: str = field(default=GO_START, init=False)
    pos: float = field(default=0.0, init=False)        # 시작 자세 기준 변위
    best_pos: float = field(default=0.0, init=False)
    best_value: float = field(default=float("-inf"), init=False)
    _t0: Optional[float] = field(default=None, init=False)
    _prev_t: Optional[float] = field(default=None, init=False)

    def reset(self, t: float) -> None:
        self.phase, self._t0, self._prev_t = GO_START, t, t
        self.pos, self.best_pos, self.best_value = 0.0, 0.0, float("-inf")

    @property
    def done(self) -> bool:
        return self.phase == DONE

    @property
    def found(self) -> bool:
        return self.best_value > self.min_value

    def update(self, t: float, value: float) -> float:
        """지금 낼 속도. ``value`` 는 최대로 만들고 싶은 양 (면적)."""
        if self._t0 is None:
            self.reset(t)
        dt = max(0.0, t - (self._prev_t if self._prev_t is not None else t))
        self._prev_t = t

        if self.phase == GO_START:
            self.pos -= self.rate * dt
            if self.pos <= -self.half_range:
                self.pos = -self.half_range
                self.phase, self._t0 = SCAN, t
            return -self.rate

        if self.phase == SCAN:
            self.pos += self.rate * dt
            if value == value and value > self.best_value:
                self.best_value, self.best_pos = float(value), self.pos
            # 충분하다 — 지금 자리가 답이다. 되돌아갈 것도 없다.
            if value == value and value >= self.good_enough:
                self.best_value, self.best_pos = float(value), self.pos
                self.phase = DONE
                return 0.0
            if self.pos >= self.half_range:
                self.pos = self.half_range
                self.phase, self._t0 = RETURN, t
                return 0.0
            return +self.rate

        if self.phase == RETURN:
            # **못 봤으면 최댓값 자리가 아니라 출발점으로 돌아간다.** 아무것도 못 본
            # 훑기의 "최댓값" 은 첫 표본(=-half_range)이라, 거기 선 채 다음 바퀴를 시작하면
            # 그 자리가 새 원점이 된다 — 바퀴마다 한쪽으로 half_range 씩 밀린다.
            # 2026-09-14 실기에서 프로브 시작 자세가 실행을 거듭할수록 첫 실행에서
            # 8 → 29 → 33 → 82 mm 로 멀어진 것이 이 표류다.
            target = self.best_pos if self.found else 0.0
            gap = target - self.pos
            if abs(gap) <= self.rate * max(dt, 1e-3):
                self.pos = target
                self.phase = DONE
                return 0.0
            v = self.rate if gap > 0 else -self.rate
            self.pos += v * dt
            return v

        return 0.0

    def describe(self) -> str:
        best = f"{self.best_value:.4f}" if self.best_value > float("-inf") else "—"
        return (f"{self.phase} 위치 {self.pos:+.1f} · 최대 {best} @ {self.best_pos:+.1f} "
                f"(±{self.half_range:.0f})")



@dataclass
class RuleController:
    """찾기 → 유지 두 국면. rclpy 없이 돌아 시험할 수 있다.

    Args:
        gains: ``axis_probe`` 로 잰 기하 이득. 축과 계수가 여기서 나온다.
        area_found: 이 면적비를 넘으면 찾았다고 본다.
        area_lost: 이 아래로 떨어지면 놓쳤다고 보고 다시 찾는다.
        deadband: 중심이 이 안이면 가만히 둔다. 1 s 잡음 표준편차 0.014 의 두 배.
        k_p: 비례 이득 [1/s]. 필요한 변위의 몇 분의 몇을 매 초 낸다.
        max_mm_s: 병진 속도 상한 [mm/s].
        max_deg_s: 회전 속도 상한 [°/s].
        find_half_range: 찾기에서 중심 기준 ± 이만큼 훑는다 [mm].
        find_half_range_deg: 회전 축으로 훑을 때의 반경 [°].
        find_rate: 훑는 속도 [mm/s].
        find_axes: 훑을 축 목록. 비우면 이득이 고른 축 하나. 여러 개면 차례로 훑고
            그중 면적이 가장 컸던 축·자리로 간다 — 병진과 회전을 함께 쓰려면 이것을 준다.
        align_max_mm / align_max_deg: 유지 중 정렬 축 누적 변위의 상한. 축의 단위로 고른다.
        keep_fraction: 면적이 유지 중 최고의 이 비율 아래로 떨어지면 되돌아간다.
    """

    gains: AxisGains
    area_found: float = 0.015
    area_lost: float = 0.005
    deadband: float = 0.03
    k_p: float = 0.4
    max_mm_s: float = 3.0
    max_deg_s: float = 3.0
    find_half_range: float = 25.0     # 훑는 반경 [mm]
    find_half_range_deg: float = 12.0  # 회전 축으로 훑을 때 [°]
    find_rate: float = 3.0            # 훑는 속도 [mm/s]
    find_axes: Sequence[str] = ()     # 비우면 이득이 고른 축 하나
    #: **면외 축은 이 비율만큼만 훑는다.** 면외 기울임은 조금만 가도 절단면이 크게 바뀌고
    #: (thx 면적 이득 −0.00189/°), 멀리 가면 방광을 지나쳐 버린다. 정렬 축(면내)만 넓게
    #: 훑는다 — 그쪽은 같은 절단면 안에서 좌우로 움직이는 것이라 안전하다.
    find_oop_scale: float = 0.5
    #: 시작 자세에서 회전 축이 이만큼 벗어나면 **되돌아간다.** 러너의 상한(30°)보다 낮게
    #: 잡는다 — 러너가 먼저 닿으면 지령이 통째로 끊기고, 지령이 없으면 되돌아올 수도 없다.
    #: 2026-09-14 rule_v3 attempt_003 이 그 교착으로 112 s 에 멈춰 남은 68 s 를 버렸다.
    max_excursion_deg: float = 24.0
    #: 복귀를 끝냈다고 보는 여유 [° 또는 mm].
    home_tol: float = 1.0
    # 유지에서 목표로 하는 Q_seg — 조작자가 정한 합격선. Thresholds.q_view_min 과 같은 값.
    #
    # 🔁 2026-09-14 저녁: 0.97 → 0.936 (통과율 보존) → 밤: **0.977** (조작자가 의도
    # 보존을 골랐다). 근거는 Thresholds.q_view_min 주석에 있다. 두 값은 늘 같아야 한다.
    #
    # ⚠️ **이 값은 합격 선언과 표시에만 쓴다.** 예전에는 이 하나가 세 가지 일을 했다 —
    # 합격 선언 · 훑기 정지선 · 법선 정렬의 여유 판정. 뜻이 다른 셋을 한 숫자에 묶어
    # 놓으면 눈금이 바뀔 때마다 셋이 함께 틀어진다. 실제로 0.977 로 올리는 순간 최근
    # 실측 최대(0.951~0.997)에 걸려 **훑기가 영영 안 멈추고 법선 정렬이 영영 안 열리는**
    # 상태가 됐다. 아래 q_good_enough 와 _axes_idle 로 갈랐다.
    q_target: float = 0.977
    #: 훑기를 도중에 멈춰도 되는 값 — "최댓값" 이 아니라 "쓸 만한 뷰".
    #:
    #: 합격선과 **일부러 다르다.** 끝까지 훑고 최고점으로 되돌아가는 사이에 그 최고점은
    #: 이미 지난 자리가 된다 (``_good_enough`` 독스트링). 그래서 이 값은 합격선이 아니라
    #: "여기서 멈춰도 손해가 아닌 자리" 여야 하고, 무엇보다 **닿을 수 있어야** 한다.
    #:
    #: 2026-09-14 밤 실측 (채운 팬텀 세션, 검출 17,202 프레임): q_view ≥ 0.75 부터 판정
    #: 세 조건(면적·연결성분·중심)이 100 % 통과한다. 0.85 는 그 위의 p75 자리이고 전체의
    #: 25 % 가 닿는다 — 쓸 만하다고 부르기에 충분하고 멈추기에 흔하다.
    q_good_enough: float = 0.85
    #: 합격한 여유에서 결합(Q_raw)을 개선할 것인가. 끄면 예전처럼 가만히 있는다.
    normal_align: bool = True
    # Q_seg 로 돌 때의 찾음/놓침 문턱 — 면적 문턱(0.015 / 0.005)을 Q_seg 눈금으로 옮긴 값.
    #
    # 🔁 2026-09-14 저녁: 0.50/0.30 → 0.43/0.13. 옛 값은 **히스테리시스가 이름뿐이었다.**
    # 옛 눈금에서 0.30 이상의 통과율이 65.90 % 로 검출률과 정확히 같았고 (검출된 프레임은
    # 0.377 아래로 내려간 적이 없다), 0.50 도 65.24 % 로 사실상 같았다 — 두 선 사이에
    # 프레임이 없으니 "놓쳤다" 판정이 has_mask 와 구별되지 않았다.
    #
    # 새 값은 분위수 대응이 아니라 **원래 의도한 면적 자리**에서 다시 읽었다. 기록에서
    # 면적 [0.013, 0.017) 프레임의 Q_seg 중앙값이 0.428, [0.003, 0.007) 이 0.124 다.
    q_found: float = 0.43
    q_lost: float = 0.13
    #: 면적이 모자라 Q 가 낮을 때 다시 훑는 범위 — 설정 반경의 이 비율. 좁게 훑는 이유는
    #: 이미 방광을 보고 있기 때문이다. 넓게 훑으면 있는 것을 놓친다.
    refine_fraction: float = 0.35
    #: 다시 훑고 이 시간 안에는 또 훑지 않는다. 없으면 합격선 바로 아래에서 찾기와 유지를
    #: 오간다 (한 번 훑어 나아지지 않는 상황이 있다).
    refine_cooldown_s: float = 5.0
    find_widen: float = 1.6           # 한 바퀴 다 훑고 못 찾으면 반경을 이만큼 곱한다
    find_max_range_deg: float = 30.0  # 넓혀도 여기까지 [°]
    find_max_range_mm: float = 60.0   # 넓혀도 여기까지 [mm]
    align_max_mm: float = 20.0        # 유지 중 정렬 축 누적 변위 상한 [mm]
    align_max_deg: float = 8.0        # 같은 것, 회전 축일 때 [°]
    keep_fraction: float = 0.5        # 최고 면적의 이 비율 아래면 되돌아간다

    phase: str = field(default=FIND, init=False)
    _scan: Optional[SweepScan] = field(default=None, init=False)
    last_note: str = field(default="", init=False)
    #: 유지 중 정렬 축 누적 변위와, 면적이 가장 좋았던 자리
    align_pos: float = field(default=0.0, init=False)
    _hold_best: float = field(default=float("-inf"), init=False)
    _hold_best_pos: float = field(default=0.0, init=False)
    _prev_t: Optional[float] = field(default=None, init=False)
    _dt: float = field(default=0.0, init=False)      # 이번 tick 의 간격
    #: 마지막으로 좁게 다시 훑은 시각. 냉각 시간이 없으면 합격선 바로 아래에서
    #: 찾기와 유지를 오간다.
    _last_refine_t: float = field(default=float("-inf"), init=False)
    #: 러너가 Q_seg 를 주고 있는가. 주면 훑기의 조기 정지선이 q_target 이 된다.
    _use_q: bool = field(default=False, init=False)
    #: 시작 자세 기준 축별 누적 변위. 지령을 적분해 따라간다 — 훑기의 pos 는 바퀴마다
    #: 0 으로 돌아가므로 "지금까지 얼마나 갔나" 를 모른다.
    axis_pos: dict = field(default_factory=dict, init=False)
    #: 합격선 위에서만 도는 법선 정렬 (Q_raw 는 힘만으로는 더 안 오른다).
    na: NormalAlign = field(default_factory=NormalAlign, init=False)
    #: 지금 훑고 있는 반경. 못 찾을 때마다 넓어지고, 찾으면 처음 값으로 돌아간다.
    find_range_now: float = field(default=0.0, init=False)
    find_range_now_deg: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.align_axis = self.gains.centering_axis()
        self.find_axis = self.gains.search_axis()
        if self.align_axis is None:
            raise ValueError("정렬 축이 없다 — axis_probe 로 이득을 먼저 재라 "
                             "(중심 이득이 문턱 아래이거나 상관이 낮다)")
        # 훑을 축 목록. 설정이 있으면 그대로, 없으면 이득이 고른 축 하나.
        axes = [a for a in (self.find_axes or ([self.find_axis] if self.find_axis else []))
                if a in AXIS_INDEX]
        bad = [a for a in (self.find_axes or ()) if a not in AXIS_INDEX]
        if bad:
            raise ValueError(f"모르는 찾기 축: {bad} (가능: {sorted(AXIS_INDEX)})")
        # 빔축 회전은 손으로 줘도 막는다. 자동 선택만 막으면 명령줄로 되살아난다 —
        # 실제로 2026-09-14 의 세 실행이 `--rule-find-axes thx,thz` 로 돌았다.
        banned = [a for a in axes if a in EXCLUDED_AXES]
        if banned:
            raise ValueError(
                f"찾기에 쓸 수 없는 축: {banned}. thz 는 빔축 둘레 회전이라 프로브를 옮기는 "
                "것이 아니라 영상면을 자기 축으로 돌린다 — 장축/단축이 바뀌어 횡단면이 "
                "사면이 되고, 그때의 면적·중심은 다른 절단면의 값이다 "
                f"(axis_gains.EXCLUDED_AXES). 남는 축으로 주십시오: "
                f"{sorted(set(AXIS_INDEX) - set(EXCLUDED_AXES))}")
        self.find_axis_list: list[str] = axes
        self._reset_range()
        self._scans = [(a, self._make_scan(a, self._good_enough())) for a in axes]
        self._scan_i = 0
        self._scan = self._scans[0][1] if self._scans else None
        self._find_index = AXIS_INDEX[axes[0]] if axes else 0
        self._best: tuple[float, int, float] = (float("-inf"), 0, 0.0)  # (값, 축, 위치)

    #: 이보다 작은 가중 손실은 잡음으로 본다 (Q 전체의 1 %).
    LOSS_FLOOR = 0.01
    #: 넓히기가 쓸 수 있는 회전 예산의 비율. 나머지는 정렬 보정과 여유로 남긴다.
    HOME_MARGIN = 0.8

    @staticmethod
    def q_losses(parts: Optional[dict]) -> dict:
        """항별 **가중 손실** — 이 항 때문에 Q 가 얼마나 깎였는가.

        하나로 합쳐진 점수지만 제어에는 **항마다 다른 함수**로 들어간다. 중심이 깎은
        0.05 와 면적이 깎은 0.05 는 같은 크기의 손실이되 **서로 다른 축**이 고친다.
        합쳐 놓은 총점만 보면 어느 쪽으로 움직여야 할지 알 수 없다.
        """
        if not parts:
            return {}
        tot = sum(Q_WEIGHTS.values())
        return {k: Q_WEIGHTS[k] * (1.0 - float(v)) / tot
                for k, v in parts.items() if k in Q_WEIGHTS}

    def _demand(self, parts: Optional[dict]) -> dict:
        """움직임별 요구량 — 같은 축이 고치는 항들의 손실을 모은다."""
        out = {ALIGN: 0.0, SWEEP: 0.0}
        for k, L in self.q_losses(parts).items():
            m = Q_REMEDY.get(k)
            if m:
                out[m] += L
        return out

    def _on_q(self, q_view: Optional[float]) -> bool:
        return q_view is not None and q_view == q_view

    @staticmethod
    def _vname(q_view: Optional[float]) -> str:
        return "Q_seg" if (q_view is not None and q_view == q_view) else "면적"

    def _mark(self, q_view: Optional[float]) -> str:
        """Q_seg 와 합격선을 앞에 붙인다. 목표에 닿았는지가 한눈에 보여야 한다."""
        if not self._on_q(q_view):
            return ""
        ok = "○" if q_view >= self.q_target else "·"
        return f"Q_seg {q_view:.3f} {ok}{self.q_target:.2f} · "

    def _blocked_by(self, parts: Optional[dict]) -> str:
        """제어기가 못 고치는 항이 남아 있으면 무엇인지 말한다.

        조작자가 "왜 0.90 에 못 가나" 를 묻게 되는 자리다. 대비가 낮으면 힘·이득이고
        신뢰도·연결성이 낮으면 U-Net 이다 — 둘 다 이 제어기가 할 수 있는 일이 아니다.
        """
        L = {k: v for k, v in self.q_losses(parts).items()
             if k in Q_NOT_OURS and v > self.LOSS_FLOOR}
        if not L:
            return ""
        k = max(L, key=L.get)
        return f"{k} {L[k]:.3f} — {Q_NOT_OURS[k]}"

    def _axes_idle(self, want: dict, q_view: Optional[float],
                   centroid_dx: float) -> bool:
        """기울임 축(thx·thy)에 지금 할 일이 없는가 — 법선 정렬이 열리는 조건.

        🔁 2026-09-14 밤: 예전에는 ``q_view >= q_target`` **하나뿐**이었다. 그것이
        문제가 된 이유는 합격선이 세 가지 일을 겸하고 있었기 때문이다. 합격선을 의도
        보존으로 0.977 까지 올리는 순간 최근 실측 최대에 걸려 **법선 정렬이 영영 안
        열리는** 상태가 됐다. Q_raw 를 자세로 고치는 고리가 통째로 죽는다.

        그래서 문을 둘로 나눴다:

          합격했다            예전 그대로. 합격하면 축은 확실히 한가하다.
          합격은 아니지만 한가 요구량(``_demand``)이 바닥이고 중심이 불감대 안이다.

        두 번째 줄이 새로 생긴 것이다. 총점과 요구량은 다른 것을 말한다 — 총점은
        "이 뷰가 좋은가", 요구량은 "내가 쓰려는 축을 지금 다른 목적이 쓰고 있는가" 다.
        모듈 독스트링이 처음부터 뒤쪽을 요구했다: "기울임은 이미 Q_seg 의 것이다.
        같은 축을 두 목적이 다투면 그 축은 둘 중 누구의 의도도 아니게 된다."

        실측도 둘이 다르다고 말한다 (채운 팬텀 17,202 프레임): 정렬 요구는 q_view
        0.80 부근에서 이미 바닥(중앙 0.0060 < LOSS_FLOOR)인데 훑기 요구는 0.90 에서도
        0.0264 로 남아 있다. 총점 하나로는 이 둘을 못 가른다.

        **불감대를 함께 보는 이유.** 요구량은 ``q_parts`` 에서 오고, 그 값이 중심과
        어긋날 수 있다 (합성 입력·전파된 마스크·눈금 변경 직후). 중심이 불감대 밖인데
        한가하다고 선언하면 제어기가 고칠 수 있는 오차를 두고 다른 일을 하러 간다.
        불감대는 이 제어기가 "중심은 됐다" 를 말하는 자기 기준이므로 그것을 따른다.

        **둘 다** 한가해야 연다 — 훑기도 회전축을 쓴다 (``--axes rot``).
        """
        if self._on_q(q_view) and q_view >= self.q_target:
            return True
        if not want:
            return False
        return (max(want.get(ALIGN, 0.0), want.get(SWEEP, 0.0)) <= self.LOSS_FLOOR
                and abs(centroid_dx) <= self.deadband)

    def _to_refine(self, t: float) -> None:
        """좁게 다시 훑는다. 이미 방광을 보고 있으므로 넓게 가면 있는 것을 놓친다."""
        self.find_range_now = self.find_half_range * self.refine_fraction
        self.find_range_now_deg = self.find_half_range_deg * self.refine_fraction
        self._to_find(t, 0.0)

    def _reset_range(self) -> None:
        """훑는 반경을 설정값으로 되돌린다. 방광을 봤을 때 부른다."""
        self.find_range_now = self.find_half_range
        self.find_range_now_deg = self.find_half_range_deg

    def _widen_range(self) -> bool:
        """반경을 넓힌다. 이미 상한이면 False.

        **한 바퀴를 다 훑고도 못 봤다는 것은 방광이 그 범위 밖에 있다는 증거다.** 같은
        범위를 그대로 되풀이하면 영원히 못 찾는다 — 2026-09-14 attempt_005 가 그렇게
        180 s 동안 ±12° 를 왕복하며 마스크 1 % 로 끝났다. 그날 프로브 시작 자세는 첫
        실행에서 82 mm 떨어져 있었고, ±12° 로는 닿을 수 없는 거리였다.
        """
        a = min(self.find_range_now * self.find_widen, self.find_max_range_mm)
        b = min(self.find_range_now_deg * self.find_widen, self.find_max_range_deg)
        # **회전 예산을 넘겨 넓히지 않는다.** 회전 찾기 축들의 반경 합이 복귀 상한을 넘으면
        # 한 바퀴 도는 것만으로 상한에 닿아 훑기가 복귀에 잡아먹힌다.
        share = sum((self.find_oop_scale if self.is_oop(x) else 1.0)
                    for x in self.find_axis_list if is_rotation(x))
        if share:
            b = min(b, self.HOME_MARGIN * self.max_excursion_deg / share)
        # **쓰는 단위만 본다.** 회전 축만 훑는데 병진 반경이 아직 안 찼다고 "넓혔다" 고
        # 말하면, 상한에 닿았다는 신호가 영영 안 나온다.
        use_rot = any(is_rotation(x) for x in self.find_axis_list)
        use_lin = any(not is_rotation(x) for x in self.find_axis_list)
        grew = ((use_lin and a > self.find_range_now + 1e-9)
                or (use_rot and b > self.find_range_now_deg + 1e-9))
        self.find_range_now, self.find_range_now_deg = a, b
        return grew

    #: 이름 → vel6 자리. 누적 변위를 축 이름으로 들고 있기 위해.
    def _accumulate(self, vel, dt: float) -> None:
        """낸 지령을 시간으로 적분해 축별 누적 변위를 따라간다."""
        if dt <= 0.0:
            return
        for name, i in AXIS_INDEX.items():
            v = float(vel[i])
            if v:
                self.axis_pos[name] = self.axis_pos.get(name, 0.0) + v * dt

    def excursion_deg(self) -> float:
        """시작 자세에서 회전 축이 얼마나 벗어났나 [°].

        러너의 ``_excursion_deg`` 는 자세 사원수로 잰다 — 이쪽은 지령 적분이라 값이 조금
        다르지만, **먼저 닿는 쪽이 이쪽이어야** 한다. 러너가 먼저 닿으면 지령이 끊긴다.
        """
        return float(sum(abs(v) for k, v in self.axis_pos.items() if is_rotation(k)))

    def _home_step(self, dt: float) -> tuple[np.ndarray, bool]:
        """시작 자세 쪽으로 한 걸음. (속도, 다 돌아왔는가)."""
        vel = np.zeros(6, float)
        far = [(k, v) for k, v in self.axis_pos.items() if abs(v) > self.home_tol]
        if not far:
            self.axis_pos = {k: 0.0 for k in self.axis_pos}
            return vel, True
        name, pos = max(far, key=lambda kv: abs(kv[1]))
        lim = self.max_deg_s if is_rotation(name) else self.max_mm_s
        vel[AXIS_INDEX[name]] = -lim if pos > 0 else lim
        return vel, False

    def _good_enough(self) -> float:
        """훑기를 도중에 멈춰도 되는 값.

        **최댓값을 찾는 것이 목적이 아니다.** 쓸 만한 뷰에 닿으면 거기서 멈춰야 한다 —
        끝까지 훑고 최고점으로 되돌아가는 사이에 그 최고점은 이미 지난 자리가 된다.
        """
        return self.q_good_enough if self._use_q else float("inf")

    def is_oop(self, axis: str) -> bool:
        """면외 축인가 — 정렬 축(면내) 이 아닌 모든 찾기 축.

        실측이 면내로 고른 축은 하나뿐이다 (thy, 중심 이득 +0.01913/°). 나머지는 절단면을
        바꾸는 축이고, 그쪽으로는 조금만 가도 방광을 지나친다.
        """
        return axis != self.align_axis

    def _make_scan(self, axis: str, good_enough: float = float("inf")) -> "SweepScan":
        """축의 단위에 맞는 훑기. 회전과 병진은 상한도 반경도 다르다."""
        rot = is_rotation(axis)
        r = self.find_range_now_deg if rot else self.find_range_now
        if self.is_oop(axis):
            r *= self.find_oop_scale
        return SweepScan(
            half_range=r,
            rate=min(self.find_rate if not rot else self.max_deg_s,
                     self.max_deg_s if rot else self.max_mm_s),
            min_value=self.area_lost, good_enough=good_enough)

    @property
    def align_max(self) -> float:
        """정렬 축 누적 변위 상한 — 단위가 축에 따라 다르다."""
        return self.align_max_deg if is_rotation(self.align_axis) else self.align_max_mm

    @property
    def max_rate(self) -> float:
        """정렬 축의 속도 상한 — 단위가 축에 따라 다르다."""
        return self.max_deg_s if is_rotation(self.align_axis) else self.max_mm_s

    # ------------------------------------------------------------------ 국면
    def _to_hold(self) -> None:
        self.phase = HOLD
        self.align_pos, self._hold_best, self._hold_best_pos = 0.0, float("-inf"), 0.0

    def _to_find(self, t: float, area: float) -> None:
        self.phase = FIND
        self._scan_i, self._best = 0, (float("-inf"), 0, 0.0)
        # 반경이 바뀌었을 수 있으므로 훑기를 다시 만든다 — SweepScan.reset 은 반경을
        # 건드리지 않는다.
        self._scans = [(a, self._make_scan(a, self._good_enough())) for a in self.find_axis_list]
        for _, sc in self._scans:
            sc.reset(t)
        self._scan = self._scans[0][1] if self._scans else None
        self._find_index = AXIS_INDEX[self._scans[0][0]] if self._scans else 0

    def update(self, t: float, has_mask: bool, centroid_dx: float,
               area: float, q_view: Optional[float] = None,
               q_parts: Optional[dict] = None,
               force_settled: bool = True,
               shear_xy: Optional[Sequence[float]] = None) -> tuple[np.ndarray, str]:
        """한 tick. 시각을 갱신하고 지령을 낸 뒤 누적 변위에 더한다.

        **시각 갱신을 여기 한 군데서만 하는 이유.** 예전에는 유지 분기에서만 갱신해서,
        찾기·복귀 구간의 dt 가 0 이 되고 누적 변위가 영원히 안 줄었다.
        """
        self._dt = max(0.0, t - (self._prev_t if self._prev_t is not None else t))
        self._prev_t = t
        vel, note = self._update(t, has_mask, centroid_dx, area, q_view, q_parts,
                                 force_settled, shear_xy)
        self._accumulate(vel, self._dt)
        return vel, note

    def _update(self, t: float, has_mask: bool, centroid_dx: float,
                area: float, q_view: Optional[float] = None,
                q_parts: Optional[dict] = None,
                force_settled: bool = True,
                shear_xy: Optional[Sequence[float]] = None) -> tuple[np.ndarray, str]:
        """지금 내보낼 vel6 과 한 줄 설명.

        ``q_view`` 를 주면 **훑기와 유지의 목적량이 Q_seg 가 된다.** 면적만 보면
        "크지만 화면 끝에 걸린 방광" 을 "작지만 가운데 있는 방광" 보다 위에 놓는다 —
        유지의 목적이 중심을 맞추는 것인데 찾기는 그것을 안 보는 셈이다. Q_seg 는
        유무·면적·중심을 76 % 가중치로 한꺼번에 담는다 (view_quality).
        면적은 q_view 가 없을 때의 대체이자, 사람이 읽는 기록용으로 남는다.
        """
        vel = np.zeros(6, float)
        seen = bool(has_mask) and area == area
        step = self._dt

        # 너무 멀리 왔으면 **되돌아가는 것이 먼저다.** 러너의 상한에 닿으면 지령이 통째로
        # 끊기고, 그러면 영원히 그 자리에 선다 (2026-09-14 rule_v3 attempt_003: 112 s 에
        # 30° 에 닿아 남은 68 s 동안 마스크 1 %).
        if self.phase != HOME and self.excursion_deg() > self.max_excursion_deg:
            self.phase = HOME
        if self.phase == HOME:
            v, done = self._home_step(step)
            if done:
                self._to_find(t, 0.0)
                self.last_note = "복귀 — 시작 자세로 돌아왔다. 다시 찾는다"
                return vel, self.last_note
            self.last_note = (f"복귀 — 시작에서 {self.excursion_deg():.0f}° "
                              f"(상한 {self.max_excursion_deg:.0f}°) · 돌아가는 중")
            return v, self.last_note

        # 훑기·유지가 최대로 만들 양. 단위가 다르므로 문턱도 함께 갈린다.
        if self._on_q(q_view) and not self._use_q:
            # 처음 Q 가 들어온 순간부터 훑기의 정지선이 생긴다 — 지금 돌고 있는 훑기에도
            # 넣어 준다. 다음 바퀴를 기다리면 이번 바퀴가 통째로 길어진다.
            self._use_q = True
            for _, sc in self._scans:
                sc.good_enough = self.q_good_enough
        val = float(q_view) if (q_view is not None and q_view == q_view) else area
        lost = self.q_lost if self._on_q(q_view) else self.area_lost
        found = self.q_found if self._on_q(q_view) else self.area_found

        if self.phase == HOLD and (not seen or val < lost):
            self._to_find(t, val if seen else 0.0)

        if self.phase == FIND:
            if self._scan is None:
                self.last_note = "찾기 축이 없다 — 정렬만 한다"
                self._to_hold()
            elif self._scan.done:
                # 이 축을 다 훑었다. 다음 축이 남아 있으면 이어서 훑는다 —
                # 병진과 회전을 함께 주면 둘 다 해 보고 나은 쪽으로 간다.
                if self._scan.best_value > self._best[0]:
                    self._best = (self._scan.best_value, self._scan_i, self._scan.best_pos)
                # 이 축에서 이미 합격선에 닿았으면 남은 축은 안 훑는다. 더 나은 자리를
                # 찾자고 지금 있는 좋은 자리를 떠날 이유가 없다.
                enough = self._scan.best_value >= self._good_enough()
                if not enough and self._scan_i + 1 < len(self._scans):
                    self._scan_i += 1
                    name, self._scan = self._scans[self._scan_i]
                    self._find_index = AXIS_INDEX[name]
                    self.last_note = f"찾기 — 다음 축 {name} 로"
                    return vel, self.last_note
                # 전부 훑었다. 방광을 봤으면 유지로, 못 봤으면 처음부터 다시 —
                # 팬텀이 변하거나 조작자가 프로브를 옮겼을 수 있다.
                if self._best[0] > lost and seen and val >= found:
                    self._reset_range()     # 찾았다 — 다음 번은 좁은 범위부터
                    self._to_hold()
                else:
                    grew = self._widen_range()
                    self.last_note = (
                        f"찾기 — {len(self._scans)} 축을 다 훑었지만 방광을 못 봤다 "
                        f"(최대 {max(self._best[0], 0.0):.4f}). "
                        + (f"반경을 ±{self.find_range_now_deg:.0f}°/±"
                           f"{self.find_range_now:.0f}mm 로 넓혀 다시 훑는다" if grew
                           else f"반경이 이미 상한(±{self.find_range_now_deg:.0f}°/±"
                                f"{self.find_range_now:.0f}mm)이다 — 프로브 위치를 옮겨야 한다"))
                    self._to_find(t, area)
                    return vel, self.last_note
            else:
                name = self._scans[self._scan_i][0]
                vel[self._find_index] = self._scan.update(t, val if seen else 0.0)
                self.last_note = (f"찾기 [{name} {self._scan_i + 1}/{len(self._scans)}] "
                                  f"{self._scan.describe()} · {self._vname(q_view)} {val:.4f}")
                return vel, self.last_note

        # 유지: 중심을 0 으로 — 단, **시야를 희생하지 않는다**
        dt = self._dt
        lim, u = self.max_rate, unit(self.align_axis)
        i = AXIS_INDEX[self.align_axis]

        if not seen:
            self.last_note = "유지 — 방광이 잠깐 안 보인다 (멈춤)"
            return vel, self.last_note
        if val > self._hold_best:
            self._hold_best, self._hold_best_pos = val, self.align_pos

        # 시야가 무너졌다 — 가장 좋았던 자리로 되돌아가고 다시 찾는다.
        if self._hold_best > 0 and val < self.keep_fraction * self._hold_best:
            gap = self._hold_best_pos - self.align_pos
            if abs(gap) > lim * max(dt, 1e-3):
                v = lim if gap > 0 else -lim
                self.align_pos += v * dt
                vel[i] = v
                self.last_note = (f"유지 — {self._vname(q_view)} 이 {val:.4f} 로 무너졌다 "
                                  f"(최고 {self._hold_best:.4f}) · 되돌아가는 중 "
                                  f"{self.align_pos:+.1f}/{self._hold_best_pos:+.1f} {u}")
                return vel, self.last_note
            self.align_pos = self._hold_best_pos
            self._to_find(t, area)
            self.last_note = "유지 — 되돌아왔다. 다시 찾는다"
            return vel, self.last_note

        # --- 여기서부터 Q 의 **항별 요구**가 움직임을 고른다 -------------------
        #
        # 합쳐진 총점은 "좋다/나쁘다" 만 말한다. 무엇을 해야 하는지는 **어느 항이 깎았는가**
        # 가 말한다 — 중심이 깎았으면 정렬 축을 돌려야 하고, 면적이 깎았으면 면외로 다시
        # 훑어야 한다. 둘은 정반대 동작이다.
        want = self._demand(q_parts)
        if self._axes_idle(want, q_view, centroid_dx):
            # **이 축들이 한가하다.** 그 여유로 결합(Q_raw)을 고친다. Q_raw 는 힘만으로는
            # 더 안 오른다 — 실측에서 결합은 힘보다 자세에 훨씬 강하게 반응했고
            # (r +0.49·+0.68 vs −0.09·−0.27) 힘 탐색은 이미 수렴해 있었다.
            # 한가함이 깨지면 다음 tick 에 이 분기를 안 타므로 즉시 원래 일로 돌아간다.
            # 합격했는가, 아니면 "우리 축은 한가한데 총점이 낮은가" 를 구별해 적는다.
            # 뒤쪽이면 남은 부족분은 이 제어기의 것이 아니다 — 무엇인지 말해 준다.
            passed = self._on_q(q_view) and q_view >= self.q_target
            why = self._blocked_by(q_parts)
            head = "합격" if passed else ("여유" + (f" · 남은 부족분: {why}" if why else ""))
            if self.normal_align and shear_xy is not None:
                nv, nnote = self.na.step(t, shear_xy)
                self.last_note = f"유지 — {self._mark(q_view)}{head} · {nnote}"
                return (nv if np.any(nv) else vel), self.last_note
            self.last_note = (f"유지 — {self._mark(q_view)}{head} · 중심 {centroid_dx:+.3f} "
                              f"· 면적 {area:.4f} (가만히 둔다)")
            return vel, self.last_note

        # **힘이 아직 자리를 못 잡았으면 면적 부족은 우리 일이 아니다.**
        #
        # 방광 용적이 바뀌면 접촉 강성이 바뀌고, 힘 조절기가 z 로 파고들거나 물러난다.
        # 그 동안 깊이가 변하므로 면적도 변한다 — 그것은 **z 가 이미 고치고 있는 것**이다.
        # 여기서 훑기를 시작하면 같은 오차를 두 축이 서로 모르게 쫓는다. 중심은 z 와
        # 무관하므로 그대로 고친다 (조작자 지시: 좌우 이동은 힘 조절에 쓰지 않는다).
        if not force_settled and want[SWEEP] > self.LOSS_FLOOR:
            self.last_note = (f"유지 — 부족분이 면적/유무 ({want[SWEEP]:.3f}) 이지만 "
                              f"**힘이 아직 움직이는 중**이다 · z 에 맡긴다 · "
                              f"{self._mark(q_view)}면적 {area:.4f}")
            want = dict(want, **{SWEEP: 0.0})
            if want[ALIGN] <= self.LOSS_FLOOR:
                return vel, self.last_note
        if want[SWEEP] > want[ALIGN] and want[SWEEP] > self.LOSS_FLOOR:
            # 면적·유무가 모자라다 — 정렬로는 못 고친다. 좁게 다시 훑는다.
            if t - self._last_refine_t >= self.refine_cooldown_s:
                self._last_refine_t = t
                self._to_refine(t)
                self.last_note = (f"유지 → 찾기 — {self._mark(q_view)}부족분이 면적/유무 "
                                  f"({want[SWEEP]:.3f} > 중심 {want[ALIGN]:.3f}) · "
                                  f"±{self.find_range_now_deg:.0f}°/±{self.find_range_now:.0f}mm "
                                  "로 좁게 다시 훑는다")
                return vel, self.last_note
            self.last_note = (f"유지 — 부족분이 면적/유무 ({want[SWEEP]:.3f}) 이지만 "
                              f"다시 훑기 대기 {self.refine_cooldown_s - (t - self._last_refine_t):.0f} s "
                              f"· {self._mark(q_view)}면적 {area:.4f}")
            return vel, self.last_note

        # 중심(·경계)이 모자라다 → 정렬 축. q_parts 가 없으면 예전처럼 불감대로 판단한다.
        gate = (want[ALIGN] <= self.LOSS_FLOOR) if q_parts else (abs(centroid_dx) <= self.deadband)
        if gate or abs(centroid_dx) <= self.deadband:
            why = self._blocked_by(q_parts)
            self.last_note = (f"유지 — 중심 {centroid_dx:+.3f} (불감대 안) · "
                              f"{self._mark(q_view)}면적 {area:.4f}"
                              + (f" · 남은 부족분: {why}" if why else ""))
            return vel, self.last_note

        need = self.gains.centering_move(centroid_dx, self.align_axis)
        v = float(np.clip(self.k_p * need, -lim, lim))
        # 누적 변위 상한. 한 번의 보정이 시야를 벗어날 만큼 커지면 안 된다.
        if abs(self.align_pos + v * dt) > self.align_max:
            self.last_note = (f"유지 — 중심 {centroid_dx:+.3f} 이지만 정렬 변위가 "
                              f"{self.align_pos:+.1f} {u} (상한 {self.align_max:.0f}) — 멈춘다")
            return vel, self.last_note
        self.align_pos += v * dt
        vel[i] = v
        self.last_note = (f"유지 [중심 부족 {want[ALIGN]:.3f}] {centroid_dx:+.3f} → "
                          f"{self.align_axis} {need:+.1f} {u} 필요, {v:+.2f} {u}/s "
                          f"(누적 {self.align_pos:+.1f}) · {self._mark(q_view)}면적 {area:.4f}")
        return vel, self.last_note

    def describe(self) -> str:
        return (f"규칙 제어 [{self.phase}] 정렬 {self.align_axis} "
                f"(이득 {self.gains.centroid_per_unit.get(self.align_axis, 0):+.5f} "
                f"/{unit(self.align_axis)}) · "
                f"찾기 {'·'.join(self.find_axis_list) or '없음'}")
