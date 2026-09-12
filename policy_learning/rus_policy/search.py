"""측정한 Q 로 오르막을 오른다 — 방광을 찾는 탐색기.

    s = HillClimbSearch()
    omega_deg_s = s.update(t, q_measured)      # (wx, wy, wz), 프로브 프레임

왜 학습된 Q̂ 로 고르지 않는가
----------------------------
2026-09-11~12 실측: 디코더가 낸 후보 64 개는 θx 가 51 % 양수로 고르게 퍼져 있는데, Q̂ 로 고른
뒤에는 32 % (Q_area 재학습 헤드는 9 %) 가 된다 — 64 개 중 최댓값을 고르다 보니 Q̂ 의 작은
기울어짐이 고정된 방향이 된다. 그 Q̂ 는 방향을 모른다: 크기가 같고 방향만 반대인 행동을
구별하는 정확도가 52~54 % 다 (찍으면 50 %). 학습 데이터가 조작자가 보면서 고른 움직임이라
방향의 효과가 선택과 뒤섞여 있어서, 라벨을 Q_area 로 바꾸고 증강을 넣어도 그대로였다.

**방향은 예측하지 말고 재면 된다.** 실제로 조금 움직여 보고 Q 가 올랐는지 보는 것은 관찰이
아니라 개입이라 뒤섞임이 없다. 힘 축에서 ForceSetpointAdapter 가 하는 일과 같고, 여기서는
회전 축에 대해 한다.

어떻게 도는가
-------------
한 방향으로 ``step_deg`` 만큼 돌리고, 영상이 따라올 때까지 기다렸다 Q 를 재서

  올랐으면   그 방향을 계속 간다 (되돌아오는 이동이 없다)
  안 올랐으면 **한 걸음 되돌아가** (직전이 곧 최고점이다) 다음 방향으로 넘어간다

여섯 방향(±θx·±θy·±θz)을 다 해도 안 오르면 지역 최고점이다. ``dwell_s`` 만큼 쉬었다가 다시 훑는다 —
팬텀이 변하면 최고점도 움직이므로 멈춰 있기만 하면 안 된다.

걸음 크기와 문턱은 실기 데이터로 정했다 (2026-09-12, eval_v1 15 에피소드 · 8975 표본)
------------------------------------------------------------------------------------
처음 값(2° · min_gain 0.01)은 **걸음 효과가 잡음에 묻혀** 제대로 돌지 않았다. 실측:

    1.3 s 동안의 실제 자세 변화        |ΔQ| 중앙    75 %p
      0.0~0.5°  (사실상 정지)            0.001      0.007
      0.5~1.5°                           0.007      0.039
      1.5~3.0°                           0.012      0.029

즉 **dQ/dθ ≈ 0.005 /°** 이고, 프로브가 멈춰 있어도 Q 는 0.007 (75 %p) 만큼 저 혼자 움직인다.
2° 걸음이 만드는 0.010 은 그 잡음과 거의 같아, 실제 탐색 170 걸음에서 걸음당 |ΔQ| 중앙이
0.018 인데 멈춤 중 가짜 차이의 90 %p 가 0.077 이었다 — 판정의 상당수가 잡음이었다.

**평균을 길게 잡아도 소용없다.** 창을 0.6 s 에서 4 s 로 늘려도 창 간 변동은 0.148 → 0.210 으로
줄지 않는다 (수 초 규모의 실제 시야 변동이라 고주파 잡음이 아니다). A-B-A 대칭 비교로 선형
드리프트를 상쇄해도 0.004 → 0.003 뿐이다 (간헐적 점프라 선형이 아니다). 남은 지렛대는
**걸음을 키우는 것**뿐이다: 5° 면 기대 ΔQ 0.025 로 정지 잡음 75 %p(0.007) 의 3.5 배가 된다.

``min_gain`` 0.02 는 정지 잡음 75 %p 위, 5° 걸음의 기대 효과 아래에 둔 값이다.

걸음은 **찾을 때 거칠게, 봉우리에서 좁게**
-------------------------------------------
큰 걸음은 잡음을 이기지만 봉우리를 거칠게 잡는다 (가상 지형에서 2° 는 Q 0.97 까지, 5° 는
0.85 까지). 평가가 "방광을 찾고 → 용적이 변하는 동안 유지" 라 둘 다 필요하다. 그래서

  모든 방향이 실패하면 (지역 최고점)  걸음을 ``step_decay`` 배로 줄인다 (``step_min_deg`` 까지)
  Q 가 최고점보다 ``reset_drop`` 넘게 떨어지면  시야가 바뀐 것이므로 거친 걸음으로 되돌린다

``step_min_deg`` 3° 는 기대 효과 0.015 로 정지 잡음(0.007) 의 두 배 — 더 줄이면 잡음에 묻힌다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

#: 시험할 방향 — (축, 부호). 축 0·1·2 = θx·θy·θz (프로브 프레임).
#:
#: 세 축을 다 쓴다. 처음에는 θz(빔축 둘레 = 프로브 장축 회전)를 뺐는데, "영상면의 방향만
#: 바꾼다" 는 근거가 틀렸다 — 빔축 둘레로 돌리면 **보는 단면 자체가 바뀌므로** 화면 밖에 있던
#: 방광이 들어올 수 있다 (방광이 빔축 위에 정확히 놓여 있을 때만 안 바뀐다). 시연자도 θx 만큼
#: (chunk 순회전 산포 θx 6.76° · θz 6.29°) 썼다. 정책의 디코더는 θz 를 거의 못 만들지만
#: (후보 산포 0.46°) 제어 스택은 그대로 실행한다 — gate_axes 가 회전 세 축을 다 통과시킨다.
AXES_XYZ: tuple[tuple[int, float], ...] = (
    (0, +1.0), (0, -1.0), (1, +1.0), (1, -1.0), (2, +1.0), (2, -1.0))
#: 장축 회전을 빼고 싶을 때 (접촉면에 비틀림을 주기 싫은 경우).
AXES_XY: tuple[tuple[int, float], ...] = ((0, +1.0), (0, -1.0), (1, +1.0), (1, -1.0))
DEFAULT_AXES = AXES_XYZ

MOVE, SETTLE, UNDO, UNDO_SETTLE, DWELL = "이동", "측정", "되돌림", "재측정", "대기"


@dataclass
class HillClimbSearch:
    """측정한 Q 의 오르막을 따라간다. rclpy 없이 돈다 — 시간과 Q 를 밖에서 받는다.

    Args:
        axes: 시험할 (축, 부호) 목록.
        step_deg: 한 걸음의 회전각 [°].
        rate_deg_s: 회전 속도 [°/s]. 러너의 상한과 같게 둔다.
        settle_s: 걸음 뒤 Q 를 모으는 시간 [s]. 초음파 지연(0.2 s)보다 넉넉해야 한다.
        min_gain: 이만큼은 올라야 "나아졌다" 로 본다.
        dwell_s: 모든 방향이 실패한 뒤 쉬는 시간 [s].
    """

    axes: Sequence[tuple[int, float]] = DEFAULT_AXES
    step_deg: float = 5.0        # 2° 는 잡음에 묻혔다 — 위 표 참조
    rate_deg_s: float = 3.0
    settle_s: float = 0.6        # 늘려도 창 간 변동이 안 줄어 그대로 둔다
    min_gain: float = 0.02       # 정지 잡음 75 %p(0.007) 위
    step_min_deg: float = 3.0    # 이보다 줄이면 걸음 효과가 잡음에 묻힌다
    step_decay: float = 0.6      # 지역 최고점에서 걸음을 좁히는 배율
    reset_drop: float = 0.15     # 최고점 대비 이만큼 떨어지면 거친 걸음으로 되돌린다
    dwell_s: float = 2.0

    step: float = field(default=float("nan"), init=False)   # 지금 쓰는 걸음 [°]
    phase: str = field(default=MOVE, init=False)
    best_q: float = field(default=float("nan"), init=False)
    idx: int = field(default=0, init=False)          # 지금 시험 중인 방향
    n_fail: int = field(default=0, init=False)       # 연속 실패한 방향 수
    n_steps: int = field(default=0, init=False)      # 지금 방향으로 연속 전진한 걸음
    _t0: Optional[float] = field(default=None, init=False)
    _samples: list = field(default_factory=list, init=False)
    #: 진단용 — 마지막 판정 ("나아짐" / "제자리")
    last_decision: str = field(default="", init=False)

    @property
    def move_s(self) -> float:
        return self.current_step / max(self.rate_deg_s, 1e-6)

    @property
    def current_step(self) -> float:
        """지금 쓰는 걸음. ``reset`` 전에는 설정값."""
        return self.step_deg if self.step != self.step else self.step

    def reset(self, t: float, q: float = float("nan")) -> None:
        self.phase, self._t0, self._samples = MOVE, t, []
        self.best_q, self.idx, self.n_fail, self.n_steps = q, 0, 0, 0
        self.step, self.last_decision = float(self.step_deg), ""

    def _omega(self, sign: float) -> tuple[float, float, float]:
        out = [0.0, 0.0, 0.0]
        axis, direction = self.axes[self.idx]
        out[axis] = sign * direction * self.rate_deg_s
        return tuple(out)

    def _enter(self, phase: str, t: float) -> None:
        self.phase, self._t0, self._samples = phase, t, []

    def update(self, t: float, q: float, blocked: bool = False) -> tuple[float, float, float]:
        """지금 내보낼 각속도 [°/s]. ``blocked`` 면 이 방향은 더 못 간다고 보고 되돌린다."""
        if self._t0 is None:
            self.reset(t, q)
        dt = t - self._t0

        if self.phase == MOVE:
            if blocked:                      # 안전 한계 — 간 만큼 되돌리고 다음 방향
                self.last_decision = "막힘"
                self._enter(UNDO, t - (self.move_s - min(dt, self.move_s)))
                return self._omega(-1.0)
            if dt < self.move_s:
                return self._omega(+1.0)
            self._enter(SETTLE, t)
            return (0.0, 0.0, 0.0)

        if self.phase in (SETTLE, UNDO_SETTLE):
            if q == q:                       # NaN 이 아니면 모은다
                self._samples.append(q)
            if dt < self.settle_s:
                return (0.0, 0.0, 0.0)
            measured = sum(self._samples) / len(self._samples) if self._samples else float("nan")
            if self.phase == UNDO_SETTLE:
                # 되돌아온 자리에서 다시 재서 기준을 새로 잡는다. 팬텀이 변하면 예전 최고값은
                # 더 이상 그 자리의 값이 아니다 — 옛 값을 들고 있으면 영영 못 넘는다.
                if measured == measured:
                    if (self.best_q == self.best_q
                            and measured < self.best_q - self.reset_drop):
                        # 방광이 시야에서 크게 벗어났다 — 좁은 걸음으로는 못 돌아온다
                        self.step = float(self.step_deg)
                        self.n_fail = 0
                    self.best_q = measured
                self._advance()
                self._enter(DWELL if self.n_fail >= len(self.axes) else MOVE, t)
                return (0.0, 0.0, 0.0)
            improved = (measured == measured) and (
                self.best_q != self.best_q or measured > self.best_q + self.min_gain)
            if improved:
                self.best_q = measured
                self.n_fail, self.n_steps = 0, self.n_steps + 1
                self.last_decision = "나아짐"
                self._enter(MOVE, t)
                return self._omega(+1.0)
            self.last_decision = "제자리"
            self._enter(UNDO, t)
            return self._omega(-1.0)

        if self.phase == UNDO:
            if dt < self.move_s:
                return self._omega(-1.0)
            self._enter(UNDO_SETTLE, t)
            return (0.0, 0.0, 0.0)

        # DWELL — 지역 최고점. 쉬었다가 다시 훑는다 (팬텀이 변한다).
        if dt < self.dwell_s:
            return (0.0, 0.0, 0.0)
        self.n_fail = 0
        self._enter(MOVE, t)
        return self._omega(+1.0)

    def _advance(self) -> None:
        """다음 방향으로. 실패가 한 바퀴를 채우면 지역 최고점이다 — 걸음을 좁힌다."""
        self.n_fail += 1
        self.n_steps = 0
        self.idx = (self.idx + 1) % len(self.axes)
        if self.n_fail >= len(self.axes):
            self.step = max(self.step_min_deg, self.current_step * self.step_decay)

    def describe(self) -> str:
        axis, direction = self.axes[self.idx]
        name = ("θx", "θy", "θz")[axis]
        best = f"{self.best_q:.3f}" if self.best_q == self.best_q else "—"
        return (f"{self.phase} {'+' if direction > 0 else '−'}{name}  최고 Q {best}  "
                f"걸음 {self.current_step:.1f}°  "
                f"연속전진 {self.n_steps}  실패 {self.n_fail}/{len(self.axes)}")
