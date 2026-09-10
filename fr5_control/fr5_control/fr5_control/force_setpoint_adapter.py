"""배경 힘 적응기 — 힘 설정값을 영상 품질의 경사 방향으로 천천히 옮긴다.

DESIGN_NOTES §8.4 (2026-09-08 개정) 의 Stage 1 잔존 루프다. FORCE LOCK 은 완전 잠금이
아니라 **설정값 기능**이다: Stage 1 이 찾은 F* 를 고정한 채 5-DoF 로 움직이면 프로브는
곡률과 조직이 다른 자리로 가고, 그 자리의 최적 힘은 F* 가 아니다. 전체 재탐색 대신
극값 탐색(extremum seeking) 으로 따라간다 — 설정값에 구형파 디더를 걸고, 품질 응답을
같은 위상으로 복조해 경사를 얻고, 그 방향으로 작은 걸음을 옮긴다.

``contact_state`` 와 같은 이유로 rclpy 없이 돈다. 이 루프의 버그는 "설정값이 어느
날 조용히 한계까지 기어올라갔다" 는 종류라, 실로봇이 아니라 pytest 로 잡아야 한다.
시간은 밖에서 받는다 (``t``). 벽시계를 안 읽으므로 기록을 되감아 같은 결과를 낸다.

**시간 규모 분리 — 세 층이 한 자릿수씩 떨어져 있어야 성립한다.**

    힘 유지 · Q̄ 홀드 창   1 s     (§7.2 홀드 윈도우, 힘 루프 정착 ±0.05 N 검증)
    <  디더 반주기          2.5 s   (period_s / 2)
    <  적응                 한 주기에 ≤ step_max_n (0.1 N) — 1 N 옮기는 데 ≥ 50 s

힘 루프가 반주기 안에 새 레벨에 앉아야 복조가 "힘의 차이에 대한 품질의 차이" 를 재고,
적응이 반주기보다 훨씬 느려야 디더가 경사를 재는 동안 기준점이 움직이지 않는다.
반주기 시작 뒤 ``settle_s`` 동안의 표본은 버린다 — 조직 점탄성 완화 구간이라 그 품질은
이전 레벨의 잔상이다. 남는 2.0 s 가 홀드 창 1 s 보다 길다.

⚠️ 위 첫 줄의 "1 s" 는 **전제이지 실측이 아니다.** probe.yaml 의 ``admittance_b_z``
주석은 B_z 4500 에서 "정착까지 십수 초" 를 적고 있고, 시정수는 B_z / k_c 로 k_c
0.66 ~ 1.8 N/mm 에서 2.5 ~ 7 s 다 — 반주기와 같거나 길다. 그러면 실현되는 힘의
진폭이 지령 ``amplitude_n`` 보다 작고, 복조가 **지령 진폭으로 나누므로** 경사의 크기가
과소평가된다. 부호는 지켜지고 걸음은 ``step_max_n`` 으로 잘리므로 수렴은 하되 느리다.
바로잡으려면 실현된 ‖F‖ 진폭으로 나누거나 주기를 늘려야 하며, 어느 쪽이든 팬텀에서
k_c 를 다시 잰 뒤의 일이다.

**반송파는 걸러내지 않고 상태로 드러낸다.** policy 가 볼 관측에 디더가 섞이면 5 s
주기 신호를 elevational 로 오학습한다 (POLICY_LEARNING_MATH §1.4). 설정값을 필터로
씻어 내는 길은 두 가지로 나쁘다. 구형파는 홀수 고조파(0.6, 1.0 Hz …)를 다 가지고 있어
노치 하나로 안 지워지고, 그 3차 고조파 0.6 Hz 는 §1.4 의 면외 디더 0.5 Hz 바로 옆이라
새는 만큼이 최악의 오염이 된다. 게다가 필터는 수 초의 지연을 붙인다. 디더는 **우리가
만든 신호**라 추정할 것이 없다 — 반송파 없는 값 ``f_bar`` 가 이 클래스의 상태이고,
:meth:`setpoint` 가 거기에 디더를 얹으며, :meth:`observation` 은 ``f_bar`` 와 디더
위상을 **따로** 낸다. policy 는 위상 채널을 받거나 아예 안 받거나이지, 섞인 값을 받지
않는다.

**한계는 ‖F‖ 기준이다.** ``f_min_n`` / ``f_max_n`` 은 probe.yaml ``safety`` 절과 같은
양이다 (``max_contact_force_n: 5.0``, ``ft_sensor.contact_force_mode: magnitude``).
clamp 는 ``f_bar`` 가 아니라 **디더를 얹은 설정값**에 걸린다 — ``f_bar`` 는
``[f_min + amplitude, f_max − amplitude]`` 안에 묶여 ``setpoint`` 가 한계를 넘는 순간이
없다. ``warn_contact_force_n: 4.5`` 는 "상승 금지" 선이므로 supervisor 는 ``f_max_n``
에 하드 한계 5.0 이 아니라 그 값을 넘기는 것이 맞다. 시작점 ``target_force_n: 3.0``
은 Stage 1 이 찾을 값의 임시 대체다.

**재탐색 트리거는 Q̄_seg 에 건다.** :func:`quality_degraded` 는 Stage 1 종료 시점의
Q̄_seg 를 기준으로 25 % 열화를 본다. Q̄_raw 가 아니다 — Q̄_raw 는 접촉만 되면 힘에
대해 평탄해서 "닿아 있다" 는 말고는 열화를 거의 안 보이고, 그것으로 트리거를 걸면
방광이 사라져도 재탐색이 안 뜬다. 접촉 자체가 무너지는 경우는 rus_perception 의
결합 게이트(``coupling_gate`` False)가 잡고, 그때 supervisor 는 Stage 1b 가 아니라
**Stage 1a 로** 돌아간다 — 세그가 없는 상태에서 Q̄_seg 는 정의되지 않으므로.

🟡 **이 모듈의 숫자 기본값은 전부 미확정이다.** 진폭 0.25 N, 주기 5 s, 완화 0.5 s,
이득 1.0, 걸음 0.1 N, 한계 1.0 / 5.0 N, 반주기당 최소 표본 4 — 팬텀 Q(F) 곡선을 얻기
전에는 근거가 §8.4 의 제안값뿐이다. probe.yaml ``force_search`` (주석 상태) 가 같은
숫자를 들고 있고, supervisor 가 생기면 그쪽이 진실이 된다.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

__all__ = ["ForceSetpointAdapter", "SetpointPusher", "quality_degraded"]


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class SetpointPusher:
    """같은 값을 반복해 밀지 않는다. 파라미터 set 은 조절기를 다시 만드는 일이라 싸지 않다."""

    def __init__(self, eps_n: float = 0.01) -> None:
        self.eps = float(eps_n)
        self.last: Optional[float] = None

    def should_push(self, setpoint: float) -> bool:
        if not math.isfinite(setpoint):
            return False
        if self.last is None:
            return True
        return abs(setpoint - self.last) > self.eps

    def mark(self, setpoint: float) -> None:
        self.last = float(setpoint)

class ForceSetpointAdapter:
    """구형파 디더 극값 탐색으로 힘 설정값을 적응시킨다.

    한 주기 = ``period_s``. 앞 반주기 부호 +1, 뒤 반주기 −1 (위상은 ``t − t0`` 에서).
    품질 표본은 :meth:`update` 로 들어오고, 위상이 한 바퀴 돌 때 한 번 판정한다.
    """

    def __init__(
        self,
        f_bar0: float,
        *,
        amplitude_n: float = 0.25,
        period_s: float = 5.0,
        settle_s: float = 0.5,
        gain_n_per_unit: float = 1.0,
        step_max_n: float = 0.1,
        f_min_n: float = 1.0,
        f_max_n: float = 5.0,
        min_samples_per_half: int = 4,
        t0: float = 0.0,
    ) -> None:
        """적응기를 만든다.

        Args:
            f_bar0: 시작 설정값 [N], 디더 없는 값. Stage 1 이 찾은 F*. 디더를 얹어도
                한계를 안 넘도록 ``[f_min + amplitude, f_max − amplitude]`` 로 묶는다 —
                묶였는지는 :attr:`f_bar` 를 ``f_bar0`` 와 비교하면 안다.
            amplitude_n: 디더 반폭 [N]. 🟡
            period_s: 디더 한 주기 [s]. 반주기가 힘 루프 정착보다 길어야 한다. 🟡
            settle_s: 반주기 전환 뒤 버리는 시간 [s]. 조직 완화 구간. 🟡
            gain_n_per_unit: 경사 → 걸음 이득 [N / (품질 단위 / N)]. 🟡
            step_max_n: 한 주기에 옮길 수 있는 최대 [N]. 적응 시간 규모를 정한다. 🟡
            f_min_n: 디더를 얹은 설정값의 하한 [N], ‖F‖ 기준. 🟡
            f_max_n: 상한 [N], ‖F‖ 기준. probe.yaml ``safety.max_contact_force_n``
                과 같은 양이며, 상승 금지 선(``warn_contact_force_n``)을 넘기는 것이
                맞다. 🟡
            min_samples_per_half: 반주기마다 이 수 이상의 유효 표본이 있어야 경사를
                믿는다. 모자라면 그 주기는 측정 실패다. 🟡
            t0: 위상 기준 시각 [s].

        Raises:
            ValueError: 진폭·주기가 양수가 아니거나, 완화 시간이 반주기를 다 먹거나,
                한계 안에 디더가 들어갈 자리가 없다.
        """
        if amplitude_n <= 0.0:
            raise ValueError(
                f"amplitude_n({amplitude_n}) 은 양수여야 한다 — 0 이면 경사를 잴 수 없다"
            )
        if period_s <= 0.0:
            raise ValueError(f"period_s({period_s}) 은 양수여야 한다")
        if settle_s < 0.0 or settle_s >= period_s / 2.0:
            raise ValueError(
                f"settle_s({settle_s}) 는 0 이상, 반주기({period_s / 2.0}) 미만이어야 한다 — "
                "반주기를 다 버리면 유효 표본이 없다"
            )
        if gain_n_per_unit < 0.0 or step_max_n < 0.0:
            raise ValueError("이득과 최대 걸음은 음수일 수 없다")
        if min_samples_per_half < 1:
            raise ValueError(
                f"min_samples_per_half({min_samples_per_half}) 는 1 이상이어야 한다"
            )
        if f_min_n + amplitude_n > f_max_n - amplitude_n:
            raise ValueError(
                f"[f_min {f_min_n}, f_max {f_max_n}] 안에 디더 ±{amplitude_n} 가 "
                "들어갈 자리가 없다"
            )

        self.amplitude = float(amplitude_n)
        self.period = float(period_s)
        self.settle_s = float(settle_s)
        self.gain = float(gain_n_per_unit)
        self.step_max = float(step_max_n)
        self.f_min = float(f_min_n)
        self.f_max = float(f_max_n)
        self.min_samples_per_half = int(min_samples_per_half)

        # 아래 상태는 reset 이 채운다.
        self._f_bar = 0.0
        self._t0 = 0.0
        self._last_t = 0.0
        self._period_idx = 0
        self._plus: List[float] = []
        self._minus: List[float] = []
        self._n_seen = 0
        self._n_discarded = 0
        self._n_updates = 0
        self._n_failed = 0
        self.reset(f_bar0, t0)

    # ---- 위상 ---------------------------------------------------------------

    @property
    def half_period_s(self) -> float:
        """반주기 [s]. 힘 루프 정착과 비교해 볼 값이다."""
        return self.period / 2.0

    def _elapsed(self, t: float) -> float:
        return float(t) - self._t0

    def _period_index(self, t: float) -> int:
        return int(math.floor(self._elapsed(t) / self.period))

    def dither_sign(self, t: float) -> float:
        """시각 ``t`` 의 디더 부호. 앞 반주기 +1.0, 뒤 반주기 −1.0."""
        phase = self._elapsed(t) % self.period
        return 1.0 if phase < self.half_period_s else -1.0

    def _in_settle(self, t: float) -> bool:
        """반주기 전환 직후 ``settle_s`` 안인가. 그 표본은 이전 레벨의 잔상이다."""
        tau = self._elapsed(t) % self.half_period_s
        return tau < self.settle_s

    # ---- 밖으로 나가는 값 ---------------------------------------------------

    def setpoint(self, t: float) -> float:
        """admittance 가 받을 힘 설정값 [N] = ``f_bar + amplitude · sign(t)``.

        ``f_bar`` 가 ``[f_min + a, f_max − a]`` 에 묶여 있으므로 이 값은 ``[f_min, f_max]``
        를 벗어나지 않는다.
        """
        return self._f_bar + self.amplitude * self.dither_sign(t)

    def observation(self, t: float) -> Dict[str, float]:
        """policy 관측에 들어갈 것 (POLICY_LEARNING_MATH §1.4).

        ``f_bar`` 에는 디더가 없다. 디더는 부호와 정현 위상으로 **따로** 준다 — 관측에
        넣을지 말지는 학습 쪽이 정하되, 섞인 값을 받는 일은 없다.
        """
        angle = 2.0 * math.pi * self._elapsed(t) / self.period
        return {
            "f_bar": self._f_bar,
            "dither_sign": self.dither_sign(t),
            "dither_sin": math.sin(angle),
            "dither_cos": math.cos(angle),
        }

    # ---- 갱신 ---------------------------------------------------------------

    def reset(self, f_bar0: float, t0: float = 0.0) -> None:
        """Stage 1 재탐색 뒤 새 F* 로 다시 시작한다.

        위상·누적 표본·횟수를 모두 비운다. 이전 세션의 횟수까지 잇고 싶으면
        supervisor 가 따로 센다 — 재탐색을 거친 값은 새 출발이지 이어짐이 아니다.
        """
        self._f_bar = _clamp(
            float(f_bar0), self.f_min + self.amplitude, self.f_max - self.amplitude
        )
        self._t0 = float(t0)
        self._last_t = float(t0)
        self._period_idx = 0
        self._plus = []
        self._minus = []
        self._n_seen = 0
        self._n_discarded = 0
        self._n_updates = 0
        self._n_failed = 0

    def update(self, t: float, q: Optional[float]) -> Optional[Dict[str, object]]:
        """품질 표본 하나를 넣는다. 주기가 닫히는 표본에서만 판정 결과를 돌려준다.

        Args:
            t: 표본 시각 [s]. 단조여야 한다 — 뒤로 가면 위상이 무너진다.
            q: 품질 Q̄ 표본. ``None`` (또는 NaN) 은 "측정 안 됨" — 세기만 하고 안 쓴다.
                게이트가 닫힌 프레임이 이렇게 들어온다.

        Returns:
            주기가 닫히지 않았으면 ``None``. 닫혔으면 ``q_plus`` / ``q_minus`` (반주기
            평균), ``gradient`` [품질/N], ``step_n`` [N], 갱신된 ``f_bar``,
            ``measurement_failed`` 와 진단용 ``n_plus`` / ``n_minus`` / ``period`` 를
            담은 dict. 측정 실패면 평균·경사는 ``None``, 걸음은 0, ``f_bar`` 는 그대로다.

        Raises:
            ValueError: ``t`` 가 직전 표본(또는 ``t0``)보다 앞이다.
        """
        t = float(t)
        if t < self._last_t:
            raise ValueError(
                f"t({t}) 가 직전 시각({self._last_t}) 보다 앞이다 — 시간은 단조여야 한다"
            )
        self._last_t = t

        result: Optional[Dict[str, object]] = None
        idx = self._period_index(t)
        if idx > self._period_idx:
            # 위상이 한 바퀴 돌았다. 지금까지 모은 주기를 닫고 새 주기를 연다.
            # 여러 주기를 건너뛴 표본이 와도 닫는 것은 모으고 있던 그 주기 하나다 —
            # 건너뛴 주기는 표본이 없으니 판정할 것도 없다.
            result = self._close_period()
            self._period_idx = idx
            self._plus = []
            self._minus = []

        self._n_seen += 1
        usable = q is not None and math.isfinite(q)
        if self._in_settle(t):
            self._n_discarded += 1
        elif usable:
            (self._plus if self.dither_sign(t) > 0.0 else self._minus).append(float(q))

        return result

    def _close_period(self) -> Dict[str, object]:
        """모인 반주기 표본으로 경사를 재고 한 걸음 옮긴다."""
        n_plus, n_minus = len(self._plus), len(self._minus)
        out: Dict[str, object] = {
            "period": self._period_idx,
            "n_plus": n_plus,
            "n_minus": n_minus,
            "q_plus": None,
            "q_minus": None,
            "gradient": None,
            "step_n": 0.0,
            "f_bar": self._f_bar,
            "measurement_failed": True,
        }
        if n_plus < self.min_samples_per_half or n_minus < self.min_samples_per_half:
            self._n_failed += 1
            return out

        q_plus = sum(self._plus) / n_plus
        q_minus = sum(self._minus) / n_minus
        # 동기 복조. 두 레벨의 차이를 레벨 간격(2a)으로 나눈 유한차분이다.
        gradient = (q_plus - q_minus) / (2.0 * self.amplitude)
        step = _clamp(self.gain * gradient, -self.step_max, self.step_max)
        self._f_bar = _clamp(
            self._f_bar + step, self.f_min + self.amplitude, self.f_max - self.amplitude
        )
        self._n_updates += 1

        out.update(
            q_plus=q_plus,
            q_minus=q_minus,
            gradient=gradient,
            step_n=step,
            f_bar=self._f_bar,
            measurement_failed=False,
        )
        return out

    # ---- 읽기 ---------------------------------------------------------------

    @property
    def f_bar(self) -> float:
        """디더 없는 설정값 [N]. policy 가 보는 것이고, 적응이 옮기는 것이다."""
        return self._f_bar

    @property
    def period_s(self) -> float:
        """디더 주기 [s]."""
        return self.period

    @property
    def amplitude_n(self) -> float:
        """디더 반폭 [N]."""
        return self.amplitude

    @property
    def n_updates(self) -> int:
        """경사를 믿고 판정한 주기 수. 걸음이 0 이어도 센다."""
        return self._n_updates

    @property
    def n_failed(self) -> int:
        """표본이 모자라 판정을 건너뛴 주기 수."""
        return self._n_failed

    @property
    def n_discarded(self) -> int:
        """완화 구간이라 버린 표본 수. 진단용."""
        return self._n_discarded

    def describe(self) -> str:
        """사람이 읽을 한 줄 요약. 감시 화면과 로그에 그대로 쓴다."""
        return (
            f"F̄ = {self._f_bar:.3f} N  ±{self.amplitude:.2f} @ {self.period:.1f} s  "
            f"갱신 {self._n_updates} · 실패 {self._n_failed}"
        )


def quality_degraded(q_ref: float, q_now: float, fraction: float = 0.25) -> bool:
    """전체 재탐색 트리거 — Stage 1 종료 시점 대비 ``fraction`` 이상 열화했는가.

    **Q̄_seg 에 쓴다.** Q̄_raw 는 접촉만 유지되면 힘에 대해 평탄해 이 판정에 안 걸린다
    (모듈 설명 참조). 지속 시간(probe.yaml ``retrigger_hold_s``, 3 s 🟡)은 여기 없다 —
    이 함수는 순간 판정이고, "3 s 동안 참" 을 세는 것은 supervisor 의 일이다.

    Args:
        q_ref: 기준 품질 (Stage 1b 종료 시점의 Q̄_seg).
        q_now: 지금 품질.
        fraction: 열화 비율. 0.25 면 기준의 75 % 아래에서 참. 🟡

    Returns:
        ``q_now < (1 − fraction) · q_ref``. 경계값은 거짓이다 — 정확히 75 % 는 아직
        열화가 아니다.
    """
    return float(q_now) < (1.0 - float(fraction)) * float(q_ref)
