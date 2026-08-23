"""서보 지령 정형기 — 속도·가속도 상한, 관절 한계, 정지 시 지령 되감기.

``us_servo_node`` 안에 있던 지령 적분부를 떼어낸 것이다. 떼어낸 이유는 두 가지다.

1. **시험.** 이 로직의 버그는 전부 "시간이 지나야" 드러나는 종류다 (선행분 누적,
   정지 후 관성). 실로봇 앞에서 눈으로 찾을 것이 아니라 초 단위 시뮬레이션으로
   찾아야 한다. rclpy 를 걷어내면 그냥 pytest 로 돌릴 수 있다.
2. **경계.** 노드는 ROS 배선만, 여기는 "지령을 어떻게 깎을 것인가" 만 안다.

핵심은 ``commanded_deg`` 가 **개루프 적분값**이라는 점이다. 실제 관절각이 아니라
"지금까지 보낸 지령의 합"이며, 로봇은 이것을 지연을 두고 따라온다. 그 지연 동안
지령은 계속 앞서 나가므로, 손을 멈춘 시점에 **아직 실행되지 않은 각도**가 관절마다
남는다. 이것이 teleop 에서 "멈췄는데 더 간다 → 되돌리려면 그만큼 헛돈다" 로 나타나며,
빠르게 움직이는 손목축(J4~J6)에서 먼저 눈에 띈다.

이전 코드에는 이 선행분을 **없애는 경로가 없었다**. 추종오차 밴드
(``max_follow_error``)는 선행분의 크기를 묶을 뿐, 묶인 만큼은 결국 로봇이 실행한다.
그래서 정지할 때마다 밴드 폭만큼의 관성이 그대로 남았다.

여기서 두 가지를 더한다.

* **감속 상한 분리** (``max_decel``). 이전에는 가속과 감속이 같은 상한을 썼다.
  가속 상한은 특이점 근처 DLS 해의 **부호 반전**을 막으려고 넣은 것인데
  (probe.yaml §safety, 2026-08-20), 0 을 향해 크기가 줄기만 하는 변화에는 그 위험이
  없다. 같은 값으로 묶어 두면 정지에 그대로 가속 램프 시간이 든다.
* **정지 중 되감기** (``resync``). 정지 상태에서 목표속도를 0 이 아니라 "실제
  관절각으로 돌아가는 속도" 로 준다. 남은 선행분이 실행되는 대신 취소된다.
  0 을 주는 것과 달리 속도·가속도·관절한계를 전부 그대로 통과하므로, 되감기 자체가
  규격 밖 지령이 되는 일이 없다.
"""
from __future__ import annotations

import math

__all__ = ["JointCommandLimiter"]

#: 이보다 작은 관절속도 지령은 "정지" 로 본다 [rad/s].
#: 0.001 rad/s = 0.057 °/s — 1 분을 줘도 3.4° 다. 조작 의도로 볼 수 없는 크기다.
IDLE_VEL_EPS_RAD_S = 1e-3

#: 되감기 불감대 [도]. 이보다 가까우면 되감지 않는다.
#: 정지 중에 지령과 실제가 이 폭 안에서 서로를 쫓으며 떠는 것을 막는다.
RESYNC_DEADBAND_DEG = 0.05


def _clamp(value: float, low: float, high: float) -> float:
    return max(min(value, high), low)


class JointCommandLimiter:
    """관절속도 지령을 받아 ServoJ 로 보낼 관절각 지령을 만든다.

    Args:
        start_deg: 기동 시점의 실제 관절각 [도]. 지령 적분의 출발점이 된다.
        max_vel: 관절속도 상한 [rad/s].
        max_accel: 관절 가속도 상한 [rad/s^2]. 가속과 **방향 반전**에 적용된다.
        max_decel: 관절 감속 상한 [rad/s^2]. 부호를 바꾸지 않고 크기만 줄어드는
            변화에만 적용된다. ``max_accel`` 보다 크게 잡아 정지를 빠르게 한다.
        max_follow_error: 지령이 실제 관절각을 앞설 수 있는 최대치 [도].
        limits_lower: 관절 하한 6개 [도].
        limits_upper: 관절 상한 6개 [도].
        resync_rate_deg_s: 정지 중 되감기 속도의 상한 [도/s]. 되감기도 로봇이
            실제로 내는 운동이므로 ``max_vel`` 을 넘을 수 없다 — 넘겨 주면 내부에서
            잘린다.
        resync_gain: 되감기 P 게인 [1/s]. 되감기 속도는 ``gain × 선행분`` 이며
            ``resync_rate_deg_s`` 로 잘린다. 선행분이 줄면 속도도 함께 줄어
            지수적으로 붙는다 (시정수 ``1/gain``).
    """

    def __init__(
        self,
        start_deg,
        max_vel: float,
        max_accel: float,
        max_decel: float,
        max_follow_error: float,
        limits_lower,
        limits_upper,
        resync_rate_deg_s: float,
        resync_gain: float,
    ) -> None:
        self.commanded_deg = [float(v) for v in start_deg]
        self.applied_vel_rad = [0.0] * 6

        self.max_vel = float(max_vel)
        self.max_accel = float(max_accel)
        self.max_decel = max(float(max_decel), float(max_accel))
        self.max_follow_error = float(max_follow_error)
        self.limits_lower = [float(v) for v in limits_lower]
        self.limits_upper = [float(v) for v in limits_upper]

        # 되감기는 지령이지 텔레포트가 아니다. 속도 상한 위로 올라가면 컨트롤러가
        # 관절 초과(오류 29)로 거부한다 — 손으로 조작할 때와 똑같은 한계를 받는다.
        self.resync_rate_deg_s = min(float(resync_rate_deg_s), math.degrees(self.max_vel))
        self.resync_gain = float(resync_gain)

        self.idle = True
        #: 직전 정지 진입 시점에 남아 있던 선행분의 최대 크기 [도]. 진단용이다 —
        #: 이 값이 추종오차 밴드에 붙어 있으면 밴드가 포화했다는 뜻이고, 그때가
        #: 조작이 뻣뻣하게 느껴지는 구간이다.
        self.lead_at_stop_deg = 0.0

    # -- 상태 ------------------------------------------------------------

    def lead_deg(self, actual_deg) -> float:
        """지령이 실제 관절각을 앞선 정도의 최대치 [도]."""
        return max(abs(c - a) for c, a in zip(self.commanded_deg, actual_deg))

    def hard_reset(self, actual_deg) -> None:
        """지령 적분을 실제 관절각으로 되돌리고 속도를 버린다.

        서보를 흘리지 **않는** 동안에만 쓴다 (폴트 정지, 서보 재시작 직후). 서보가
        살아 있는 채로 부르면 그 자리에서 각도 계단이 나가고, 그것이 곧 급발진이다.
        정상 조작 중의 되감기는 :meth:`step` 이 속도로 처리한다.
        """
        self.commanded_deg = [float(v) for v in actual_deg]
        self.applied_vel_rad = [0.0] * 6
        self.idle = True
        self.lead_at_stop_deg = 0.0

    # -- 지령 정형 --------------------------------------------------------

    def _resync_velocity(self, actual_deg) -> list[float]:
        """정지 중에 쓸 목표속도 — 지령을 실제 관절각으로 되돌리는 방향 [rad/s]."""
        out = []
        for i in range(6):
            gap = actual_deg[i] - self.commanded_deg[i]
            if abs(gap) < RESYNC_DEADBAND_DEG:
                out.append(0.0)
                continue
            rate = _clamp(
                self.resync_gain * gap, -self.resync_rate_deg_s, self.resync_rate_deg_s
            )
            out.append(math.radians(rate))
        return out

    def step(self, target_vel_rad, actual_deg, dt: float) -> list[float]:
        """한 주기. 관절속도 지령을 받아 ServoJ 에 보낼 관절각을 돌려준다.

        Args:
            target_vel_rad: 상위(미분 IK)가 낸 관절속도 6개 [rad/s].
            actual_deg: 로봇이 보고한 실제 관절각 6개 [도].
            dt: 이번 주기 길이 [s].

        Returns:
            관절각 지령 6개 [도].
        """
        commanding = any(abs(v) > IDLE_VEL_EPS_RAD_S for v in target_vel_rad)

        if commanding:
            self.idle = False
            target = [_clamp(float(v), -self.max_vel, self.max_vel) for v in target_vel_rad]
        else:
            # 정지 진입 순간의 선행분을 기록해 둔다. 되감기가 이것을 없앤다.
            if not self.idle:
                self.lead_at_stop_deg = self.lead_deg(actual_deg)
                self.idle = True
            target = self._resync_velocity(actual_deg)

        dv_accel = self.max_accel * dt
        dv_decel = self.max_decel * dt

        for i in range(6):
            applied = self.applied_vel_rad[i]
            want = target[i]

            # **0 을 향해 가는 구간만** 감속이다. 목표가 반대 부호여도(되감기·방향
            # 반전) 0 까지 내려오는 동안은 크기가 줄기만 하므로 감속으로 친다.
            # 0 을 넘어선 뒤부터는 다시 가속이라 가속 상한을 받는다.
            #
            # 이렇게 나눠도 한 주기 변화량은 dv_decel 로 묶여 있다. 오류 29 를
            # 일으켰던 것은 "한 주기 안의 부호 반전"(+1.5 → -1.5, Δ3 rad/s)이었고,
            # 여기서는 0 에서 반드시 한 번 멈추므로 그 일이 일어나지 않는다.
            toward_zero = applied != 0.0 and (want - applied) * applied < 0.0
            dv_max = dv_decel if toward_zero else dv_accel

            new_applied = applied + _clamp(want - applied, -dv_max, dv_max)
            # 감속으로 0 을 지나치지 않는다. 넘어갈 몫은 다음 주기에 가속 상한을
            # 받아 나간다 — 방향 전환은 언제나 0 을 한 번 밟고 간다.
            if toward_zero and new_applied * applied < 0.0:
                new_applied = 0.0
            self.applied_vel_rad[i] = new_applied

            candidate = self.commanded_deg[i] + math.degrees(new_applied) * dt
            candidate = _clamp(candidate, self.limits_lower[i], self.limits_upper[i])

            # 추종오차 밴드. 지령이 실제에서 이 폭 이상 달아나지 못하게 묶는다.
            # 폭주 방지용이지 정지용이 아니다 — 정지는 위의 되감기가 한다.
            self.commanded_deg[i] = _clamp(
                candidate,
                actual_deg[i] - self.max_follow_error,
                actual_deg[i] + self.max_follow_error,
            )

        return list(self.commanded_deg)
