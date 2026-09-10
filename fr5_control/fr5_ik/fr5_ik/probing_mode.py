"""접근 / 접촉 프로빙 모드 전환기 (DESIGN_NOTES §10.6, 2026-08-25).

teleop 한 세션은 두 단계로 나뉜다.

``approach``
    프로브가 아무것도 만지지 않은 단계. 초기 자세로 데려가는 것이 목적이므로
    자유공간 스케일(150 mm/s · 1.5 rad/s)을 쓴다.

``contact_probing``
    프로브가 조직·팬텀에 닿은 뒤. 접촉용 상한(10 mm/s · 0.2 rad/s)으로 내려간다.

전환은 **접촉력이 문턱을 넘는 순간** 일어난다. 조작자가 손으로 바꾸는 것이 아니다 —
접촉은 예고 없이 시작되고, 그 순간에 손이 파라미터를 만지고 있을 리 없다.

``joint_command_limiter`` 와 같은 이유로 rclpy 없이 돈다. 이 판정의 버그는 실기 앞이
아니라 pytest 로 찾아야 하는 종류다.

무엇이 문턱을 넘는가 — 2026-08-31 변경
--------------------------------------
예전에는 **법선력** ``F_n = sign x F_z`` 하나를 봤고 문턱이 8 N 이었다. 이제는
**접촉력 크기** ``‖F‖ = √(Fx²+Fy²+Fz²)`` (중력보상 후, 프로브 프레임) 를 보고
문턱이 한 자릿수 N 으로 내려왔다 (probe.yaml 현재 2.0). 무엇을 넣을지는 이 모듈이
아니라 호출자가 정한다 — 여기는 스칼라
하나를 받아 문턱과 비교할 뿐이다 (``us_diff_ik_node`` 의
``ft_sensor.contact_force_mode``).

바꾼 이유는 낮은 문턱과 함께 온다. 8 N 은 프로브가 어느 방향으로 닿든 F_z 에 충분히
실리는 크기였지만, 몇 N 짜리 문턱은 아니다. 프로브가 조금만 기울어 닿으면 접촉력의 상당 부분이
횡력(Fx, Fy)으로 가고 F_z 하나로는 문턱을 못 넘는다 — 즉 **닿았는데 접근 속도를
유지한다.** 크기는 방향에 무관하므로 그 구멍이 없다.

되돌림 — 2026-08-31 변경
------------------------
예전에는 전환이 **단방향** 이었다. 문턱이 8 N 이던 동안에는 맞는 선택이었다: 그 힘에
닿았다면 이미 작업 중이고, 프로브를 살짝 떼는 것은 작업의 일부이지 끝이 아니다.
그때마다 상한이 15 배로 뛰면 조작자가 같은 손동작을 하는 동안 로봇의 반응이 달라진다.

문턱이 내려오면서 그 논리가 뒤집혔다. 이 크기는 **스치기만 해도 넘는다.** 단방향이면
세션 초반의 우연한 접촉 한 번으로 남은 세션 전체가 10 mm/s 에 묶이고, 150 mm 를
옮기는 데 15 초가 걸린다. 게다가 접촉 프로빙에서는 조작자의 여섯 축이 모두 0 이므로
(``contact_control.allow_teleop_lateral: false``) **조작자가 스스로 빠져나올 수단이
없다** — 데드맨을 놓아 워치독 후퇴를 부르는 것이 유일한 길이고, 그 후퇴의 종료 조건이
바로 아래 이탈 문턱과 같은 자리(0.2~0.3 N)다.

그래서 이탈을 연다. 다만 ``fr5_control.contact_state`` 와 같은 규약을 쓴다:

* **이력.** 이탈 문턱은 진입보다 낮다. 하나로 두면 문턱 근처에서 모드가 떨리고,
  그 떨림이 곧 속도 상한의 on/off 라 로봇이 덜컥거린다.
* **이탈 확인 창이 진입보다 길다.** 누르는 중의 순간적인 힘 감소로 접근 속도가
  되살아나는 것이 이 판정에서 가장 위험한 오작동이다.
* **걸쇠는 남는다.** ``has_contacted`` 는 접촉이 한 번이라도 있었는지를 기억하며
  이탈해도 지워지지 않는다. 되돌리면 안 되는 판단(예: teleop 종결)은 모드가 아니라
  걸쇠를 봐야 한다.

``release_force_n=None`` 이면 예전 단방향 거동 그대로다. 8 N 문턱으로 되돌리는 검증
절차가 그 조합을 쓴다.
"""
from __future__ import annotations

__all__ = ["APPROACH", "CONTACT_PROBING", "ProbingModeSwitch"]

#: 자유공간 스케일. 초기 자세 접근용.
APPROACH = "approach"

#: 접촉용 상한. 프로브가 닿은 뒤.
CONTACT_PROBING = "contact_probing"


class ProbingModeSwitch:
    """접촉력으로 접근/접촉 프로빙 모드를 가른다.

    시간은 밖에서 받는다 (``dt_s``). 벽시계를 안 읽으므로 기록을 되돌려 그대로 다시
    판정할 수 있다.
    """

    def __init__(
        self,
        enter_force_n: float,
        confirm_s: float = 0.02,
        release_force_n: float | None = None,
        release_confirm_s: float = 0.5,
        force_trigger_enabled: bool = True,
    ) -> None:
        """전환기를 만든다.

        Args:
            enter_force_n: 접촉 프로빙으로 넘어가는 접촉력 [N]. 양수 = 누름.
            confirm_s: 연속으로 문턱을 넘어야 하는 시간 [s]. 0 이면 표본 하나로 전환.
            release_force_n: 접근으로 되돌아가는 접촉력 [N]. ``None`` 이면 되돌아가지
                않는다(예전 단방향 거동). 진입 문턱보다 **낮아야** 한다.
            release_confirm_s: 이탈을 확정하기까지 연속으로 아래에 머물러야 하는
                시간 [s]. 진입보다 길게 둔다.
            force_trigger_enabled: 접촉력으로 régime 을 가를 것인가. 거짓이면 ``update``
                가 힘을 보고도 모드를 바꾸지 않는다 — 정책 추론이 régime 을 정하는
                운용에서 쓴다 (2026-09-11). 문턱 값 자체는 지우지 않는다: 파라미터
                하나로 예전 거동으로 돌아갈 수 있어야 한다.

        Raises:
            ValueError: 문턱이 0 이하이거나, 확인 시간이 음수이거나, 이탈 문턱이
                진입 문턱보다 낮지 않다.
        """
        if enter_force_n <= 0.0:
            raise ValueError(f"enter_force_n 은 양수여야 한다: {enter_force_n}")
        if confirm_s < 0.0:
            raise ValueError(f"confirm_s 는 음수일 수 없다: {confirm_s}")
        if release_force_n is not None:
            if release_force_n >= enter_force_n:
                # 이력이 없으면 문턱 근처에서 모드가 떨린다. 그 떨림은 곧 속도
                # 상한의 on/off 이고, 조작자에게는 로봇이 덜컥거리는 것으로 나타난다.
                raise ValueError(
                    f"release_force_n({release_force_n}) 은 "
                    f"enter_force_n({enter_force_n}) 보다 낮아야 한다"
                )
            if release_confirm_s < 0.0:
                raise ValueError(
                    f"release_confirm_s 는 음수일 수 없다: {release_confirm_s}"
                )

        self.enter_force_n = float(enter_force_n)
        self.confirm_s = float(confirm_s)
        self.release_force_n = (
            None if release_force_n is None else float(release_force_n)
        )
        self.release_confirm_s = float(release_confirm_s)
        self.force_trigger_enabled = bool(force_trigger_enabled)
        #: 힘 판정이 내린 모드. 정책 요청은 여기 섞지 않는다 — 둘을 한 변수에 담으면
        #: "왜 접촉 régime 인가" 를 되짚을 수 없다.
        self.mode = APPROACH
        #: 정책 추론이 régime 을 요청했는가. 힘 판정과 **독립**이며 OR 로 합쳐진다.
        self.policy_requested = False
        #: 접촉이 **한 번이라도** 있었는가. 이탈해도 지워지지 않는다.
        self.has_contacted = False
        self._above_s = 0.0
        self._below_s = 0.0

    @property
    def reversible(self) -> bool:
        """이탈이 열려 있는가. 거짓이면 예전 단방향 거동이다."""
        return self.release_force_n is not None

    def update(self, contact_force_n: float, dt_s: float) -> str:
        """표본 하나를 넣고 현재 모드를 돌려준다.

        Args:
            contact_force_n: 접촉력 [N]. 무엇을 넣을지는 호출자가 정한다 — 크기
                ``‖F‖`` 이거나 법선력 ``F_n``. 양수 = 누름.
            dt_s: 직전 표본과의 간격 [s].

        Returns:
            :data:`APPROACH` 또는 :data:`CONTACT_PROBING`.
        """
        step = max(0.0, float(dt_s))
        if not self.force_trigger_enabled:
            # 힘으로는 가르지 않는다. 값을 세지도 않는다 — 꺼 둔 동안 쌓인 창이
            # 다시 켜는 순간 즉시 전환을 만들면, 켠 사람이 예상하지 못한 일이 된다.
            self._above_s = self._below_s = 0.0
            return self.mode

        if self.mode == APPROACH:
            if contact_force_n >= self.enter_force_n:
                self._above_s += step
                if self._above_s >= self.confirm_s:
                    self.mode = CONTACT_PROBING
                    self.has_contacted = True
                    self._below_s = 0.0
            else:
                # 문턱 아래로 한 번만 내려가도 처음부터 다시 센다. "대체로 넘었다" 는
                # 접촉이 아니다.
                self._above_s = 0.0
            return self.mode

        if self.release_force_n is None:
            return self.mode

        if contact_force_n <= self.release_force_n:
            self._below_s += step
            if self._below_s >= self.release_confirm_s:
                self.mode = APPROACH
                self._above_s = 0.0
        else:
            # 이탈도 연속이어야 한다. 누르는 중의 한 표본짜리 힘 감소로 접근 속도가
            # 되살아나면, 그것이 이 판정에서 가장 위험한 오작동이다.
            self._below_s = 0.0
        return self.mode

    def request_policy(self, enabled: bool) -> None:
        """정책 추론이 régime 을 잡는다 / 놓는다.

        힘 판정과 **독립**이다. 정책이 잡고 있는 동안 힘이 이탈 문턱 아래로 내려가도
        régime 은 유지된다 — 정책이 프로브를 들어 다시 찾는 동작이 régime 을 놓는
        것으로 읽히면, 그 순간 z 가 조작자에게 돌아가고 정책은 허공에 지령한다.
        """
        self.policy_requested = bool(enabled)
        if self.policy_requested:
            self.has_contacted = True

    @property
    def in_contact_probing(self) -> bool:
        """접촉 régime 인가 — 힘 판정 **또는** 정책 요청."""
        return self.mode == CONTACT_PROBING or self.policy_requested

    @property
    def effective_mode(self) -> str:
        """régime 을 하나의 문자열로. 무엇이 régime 을 잡았는지는 따로 본다."""
        return CONTACT_PROBING if self.in_contact_probing else APPROACH

    @property
    def confirm_progress(self) -> float:
        """지금 열려 있는 확인 창을 몇 % 지났는가. 0..1.

        접근에서는 진입 창, 접촉 프로빙에서는 이탈 창이다. 되돌아가지 않는 설정에서는
        접촉 프로빙에 들어선 순간 1.0 으로 남는다 — 기다릴 창이 없다.
        """
        if self.mode == CONTACT_PROBING:
            if self.release_force_n is None:
                return 1.0
            if self.release_confirm_s <= 0.0:
                return 1.0 if self._below_s > 0.0 else 0.0
            return min(1.0, self._below_s / self.release_confirm_s)
        if self.confirm_s <= 0.0:
            return 1.0 if self._above_s > 0.0 else 0.0
        return min(1.0, self._above_s / self.confirm_s)

    def reset(self, *, unlatch: bool = False) -> None:
        """접근 모드로 되돌린다. 새 세션을 시작할 때만 쓴다.

        Args:
            unlatch: 참이면 ``has_contacted`` 걸쇠까지 지운다. 기본은 거짓 — 걸쇠는
                "이 조립이 무언가에 닿은 적이 있다" 는 세션의 기록이고, 모드를
                되돌린다고 그 사실이 사라지지는 않는다.
        """
        self.mode = APPROACH
        self.policy_requested = False
        self._above_s = 0.0
        self._below_s = 0.0
        if unlatch:
            self.has_contacted = False
