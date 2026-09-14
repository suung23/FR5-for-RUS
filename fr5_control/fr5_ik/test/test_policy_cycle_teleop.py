"""추론 → 정지 → teleop 을 **여러 번** 반복해도 teleop 이 같은가.

지금의 평가 루프는 한 세션 안에서 Start/Stop 을 반복한다 (run_policy --loop, 그리고
조작자가 자세를 고쳐 잡고 다시 누르는 경우). 그때 제어 스택이 직전 사이클의 부스러기를
들고 다음 사이클로 넘어가면, 조작자에게는 **같은 손동작에 로봇이 다르게 반응하는 것**
으로만 보인다. 무엇이 달라졌는지는 화면 어디에도 없다 — 파라미터는 그대로이고, 로그는
"정책 정지" 한 줄이다.

여기서 고정하는 것은 두 가지다.

1. **Stop 뒤의 상태가 매번 같다.** 사이클 1 이 끝난 자리와 사이클 5 가 끝난 자리가
   비트 단위로 같아야 한다. 속도 상한, 지령, 힘 감시, 조절기 이력, 한 번만 찍는 표시.
2. **Start 가 직전 Stop 의 동작을 물려받지 않는다.** 이탈 후퇴는 제어 루프에서 힘
   조절보다 앞에 있으므로, 살아 있는 채로 Start 를 받으면 정책이 지령하는 동안
   로봇은 −z 로 물러난다.

rclpy 없이 돈다 — 노드를 만들지 않고 껍데기에 메서드를 붙여 부른다
(``test_force_hold_toggle`` 과 같은 규약).
"""
from fr5_ik.force_limit_override import ForceLimitOverride
from fr5_ik.force_regulator import ForceRegulator
from fr5_ik.probing_mode import APPROACH, CONTACT_PROBING, ProbingModeSwitch
from fr5_ik.release_retreat import ReleaseRetreat
from fr5_ik.us_diff_ik_node import UsDiffIkNode
import numpy as np
import pytest

APPROACH_LIN, APPROACH_ANG = 0.15, 1.5
CONTACT_LIN, CONTACT_ANG = 0.01, 0.2
#: 정책이 놓는 순간의 접촉력. 2026-09-12 마지막 에피소드의 실측이다.
PRESSED_N = 3.39


class _Stamp:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __sub__(self, other: "_Stamp") -> "_Stamp":
        return _Stamp(self.seconds - other.seconds)

    @property
    def nanoseconds(self) -> float:
        return self.seconds * 1e9


class _Clock:
    def __init__(self, node: "_Node") -> None:
        self.node = node

    def now(self) -> _Stamp:
        return _Stamp(self.node.t)


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def error(self, msg, **kw):
        self.lines.append(str(msg))

    warn = info = error


class _Pub:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def publish(self, msg):
        self.sent.append(getattr(msg, "data", msg))


class _Msg:
    def __init__(self, data) -> None:
        self.data = data


class _Node:
    """régime 경계가 실제로 만지는 상태만 가진 껍데기."""

    def __init__(self) -> None:
        self.t = 0.0
        self.log = _Logger()
        self.mode_pub = _Pub()
        self.force = 0.0

        self.mode_switch = ProbingModeSwitch(
            enter_force_n=2.0, confirm_s=0.02,
            release_force_n=0.3, release_confirm_s=0.5,
            # 지금의 운용값 — régime 을 정하는 것은 정책이고 힘 판정은 꺼져 있다.
            force_trigger_enabled=False,
        )
        self.approach_linear, self.approach_angular = APPROACH_LIN, APPROACH_ANG
        self.contact_linear, self.contact_angular = CONTACT_LIN, CONTACT_ANG
        self.max_linear, self.max_angular = APPROACH_LIN, APPROACH_ANG

        self.regulator = ForceRegulator(
            target_force_n=3.0, deadband_n=0.05, admittance_b_z=3000.0,
            max_speed_m_s=0.01, warn_force_n=4.5, max_force_n=5.0,
            retreat_speed_m_s=0.005,
        )
        self.override = ForceLimitOverride(
            limit_n=4.0, resume_n=3.0, warn_n=3.5,
            retreat_speed_m_s=0.005, max_override_s=5.0,
        )
        self.retreat_speed, self.retreat_force, self.max_retreat = 0.005, 0.2, 5.0
        self.release_retreat = ReleaseRetreat(
            speed_m_s=self.retreat_speed, until_force_n=self.retreat_force,
            max_s=self.max_retreat,
        )
        self._release_retreat_started = None
        self._release_retreat_announced = False

        self.twist_cmd = np.zeros(6)
        self.last_twist_time = _Stamp(0.0)
        self.retreat_started = None
        self._retreat_announced = False
        self._retreat_gave_up = False
        self.wrench_stamp = _Stamp(0.0)

        self.contact_force_mode = "magnitude"
        self.contact_force_mag = 0.0
        self.normal_force = 0.0
        self.allow_teleop_lateral = False
        self.allow_inplane_rotation = False
        self.policy_lateral = False
        self.force_hold_enabled = True
        self.regulator_reason = ""
        self._last_reason = None
        self._last_v_z = 0.0
        self._last_override_key = None
        self._force_hold_announced = False
        self._calibration_blocked_announced = False
        self._mode_gate_warned = False

    # -- 노드 흉내 ---------------------------------------------------------
    def get_clock(self):
        return _Clock(self)

    def get_logger(self):
        return self.log

    def _control_force(self) -> float:
        return self.force

    _publish_mode = UsDiffIkNode._publish_mode
    _forget_command = UsDiffIkNode._forget_command
    _cancel_retreats = UsDiffIkNode._cancel_retreats
    _enter_contact_probing = UsDiffIkNode._enter_contact_probing
    _leave_contact_probing = UsDiffIkNode._leave_contact_probing
    _on_policy_enable = UsDiffIkNode._on_policy_enable
    _release_retreat_step = UsDiffIkNode._release_retreat_step

    # -- 사이클 ------------------------------------------------------------
    def start(self, *, force: float = 0.5, command=None):
        """Start 를 누른다. ``command`` 는 그 직전 조작자가 내고 있던 지령."""
        self.force = force
        if command is not None:
            self.twist_cmd = np.asarray(command, dtype=float)
        self._on_policy_enable(_Msg(True))

    def stop(self, *, force: float = PRESSED_N, command=None):
        """Stop 을 누른다. ``command`` 는 정책이 마지막으로 낸 지령."""
        self.force = force
        if command is not None:
            self.twist_cmd = np.asarray(command, dtype=float)
        self._on_policy_enable(_Msg(False))

    def snapshot(self) -> dict:
        """teleop 구간의 거동을 정하는 값 전부. 사이클마다 같아야 한다."""
        return {
            "max_linear": self.max_linear,
            "max_angular": self.max_angular,
            "mode": self.mode_switch.mode,
            "policy_requested": self.mode_switch.policy_requested,
            "effective_mode": self.mode_switch.effective_mode,
            "confirm_progress": self.mode_switch.confirm_progress,
            "twist_cmd": tuple(self.twist_cmd),
            "override_state": self.override.state,
            "override_elapsed": self.override.elapsed_s,
            "regulator_moving": self.regulator._moving,
            "regulator_reason": self.regulator_reason,
            "retreat_started": self.retreat_started,
            "retreat_gave_up": self._retreat_gave_up,
            "release_active": self.release_retreat.active,
            "force_hold_announced": self._force_hold_announced,
            "calibration_blocked": self._calibration_blocked_announced,
            "mode_gate_warned": self._mode_gate_warned,
        }


def test_a_full_cycle_puts_teleop_back_where_it_started():
    """한 사이클. Stop 뒤의 상태가 Start 전과 같아야 한다.

    다른 것은 이탈 후퇴 하나뿐이다 — 그것은 남은 접촉을 푸는 **동작**이라 상태가
    아니라 진행 중인 일이다.
    """
    node = _Node()
    before = node.snapshot()
    node.start()
    node.t += 60.0
    node.stop()
    after = node.snapshot()
    assert after["release_active"] is True          # 눌린 채였으므로 물러난다
    del before["release_active"], after["release_active"]
    assert after == before


def test_five_cycles_land_on_the_same_place():
    """다섯 번 반복. 두 번째 자리와 다섯 번째 자리가 같아야 한다.

    사이클마다 조작자와 정책이 내는 지령이 다르고 힘도 다르다 — 실제 세션이 그렇다.
    그래도 Stop 뒤의 teleop 은 매번 같은 것을 받아야 한다.
    """
    node = _Node()
    seen = []
    for i in range(5):
        node.start(force=0.4 + 0.1 * i,
                   command=[0.001 * (i + 1), 0.0, 0.0, 0.0, 0.02, 0.0])
        node.t += 30.0
        node.stop(force=PRESSED_N, command=[0.0, 0.0, 0.0, 0.01 * (i + 1), 0.0, 0.0])
        node.t += 5.0
        # 다음 Start 전에 후퇴가 끝난 상태로 맞춘다 (실제로는 힘이 떨어지면 끝난다).
        node.release_retreat.cancel()
        node._release_retreat_started = None
        seen.append(node.snapshot())
    assert seen[1:] == seen[:-1]


def test_stop_restores_the_approach_speed_limits_every_time():
    """속도 상한. 접촉 상한에 묶인 채로 teleop 에 돌아가면 150 mm 에 15 초다."""
    node = _Node()
    for _ in range(3):
        node.start()
        assert (node.max_linear, node.max_angular) == (CONTACT_LIN, CONTACT_ANG)
        node.t += 10.0
        node.stop()
        assert (node.max_linear, node.max_angular) == (APPROACH_LIN, APPROACH_ANG)


def test_stop_does_not_inherit_the_last_policy_command():
    """정책이 마지막에 낸 지령이 조작자의 지령으로 읽히면 안 된다.

    이탈 후퇴는 "조작자가 지령을 내면 즉시 접는다" 인데, 그 판정이 보는 것이
    ``twist_cmd`` 다. 정책의 마지막 회전이 남아 있으면 첫 주기에 접히고 — 프로브는
    3.39 N 으로 눌린 채 teleop 에 넘어간다. 그것이 "Stop 뒤에 텔레옵이 안 된다" 다.
    """
    node = _Node()
    node.start()
    node.t += 20.0
    node.stop(command=[0.0, 0.0, 0.0, 0.0, 0.05, 0.0])   # 정책의 마지막 회전
    assert np.allclose(node.twist_cmd, 0.0)
    # 첫 주기가 후퇴로 나가는지 실제로 확인한다.
    node.t += 0.01
    step = node._release_retreat_step(node.get_clock().now())
    assert step is not None and step[2] == pytest.approx(-node.retreat_speed)


def test_the_operator_still_cancels_the_release_retreat():
    """지우는 것은 전 주인의 지령뿐이다. 조작자가 손을 쓰면 후퇴는 그대로 접힌다."""
    node = _Node()
    node.start()
    node.t += 20.0
    node.stop()
    node.twist_cmd = np.array([0.0, 0.0, 0.0, 0.0, 0.01, 0.0])   # 조작자가 비틀었다
    node.t += 0.01
    assert node._release_retreat_step(node.get_clock().now()) is None
    assert node.release_retreat.active is False


def test_stop_refreshes_the_command_clock():
    """시계도 함께 맞춘다. 정책 60 초 뒤의 Stop 이 곧바로 워치독 후퇴가 되면 안 된다."""
    node = _Node()
    node.start()
    node.t += 60.0
    node.stop()
    age = (node.get_clock().now() - node.last_twist_time).nanoseconds / 1e9
    assert age == pytest.approx(0.0)


def test_start_cancels_a_retreat_that_is_still_running():
    """Start 가 직전 Stop 의 후퇴를 물려받지 않는다.

    이탈 후퇴는 제어 루프에서 힘 조절보다 **앞**에 있다. 살아 있는 채로 다음 추론이
    시작되면 정책은 지령하는데 로봇은 −z 로 물러난다.
    """
    node = _Node()
    node.start()
    node.t += 20.0
    node.stop()
    assert node.release_retreat.active is True
    node.t += 0.2
    node.start()
    assert node.release_retreat.active is False
    assert node._release_retreat_started is None
    assert node.retreat_started is None


def test_the_regime_is_released_even_if_force_latched_it_first():
    """정책 전에 힘으로 걸린 régime 도 Stop 에서 풀린다.

    힘 판정이 꺼져 있으면 ``update`` 는 모드를 되돌리지 못한다. 놓는 순간이 그것을
    풀 수 있는 유일한 자리다 — 안 풀면 조작자의 여섯 축이 세션 끝까지 0 이다.
    """
    node = _Node()
    node.mode_switch.mode = CONTACT_PROBING      # 정책 전에 힘으로 걸렸다
    node.max_linear, node.max_angular = CONTACT_LIN, CONTACT_ANG
    node.start()
    node.t += 10.0
    node.stop()
    assert node.mode_switch.in_contact_probing is False
    assert node.mode_switch.effective_mode == APPROACH
    assert (node.max_linear, node.max_angular) == (APPROACH_LIN, APPROACH_ANG)


def test_the_force_window_does_not_survive_the_policy_run():
    """힘 판정의 확인 창을 들고 나가지 않는다.

    힘 판정을 켜 둔 운용에서, 정책 구간 내내 힘은 유지 목표(수 N)에 있다. 그 창을
    그대로 들고 teleop 에 나가면 다음 접촉은 **닿기도 전에** 접촉 상한으로 떨어진다.
    """
    switch = ProbingModeSwitch(enter_force_n=2.0, confirm_s=0.5,
                               release_force_n=0.3, release_confirm_s=0.5)
    switch.request_policy(True)
    switch.update(3.0, 0.4)                      # 진입 창이 0.4/0.5 까지 찼다
    assert switch.confirm_progress > 0.0
    switch.request_policy(False)
    assert switch.confirm_progress == 0.0
    # 다시 닿을 때는 처음부터 센다.
    assert switch.update(3.0, 0.4) == APPROACH
    assert switch.update(3.0, 0.2) == CONTACT_PROBING


def test_the_contact_latch_survives():
    """지우는 것은 régime 이지 사실이 아니다 — 닿았다는 기록은 남는다."""
    node = _Node()
    node.start()
    node.t += 10.0
    node.stop()
    assert node.mode_switch.has_contacted is True
