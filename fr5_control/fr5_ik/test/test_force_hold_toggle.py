"""위약 대조군 — 힘 유지를 끈 채로도 안전층은 살아 있는가.

``contact_control.force_hold_enabled=false`` 는 교란 시험의 반쪽을 만든다. 같은
주사기 조작을 제어 없이 한 번 더 받아야 "힘이 목표 근처에 머물렀다" 가 제어 덕분인지
교란이 원래 그 정도였는지 갈린다. 지금까지 그 반사실은 팬텀 강성으로 외삽했고
(``fh.analysis.regulation_evidence``), 외삽인 만큼만 믿을 수 있었다.

여기서 고정하는 것은 그 스위치가 **무엇을 끄고 무엇을 끄지 않는가** 다.

  끈다   z 조절. 프로브는 접촉 자세 그대로 서고 힘은 팬텀이 정한다.
  안 끈다 한계 힘 후퇴. 물을 넣는 것은 조작자가 아니라 주사기이고, 대조군이라고
          해서 팬텀이 프로브를 5 N 으로 밀어 올릴 권리가 생기지는 않는다.

두 번째 줄이 이 파일의 이유다. "제어를 끈다" 를 "안전을 끈다" 로 구현하는 것은
한 줄 차이이고, 그 한 줄은 실물에서만 드러난다.
"""
from fr5_ik.force_regulator import ForceRegulator
from fr5_ik.us_diff_ik_node import UsDiffIkNode
import numpy as np
import pytest

TARGET = 3.0
BAND = 0.05
B_Z = 3000.0
WARN = 4.5
MAX = 5.0
RETREAT = 0.005


class _Stamp:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __sub__(self, other: "_Stamp") -> "_Stamp":
        return _Stamp(self.seconds - other.seconds)

    @property
    def nanoseconds(self) -> float:
        return self.seconds * 1e9


class _Clock:
    def now(self) -> _Stamp:
        return _Stamp(0.0)


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def error(self, msg, **kw):
        self.lines.append(msg)

    def warn(self, msg, **kw):
        self.lines.append(msg)

    def info(self, msg, **kw):
        self.lines.append(msg)


class _Switch:
    in_contact_probing = True


class _Node:
    """``_apply_force_regulation`` 이 실제로 만지는 상태만 가진 껍데기."""

    def __init__(self, force: float, hold: bool) -> None:
        self.mode_switch = _Switch()
        self.contact_probing_enabled = True
        self.require_calibration = True
        self.calibration_valid = True
        self._calibration_blocked_announced = False
        self.wrench_stamp = _Stamp(0.0)
        self.allow_teleop_lateral = False
        self.allow_inplane_rotation = False
        self.force_hold_enabled = hold
        self.force = force
        self.regulator_reason = ""
        self._last_v_z = 0.0
        self.log = _Logger()
        self.regulator = ForceRegulator(
            target_force_n=TARGET, deadband_n=BAND, admittance_b_z=B_Z,
            max_speed_m_s=0.01, warn_force_n=WARN, max_force_n=MAX,
            retreat_speed_m_s=RETREAT,
        )

    def get_clock(self):
        return _Clock()

    def get_logger(self):
        return self.log

    def _control_force(self) -> float:
        return self.force

    def regulate(self):
        return UsDiffIkNode._apply_force_regulation(self, np.zeros(6))


def v_z(node) -> float:
    return float(node.regulate()[2])


def test_with_hold_on_it_presses_toward_the_target():
    """기준선. 켜져 있으면 오차만큼 전진한다."""
    node = _Node(force=TARGET - 1.0, hold=True)
    assert v_z(node) == pytest.approx(1.0 / B_Z)


def test_with_hold_off_it_does_not_move():
    """같은 오차, 같은 접촉 — 그런데 z 를 놓는다. 이것이 대조군이다."""
    node = _Node(force=TARGET - 1.0, hold=False)
    assert v_z(node) == 0.0


def test_with_hold_off_it_does_not_retreat_either():
    """힘이 목표보다 높아도 물러나지 않는다. 놓는다는 것은 양쪽 다 놓는 것이다."""
    node = _Node(force=TARGET + 1.0, hold=False)
    assert v_z(node) == 0.0
    # 켜져 있었다면 물러났을 자리다 — 대조가 성립하는지 함께 확인한다.
    assert v_z(_Node(force=TARGET + 1.0, hold=True)) < 0.0


@pytest.mark.parametrize("force", [MAX, MAX + 0.5, MAX * 2])
def test_the_hard_limit_still_retreats_with_hold_off(force):
    """**대조군에서도 한계는 살아 있다.**

    물을 넣는 것은 주사기다. 조작자가 손을 대지 않아도 힘은 오를 수 있고, 그때
    한계까지 놓아 버리면 대조군이 아니라 사고다.
    """
    node = _Node(force=force, hold=False)
    assert v_z(node) == pytest.approx(-RETREAT)


def test_below_the_limit_nothing_moves_but_at_the_limit_it_does():
    """경계가 한계 힘에 정확히 걸려 있는지 — 한 칸 아래는 정지, 한계는 후퇴."""
    assert v_z(_Node(force=MAX - 0.01, hold=False)) == 0.0
    assert v_z(_Node(force=MAX, hold=False)) == pytest.approx(-RETREAT)


def test_the_reason_says_it_is_the_control_arm():
    """캡처가 남기는 사유 문자열로 두 팔이 구별돼야 한다."""
    node = _Node(force=TARGET - 1.0, hold=False)
    node.regulate()
    assert "대조군" in node.regulator_reason


def test_hold_off_does_not_disable_the_calibration_gate():
    """교정이 무효면 대조군이라도 힘 축을 열지 않는다.

    대조군은 "제어를 안 한다" 이지 "검사를 건너뛴다" 가 아니다.
    """
    node = _Node(force=TARGET, hold=False)
    node.calibration_valid = False
    assert np.allclose(node.regulate(), np.zeros(6))
    assert "교정" in node.regulator_reason


def test_hold_off_still_stops_when_the_wrench_dies():
    """힘을 모르면 한계 판정도 못 한다 — 그때는 대조군도 정지다."""
    node = _Node(force=TARGET, hold=False)
    node.wrench_stamp = _Stamp(-10.0)   # 10 초 묵었다
    assert np.allclose(node.regulate(), np.zeros(6))
    assert "wrench" in node.regulator_reason
