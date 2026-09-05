"""보상 전 렌치가 접촉 판정에 닿지 않는지 — 프레임 이름으로 거른다.

브리지는 ``/…/wrench_px6d`` 하나로 두 가지를 낸다: 교정이 실렸으면 중력보상된
프로브 프레임 값, 아니면 센서 프레임 원값. 값만 보면 둘은 구별되지 않는다.

예전에는 ``us_diff_ik`` 가 **다른 토픽**(``calibration_valid``)의 게이트로 그것을
막는다고 보았다. 두 토픽은 각자의 시각에 도착하므로 잠깐 어긋날 수 있고,
2026-09-04 02:15:09 에 어긋났다:

    09.851  교정 유효성: True                      ← 게이트 열림
    09.871  접촉 프로빙 전환 — F 10.52 N           ← 20 ms 뒤

같은 자세의 보상값은 0.71 N 이었다. 10.52 N 은 마운트·프로브 자중, 곧 큐에 남아
있던 보상 전 표본이다. 그 값은 한계 5.0 N 을 넘으므로 조절기의 첫 분기가 강제
후퇴다 — 아무것도 닿지 않은 세션 시작에 로봇이 움직였고, 단방향 걸쇠
(``has_contacted``)까지 남았다.

프레임 이름은 그 메시지와 함께 온다. 토픽 사이의 시간차가 끼어들 자리가 없다.
"""
from fr5_ik.us_diff_ik_node import UsDiffIkNode
from geometry_msgs.msg import WrenchStamped
import pytest

RAW_FRAME = "fr5_right_ft_sensor"      # telemetry_bridge 의 _wrench_frame
COMPENSATED = "fr5_right" + UsDiffIkNode.COMPENSATED_FRAME_SUFFIX
DEAD_WEIGHT = 10.52                    # 그날 실제로 들어온 보상 전 크기 [N]
ENTER = 2.0                            # teleop.contact_probing_force_n


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
        return _Stamp(1.0)


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def warn(self, msg, **kw):
        self.lines.append(msg)

    error = info = warn


class _Switch:
    """문턱 하나짜리 모드 스위치. 확인 창은 이 시험의 관심사가 아니다."""

    def __init__(self) -> None:
        self.in_contact_probing = False
        self.seen: list[float] = []

    def update(self, force, dt):
        self.seen.append(force)
        if force >= ENTER:
            self.in_contact_probing = True


class _Node:
    # 진짜 클래스의 상수를 그대로 쓴다. 여기에 문자열을 다시 적으면 이름이 바뀌어도
    # 시험만 통과한다.
    COMPENSATED_FRAME_SUFFIX = UsDiffIkNode.COMPENSATED_FRAME_SUFFIX

    def __init__(self) -> None:
        self.wrench_stamp = None
        self.normal_sign = -1.0
        self.normal_force = 0.0
        self.contact_force_mag = 0.0
        self.contact_probing_enabled = True
        self.require_calibration = True
        self.calibration_valid = True          # 게이트는 열려 있다
        self._mode_gate_warned = False
        self._raw_wrench_warned = False
        self.mode_switch = _Switch()
        self.entered = 0
        self.log = _Logger()

    def get_clock(self):
        return _Clock()

    def get_logger(self):
        return self.log

    def _control_force(self) -> float:
        return self.contact_force_mag

    def _enter_contact_probing(self):
        self.entered += 1

    def _leave_contact_probing(self):
        pass

    def feed(self, magnitude, frame):
        msg = WrenchStamped()
        msg.header.frame_id = frame
        msg.wrench.force.z = -float(magnitude)   # 압축은 −z
        UsDiffIkNode._on_wrench(self, msg)


def test_the_uncompensated_sample_from_that_night_is_dropped():
    """10.52 N 이 센서 프레임으로 오면 접촉 전환이 일어나지 않는다."""
    node = _Node()
    node.feed(DEAD_WEIGHT, RAW_FRAME)
    assert node.entered == 0
    assert node.mode_switch.seen == []          # 스위치까지 가지도 않는다
    assert not node.mode_switch.in_contact_probing


def test_the_same_number_compensated_would_transition():
    """거르는 것이 값이 아니라 **출처** 임을 확인한다.

    같은 10.52 N 이라도 보상된 프레임으로 오면 그것은 진짜 접촉이고, 막으면 안 된다.
    """
    node = _Node()
    node.feed(DEAD_WEIGHT, COMPENSATED)
    assert node.entered == 1
    assert node.mode_switch.in_contact_probing


def test_a_quiet_compensated_reading_passes_through_without_contact():
    """그날 같은 자세의 보상값 0.71 N — 통과하되 접촉은 아니다."""
    node = _Node()
    node.feed(0.706, COMPENSATED)
    assert node.entered == 0
    assert node.mode_switch.seen == [pytest.approx(0.706)]


def test_the_queued_raw_sample_cannot_slip_in_when_the_gate_opens():
    """그날의 순서 그대로: 게이트가 열린 직후 큐에 남은 원값이 들어온다."""
    node = _Node()
    node.calibration_valid = True              # 방금 열렸다
    node.feed(DEAD_WEIGHT, RAW_FRAME)          # 큐에 남아 있던 보상 전 표본
    node.feed(0.706, COMPENSATED)              # 그 뒤 정상 표본
    assert node.entered == 0
    assert node.mode_switch.seen == [pytest.approx(0.706)]


def test_it_says_so_once_not_every_sample():
    """센서는 1 kHz 다. 버릴 때마다 찍으면 그 자체로 로그가 막힌다."""
    node = _Node()
    for _ in range(500):
        node.feed(DEAD_WEIGHT, RAW_FRAME)
    assert len(node.log.lines) == 1
    assert "보상 전" in node.log.lines[0]


def test_it_speaks_again_after_recovering():
    """보상이 실렸다가 다시 끊기면 새 사건이다."""
    node = _Node()
    node.feed(DEAD_WEIGHT, RAW_FRAME)
    node.feed(0.706, COMPENSATED)              # 복구
    node.feed(DEAD_WEIGHT, RAW_FRAME)          # 다시 끊김
    assert len(node.log.lines) == 2


def test_the_calibration_gate_is_still_there_underneath():
    """프레임 검사가 교정 게이트를 대신하지 않는다 — 둘 다 있어야 한다."""
    node = _Node()
    node.calibration_valid = False
    node.feed(0.706, COMPENSATED)              # 프레임은 맞지만 교정이 무효다
    assert node.mode_switch.seen == []
    assert node.entered == 0
