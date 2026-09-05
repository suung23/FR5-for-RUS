"""후퇴 워치독이 **한 번만** 말하는지 고정한다.

twist 가 끊기면 us_diff_ik 는 프로브 −z 로 물러나고, 힘이 안 풀리면 시간 상한에서
포기하고 선다. 거기까지는 옳다 — 로봇은 정지해 있고 안전하다.

문제는 그 뒤였다. 포기 분기는 ``None`` 을 내지만 ``retreat_started`` 를 지우지
않으므로, twist 가 돌아올 때까지 **매 주기 같은 가지로 다시 들어온다.** 로그를
그 안에서 찍으면 100 Hz 로 같은 줄이 쏟아진다.

2026-09-04 에 그 대가를 치렀다. touch_twist 가 햅틱 장치를 못 잡고 죽었는데,
그 사실을 말하는 유일한 줄("No haptic devices found")이 몇 초 만에 수천 줄 밑으로
밀려났다. 화면에 남은 것은 로봇이 무엇을 못 하는지 반복하는 문장뿐이고, **왜**
그런지는 스크롤 밖에 있었다. 원인을 가리는 로그는 없느니만 못하다.

그래서 여기서 고정하는 것은 거동이 아니라 **말수**다. 후퇴 한 번에 시작 한 줄,
포기 한 줄. twist 가 돌아왔다 다시 끊기면 그때는 새 사건이므로 다시 말한다.
"""
from fr5_ik.us_diff_ik_node import UsDiffIkNode
import numpy as np
import pytest

MAX_RETREAT = 3.0     # probe.yaml watchdog.max_retreat_s
RETREAT_FORCE = 0.5   # 이 아래로 떨어지면 접촉이 풀린 것으로 본다 [N]
RETREAT_SPEED = 0.005
DT = 0.01             # 100 Hz. us_diff_ik 제어 주기다.


class _Stamp:
    """``rclpy.time.Time`` 대신. ``-`` 가 ``nanoseconds`` 를 가진 것을 내면 된다."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __sub__(self, other: "_Stamp") -> "_Stamp":
        return _Stamp(self.seconds - other.seconds)

    @property
    def nanoseconds(self) -> float:
        return self.seconds * 1e9


class _Logger:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warns: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warns.append(msg)


class _Node:
    """``_retreat_twist`` 가 실제로 만지는 상태만 가진 껍데기.

    노드를 통째로 세우면 rclpy·파라미터·퍼블리셔가 따라오고, 그러면 이 시험이
    재는 것이 로그 횟수인지 기동 절차인지 흐려진다.
    """

    def __init__(self, force: float) -> None:
        self.retreat_started = None
        self._retreat_announced = False
        self._retreat_gave_up = False
        self.wrench_stamp = _Stamp(0.0)
        self.max_retreat = MAX_RETREAT
        self.retreat_force = RETREAT_FORCE
        self.retreat_speed = RETREAT_SPEED
        self.force = force
        self.log = _Logger()

    def get_logger(self) -> _Logger:
        return self.log

    def _control_force(self) -> float:
        return self.force

    # 실제 노드의 메서드를 그대로 부른다. 복사하면 갈라진다.
    def retreat(self, now):
        return UsDiffIkNode._retreat_twist(self, now)


def _run(node, seconds, start=0.0):
    """제어 주기마다 후퇴를 부른다. 마지막 반환값을 준다."""
    out = None
    t = start
    while t < start + seconds:
        node.wrench_stamp = _Stamp(t)     # wrench 는 계속 신선하다
        out = node.retreat(_Stamp(t))
        t += DT
    return out


def test_gives_up_after_the_time_limit():
    """힘이 안 풀리면 상한에서 선다 — 거동은 그대로다."""
    node = _Node(force=RETREAT_FORCE * 4)   # 눌린 채 안 풀린다
    assert node.retreat(_Stamp(0.0)) is not None
    assert _run(node, MAX_RETREAT + 1.0) is None


def test_the_give_up_is_said_once_not_every_cycle():
    """상한 뒤 10 초를 더 돌아도 error 는 한 줄이다.

    이 시험이 없던 동안 여기서 1000 줄이 나왔다 (100 Hz × 10 s).
    """
    node = _Node(force=RETREAT_FORCE * 4)
    _run(node, MAX_RETREAT + 10.0)
    assert len(node.log.errors) == 1, f"{len(node.log.errors)} 줄이 나왔다"
    # 시작 알림도 한 번뿐이다. 같은 성질을 반대편에서 잡는다.
    assert len(node.log.warns) == 1


def test_the_message_points_upstream():
    """원인은 이 노드 밖에 있다. 어디를 볼지 문장이 말해야 한다."""
    node = _Node(force=RETREAT_FORCE * 4)
    _run(node, MAX_RETREAT + 1.0)
    assert "touch_twist" in node.log.errors[0]


def test_a_new_retreat_speaks_again():
    """twist 가 돌아왔다 다시 끊기면 새 사건이다 — 그때는 다시 말한다.

    한 번만 말하게 만드느라 **영원히** 입을 다물면, 두 번째 고장은 조용히 지나간다.
    """
    node = _Node(force=RETREAT_FORCE * 4)
    _run(node, MAX_RETREAT + 1.0)
    assert len(node.log.errors) == 1

    node.retreat_started = None            # 호출자가 twist 복귀 때 하는 일
    _run(node, MAX_RETREAT + 1.0, start=100.0)
    assert len(node.log.errors) == 2


def test_released_contact_ends_the_retreat_quietly():
    """힘이 풀리면 상한 전에 끝난다 — 그때는 error 가 없다."""
    node = _Node(force=RETREAT_FORCE / 2)  # 이미 풀려 있다
    assert node.retreat(_Stamp(0.0)) is None
    assert node.log.errors == []


def test_retreat_is_along_probe_minus_z():
    """후퇴 방향은 프로브 −z 다. 침투 방향의 반대."""
    node = _Node(force=RETREAT_FORCE * 4)
    twist = node.retreat(_Stamp(0.0))
    assert np.allclose(twist, [0.0, 0.0, -RETREAT_SPEED, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("fresh", [True, False])
def test_it_says_whether_the_wrench_is_alive(fresh):
    """wrench 까지 죽었는지 아닌지는 다음에 볼 곳을 가른다."""
    node = _Node(force=RETREAT_FORCE * 4)
    t = 0.0
    while t < MAX_RETREAT + 1.0:
        # 신선하지 않은 쪽은 wrench 시각을 붙잡아 둔다 (0.5 s 넘게 늙는다).
        node.wrench_stamp = _Stamp(t if fresh else 0.0)
        node.retreat(_Stamp(t))
        t += DT
    expected = "힘이 안 떨어진다" if fresh else "wrench 도 두절이다"
    assert expected in node.log.errors[0]
