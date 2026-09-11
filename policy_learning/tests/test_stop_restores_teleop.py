"""GUI 의 Stop 이 teleop 을 **처음 상태로** 돌려주는가 — 러너 쪽.

제어 스택(us_diff_ik)은 policy_enable=false 에 régime 을 놓고 approach 로 돌아간다. 그런데
러너가 그 뒤에도 desired_twist 에 매 tick 0 을 내면 teleop 은 처음과 같지 않다: 쥐고 있으면
조작자 지령이 200 ms 마다 0 으로 덮이고, 놓으면 그 0 이 워치독 후퇴를 매번 푼다.

그리고 에피소드 모드에서 Stop 이 에피소드를 닫지 않으면 러너가 영영 떠 있어 드라이버가
다음 자세를 묻지 못한다.

ROS 없이 돈다 — 노드를 만들지 않고 메서드만 부른다.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_policy  # noqa: E402
from run_policy import PolicyRunner  # noqa: E402


class _Log:
    def warn(self, *_a, **_k): pass
    info = error = warn


class _Pub:
    def __init__(self):
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


@pytest.fixture
def runner(monkeypatch):
    # ROS 가 없으면 Twist 가 정의되지 않는다. 0 twist 는 무엇이든 세기만 하면 된다.
    monkeypatch.setattr(run_policy, "Twist", lambda: "zero", raising=False)
    r = object.__new__(PolicyRunner)
    r.get_logger = lambda: _Log()
    r.args = SimpleNamespace(execute=True, loop=False, duration=90.0,
                             enable_topic="/fr5_right/policy_enable",
                             robot_namespace="/fr5_right")
    r.twist_pub = _Pub()
    r.enabled = True
    r.retreating = False
    r.wrench = None
    r._uncompensated = ""
    r.t_start = None
    r.finished = False
    r.stopped_by_operator_s = None
    r._held = None
    r._idle_reason, r._idle_logged = "", 0.0
    return r


def _stop(r):
    r._on_enable(SimpleNamespace(data=False))


def test_stop_sends_one_zero_then_the_channel_is_touchs(runner):
    _stop(runner)
    assert runner.twist_pub.sent == ["zero"], "전환 순간의 0 한 번은 나가야 한다"
    for _ in range(25):                       # 5 s — 예전에는 여기서 25 번 0 이 나갔다
        runner._tick()
    assert runner.twist_pub.sent == ["zero"], (
        "Stop 뒤에도 desired_twist 에 내고 있다 — Touch 지령을 덮고 워치독 후퇴를 무른다")


def test_retreat_is_not_masked_by_zeros(runner):
    runner.retreating = True
    for _ in range(5):
        runner._tick()
    assert runner.twist_pub.sent == [], "후퇴 중에 0 을 내면 us_diff_ik 가 후퇴를 푼다"


def test_enabled_idle_still_holds_the_channel(runner):
    """켜져 있는데 기다리는 동안(접촉 부족 등)은 예전처럼 0 을 낸다 — 그때는 러너의 채널이다."""
    runner._tick()                            # wrench 가 없다 → 대기
    assert runner.twist_pub.sent == ["zero"]


def test_stop_closes_a_started_episode(runner):
    runner.t_start = run_policy.time.time() - 40.0
    _stop(runner)
    assert runner.finished, "Stop 이 에피소드를 닫지 않으면 드라이버가 다음 자세를 묻지 못한다"
    assert runner.stopped_by_operator_s == pytest.approx(40.0, abs=1.0)


def test_stop_before_start_keeps_waiting(runner):
    """아직 시작 전이면 닫지 않는다 — 다시 닿고 Start 하면 그 에피소드가 이어진다."""
    _stop(runner)
    assert not runner.finished and runner.stopped_by_operator_s is None


def test_unlimited_run_is_not_closed_by_stop(runner):
    """--duration 0 (옛 live 모드) 은 Start/Stop 을 오가며 한 프로세스로 쓴다."""
    runner.args.duration = 0.0
    runner.t_start = run_policy.time.time() - 10.0
    _stop(runner)
    assert not runner.finished
