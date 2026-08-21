"""지령 정형기 단위 시험 — 특히 정지 거동 (2026-08-21).

여기서 잡으려는 증상은 실로봇 앞에서 눈으로 찾기 어려운 종류다. "시간이 좀 지나면
손목 조작이 뻣뻣해진다" 는 한 주기를 봐서는 안 보이고, 수십 번의 시작·정지가 쌓여야
드러난다. 그래서 서보 지연을 흉내내는 1차 지연 모형 위에서 초 단위로 돌린다.

모형: 컨트롤러는 지령을 시정수 ``LAG_S`` 로 따라간다. probe.yaml 이 적어 둔
"서보 지연 ~20 ms" 와 같은 값이며, 그 지연이 곧 지령 선행분의 원천이다.
"""
import math

import pytest

from fr5_control.joint_command_limiter import (
    IDLE_VEL_EPS_RAD_S,
    RESYNC_DEADBAND_DEG,
    JointCommandLimiter,
)

DT = 0.008                 # 125 Hz
LAG_S = 0.02               # 서보 추종 시정수
MAX_VEL = 1.5              # rad/s — 자유공간 상한
MAX_ACCEL = 8.0
MAX_DECEL = 24.0
FOLLOW_BAND = 5.0          # 도
START = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
LOWER = [-175.0, -265.0, -160.0, -265.0, -175.0, -165.0]
UPPER = [175.0, 85.0, 160.0, 85.0, 175.0, 165.0]

WRIST = 4                  # J5 — teleop 에서 가장 빠르게 도는 축


def make_limiter(**overrides) -> JointCommandLimiter:
    kwargs = dict(
        start_deg=list(START),
        max_vel=MAX_VEL,
        max_accel=MAX_ACCEL,
        max_decel=MAX_DECEL,
        max_follow_error=FOLLOW_BAND,
        limits_lower=list(LOWER),
        limits_upper=list(UPPER),
        resync_rate_deg_s=90.0,
        resync_gain=12.0,
    )
    kwargs.update(overrides)
    return JointCommandLimiter(**kwargs)


class LaggingRobot:
    """지령을 1차 지연으로 따라가는 컨트롤러 흉내."""

    def __init__(self, start_deg, lag_s: float = LAG_S) -> None:
        self.actual = [float(v) for v in start_deg]
        self.lag_s = lag_s

    def follow(self, commanded_deg, dt: float) -> list[float]:
        alpha = min(1.0, dt / self.lag_s)
        self.actual = [a + (c - a) * alpha for a, c in zip(self.actual, commanded_deg)]
        return list(self.actual)


def run(limiter, robot, target_vel_rad, seconds: float, dt: float = DT):
    """``seconds`` 동안 같은 관절속도를 넣고 돌린다. 마지막 실제 관절각을 돌려준다."""
    actual = list(robot.actual)
    for _ in range(int(round(seconds / dt))):
        commanded = limiter.step(target_vel_rad, actual, dt)
        actual = robot.follow(commanded, dt)
    return actual


def vel(joint: int, value: float) -> list[float]:
    out = [0.0] * 6
    out[joint] = value
    return out


# -- 정지 거동 ------------------------------------------------------------

def test_stop_cancels_the_unexecuted_lead():
    """손을 멈추면 남은 선행분이 실행되지 않고 취소된다.

    이것이 이번 수정의 본체다. 되감기가 없으면 지령이 실제보다 앞선 만큼을 로봇이
    나중에 다 실행한다 — 조작자에게는 "멈췄는데 더 간다" 로 나타난다.
    """
    limiter = make_limiter()
    robot = LaggingRobot(START)

    run(limiter, robot, vel(WRIST, MAX_VEL), 1.0)
    at_release = robot.actual[WRIST]
    lead_at_release = limiter.commanded_deg[WRIST] - at_release
    assert lead_at_release > 1.0, "전속 주행 중이면 선행분이 있어야 시험이 성립한다"

    run(limiter, robot, [0.0] * 6, 0.5)
    coast = robot.actual[WRIST] - at_release

    # 정지 후 남은 선행분은 0 으로 붙는다 — 다음 조작이 헛돌 여지가 없다.
    assert abs(limiter.commanded_deg[WRIST] - robot.actual[WRIST]) < 0.1
    # 같은 모형에서 이전 거동(감속=가속, 되감기 없음)은 8.7° 였다.
    # 감속 상한 분리로 감속 램프가 187 ms → 62 ms 로 줄어든 몫이 대부분이다.
    assert coast < 4.0, f"정지 후 {coast:.2f}° 더 갔다"


def test_resync_pulls_the_command_back_when_the_robot_stalls():
    """로봇이 멈춰 서 있어도 지령이 실제 관절각으로 되돌아온다.

    되감기의 핵심 성질이다. 이전 코드에는 지령을 실제값 쪽으로 되돌리는 경로가
    아예 없어서, 로봇이 못 따라온 만큼(추종오차 밴드 폭)이 그대로 남았다. 남은
    선행분은 다음 조작에서 먼저 소진되므로 "조작이 한 박자 늦게 먹는다" 가 된다.
    """
    limiter = make_limiter()
    robot = LaggingRobot(START)
    run(limiter, robot, vel(WRIST, MAX_VEL), 1.0)

    stalled = list(robot.actual)                # 이 시점부터 로봇은 움직이지 않는다
    lead_before = limiter.commanded_deg[WRIST] - stalled[WRIST]
    assert lead_before > 1.0

    for _ in range(int(1.0 / DT)):
        limiter.step([0.0] * 6, stalled, DT)

    assert limiter.commanded_deg[WRIST] == pytest.approx(stalled[WRIST], abs=RESYNC_DEADBAND_DEG)


def test_short_pauses_do_not_leave_lead_behind():
    """짧은 정지를 반복해도 선행분이 남지 않는다.

    "초반에는 잘 되다가 시간이 좀 지나면 뻣뻣해진다" 는, 미세조정하며 짧게 끊어
    움직이는 구간에서 정지마다 선행분이 덜 소진되어 쌓이는 모습이다. 여기서는
    컨트롤러를 일부러 느리게(시정수 60 ms) 두어 그 조건을 만든다.
    """
    def residual_leads(**overrides):
        limiter = make_limiter(**overrides)
        robot = LaggingRobot(START, lag_s=0.06)
        out = []
        for cycle in range(40):
            direction = 1.0 if cycle % 2 == 0 else -1.0
            run(limiter, robot, vel(WRIST, direction * MAX_VEL), 0.2)
            run(limiter, robot, [0.0] * 6, 0.1)
            out.append(abs(limiter.commanded_deg[WRIST] - robot.actual[WRIST]))
        return out

    now = residual_leads()
    # 되감기를 끈 것이 이전 거동이다.
    before = residual_leads(max_decel=MAX_ACCEL, resync_rate_deg_s=1e-9)

    assert max(now) < 1.5, f"정지 시점 선행분이 남는다: {max(now):.2f}°"
    assert max(now) < 0.6 * max(before)
    # 40 회를 돌아도 첫 회와 다르지 않다 — 누적되지 않는다.
    assert max(now) - min(now) < 0.2


def test_resync_never_exceeds_joint_velocity_limit():
    """되감기도 로봇이 내는 운동이다 — 속도 상한을 넘으면 컨트롤러가 거부한다."""
    limiter = make_limiter(resync_rate_deg_s=10_000.0)
    assert limiter.resync_rate_deg_s == pytest.approx(math.degrees(MAX_VEL))

    robot = LaggingRobot(START)
    run(limiter, robot, vel(WRIST, MAX_VEL), 0.5)

    previous = list(limiter.commanded_deg)
    actual = list(robot.actual)
    for _ in range(80):
        commanded = limiter.step([0.0] * 6, actual, DT)
        for before, after in zip(previous, commanded):
            assert abs(after - before) <= math.degrees(MAX_VEL) * DT + 1e-9
        previous = list(commanded)
        actual = robot.follow(commanded, DT)


def test_resync_is_gentle_when_robot_is_already_synced():
    """지령과 실제가 이미 붙어 있으면 되감기는 아무것도 하지 않는다."""
    limiter = make_limiter()
    actual = list(START)
    for _ in range(200):
        commanded = limiter.step([0.0] * 6, actual, DT)
        assert commanded == pytest.approx(START, abs=1e-9)


# -- 속도 램프 ------------------------------------------------------------

def test_deceleration_uses_the_faster_limit():
    """0 을 향해 크기만 줄어드는 변화는 감속 상한을 쓴다."""
    limiter = make_limiter()
    robot = LaggingRobot(START)
    actual = run(limiter, robot, vel(WRIST, MAX_VEL), 1.6)   # 전속까지 올린다
    assert limiter.applied_vel_rad[WRIST] == pytest.approx(MAX_VEL, abs=1e-6)

    limiter.step([0.0] * 6, actual, DT)
    dropped = MAX_VEL - limiter.applied_vel_rad[WRIST]
    assert dropped == pytest.approx(MAX_DECEL * DT, rel=1e-6)


def test_reversal_still_uses_the_acceleration_limit():
    """부호가 바뀌는 변화는 감속이 아니다 — 오류 29 를 부른 그 경우다."""
    limiter = make_limiter()
    robot = LaggingRobot(START)
    actual = run(limiter, robot, vel(WRIST, MAX_VEL), 1.6)

    # +1.5 에서 -1.5 로 뒤집는다. 첫 주기의 변화량은 가속 상한을 넘지 않아야 한다.
    before = limiter.applied_vel_rad[WRIST]
    limiter.step(vel(WRIST, -MAX_VEL), actual, DT)
    # 0 까지는 감속 구간이다. 중요한 것은 한 주기 안에 부호가 뒤집히지 않는 것.
    assert before - limiter.applied_vel_rad[WRIST] == pytest.approx(MAX_DECEL * DT, rel=1e-6)
    assert limiter.applied_vel_rad[WRIST] > 0.0

    # 0 을 밟고 넘어간 뒤부터는 가속 상한을 받는다.
    while limiter.applied_vel_rad[WRIST] > 0.0:
        limiter.step(vel(WRIST, -MAX_VEL), actual, DT)
    assert limiter.applied_vel_rad[WRIST] == pytest.approx(0.0, abs=1e-12)
    limiter.step(vel(WRIST, -MAX_VEL), actual, DT)
    assert limiter.applied_vel_rad[WRIST] == pytest.approx(-MAX_ACCEL * DT, rel=1e-6)


def test_velocity_magnitude_is_clamped():
    limiter = make_limiter()
    run(limiter, LaggingRobot(START), vel(WRIST, 10.0), 4.0)
    assert abs(limiter.applied_vel_rad[WRIST]) <= MAX_VEL + 1e-9


def test_decel_below_accel_is_rejected_by_construction():
    """감속 상한을 가속보다 낮게 주면 가속 값으로 올려 쓴다 (노드는 아예 거부한다)."""
    limiter = make_limiter(max_decel=1.0)
    assert limiter.max_decel == pytest.approx(MAX_ACCEL)


# -- 기존 보호장치가 그대로인지 --------------------------------------------

def test_follow_error_band_still_bounds_the_lead():
    """로봇이 아예 안 따라와도 지령은 밴드 밖으로 못 나간다."""
    limiter = make_limiter()
    frozen = list(START)                        # 상태가 얼어붙은 로봇
    for _ in range(2000):
        limiter.step(vel(WRIST, MAX_VEL), frozen, DT)
    assert limiter.commanded_deg[WRIST] - frozen[WRIST] == pytest.approx(FOLLOW_BAND, abs=1e-9)


def test_joint_limits_are_respected():
    limiter = make_limiter()
    robot = LaggingRobot(START)
    run(limiter, robot, vel(5, MAX_VEL), 8.0)
    assert limiter.commanded_deg[5] <= UPPER[5] + 1e-9


def test_hard_reset_drops_velocity_and_lead():
    limiter = make_limiter()
    robot = LaggingRobot(START)
    run(limiter, robot, vel(WRIST, MAX_VEL), 0.5)

    limiter.hard_reset(robot.actual)
    assert limiter.commanded_deg == pytest.approx(robot.actual)
    assert limiter.applied_vel_rad == [0.0] * 6
    assert limiter.idle


def test_idle_epsilon_ignores_numerical_dust():
    """IK 가 정확히 0 을 못 내도 정지로 인식해야 한다."""
    limiter = make_limiter()
    robot = LaggingRobot(START)
    run(limiter, robot, vel(WRIST, MAX_VEL), 0.5)
    run(limiter, robot, [IDLE_VEL_EPS_RAD_S / 2] * 6, 0.5)
    assert limiter.idle
    assert abs(limiter.commanded_deg[WRIST] - robot.actual[WRIST]) < 0.1
