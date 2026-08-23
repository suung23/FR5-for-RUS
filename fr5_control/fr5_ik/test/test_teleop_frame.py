"""병진 프레임 매핑 시험 — 표류가 실제로 사라지는지 (2026-08-21).

2026-08-21 실측에서 "스타일러스 +X 밀기" 방향이 한 세션에 10° → 104° 로 표류했다
(최대 135°, 데드맨 재파지로도 복구 안 됨). 여기서는 그 메커니즘을 그대로 재현한
모형 위에서, 기존 매핑은 표류하고 새 매핑은 표류하지 않음을 확인한다.

모형: 손이 스타일러스를 계속 비틀고, 프로브는 그 회전을 **불완전하게** 따라간다.
불완전함의 원천은 실제 경로와 같다 — angular_scale 1.2 배와, 느린 회전을 통째로
버리는 데드존(정류기). 재현에 필요한 것은 이 둘뿐이다.
"""
import math

import numpy as np
import pytest

from fr5_ik.teleop_frame import LinearFrameMapper, axis_mapping

TIP_ROLL = 90.0
A = axis_mapping(TIP_ROLL)

ANGULAR_SCALE = 1.2      # probe.yaml teleop.freespace.angular_scale
ANGULAR_DEADZONE = 0.05  # rad/s — 이 아래는 통째로 버려진다
DT = 0.02                # Touch 발행 주기 50 Hz


def rot_z(angle: float) -> np.ndarray:
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])


def rot_x(angle: float) -> np.ndarray:
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]])


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """두 방향 사이 각 [도]."""
    cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.degrees(math.acos(min(max(cosine, -1.0), 1.0)))


class Session:
    """손은 비틀고, 프로브는 불완전하게 따라오는 한 파지.

    ``step`` 한 번이 Touch 한 주기다. 손 각속도를 주면 스타일러스 자세는 그대로,
    프로브 자세는 데드존과 스케일을 통과한 만큼만 돈다.
    """

    def __init__(self) -> None:
        self.stylus_angle = 0.0
        self.probe_angle = 0.0

    @property
    def rot_stylus(self) -> np.ndarray:
        return rot_z(self.stylus_angle)

    @property
    def rot_base_probe(self) -> np.ndarray:
        return rot_x(0.3) @ rot_z(self.probe_angle)   # 임의의 초기 자세 + 누적 회전

    def step(self, hand_rate: float) -> None:
        self.stylus_angle += hand_rate * DT
        commanded = 0.0 if abs(hand_rate) < ANGULAR_DEADZONE else hand_rate * ANGULAR_SCALE
        self.probe_angle += commanded * DT


def hand_profile(seconds: float):
    """빠르게 비틀고 천천히 되돌리는 동작. 데드존이 되돌림만 잡아먹는다."""
    steps = int(seconds / DT)
    for i in range(steps):
        phase = (i * DT) % 1.5
        yield 0.6 if phase < 0.5 else -0.3 * (0.5 / 1.0)   # 되돌림은 0.15 < 데드존


PUSH_X = A @ np.array([1.0, 0.0, 0.0])   # 스타일러스 자세가 단위행렬일 때의 지령
PUSH_WORLD = np.array([1.0, 0.0, 0.0])   # 조작자가 "오른쪽으로 민다" 는 의도


def replay(seconds: float):
    """같은 world 의도를 계속 밀면서, 옛 경로와 새 경로가 각각 어디로 가는지 본다.

    teleop 은 매 주기 ``v_cmd = A · R_stylusᵀ · v_world`` 를 발행한다. 손 자세가 바뀌면
    발행되는 지령도 함께 바뀌므로, "의도가 같다" 를 보려면 지령을 매 주기 다시 만들어야
    한다. 처음 이 시험을 지령 고정으로 썼다가 틀렸다 — 그러면 손 의도 자체가 도는
    상황을 재는 것이 된다.

    Returns:
        ``(옛 경로 최대 표류°, 새 경로 최대 표류°)``.
    """
    session = Session()
    mapper = LinearFrameMapper(TIP_ROLL)
    mapper.engage(session.rot_base_probe, session.rot_stylus)

    def command():
        return A @ session.rot_stylus.T @ PUSH_WORLD

    def old_direction():
        return session.rot_base_probe @ command()

    def new_direction():
        return session.rot_base_probe @ mapper.to_probe(
            command(), session.rot_base_probe, session.rot_stylus
        )

    old_reference, new_reference = old_direction(), new_direction()
    worst_old = worst_new = 0.0
    for rate in hand_profile(seconds):
        session.step(rate)
        worst_old = max(worst_old, angle_between(old_direction(), old_reference))
        worst_new = max(worst_new, angle_between(new_direction(), new_reference))
    return worst_old, worst_new


# -- 재현: 기존 매핑은 표류한다 --------------------------------------------

def test_body_frame_mapping_drifts():
    """기존 경로(``v_base = R_base_probe · v_cmd``)는 축이 어긋난다.

    이것이 실측된 증상이다 (10° → 104°, 최대 135°). 이 시험이 깨지면 재현 모형이
    더는 현실을 안 닮은 것이므로, 아래 시험들의 의미도 함께 사라진다.
    """
    worst_old, _ = replay(40.0)
    assert worst_old > 45.0, f"표류가 재현되지 않았다 ({worst_old:.1f}°)"


# -- 새 매핑 --------------------------------------------------------------

def test_latched_mapping_does_not_drift():
    """같은 조건에서 새 경로는 표류가 **0** 이다. 줄어드는 게 아니라 없다."""
    worst_old, worst_new = replay(40.0)
    # 임계 1e-4° 는 acos 의 수치 바닥이다 — 인수가 1 에 붙으면 오차가 sqrt(eps) 로
    # 증폭되어 각도로 약 1e-6° 가 남는다. 그보다 작게 잡으면 잡음을 재게 된다.
    assert worst_new < 1e-4, f"고정 매핑인데 {worst_new:.6f}° 표류했다"
    assert worst_old > 45.0   # 같은 재생에서 옛 경로는 무너진다


def test_mapping_is_identity_at_the_moment_of_engage():
    """파지를 시작하는 순간에는 기존 매핑과 정확히 같다 — 축이 갑자기 안 바뀐다."""
    session = Session()
    mapper = LinearFrameMapper(TIP_ROLL)
    mapper.engage(session.rot_base_probe, session.rot_stylus)

    for command in (PUSH_X, np.array([0.03, -0.01, 0.02]), np.zeros(3)):
        out = mapper.to_probe(command, session.rot_base_probe, session.rot_stylus)
        assert out == pytest.approx(command, abs=1e-12)


def test_magnitude_is_preserved():
    """방향만 바꾼다. 속도 크기가 바뀌면 안전 클램프의 의미가 흐려진다."""
    session = Session()
    mapper = LinearFrameMapper(TIP_ROLL)
    mapper.engage(session.rot_base_probe, session.rot_stylus)

    for rate in hand_profile(5.0):
        session.step(rate)
        command = np.array([0.05, -0.02, 0.01])
        out = mapper.to_probe(command, session.rot_base_probe, session.rot_stylus)
        assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(command), rel=1e-12)


# -- 기준 잡기 정책 --------------------------------------------------------

def test_session_latch_survives_regrip():
    """기본값에서는 재파지해도 기준이 그대로다 — 파지마다 축이 바뀌지 않는다."""
    session = Session()
    mapper = LinearFrameMapper(TIP_ROLL)
    assert mapper.engage(session.rot_base_probe, session.rot_stylus) is True
    latched = mapper.latched.copy()

    for rate in hand_profile(10.0):
        session.step(rate)
    assert mapper.engage(session.rot_base_probe, session.rot_stylus) is False
    assert mapper.latched == pytest.approx(latched)


def test_relatch_on_engage_resets_the_reference():
    """켜 두면 파지마다 다시 잡는다 — 조작자가 자리를 옮겼을 때를 위한 선택지."""
    session = Session()
    mapper = LinearFrameMapper(TIP_ROLL, relatch_on_engage=True)
    mapper.engage(session.rot_base_probe, session.rot_stylus)
    latched = mapper.latched.copy()

    for rate in hand_profile(10.0):
        session.step(rate)
    assert mapper.engage(session.rot_base_probe, session.rot_stylus) is True
    assert not np.allclose(mapper.latched, latched)


def test_falls_back_to_passthrough_before_first_engage():
    """기준을 잡기 전에는 손대지 않는다 — stylus_pose 가 없어도 기존 거동으로 돈다."""
    session = Session()
    mapper = LinearFrameMapper(TIP_ROLL)
    assert not mapper.ready
    out = mapper.to_probe(PUSH_X, session.rot_base_probe, session.rot_stylus)
    assert out == pytest.approx(PUSH_X)


def test_tip_roll_must_match_touch_node():
    """A 는 teleop 노드와 같은 행렬이어야 한다 — 갈라지면 병진이 90° 틀어진다."""
    assert axis_mapping(0.0) == pytest.approx(np.diag([1.0, -1.0, -1.0]))
    # tip_roll 90 에서 스타일러스 +X 는 프로브 +y (elevational) 로 간다 — 실측 로그와 동일
    assert axis_mapping(90.0) @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])
