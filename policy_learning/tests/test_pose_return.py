"""시작 자세로 되돌아가기.

한 자세에서 여러 조건을 비교하려면 셋이 같은 자리에서 출발해야 하는데, 에피소드가 끝나면
로봇은 이미 옮겨져 있다 (2026-09-14 본 시험 ep0001 은 50 mm 를 옮겨 놓고 끝났다).
손으로 되돌리면 그 오차가 조건 차이에 섞인다.
"""

import numpy as np
import pytest

from rus_policy.pose_return import PoseTarget, parse_optional

I = (1.0, 0.0, 0.0, 0.0)                      # 단위 사원수


def rotz(deg):
    h = np.radians(deg) / 2.0
    return (np.cos(h), 0.0, 0.0, np.sin(h))


def test_parses_the_recorded_order():
    t = PoseTarget.parse("-0.4826,-0.3490,0.4156,1,0,0,0")
    assert np.allclose(t.p, [-0.4826, -0.3490, 0.4156])
    assert np.allclose(t.q, [1, 0, 0, 0])


def test_refuses_a_malformed_pose():
    with pytest.raises(ValueError, match="일곱 개"):
        PoseTarget.parse("1,2,3")


def test_no_target_is_the_normal_path():
    """자세의 첫 에피소드에는 돌아갈 곳이 없다."""
    assert parse_optional(None) is None and parse_optional("") is None


def test_already_there_means_no_command():
    t = PoseTarget(np.zeros(3), np.array(I))
    vel, (mm, deg) = t.step(np.zeros(3), I, max_mm_s=3.0, max_deg_s=3.0)
    assert not np.any(vel) and mm == 0.0 and deg == 0.0
    assert t.reached(np.zeros(3), I)


def step_world(p, q, vel, dt):
    """지령(프로브 프레임)을 기저 프레임의 자세 변화로 적분한다."""
    from rus_policy.pose_return import _mul, _rot
    p = np.asarray(p, float) + _rot(q) @ (np.asarray(vel[:3], float) * dt / 1000.0)
    w = np.radians(np.asarray(vel[3:], float) * dt)          # 프로브 프레임 회전벡터
    a = float(np.linalg.norm(w))
    if a > 1e-12:
        h = a / 2.0
        dq = np.concatenate(([np.cos(h)], np.sin(h) * w / a))
        q = _mul(q, dq)                                       # 오른쪽 곱 = 몸통 좌표
        q = q / np.linalg.norm(q)
    return p, q


def test_it_drives_toward_the_target_and_gets_there():
    """폐루프로 돌려 실제로 도착하는지 본다 — 부호가 뒤집히면 멀어진다."""
    t = PoseTarget(np.array([0.020, -0.015, 0.0]), np.array(rotz(6.0)))
    p, q = np.zeros(3), np.array(I, float)
    for _ in range(600):
        vel, _ = t.step(p, q, max_mm_s=3.0, max_deg_s=3.0)
        p, q = step_world(p, q, vel, 0.1)
        if t.reached(p, q):
            break
    mm, deg = t.error(p, q)
    assert t.reached(p, q), f"{mm:.1f} mm · {deg:.1f}° 남았다"


def test_a_tilted_probe_still_converges():
    """프로브가 기울어 있어도 도착해야 한다 — 프레임 변환이 틀리면 여기서 발산한다."""
    start_q = np.array(rotz(35.0), float)
    t = PoseTarget(np.array([0.018, 0.012, 0.0]), np.array(rotz(20.0)))
    p, q = np.zeros(3), start_q
    d0 = t.error(p, q)[0]
    for _ in range(800):
        vel, _ = t.step(p, q, max_mm_s=3.0, max_deg_s=3.0)
        p, q = step_world(p, q, vel, 0.1)
        if t.reached(p, q):
            break
    mm, deg = t.error(p, q)
    assert mm < d0, f"멀어졌다: {d0:.1f} → {mm:.1f} mm"
    assert t.reached(p, q), f"{mm:.1f} mm · {deg:.1f}° 남았다"


def test_the_beam_axis_is_never_commanded():
    """z 는 힘 조절기의 것이다 — 되돌리기가 그 축을 밀면 둘이 다툰다."""
    t = PoseTarget(np.array([0.0, 0.0, 0.050]), np.array(I))
    vel, (mm, _) = t.step(np.zeros(3), I, max_mm_s=3.0, max_deg_s=3.0)
    assert vel[2] == 0.0, vel
    assert mm == 0.0, "빔축 오차를 위치 오차로 세고 있다 — 영원히 도착하지 못한다"
    assert t.reached(np.zeros(3), I)


def test_error_is_measured_in_the_probe_frame():
    """기저 프레임에서 재면 프로브가 기울어져 있을 때 엉뚱한 축으로 민다."""
    t = PoseTarget(np.array([0.0, 0.0, 0.010]), np.array(I))   # 기저 z 로 10 mm
    # 프로브가 90° 기울어 있으면 그 10 mm 는 프로브의 **면내** 성분이다
    vel, (mm, _) = t.step(np.zeros(3), (np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0),
                          max_mm_s=3.0, max_deg_s=3.0)
    assert mm > 5.0, f"기울어진 프로브에서 오차를 못 본다: {mm:.1f} mm"


def test_tolerance_matches_the_measurement():
    """10 mm 면 마스크 보임 90 %, 30 mm 면 44 % — 허용은 그보다 빡빡해야 한다."""
    from rus_policy.pose_return import TOL_MM
    assert TOL_MM <= 10.0


def test_speed_limits_are_respected():
    t = PoseTarget(np.array([1.0, 1.0, 0.0]), np.array(rotz(90.0)))
    vel, _ = t.step(np.zeros(3), I, max_mm_s=3.0, max_deg_s=2.0)
    assert np.all(np.abs(vel[:3]) <= 3.0 + 1e-9)
    assert np.all(np.abs(vel[3:]) <= 2.0 + 1e-9)


def test_the_runner_claims_the_probe_frame_for_translation():
    """_remap_twist 는 gate_axes 보다 **앞**에 있다.

    teleop.linear_frame 이 "latched" 면 러너가 프로브 프레임으로 낸 x·y 가 조작자 기준
    프레임으로 재사상된 뒤에야 게이트를 지난다 — 러너는 자기가 낸 방향으로 가고 있다고
    믿지만 로봇은 다른 쪽으로 간다. 2026-09-14 본 시험 ep0002 의 복귀가 그래서 발산했다
    (45.9 → 7.0 → 35.3 mm 를 오가며 45 s 를 썼고 각도 오차는 5.1° → 13.0° 로 커졌다).
    """
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text(encoding="utf-8")
    assert 'LINEAR_FRAME_PARAM = "teleop.linear_frame"' in src
    fn = re.search(r"def _enable_lateral_if_needed.*?\n    def ", src, re.S).group(0)
    assert "self.LINEAR_FRAME_PARAM" in fn, "병진 프레임을 안 맞춘다"
    assert 'string_value="body"' in fn, "프로브 프레임으로 받겠다고 말하지 않는다"
    # 못 맞추면 시작하지 않는다 — 엉뚱한 방향으로 가는 것보다 안 가는 편이 낫다
    assert "SystemExit" in fn, fn
    assert '"linear_frame"' in src, "무엇으로 돌았는지 기록에 안 남는다"


def test_the_timeout_is_reported_not_swallowed():
    """못 돌아갔으면 그 에피소드는 짝이 안 맞는다 — 조용히 시작하면 안 된다."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text(encoding="utf-8")
    fn = re.search(r"def _return_step.*?\n    def ", src, re.S).group(0)
    assert "return_timeout_s" in fn and "복귀 실패" in fn, fn
    assert 'self.return_state.update(ok=False' in fn, "실패를 기록에 안 남긴다"
