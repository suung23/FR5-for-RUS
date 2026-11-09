"""복귀 — 빵부스러기 기록과 역순 되밟기."""

import math

import pytest

from safety_estop.recovery import PathRecorder, RecoveryLimits, ReturnPlanner

RAD2DEG = 180.0 / math.pi


def pose(j1):
    return [j1, -90.0, 0.0, -90.0, 0.0, 0.0]


# -- 기록 ----------------------------------------------------------------


def test_first_point_is_the_initial_pose():
    r = PathRecorder(min_step_deg=1.0)
    r.reset(pose(0.0))
    r.record(pose(5.0))
    r.record(pose(10.0))
    assert r.home_deg == pose(0.0)
    assert len(r) == 3


def test_small_motion_is_not_recorded():
    r = PathRecorder(min_step_deg=1.0)
    r.reset(pose(0.0))
    for i in range(100):
        r.record(pose(i * 0.005))      # 총 0.5° — 한 점도 안 쌓인다
    assert len(r) == 1


def test_oldest_points_fall_off_and_that_is_reported():
    r = PathRecorder(min_step_deg=1.0, max_points=3)
    r.reset(pose(0.0))
    for j in (1.0, 2.0, 3.0):
        r.record(pose(j))
    assert len(r) == 3
    assert r.truncated is True
    # 버린 뒤의 "초기 자세" 는 원래 자세가 아니다. 이 사실을 콘솔이 말해야 한다.
    assert r.home_deg != pose(0.0)


# -- 되밟기 --------------------------------------------------------------


def run(trail, start, limits=None, force=None, dt=0.05, max_steps=4000):
    """지령대로 움직이는 이상적인 로봇으로 복귀를 끝까지 돌린다."""
    limits = limits or RecoveryLimits()
    planner = ReturnPlanner(trail, limits, started_at=0.0)
    q = list(start)
    path = [list(q)]
    t = 0.0
    for _ in range(max_steps):
        t += dt
        f = force(t) if callable(force) else force
        vel, done, reason = planner.step(t, q, f)
        if reason:
            return path, reason, planner
        if done:
            return path, "", planner
        for i in range(6):
            q[i] += vel[i] * RAD2DEG * dt
        path.append(list(q))
    raise AssertionError("복귀가 끝나지 않았다")


def test_return_walks_the_trail_backwards_to_the_initial_pose():
    trail = [pose(0.0), pose(10.0), pose(20.0), pose(30.0)]
    path, reason, _ = run(trail, start=pose(30.0), force=0.0)
    assert reason == ""
    assert path[-1][0] == pytest.approx(0.0, abs=0.5)
    # 단조롭게 되돌아온다 — 중간에 앞으로 가는 구간이 없다.
    assert all(path[i + 1][0] <= path[i][0] + 1e-6 for i in range(len(path) - 1))


def test_it_does_not_cut_the_corner():
    # 들어올 때 J1 으로 돌고 J3 으로 내려왔다면, 나갈 때도 J3 → J1 순서다.
    # 관절공간 직선(MoveJ)이었다면 두 축이 동시에 움직이며 모서리를 잘랐을 것이다.
    trail = [
        [0.0, -90.0, 0.0, -90.0, 0.0, 0.0],
        [40.0, -90.0, 0.0, -90.0, 0.0, 0.0],      # J1 으로 돌고
        [40.0, -90.0, 40.0, -90.0, 0.0, 0.0],     # J3 으로 내려왔다
    ]
    path, reason, _ = run(trail, start=trail[-1], force=0.0)
    assert reason == ""
    # J3 이 거의 다 돌아오기 전에는 J1 이 크게 움직이지 않는다.
    while_j3_moving = [p for p in path if p[2] > 20.0]
    assert max(abs(p[0] - 40.0) for p in while_j3_moving) < 1.0


def test_contact_slows_the_first_segment():
    trail = [pose(0.0), pose(20.0)]
    slow = RecoveryLimits(release_speed_deg_s=2.0, return_speed_deg_s=20.0,
                          release_force_n=0.5)
    planner = ReturnPlanner(trail, slow, started_at=0.0)
    vel_contact, _, _ = planner.step(0.1, pose(20.0), 3.0)
    vel_free, _, _ = planner.step(0.2, pose(20.0), 0.1)
    assert abs(vel_contact[0]) * RAD2DEG == pytest.approx(2.0, abs=1e-6)
    assert abs(vel_free[0]) * RAD2DEG == pytest.approx(20.0, abs=1e-6)


def test_unknown_force_is_treated_as_contact():
    # 모르는 쪽에서 더 느리게 움직이는 것이 틀릴 때 덜 해롭다.
    trail = [pose(0.0), pose(20.0)]
    planner = ReturnPlanner(trail, RecoveryLimits(), started_at=0.0)
    vel, _, _ = planner.step(0.1, pose(20.0), None)
    assert planner.in_contact is True
    assert abs(vel[0]) * RAD2DEG == pytest.approx(2.0, abs=1e-6)


def test_it_stops_instead_of_retreating_forever():
    # 물러나는데 힘이 안 떨어진다 — 더 물러나는 것은 이 고장에 대한 답이 아니다.
    trail = [pose(0.0), pose(20.0)]
    limits = RecoveryLimits(max_release_s=1.0)
    _, reason, _ = run(trail, start=pose(20.0), limits=limits, force=5.0)
    assert "안 떨어진다" in reason


def test_a_stuck_arm_halts():
    limits = RecoveryLimits(stall_s=0.5, stall_deg=0.2)
    planner = ReturnPlanner([pose(0.0), pose(20.0)], limits, started_at=0.0)
    reason = ""
    for k in range(1, 40):
        _, _, reason = planner.step(k * 0.1, pose(20.0), 0.0)   # 전혀 안 움직인다
        if reason:
            break
    assert "걸려 있다" in reason


def test_already_home_finishes_immediately():
    planner = ReturnPlanner([pose(0.0)], RecoveryLimits(), started_at=0.0)
    _, done, reason = planner.step(0.1, pose(0.0), 0.0)
    assert done and reason == ""


def test_release_cannot_be_faster_than_the_free_return():
    with pytest.raises(ValueError, match="닿아 있는 동안 더 느려야"):
        RecoveryLimits(release_speed_deg_s=10.0, return_speed_deg_s=5.0).validated()
