"""지령한 회전이 실제로 일어났는가 — motion_check.

2026-09-11 까지 "지령은 나가는데 로봇이 안 움직인다" 가 세 번 있었고, 셋 다 코드를
읽어서야 찾았다. 기록에 실제 자세가 있으면 첫 에피소드에서 드러난다.
"""

import math

import numpy as np

from rus_policy.episode import _quat_angle_deg, motion_check


def _rot_y(deg):
    h = math.radians(deg) / 2.0
    return (math.cos(h), 0.0, math.sin(h), 0.0)


def _episode(rate_deg_s, realized_fraction, seconds=10.0, hz=5.0):
    """ω_y 를 rate 로 지령하고, 실제로는 그 fraction 만큼만 돈 에피소드."""
    n = int(seconds * hz) + 1
    t = np.arange(n) / hz
    w = np.tile([0.0, rate_deg_s, 0.0], (n, 1))
    q = np.array([_rot_y(rate_deg_s * realized_fraction * ti) for ti in t])
    return t, w, q


def test_quat_angle_is_sign_invariant():
    q = _rot_y(30.0)
    assert abs(_quat_angle_deg(q, q)) < 1e-6
    assert abs(_quat_angle_deg(q, tuple(-v for v in q))) < 1e-6   # q ≡ −q
    assert abs(_quat_angle_deg(_rot_y(0.0), _rot_y(30.0)) - 30.0) < 1e-6


def test_commanded_and_realized_rotation_agree():
    t, w, q = _episode(3.0, 1.0)
    r = motion_check(t, w, q)
    assert abs(r["commanded_deg"] - 30.0) < 0.1          # 3 °/s × 10 s
    assert abs(r["realized_deg"] - 30.0) < 0.1
    assert r["ratio"] > 0.95 and "지령대로" in r["verdict"]


def test_detects_command_that_never_reached_the_robot():
    """오늘의 증상: 지령은 나가는데 자세가 안 변한다."""
    t, w, q = _episode(3.0, 0.0)
    r = motion_check(t, w, q)
    assert r["commanded_deg"] > 29.0
    assert r["realized_deg"] < 0.01
    assert "안 움직였다" in r["verdict"]


def test_detects_the_half_duty_watchdog_symptom():
    """5 Hz 발행이 워치독에 절반씩 버려지던 상태 — 실현 ~35 %."""
    t, w, q = _episode(3.0, 0.35)
    r = motion_check(t, w, q)
    assert 0.3 < r["ratio"] < 0.4 and "일부만" in r["verdict"]


def test_hold_is_reported_as_no_command_not_as_failure():
    """hold 는 지령하지 않는 게 정상이다 — '안 움직였다' 경보를 내면 안 된다."""
    t, _, q = _episode(3.0, 0.0)
    r = motion_check(t, np.zeros((len(t), 3)), q)
    assert "지령 없음" in r["verdict"] and "안 움직였다" not in r["verdict"]


def test_placebo_path_length_ignores_direction():
    """위약은 크기는 같고 방향만 무작위다. 경로 길이는 방향과 무관해야 한다."""
    hz, n = 5.0, 51
    t = np.arange(n) / hz
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=n)
    w = np.stack([np.zeros(n), 3.0 * signs, np.zeros(n)], 1)
    ang = np.concatenate([[0.0], np.cumsum(3.0 * signs[:-1] / hz)])
    q = np.array([_rot_y(a) for a in ang])
    r = motion_check(t, w, q)
    assert r["ratio"] > 0.95, r
    # 왕복하므로 알짜 회전은 경로보다 훨씬 작다 — 그래서 알짜로는 판정하지 않는다.
    assert r["realized_net_deg"] < r["realized_deg"]


def test_missing_pose_is_reported_not_guessed():
    t, w, _ = _episode(3.0, 1.0)
    q = np.full((len(t), 4), np.nan)
    r = motion_check(t, w, q)
    assert "자세 없음" in r["verdict"]
