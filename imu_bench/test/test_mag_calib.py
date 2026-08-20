"""자력계 보정 피팅 검증.

실제 센서 없이도 돌아가야 한다: 알려진 하드아이언/스케일 왜곡을 합성해 넣고
피팅이 그 값을 되찾는지 본다.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "host"))

import mag_calib  # noqa: E402


def sphere_dirs(n=2000, seed=0):
    """단위구 위에 고르게 뿌린 방향벡터 — 센서를 모든 방향으로 돌린 상황."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_recovers_hard_iron_and_scale():
    earth = 45.0
    bias = np.array([100.0, -30.0, 20.0])       # 지구 자기장보다 큰 하드아이언
    scale = np.array([1.0, 0.85, 1.2])          # 축별 감도 왜곡
    meas = sphere_dirs() * earth / scale + bias

    cal = mag_calib.fit(meas)

    assert np.allclose(cal["hard_iron"], bias, atol=1.0)
    assert np.allclose(cal["scale"], scale, rtol=0.02)
    assert cal["radius_uT"] == pytest.approx(earth, rel=0.05)
    assert cal["residual_pct"] < 2.0
    assert cal["coverage_deg"] > 90.0


def test_apply_makes_norm_constant():
    """보정을 적용하면 자세와 무관하게 |m| 이 일정해져야 한다 — 그게 보정의 정의다."""
    meas = sphere_dirs(seed=1) * 50.0 / np.array([1.1, 0.9, 1.0]) + np.array([-80.0, 40.0, 5.0])
    cal = mag_calib.fit(meas)

    norms = np.linalg.norm(mag_calib.apply(cal, meas), axis=1)
    assert norms.std() / norms.mean() < 0.02


def test_apply_without_calibration_is_identity():
    m = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(mag_calib.apply(None, m), m)


def test_partial_coverage_is_reported_low():
    """한쪽 방향만 돌린 데이터는 커버리지가 낮게 나와야 한다 — 사용자가 다시 돌리도록."""
    dirs = sphere_dirs(seed=2)
    dirs = dirs[dirs[:, 2] > 0.5]               # 반구의 일부만
    meas = dirs * 45.0 + np.array([10.0, 0.0, 0.0])

    assert mag_calib.fit(meas)["coverage_deg"] < 50.0


# --- TRIAD seeding -----------------------------------------------------------

import fusion as ifv  # noqa: E402


def rot_from_quat(q):
    """[w,x,y,z] -> 3x3 회전행렬 (quat_from_matrix 의 역변환)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def test_quat_from_matrix_roundtrips():
    rng = np.random.default_rng(3)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        back = ifv.quat_from_matrix(rot_from_quat(q))
        if back[0] < 0:
            back = -back
        assert np.allclose(back, q, atol=1e-6)


def test_seed_quat_matches_madgwick_fixed_point():
    """seed 가 옳다면 그 자세에서 Madgwick 의 보정항이 0 이어야 한다.

    즉 정지 상태에서 seed 한 뒤 자이로 0 으로 갱신해도 자세가 움직이지 않는다.
    """
    rng = np.random.default_rng(7)
    for _ in range(20):
        # 임의 자세에서 관측될 중력/자기장을 만든다 (지구 프레임은 NWU, 자기장은
        # 북쪽 성분 + 하향 성분)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R_be = rot_from_quat(q).T          # 지구 -> 센서
        acc = R_be @ np.array([0.0, 0.0, 9.81])
        mag = R_be @ np.array([20.0, 0.0, -44.0])

        seed = ifv.seed_quat(acc, mag)
        assert seed is not None

        filt = ifv.Madgwick(beta=0.05)
        filt.q = seed.copy()
        for _ in range(200):               # 1초분, 완전 정지
            filt.update(np.zeros(3), acc, mag, 0.005)

        assert ifv.quat_angle_deg(filt.q, seed) < 1.0


def test_seed_quat_rejects_degenerate_input():
    assert ifv.seed_quat(np.zeros(3), np.array([1.0, 0.0, 0.0])) is None
    # 자기장이 중력과 평행하면 방위를 정할 수 없다
    assert ifv.seed_quat(np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 50.0])) is None
