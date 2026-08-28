"""교정 추정 회귀 테스트 — 알려진 물리를 심고 되찾는지.

합성 데이터를 쓰는 이유: 실기 데이터로는 "맞았다" 를 증명할 수 없다. 참값을 모르기
때문이다. 질량과 COM 을 심어 두면 추정이 그것을 되찾는지, 그리고 **되찾지 못하는
상황을 스스로 거부하는지**를 둘 다 볼 수 있다. 후자가 더 중요하다 — 조용히 틀린 교정이
접촉 제어로 들어가는 것이 이 모듈이 막아야 할 실패다.
"""
import math

import numpy as np
import pytest

from fr5_control.wrench_calibration import (
    CalibrationPose,
    estimate_static_bias,
    fit_gravity_model,
    pose_coverage,
    solve_axial_alignment,
    solve_sensor_alignment,
)
from fr5_control.wrench_frames import GRAVITY_B

TRUE_MASS = 0.42          # kg — 프로브 + 마운트 정도
TRUE_COM = np.array([0.004, -0.002, 0.061])
TRUE_BIAS = np.array([0.21, -0.13, 0.44, 0.0021, -0.0014, 0.0008])
RNG = np.random.default_rng(20260826)


def _rot(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _synth_pose(rot_sb, noise=0.0, extra=None):
    """참 물리로 한 자세의 wrench 를 만든다."""
    g = rot_sb @ GRAVITY_B
    f = TRUE_MASS * g
    t = np.cross(TRUE_COM, f)
    w = np.concatenate([f, t]) + TRUE_BIAS
    if noise:
        w = w + RNG.normal(0.0, noise, 6)
    if extra is not None:
        w = w + extra
    return CalibrationPose(gravity_sensor=g, wrench=w)


def _diverse_poses(n=14, noise=0.0):
    """중력 방향을 넓게 흩은 자세 묶음."""
    out = []
    for i in range(n):
        rot = (_rot('x', 0.9 * math.sin(i * 1.7))
               @ _rot('y', 0.9 * math.cos(i * 1.1))
               @ _rot('z', i * 0.5))
        out.append(_synth_pose(rot, noise))
    return out


# -- 전자 영점 ---------------------------------------------------------------

#: 실측 잡음 (2026-08-24 벤치): 힘 축당 ±0.05 N 수준, 모멘트는 그보다 훨씬 작다.
NOISE_SD = np.array([0.02, 0.02, 0.02, 0.0005, 0.0005, 0.0005])


def test_static_bias_recovers_the_channel_means():
    samples = TRUE_BIAS + RNG.normal(0.0, 1.0, (2000, 6)) * NOISE_SD
    result = estimate_static_bias(samples, duration_s=4.0)
    assert result.accepted
    assert result.bias == pytest.approx(TRUE_BIAS, abs=0.005)


def test_static_bias_rejects_a_noisy_capture():
    """시끄러운 수집은 거부한다.

    조용해야 할 신호가 시끄러우면 무언가 닿아 있다는 뜻이다. 그 평균을 영점으로
    저장하면 그 접촉이 영점 안으로 들어가 이후 모든 측정이 조용히 틀어진다.
    """
    samples = TRUE_BIAS + RNG.normal(0.0, 0.6, (2000, 6))
    result = estimate_static_bias(samples, duration_s=4.0)
    assert not result.accepted
    assert "잡음" in result.reason


def test_static_bias_rejects_a_short_capture():
    samples = TRUE_BIAS + RNG.normal(0.0, 1.0, (2000, 6)) * NOISE_SD
    assert not estimate_static_bias(samples, duration_s=1.0).accepted


def test_static_bias_rejects_too_few_samples():
    samples = TRUE_BIAS + RNG.normal(0.0, 1.0, (50, 6)) * NOISE_SD
    assert not estimate_static_bias(samples, duration_s=4.0).accepted


def test_static_bias_requires_six_channels():
    with pytest.raises(ValueError):
        estimate_static_bias(np.zeros((100, 3)), duration_s=4.0)


# -- 자세 다양성 -------------------------------------------------------------

def test_coverage_is_high_for_diverse_poses():
    cov, _ = pose_coverage([p.gravity_sensor for p in _diverse_poses()])
    assert cov > 0.3


def test_coverage_collapses_when_all_poses_are_alike():
    """같은 자세만 모으면 다양성이 무너진다.

    중력 방향 행렬의 rank 가 1 이 되어 COM 이 관측되지 않는다. 그런데 잔차는 작게
    나오므로, 잔차만 보고 판정하면 이 실패를 못 잡는다.
    """
    same = [_synth_pose(np.eye(3)) for _ in range(20)]
    cov, _ = pose_coverage([p.gravity_sensor for p in same])
    assert cov < 0.01


# -- 중력 모델 ---------------------------------------------------------------

def test_gravity_fit_recovers_mass_and_com():
    model = fit_gravity_model(_diverse_poses(noise=0.002))
    assert model.valid, model.issues
    assert model.mass_kg == pytest.approx(TRUE_MASS, abs=0.005)
    assert model.com_sensor_m == pytest.approx(TRUE_COM, abs=0.003)


def test_gravity_fit_recovers_the_residual_bias():
    model = fit_gravity_model(_diverse_poses(noise=0.002))
    assert model.residual_bias == pytest.approx(TRUE_BIAS, abs=0.01)


def test_gravity_prediction_cancels_the_synthetic_wrench():
    """모델이 맞으면 무부하 자세의 예측이 측정과 같아야 한다 — 그것이 보상이다."""
    poses = _diverse_poses(noise=0.0)
    model = fit_gravity_model(poses)
    for p in poses:
        assert model.predict(p.gravity_sensor) == pytest.approx(p.wrench, abs=1e-6)


def test_gravity_fit_rejects_too_few_poses():
    model = fit_gravity_model(_diverse_poses(n=5))
    assert not model.valid
    assert any("자세 부족" in m for m in model.issues)


def test_gravity_fit_rejects_poor_coverage():
    """한 방향만 본 데이터로는 COM 을 관측할 수 없다. 잔차가 작아도 거부해야 한다."""
    poses = [_synth_pose(_rot('z', i * 0.3), noise=0.001) for i in range(16)]
    model = fit_gravity_model(poses)
    assert not model.valid
    assert any("다양성" in m for m in model.issues)


def test_gravity_fit_rejects_implausible_mass():
    heavy = []
    for p in _diverse_poses():
        w = p.wrench.copy()
        w[:3] *= 40.0
        heavy.append(CalibrationPose(p.gravity_sensor, w))
    model = fit_gravity_model(heavy)
    assert not model.valid
    assert any("질량" in m for m in model.issues)


def test_robust_fit_survives_one_bad_pose():
    """한 자세에서 케이블이 걸려도 전체 해가 끌려가지 않는다.

    보통 최소자승이면 그 하나가 질량을 크게 흔든다. Huber 가중이 영향력을 줄인다.
    """
    poses = _diverse_poses(noise=0.002)
    bad = poses[3]
    poses[3] = CalibrationPose(
        bad.gravity_sensor, bad.wrench + np.array([6.0, -4.0, 5.0, 0.3, -0.2, 0.1])
    )
    model = fit_gravity_model(poses)
    assert model.mass_kg == pytest.approx(TRUE_MASS, abs=0.05)
    assert model.com_sensor_m == pytest.approx(TRUE_COM, abs=0.02)


def test_per_axis_residuals_are_reported():
    model = fit_gravity_model(_diverse_poses(noise=0.01))
    assert model.per_axis_force_n.shape == (3,)
    assert model.per_axis_torque_nm.shape == (3,)
    assert np.all(model.per_axis_force_n >= 0)


def test_empty_pose_list_is_an_error():
    with pytest.raises(ValueError):
        fit_gravity_model([])


def test_model_round_trips_through_json():
    from fr5_control.wrench_calibration import GravityModel
    model = fit_gravity_model(_diverse_poses(noise=0.001))
    back = GravityModel.from_dict(model.to_dict())
    assert back.mass_kg == pytest.approx(model.mass_kg)
    assert back.com_sensor_m == pytest.approx(model.com_sensor_m)
    assert back.predict([0.0, 0.0, -9.8]) == pytest.approx(model.predict([0.0, 0.0, -9.8]))


# -- 플랜지→센서 정렬을 데이터에서 푼다 ----------------------------------


def _rotation(roll, pitch, yaw):
    """고정축 XYZ 회전행렬. 시험 안에서만 쓴다."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _spread_gravity(count, seed=0):
    """서로 충분히 다른 방향의 {F}g 를 만든다."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        d = rng.normal(size=3)
        norm = np.linalg.norm(d)
        if norm > 1e-6:
            out.append(d / norm * 9.80665)
    return np.array(out)


def _angle_between(a, b):
    """두 회전 사이 측지각 [도]."""
    trace = float(np.trace(np.asarray(a) @ np.asarray(b).T))
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))


def test_alignment_recovers_a_planted_rotation_and_mass():
    truth = _rotation(0.3, -0.5, 1.1)
    mass, bias = 1.04, np.array([0.4, -0.2, 0.9])
    gravity = _spread_gravity(14, seed=1)
    forces = np.array([mass * (truth @ g) + bias for g in gravity])

    result = solve_sensor_alignment(gravity, forces)
    assert result.mass_kg == pytest.approx(mass, abs=1e-6)
    assert _angle_between(result.rotation_sensor_from_flange, truth) < 1e-3
    assert np.allclose(result.bias_force, bias, atol=1e-6)


def test_alignment_result_is_a_proper_rotation():
    """반사가 섞이면 힘은 맞아도 모멘트가 거울상이 된다."""
    gravity = _spread_gravity(12, seed=2)
    forces = np.array([0.9 * (_rotation(1.2, 0.4, -0.8) @ g) for g in gravity])
    rotation = solve_sensor_alignment(gravity, forces).rotation_sensor_from_flange
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_alignment_needs_gravity_out_of_one_plane():
    """한 평면에 몰린 방향들은 회전의 한 축을 결정하지 못한다."""
    flat = np.array([
        [9.8 * math.cos(a), 9.8 * math.sin(a), 0.0]
        for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)
    ])
    with pytest.raises(ValueError, match="한 평면"):
        solve_sensor_alignment(flat, flat * 1.04)


def test_alignment_needs_four_poses():
    gravity = _spread_gravity(3, seed=3)
    with pytest.raises(ValueError):
        solve_sensor_alignment(gravity, gravity)


def test_alignment_spread_is_small_for_a_true_rotation():
    gravity = _spread_gravity(14, seed=4)
    forces = np.array([1.04 * (_rotation(0.2, 0.9, -1.4) @ g) for g in gravity])
    assert solve_sensor_alignment(gravity, forces).scale_spread < 1e-6


def test_alignment_spread_grows_when_the_map_is_not_a_rotation():
    """축마다 배율이 다르면 회전이 아니다. 그것이 산포로 드러나야 한다."""
    gravity = _spread_gravity(14, seed=5)
    stretch = np.diag([1.4, 1.0, 0.7])
    forces = np.array([stretch @ g for g in gravity])
    assert solve_sensor_alignment(gravity, forces).scale_spread > 0.3


def test_assumed_identity_produces_the_negative_mass_that_was_observed():
    """왜 이 솔버가 필요한지를 고정한다.

    센서가 실제로 돌아 앉아 있는데 회전을 단위행렬로 가정하면, 모델 중력이
    측정된 힘과 다른 방향을 가리킨다. 크기는 맞는데 방향이 틀리므로 최소제곱은
    질량을 0 쪽으로 밀고, 실측에서 음수로 나왔다.
    """
    truth = _rotation(0.9, -0.7, 1.3)
    gravity = _spread_gravity(14, seed=6)
    forces = np.array([1.04 * (truth @ g) for g in gravity])

    assumed = [
        CalibrationPose(gravity_sensor=g, wrench=np.concatenate([f, np.zeros(3)]))
        for g, f in zip(gravity, forces)
    ]
    bad = fit_gravity_model(assumed, solve_alignment=False)
    assert not bad.valid
    assert bad.rms_force_n > 1.0

    solved = [
        CalibrationPose(
            gravity_sensor=g, gravity_flange=g, wrench=np.concatenate([f, np.zeros(3)])
        )
        for g, f in zip(gravity, forces)
    ]
    # 심은 것이 자유 회전이므로 z 제약으로는 못 잡는다. 제약을 풀고 푼다.
    good = fit_gravity_model(solved, solve_alignment=True, axial_alignment=False)
    assert good.mass_kg == pytest.approx(1.04, abs=1e-4)
    assert good.rms_force_n < 1e-6
    assert _angle_between(good.rotation_sensor_from_flange, truth) < 1e-3


def test_fit_keeps_identity_when_flange_gravity_is_missing():
    """옛 기록에는 플랜지 중력이 없다. 그때는 조용히 예전 거동을 쓴다."""
    gravity = _spread_gravity(14, seed=7)
    poses = [
        CalibrationPose(gravity_sensor=g, wrench=np.concatenate([1.0 * g, np.zeros(3)]))
        for g in gravity
    ]
    model = fit_gravity_model(poses, solve_alignment=True)
    assert np.allclose(model.rotation_sensor_from_flange, np.eye(3))
    assert model.mass_kg == pytest.approx(1.0, abs=1e-6)


def test_axial_alignment_recovers_a_rotation_about_the_stack_axis():
    """적층이 동축일 때 남는 자유도는 축 둘레 회전 하나뿐이다."""
    truth = _rotation(0.0, 0.0, math.radians(136.5))
    mass, bias = 0.206, np.array([-1.24, 1.08, 8.52])
    gravity = _spread_gravity(13, seed=11)
    forces = np.array([mass * (truth @ g) + bias for g in gravity])

    result = solve_axial_alignment(gravity, forces)
    assert result.mass_kg == pytest.approx(mass, abs=1e-4)
    assert _angle_between(result.rotation_sensor_from_flange, truth) < 0.3
    assert np.allclose(result.bias_force, bias, atol=1e-3)
    assert result.scale_spread < 1e-3


def test_axial_alignment_fits_a_weak_signal_far_more_tightly():
    """왜 제약이 필요한가.

    자중 200 g 에 오프셋 8.5 N 이면 중력 신호가 오프셋의 4 분의 1 밖에 안 된다.
    그 조건에서 자유 3 자유도는 잡음까지 회전으로 흡수해 잔차를 남기고, 제약은
    그러지 않는다.

    각도 정확도까지 항상 낫다고는 하지 않는다 — 이 잡음 수준에서 둘은 1.5° 안팎으로
    사실상 동률이고, 축 제약 쪽은 0.25° 격자 분해능에 걸려 있다. 이득은 **적합이
    얼마나 조여지는가** 에 있다.

    이 합성 시험의 잡음(σ = 0.05 N)은 실기보다 작아서, 여기서는 자유 해도
    산포 문턱을 통과한다. 실제 13 자세에서는 자유 해가 0.30 으로 걸렸고 제약 해가
    잔차를 0.183 → 0.047 N 으로 낮췄다 — 그 차이는 시험이 아니라 기록으로 남긴다.
    """
    truth = _rotation(0.0, 0.0, math.radians(136.5))
    rng = np.random.default_rng(12)
    gravity = _spread_gravity(13, seed=12)
    forces = np.array([
        0.206 * (truth @ g) + np.array([-1.24, 1.08, 8.52]) + rng.normal(scale=0.05, size=3)
        for g in gravity
    ])

    free = solve_sensor_alignment(gravity, forces)
    axial = solve_axial_alignment(gravity, forces)

    assert axial.residual_n < free.residual_n / 2.0
    assert axial.scale_spread < free.scale_spread
    assert axial.mass_kg == pytest.approx(0.206, abs=0.01)


def test_axial_alignment_keeps_the_rotation_about_z():
    """Z 성분이 섞이면 동축이라는 사실을 버린 것이다."""
    gravity = _spread_gravity(12, seed=13)
    forces = np.array([0.3 * g for g in gravity])
    rotation = solve_axial_alignment(gravity, forces).rotation_sensor_from_flange
    assert np.allclose(rotation[2, :], [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-12)
