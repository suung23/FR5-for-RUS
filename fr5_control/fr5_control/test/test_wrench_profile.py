"""프로파일과 런타임 보상 회귀 테스트.

사양이 마지막에 요구하는 것을 그대로 시험한다: **무부하 프로브가, 교정한 자세가
아니라 여러 다른 자세에서, 0 근처의 보상 wrench 를 내는가.** 초기 자세에서만 0 이
나오는 것은 전자 영점만 한 것과 구별되지 않는다.
"""
import math
import time

import numpy as np
import pytest

from fr5_control.wrench_calibration import (
    BiasResult,
    CalibrationPose,
    fit_gravity_model,
)
from fr5_control.wrench_frames import FrameRegistration, GRAVITY_B
from fr5_control.wrench_profile import CalibrationProfile, compensate

TRUE_MASS = 0.42
TRUE_COM = np.array([0.004, -0.002, 0.061])
TRUE_BIAS = np.array([0.21, -0.13, 0.44, 0.0021, -0.0014, 0.0008])
LEVER = np.array([0.0, 0.0, 0.085])          # 센서 원점 → 프로브 접촉점


def _rot(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _flange_orientations(n=14):
    """중력 방향을 넓게 흩는 플랜지 자세들."""
    return [
        _rot('x', 0.9 * math.sin(i * 1.7)) @ _rot('y', 0.9 * math.cos(i * 1.1)) @ _rot('z', i * 0.5)
        for i in range(n)
    ]


def _raw_unloaded(reg, rot_bf):
    """무부하일 때 센서가 실제로 낼 원값 (참 물리)."""
    g = reg.gravity_in_sensor(rot_bf)
    f = TRUE_MASS * g
    return np.concatenate([f, np.cross(TRUE_COM, f)]) + TRUE_BIAS


def _profile(reg, poses_rot, gravity=True):
    bias_pose = np.eye(3)
    bias_samples = np.tile(_raw_unloaded(reg, bias_pose), (2000, 1))
    bias = BiasResult(bias_samples.mean(axis=0), np.zeros(6), 2000, 4.0, True, "합성")
    model = None
    if gravity:
        model = fit_gravity_model([
            CalibrationPose(reg.gravity_in_sensor(r), _raw_unloaded(reg, r))
            for r in poses_rot
        ])
    return CalibrationProfile(registration=reg, bias=bias, gravity=model)


# -- 핵심 요구 ---------------------------------------------------------------

def test_unloaded_probe_is_near_zero_at_many_orientations():
    """사양의 최종 확인 항목. 교정 자세가 아닌 곳에서도 0 이어야 한다."""
    reg = FrameRegistration(mounting_angle_deg=45.0, r_sensor_to_probe_m=LEVER)
    profile = _profile(reg, _flange_orientations())

    # 교정에 쓰지 않은 자세들
    unseen = [_rot('y', 1.2) @ _rot('x', -0.7), _rot('x', 2.0), _rot('z', 1.0) @ _rot('y', -1.4)]
    for rot_bf in unseen:
        out = compensate(profile, _raw_unloaded(reg, rot_bf), rot_bf)
        assert np.abs(out.contact_probe[:3]).max() < 1e-6
        assert np.abs(out.contact_probe[3:]).max() < 1e-6


def test_bias_only_profile_drifts_when_orientation_changes():
    """전자 영점만으로는 왜 부족한지 — 이것이 두 절차를 나눈 이유다.

    영점을 잰 자세에서는 0 이지만, 자세를 바꾸면 수 N 씩 어긋난다.
    """
    reg = FrameRegistration(mounting_angle_deg=45.0, r_sensor_to_probe_m=LEVER)
    profile = _profile(reg, _flange_orientations(), gravity=False)

    at_home = compensate(profile, _raw_unloaded(reg, np.eye(3)), np.eye(3))
    assert np.abs(at_home.contact_probe[:3]).max() < 1e-9

    tipped = _rot('x', math.pi / 2)
    out = compensate(profile, _raw_unloaded(reg, tipped), tipped)
    assert np.abs(out.contact_probe[:3]).max() > 3.0


def test_gravity_compensation_is_skipped_without_a_pose():
    """자세를 모르면 중력을 빼지 않는다. 추측은 보상이 아니다."""
    reg = FrameRegistration(r_sensor_to_probe_m=LEVER)
    profile = _profile(reg, _flange_orientations())
    out = compensate(profile, _raw_unloaded(reg, np.eye(3)), None)
    assert out.external_sensor == pytest.approx(out.bias_corrected_sensor)


# -- 접촉 신호 ---------------------------------------------------------------

def test_axial_push_appears_as_positive_normal_force():
    """사양 §6 시험 2. 압축이 +F_zP 로 나와야 한다."""
    reg = FrameRegistration(mounting_angle_deg=45.0, r_sensor_to_probe_m=LEVER)
    profile = _profile(reg, _flange_orientations())
    rot_bf = np.eye(3)
    # 프로브 +z 로 3 N 압축. 센서 z 는 프로브 z 와 같으므로 센서에도 +z 로 실린다.
    contact = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    raw = _raw_unloaded(reg, rot_bf) + contact
    out = compensate(profile, raw, rot_bf)
    assert out.normal_force_n == pytest.approx(3.0, abs=1e-6)


def test_lateral_push_lands_on_the_image_axis():
    """사양 §6 시험 3. 영상면 내 lateral 은 F_xP 에 주로 실린다."""
    reg = FrameRegistration(mounting_angle_deg=45.0, r_sensor_to_probe_m=LEVER)
    profile = _profile(reg, _flange_orientations())
    rot_bf = np.eye(3)
    # 프로브 +x 로 2 N. 45° 장착이므로 센서에서는 x, y 로 나뉜다.
    rot_ps = reg.rotation_probe_from_sensor()
    sensor_force = rot_ps.T @ np.array([2.0, 0.0, 0.0])
    raw = _raw_unloaded(reg, rot_bf) + np.concatenate([sensor_force, np.zeros(3)])
    out = compensate(profile, raw, rot_bf)
    assert out.contact_probe[0] == pytest.approx(2.0, abs=1e-6)
    assert abs(out.contact_probe[1]) < 1e-6
    assert abs(out.contact_probe[2]) < 1e-6


def test_elevational_push_lands_on_the_out_of_plane_axis():
    """사양 §6 시험 4."""
    reg = FrameRegistration(mounting_angle_deg=45.0, r_sensor_to_probe_m=LEVER)
    profile = _profile(reg, _flange_orientations())
    rot_bf = np.eye(3)
    rot_ps = reg.rotation_probe_from_sensor()
    sensor_force = rot_ps.T @ np.array([0.0, 1.5, 0.0])
    raw = _raw_unloaded(reg, rot_bf) + np.concatenate([sensor_force, np.zeros(3)])
    out = compensate(profile, raw, rot_bf)
    assert out.contact_probe[1] == pytest.approx(1.5, abs=1e-6)
    assert abs(out.contact_probe[0]) < 1e-6


def test_lever_arm_removes_the_offset_moment():
    """접촉점에서 순수 lateral 을 밀면 접촉 모멘트는 0 이어야 한다.

    센서 원점에서는 지렛대 때문에 모멘트가 보이지만, 기준점을 옮기면 사라진다.
    이 항을 빼먹으면 rx/ry 정렬이 원리적으로 동작하지 않는다 (§4.3).
    """
    reg = FrameRegistration(mounting_angle_deg=0.0, r_sensor_to_probe_m=LEVER)
    profile = _profile(reg, _flange_orientations())
    rot_bf = np.eye(3)
    contact_force = np.array([1.0, 0.0, 0.0])
    sensor_moment = np.cross(LEVER, contact_force)
    raw = _raw_unloaded(reg, rot_bf) + np.concatenate([contact_force, sensor_moment])
    out = compensate(profile, raw, rot_bf)
    assert out.external_probe[4] == pytest.approx(sensor_moment[1], abs=1e-9)
    assert out.contact_probe[3:] == pytest.approx(np.zeros(3), abs=1e-9)


def test_axial_flip_reverses_the_reported_normal_force():
    """부호 뒤집기가 등록 층 하나에서만 일어난다는 것을 런타임까지 확인한다."""
    lever = np.array([0.0, 0.0, 0.085])
    plain = FrameRegistration(mounting_angle_deg=0.0, r_sensor_to_probe_m=lever)
    flipped = FrameRegistration(mounting_angle_deg=0.0, axial_flip=True,
                                r_sensor_to_probe_m=lever)
    push = np.array([0.0, 0.0, 2.5, 0.0, 0.0, 0.0])
    for reg, expect in ((plain, 2.5), (flipped, -2.5)):
        profile = _profile(reg, _flange_orientations())
        raw = _raw_unloaded(reg, np.eye(3)) + push
        assert compensate(profile, raw, np.eye(3)).normal_force_n == pytest.approx(expect, abs=1e-6)


# -- 유효성과 저장 -----------------------------------------------------------

def test_profile_without_gravity_is_invalid():
    """A 단계만으로 접촉 제어를 열면 자세가 바뀌는 순간 값이 어긋난다."""
    reg = FrameRegistration()
    profile = _profile(reg, _flange_orientations(), gravity=False)
    valid, issues = profile.validity()
    assert not valid
    assert any("중력 보상 미완료" in m for m in issues)


def test_profile_with_rejected_bias_is_invalid():
    reg = FrameRegistration()
    profile = _profile(reg, _flange_orientations())
    profile.bias = BiasResult(np.zeros(6), np.zeros(6), 10, 0.1, False, "잡음 과다")
    valid, issues = profile.validity()
    assert not valid
    assert any("전자 영점" in m for m in issues)


def test_old_profile_is_flagged_not_silently_used():
    """오래된 교정을 조용히 쓰지 않는다 (§7)."""
    reg = FrameRegistration()
    profile = _profile(reg, _flange_orientations())
    profile.created_at = time.time() - 40 * 3600
    valid, issues = profile.validity()
    assert not valid
    assert any("오래됐다" in m for m in issues)


def test_fresh_complete_profile_is_valid():
    reg = FrameRegistration(mounting_angle_deg=45.0, r_sensor_to_probe_m=LEVER)
    profile = _profile(reg, _flange_orientations())
    valid, issues = profile.validity()
    assert valid, issues


def test_profile_round_trips_through_disk(tmp_path):
    reg = FrameRegistration(mounting_angle_deg=43.0, r_sensor_to_probe_m=LEVER,
                            flange_to_sensor_rpy=np.array([0.0, 0.0, 0.2]))
    profile = _profile(reg, _flange_orientations())
    profile.mounting_note = "convex probe + v2 mount"
    path = tmp_path / "profile.json"
    profile.save(path)
    back = CalibrationProfile.load(path)

    rot_bf = _rot('y', 0.8)
    a = compensate(profile, _raw_unloaded(reg, rot_bf), rot_bf)
    b = compensate(back, _raw_unloaded(reg, rot_bf), rot_bf)
    assert b.contact_probe == pytest.approx(a.contact_probe, abs=1e-9)
    assert back.mounting_note == "convex probe + v2 mount"


def test_compensation_reports_every_stage():
    """검증 화면이 단계별로 보여 줘야 하므로 중간 결과를 버리지 않는다."""
    reg = FrameRegistration(mounting_angle_deg=45.0, r_sensor_to_probe_m=LEVER)
    profile = _profile(reg, _flange_orientations())
    out = compensate(profile, _raw_unloaded(reg, np.eye(3)), np.eye(3))
    d = out.to_dict()
    for key in ("rawSensor", "biasCorrectedSensor", "externalSensor",
                "externalProbe", "contactProbe", "normalForceN"):
        assert key in d


def test_gravity_vector_magnitude_is_standard():
    assert np.linalg.norm(GRAVITY_B) == pytest.approx(9.80665)


def _rotation_z(angle):
    """Z 둘레 회전. 시험 안에서만 쓴다."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _flange_rotation_for(gravity_flange):
    """``{F}g`` 가 나오는 ``{B}R{F}`` 하나를 만든다.

    ``g_F = R^T · g_B`` 이므로, ``g_F`` 를 첫 축으로 삼는 정규직교기저를 세우고
    ``g_B`` 를 같은 자리에 놓으면 조건을 만족하는 R 이 나온다.
    """
    target = np.asarray(gravity_flange, dtype=float).reshape(3)
    base = np.array([0.0, 0.0, -9.80665])

    def frame(vector):
        first = vector / np.linalg.norm(vector)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(first @ helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        second = np.cross(first, helper)
        second /= np.linalg.norm(second)
        return np.column_stack([first, second, np.cross(first, second)])

    return frame(base) @ frame(target).T


def _poses_for(mass, com, rotation, count=14, seed=21):
    """심은 질량·COM·회전이 만든 자세들."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        gravity_flange = direction / norm * 9.80665
        g_s = rotation @ gravity_flange
        force = mass * g_s
        torque = np.cross(com, force)
        out.append(CalibrationPose(
            gravity_sensor=g_s,
            gravity_flange=gravity_flange,
            wrench=np.concatenate([force, torque]),
            label=f"p{len(out)}",
        ))
    return out


def test_compensation_uses_the_rotation_the_model_was_fitted_with():
    """적합이 푼 회전과 등록의 가정된 회전이 다르면 후자를 쓰면 안 된다.

    실기에서 이 둘이 89° 어긋나 있었다. 모델을 직접 불러 계산하면 잔여 힘이
    0.17 N 인 자세에서, 등록의 회전으로 중력을 만든 보상은 4.09 N 을 남겼다 —
    적합 잔차로는 절대 드러나지 않는 종류의 어긋남이다. 적합은 자기가 푼 회전
    위에서 잔차를 재기 때문이다.
    """
    fitted = _rotation_z(math.radians(136.25))
    poses = _poses_for(mass=0.2, com=np.array([0.0, 0.0, 0.054]), rotation=fitted)
    model = fit_gravity_model(poses)
    assert not np.allclose(model.rotation_sensor_from_flange, np.eye(3), atol=1e-3)

    profile = CalibrationProfile(
        # 등록은 **다른** 각도를 들고 있다. 실기의 probe.yaml 이 그랬다.
        registration=FrameRegistration(mounting_angle_deg=43.0, flange_to_sensor_rpy=[0, 0, 0.82]),
        bias=BiasResult(
            bias=np.zeros(6), std=np.zeros(6), samples=4000,
            duration_s=4.0, accepted=True, reason="시험",
        ),
        gravity=model,
    )

    # 적합에 쓰인 자세 하나를 그대로 되먹인다. 무접촉이었으니 0 이어야 한다.
    pose = poses[3]
    flange = _flange_rotation_for(pose.gravity_flange)
    result = compensate(profile, pose.wrench, flange)
    assert np.linalg.norm(result.external_sensor[:3]) < 1e-6


def test_working_tare_zeroes_the_pose_it_was_taken_at():
    """작업 자세에서 무접촉이 정확히 0 으로 읽혀야 한다."""
    fitted = _rotation_z(math.radians(136.25))
    poses = _poses_for(mass=0.2, com=np.array([0.0, 0.0, 0.054]), rotation=fitted)
    model = fit_gravity_model(poses)
    profile = CalibrationProfile(
        registration=FrameRegistration(mounting_angle_deg=43.0),
        bias=BiasResult(
            bias=np.zeros(6), std=np.zeros(6), samples=4000,
            duration_s=4.0, accepted=True, reason="시험",
        ),
        gravity=model,
    )

    # 보상이 다 맞아도 남는 잔여를 심는다 — 실기에서 0.2 N 남짓이었다.
    leftover = np.array([0.12, -0.09, 0.18, 0.001, -0.002, 0.0])
    pose = poses[0]
    flange = _flange_rotation_for(pose.gravity_flange)
    loaded = pose.wrench + leftover

    before = compensate(profile, loaded, flange).contact_probe
    assert np.linalg.norm(before[:3]) > 0.2

    profile.working_tare = np.asarray(before, dtype=float)
    after = compensate(profile, loaded, flange).contact_probe
    assert np.allclose(after, np.zeros(6), atol=1e-12)


def test_working_tare_is_a_constant_and_moves_error_elsewhere():
    """이 영점은 잰 자세에서만 정확하다. 그 사실을 시험이 들고 있어야 한다.

    다른 자세로 가면 뺀 만큼이 그대로 오차로 돌아온다 — 숨길 수 없는 성질이고,
    그래서 프로파일이 어느 자세에서 쟀는지를 함께 들고 다닌다.
    """
    fitted = _rotation_z(math.radians(136.25))
    poses = _poses_for(mass=0.2, com=np.array([0.0, 0.0, 0.054]), rotation=fitted)
    model = fit_gravity_model(poses)
    tare = np.array([0.12, -0.09, 0.18, 0.0, 0.0, 0.0])
    profile = CalibrationProfile(
        registration=FrameRegistration(mounting_angle_deg=43.0),
        bias=BiasResult(
            bias=np.zeros(6), std=np.zeros(6), samples=4000,
            duration_s=4.0, accepted=True, reason="시험",
        ),
        gravity=model,
        working_tare=tare,
    )

    other = poses[5]
    flange = _flange_rotation_for(other.gravity_flange)
    result = compensate(profile, other.wrench, flange).contact_probe
    # 다른 자세에서는 뺀 것이 그대로 남는다.
    assert np.allclose(result, -tare, atol=1e-9)


def test_working_tare_survives_a_save_and_load(tmp_path):
    profile = CalibrationProfile(
        registration=FrameRegistration(mounting_angle_deg=43.0),
        bias=BiasResult(
            bias=np.zeros(6), std=np.zeros(6), samples=4000,
            duration_s=4.0, accepted=True, reason="시험",
        ),
        working_tare=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]),
        tare_flange_z=-0.98,
        tare_pose_deg=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    path = tmp_path / "profile.json"
    profile.save(str(path))
    back = CalibrationProfile.load(str(path))
    assert np.allclose(back.working_tare, profile.working_tare)
    assert back.tare_flange_z == profile.tare_flange_z
    assert back.tare_pose_deg == profile.tare_pose_deg


def test_profile_without_a_tare_is_unchanged():
    """영점이 없으면 예전 거동 그대로여야 한다.

    프로브 프레임 회전과 레버암은 영점과 무관하게 계속 걸린다 — 여기서 보는 것은
    **영점이 아무것도 바꾸지 않는가** 뿐이다.
    """
    def make(tare):
        return CalibrationProfile(
            registration=FrameRegistration(mounting_angle_deg=43.0),
            bias=BiasResult(
                bias=np.zeros(6), std=np.zeros(6), samples=4000,
                duration_s=4.0, accepted=True, reason="시험",
            ),
            working_tare=tare,
        )

    raw = np.array([1.0, 2.0, 3.0, 0.01, 0.02, 0.03])
    none_applied = compensate(make(None), raw, None).contact_probe
    zeros_applied = compensate(make(np.zeros(6)), raw, None).contact_probe
    assert np.allclose(none_applied, zeros_applied, atol=1e-12)

    shift = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])
    shifted = compensate(make(shift), raw, None).contact_probe
    assert np.allclose(shifted, none_applied - shift, atol=1e-12)
