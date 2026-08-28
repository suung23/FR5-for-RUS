"""프레임 등록과 wrench 변환 회귀 테스트.

부호 규약이 여기 한 곳에서만 정해지므로, 여기가 틀리면 제어 전체가 조용히 틀린다.
특히 **모멘트가 유사벡터**라는 점을 고정한다 — 반사가 섞인 행렬을 쓰면 힘은 맞는데
모멘트만 거울상이 되고, 화면으로는 알아채기 어렵다.
"""
import math

import numpy as np
import pytest

from fr5_control.wrench_frames import (
    FrameRegistration,
    GRAVITY_B,
    rotation_z,
    skew,
    wrench_rotate,
    wrench_translate,
)

SQRT2_2 = math.sqrt(2.0) / 2.0


def test_skew_matches_cross_product():
    a = np.array([0.3, -1.2, 4.0])
    b = np.array([-2.0, 0.5, 0.7])
    assert skew(a) @ b == pytest.approx(np.cross(a, b))


def test_registration_is_a_proper_rotation():
    """직교이고 det = +1. 모멘트를 벡터와 같게 변환해도 되는 근거 전체다."""
    for angle in (0.0, 45.0, -30.0, 91.7):
        for flip in (False, True):
            rot = FrameRegistration(mounting_angle_deg=angle, axial_flip=flip
                                    ).rotation_probe_from_sensor()
            assert rot.T @ rot == pytest.approx(np.eye(3), abs=1e-12)
            assert np.linalg.det(rot) == pytest.approx(1.0)


def test_mounting_angle_places_the_sensor_axes_as_measured():
    """사양의 실측 장착: +y_S 는 +45°, +x_S 는 −45° 방향이다.

    {P}R{S} 의 열이 곧 P 에서 본 센서 기저다.
    """
    rot = FrameRegistration(mounting_angle_deg=45.0).rotation_probe_from_sensor()
    x_s, y_s, z_s = rot[:, 0], rot[:, 1], rot[:, 2]
    assert x_s == pytest.approx([SQRT2_2, -SQRT2_2, 0.0], abs=1e-12)
    assert y_s == pytest.approx([SQRT2_2, SQRT2_2, 0.0], abs=1e-12)
    # 축 방향은 돌아가도 침투축은 공유한다 — 회전이 z 둘레이기 때문이다.
    assert z_s == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)


def test_zero_angle_is_identity():
    rot = FrameRegistration(mounting_angle_deg=0.0).rotation_probe_from_sensor()
    assert rot == pytest.approx(np.eye(3), abs=1e-12)


def test_axial_flip_reverses_the_normal_axis():
    """뒤집어 달면 압축 방향이 반대가 된다. 부호 뒤집기는 여기 한 곳뿐이다."""
    plain = FrameRegistration(mounting_angle_deg=0.0, axial_flip=False)
    flipped = FrameRegistration(mounting_angle_deg=0.0, axial_flip=True)
    push = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    assert wrench_rotate(plain.rotation_probe_from_sensor(), push)[2] == pytest.approx(3.0)
    assert wrench_rotate(flipped.rotation_probe_from_sensor(), push)[2] == pytest.approx(-3.0)


def test_axial_flip_moves_force_and_moment_together():
    """스칼라 부호였다면 모멘트가 잊혔을 자리다. 180° 회전은 둘에 같이 작용한다."""
    flipped = FrameRegistration(mounting_angle_deg=0.0, axial_flip=True)
    rot = flipped.rotation_probe_from_sensor()
    w = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.4])
    out = wrench_rotate(rot, w)
    assert out[2] == pytest.approx(-2.0)
    assert out[5] == pytest.approx(-0.4)


def test_rotation_preserves_magnitudes():
    """프레임만 바꾸는 변환이므로 크기가 달라지면 안 된다."""
    rot = FrameRegistration(mounting_angle_deg=45.0).rotation_probe_from_sensor()
    w = np.array([1.0, -2.0, 3.0, 0.1, 0.2, -0.3])
    out = wrench_rotate(rot, w)
    assert np.linalg.norm(out[:3]) == pytest.approx(np.linalg.norm(w[:3]))
    assert np.linalg.norm(out[3:]) == pytest.approx(np.linalg.norm(w[3:]))


def test_lateral_sensor_force_lands_on_the_probe_axes_as_expected():
    """센서 +x 로 민 힘은 45° 장착에서 P 의 x 와 −y 에 같은 크기로 나뉜다."""
    rot = FrameRegistration(mounting_angle_deg=45.0).rotation_probe_from_sensor()
    out = wrench_rotate(rot, np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert out[0] == pytest.approx(2.0 * SQRT2_2)
    assert out[1] == pytest.approx(-2.0 * SQRT2_2)
    assert out[2] == pytest.approx(0.0, abs=1e-12)


def test_wrench_translate_leaves_force_untouched():
    w = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    out = wrench_translate(w, [0.0, 0.0, 0.1])
    assert out[:3] == pytest.approx(w[:3])


def test_wrench_translate_removes_the_lever_term():
    """센서 원점에서 본 모멘트가 순수 지렛대 항이면, 접촉점에서는 0 이 되어야 한다.

    r ≈ 0.1 m, F_z ≈ 5 N 이면 이 항이 0.5 N·m 다 — 정렬 신호(수십 mN·m)를 통째로
    덮는다. 빼먹으면 rx/ry 정렬이 원리적으로 동작하지 않는다.
    """
    lever = np.array([0.0, 0.0, 0.1])
    force = np.array([2.0, -1.0, 5.0])
    w = np.concatenate([force, np.cross(lever, force)])
    assert wrench_translate(w, lever)[3:] == pytest.approx(np.zeros(3), abs=1e-12)


def test_gravity_in_sensor_points_down_when_frames_align():
    """플랜지가 베이스와 정렬돼 있으면 센서가 보는 중력도 −z 다."""
    reg = FrameRegistration()
    g = reg.gravity_in_sensor(np.eye(3))
    assert g == pytest.approx(GRAVITY_B)


def test_gravity_in_sensor_follows_the_flange():
    """플랜지를 x 둘레로 +90° 돌리면 중력이 센서 −y 로 온다.

    손으로 확인: 플랜지 +y 가 베이스 +z(위)를 향하게 되므로, 베이스 −z 인 중력은
    플랜지 −y 성분이 된다. 크기는 보존된다.
    """
    reg = FrameRegistration()
    rot_bf = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    g = reg.gravity_in_sensor(rot_bf)
    assert np.linalg.norm(g) == pytest.approx(9.80665)
    assert g == pytest.approx([0.0, -9.80665, 0.0], abs=1e-9)


def test_flange_to_sensor_rotation_is_applied():
    """장착 회전이 있으면 중력도 그만큼 돌아서 들어온다."""
    reg = FrameRegistration(flange_to_sensor_rpy=[0.0, 0.0, math.pi / 2])
    g = reg.gravity_in_sensor(np.eye(3))
    assert g == pytest.approx(GRAVITY_B, abs=1e-9)   # z 둘레 회전은 중력을 안 바꾼다
    reg2 = FrameRegistration(flange_to_sensor_rpy=[math.pi / 2, 0.0, 0.0])
    g2 = reg2.gravity_in_sensor(np.eye(3))
    assert g2[1] == pytest.approx(-9.80665, abs=1e-9)


def test_registration_round_trips_through_json():
    reg = FrameRegistration(
        mounting_angle_deg=43.5, axial_flip=True,
        r_sensor_to_probe_m=np.array([0.001, -0.002, 0.085]),
        flange_to_sensor_rpy=np.array([0.0, 0.0, 0.3]),
    )
    back = FrameRegistration.from_dict(reg.to_dict())
    assert back.rotation_probe_from_sensor() == pytest.approx(reg.rotation_probe_from_sensor())
    assert back.r_sensor_to_probe_m == pytest.approx(reg.r_sensor_to_probe_m)


def test_rotation_z_matches_hand_computation():
    assert rotation_z(math.pi / 2) @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)


# ---- probe.yaml 적층에서 유도하기 (2026-08-27) -----------------------------
#
# 이 값들이 노드에 **따로** 선언돼 있던 동안, telemetry_bridge 는 probe.yaml 을
# 받지 않아 레버암 0 으로 돌았다. 축방향 압축에서는 r ∥ f 라 차이가 0 이어서
# 증상이 없었다 — 그래서 아래 테스트는 "횡력에서 차이가 난다" 를 함께 고정한다.

#: 2026-08-27 적층. probe.yaml 과 같은 값이다 (마운트는 CAD, 나머지는 실측).
STACK = dict(
    sensor_xyz=[0.0, 0.0, 0.033], sensor_rpy=[0.0, 0.0, 0.8203],
    probe_xyz=[0.0, 0.0, 0.234], probe_rpy=[0.0, 0.0, math.pi / 2],
)


def test_from_stack_reproduces_measured_registration():
    reg = FrameRegistration.from_stack(**STACK)
    assert reg.mounting_angle_deg == pytest.approx(43.0, abs=0.01)   # 90 - 47
    assert reg.axial_flip is False
    assert reg.r_sensor_to_probe_m == pytest.approx([0.0, 0.0, 0.201], abs=1e-9)
    assert reg.flange_to_sensor_rpy == pytest.approx([0.0, 0.0, 0.8203])


def test_from_stack_lever_arm_only_bites_on_lateral_force():
    reg = FrameRegistration.from_stack(**STACK)
    axial = wrench_translate([0.0, 0.0, 5.0, 0.0, 0.0, 0.0], reg.r_sensor_to_probe_m)
    assert axial[3:] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)    # r ∥ f
    lateral = wrench_translate([1.49, 0.0, 0.0, 0.0, 0.0, 0.0], reg.r_sensor_to_probe_m)
    assert np.linalg.norm(lateral[3:]) == pytest.approx(0.300, abs=1e-3)   # max_moment_nm


def test_from_stack_detects_a_flipped_assembly():
    reg = FrameRegistration.from_stack(
        sensor_xyz=[0.0, 0.0, 0.033], sensor_rpy=[math.pi, 0.0, 0.8203],
        probe_xyz=[0.0, 0.0, 0.234], probe_rpy=[0.0, 0.0, math.pi / 2],
    )
    assert reg.axial_flip is True
    rot = reg.rotation_probe_from_sensor()
    # 뒤집혔으므로 센서 +z 가 프로브 -z 를 향한다. 그래도 진짜 회전이어야 한다 —
    # 부호 대각행렬로 흉내내면 모멘트만 거울상이 된다.
    assert rot[2, 2] == pytest.approx(-1.0)
    assert np.linalg.det(rot) == pytest.approx(1.0)
    assert rot @ rot.T == pytest.approx(np.eye(3), abs=1e-12)


def test_from_stack_refuses_unmeasured_values():
    """.nan 은 0 으로 대체하지 않는다 — 조용히 틀린 기하로 도는 쪽이 더 위험하다."""
    with pytest.raises(ValueError, match="NaN"):
        FrameRegistration.from_stack(
            sensor_xyz=[0.0, 0.0, float("nan")], sensor_rpy=[0.0, 0.0, 0.0],
            probe_xyz=[0.0, 0.0, 0.234], probe_rpy=[0.0, 0.0, math.pi / 2],
        )


def test_from_stack_refuses_geometry_an_angle_cannot_describe():
    """z 둘레가 아닌 장착은 θ 하나로 표현되지 않는다. 그때는 멈춘다."""
    with pytest.raises(ValueError, match="z 둘레 회전이 아니다"):
        FrameRegistration.from_stack(
            sensor_xyz=[0.0, 0.0, 0.033], sensor_rpy=[0.3, 0.0, 0.8203],
            probe_xyz=[0.0, 0.0, 0.234], probe_rpy=[0.0, 0.0, math.pi / 2],
        )


def test_from_stack_agrees_with_the_probe_frame_lever_convention():
    """레버암은 {P} 기준이다. 센서가 프로브 축에서 벗어나 달리면 x·y 가 생긴다."""
    reg = FrameRegistration.from_stack(
        sensor_xyz=[0.01, 0.0, 0.033], sensor_rpy=[0.0, 0.0, 0.8203],
        probe_xyz=[0.0, 0.0, 0.234], probe_rpy=[0.0, 0.0, math.pi / 2],
    )
    # {F} 에서 (-0.01, 0, 0.21) 이고, {P} 는 그보다 +90° 돌아가 있다.
    assert reg.r_sensor_to_probe_m == pytest.approx([0.0, 0.01, 0.201], abs=1e-9)
