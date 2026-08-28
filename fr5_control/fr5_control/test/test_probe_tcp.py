"""프로브 TCP 식별 시험.

합성 자세로 검증한다 — 오프셋을 심어 두고 되찾을 수 있어야 하고, 되찾을 수 없는
조건(자세가 좁다, 점을 헛짚었다)에서는 **조용히 틀린 값을 내는 대신 무효라고
말해야** 한다. 그것이 이 모듈의 존재 이유다.
"""

import math

import numpy as np
import pytest

from fr5_control.probe_tcp import (
    MIN_ORIENTATION_SPREAD_DEG,
    _pivot_lstsq,
    PivotPose,
    axis_in_flange,
    fit_plane,
    orientation_spread_deg,
    pose_from_joints,
    rotation_from_axes,
    StackSegment,
    compose_stack,
    rotation_from_rpy,
    rpy_from_rotation,
    solve_pivot,
)

TRUE_OFFSET = np.array([0.012, -0.034, 0.171])

WIDE_JOINTS = [
    (0.0, -90.0, 90.0, -90.0, -90.0, 0.0),
    (20.0, -70.0, 80.0, -100.0, -70.0, 30.0),
    (-25.0, -100.0, 100.0, -80.0, -110.0, -40.0),
    (10.0, -80.0, 70.0, -60.0, -90.0, 60.0),
    (-15.0, -95.0, 110.0, -110.0, -60.0, -20.0),
]

NARROW_JOINTS = [
    (0.0, -90.0, 90.0, -90.0, -90.0, 0.0),
    (1.0, -90.0, 90.0, -90.0, -90.0, 2.0),
    (-1.0, -89.0, 90.0, -90.0, -90.0, -2.0),
    (0.5, -90.5, 90.0, -90.0, -90.0, 1.0),
]


def _touching(joint_sets, offset=TRUE_OFFSET, jitter_m=0.0, seed=0):
    """모두 같은 고정점을 짚은 자세들을 만든다.

    순기구학으로 회전은 실제 관절각에서 가져오고, 위치는 고정점 조건을 만족하도록
    되맞춘다. ``jitter_m`` 은 조작자가 점을 헛짚은 양이다.
    """
    rng = np.random.default_rng(seed)
    raw = [pose_from_joints(j, f"P{i}") for i, j in enumerate(joint_sets)]
    pivot = raw[0].rotation @ offset + raw[0].position
    out = []
    for pose in raw:
        error = rng.normal(scale=jitter_m, size=3) if jitter_m else np.zeros(3)
        out.append(
            PivotPose(
                label=pose.label,
                joint_deg=pose.joint_deg,
                rotation=pose.rotation,
                position=pivot + error - pose.rotation @ offset,
            )
        )
    return out


# -- 피벗 ---------------------------------------------------------------


def test_recovers_a_planted_offset():
    result = solve_pivot(_touching(WIDE_JOINTS))
    assert np.allclose(result.offset_flange_m, TRUE_OFFSET, atol=1e-9)
    assert result.valid
    assert result.rms_mm < 1e-6


def test_all_poses_agree_on_one_pivot_point():
    poses = _touching(WIDE_JOINTS)
    result = solve_pivot(poses)
    for pose in poses:
        tip = pose.rotation @ result.offset_flange_m + pose.position
        assert np.allclose(tip, result.pivot_base_m, atol=1e-9)


def test_narrow_poses_are_rejected_not_silently_solved():
    """자세가 좁으면 계수행렬이 병약하다. 값이 아니라 판정이 중요하다."""
    result = solve_pivot(_touching(NARROW_JOINTS))
    assert not result.valid
    assert any("다양성" in issue for issue in result.issues)


def test_too_few_poses_are_flagged():
    result = solve_pivot(_touching(WIDE_JOINTS[:2]))
    assert not result.valid
    assert any("자세가" in issue for issue in result.issues)


def test_two_poses_can_never_solve_and_say_so():
    """식 6 개 미지수 6 개인데도 안 풀린다.

    ``R_i - R_j = R_i(I - R_i^T R_j)`` 이고 회전행렬은 고유값 1 을 가지므로
    ``I - R`` 은 항상 특이하다. 상대회전축 방향 성분이 구속되지 않는다는 뜻이고,
    lstsq 는 그 자리에 최소노름 해를 조용히 채워 넣는다 — 그래서 명시적으로
    특이하다고 말해야 한다.
    """
    result = solve_pivot(_touching(WIDE_JOINTS[:2]))
    assert result.singular_values[-1] == pytest.approx(0.0, abs=1e-12)
    assert any("구속되지 않는다" in issue for issue in result.issues)


def test_three_poses_are_the_true_minimum():
    result = solve_pivot(_touching(WIDE_JOINTS[:3]))
    assert result.singular_values[-1] > 1e-3
    assert np.allclose(result.offset_flange_m, TRUE_OFFSET, atol=1e-9)


def test_sloppy_touches_show_up_as_residual():
    result = solve_pivot(_touching(WIDE_JOINTS, jitter_m=0.004, seed=3))
    assert result.rms_mm > 1.0
    assert not result.valid
    assert any("잔차" in issue for issue in result.issues)


def test_one_bad_touch_does_not_drag_the_solution():
    """Huber 가중의 존재 이유. 한 자세만 20 mm 빗나가게 만든다.

    이상치를 없애 주지는 않는다 — 끌려가는 정도를 줄일 뿐이다. 평범한
    최소제곱과 나란히 재서, 실제로 얼마나 나은지를 시험이 붙들고 있게 한다.
    """
    poses = _touching(WIDE_JOINTS)
    bad = poses[2]
    poses[2] = PivotPose(
        label=bad.label,
        joint_deg=bad.joint_deg,
        rotation=bad.rotation,
        position=bad.position + np.array([0.02, 0.0, 0.0]),
    )
    plain, _, _ = _pivot_lstsq(poses, np.ones(len(poses)))
    result = solve_pivot(poses)

    plain_error = np.abs(plain - TRUE_OFFSET).max()
    huber_error = np.abs(result.offset_flange_m - TRUE_OFFSET).max()
    assert huber_error < plain_error / 3.0
    # 그리고 그 자세는 잔차로 드러난다 — 조작자가 다시 잡을 수 있게.
    assert result.residuals_mm[2] > 5.0


def test_requires_at_least_two_poses():
    with pytest.raises(ValueError):
        solve_pivot(_touching(WIDE_JOINTS[:1]))


def test_spread_is_zero_for_a_single_orientation():
    identity = np.eye(3)
    assert orientation_spread_deg([identity]) == 0.0
    assert orientation_spread_deg([identity, identity]) == pytest.approx(0.0)


def test_spread_measures_the_widest_pair():
    turn = rotation_from_rpy(0.0, 0.0, math.radians(60.0))
    spread = orientation_spread_deg([np.eye(3), turn])
    assert spread == pytest.approx(60.0, abs=1e-6)


def test_wide_pose_set_clears_the_spread_floor():
    poses = _touching(WIDE_JOINTS)
    assert solve_pivot(poses).spread_deg > MIN_ORIENTATION_SPREAD_DEG


# -- 평면과 축 ----------------------------------------------------------


def test_plane_fit_recovers_a_tilted_plane():
    normal = np.array([0.1, -0.05, 1.0])
    normal /= np.linalg.norm(normal)
    basis = np.linalg.svd(normal.reshape(1, 3))[2][1:]
    points = [0.5 * normal + u * basis[0] + v * basis[1] for u, v in
              [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (-0.08, 0.05)]]
    fitted, _, residual = fit_plane(points)
    assert np.allclose(fitted, normal, atol=1e-9)
    assert residual < 1e-6


def test_plane_normal_is_oriented_upward():
    """법선 부호가 자세마다 뒤집히면 쓰는 쪽에서 축이 뒤집힌다."""
    points = [(0.0, 0.0, 0.2), (0.1, 0.0, 0.2), (0.0, 0.1, 0.2)]
    normal, _, _ = fit_plane(points)
    assert normal[2] > 0


def test_plane_fit_needs_three_points():
    with pytest.raises(ValueError):
        fit_plane([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])


def test_axis_recovers_the_probe_direction_from_flat_contacts():
    """프로브 축을 심어 두고, 그 축이 아래를 향하는 자세들을 만들어 되찾는다."""
    true_axis = np.array([0.0, 0.0, 1.0])
    down = np.array([0.0, 0.0, -1.0])
    # 프로브 축이 정확히 아래를 향하는 자세 여럿 (z 둘레로만 다르다)
    rotations = []
    for yaw in (0.0, 40.0, -70.0):
        spin = rotation_from_rpy(math.pi, 0.0, math.radians(yaw))
        rotations.append(spin)
    result = axis_in_flange(rotations, down)
    assert result.valid
    assert np.allclose(np.abs(result.axis_flange), np.abs(true_axis), atol=1e-9)


def test_axis_flags_a_pose_that_was_not_flush():
    flush = rotation_from_rpy(math.pi, 0.0, 0.0)
    tilted = rotation_from_rpy(math.pi - math.radians(12.0), 0.0, 0.0)
    result = axis_in_flange([flush, flush, tilted], np.array([0.0, 0.0, -1.0]))
    assert not result.valid
    assert any("불일치" in issue for issue in result.issues)


def test_single_flat_contact_is_flagged_as_unverifiable():
    result = axis_in_flange([np.eye(3)], np.array([0.0, 0.0, -1.0]))
    assert not result.valid


# -- 프레임 조립 --------------------------------------------------------


def test_rotation_from_axes_is_orthonormal_and_right_handed():
    r = rotation_from_axes([0.0, 0.0, 1.0], [1.0, 0.2, 0.0])
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_rotation_from_axes_puts_z_where_asked():
    axis = np.array([0.3, -0.4, 0.866])
    axis /= np.linalg.norm(axis)
    r = rotation_from_axes(axis, [1.0, 0.0, 0.0])
    assert np.allclose(r[:, 2], axis, atol=1e-12)


def test_rotation_from_axes_only_uses_the_orthogonal_part_of_the_hint():
    """힌트가 직교하지 않아도 회전행렬이어야 한다."""
    r = rotation_from_axes([0.0, 0.0, 1.0], [1.0, 0.0, 5.0])
    assert np.allclose(r[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-12)


def test_rotation_from_axes_rejects_a_parallel_hint():
    with pytest.raises(ValueError):
        rotation_from_axes([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])


def test_rpy_round_trips():
    for rpy in [(0.0, 0.0, 0.0), (0.3, -0.2, 1.1), (math.pi, 0.0, 0.5), (-0.7, 0.9, -2.0)]:
        rotation = rotation_from_rpy(*rpy)
        assert np.allclose(rotation_from_rpy(*rpy_from_rotation(rotation)), rotation, atol=1e-12)


def test_rpy_matches_the_kdl_convention_used_by_the_chain():
    """probe.yaml 의 값은 kdl.Rotation.RPY 로 들어간다. 규약이 갈리면 조용히 틀린다."""
    from fr5_control.robot_backend import _rpy

    for rpy in [(0.2, 0.3, -0.4), (1.0, -0.5, 2.0)]:
        assert np.allclose(rotation_from_rpy(*rpy), _rpy(*rpy), atol=1e-12)


def test_rpy_handles_gimbal_lock_without_nan():
    rotation = rotation_from_rpy(0.0, math.pi / 2.0, 0.0)
    roll, pitch, yaw = rpy_from_rotation(rotation)
    assert all(math.isfinite(v) for v in (roll, pitch, yaw))
    assert np.allclose(rotation_from_rpy(roll, pitch, yaw), rotation, atol=1e-9)


# -- 실측 치수 조립 ------------------------------------------------------


def _example_stack(bend_deg=45.0):
    """플랜지 → 마운트① → 센서 → 마운트② → 프로브.

    숫자는 자리표시자다. 시험이 보는 것은 값이 아니라 **사슬이 정합적인가** 다.
    """
    return [
        StackSegment("adapter1", xyz_mm=(0.0, 0.0, 12.0)),
        StackSegment("sensor", xyz_mm=(0.0, 0.0, 31.0)),
        StackSegment("adapter2", xyz_mm=(0.0, 0.0, 18.0), rpy_deg=(0.0, 0.0, bend_deg)),
        StackSegment("probe", xyz_mm=(0.0, 0.0, 96.0)),
    ]


def test_stack_accumulates_axial_lengths():
    result = compose_stack(_example_stack())
    assert result.j6_to_sensor_xyz[2] * 1000.0 == pytest.approx(43.0)
    assert result.j6_to_probe_xyz[2] * 1000.0 == pytest.approx(157.0)


def test_stack_lever_arm_matches_the_two_frame_origins():
    """레버암은 두 원점의 차이여야 한다 — 따로 적어 넣으면 어긋날 수 있는 부분."""
    result = compose_stack(_example_stack())
    sensor = result.frames["sensor"][:3, 3]
    probe = result.frames["probe"][:3, 3]
    expected = result.frames["probe"][:3, :3].T @ (probe - sensor)
    assert np.allclose(result.r_sensor_to_probe_m, expected, atol=1e-12)
    assert np.linalg.norm(result.r_sensor_to_probe_m) * 1000.0 == pytest.approx(114.0)


def test_stack_carries_the_mount_bend_into_the_probe_rotation():
    result = compose_stack(_example_stack(bend_deg=45.0))
    assert math.degrees(result.j6_to_probe_rpy[2]) == pytest.approx(45.0, abs=1e-9)
    # 센서는 꺾임 앞이므로 돌아가 있지 않다.
    assert np.allclose(result.j6_to_sensor_rpy, np.zeros(3), atol=1e-12)


def test_stack_rotation_is_a_proper_rotation():
    result = compose_stack(_example_stack(bend_deg=45.0))
    rotation = result.frames["probe"][:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_stack_output_round_trips_through_the_chain_convention():
    """조립한 rpy 를 다시 행렬로 만들면 원래 프레임이어야 한다.

    이 값은 ``build_chain`` 이 ``kdl.Rotation.RPY`` 로 되읽는다. 여기서 갈리면
    로봇이 조용히 틀린 기하로 돈다.
    """
    result = compose_stack(_example_stack(bend_deg=45.0))
    assert np.allclose(
        rotation_from_rpy(*result.j6_to_probe_rpy),
        result.frames["probe"][:3, :3],
        atol=1e-12,
    )


def test_stack_rejects_an_unknown_frame_name():
    with pytest.raises(KeyError):
        compose_stack(_example_stack(), probe_frame="tip")


def test_lateral_offset_shows_up_off_axis():
    """축에서 벗어난 마운트도 표현된다 — 꺾임만 있는 게 아닐 수 있다."""
    stack = [
        StackSegment("adapter1", xyz_mm=(0.0, 0.0, 12.0)),
        StackSegment("sensor", xyz_mm=(0.0, 0.0, 31.0)),
        StackSegment("adapter2", xyz_mm=(8.0, -3.0, 18.0), rpy_deg=(0.0, 0.0, 45.0)),
        StackSegment("probe", xyz_mm=(0.0, 0.0, 96.0)),
    ]
    result = compose_stack(stack)
    assert result.j6_to_probe_xyz[0] * 1000.0 == pytest.approx(8.0)
    assert result.j6_to_probe_xyz[1] * 1000.0 == pytest.approx(-3.0)


def test_pivot_and_stack_agree_when_the_stack_is_the_truth():
    """두 경로가 만나는 지점. 캘리퍼 값을 심어 피벗으로 되찾는다."""
    stack = compose_stack(_example_stack())
    truth = stack.j6_to_probe_xyz
    poses = _touching(WIDE_JOINTS, offset=truth)
    recovered = solve_pivot(poses)
    assert np.allclose(recovered.offset_flange_m, truth, atol=1e-9)


def test_segment_translation_is_measured_before_its_own_rotation():
    """``[R | p]`` 의 p 는 부모 프레임 값이다.

    회전 마디에 병진을 같이 적으면 그 병진은 **회전 전** 축을 따라간다. 꺾인
    마운트를 한 마디로 적으면 길이는 맞는데 방향이 틀리는 오류가 나므로,
    규약을 시험으로 못박아 둔다.
    """
    one = compose_stack([
        StackSegment("sensor"),
        StackSegment("probe", xyz_mm=(0.0, 0.0, 50.0), rpy_deg=(-90.0, 0.0, 0.0)),
    ])
    # 회전을 같이 적었어도 병진은 부모 +z 를 따라갔다.
    assert np.allclose(one.j6_to_probe_xyz * 1000.0, [0.0, 0.0, 50.0], atol=1e-9)

    # 꺾인 뒤 그 방향으로 나가려면 마디를 나눈다.
    two = compose_stack([
        StackSegment("sensor"),
        StackSegment("bend", rpy_deg=(-90.0, 0.0, 0.0)),
        StackSegment("probe", xyz_mm=(0.0, 0.0, 50.0)),
    ])
    assert np.allclose(two.j6_to_probe_xyz * 1000.0, [0.0, 50.0, 0.0], atol=1e-9)
